from __future__ import annotations

import json
import logging
import os
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import qlib
from qlib.backtest import backtest

# Keep logs compact for long optimization runs.
os.environ.setdefault("TQDM_DISABLE", "1")
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")


def compute_supertrend(df: pd.DataFrame, period: int = 14, multiplier: float = 3.0) -> pd.DataFrame:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    hl2 = (high + low) / 2.0

    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()

    for i in range(1, len(df)):
        if pd.isna(final_upper.iloc[i - 1]):
            final_upper.iloc[i] = basic_upper.iloc[i]
            final_lower.iloc[i] = basic_lower.iloc[i]
            continue

        if basic_upper.iloc[i] < final_upper.iloc[i - 1] or close.iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        if basic_lower.iloc[i] > final_lower.iloc[i - 1] or close.iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

    st = pd.Series(index=df.index, dtype=float)
    first_valid = atr.first_valid_index()
    if first_valid is not None:
        first_pos = df.index.get_loc(first_valid)
        st.iloc[first_pos] = final_upper.iloc[first_pos]
        for i in range(first_pos + 1, len(df)):
            prev_st = st.iloc[i - 1]
            prev_upper = final_upper.iloc[i - 1]
            cur_upper = final_upper.iloc[i]
            cur_lower = final_lower.iloc[i]
            cur_close = close.iloc[i]
            if prev_st == prev_upper:
                st.iloc[i] = cur_upper if cur_close <= cur_upper else cur_lower
            else:
                st.iloc[i] = cur_lower if cur_close >= cur_lower else cur_upper

    out = pd.DataFrame(index=df.index)
    out["SUPERTREND"] = st
    out["SUPERTREND_DIR"] = np.where(close >= st, 1.0, -1.0)
    out["SUPERTREND_GAP"] = (close - st) / close.replace(0, np.nan)
    return out


def build_all_df(raw_by_code: dict[str, pd.DataFrame], st_period: int, st_multiplier: float) -> pd.DataFrame:
    parts = []
    for code, raw in raw_by_code.items():
        st = compute_supertrend(raw, period=st_period, multiplier=st_multiplier)

        feat = pd.DataFrame(
            {
                "OPEN": raw["open"].astype(float),
                "HIGH": raw["high"].astype(float),
                "LOW": raw["low"].astype(float),
                "CLOSE": raw["close"].astype(float),
                "VOLUME": raw["volume"].astype(float),
                "CHANGE": raw["change"].astype(float),
                "SUPERTREND": st["SUPERTREND"].astype(float),
                "SUPERTREND_DIR": st["SUPERTREND_DIR"].astype(float),
                "SUPERTREND_GAP": st["SUPERTREND_GAP"].astype(float),
            }
        )
        label = pd.DataFrame({"LABEL0": raw["close"].shift(-1) / raw["close"] - 1.0})

        idx = pd.MultiIndex.from_arrays(
            [raw["date"], pd.Series([code] * len(raw))], names=["datetime", "instrument"]
        )
        feat.index = idx
        label.index = idx

        feat.columns = pd.MultiIndex.from_product([["feature"], feat.columns])
        label.columns = pd.MultiIndex.from_product([["label"], label.columns])
        parts.append(pd.concat([feat, label], axis=1))

    all_df = pd.concat(parts, axis=0)
    all_df = all_df.reset_index().sort_values(["datetime", "instrument"]).set_index(["datetime", "instrument"])
    return all_df


def train_and_predict(
    all_df: pd.DataFrame,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> tuple[pd.Series, pd.Series]:
    feat = all_df["feature"]
    label = all_df["label"]["LABEL0"]
    dt_index = feat.index.get_level_values("datetime")

    train_mask = (dt_index >= pd.Timestamp("2018-01-01")) & (dt_index <= pd.Timestamp("2022-12-31"))
    valid_mask = (dt_index >= pd.Timestamp("2023-01-01")) & (dt_index <= pd.Timestamp("2024-12-31"))
    test_mask = (dt_index >= pd.Timestamp("2025-01-01")) & (dt_index <= pd.Timestamp("2026-02-25"))

    x_train = feat[train_mask]
    y_train = label[train_mask]
    x_valid = feat[valid_mask]
    y_valid = label[valid_mask]
    x_test = feat[test_mask]

    trn_notna = y_train.notna()
    val_notna = y_valid.notna()
    x_train = x_train[trn_notna]
    y_train = y_train[trn_notna]
    x_valid = x_valid[val_notna]
    y_valid = y_valid[val_notna]

    params = {
        "objective": "mse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 5,
        "verbosity": -1,
        "num_threads": 4,
        "seed": 42,
        "feature_pre_filter": False,
    }

    dtrain = lgb.Dataset(x_train.values, label=y_train.values)
    dvalid = lgb.Dataset(x_valid.values, label=y_valid.values, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )

    pred = pd.Series(model.predict(x_test.values), index=x_test.index)
    gap = feat.loc[x_test.index, "SUPERTREND_GAP"].astype(float)
    return pred, gap


def evaluate_strategy(
    pred: pd.Series,
    gap: pd.Series,
    topk: int,
    n_drop: int,
    hold_thresh: int,
    risk_degree: float,
    gap_abs_threshold: float,
    lambda_mdd: float,
):
    keep_mask = gap.abs() <= gap_abs_threshold
    signal = pred.where(keep_mask)

    executor_config = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
    }
    strategy_config = {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy.signal_strategy",
        "kwargs": {
            "signal": signal,
            "topk": topk,
            "n_drop": n_drop,
            "hold_thresh": hold_thresh,
            "risk_degree": risk_degree,
        },
    }

    portfolio_metric_dict, _ = backtest(
        start_time="2025-01-01",
        end_time="2026-02-25",
        strategy=strategy_config,
        executor=executor_config,
        benchmark="1489.T",
        account=10_000_000,
        exchange_kwargs={
            "freq": "day",
            "limit_threshold": None,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    )

    report = portfolio_metric_dict["1day"][0]
    curve = pd.DataFrame(index=report.index)
    curve["cum_return_wo_cost"] = report["return"].cumsum()
    curve["cum_return_w_cost"] = (report["return"] - report["cost"]).cumsum()
    curve["bench_cum"] = report["bench"].cumsum()

    ret_w_cost = float(curve["cum_return_w_cost"].iloc[-1])
    running_max = curve["cum_return_w_cost"].cummax()
    max_drawdown = float((running_max - curve["cum_return_w_cost"]).max())
    objective_value = ret_w_cost - lambda_mdd * max_drawdown

    return {
        "objective": objective_value,
        "curve": curve,
        "report": report,
        "ret_w_cost": ret_w_cost,
        "ret_wo_cost": float(curve["cum_return_wo_cost"].iloc[-1]),
        "bench": float(curve["bench_cum"].iloc[-1]),
        "max_drawdown": max_drawdown,
        "coverage": float(keep_mask.fillna(False).mean()),
        "mean_cost": float(report["cost"].mean()),
        "mean_turnover": float(report["turnover"].mean()),
    }


def main() -> None:
    root = Path('/Users/kei/git/src/github.com/microsoft/qlib')
    source_dir = root / 'data' / 'jp_multi_etf' / 'lgb_source'
    output_dir = root / 'outputs' / 'jp_multi_optuna_gapdrop_500'
    output_dir.mkdir(parents=True, exist_ok=True)

    n_trials = 500
    lambda_mdd = 0.5

    codes = [
        "1489.T",
        "1615.T", "1624.T", "1628.T", "1631.T", "1620.T", "1622.T", "1617.T", "1630.T", "1629.T",
        "1619.T", "1632.T", "1633.T", "1618.T", "1621.T", "1623.T", "1626.T", "1627.T", "1625.T",
    ]

    raw_by_code: dict[str, pd.DataFrame] = {}
    for code in codes:
        raw = pd.read_csv(source_dir / f"{code}.csv")
        raw["date"] = pd.to_datetime(raw["date"])
        raw = raw.sort_values("date").reset_index(drop=True)
        raw_by_code[code] = raw

    qlib.init(provider_uri=str(root / 'data' / 'jp_multi_etf' / 'qlib_day'), region='cn', kernels=1)
    logging.getLogger("qlib").setLevel(logging.ERROR)

    df_cache: dict[tuple[int, float], pd.DataFrame] = {}
    pred_cache: dict[tuple[int, float, int, int], tuple[pd.Series, pd.Series]] = {}

    def get_pred_gap(st_period: int, st_multiplier: float, num_boost_round: int, early_stopping_rounds: int):
        key_df = (st_period, round(st_multiplier, 4))
        if key_df not in df_cache:
            df_cache[key_df] = build_all_df(raw_by_code, st_period=st_period, st_multiplier=st_multiplier)

        key_pred = (st_period, round(st_multiplier, 4), num_boost_round, early_stopping_rounds)
        if key_pred not in pred_cache:
            pred_cache[key_pred] = train_and_predict(
                df_cache[key_df],
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds,
            )
        return pred_cache[key_pred]

    def objective(trial: optuna.Trial) -> float:
        st_period = trial.suggest_int("st_period", 7, 40)
        st_multiplier = trial.suggest_float("st_multiplier", 1.5, 5.0, step=0.1)

        topk = trial.suggest_int("topk", 2, 8)
        n_drop = trial.suggest_int("n_drop", 0, min(4, topk - 1))
        gap_abs_threshold = trial.suggest_float("gap_abs_threshold", 0.005, 0.10)
        hold_thresh = trial.suggest_int("hold_thresh", 1, 7)
        risk_degree = trial.suggest_float("risk_degree", 0.60, 1.00)

        try:
            pred, gap = get_pred_gap(st_period, st_multiplier, num_boost_round=120, early_stopping_rounds=20)
            ev = evaluate_strategy(
                pred=pred,
                gap=gap,
                topk=topk,
                n_drop=n_drop,
                hold_thresh=hold_thresh,
                risk_degree=risk_degree,
                gap_abs_threshold=gap_abs_threshold,
                lambda_mdd=lambda_mdd,
            )
            trial.set_user_attr("cum_return_w_cost_last", ev["ret_w_cost"])
            trial.set_user_attr("max_drawdown", ev["max_drawdown"])
            trial.set_user_attr("objective", ev["objective"])
            trial.set_user_attr("bench_cum_last", ev["bench"])
            trial.set_user_attr("coverage", ev["coverage"])
            trial.set_user_attr("mean_turnover", ev["mean_turnover"])
            return ev["objective"]
        except Exception as ex:
            trial.set_user_attr("error", str(ex)[:300])
            return -1e9

    def on_trial_end(study: optuna.Study, trial: optuna.Trial) -> None:
        n = trial.number + 1
        if n % 20 == 0 or n == 1 or n == n_trials:
            print(
                f"progress={n}/{n_trials} best_objective={study.best_value:.6f} "
                f"best_trial={study.best_trial.number}"
            )

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, callbacks=[on_trial_end], show_progress_bar=False)

    best_params = study.best_trial.params

    pred_final, gap_final = get_pred_gap(
        st_period=int(best_params["st_period"]),
        st_multiplier=float(best_params["st_multiplier"]),
        num_boost_round=300,
        early_stopping_rounds=30,
    )
    final = evaluate_strategy(
        pred=pred_final,
        gap=gap_final,
        topk=int(best_params["topk"]),
        n_drop=int(best_params["n_drop"]),
        hold_thresh=int(best_params["hold_thresh"]),
        risk_degree=float(best_params["risk_degree"]),
        gap_abs_threshold=float(best_params["gap_abs_threshold"]),
        lambda_mdd=lambda_mdd,
    )

    final["report"].to_csv(output_dir / "report_normal_1day.csv")
    final["curve"].to_csv(output_dir / "equity_curve.csv")

    summary = {
        "objective_formula": "cum_return_w_cost_last - lambda_mdd * max_drawdown",
        "lambda_mdd": lambda_mdd,
        "n_trials": n_trials,
        "best_trial_number": int(study.best_trial.number),
        "best_trial_objective": float(study.best_trial.value),
        "best_st_period": int(best_params["st_period"]),
        "best_st_multiplier": float(best_params["st_multiplier"]),
        "best_topk": int(best_params["topk"]),
        "best_n_drop": int(best_params["n_drop"]),
        "best_gap_abs_threshold": float(best_params["gap_abs_threshold"]),
        "best_hold_thresh": int(best_params["hold_thresh"]),
        "best_risk_degree": float(best_params["risk_degree"]),
        "final_objective": final["objective"],
        "final_cum_return_wo_cost_last": final["ret_wo_cost"],
        "final_cum_return_w_cost_last": final["ret_w_cost"],
        "final_max_drawdown": final["max_drawdown"],
        "final_bench_cum_last": final["bench"],
        "final_mean_cost": final["mean_cost"],
        "final_mean_turnover": final["mean_turnover"],
        "final_signal_coverage": final["coverage"],
    }
    pd.Series(summary).to_csv(output_dir / "summary.csv", header=["value"])

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state", "user_attrs"))
    trials_df.to_csv(output_dir / "optuna_trials.csv", index=False)

    with (output_dir / "best_params.json").open("w") as f:
        json.dump(best_params, f, indent=2)

    curve_plot = pd.read_csv(output_dir / "equity_curve.csv", parse_dates=["datetime"])
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(curve_plot["datetime"], curve_plot["cum_return_wo_cost"], label="Strategy (wo cost)")
    ax.plot(curve_plot["datetime"], curve_plot["cum_return_w_cost"], label="Strategy (w cost)")
    ax.plot(curve_plot["datetime"], curve_plot["bench_cum"], label="Benchmark 1489.T")
    ax.set_title("JP Multi ETF Optuna 500 Trials (SuperTrend + Topk/Dropout)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "equity_curve.png", dpi=150)

    plt.figure(figsize=(10, 4))
    trials_sorted = trials_df.sort_values("number")
    plt.plot(trials_sorted["number"], trials_sorted["value"], linewidth=1.0)
    plt.title("Optuna Objective by Trial")
    plt.xlabel("Trial")
    plt.ylabel("Objective")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "objective_history.png", dpi=150)

    print("DONE")
    print("output_dir=", output_dir)
    print("best_params=", best_params)


if __name__ == "__main__":
    main()
