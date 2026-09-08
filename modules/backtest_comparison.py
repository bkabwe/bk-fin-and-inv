from __future__ import annotations

import random

from modules.backtester import DEFAULT_WALK_FORWARD_HORIZON, get_walk_forward_window_config, run_walk_forward
from modules.data_fetcher import get_sp500_tickers, get_stock_data
from modules.logger import get_logger

logger = get_logger(__name__)

DEFAULT_SAMPLE_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "JPM",
    "BAC",
    "WFC",
    "XOM",
    "CVX",
    "COP",
    "UNH",
    "JNJ",
    "PFE",
    "MRK",
    "WMT",
    "COST",
    "HD",
    "CAT",
    "BA",
    "GE",
    "LMT",
    "DIS",
    "NFLX",
    "KO",
    "PEP",
    "NKE",
    "MCD",
]


def select_sample_tickers(
    tickers: list[str],
    sample_size: int,
    random_seed: int | None = None,
    ticker_offset: int = 0,
) -> list[str]:
    sample = list(tickers)
    if random_seed is not None:
        random.Random(random_seed).shuffle(sample)
    if ticker_offset > 0:
        sample = sample[ticker_offset:]
    return sample[:sample_size]


def _weighted_mean(weighted_values: list[tuple[float, int]]) -> float | None:
    if not weighted_values:
        return None
    total_weight = sum(weight for _, weight in weighted_values)
    if total_weight <= 0:
        return None
    weighted_sum = sum(value * weight for value, weight in weighted_values)
    return round(float(weighted_sum / total_weight), 6)


def _weighted_median(weighted_values: list[tuple[float, int]]) -> float | None:
    if not weighted_values:
        return None
    expanded = sorted(weighted_values, key=lambda item: item[0])
    total_weight = sum(weight for _, weight in expanded)
    if total_weight <= 0:
        return None
    midpoint = total_weight / 2.0
    cumulative = 0
    for value, weight in expanded:
        cumulative += weight
        if cumulative >= midpoint:
            return round(float(value), 6)
    return round(float(expanded[-1][0]), 6)


def summarize_backtest_results(per_ticker: list[dict]) -> dict:
    models = ["arima", "trend", "lightgbm"]
    if any(("naive_rmse" in row or "naive_windows" in row) for row in per_ticker):
        models.append("naive")
    rmse_values: dict[str, list[tuple[float, int]]] = {model: [] for model in models}
    ticker_counts: dict[str, int] = {model: 0 for model in models}
    window_counts: dict[str, int] = {model: 0 for model in models}
    lightgbm_wins = 0
    lightgbm_compared = 0
    lightgbm_vs_naive_wins = 0
    lightgbm_vs_naive_compared = 0

    for row in per_ticker:
        for model in models:
            windows = int(row.get(f"{model}_windows", 0) or 0)
            rmse = row.get(f"{model}_rmse")
            window_counts[model] += windows
            if windows > 0 and rmse is not None:
                ticker_counts[model] += 1
                rmse_values[model].append((float(rmse), windows))

        if (
            int(row.get("lightgbm_windows", 0) or 0) > 0
            and int(row.get("arima_windows", 0) or 0) > 0
            and int(row.get("trend_windows", 0) or 0) > 0
        ):
            lightgbm_compared += 1
            if float(row["lightgbm_rmse"]) < float(row["arima_rmse"]) and float(row["lightgbm_rmse"]) < float(row["trend_rmse"]):
                lightgbm_wins += 1
        if int(row.get("lightgbm_windows", 0) or 0) > 0 and int(row.get("naive_windows", 0) or 0) > 0:
            lightgbm_vs_naive_compared += 1
            if float(row["lightgbm_rmse"]) < float(row["naive_rmse"]):
                lightgbm_vs_naive_wins += 1

    model_summary = {
        model: {
            "mean_rmse": _weighted_mean(rmse_values[model]),
            "median_rmse": _weighted_median(rmse_values[model]),
            "tickers_evaluated": int(ticker_counts[model]),
            "windows_evaluated": int(window_counts[model]),
        }
        for model in models
    }
    win_pct = round((lightgbm_wins / lightgbm_compared) * 100.0, 2) if lightgbm_compared > 0 else 0.0
    naive_win_pct = (
        round((lightgbm_vs_naive_wins / lightgbm_vs_naive_compared) * 100.0, 2)
        if lightgbm_vs_naive_compared > 0
        else 0.0
    )
    return {
        "total_tickers": int(len(per_ticker)),
        "models": model_summary,
        "lightgbm_wins_vs_both": {
            "wins": int(lightgbm_wins),
            "comparable_tickers": int(lightgbm_compared),
            "win_pct": win_pct,
        },
        "lightgbm_wins_vs_naive": {
            "wins": int(lightgbm_vs_naive_wins),
            "comparable_tickers": int(lightgbm_vs_naive_compared),
            "win_pct": naive_win_pct,
        },
    }


def _format_per_ticker_rmse(row: dict, model: str, width: int) -> str:
    windows = int(row.get(f"{model}_windows", 0) or 0)
    rmse = row.get(f"{model}_rmse")
    if windows <= 0 or rmse is None:
        return f"{'n/a':>{width}}"
    return f"{float(rmse):>{width}.6f}"


def format_per_ticker_diagnostics_table(per_ticker: list[dict]) -> str:
    lines = [
        "Per-ticker RMSE and LightGBM prediction variance diagnostics",
        (
            "Ticker  ARIMA RMSE  Trend RMSE  LightGBM RMSE  Naive RMSE  "
            "LightGBM Pred Return Std  LightGBM Pred Return Min  LightGBM Pred Return Max"
        ),
        (
            "------  ----------  ----------  -------------  ----------  "
            "----------------------  -----------------------  -----------------------"
        ),
    ]
    for row in per_ticker:
        diagnostics = row.get("lightgbm_diagnostics") or {}
        stats = diagnostics.get("prediction_stats") if isinstance(diagnostics, dict) else None
        pred_std = stats.get("std") if isinstance(stats, dict) else None
        pred_min = stats.get("min") if isinstance(stats, dict) else None
        pred_max = stats.get("max") if isinstance(stats, dict) else None
        std_text = f"{pred_std:.8f}" if pred_std is not None else "n/a"
        min_text = f"{pred_min:.8f}" if pred_min is not None else "n/a"
        max_text = f"{pred_max:.8f}" if pred_max is not None else "n/a"
        lines.append(
            f"{row.get('ticker', ''):<6}  "
            f"{_format_per_ticker_rmse(row, 'arima', 10)}  "
            f"{_format_per_ticker_rmse(row, 'trend', 10)}  "
            f"{_format_per_ticker_rmse(row, 'lightgbm', 13)}  "
            f"{_format_per_ticker_rmse(row, 'naive', 10)}  "
            f"{std_text:>22}  {min_text:>23}  {max_text:>23}"
        )
    return "\n".join(lines)


def run_lightgbm_backtest_comparison(
    tickers: list[str] | None = None,
    sample_size: int = 30,
    period: str | None = None,
    interval: str = "1d",
    horizon: int = DEFAULT_WALK_FORWARD_HORIZON,
    random_seed: int | None = None,
    ticker_offset: int = 0,
    evaluate_naive_baseline: bool = False,
    lightgbm_diagnostics: bool = False,
) -> dict:
    config = get_walk_forward_window_config(horizon)
    resolved_period = str(period or config.default_period)
    if tickers is None:
        try:
            universe = get_sp500_tickers()
        except Exception as exc:
            logger.warning("Falling back to built-in sample tickers: %s", exc)
            universe = DEFAULT_SAMPLE_TICKERS
    else:
        universe = tickers

    sample = select_sample_tickers(
        universe,
        sample_size=sample_size,
        random_seed=random_seed,
        ticker_offset=ticker_offset,
    )

    per_ticker: list[dict] = []
    for ticker in sample:
        data = get_stock_data(ticker, period=resolved_period, interval=interval)
        result = run_walk_forward(
            ticker,
            data,
            horizon=int(config.horizon),
            evaluate_lightgbm=True,
            evaluate_naive_baseline=evaluate_naive_baseline,
            lightgbm_diagnostics=lightgbm_diagnostics,
        )
        per_ticker.append({"ticker": ticker, **result})

    summary = summarize_backtest_results(per_ticker)
    return {
        "horizon": int(config.horizon),
        "period": resolved_period,
        "window_config": {
            "train_len": int(config.train_len),
            "test_len": int(config.test_len),
            "stride": int(config.stride),
            "max_history_rows": int(config.max_history_rows),
        },
        "sample_tickers": sample,
        "per_ticker": per_ticker,
        "summary": summary,
    }


def format_comparison_summary(summary: dict) -> str:
    models = summary.get("models", {})
    lines = [
        "Model      Mean RMSE  Median RMSE  Tickers  Windows",
        "---------  ---------  -----------  -------  -------",
    ]
    ordered_models = ["arima", "trend", "lightgbm"]
    if "naive" in models:
        ordered_models.append("naive")
    for model in ordered_models:
        values = models.get(model, {})
        mean_rmse = values.get("mean_rmse")
        median_rmse = values.get("median_rmse")
        lines.append(
            f"{model:<9}  "
            f"{(f'{mean_rmse:.6f}' if mean_rmse is not None else 'n/a'):>9}  "
            f"{(f'{median_rmse:.6f}' if median_rmse is not None else 'n/a'):>11}  "
            f"{int(values.get('tickers_evaluated', 0)):>7}  "
            f"{int(values.get('windows_evaluated', 0)):>7}"
        )
    wins = summary.get("lightgbm_wins_vs_both", {})
    naive_wins = summary.get("lightgbm_wins_vs_naive", {})
    lines.extend(
        [
            "",
            f"LightGBM better than both ARIMA and trend: {wins.get('wins', 0)}/{wins.get('comparable_tickers', 0)} "
            f"tickers ({wins.get('win_pct', 0.0):.2f}%)",
        ]
    )
    if int(naive_wins.get("comparable_tickers", 0)) > 0:
        lines.append(
            f"LightGBM better than naive no-change baseline: {naive_wins.get('wins', 0)}/"
            f"{naive_wins.get('comparable_tickers', 0)} tickers ({naive_wins.get('win_pct', 0.0):.2f}%)"
        )
    return "\n".join(lines)
