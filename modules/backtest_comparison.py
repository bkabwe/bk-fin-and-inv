from __future__ import annotations

from modules.backtester import run_walk_forward
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


def run_lightgbm_backtest_comparison(
    tickers: list[str] | None = None,
    sample_size: int = 30,
    period: str = "2y",
    interval: str = "1d",
    evaluate_naive_baseline: bool = False,
    lightgbm_diagnostics: bool = False,
) -> dict:
    if tickers is None:
        try:
            sample = get_sp500_tickers()[:sample_size]
        except Exception as exc:
            logger.warning("Falling back to built-in sample tickers: %s", exc)
            sample = DEFAULT_SAMPLE_TICKERS[:sample_size]
    else:
        sample = tickers[:sample_size]

    per_ticker: list[dict] = []
    for ticker in sample:
        data = get_stock_data(ticker, period=period, interval=interval)
        result = run_walk_forward(
            ticker,
            data,
            evaluate_lightgbm=True,
            evaluate_naive_baseline=evaluate_naive_baseline,
            lightgbm_diagnostics=lightgbm_diagnostics,
        )
        per_ticker.append({"ticker": ticker, **result})

    summary = summarize_backtest_results(per_ticker)
    return {"sample_tickers": sample, "per_ticker": per_ticker, "summary": summary}


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
