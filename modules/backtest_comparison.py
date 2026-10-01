from __future__ import annotations

import math
import random

import numpy as np

from modules.backtester import DEFAULT_WALK_FORWARD_HORIZON, get_walk_forward_window_config, run_walk_forward
from modules.data_fetcher import get_sp500_tickers, get_stock_data
from modules.logger import get_logger

logger = get_logger(__name__)

try:
    # scipy is an existing transitive dependency of statsmodels (already a
    # direct requirement for ARIMA), so this gives an exact Student's-t
    # p-value whenever it's importable without adding a new direct
    # dependency. Falls back to a normal-distribution approximation (via
    # math.erf) when scipy isn't importable, which is still a reasonable
    # p-value estimate for the ticker-sample sizes used here.
    from scipy import stats as _scipy_stats

    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover
    SCIPY_AVAILABLE = False

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


def _two_sided_p_value(t_stat: float, df: int) -> float | None:
    """Two-sided p-value for a t-statistic with ``df`` degrees of freedom."""
    if df < 1:
        return None
    if SCIPY_AVAILABLE:
        return float(2.0 * _scipy_stats.t.sf(abs(t_stat), df))
    # Normal-distribution approximation (valid for reasonably large df,
    # acceptable given this tool only runs when scipy is unavailable).
    return float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2.0)))))


def diebold_mariano_test(per_ticker: list[dict], model_a: str, model_b: str) -> dict:
    """Pooled cross-ticker significance test comparing two models' walk-forward
    RMSE, in the spirit of the Diebold-Mariano test.

    Per-ticker window counts in ``run_walk_forward`` can be as low as 1-4,
    which makes a single ticker's raw RMSE difference unreliable evidence of
    real forecasting skill on its own. This pools one loss-differential
    observation per comparable ticker -- ``model_a``'s squared RMSE minus
    ``model_b``'s squared RMSE -- across every ticker where both models have
    at least one evaluated window, then runs a one-sample t-test (mean
    difference / standard error) against the null hypothesis of no skill
    difference. A negative ``mean_diff`` favors ``model_a`` (lower loss);
    positive favors ``model_b``.

    Returns a dict with ``n_tickers``, ``mean_diff``, ``t_statistic``,
    ``p_value``, and ``significant_at_5pct`` (None/False when fewer than 2
    comparable tickers are available, since a t-test needs at least 2
    observations to estimate variance).
    """
    diffs: list[float] = []
    for row in per_ticker:
        a_windows = int(row.get(f"{model_a}_windows", 0) or 0)
        b_windows = int(row.get(f"{model_b}_windows", 0) or 0)
        a_rmse = row.get(f"{model_a}_rmse")
        b_rmse = row.get(f"{model_b}_rmse")
        if a_windows > 0 and b_windows > 0 and a_rmse is not None and b_rmse is not None:
            diffs.append(float(a_rmse) ** 2 - float(b_rmse) ** 2)

    result: dict = {
        "model_a": model_a,
        "model_b": model_b,
        "n_tickers": int(len(diffs)),
        "mean_diff": None,
        "t_statistic": None,
        "p_value": None,
        "significant_at_5pct": None,
    }
    if len(diffs) < 2:
        return result

    diffs_arr = np.array(diffs, dtype=float)
    mean_diff = float(np.mean(diffs_arr))
    std_diff = float(np.std(diffs_arr, ddof=1))
    result["mean_diff"] = round(mean_diff, 8)
    if std_diff == 0:
        result["significant_at_5pct"] = bool(mean_diff != 0)
        result["p_value"] = 0.0 if mean_diff != 0 else 1.0
        return result

    se = std_diff / math.sqrt(len(diffs_arr))
    t_stat = mean_diff / se
    p_value = _two_sided_p_value(t_stat, df=len(diffs_arr) - 1)
    result["t_statistic"] = round(float(t_stat), 6)
    result["p_value"] = round(float(p_value), 6) if p_value is not None else None
    result["significant_at_5pct"] = bool(p_value is not None and p_value < 0.05)
    return result


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
    has_path_rmse = any(f"{model}_path_rmse" in row for row in per_ticker for model in ("arima", "trend", "lightgbm"))
    rmse_values: dict[str, list[tuple[float, int]]] = {model: [] for model in models}
    path_rmse_values: dict[str, list[tuple[float, int]]] = {model: [] for model in ("arima", "trend", "lightgbm")}
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
            if has_path_rmse and model in path_rmse_values:
                path_rmse = row.get(f"{model}_path_rmse")
                if windows > 0 and path_rmse is not None:
                    path_rmse_values[model].append((float(path_rmse), windows))

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
            **(
                {
                    "mean_path_rmse": _weighted_mean(path_rmse_values[model]),
                    "median_path_rmse": _weighted_median(path_rmse_values[model]),
                }
                if has_path_rmse and model in path_rmse_values
                else {}
            ),
        }
        for model in models
    }
    win_pct = round((lightgbm_wins / lightgbm_compared) * 100.0, 2) if lightgbm_compared > 0 else 0.0
    naive_win_pct = (
        round((lightgbm_vs_naive_wins / lightgbm_vs_naive_compared) * 100.0, 2)
        if lightgbm_vs_naive_compared > 0
        else 0.0
    )
    significance_tests = [
        diebold_mariano_test(per_ticker, "lightgbm", "arima"),
        diebold_mariano_test(per_ticker, "lightgbm", "trend"),
    ]
    if "naive" in models:
        significance_tests.append(diebold_mariano_test(per_ticker, "lightgbm", "naive"))
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
        "significance_tests": significance_tests,
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
    evaluate_path_rmse: bool = False,
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
            evaluate_path_rmse=evaluate_path_rmse,
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
    significance_tests = summary.get("significance_tests") or []
    if significance_tests:
        lines.extend(["", "Significance tests (pooled cross-ticker, Diebold-Mariano style):"])
        for test in significance_tests:
            n_tickers = test.get("n_tickers", 0)
            if n_tickers < 2:
                lines.append(
                    f"  {test.get('model_a')} vs {test.get('model_b')}: insufficient comparable tickers "
                    f"({n_tickers}) for a significance test"
                )
                continue
            p_value = test.get("p_value")
            significant = test.get("significant_at_5pct")
            verdict = "significant" if significant else "not significant"
            lines.append(
                f"  {test.get('model_a')} vs {test.get('model_b')}: mean_diff={test.get('mean_diff'):.8f} "
                f"t={test.get('t_statistic')} p={p_value if p_value is not None else 'n/a'} "
                f"({verdict} at 5%, n={n_tickers} tickers)"
            )
    return "\n".join(lines)
