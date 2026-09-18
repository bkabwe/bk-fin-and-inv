from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.data_fetcher import get_stock_data
from modules.email_reports import (
    csv_attachment,
    render_html_table,
    render_legend,
    render_metric_tiles,
    render_report_html,
    send_brevo_email,
)
from modules.fred_client import get_macro_feature_table
from modules.lightgbm_batch import (
    batch_asset_urls_from_manifest,
    fetch_live_manifest,
    load_return_model_batch_from_urls,
    resolve_release_repository,
    shard_asset_urls_by_index,
)
from modules.prediction_tracker import SUBSCORE_ROW_FIELDS, record_predictions_from_scan
from modules.profit_opportunities import (
    DEFAULT_FAST_SCREEN_MARGIN,
    HORIZON_SETTINGS as _ALL_HORIZON_SETTINGS,
    filter_by_upside,
    profit_row_from_analysis,
    scan_profit_opportunities,
)
from modules.scoring_engine import analyze_stock, fast_screen_score
from modules.screener import DEFAULT_FAST_SCREEN_MARGIN as SCREENER_FAST_SCREEN_MARGIN, screener_row_from_analysis

# Scheduled scans only cover short/medium term horizons; keep the same
# restricted subset here (rather than the full three-horizon superset in
# modules.profit_opportunities, which also serves the long-term Streamlit page).
HORIZON_SETTINGS = {k: v for k, v in _ALL_HORIZON_SETTINGS.items() if k in ("short_term", "medium_term")}
DEFAULT_MIN_SCORE = 50
DEFAULT_MIN_UPSIDE_PCT = 15.0
DEFAULT_FAST_SCREEN_PROXY_THRESHOLD = 15
TOP_RESULTS_LIMIT = 75
PREVIEW_LIMIT = 10
# Each ticker's `_run_shard_ticker` call is independent (no shared mutable
# state) and is I/O-bound (network fetches, live FRED/SEC calls) far more
# than CPU-bound, matching `modules.screener.run_screener`'s existing
# ThreadPoolExecutor pattern -- mirrored here so a shard's ~tickers-per-shard
# analyses overlap their I/O wait instead of running strictly sequentially.
DEFAULT_SCAN_SHARD_MAX_WORKERS = 8

# The two fast-screen thresholds `run_screener` and
# `scan_and_filter_profit_opportunities` each applied independently before
# `run_scan_shard` unified them into a single per-ticker pass. A ticker can
# clear one and not the other, so `_run_shard_ticker` evaluates both from one
# shared `fast_screen_score` call rather than reusing a single threshold.
SCREENER_FAST_SCREEN_THRESHOLD = DEFAULT_MIN_SCORE - SCREENER_FAST_SCREEN_MARGIN  # 50 - 15 = 35
PROFIT_FAST_SCREEN_THRESHOLD = DEFAULT_FAST_SCREEN_PROXY_THRESHOLD  # matches scan_and_filter_profit_opportunities's effective threshold


@contextmanager
def live_scoring_context(manifest: dict[str, Any], batch: dict[str, Any], shared_macro_table: pd.DataFrame):
    import modules.backtester as backtester
    import modules.scoring_engine as scoring_engine

    original_build_feature_table = scoring_engine.build_feature_table

    def _build_feature_table_with_shared_macro(*args, **kwargs):
        kwargs.setdefault("shared_macro_table", shared_macro_table)
        return original_build_feature_table(*args, **kwargs)

    with (
        patch.object(scoring_engine, "_load_live_lightgbm_manifest", return_value=manifest),
        patch.object(scoring_engine, "_load_live_lightgbm_batch", return_value=batch),
        # The scan scripts preload the full combined batch above (`batch`)
        # once for every ticker in the manifest, so a per-ticker shard
        # download here would be pure redundant network I/O. Force shard
        # lookups to report "not found" so `_load_live_lightgbm_models`
        # always falls through to the already-loaded combined batch instead.
        patch.object(scoring_engine, "_load_live_lightgbm_shard_batch", return_value={}),
        patch.object(scoring_engine, "build_feature_table", side_effect=_build_feature_table_with_shared_macro),
        # `modules.backtester` imports its own `build_feature_table` reference
        # (rather than calling through `scoring_engine`), so `run_walk_forward`'s
        # per-window LightGBM feature building needs this patched separately --
        # otherwise every walk-forward window re-fetches macro data live from
        # FRED for its own date slice instead of reusing the shared table
        # already fetched once for the whole scan (up to ~33 redundant FRED
        # HTTP calls per ticker: 3 series x up to 11 windows across the 30d/180d
        # horizons). Same underlying function, same resulting feature values --
        # this only removes redundant network I/O, not a modeling change.
        patch.object(backtester, "build_feature_table", side_effect=_build_feature_table_with_shared_macro),
    ):
        yield


def manifest_tickers(manifest: dict[str, Any] | None) -> list[str]:
    return sorted(
        str(ticker).strip().upper()
        for ticker in ((manifest or {}).get("tickers") or {})
        if str(ticker).strip()
    )


def manifest_shard_count(manifest: dict[str, Any] | None) -> int:
    return max(1, int(((manifest or {}).get("latest_batch") or {}).get("shard_count") or 1))


def tickers_for_shard(manifest: dict[str, Any] | None, shard_index: int, shard_count: int) -> list[str]:
    """Return the sorted tickers assigned to `shard_index` of `shard_count`.

    When `shard_count <= 1` (old manifests without shard partitioning, or a
    single-shard run) every manifest ticker belongs to the one shard. Any
    ticker missing a `shard_index` (shouldn't happen once shards exist, see
    `build_live_manifest`) defaults to shard 0 rather than being dropped.
    """
    tickers_meta = (manifest or {}).get("tickers") or {}
    if shard_count <= 1:
        return manifest_tickers(manifest)
    selected = []
    for ticker, meta in tickers_meta.items():
        clean_ticker = str(ticker).strip().upper()
        if not clean_ticker:
            continue
        ticker_shard = int((meta or {}).get("shard_index") or 0)
        if ticker_shard == int(shard_index):
            selected.append(clean_ticker)
    return sorted(selected)


def _shard_batch_asset_urls(manifest: dict[str, Any], shard_index: int, shard_count: int) -> list[str]:
    if shard_count > 1:
        urls = shard_asset_urls_by_index(manifest, shard_index)
        if urls:
            return urls
    return batch_asset_urls_from_manifest(manifest)


def discover_scan_inputs(repository: str | None, output_manifest: Path, output_macro: Path) -> dict[str, Any]:
    manifest = fetch_live_manifest(repository=resolve_release_repository(repository))
    tickers = manifest_tickers(manifest)
    if not tickers:
        raise RuntimeError("Live manifest is empty or unreachable; refusing to run an empty scan")

    shared_macro_table = fetch_shared_macro_table()

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    output_macro.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(shared_macro_table, output_macro)

    return {"tickers": tickers, "shard_count": manifest_shard_count(manifest)}


def fetch_shared_macro_table(lookback_days: int = 365 * 5) -> pd.DataFrame:
    end_date = datetime.now(UTC).date()
    start_date = end_date - timedelta(days=int(lookback_days))
    macro_table = get_macro_feature_table(start_date, end_date, realtime_end=end_date)
    # stderr, not stdout: the `discover` subcommand's stdout is redirected
    # straight into `$GITHUB_OUTPUT` by the workflow, which requires every
    # stdout line to be a strict `key=value` pair.
    print(f"Loaded shared FRED macro table rows: {len(macro_table)}", file=sys.stderr)
    return macro_table


def _require_lightgbm_backtested(results: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows whose LightGBM ensemble contribution (if any) for this
    horizon wasn't backed by genuine per-ticker walk-forward backtest
    evidence -- i.e. rows that either had no live LightGBM prediction at all,
    or would only have relied on the small fixed fallback weight (see
    modules.scoring_engine._has_lightgbm_backtest_support).

    Every profit-opportunities result the scheduled report ships must have a
    real, backtested LightGBM signal behind it. Returns
    `(filtered_results, dropped_count)`.
    """
    if results is None or results.empty or "_lightgbm_backtested" not in results.columns:
        return results, 0
    confirmed_mask = results["_lightgbm_backtested"] == True  # noqa: E712
    dropped = int((~confirmed_mask).sum())
    return results[confirmed_mask].reset_index(drop=True), dropped


def scan_and_filter_profit_opportunities(
    tickers: list[str],
    horizon: str,
    *,
    min_upside_pct: float = DEFAULT_MIN_UPSIDE_PCT,
    max_results: int | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Scan `tickers` and filter by upside without stripping internal scoring
    fields or recording predictions.

    Rows keep their internal `_rsi`/`SUBSCORE_ROW_FIELDS` columns intact, and
    `max_results` defaults to unbounded, so a sharded caller can defer both
    final top-N truncation and prediction-recording until after all shard
    partials are merged (see `finalize_profit_results`).
    """
    scanned = scan_profit_opportunities(
        tickers,
        horizon,
        use_fast_screen=True,
        fast_screen_base=DEFAULT_FAST_SCREEN_PROXY_THRESHOLD + DEFAULT_FAST_SCREEN_MARGIN,
        fast_screen_margin=DEFAULT_FAST_SCREEN_MARGIN,
        parallel=False,
        # 5y (not just 1y) so a genuine 180d LightGBM walk-forward backtest has
        # enough history to run inside analyze_stock; it's still a single
        # Polygon fetch per ticker, just a larger payload.
        prefetch_price_period="5y",
    )
    confirmed, lightgbm_unconfirmed_count = _require_lightgbm_backtested(scanned)
    filtered = filter_by_upside(confirmed, horizon, min_upside_pct=min_upside_pct, max_results=max_results)
    stats_base = {
        "scanned_count": int(scanned.attrs.get("scanned_count", len(tickers))),
        "fast_filtered_count": int(scanned.attrs.get("fast_filtered_count", 0)),
        "passed_fast_screen_count": int(scanned.attrs.get("fully_analyzed_count", 0)),
        "failed_count": int(scanned.attrs.get("failed_count", 0)),
        "lightgbm_unconfirmed_count": lightgbm_unconfirmed_count,
    }
    return filtered, stats_base


def finalize_profit_results(
    results: pd.DataFrame,
    horizon: str,
    *,
    record: bool = True,
    stats_base: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Strip internal scoring fields from a filtered/sorted/truncated profit
    results frame, optionally recording predictions from the pre-strip rows.

    Recording must happen exactly once, on the final globally-truncated set,
    so callers merging multiple scan shards should only pass `record=True`
    here at the reduce stage (not per-shard).
    """
    recorded_rows = results.to_dict("records") if not results.empty else []
    drop_columns = ["_rsi", "_lightgbm_backtested", *SUBSCORE_ROW_FIELDS]
    if horizon == "short_term":
        # Confidence (Full/Limited/Technical Only) is derived from a
        # ticker-wide models_used list that includes Fundamental Fair Value
        # even though the short-term ensemble never uses it -- the label is
        # effectively meaningless at this horizon, so it's dropped from the
        # short-term dashboard/attachment (medium-term keeps it).
        drop_columns.append("Confidence")
    display_results = results.drop(columns=drop_columns, errors="ignore")

    if record and recorded_rows:
        recorded_count = record_predictions_from_scan(recorded_rows, horizon=horizon, source="profit_opportunities")
    else:
        recorded_count = 0

    stats = dict(stats_base or {})
    stats["qualified_count"] = len(display_results)
    stats["recorded_count"] = recorded_count
    return display_results, stats


def run_profit_opportunities_scan(
    tickers: list[str],
    horizon: str,
    *,
    min_upside_pct: float = DEFAULT_MIN_UPSIDE_PCT,
    max_results: int | None = TOP_RESULTS_LIMIT,
    record: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Single-shot scan + filter + finalize, used by non-sharded callers."""
    filtered, stats_base = scan_and_filter_profit_opportunities(
        tickers, horizon, min_upside_pct=min_upside_pct, max_results=max_results
    )
    return finalize_profit_results(filtered, horizon, record=record, stats_base=stats_base)


def _format_value(value: Any, *, currency: bool = False, pct: bool = False) -> str:
    if value is None or value == "":
        return "—"
    try:
        number = float(value)
    except Exception:
        return str(value)
    if currency:
        return f"${number:,.2f}"
    if pct:
        return f"{number:,.2f}%"
    return f"{number:,.2f}"


def _preview_rows(df: pd.DataFrame, columns: list[str], *, limit: int = PREVIEW_LIMIT) -> list[dict[str, object]]:
    preview = df.head(limit).copy() if not df.empty else pd.DataFrame(columns=columns)
    records = preview.to_dict("records") if not preview.empty else []
    formatted = []
    for row in records:
        formatted.append(
            {
                **row,
                "Score": _format_value(row.get("Score")),
                "Current Price": _format_value(row.get("Current Price"), currency=True),
                "Target Price": _format_value(row.get("Target Price"), currency=True),
                "Projected Upside %": _format_value(row.get("Projected Upside %"), pct=True),
            }
        )
    return formatted


def _profit_subsection(
    title: str,
    sort_label: str,
    results: pd.DataFrame,
    profit_stats: dict[str, int],
    horizon: str,
    *,
    show_recorded_tile: bool = False,
) -> str:
    top = results.iloc[0] if not results.empty else None
    avg_upside = float(results["Projected Upside %"].mean()) if not results.empty else None
    median_upside = float(results["Projected Upside %"].median()) if not results.empty else None

    columns = ["Ticker", "Company", "Score", "Current Price", "Target Price", "Projected Upside %"]
    tiles = [
        {"label": "Top ticker", "value": str(top["Ticker"]) if top is not None else "—"},
        {
            "label": "Avg / median upside",
            "value": ("—" if avg_upside is None else f"{avg_upside:.2f}% / {median_upside:.2f}%"),
        },
    ]
    if show_recorded_tile:
        tiles = [
            {"label": "Passed fast-screen", "value": str(int(profit_stats.get("passed_fast_screen_count") or 0))},
            {"label": "Recorded predictions", "value": str(int(profit_stats.get("recorded_count") or 0))},
            {
                "label": "LightGBM-unconfirmed dropped",
                "value": str(int(profit_stats.get("lightgbm_unconfirmed_count") or 0)),
            },
            *tiles,
        ]

    legend_html = ""
    if horizon != "short_term":
        # Confidence (Full/Limited/Technical Only) is derived from a
        # ticker-wide models_used list that includes models the short-term
        # ensemble never uses (see modules.scoring_engine), so it's omitted
        # entirely from short-term output where the label is close to
        # meaningless; medium-term keeps it.
        columns = [*columns, "Confidence"]
        tiles.append(
            {
                "label": "High-confidence picks",
                "value": str(int((results["Confidence"] == "Full").sum())) if not results.empty else "0",
            }
        )
        legend_html = (
            '<div style="margin-top:14px;"><div style="color:#a9bdd7;font-size:12px;text-transform:uppercase;'
            'letter-spacing:0.06em;margin-bottom:6px;">Confidence legend</div>'
            f"{render_legend([{'label': 'Full', 'description': 'ARIMA, trend, and fundamental/DCF models all contributed to the projection — highest-confidence tier.'}, {'label': 'Limited', 'description': 'Partial model coverage (e.g. missing fundamentals or a fitted trend) — treat with more caution.'}, {'label': 'Technical Only', 'description': 'Speculative/OTC ticker with no fundamental EPS data — projection is technical-trend only, higher risk.'}])}"
            "</div>"
        )

    return (
        f'<h3 style="margin:18px 0 8px 0;color:#f4fff8;">{title}</h3>'
        f'<p style="margin:0 0 12px 0;color:#a9bdd7;">Ranked by {sort_label}, all LightGBM-backtest-confirmed for '
        f"{HORIZON_SETTINGS[horizon]['label']}. Top 10 preview below; full top 75 is attached as CSV.</p>"
        f"{render_metric_tiles(tiles)}"
        f'<div style="margin-top:18px;">{render_html_table(columns, _preview_rows(results, columns))}</div>'
        f"{legend_html}"
    )


def build_scan_report(
    manifest: dict[str, Any],
    tickers: list[str],
    screener_results: pd.DataFrame,
    profit_by_upside: pd.DataFrame,
    profit_by_score: pd.DataFrame,
    profit_stats: dict[str, int],
    horizon: str,
    run_date: str,
) -> str:
    current_run = (manifest or {}).get("current_run") or {}
    screener_top = screener_results.iloc[0] if not screener_results.empty else None
    screener_avg_score = float(screener_results["Score"].mean()) if not screener_results.empty else None
    screener_median_score = float(screener_results["Score"].median()) if not screener_results.empty else None

    sections = [
        (
            f"<h2 style=\"margin:0 0 10px 0;color:#f4fff8;\">Screener top 75</h2>"
            f"<p style=\"margin:0 0 16px 0;color:#a9bdd7;\">Score-based scan for {HORIZON_SETTINGS[horizon]['label']} workflows. Top 10 preview below; full top 75 is attached as CSV.</p>"
            f"{render_metric_tiles([{'label': 'Passed fast-screen', 'value': str(int(screener_results.attrs.get('fully_analyzed_count') or 0))}, {'label': 'Fast-filtered', 'value': str(int(screener_results.attrs.get('fast_filtered_count') or 0))}, {'label': 'Top ticker', 'value': str(screener_top['Ticker']) if screener_top is not None else '—'}, {'label': 'Avg / median score', 'value': ('—' if screener_avg_score is None else f'{screener_avg_score:.1f} / {screener_median_score:.1f}')}, {'label': 'High-confidence picks', 'value': str(int((screener_results['Score'] >= 80).sum())) if not screener_results.empty else '0'}])}"
            f"<div style=\"margin-top:18px;\">{render_html_table(['Ticker', 'Company', 'Score', 'Recommendation', 'Current Price', 'Target Price'], _preview_rows(screener_results, ['Ticker', 'Company', 'Score', 'Recommendation', 'Current Price', 'Target Price']))}</div>"
        ),
        (
            f"<h2 style=\"margin:0 0 10px 0;color:#f4fff8;\">Profit opportunities top 75</h2>"
            f"<p style=\"margin:0 0 16px 0;color:#a9bdd7;\">Split into two rankings over the same qualifying pool for {HORIZON_SETTINGS[horizon]['label']}: one by projected upside, one by overall score. Full top 75 of each is attached as CSV.</p>"
            + _profit_subsection(
                "By upside", "projected upside %", profit_by_upside, profit_stats, horizon, show_recorded_tile=True
            )
            + _profit_subsection("By score", "overall score", profit_by_score, profit_stats, horizon)
        ),
    ]

    return render_report_html(
        title=f"{HORIZON_SETTINGS[horizon]['label']} scan report",
        subtitle=f"Run date {run_date} · ticker pool {len(tickers)} · manifest-trained universe from the promoted LightGBM release",
        metrics=[
            {"label": "Tickers scanned", "value": str(len(tickers))},
            {"label": "Newly trained", "value": str(len(current_run.get('trained_tickers') or []))},
            {"label": "Carried forward", "value": str(len(current_run.get('carried_forward_tickers') or []))},
            {"label": "Dropped stale", "value": str(len(current_run.get('dropped_stale_tickers') or []))},
        ],
        sections=sections,
    )


def _run_shard_ticker(ticker: str, horizon: str) -> dict[str, Any]:
    """Run a single shared full analysis for `ticker` and derive both a
    screener-style row and a horizon-specific profit-opportunities row from
    it.

    This replaces what used to be two fully independent `analyze_stock()`
    passes per ticker inside `run_scan_shard` -- once via `run_screener`
    (1y data) and once via `scan_and_filter_profit_opportunities` (5y data).
    Both passes fed into `get_price_projections`, which runs a genuine 30d
    *and* 180d ARIMA/LightGBM walk-forward backtest, so every ticker's
    backtests were being computed twice per shard; with ~277 tickers/shard
    that duplication was the root cause of shards exceeding GitHub's 6h job
    limit once PR 58 fixed a caching bug that had been silently swallowing
    these calls. Fetching one 5y OHLCV window (a superset of the screener's
    former 1y window) and calling `analyze_stock()` exactly once collapses
    that back down to one backtest pass per ticker per shard, with zero
    change to the models, windows, or evidence used.

    Returns a status dict:
    ``{"screener_row": dict | None, "profit_row": dict | None,
       "screener_fast_filtered": bool, "profit_fast_filtered": bool,
       "screener_fully_analyzed": bool, "profit_fully_analyzed": bool,
       "failed": bool, "reason": str | None}``.

    The screener and profit-opportunities fast-screen thresholds
    (`SCREENER_FAST_SCREEN_THRESHOLD`/`PROFIT_FAST_SCREEN_THRESHOLD`) are
    independent -- a ticker can clear one and not the other -- so both are
    checked against a single shared `fast_screen_score` call, and the
    expensive `analyze_stock()` call is skipped entirely only when a ticker
    fails both.
    """
    outcome: dict[str, Any] = {
        "screener_row": None,
        "profit_row": None,
        "screener_fast_filtered": False,
        "profit_fast_filtered": False,
        "screener_fully_analyzed": False,
        "profit_fully_analyzed": False,
        "failed": False,
        "reason": None,
    }
    try:
        try:
            # 5y (not 1y) so a genuine 180d LightGBM walk-forward backtest has
            # enough history to run inside analyze_stock -- see the identical
            # rationale documented at modules.scoring_engine's
            # _get_price_projections_core, which this mirrors.
            price_data = get_stock_data(ticker, period="5y", interval="1d")
        except Exception:
            # Fall through with price_data=None: fast_screen_score and
            # analyze_stock each retry their own default-period fetch below,
            # matching scan_profit_opportunities's existing graceful
            # degradation on a prefetch failure.
            price_data = None

        fast_score, err = fast_screen_score(ticker, data=price_data)
        if err is not None:
            outcome["failed"] = True
            outcome["reason"] = f"fast-screen error: {err}"
            return outcome

        passes_screener = fast_score >= SCREENER_FAST_SCREEN_THRESHOLD
        passes_profit = fast_score >= PROFIT_FAST_SCREEN_THRESHOLD
        outcome["screener_fast_filtered"] = not passes_screener
        outcome["profit_fast_filtered"] = not passes_profit
        if not passes_screener and not passes_profit:
            return outcome

        analyze_kwargs: dict[str, Any] = {"investment_horizon": horizon}
        if price_data is not None:
            analyze_kwargs["data_override"] = price_data
            analyze_kwargs["projection_data_override"] = price_data
        analysis = analyze_stock(ticker, **analyze_kwargs)

        if passes_screener:
            outcome["screener_fully_analyzed"] = True
            outcome["screener_row"] = screener_row_from_analysis(analysis, DEFAULT_MIN_SCORE)
        if passes_profit:
            outcome["profit_fully_analyzed"] = True
            outcome["profit_row"] = profit_row_from_analysis(ticker, horizon, analysis)
        return outcome
    except Exception as exc:
        outcome["failed"] = True
        outcome["reason"] = str(exc) or type(exc).__name__
        return outcome


def run_scan_shard(
    horizon: str,
    manifest: dict[str, Any],
    macro_table: pd.DataFrame,
    shard_index: int,
    shard_count: int,
    max_workers: int = DEFAULT_SCAN_SHARD_MAX_WORKERS,
) -> dict[str, Any]:
    """Run the screener + profit-opportunities analysis for one ticker shard.

    Each ticker is analyzed exactly once via `_run_shard_ticker`, which
    derives both the screener-style row and the horizon-specific
    profit-opportunities row from a single `analyze_stock()` call (see its
    docstring for why this eliminates the duplicate-backtest root cause of
    shard timeouts).

    Tickers within the shard are processed concurrently via a
    `ThreadPoolExecutor` (mirroring `modules.screener.run_screener`'s
    existing pattern): `_run_shard_ticker` is a pure, side-effect-free
    function per ticker (no shared mutable state passed in), and its cost is
    dominated by I/O-bound work (Polygon/SEC/FRED HTTP fetches) rather than
    CPU, so overlapping tickers' I/O wait materially shortens shard wall-clock
    time without changing what is computed for any individual ticker.

    Returns unrecorded, untruncated partial results (internal scoring fields
    intact) plus raw per-shard stats, ready to be merged with other shards'
    partials and finalized exactly once at the reduce stage.
    """
    empty_screener_attrs = {"fast_filtered_count": 0, "fully_analyzed_count": 0, "failed_count": 0, "source_ticker_count": 0}
    empty_profit_stats = {
        "scanned_count": 0,
        "fast_filtered_count": 0,
        "passed_fast_screen_count": 0,
        "failed_count": 0,
        "lightgbm_unconfirmed_count": 0,
    }

    shard_tickers = tickers_for_shard(manifest, shard_index, shard_count)
    if not shard_tickers:
        print(f"Shard {shard_index}/{shard_count} has no assigned tickers; writing empty partial output")
        return {
            "shard_index": shard_index,
            "screener": pd.DataFrame(),
            "screener_attrs": empty_screener_attrs,
            "profit": pd.DataFrame(),
            "profit_stats_base": empty_profit_stats,
        }

    asset_urls = _shard_batch_asset_urls(manifest, shard_index, shard_count)
    if not asset_urls:
        raise RuntimeError(f"No batch asset URLs resolved for shard {shard_index}/{shard_count}")
    batch = load_return_model_batch_from_urls(asset_urls)
    if not ((batch or {}).get("models") or {}):
        raise RuntimeError(f"Shard {shard_index}/{shard_count} batch artifact is empty or unreachable")

    total = len(shard_tickers)
    screener_rows: list[dict] = []
    profit_rows: list[dict] = []
    screener_fast_filtered_count = 0
    screener_fully_analyzed_count = 0
    profit_fast_filtered_count = 0
    profit_fully_analyzed_count = 0
    # A single shared fast-screen/analyze_stock call now backs both the
    # screener and profit-opportunities views of each ticker (see
    # `_run_shard_ticker`), so a failure is necessarily shared too -- unlike
    # the old independent `run_screener`/`scan_and_filter_profit_opportunities`
    # calls, there's no longer a scenario where one path fails and the other
    # succeeds for the same ticker.
    failed_count = 0
    processed_count = 0
    _lock = threading.Lock()

    start_time = time.monotonic()

    def _process_ticker(ticker: str) -> None:
        nonlocal screener_fast_filtered_count, screener_fully_analyzed_count
        nonlocal profit_fast_filtered_count, profit_fully_analyzed_count
        nonlocal failed_count, processed_count

        result = _run_shard_ticker(ticker, horizon)
        with _lock:
            if result["failed"]:
                failed_count += 1
            if result["screener_fast_filtered"]:
                screener_fast_filtered_count += 1
            if result["screener_fully_analyzed"]:
                screener_fully_analyzed_count += 1
            if result["screener_row"] is not None:
                screener_rows.append(result["screener_row"])
            if result["profit_fast_filtered"]:
                profit_fast_filtered_count += 1
            if result["profit_fully_analyzed"]:
                profit_fully_analyzed_count += 1
            if result["profit_row"] is not None:
                profit_rows.append(result["profit_row"])

            processed_count += 1
            i = processed_count
            # ETA log line: cheap, run-log-only visibility so a future
            # per-ticker slowdown shows up long before it turns into a full
            # 6h job timeout (see notify-on-failure's timeout messaging).
            if i == 1 or i % 25 == 0 or i == total:
                elapsed = time.monotonic() - start_time
                per_ticker = elapsed / i
                print(
                    f"Shard {shard_index}/{shard_count} progress: {i}/{total} tickers processed "
                    f"| elapsed={elapsed / 60:.1f}m | ~{per_ticker:.1f}s/ticker "
                    f"| projected shard total ~{(per_ticker * total) / 60:.1f}m",
                    flush=True,
                )

    with (
        live_scoring_context(manifest, batch, macro_table),
        concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor,
    ):
        futures = {executor.submit(_process_ticker, ticker): ticker for ticker in shard_tickers}
        for future in concurrent.futures.as_completed(futures):
            ticker = futures[future]
            try:
                future.result()
            except Exception as exc:
                # `_run_shard_ticker` already catches and reports its own
                # per-ticker failures in its returned outcome dict, so
                # this is a defensive backstop for a truly unexpected
                # error escaping that boundary -- not the normal failure path.
                print(f"Shard {shard_index}/{shard_count}: unexpected error for {ticker}: {exc}", flush=True)

    screener_partial = pd.DataFrame(screener_rows) if screener_rows else pd.DataFrame()

    profit_scanned = pd.DataFrame(profit_rows) if profit_rows else pd.DataFrame()
    profit_confirmed, lightgbm_unconfirmed_count = _require_lightgbm_backtested(profit_scanned)
    # max_results=None: final top-N truncation is deferred to reduce_scan_shards
    # once all shards' partials are combined, matching the old
    # scan_and_filter_profit_opportunities(..., max_results=None) call here.
    profit_partial = filter_by_upside(profit_confirmed, horizon, min_upside_pct=DEFAULT_MIN_UPSIDE_PCT, max_results=None)

    print(
        f"Shard {shard_index}/{shard_count}: tickers={total} "
        f"screener_qualified={len(screener_partial)} profit_qualified={len(profit_partial)}"
    )
    return {
        "shard_index": shard_index,
        "screener": screener_partial,
        "screener_attrs": {
            "fast_filtered_count": screener_fast_filtered_count,
            "fully_analyzed_count": screener_fully_analyzed_count,
            "failed_count": failed_count,
            "source_ticker_count": total,
        },
        "profit": profit_partial,
        "profit_stats_base": {
            "scanned_count": total,
            "fast_filtered_count": profit_fast_filtered_count,
            "passed_fast_screen_count": profit_fully_analyzed_count,
            "failed_count": failed_count,
            "lightgbm_unconfirmed_count": lightgbm_unconfirmed_count,
        },
    }


def merge_shard_outputs(partial_dir: Path) -> dict[str, Any]:
    partial_files = sorted(Path(partial_dir).glob("*.joblib"))
    if not partial_files:
        raise RuntimeError(f"No scan-shard partial files found under {partial_dir}")

    screener_frames: list[pd.DataFrame] = []
    profit_frames: list[pd.DataFrame] = []
    screener_attrs_totals = {"fast_filtered_count": 0, "fully_analyzed_count": 0, "failed_count": 0, "source_ticker_count": 0}
    profit_stats_totals = {
        "scanned_count": 0,
        "fast_filtered_count": 0,
        "passed_fast_screen_count": 0,
        "failed_count": 0,
        "lightgbm_unconfirmed_count": 0,
    }

    for path in partial_files:
        partial = joblib.load(path)
        screener_df = partial.get("screener")
        if isinstance(screener_df, pd.DataFrame) and not screener_df.empty:
            screener_frames.append(screener_df)
        for key in screener_attrs_totals:
            screener_attrs_totals[key] += int((partial.get("screener_attrs") or {}).get(key, 0))

        profit_df = partial.get("profit")
        if isinstance(profit_df, pd.DataFrame) and not profit_df.empty:
            profit_frames.append(profit_df)
        for key in profit_stats_totals:
            profit_stats_totals[key] += int((partial.get("profit_stats_base") or {}).get(key, 0))

    return {
        "screener_frames": screener_frames,
        "screener_attrs_totals": screener_attrs_totals,
        "profit_frames": profit_frames,
        "profit_stats_totals": profit_stats_totals,
    }


def reduce_scan_shards(horizon: str, manifest: dict[str, Any], partial_dir: Path, run_date: str) -> int:
    tickers = manifest_tickers(manifest)
    merged = merge_shard_outputs(partial_dir)

    if merged["screener_frames"]:
        screener_results = pd.concat(merged["screener_frames"], ignore_index=True)
        screener_results = screener_results.sort_values("Score", ascending=False).head(TOP_RESULTS_LIMIT).reset_index(drop=True)
    else:
        screener_results = pd.DataFrame()
    screener_results.attrs.update(merged["screener_attrs_totals"])

    if merged["profit_frames"]:
        profit_pool = pd.concat(merged["profit_frames"], ignore_index=True)
    else:
        profit_pool = pd.DataFrame()

    if not profit_pool.empty:
        profit_by_upside_raw = profit_pool.sort_values("Projected Upside %", ascending=False).head(TOP_RESULTS_LIMIT).reset_index(drop=True)
        profit_by_score_raw = profit_pool.sort_values("Score", ascending=False).head(TOP_RESULTS_LIMIT).reset_index(drop=True)
        # A ticker can rank in the top 75 by both upside and score; predictions
        # must be recorded exactly once per ticker, so record from the
        # deduplicated union of both cuts rather than once per cut.
        union_raw = pd.concat([profit_by_upside_raw, profit_by_score_raw], ignore_index=True).drop_duplicates(
            subset="Ticker", keep="first"
        ).reset_index(drop=True)
    else:
        profit_by_upside_raw = profit_by_score_raw = union_raw = profit_pool

    _, profit_stats = finalize_profit_results(union_raw, horizon, record=True, stats_base=merged["profit_stats_totals"])
    profit_by_upside, _ = finalize_profit_results(profit_by_upside_raw, horizon, record=False)
    profit_by_score, _ = finalize_profit_results(profit_by_score_raw, horizon, record=False)

    html_content = build_scan_report(
        manifest, tickers, screener_results, profit_by_upside, profit_by_score, profit_stats, horizon, run_date
    )
    attachments = [
        csv_attachment(f"screener_top75_{run_date}.csv", screener_results.to_csv(index=False)),
        csv_attachment(f"profit_opportunities_by_upside_top75_{run_date}.csv", profit_by_upside.to_csv(index=False)),
        csv_attachment(f"profit_opportunities_by_score_top75_{run_date}.csv", profit_by_score.to_csv(index=False)),
    ]
    subject = f"BK Self {HORIZON_SETTINGS[horizon]['label']} scan report — {run_date}"
    sent_count = send_brevo_email(subject=subject, html_content=html_content, attachments=attachments)
    print(f"Sent scan report to {sent_count} recipient(s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and send scheduled scan reports for the live LightGBM ticker "
            "universe, split across discover/scan-shard/reduce stages so large "
            "universes can be scanned in parallel without loading the full batch "
            "artifact in a single job."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover_parser = subparsers.add_parser(
        "discover", help="Fetch the live manifest and shared macro table once for all scan shards."
    )
    discover_parser.add_argument("--repository", default=None)
    discover_parser.add_argument("--output-manifest", required=True)
    discover_parser.add_argument("--output-macro", required=True)

    scan_shard_parser = subparsers.add_parser(
        "scan-shard", help="Run the screener/profit-opportunities scans for one ticker shard."
    )
    scan_shard_parser.add_argument("--horizon", choices=sorted(HORIZON_SETTINGS.keys()), required=True)
    scan_shard_parser.add_argument("--manifest-file", required=True)
    scan_shard_parser.add_argument("--macro-file", required=True)
    scan_shard_parser.add_argument("--shard-index", type=int, required=True)
    scan_shard_parser.add_argument("--shard-count", type=int, required=True)
    scan_shard_parser.add_argument("--output", required=True)

    reduce_parser = subparsers.add_parser(
        "reduce",
        help="Merge scan-shard partial outputs, truncate to the final top-N, "
        "record predictions once, and send the email report.",
    )
    reduce_parser.add_argument("--horizon", choices=sorted(HORIZON_SETTINGS.keys()), required=True)
    reduce_parser.add_argument("--manifest-file", required=True)
    reduce_parser.add_argument("--partial-dir", required=True)
    reduce_parser.add_argument("--run-date", default=None)

    return parser


def _run_discover(args: argparse.Namespace) -> int:
    run_date = datetime.now(UTC).date().isoformat()
    info = discover_scan_inputs(args.repository, Path(args.output_manifest), Path(args.output_macro))
    # This subcommand's stdout is redirected straight into `$GITHUB_OUTPUT`
    # by the workflow (`>> "$GITHUB_OUTPUT"`), which requires every stdout
    # line to be a strict `key=value` pair, so human-readable logging goes
    # to stderr instead.
    print(f"Discovered {len(info['tickers'])} tickers across {info['shard_count']} scan shard(s)", file=sys.stderr)
    print(f"run_date={run_date}")
    print(f"shard_count={info['shard_count']}")
    print(f"shard_matrix={json.dumps(list(range(info['shard_count'])))}")
    return 0


def _run_scan_shard(args: argparse.Namespace) -> int:
    manifest = json.loads(Path(args.manifest_file).read_text(encoding="utf-8"))
    macro_table = joblib.load(Path(args.macro_file))
    partial = run_scan_shard(args.horizon, manifest, macro_table, int(args.shard_index), int(args.shard_count))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(partial, output_path)
    return 0


def _run_reduce(args: argparse.Namespace) -> int:
    manifest = json.loads(Path(args.manifest_file).read_text(encoding="utf-8"))
    run_date = args.run_date or datetime.now(UTC).date().isoformat()
    return reduce_scan_shards(args.horizon, manifest, Path(args.partial_dir), run_date)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "discover":
        return _run_discover(args)
    if args.command == "scan-shard":
        return _run_scan_shard(args)
    if args.command == "reduce":
        return _run_reduce(args)
    raise AssertionError(f"Unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
