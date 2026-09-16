from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.email_reports import csv_attachment, render_html_table, render_metric_tiles, render_report_html, send_brevo_email
from modules.fred_client import get_macro_feature_table
from modules.lightgbm_batch import (
    batch_asset_urls_from_manifest,
    fetch_live_manifest,
    load_return_model_batch_from_urls,
    resolve_release_repository,
)
from modules.prediction_tracker import SUBSCORE_ROW_FIELDS, record_predictions_from_scan
from modules.profit_opportunities import (
    DEFAULT_FAST_SCREEN_MARGIN,
    HORIZON_SETTINGS as _ALL_HORIZON_SETTINGS,
    filter_by_upside,
    scan_profit_opportunities,
)
from modules.screener import run_screener

# Scheduled scans only cover short/medium term horizons; keep the same
# restricted subset here (rather than the full three-horizon superset in
# modules.profit_opportunities, which also serves the long-term Streamlit page).
HORIZON_SETTINGS = {k: v for k, v in _ALL_HORIZON_SETTINGS.items() if k in ("short_term", "medium_term")}
DEFAULT_MIN_SCORE = 50
DEFAULT_MIN_UPSIDE_PCT = 15.0
DEFAULT_FAST_SCREEN_PROXY_THRESHOLD = 15
TOP_RESULTS_LIMIT = 75
PREVIEW_LIMIT = 10


@contextmanager
def live_scoring_context(manifest: dict[str, Any], batch: dict[str, Any], shared_macro_table: pd.DataFrame):
    import modules.scoring_engine as scoring_engine

    original_build_feature_table = scoring_engine.build_feature_table

    def _build_feature_table_with_shared_macro(*args, **kwargs):
        kwargs.setdefault("shared_macro_table", shared_macro_table)
        return original_build_feature_table(*args, **kwargs)

    with (
        patch.object(scoring_engine, "_load_live_lightgbm_manifest", return_value=manifest),
        patch.object(scoring_engine, "_load_live_lightgbm_batch", return_value=batch),
        patch.object(scoring_engine, "build_feature_table", side_effect=_build_feature_table_with_shared_macro),
    ):
        yield


def load_live_scan_inputs(repository: str | None = None) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    manifest = fetch_live_manifest(repository=resolve_release_repository(repository))
    tickers = sorted(
        str(ticker).strip().upper()
        for ticker in ((manifest or {}).get("tickers") or {})
        if str(ticker).strip()
    )
    if not tickers:
        raise RuntimeError("Live manifest is empty or unreachable; refusing to send an empty scan report")

    asset_urls = batch_asset_urls_from_manifest(manifest)
    if not asset_urls:
        raise RuntimeError("Live manifest is missing latest batch asset_url")

    batch = load_return_model_batch_from_urls(asset_urls)
    if not ((batch or {}).get("models") or {}):
        raise RuntimeError("Live batch artifact is empty or unreachable")
    return manifest, batch, tickers


def fetch_shared_macro_table(lookback_days: int = 365 * 5) -> pd.DataFrame:
    end_date = datetime.now(UTC).date()
    start_date = end_date - timedelta(days=int(lookback_days))
    macro_table = get_macro_feature_table(start_date, end_date, realtime_end=end_date)
    print(f"Loaded shared FRED macro table rows: {len(macro_table)}")
    return macro_table


def run_profit_opportunities_scan(
    tickers: list[str],
    horizon: str,
    *,
    min_upside_pct: float = DEFAULT_MIN_UPSIDE_PCT,
) -> tuple[pd.DataFrame, dict[str, int]]:
    scanned = scan_profit_opportunities(
        tickers,
        horizon,
        use_fast_screen=True,
        fast_screen_base=DEFAULT_FAST_SCREEN_PROXY_THRESHOLD + DEFAULT_FAST_SCREEN_MARGIN,
        fast_screen_margin=DEFAULT_FAST_SCREEN_MARGIN,
        parallel=False,
        prefetch_price_period="1y",
    )
    results = filter_by_upside(scanned, horizon, min_upside_pct=min_upside_pct, max_results=TOP_RESULTS_LIMIT)
    # Capture rows (with internal _rsi/_*_score fields intact) for recording
    # before stripping them from the DataFrame used for the CSV/HTML report.
    recorded_rows = results.to_dict("records")
    results = results.drop(columns=["_rsi", *SUBSCORE_ROW_FIELDS], errors="ignore")

    if recorded_rows:
        recorded_count = record_predictions_from_scan(recorded_rows, horizon=horizon, source="profit_opportunities")
    else:
        recorded_count = 0

    return results, {
        "scanned_count": int(scanned.attrs.get("scanned_count", len(tickers))),
        "fast_filtered_count": int(scanned.attrs.get("fast_filtered_count", 0)),
        "passed_fast_screen_count": int(scanned.attrs.get("fully_analyzed_count", 0)),
        "failed_count": int(scanned.attrs.get("failed_count", 0)),
        "qualified_count": len(results),
        "recorded_count": recorded_count,
    }


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


def build_scan_report(
    manifest: dict[str, Any],
    tickers: list[str],
    screener_results: pd.DataFrame,
    profit_results: pd.DataFrame,
    profit_stats: dict[str, int],
    horizon: str,
    run_date: str,
) -> str:
    current_run = (manifest or {}).get("current_run") or {}
    screener_top = screener_results.iloc[0] if not screener_results.empty else None
    profit_top = profit_results.iloc[0] if not profit_results.empty else None
    screener_avg_score = float(screener_results["Score"].mean()) if not screener_results.empty else None
    screener_median_score = float(screener_results["Score"].median()) if not screener_results.empty else None
    profit_avg_upside = float(profit_results["Projected Upside %"].mean()) if not profit_results.empty else None
    profit_median_upside = float(profit_results["Projected Upside %"].median()) if not profit_results.empty else None

    sections = [
        (
            f"<h2 style=\"margin:0 0 10px 0;color:#f4fff8;\">Screener top 75</h2>"
            f"<p style=\"margin:0 0 16px 0;color:#a9bdd7;\">Score-based scan for {HORIZON_SETTINGS[horizon]['label']} workflows. Top 10 preview below; full top 75 is attached as CSV.</p>"
            f"{render_metric_tiles([{'label': 'Passed fast-screen', 'value': str(int(screener_results.attrs.get('fully_analyzed_count') or 0))}, {'label': 'Fast-filtered', 'value': str(int(screener_results.attrs.get('fast_filtered_count') or 0))}, {'label': 'Top ticker', 'value': str(screener_top['Ticker']) if screener_top is not None else '—'}, {'label': 'Avg / median score', 'value': ('—' if screener_avg_score is None else f'{screener_avg_score:.1f} / {screener_median_score:.1f}')}, {'label': 'High-confidence picks', 'value': str(int((screener_results['Score'] >= 80).sum())) if not screener_results.empty else '0'}])}"
            f"<div style=\"margin-top:18px;\">{render_html_table(['Ticker', 'Company', 'Score', 'Recommendation', 'Current Price', 'Target Price'], _preview_rows(screener_results, ['Ticker', 'Company', 'Score', 'Recommendation', 'Current Price', 'Target Price']))}</div>"
        ),
        (
            f"<h2 style=\"margin:0 0 10px 0;color:#f4fff8;\">Profit opportunities top 75</h2>"
            f"<p style=\"margin:0 0 16px 0;color:#a9bdd7;\">Projected-upside scan for {HORIZON_SETTINGS[horizon]['label']}. Top 10 preview below; full top 75 is attached as CSV.</p>"
            f"{render_metric_tiles([{'label': 'Passed fast-screen', 'value': str(int(profit_stats.get('passed_fast_screen_count') or 0))}, {'label': 'Recorded predictions', 'value': str(int(profit_stats.get('recorded_count') or 0))}, {'label': 'Top ticker', 'value': str(profit_top['Ticker']) if profit_top is not None else '—'}, {'label': 'Avg / median upside', 'value': ('—' if profit_avg_upside is None else f'{profit_avg_upside:.2f}% / {profit_median_upside:.2f}%')}, {'label': 'High-confidence picks', 'value': str(int((profit_results['Confidence'] == 'High').sum())) if not profit_results.empty else '0'}])}"
            f"<div style=\"margin-top:18px;\">{render_html_table(['Ticker', 'Company', 'Score', 'Current Price', 'Target Price', 'Projected Upside %', 'Confidence'], _preview_rows(profit_results, ['Ticker', 'Company', 'Score', 'Current Price', 'Target Price', 'Projected Upside %', 'Confidence']))}</div>"
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


def generate_scan_report(horizon: str, repository: str | None = None) -> int:
    run_date = datetime.now(UTC).date().isoformat()
    manifest, batch, tickers = load_live_scan_inputs(repository=repository)
    shared_macro_table = fetch_shared_macro_table()

    with live_scoring_context(manifest, batch, shared_macro_table):
        screener_results = run_screener(
            universe="custom",
            custom_tickers=tickers,
            min_score=DEFAULT_MIN_SCORE,
            max_results=TOP_RESULTS_LIMIT,
            max_workers=1,
            use_fast_screen=True,
        )
        profit_results, profit_stats = run_profit_opportunities_scan(tickers, horizon)

    html_content = build_scan_report(manifest, tickers, screener_results, profit_results, profit_stats, horizon, run_date)
    attachments = [
        csv_attachment(f"screener_top75_{run_date}.csv", screener_results.to_csv(index=False)),
        csv_attachment(f"profit_opportunities_top75_{run_date}.csv", profit_results.to_csv(index=False)),
    ]
    subject = f"BK Self {HORIZON_SETTINGS[horizon]['label']} scan report — {run_date}"
    sent_count = send_brevo_email(subject=subject, html_content=html_content, attachments=attachments)
    print(f"Sent scan report to {sent_count} recipient(s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send scheduled scan reports for the live LightGBM ticker universe.")
    parser.add_argument("--horizon", choices=sorted(HORIZON_SETTINGS.keys()), required=True)
    parser.add_argument("--repository", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return generate_scan_report(args.horizon, repository=args.repository)


if __name__ == "__main__":
    raise SystemExit(main())
