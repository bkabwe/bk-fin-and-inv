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

from modules.data_fetcher import get_stock_data
from modules.email_reports import csv_attachment, render_html_table, render_metric_tiles, render_report_html, send_brevo_email
from modules.fred_client import get_macro_feature_table
from modules.lightgbm_batch import (
    batch_asset_urls_from_manifest,
    fetch_live_manifest,
    load_return_model_batch_from_urls,
    resolve_release_repository,
)
from modules.prediction_tracker import record_predictions_from_scan
from modules.scoring_engine import analyze_stock, fast_screen_score
from modules.screener import run_screener

HORIZON_SETTINGS = {
    "short_term": {
        "label": "Short-Term (1–4 weeks)",
        "target_key": "short_term_target",
        "target_low_key": "short_term_low",
        "target_high_key": "short_term_high",
        "upside_key": "short_term_upside",
        "basis_key": "short_term_basis",
    },
    "medium_term": {
        "label": "Medium-Term (1–6 months)",
        "target_key": "medium_term_target",
        "target_low_key": "medium_term_low",
        "target_high_key": "medium_term_high",
        "upside_key": "medium_term_upside",
        "basis_key": "medium_term_basis",
    },
}
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

    with patch.object(scoring_engine, "_load_live_lightgbm_manifest", return_value=manifest):
        with patch.object(scoring_engine, "_load_live_lightgbm_batch", return_value=batch):
            with patch.object(scoring_engine, "build_feature_table", side_effect=_build_feature_table_with_shared_macro):
                yield


def load_live_scan_inputs(repository: str | None = None) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    manifest = fetch_live_manifest(repository=resolve_release_repository(repository))
    tickers = sorted(
        str(ticker).strip().upper()
        for ticker in ((manifest or {}).get("tickers") or {}).keys()
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
    macro_table = get_macro_feature_table(start_date, end_date)
    print(f"Loaded shared FRED macro table rows: {len(macro_table)}")
    return macro_table


def run_profit_opportunities_scan(
    tickers: list[str],
    horizon: str,
    *,
    min_upside_pct: float = DEFAULT_MIN_UPSIDE_PCT,
) -> tuple[pd.DataFrame, dict[str, int]]:
    settings = HORIZON_SETTINGS[horizon]
    rows: list[dict[str, Any]] = []
    fast_filtered_count = 0
    passed_fast_screen_count = 0
    failed_count = 0

    for ticker in tickers:
        try:
            price_data = get_stock_data(ticker, period="1y", interval="1d")
            fast_score, err = fast_screen_score(ticker, data=price_data)
            if err is not None:
                failed_count += 1
                print(f"Fast-screen failed for {ticker}: {err}")
                continue
            if fast_score < DEFAULT_FAST_SCREEN_PROXY_THRESHOLD:
                fast_filtered_count += 1
                continue

            passed_fast_screen_count += 1
            analysis = analyze_stock(
                ticker,
                data_override=price_data,
                projection_data_override=price_data,
            )
            projections = analysis.get("projections") or {}
            current_price = float(projections.get("current_price") or analysis.get("current_price") or 0)
            target_price = float(projections.get(settings["target_key"]) or 0)
            upside = float(projections.get(settings["upside_key"]) or 0)
            if current_price <= 0 or target_price <= 0 or upside <= 0 or upside < float(min_upside_pct):
                continue
            rows.append(
                {
                    "Ticker": ticker,
                    "Company": analysis.get("company") or ticker,
                    "Score": int(analysis.get("score") or 0),
                    "Current Price": round(current_price, 4),
                    "Target Price": round(target_price, 4),
                    "Target Low": round(float(projections.get(settings["target_low_key"]) or target_price * 0.95), 4),
                    "Target High": round(float(projections.get(settings["target_high_key"]) or target_price * 1.05), 4),
                    "Projected Upside %": round(upside, 4),
                    "Confidence": str(projections.get("data_quality") or "Limited"),
                    "Basis": str(projections.get(settings["basis_key"]) or ""),
                }
            )
        except Exception as exc:
            failed_count += 1
            print(f"Profit-opportunity analysis failed for {ticker}: {exc}")

    if rows:
        results = pd.DataFrame(rows).sort_values("Projected Upside %", ascending=False).head(TOP_RESULTS_LIMIT).reset_index(drop=True)
        recorded_count = record_predictions_from_scan(results.to_dict("records"), horizon=horizon, source="profit_opportunities")
    else:
        results = pd.DataFrame(columns=["Ticker", "Company", "Score", "Current Price", "Target Price", "Target Low", "Target High", "Projected Upside %", "Confidence", "Basis"])
        recorded_count = 0

    return results, {
        "scanned_count": len(tickers),
        "fast_filtered_count": fast_filtered_count,
        "passed_fast_screen_count": passed_fast_screen_count,
        "failed_count": failed_count,
        "qualified_count": len(rows),
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
