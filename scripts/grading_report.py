from __future__ import annotations

import argparse
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.email_reports import render_html_table, render_metric_tiles, render_report_html, send_brevo_email
from modules.prediction_tracker import HORIZON_DAYS, compute_max_price_since_scan, get_all_predictions, resolve_pending_predictions

HORIZON_LABELS = {
    "short_term": "Short-Term (1–4 weeks)",
    "medium_term": "Medium-Term (1–6 months)",
}


def _target_scan_date(horizon: str, today: date | None = None) -> date:
    current_day = today or datetime.now(UTC).date()
    return current_day - timedelta(days=int(HORIZON_DAYS[horizon]))


def _format_currency(value: Any) -> str:
    if value is None:
        return "—"
    return f"${float(value):,.2f}"


def _format_pct(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):,.2f}%"


def load_prediction_batch(horizon: str, scan_date: date) -> list[dict[str, Any]]:
    records = get_all_predictions()
    return [
        record
        for record in records
        if str(record.get("source") or "") == "profit_opportunities"
        and str(record.get("horizon") or "") == horizon
        and str(record.get("scan_date") or "")[:10] == scan_date.isoformat()
    ]


def build_grading_rows(records: list[dict[str, Any]], grading_date: date) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        current_price = float(record.get("current_price_at_scan") or 0)
        target_price = float(record.get("target_price") or 0)
        target_low = float(record.get("target_low") or (target_price * 0.95 if target_price > 0 else 0))
        actual_return_pct = record.get("actual_return_pct")
        max_price = compute_max_price_since_scan(record.get("ticker", ""), record.get("scan_date", ""), grading_date)
        max_return_pct = round(((float(max_price) - current_price) / current_price) * 100, 4) if max_price is not None and current_price > 0 else None
        projected_upside_pct = record.get("projected_upside_pct")
        point_error_pct = (
            round(abs(float(actual_return_pct) - float(projected_upside_pct)), 4)
            if actual_return_pct is not None and projected_upside_pct is not None
            else None
        )
        rows.append(
            {
                "Ticker": record.get("ticker") or "",
                "Company": record.get("company") or record.get("ticker") or "",
                "Scan Price": current_price,
                "Target Price": target_price,
                "Projected Upside %": projected_upside_pct,
                "Target Date": record.get("target_date"),
                "Actual @ Target": record.get("actual_price_at_target_date"),
                "Actual Return %": actual_return_pct,
                "Point Hit": bool(record.get("hit_target")),
                "Band Hit": bool(record.get("hit_band")),
                "Max Price Since Scan": max_price,
                "Max Return %": max_return_pct,
                "Max Hit": bool(max_price is not None and target_price > 0 and float(max_price) >= target_price),
                "Max Band Hit": bool(max_price is not None and target_low > 0 and float(max_price) >= target_low),
                "Abs Error %": point_error_pct,
                "Status": record.get("status") or "unknown",
            }
        )
    return rows


def compute_batch_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    point_rows = [row for row in rows if row.get("Actual Return %") is not None]
    max_rows = [row for row in rows if row.get("Max Return %") is not None]
    errors = sorted(float(row["Abs Error %"]) for row in point_rows if row.get("Abs Error %") is not None)
    actual_returns = [float(row["Actual Return %"]) for row in point_rows]
    median_error = None
    if errors:
        mid = len(errors) // 2
        median_error = errors[mid] if len(errors) % 2 else (errors[mid - 1] + errors[mid]) / 2
    best = max(point_rows, key=lambda row: float(row.get("Actual Return %") or -10**9), default=None)
    worst = min(point_rows, key=lambda row: float(row.get("Actual Return %") or 10**9), default=None)
    return {
        "graded_count": len(rows),
        "point_rows": len(point_rows),
        "point_hit_rate": round(sum(1 for row in point_rows if row.get("Point Hit")) / len(point_rows) * 100, 1) if point_rows else None,
        "band_hit_rate": round(sum(1 for row in point_rows if row.get("Band Hit")) / len(point_rows) * 100, 1) if point_rows else None,
        "max_hit_rate": round(sum(1 for row in max_rows if row.get("Max Hit")) / len(max_rows) * 100, 1) if max_rows else None,
        "max_band_hit_rate": round(sum(1 for row in max_rows if row.get("Max Band Hit")) / len(max_rows) * 100, 1) if max_rows else None,
        "mae_pct": round(sum(errors) / len(errors), 2) if errors else None,
        "median_absolute_error_pct": round(median_error, 2) if median_error is not None else None,
        "best": best,
        "worst": worst,
        "average_actual_return_pct": round(sum(actual_returns) / len(actual_returns), 2) if actual_returns else None,
    }


def build_grading_report(horizon: str, scan_date: date, grading_date: date, rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    preview_rows = []
    for row in rows:
        preview_rows.append(
            {
                **row,
                "Scan Price": _format_currency(row.get("Scan Price")),
                "Target Price": _format_currency(row.get("Target Price")),
                "Projected Upside %": _format_pct(row.get("Projected Upside %")),
                "Actual @ Target": _format_currency(row.get("Actual @ Target")),
                "Actual Return %": _format_pct(row.get("Actual Return %")),
                "Max Price Since Scan": _format_currency(row.get("Max Price Since Scan")),
                "Max Return %": _format_pct(row.get("Max Return %")),
            }
        )

    best = summary.get("best")
    worst = summary.get("worst")
    highlight_tiles = render_metric_tiles(
        [
            {"label": "Point hit rate", "value": "—" if summary.get("point_hit_rate") is None else f"{summary['point_hit_rate']:.1f}%"},
            {"label": "Band hit rate", "value": "—" if summary.get("band_hit_rate") is None else f"{summary['band_hit_rate']:.1f}%"},
            {"label": "Max hit rate", "value": "—" if summary.get("max_hit_rate") is None else f"{summary['max_hit_rate']:.1f}%"},
            {"label": "Max band hit rate", "value": "—" if summary.get("max_band_hit_rate") is None else f"{summary['max_band_hit_rate']:.1f}%"},
            {
                "label": "MAE / median APE",
                "value": "—"
                if summary.get("mae_pct") is None
                else f"{summary['mae_pct']:.2f}% / {summary['median_absolute_error_pct']:.2f}%",
            },
        ]
    )
    callouts = ""
    if best and worst:
        callouts = (
            f"<p style=\"margin:18px 0 0 0;color:#dce8f7;\"><strong>Best performer:</strong> {best['Ticker']} "
            f"({_format_pct(best.get('Actual Return %'))}) &nbsp;|&nbsp; <strong>Worst performer:</strong> "
            f"{worst['Ticker']} ({_format_pct(worst.get('Actual Return %'))})</p>"
        )
    sections = [
        (
            f"<h2 style=\"margin:0 0 10px 0;color:#f4fff8;\">Batch highlights</h2>"
            f"<p style=\"margin:0 0 16px 0;color:#a9bdd7;\">Grading the profit-opportunities scan recorded on {scan_date.isoformat()} against data available through {grading_date.isoformat()}.</p>"
            f"{highlight_tiles}"
            f"{callouts}"
        ),
        (
            "<h2 style=\"margin:0 0 10px 0;color:#f4fff8;\">Graded predictions</h2>"
            "<p style=\"margin:0 0 16px 0;color:#a9bdd7;\">Point-in-time resolution and max-favorable-excursion are shown side by side for the exact recorded scan batch.</p>"
            + render_html_table(
                [
                    "Ticker",
                    "Scan Price",
                    "Target Price",
                    "Projected Upside %",
                    "Actual @ Target",
                    "Actual Return %",
                    "Point Hit",
                    "Band Hit",
                    "Max Price Since Scan",
                    "Max Return %",
                    "Max Hit",
                    "Max Band Hit",
                    "Status",
                ],
                preview_rows,
            )
        ),
    ]
    return render_report_html(
        title=f"{HORIZON_LABELS[horizon]} grading report",
        subtitle=f"Scan date {scan_date.isoformat()} · grading date {grading_date.isoformat()} · tracked batch size {len(rows)}",
        metrics=[
            {"label": "Predictions graded", "value": str(int(summary.get("graded_count") or 0))},
            {"label": "Resolved at target", "value": str(int(summary.get("point_rows") or 0))},
            {"label": "Average actual return", "value": "—" if summary.get("average_actual_return_pct") is None else f"{summary['average_actual_return_pct']:.2f}%"},
            {"label": "Target batch source", "value": "profit_opportunities"},
        ],
        sections=sections,
    )


def generate_grading_report(horizon: str) -> int:
    grading_date = datetime.now(UTC).date()
    scan_date = _target_scan_date(horizon, grading_date)
    resolution_summary = resolve_pending_predictions()
    print(f"Resolved pending predictions: {resolution_summary}")
    batch_records = load_prediction_batch(horizon, scan_date)
    if not batch_records:
        raise RuntimeError(
            f"No recorded profit_opportunities predictions found for horizon={horizon} scan_date={scan_date.isoformat()}"
        )
    rows = build_grading_rows(batch_records[:75], grading_date)
    summary = compute_batch_summary(rows)
    html_content = build_grading_report(horizon, scan_date, grading_date, rows, summary)
    subject = f"BK Self {HORIZON_LABELS[horizon]} grading report — {scan_date.isoformat()}"
    sent_count = send_brevo_email(subject=subject, html_content=html_content)
    print(f"Sent grading report to {sent_count} recipient(s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send scheduled grading reports for recorded profit-opportunity scans.")
    parser.add_argument("--horizon", choices=sorted(HORIZON_LABELS.keys()), required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return generate_grading_report(args.horizon)


if __name__ == "__main__":
    raise SystemExit(main())
