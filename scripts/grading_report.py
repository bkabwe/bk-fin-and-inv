from __future__ import annotations

import argparse
import csv
import io
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.email_reports import (
    StyledSection,
    csv_attachment,
    render_html_table,
    render_metric_tiles,
    render_panel,
    render_paragraph,
    render_report_html,
    render_row_title,
    send_brevo_email,
)
from modules.prediction_tracker import (
    HORIZON_DAYS,
    compute_max_price_since_scan,
    compute_score_validation_stats,
    compute_upside_validation_stats,
    get_all_predictions,
    resolve_pending_predictions,
)
from modules.price_format import format_price

HORIZON_LABELS = {
    "short_term": "Short-Term (1–4 weeks)",
    "medium_term": "Medium-Term (1–6 months)",
}

# First scheduled scan on the current setup: the SQLite prediction store (#62)
# and the scoring-model changes (#63). Earlier scans came from the legacy JSON
# store, the previous model (the 2026-10-01 medium-term batch still had the
# circular "Fundamental" ensemble slot) or one-off manual runs around the
# changeover, so they are neither graded nor counted in the validation panels.
# Move this forward whenever a scoring change makes older predictions
# incomparable.
CURRENT_MODEL_START_DATE = date(2026, 10, 5)

# Days between runs of each horizon's grading workflow (the crons in
# .github/workflows/grading-report-*.yml: every Monday, and the 1st of every
# other month, which is 59-62 days apart). A run grades the newest scan whose
# horizon completed within the last cadence, so each scheduled scan is graded
# once even though its date is rarely exactly HORIZON_DAYS before a grading
# date -- for the medium term it never is (90 days isn't a multiple of ~61).
GRADING_CADENCE_DAYS = {"short_term": 7, "medium_term": 62}

# Rows graded per report: the first 75 recorded rows (the scan's top-75 list).
GRADED_BATCH_LIMIT = 75
# Rows shown in the email's table. The CSV attachment carries every graded row,
# which keeps the HTML under Gmail's ~102KB clipping limit.
PREVIEW_LIMIT = 10


def _target_scan_date(horizon: str, today: date | None = None) -> date:
    current_day = today or datetime.now(UTC).date()
    return current_day - timedelta(days=int(HORIZON_DAYS[horizon]))


def _scan_window(horizon: str, today: date) -> tuple[date, date] | None:
    """Inclusive range of scan dates a run on ``today`` may grade, or None if
    the range is empty.

    Its newest end is the horizon cut-off (a newer scan hasn't reached its
    horizon yet); its oldest end is one grading cadence earlier, so a batch is
    only picked up by the run that first covers it, and never predates the
    current model.
    """
    last = _target_scan_date(horizon, today)
    first = max(last - timedelta(days=GRADING_CADENCE_DAYS[horizon] - 1), CURRENT_MODEL_START_DATE)
    return (first, last) if first <= last else None


def _format_currency(value: Any) -> str:
    # Sub-dollar scan/target prices keep 4-6 decimals (price_format.price_decimals).
    return format_price(value)


def _format_pct(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):,.2f}%"


def find_grading_batch(horizon: str, today: date) -> tuple[date, list[dict[str, Any]]] | None:
    """The newest recorded profit-opportunities batch a run on ``today`` should
    grade, as ``(scan_date, records)``, or None when there isn't one."""
    window = _scan_window(horizon, today)
    if window is None:
        return None
    first, last = window[0].isoformat(), window[1].isoformat()
    batches: dict[str, list[dict[str, Any]]] = {}
    for record in get_all_predictions():
        scan_day = str(record.get("scan_date") or "")[:10]
        if (
            str(record.get("source") or "") == "profit_opportunities"
            and str(record.get("horizon") or "") == horizon
            and first <= scan_day <= last
        ):
            batches.setdefault(scan_day, []).append(record)
    if not batches:
        return None
    newest = max(batches)
    return date.fromisoformat(newest), batches[newest]


def _nothing_to_grade_message(horizon: str, today: date) -> str:
    window = _scan_window(horizon, today)
    if window is None:
        return (
            f"Nothing to grade yet: a {horizon} run on {today.isoformat()} covers scans up to "
            f"{_target_scan_date(horizon, today).isoformat()}, before the current model's first scan "
            f"({CURRENT_MODEL_START_DATE.isoformat()})."
        )
    return (
        f"Nothing to grade: no profit_opportunities predictions were recorded for horizon={horizon} "
        f"with scan dates {window[0].isoformat()} to {window[1].isoformat()}."
    )


def _excursion_end(target_date: Any, grading_date: date) -> date:
    """Last day of a row's max-price window: its target date, or the grading date
    if that comes first (or the target date is missing). Medium-term batches are
    graded weeks after their target date; the max must not count that extra time."""
    try:
        return min(grading_date, date.fromisoformat(str(target_date)[:10]))
    except ValueError:
        return grading_date


def build_grading_rows(records: list[dict[str, Any]], grading_date: date) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        current_price = float(record.get("current_price_at_scan") or 0)
        target_price = float(record.get("target_price") or 0)
        target_low = float(record.get("target_low") or (target_price * 0.95 if target_price > 0 else 0))
        actual_return_pct = record.get("actual_return_pct")
        max_price = compute_max_price_since_scan(
            record.get("ticker", ""),
            record.get("scan_date", ""),
            _excursion_end(record.get("target_date"), grading_date),
        )
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


def build_graded_csv(rows: list[dict[str, Any]]) -> str:
    """Every graded row as CSV, with the table's raw (unformatted) values."""
    if not rows:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


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


def _horizon_stats(validation: dict[str, Any] | None, horizon: str) -> dict[str, Any]:
    """The report's own horizon from a ``compute_*_validation_stats`` result.
    (Its ``overall`` bucket blends every horizon, which would contradict the
    "this horizon" panel titles.)"""
    return ((validation or {}).get("by_horizon") or {}).get(horizon) or {}


def build_grading_report(
    horizon: str,
    scan_date: date,
    grading_date: date,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    score_validation: dict[str, Any] | None = None,
    upside_validation: dict[str, Any] | None = None,
    *,
    validation_since: date | None = None,
) -> str:
    preview_rows = []
    for row in rows[:PREVIEW_LIMIT]:
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

    if len(rows) > PREVIEW_LIMIT:
        list_note = (
            f"Showing the first {PREVIEW_LIMIT} of {len(rows)} graded predictions; "
            "the full list is attached as CSV."
        )
    elif rows:
        list_note = "The full list is also attached as CSV."
    else:
        list_note = ""
    if validation_since is None:
        validation_window, validation_scope = "all-time", "to date"
    else:
        validation_window = f"since {validation_since.isoformat()}"
        validation_scope = f"scanned since {validation_since.isoformat()} (the current scoring model)"

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
        callouts = render_paragraph(
            f"**Best performer:** {best['Ticker']} ({_format_pct(best.get('Actual Return %'))})  |  "
            f"**Worst performer:** {worst['Ticker']} ({_format_pct(worst.get('Actual Return %'))})"
        )
    sections: list[str] = [
        render_row_title("Batch highlights"),
        StyledSection(highlight_tiles),
        render_panel(
            "Graded predictions",
            callouts
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
            ),
            description=(
                f"Grading the profit-opportunities scan recorded on {scan_date.isoformat()} against data available "
                f"through {grading_date.isoformat()}. Point-in-time resolution and max-favorable-excursion (the "
                "highest price from the scan date through the target date) are shown side by side for the exact "
                f"recorded scan batch. {list_note}"
            ).strip(),
        ),
    ]

    score_stats = _horizon_stats(score_validation, horizon)
    if score_stats.get("count"):
        ic = score_stats.get("information_coefficient")
        sections.append(render_row_title(f"Score validation ({validation_window}, this horizon)"))
        sections.append(
            render_panel(
                "Does a higher score precede a better outcome?",
                render_metric_tiles(
                    [
                        {"label": "Information coefficient", "value": "—" if ic is None else f"{ic:.3f}"},
                        {
                            "label": "Top-third hit rate",
                            "value": "—" if score_stats.get("precision_at_top_third") is None else f"{score_stats['precision_at_top_third']:.1f}%",
                        },
                        {
                            "label": "Bottom-third hit rate",
                            "value": "—" if score_stats.get("precision_at_bottom_third") is None else f"{score_stats['precision_at_bottom_third']:.1f}%",
                        },
                    ]
                ),
                description=(
                    f"Computed across all {int(score_stats['count'])} resolved {HORIZON_LABELS[horizon]} "
                    f"predictions {validation_scope}, not just this batch."
                ),
            )
        )

    upside_stats = _horizon_stats(upside_validation, horizon)
    if upside_stats.get("count"):
        upside_ic = upside_stats.get("information_coefficient")
        sections.append(render_row_title(f"Upside forecast validation ({validation_window}, this horizon)"))
        sections.append(
            render_panel(
                "Is the projected upside systematically over-optimistic?",
                render_metric_tiles(
                    [
                        {"label": "Information coefficient", "value": "—" if upside_ic is None else f"{upside_ic:.3f}"},
                        {
                            "label": "Mean bias (actual − projected)",
                            "value": "—" if upside_stats.get("mean_bias_pct") is None else f"{upside_stats['mean_bias_pct']:.2f}%",
                        },
                        {
                            "label": "Median bias (actual − projected)",
                            "value": "—" if upside_stats.get("median_bias_pct") is None else f"{upside_stats['median_bias_pct']:.2f}%",
                        },
                        {
                            "label": "Over-optimism rate",
                            "value": "—" if upside_stats.get("overoptimism_rate") is None else f"{upside_stats['overoptimism_rate']:.1f}%",
                        },
                    ]
                ),
                description=(
                    "Is the projected upside % itself systematically over-optimistic, not just directionally "
                    f"correlated with outcomes? Computed across all {int(upside_stats['count'])} resolved "
                    f"{HORIZON_LABELS[horizon]} predictions {validation_scope}, not just this batch."
                ),
            )
        )

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


def generate_grading_report(horizon: str, today: date | None = None) -> int:
    grading_date = today or datetime.now(UTC).date()
    resolution_summary = resolve_pending_predictions()
    print(f"Resolved pending predictions: {resolution_summary}")
    batch = find_grading_batch(horizon, grading_date)
    if batch is None:
        # Not an error: scans are weekly / bi-monthly, so most runs around a
        # missed scan or the current-model cut-off simply have no batch yet.
        print(_nothing_to_grade_message(horizon, grading_date))
        return 0
    scan_date, batch_records = batch
    rows = build_grading_rows(batch_records[:GRADED_BATCH_LIMIT], grading_date)
    summary = compute_batch_summary(rows)
    score_validation = compute_score_validation_stats(since=CURRENT_MODEL_START_DATE)
    upside_validation = compute_upside_validation_stats(since=CURRENT_MODEL_START_DATE)
    html_content = build_grading_report(
        horizon,
        scan_date,
        grading_date,
        rows,
        summary,
        score_validation,
        upside_validation,
        validation_since=CURRENT_MODEL_START_DATE,
    )
    subject = f"BK Self {HORIZON_LABELS[horizon]} grading report — {scan_date.isoformat()}"
    graded_csv = csv_attachment(f"graded_predictions_{horizon}_{scan_date.isoformat()}.csv", build_graded_csv(rows))
    sent_count = send_brevo_email(subject=subject, html_content=html_content, attachments=[graded_csv])
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
