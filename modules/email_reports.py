from __future__ import annotations

import base64
import html
import os
from typing import Iterable

import requests

from modules.logger import get_logger

logger = get_logger(__name__)

BREVO_EMAIL_API_URL = "https://api.brevo.com/v3/smtp/email"

# Shared dashboard palette, reused by tiles/tables/badges/legend below.
_BG = "#06111d"
_CARD_BG = "#0f1f33"
_TABLE_BG = "#0b1725"
_BORDER = "#1f4f46"
_ROW_BORDER = "#1c314b"
_TEXT_PRIMARY = "#f4fff8"
_TEXT_BODY = "#ecf5ff"
_TEXT_SECONDARY = "#9eb3cf"
_ACCENT = "#39d98a"
_TILE_ACCENTS = ["#39d98a", "#5ab8ff", "#f5b64c", "#b58bff", "#ff6b6b"]

# Badge color styles keyed by a lowercased "style key" -- either the literal
# cell text (e.g. "full", "pending") or an explicit style_key passed to
# render_legend(). (background, foreground) hex colors.
_BADGE_STYLES: dict[str, tuple[str, str]] = {
    "full": ("#173a2b", "#3ddc84"),
    "resolved": ("#173a2b", "#3ddc84"),
    "yes": ("#173a2b", "#3ddc84"),
    "true": ("#173a2b", "#3ddc84"),
    "limited": ("#3a2f12", "#f5b64c"),
    "pending": ("#122a3a", "#5ab8ff"),
    "technical only": ("#2a1f3a", "#b58bff"),
    "unresolved_no_data": ("#3a1414", "#ff6b6b"),
    "no": ("#3a1414", "#ff6b6b"),
    "false": ("#3a1414", "#ff6b6b"),
    "unknown": ("#22242c", "#9eb3cf"),
}
# Columns whose text values should be rendered as badges (case-insensitive).
_BADGE_COLUMNS = {"confidence", "status"}
# Columns whose numeric values should be rendered with a mini bar-chart cell,
# and the fixed domain max to size the bar against ("None" means "use the
# max value observed in that column for this table").
_BAR_COLUMNS: dict[str, float | None] = {"score": 100.0, "projected upside %": None}


def parse_recipients(raw_value: str | None = None) -> list[str]:
    raw = str(raw_value if raw_value is not None else os.getenv("SCAN_EMAIL_RECIPIENTS") or "")
    recipients = [value.strip() for value in raw.split(",") if value and value.strip()]
    if not recipients:
        raise RuntimeError("SCAN_EMAIL_RECIPIENTS is empty or not configured")
    return recipients


def build_sender() -> dict[str, str]:
    sender_email = str(os.getenv("SCAN_EMAIL_FROM") or "").strip()
    if not sender_email:
        raise RuntimeError("SCAN_EMAIL_FROM is not configured")
    return {"name": "BK Self", "email": sender_email}


def csv_attachment(filename: str, content: str) -> dict[str, str]:
    return {
        "name": str(filename).strip(),
        "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
    }


def _numeric_value(value: object) -> float | None:
    """Best-effort parse of a raw number OR an already-formatted display
    string (e.g. "87.00", "$12.34", "12.34%", "1,234.5") back into a float.
    Callers may pre-format values before handing them to render_html_table,
    so bar-cell rendering has to tolerate both raw numbers and display text.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if number == number else None  # exclude NaN
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "").replace("$", "").replace("%", "")
        if not cleaned or cleaned in {"—", "-", "None"}:
            return None
        try:
            number = float(cleaned)
        except ValueError:
            return None
        return number if number == number else None
    return None


def _badge(text: str, bg: str, fg: str) -> str:
    return (
        '<span style="display:inline-block;padding:3px 10px;border-radius:999px;'
        f'background:{bg};color:{fg};font-size:12px;font-weight:700;'
        f'letter-spacing:0.02em;white-space:nowrap;">{html.escape(text)}</span>'
    )


def _bar_cell(display_text: str, value: float, domain_max: float) -> str:
    ratio = 0.0 if domain_max <= 0 else max(0.0, min(1.0, abs(value) / domain_max))
    width_pct = round(ratio * 100, 1)
    bar_color = _ACCENT if value >= 0 else "#ff6b6b"
    # Nested tables (rather than flexbox) render consistently across email
    # clients: an inner two-cell row where the first cell's width% is filled
    # with color and the second is empty, together spanning the full width.
    return (
        '<table role="presentation" cellpadding="0" cellspacing="0" style="border-collapse:collapse;width:100%;">'
        "<tr>"
        f'<td style="white-space:nowrap;padding-right:8px;color:{_TEXT_BODY};font-variant-numeric:tabular-nums;">{html.escape(display_text)}</td>'
        '<td style="width:100%;">'
        f'<table role="presentation" cellpadding="0" cellspacing="0" style="border-collapse:collapse;width:100%;min-width:60px;background:#132743;border-radius:6px;">'
        f'<tr><td style="width:{width_pct}%;background:{bar_color};border-radius:6px;line-height:8px;font-size:1px;">&nbsp;</td>'
        '<td style="line-height:8px;font-size:1px;">&nbsp;</td></tr>'
        "</table>"
        "</td>"
        "</tr>"
        "</table>"
    )


def _render_cell(column: str, value: object, *, domain_max: float | None) -> str:
    if isinstance(value, bool):
        style_key = "yes" if value else "no"
        bg, fg = _BADGE_STYLES[style_key]
        return _badge("Yes" if value else "No", bg, fg)

    column_key = column.strip().lower()
    if column_key in _BAR_COLUMNS:
        number = _numeric_value(value)
        if number is not None:
            text = str(value) if value not in (None, "") else f"{number:g}"
            effective_domain = _BAR_COLUMNS[column_key] if _BAR_COLUMNS[column_key] is not None else domain_max
            return _bar_cell(text, number, effective_domain or 0.0)

    if value is None or value == "":
        return "—"
    text = str(value)
    if column_key in _BADGE_COLUMNS:
        style_key = text.strip().lower()
        if style_key in _BADGE_STYLES:
            bg, fg = _BADGE_STYLES[style_key]
            return _badge(text, bg, fg)
    return html.escape(text)


def render_legend(items: Iterable[dict[str, str]]) -> str:
    """Render a compact badge+description legend, e.g. explaining what each
    Confidence tier or Status value means. Each item is a dict with `label`
    (the badge text), optional `style_key` (defaults to a lowercased `label`,
    for cases where the badge text itself isn't a recognized style key), and
    `description`.
    """
    rows = []
    for item in items:
        label = str(item.get("label") or "")
        style_key = str(item.get("style_key") or label).strip().lower()
        bg, fg = _BADGE_STYLES.get(style_key, _BADGE_STYLES["unknown"])
        description = html.escape(str(item.get("description") or ""))
        rows.append(
            '<table role="presentation" cellpadding="0" cellspacing="0" style="border-collapse:collapse;margin:6px 0;">'
            f'<tr><td style="padding-right:10px;vertical-align:top;">{_badge(label, bg, fg)}</td>'
            f'<td style="color:{_TEXT_SECONDARY};font-size:13px;vertical-align:top;">{description}</td></tr>'
            "</table>"
        )
    return "".join(rows)


def render_metric_tiles(metrics: Iterable[dict[str, str]]) -> str:
    tiles = []
    for index, metric in enumerate(metrics):
        label = html.escape(str(metric.get("label") or ""))
        value = html.escape(str(metric.get("value") or "—"))
        accent = _TILE_ACCENTS[index % len(_TILE_ACCENTS)]
        tiles.append(
            '<div style="display:inline-block;vertical-align:top;width:22%;min-width:130px;'
            'margin:0 3% 12px 0;box-sizing:border-box;">'
            f'<div style="background:{_CARD_BG};border:1px solid {_BORDER};border-top:3px solid {accent};'
            'border-radius:14px;padding:16px;">'
            f'<div style="color:{_TEXT_SECONDARY};font-size:12px;text-transform:uppercase;letter-spacing:0.08em;">{label}</div>'
            f'<div style="color:{_TEXT_PRIMARY};font-size:24px;font-weight:700;margin-top:8px;">{value}</div>'
            "</div>"
            "</div>"
        )
    return f'<div style="font-size:0;">{"".join(tiles)}</div>'


def render_html_table(columns: list[str], rows: list[dict[str, object]]) -> str:
    headers = "".join(
        f'<th style="padding:10px 12px;text-align:left;background:#0d1b2a;color:#dce8f7;'
        f'border-bottom:1px solid {_BORDER};white-space:nowrap;">{html.escape(column)}</th>'
        for column in columns
    )
    domain_maxes: dict[str, float] = {}
    for column in columns:
        column_key = column.strip().lower()
        if column_key in _BAR_COLUMNS and _BAR_COLUMNS[column_key] is None:
            observed = [n for row in rows if (n := _numeric_value(row.get(column))) is not None]
            domain_maxes[column_key] = max(observed) if observed else 0.0

    body_rows = []
    for row in rows:
        cells = "".join(
            f'<td style="padding:10px 12px;border-bottom:1px solid {_ROW_BORDER};color:{_TEXT_BODY};">'
            f"{_render_cell(column, row.get(column, '—'), domain_max=domain_maxes.get(column.strip().lower()))}</td>"
            for column in columns
        )
        body_rows.append(f"<tr>{cells}</tr>")
    if not body_rows:
        body_rows.append(
            f'<tr><td colspan="{max(len(columns), 1)}" style="padding:12px;color:{_TEXT_SECONDARY};">No rows available.</td></tr>'
        )
    return (
        '<div style="overflow-x:auto;">'
        f'<table role="presentation" cellpadding="0" cellspacing="0" style="width:100%;min-width:480px;border-collapse:collapse;'
        f'background:{_TABLE_BG};border:1px solid {_BORDER};border-radius:14px;overflow:hidden;">'
        f"<thead><tr>{headers}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"
        "</div>"
    )


def render_report_html(
    *,
    title: str,
    subtitle: str,
    metrics: Iterable[dict[str, str]],
    sections: Iterable[str],
) -> str:
    metrics_html = render_metric_tiles(metrics)
    body_sections = "".join(
        f'<div style="margin-top:24px;background:{_CARD_BG};border:1px solid {_BORDER};border-radius:18px;padding:22px;">{section}</div>'
        for section in sections
    )
    return (
        "<html><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(title)}</title></head>"
        f'<body style="margin:0;padding:24px;background:{_BG};font-family:Arial,Helvetica,sans-serif;">'
        f'<div style="max-width:640px;margin:0 auto;background:#081423;border:1px solid {_BORDER};border-radius:24px;padding:28px;">'
        f'<div style="color:{_ACCENT};font-size:13px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;">BK Self</div>'
        f'<h1 style="color:{_TEXT_PRIMARY};margin:10px 0 6px 0;font-size:28px;">{html.escape(title)}</h1>'
        f'<p style="color:{_TEXT_SECONDARY};margin:0 0 20px 0;font-size:15px;">{html.escape(subtitle)}</p>'
        f"{metrics_html}"
        f"{body_sections}"
        "</div></body></html>"
    )


def send_brevo_email(
    *,
    subject: str,
    html_content: str,
    attachments: list[dict[str, str]] | None = None,
    recipients: list[str] | None = None,
) -> int:
    api_key = str(os.getenv("BREVO_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("BREVO_API_KEY is not configured")

    sender = build_sender()
    target_recipients = recipients or parse_recipients()
    session = requests.Session()
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "api-key": api_key,
    }
    failures: list[str] = []
    sent_count = 0

    for recipient in target_recipients:
        payload = {
            "sender": sender,
            "to": [{"email": recipient}],
            "subject": subject,
            "htmlContent": html_content,
        }
        if attachments:
            payload["attachment"] = attachments
        try:
            response = session.post(BREVO_EMAIL_API_URL, json=payload, headers=headers, timeout=60)
            if response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
            sent_count += 1
            logger.info("Brevo email sent to %s", recipient)
        except Exception as exc:
            message = f"Brevo send failed for {recipient}: {exc}"
            logger.warning(message)
            failures.append(message)

    if failures:
        raise RuntimeError("; ".join(failures))
    return sent_count

