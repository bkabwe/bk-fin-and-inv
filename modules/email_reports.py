from __future__ import annotations

import base64
import html
import math
import os
import re
from typing import Callable, Iterable, Mapping

import requests

from modules.logger import get_logger

logger = get_logger(__name__)

BREVO_EMAIL_API_URL = "https://api.brevo.com/v3/smtp/email"

# Grafana-dark dashboard palette, shared by the panels/tiles/tables/badges/
# legend below. Colors follow Grafana's dark theme and its "classic" series
# palette so the emailed report reads like a Grafana dashboard.
_CANVAS_BG = "#111217"  # page canvas behind the panels
_PANEL_BG = "#181b1f"  # panel / stat-tile background
_PANEL_BORDER = "#2c3235"
_HEADER_BG = "#202226"  # table header row, variable chips
_ROW_ALT_BG = "#1b1e23"  # zebra stripe
_ROW_BORDER = "#26292e"
_TRACK_BG = "#2c3235"  # unfilled part of a gauge bar
_TEXT_PRIMARY = "#ffffff"
_TEXT_BODY = "#ccccdc"
_TEXT_SECONDARY = "#9aa0ac"
_FONT_STACK = "Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif"

_GREEN = "#73bf69"
_YELLOW = "#fade2a"
_ORANGE = "#ff9830"
_RED = "#f2495c"
_BLUE = "#5794f2"
_LIGHT_BLUE = "#8ab8ff"
_PURPLE = "#b877d9"
_TILE_ACCENTS = [_GREEN, _BLUE, _YELLOW, _PURPLE, _ORANGE]

# Widest the dashboard grows on a big screen. The layout is fluid below that
# (width:100% plus wrapping inline-block tiles and horizontally scrollable
# tables), so the same HTML fills a laptop browser and fits a phone.
_DASHBOARD_MAX_WIDTH_PX = 1200

# Badge color styles keyed by a lowercased "style key" -- either the literal
# cell text (e.g. "full", "pending") or an explicit style_key passed to
# render_legend(). (background, foreground) hex colors.
_GREEN_BADGE = ("#223d27", _GREEN)
_YELLOW_BADGE = ("#40371a", _YELLOW)
_ORANGE_BADGE = ("#42301a", _ORANGE)
_RED_BADGE = ("#442329", _RED)
_BLUE_BADGE = ("#1f3350", _LIGHT_BLUE)
_PURPLE_BADGE = ("#35283f", _PURPLE)
_GREY_BADGE = ("#2a2d33", _TEXT_SECONDARY)
_BADGE_STYLES: dict[str, tuple[str, str]] = {
    "full": _GREEN_BADGE,
    "resolved": _GREEN_BADGE,
    "yes": _GREEN_BADGE,
    "true": _GREEN_BADGE,
    "limited": _YELLOW_BADGE,
    "pending": _BLUE_BADGE,
    "technical only": _PURPLE_BADGE,
    "unresolved_no_data": _RED_BADGE,
    "no": _RED_BADGE,
    "false": _RED_BADGE,
    "unknown": _GREY_BADGE,
    # Risk-Adjusted Upside tiers (see modules.profit_opportunities).
    "strong": _GREEN_BADGE,
    "moderate": _YELLOW_BADGE,
    "weak": _ORANGE_BADGE,
    # Score-based recommendation tiers (see modules.scoring_engine._recommendation).
    "strong buy": _GREEN_BADGE,
    "buy": _BLUE_BADGE,
    "take small position": _YELLOW_BADGE,
    "monitor": _ORANGE_BADGE,
    "do not buy": _RED_BADGE,
    "avoid": _RED_BADGE,
}
# Columns whose text values should be rendered as badges (case-insensitive).
_BADGE_COLUMNS = {"confidence", "status", "recommendation", "risk-adjusted rating"}
# Columns whose numeric values should be rendered with a mini gauge-bar cell,
# and the fixed domain max to size the bar against ("None" means "use the
# max value observed in that column for this table").
_BAR_COLUMNS: dict[str, float | None] = {
    "score": 100.0,
    "projected upside %": None,
    "risk-adjusted upside": 1.0,
}


def _score_bar_color(value: float) -> str:
    # Same tier edges as modules.scoring_engine._recommendation (80/65/50/35),
    # so the bar color matches the recommendation emoji (green/blue/yellow/
    # orange/red); a test keeps the two in sync.
    if value >= 80:
        return _GREEN
    if value >= 65:
        return _BLUE
    if value >= 50:
        return _YELLOW
    if value >= 35:
        return _ORANGE
    return _RED


# Mirror modules.profit_opportunities.RISK_ADJUSTED_STRONG_MIN/_MODERATE_MIN
# (kept here so this renderer doesn't import the scoring engine; a test keeps
# the two in sync).
_RISK_ADJUSTED_STRONG_MIN = 0.60
_RISK_ADJUSTED_MODERATE_MIN = 0.30


def _risk_adjusted_bar_color(value: float) -> str:
    if value >= _RISK_ADJUSTED_STRONG_MIN:
        return _GREEN
    if value >= _RISK_ADJUSTED_MODERATE_MIN:
        return _YELLOW
    return _ORANGE


# Gauge-bar color rules, keyed like _BAR_COLUMNS. Columns without a rule use
# green for positive values and red for negative ones.
_BAR_COLOR_RULES: dict[str, Callable[[float], str]] = {
    "score": _score_bar_color,
    "projected upside %": lambda value: _BLUE if value >= 0 else _RED,
    "risk-adjusted upside": _risk_adjusted_bar_color,
}
# One-line "how to read this column" hints rendered under a column header.
_COLUMN_HINTS: dict[str, str] = {
    "score": "0–100 · higher = better",
    "projected upside %": "to target price",
    "forecast range %": "target low → high",
    "risk-adjusted upside": "0–1 · higher = better",
}
# Columns dropped (via the .bk-hide-sm class) on phone-sized screens in email
# clients that honor <style> media queries; the table still scrolls
# horizontally everywhere else.
_HIDE_ON_SMALL_COLUMNS = {"company"}


class StyledSection(str):
    """HTML that is already a fully styled dashboard block (a panel, a row
    heading, ...). :func:`render_report_html` places these as-is, while plain
    ``str`` sections are wrapped in a default untitled panel."""

    __slots__ = ()


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
        '<span style="display:inline-block;padding:2px 9px;border-radius:999px;'
        f'background:{bg};color:{fg};font-size:11px;font-weight:700;'
        f'letter-spacing:0.03em;white-space:nowrap;">{html.escape(text)}</span>'
    )


def _bar_cell(display_text: str, value: float, domain_max: float, *, color: str | None = None) -> str:
    ratio = 0.0 if domain_max <= 0 else max(0.0, min(1.0, abs(value) / domain_max))
    width_pct = round(ratio * 100, 1)
    bar_color = color or (_GREEN if value >= 0 else _RED)
    # A value over a thin gauge bar (like a Grafana table "gauge" cell). Plain
    # stacked divs keep this a few hundred bytes -- the whole report has to
    # stay well under Gmail's ~102KB clipping limit -- and wrap cleanly.
    return (
        f'<div style="white-space:nowrap;font-variant-numeric:tabular-nums;">{html.escape(display_text)}</div>'
        f'<div style="min-width:64px;margin-top:4px;background:{_TRACK_BG};border-radius:2px;line-height:4px;font-size:1px;">'
        f'<div style="width:{width_pct}%;background:{bar_color};border-radius:2px;line-height:4px;font-size:1px;">&nbsp;</div>'
        "</div>"
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
            color_rule = _BAR_COLOR_RULES.get(column_key)
            return _bar_cell(text, number, effective_domain or 0.0, color=color_rule(number) if color_rule else None)
    if value is None or value == "":
        return "—"
    text = str(value)
    if column_key in _BADGE_COLUMNS:
        # Tolerate an emoji prefix ("🟢 STRONG BUY") and a trailing threshold
        # note ("Strong (0.60+)") when looking up the badge style, but show the
        # text the caller passed.
        style_key = " ".join(text.split("(")[0].encode("ascii", "ignore").decode().split()).lower()
        if style_key in _BADGE_STYLES:
            bg, fg = _BADGE_STYLES[style_key]
            return _badge(text, bg, fg)
    if column_key == "ticker":
        return f'<strong style="color:{_LIGHT_BLUE};">{html.escape(text)}</strong>'
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
            f'<tr><td style="padding:4px 12px 4px 0;vertical-align:top;white-space:nowrap;">{_badge(label, bg, fg)}</td>'
            f'<td style="padding:4px 0;color:{_TEXT_SECONDARY};font-size:13px;line-height:1.45;vertical-align:top;">{description}</td></tr>'
        )
    # One table (not one per item) so the descriptions line up in a column.
    return (
        '<table role="presentation" cellpadding="0" cellspacing="0" border="0" '
        f'style="border-collapse:collapse;margin:6px 0;">{"".join(rows)}</table>'
    )


_BOLD_MARKUP = re.compile(r"\*\*(.+?)\*\*")


def _inline_markup(text: str) -> str:
    """Escape `text`, then turn ``**bold**`` spans into <strong> (escaping first
    means the only tags that can appear are the ones added here)."""
    return _BOLD_MARKUP.sub(
        lambda match: f'<strong style="color:{_TEXT_BODY};font-weight:600;">{match.group(1)}</strong>',
        html.escape(text),
    )


def render_subheading(text: str) -> str:
    """A small uppercase label that titles a block of explanatory copy."""
    return (
        f'<div style="margin:12px 0 4px 0;color:{_TEXT_BODY};font-size:12px;font-weight:600;'
        f'letter-spacing:0.06em;text-transform:uppercase;">{html.escape(text)}</div>'
    )


def render_paragraph(text: str) -> str:
    """Muted explanatory copy; ``**bold**`` spans are emphasized."""
    return (
        f'<div style="margin:0 0 8px 0;color:{_TEXT_SECONDARY};font-size:13px;line-height:1.55;">'
        f"{_inline_markup(text)}</div>"
    )


def render_callout(text: str) -> str:
    """A highlighted one-line formula or definition."""
    return (
        f'<div style="margin:6px 0 10px 0;padding:8px 12px;background:{_HEADER_BG};border-left:3px solid {_BLUE};'
        f'color:{_TEXT_BODY};font-size:13px;line-height:1.5;">{_inline_markup(text)}</div>'
    )


def render_glossary(groups: Mapping[str, Iterable[dict[str, str]]]) -> str:
    """Render glossary entries as one compact two-column table. `groups` maps a
    heading to its entries (dicts with a ``term`` and a ``definition``;
    ``**bold**`` spans are allowed in the definition). Markup is kept minimal on
    purpose -- the glossary sits at the very bottom of a report that has to stay
    under Gmail's ~102KB clipping limit, where it would be the first part
    clipped. Phones that honor <style> stack each definition under its term
    (see ``.bk-gloss`` in _RESPONSIVE_CSS)."""
    rows: list[str] = []
    for title, entries in groups.items():
        group_rows: list[str] = []
        for index, entry in enumerate(entries):
            term = html.escape(str(entry.get("term") or ""))
            definition = _inline_markup(str(entry.get("definition") or ""))
            stripe = f' style="background:{_ROW_ALT_BG};"' if index % 2 else ""
            # The first term cell sizes the whole column.
            width = ' width="150"' if not rows and not group_rows else ""
            group_rows.append(
                f'<tr valign="top"{stripe}><th align="left"{width}>{term}</th><td>{definition}</td></tr>'
            )
        if group_rows and title:
            gap = 16 if rows else 4
            rows.append(
                f'<tr><th colspan="2" align="left" style="padding-top:{gap}px;font-size:12px;'
                f'text-transform:uppercase;border-bottom:1px solid {_PANEL_BORDER};">{html.escape(title)}</th></tr>'
            )
        rows.extend(group_rows)
    if not rows:
        return ""
    # cellpadding is an HTML attribute, so it survives clients that ignore
    # <style>; color/size/line-height are inherited by every cell.
    return (
        '<table class="bk-gloss" role="presentation" cellpadding="7" cellspacing="0" border="0" width="100%" '
        f'style="width:100%;border-collapse:collapse;color:{_TEXT_BODY};font-size:13px;line-height:1.5;">'
        f'{"".join(rows)}</table>'
    )


_MIN_TILE_SLOTS = 4


def _tiles_per_row(count: int) -> int:
    """Tile slots per row for `count` stat tiles: at most six across, spread
    evenly (7 tiles -> 4 + 3, not 6 + 1) so a wrapped last row isn't a lone
    orphan, and never fewer than four slots so a short strip (two tiles)
    doesn't stretch into huge half-width panels."""
    if count <= 0:
        return _MIN_TILE_SLOTS
    rows = math.ceil(count / 6)
    return max(math.ceil(count / rows), _MIN_TILE_SLOTS)


def _tile_value_font_size(value: str) -> int:
    if len(value) <= 8:
        return 26
    if len(value) <= 12:
        return 20
    return 16


def render_metric_tiles(metrics: Iterable[dict[str, str]]) -> str:
    """Render Grafana-style stat panels. Tiles are wrapping inline-block divs
    (no <table>), sized as an equal share of the row but never narrower than
    150px, so they reflow from a single row on a laptop to two-up and one-up
    on a phone without needing media-query support in the email client."""
    metric_list = list(metrics)
    width_pct = math.floor(100 / _tiles_per_row(len(metric_list)) * 100) / 100
    tiles = []
    for index, metric in enumerate(metric_list):
        label = html.escape(str(metric.get("label") or ""))
        raw_value = str(metric.get("value") or "—")
        value = html.escape(raw_value)
        accent = _TILE_ACCENTS[index % len(_TILE_ACCENTS)]
        tiles.append(
            f'<div class="bk-stat" style="display:inline-block;vertical-align:top;width:{width_pct}%;min-width:150px;'
            'padding:0 8px 8px 0;box-sizing:border-box;">'
            f'<div style="background:{_PANEL_BG};border:1px solid {_PANEL_BORDER};border-left:3px solid {accent};'
            'border-radius:2px;padding:10px 14px;">'
            f'<div style="color:{_TEXT_SECONDARY};font-size:12px;line-height:1.3;">{label}</div>'
            f'<div class="bk-stat-value" style="color:{_TEXT_PRIMARY};font-size:{_tile_value_font_size(raw_value)}px;'
            f'font-weight:600;line-height:32px;min-height:32px;margin-top:4px;word-wrap:break-word;">{value}</div>'
            "</div>"
            "</div>"
        )
    return f'<div style="font-size:0;margin:0 -8px 8px 0;">{"".join(tiles)}</div>'


def render_html_table(columns: list[str], rows: list[dict[str, object]]) -> str:
    def _extra_class(column: str) -> str:
        return ' class="bk-hide-sm"' if column.strip().lower() in _HIDE_ON_SMALL_COLUMNS else ""

    header_cells = []
    for column in columns:
        hint = _COLUMN_HINTS.get(column.strip().lower())
        hint_html = (
            f'<div style="font-size:10px;font-weight:400;color:{_TEXT_SECONDARY};">{html.escape(hint)}</div>' if hint else ""
        )
        header_cells.append(
            f'<th{_extra_class(column)} align="left" nowrap="nowrap" style="background:{_HEADER_BG};color:{_TEXT_BODY};'
            f'font-size:12px;font-weight:600;text-align:left;white-space:nowrap;border-bottom:1px solid {_PANEL_BORDER};">'
            f"{html.escape(column)}{hint_html}</th>"
        )
    headers = "".join(header_cells)

    domain_maxes: dict[str, float] = {}
    for column in columns:
        column_key = column.strip().lower()
        if column_key in _BAR_COLUMNS and _BAR_COLUMNS[column_key] is None:
            observed = [n for row in rows if (n := _numeric_value(row.get(column))) is not None]
            domain_maxes[column_key] = max(observed) if observed else 0.0

    body_rows = []
    for row_index, row in enumerate(rows):
        row_style = f"border-bottom:1px solid {_ROW_BORDER};"
        if row_index % 2:
            row_style += f"background:{_ROW_ALT_BG};"
        cells = "".join(
            f"<td{_extra_class(column)} valign=\"middle\">"
            f"{_render_cell(column, row.get(column, '—'), domain_max=domain_maxes.get(column.strip().lower()))}</td>"
            for column in columns
        )
        body_rows.append(f'<tr style="{row_style}">{cells}</tr>')
    if not body_rows:
        body_rows.append(
            f'<tr><td colspan="{max(len(columns), 1)}" style="color:{_TEXT_SECONDARY};">No rows available.</td></tr>'
        )
    # cellpadding (an HTML attribute, so it survives clients that strip or
    # ignore <style>) provides the cell padding; the phone media query only
    # tightens it. Color/size/font are inherited by every cell from the table.
    return (
        '<div style="overflow-x:auto;-webkit-overflow-scrolling:touch;">'
        f'<table class="bk-table" role="presentation" cellpadding="8" cellspacing="0" border="0" width="100%" '
        f'style="width:100%;min-width:560px;border-collapse:collapse;color:{_TEXT_BODY};font-size:13px;line-height:1.35;">'
        f"<thead><tr>{headers}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"
        "</div>"
    )


def render_panel(title: str, body_html: str, *, description: str | None = None) -> StyledSection:
    """Wrap `body_html` in a Grafana-style panel: a bordered card with a title
    bar and an optional muted description line above the body."""
    description_html = (
        f'<div style="color:{_TEXT_SECONDARY};font-size:12px;line-height:1.5;margin:0 0 10px 0;">{html.escape(description)}</div>'
        if description
        else ""
    )
    title_html = (
        f'<div style="padding:9px 14px;border-bottom:1px solid {_PANEL_BORDER};color:{_TEXT_BODY};'
        f'font-size:14px;font-weight:600;">{html.escape(title)}</div>'
        if title
        else ""
    )
    return StyledSection(
        f'<div style="background:{_PANEL_BG};border:1px solid {_PANEL_BORDER};border-radius:2px;margin:0 0 16px 0;">'
        f"{title_html}"
        f'<div class="bk-panel-body" style="padding:12px 14px;">{description_html}{body_html}</div>'
        "</div>"
    )


def render_row_title(title: str) -> StyledSection:
    """A Grafana "row" heading that groups the panels below it."""
    return StyledSection(
        f'<div style="margin:24px 0 10px 0;color:{_TEXT_BODY};font-size:16px;font-weight:600;">'
        f'<span style="color:{_TEXT_SECONDARY};">&#9662;</span>&nbsp;{html.escape(title)}</div>'
    )


# Progressive enhancement only: the layout above is fluid and fully inline-
# styled, so it already adapts without this block. Clients that honor <style>
# (Gmail web/app with a Google account, Apple Mail, Outlook.com) additionally
# get the tighter phone layout below.
_RESPONSIVE_CSS = """
@media only screen and (max-width: 720px) {
  .bk-canvas { padding: 8px !important; }
  .bk-header { padding: 12px !important; }
  .bk-title { font-size: 20px !important; }
  .bk-stat { width: 50% !important; min-width: 0 !important; }
  .bk-stat-value { font-size: 20px !important; }
  .bk-panel-body { padding: 8px !important; }
  .bk-table th, .bk-table td { padding: 5px 6px !important; font-size: 12px !important; }
  .bk-hide-sm { display: none !important; }
  .bk-gloss, .bk-gloss tbody, .bk-gloss tr, .bk-gloss th, .bk-gloss td { display: block !important; width: auto !important; }
  .bk-gloss th { padding: 8px 6px 2px 6px !important; }
  .bk-gloss td { padding: 0 6px 8px 6px !important; }
}
"""


def _render_subtitle_chips(subtitle: str) -> str:
    """Render ``"A · B · C"`` as Grafana-style variable chips."""
    parts = [part.strip() for part in subtitle.split(" · ") if part.strip()]
    return "".join(
        f'<span style="display:inline-block;margin:0 6px 6px 0;padding:3px 10px;background:{_HEADER_BG};'
        f'border:1px solid {_PANEL_BORDER};border-radius:2px;color:{_TEXT_BODY};font-size:12px;">{html.escape(part)}</span>'
        for part in parts
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
        section if isinstance(section, StyledSection) else render_panel("", section) for section in sections
    )
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="color-scheme" content="dark"><meta name="supported-color-schemes" content="dark">'
        f"<title>{html.escape(title)}</title><style>{_RESPONSIVE_CSS}</style></head>"
        f'<body style="margin:0;padding:0;background:{_CANVAS_BG};">'
        f'<div class="bk-canvas" style="background:{_CANVAS_BG};padding:16px;font-family:{_FONT_STACK};color:{_TEXT_BODY};">'
        f'<div style="width:100%;max-width:{_DASHBOARD_MAX_WIDTH_PX}px;margin:0 auto;">'
        f'<div class="bk-header" style="background:{_PANEL_BG};border:1px solid {_PANEL_BORDER};border-top:3px solid {_GREEN};'
        'border-radius:2px;padding:16px 18px;margin:0 0 16px 0;">'
        f'<div style="color:{_GREEN};font-size:12px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;">BK Self</div>'
        f'<h1 class="bk-title" style="color:{_TEXT_PRIMARY};margin:6px 0 10px 0;font-size:26px;font-weight:600;line-height:1.2;">{html.escape(title)}</h1>'
        f"<div>{_render_subtitle_chips(subtitle)}</div>"
        "</div>"
        f"{metrics_html}"
        f"{body_sections}"
        "</div></div></body></html>"
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

