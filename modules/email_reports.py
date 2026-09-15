from __future__ import annotations

import base64
import html
import os
from typing import Iterable

import requests

BREVO_EMAIL_API_URL = "https://api.brevo.com/v3/smtp/email"


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


def render_metric_tiles(metrics: Iterable[dict[str, str]]) -> str:
    tiles = []
    for metric in metrics:
        label = html.escape(str(metric.get("label") or ""))
        value = html.escape(str(metric.get("value") or "—"))
        tiles.append(
            "<td style=\"padding:8px;vertical-align:top;\">"
            "<div style=\"background:#10233f;border:1px solid #1f4f46;border-radius:14px;padding:16px;min-width:150px;\">"
            f"<div style=\"color:#9eb3cf;font-size:12px;text-transform:uppercase;letter-spacing:0.08em;\">{label}</div>"
            f"<div style=\"color:#f4fff8;font-size:24px;font-weight:700;margin-top:8px;\">{value}</div>"
            "</div>"
            "</td>"
        )
    return "<table role=\"presentation\" cellpadding=\"0\" cellspacing=\"0\" style=\"border-collapse:collapse;\"><tr>" + "".join(tiles) + "</tr></table>"


def render_html_table(columns: list[str], rows: list[dict[str, object]]) -> str:
    headers = "".join(
        f"<th style=\"padding:10px 12px;text-align:left;background:#0d1b2a;color:#dce8f7;border-bottom:1px solid #1f4f46;\">{html.escape(column)}</th>"
        for column in columns
    )
    body_rows = []
    for row in rows:
        cells = "".join(
            f"<td style=\"padding:10px 12px;border-bottom:1px solid #1c314b;color:#ecf5ff;\">{html.escape(str(row.get(column, '—')))}</td>"
            for column in columns
        )
        body_rows.append(f"<tr>{cells}</tr>")
    if not body_rows:
        body_rows.append(
            f"<tr><td colspan=\"{max(len(columns), 1)}\" style=\"padding:12px;color:#9eb3cf;\">No rows available.</td></tr>"
        )
    return (
        "<table role=\"presentation\" cellpadding=\"0\" cellspacing=\"0\" style=\"width:100%;border-collapse:collapse;background:#0b1725;border:1px solid #1f4f46;border-radius:14px;overflow:hidden;\">"
        f"<thead><tr>{headers}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"
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
        f"<div style=\"margin-top:24px;background:#0f1f33;border:1px solid #1f4f46;border-radius:18px;padding:22px;\">{section}</div>"
        for section in sections
    )
    return (
        "<html><body style=\"margin:0;padding:24px;background:#06111d;font-family:Arial,Helvetica,sans-serif;\">"
        "<div style=\"max-width:1080px;margin:0 auto;background:#081423;border:1px solid #1f4f46;border-radius:24px;padding:28px;\">"
        f"<div style=\"color:#39d98a;font-size:13px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;\">BK Self</div>"
        f"<h1 style=\"color:#f4fff8;margin:10px 0 6px 0;font-size:30px;\">{html.escape(title)}</h1>"
        f"<p style=\"color:#a9bdd7;margin:0 0 20px 0;font-size:15px;\">{html.escape(subtitle)}</p>"
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
            print(f"Brevo email sent to {recipient}")
        except Exception as exc:
            message = f"Brevo send failed for {recipient}: {exc}"
            print(message)
            failures.append(message)

    if failures:
        raise RuntimeError("; ".join(failures))
    return sent_count
