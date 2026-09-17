"""Send a lightweight email alert when a scheduled GitHub Actions workflow
fails, so pipeline breaks (e.g. the reduce-promote job failures fixed in
recent PRs) surface proactively instead of being caught by chance.

This is meant to be wired up as a `notify-on-failure` job with
`if: failure()` and `needs: [<the workflow's other jobs>]`, so it only runs
when at least one of those jobs failed:

    notify-on-failure:
      needs: [scan-email-report]
      if: failure()
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-python@v5
          with:
            python-version: ${{ env.PYTHON_VERSION }}
        - run: pip install requests
        - run: python scripts/notify_workflow_failure.py --workflow "Scan Email Short-Term"

It reuses the existing Brevo email integration (modules.email_reports), so it
only needs BREVO_API_KEY and SCAN_EMAIL_FROM (already configured for the
scan/grading email reports) -- no new secrets or notification channel
required. Run/repo context is read from the standard GitHub Actions
environment variables, so no extra inputs are required beyond --workflow.

Failure alerts deliberately do NOT go to SCAN_EMAIL_RECIPIENTS (the broader
scan/grading report distribution list) -- that list may include people who
shouldn't be paged about internal pipeline/CI failures. Instead, recipients
are resolved via WORKFLOW_ALERT_RECIPIENTS (a comma-separated list, optional)
falling back to SCAN_EMAIL_FROM alone (i.e. just the maintainer) when that
variable isn't set. See resolve_alert_recipients() below.

Deliberately lightweight: only `requests` is needed (not the full
requirements-workflows.txt), so this step can still run and notify even if
an earlier job failed during dependency installation.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.email_reports import send_brevo_email  # noqa: E402
from modules.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def build_run_url() -> str:
    """Best-effort link to the failed run, from GitHub Actions' env vars."""
    server = os.getenv("GITHUB_SERVER_URL", "").strip()
    repo = os.getenv("GITHUB_REPOSITORY", "").strip()
    run_id = os.getenv("GITHUB_RUN_ID", "").strip()
    if not server or not repo or not run_id:
        return ""
    return f"{server}/{repo}/actions/runs/{run_id}"


def build_alert_html(workflow_name: str) -> str:
    repo = os.getenv("GITHUB_REPOSITORY", "unknown/unknown")
    ref = os.getenv("GITHUB_REF_NAME", "unknown")
    run_url = build_run_url()
    link_html = (
        f'<p style="margin:18px 0 0 0;"><a href="{run_url}" style="color:#5ab8ff;font-weight:600;">View the failed run &rarr;</a></p>'
        if run_url
        else ""
    )
    return (
        "<html><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width, initial-scale=1"></head>'
        "<body style=\"font-family:Arial,Helvetica,sans-serif;background:#06111d;color:#ecf5ff;margin:0;padding:24px;\">"
        '<div style="max-width:640px;margin:0 auto;background:#0f1f33;border:1px solid #1f4f46;border-radius:18px;padding:28px;">'
        '<span style="display:inline-block;padding:3px 10px;border-radius:999px;background:#3a1414;color:#ff6b6b;'
        'font-size:12px;font-weight:700;letter-spacing:0.04em;text-transform:uppercase;">&#9888; Workflow failed</span>'
        f'<h2 style="color:#f4fff8;margin:14px 0 6px 0;">{workflow_name}</h2>'
        f'<p style="color:#9eb3cf;margin:0;">Repository: <strong style="color:#ecf5ff;">{repo}</strong><br>'
        f'Branch/ref: <strong style="color:#ecf5ff;">{ref}</strong></p>'
        f"{link_html}"
        '<p style="color:#9eb3cf;margin:18px 0 0 0;">One or more jobs in this scheduled workflow failed. Check the run logs for details.</p>'
        "</div></body></html>"
    )


def resolve_alert_recipients() -> list[str]:
    """Recipients for failure alerts -- deliberately independent of
    SCAN_EMAIL_RECIPIENTS (the broader scan/grading report distribution
    list), since a workflow/pipeline failure is an operational concern for
    the maintainer, not something every report recipient needs to see.

    Prefers WORKFLOW_ALERT_RECIPIENTS (comma-separated) when set; otherwise
    falls back to the single SCAN_EMAIL_FROM address (i.e. just the
    maintainer's own verified sender address).
    """
    raw = str(os.getenv("WORKFLOW_ALERT_RECIPIENTS") or "").strip()
    if raw:
        recipients = [value.strip() for value in raw.split(",") if value.strip()]
        if recipients:
            return recipients

    fallback = str(os.getenv("SCAN_EMAIL_FROM") or "").strip()
    if fallback:
        return [fallback]

    raise RuntimeError("Neither WORKFLOW_ALERT_RECIPIENTS nor SCAN_EMAIL_FROM is configured")


def send_failure_alert(workflow_name: str) -> bool:
    """Best-effort send; returns True if the alert was sent to at least one recipient."""
    try:
        recipients = resolve_alert_recipients()
        sent = send_brevo_email(
            subject=f"\u26a0\ufe0f Workflow failed: {workflow_name}",
            html_content=build_alert_html(workflow_name),
            recipients=recipients,
        )
        if sent:
            logger.info("Sent workflow-failure alert for %s to %d recipient(s)", workflow_name, sent)
            return True
        logger.warning("Workflow-failure alert for %s was not delivered to any recipient", workflow_name)
        return False
    except Exception as exc:
        # A broken notifier must never mask or fail the CI job further --
        # it's only trying to surface a failure that already happened.
        logger.warning("Failed to send workflow-failure alert for %s: %s", workflow_name, exc)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workflow", required=True, help="Human-readable workflow name to show in the alert email")
    args = parser.parse_args()

    send_failure_alert(args.workflow)
    # Always exit 0: this notify step's own success/failure should never
    # affect the workflow's overall status or hide the original failure.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
