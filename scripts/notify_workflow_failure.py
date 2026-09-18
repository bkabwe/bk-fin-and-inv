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

For jobs that can be cancelled by a `timeout-minutes` guardrail or GitHub's
hard 6h job limit (e.g. a sharded scan job), a *cancelled* job's conclusion is
NOT `failure`, so `if: failure()` alone will not trigger this notifier and the
gap would go unnoticed. Use `if: failure() || cancelled()` together with
`--job-results` so the alert can distinguish the two cases:

    notify-on-failure:
      needs: [discover, scan-shard, reduce]
      if: failure() || cancelled()
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-python@v5
          with:
            python-version: ${{ env.PYTHON_VERSION }}
        - run: pip install requests
        - run: |
            python scripts/notify_workflow_failure.py \
              --workflow "Scan Email Short-Term" \
              --job-results '${{ toJson(needs) }}'

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
import json
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


def any_job_cancelled(job_results_json: str | None) -> bool:
    """Best-effort detection of a cancelled/timed-out job from a JSON-encoded
    GitHub Actions `needs` context (e.g. ``${{ toJson(needs) }}``).

    A job that hits its `timeout-minutes` limit -- or the hard 6h GitHub
    Actions job limit when no explicit timeout is set -- gets a `cancelled`
    conclusion, NOT `failure`. Since the `notify-on-failure` job normally
    only runs on `if: failure()`, a cancelled shard could otherwise fail
    completely silently. Callers should combine this with an
    `if: failure() || cancelled()` job condition.
    """
    if not job_results_json:
        return False
    try:
        parsed = json.loads(job_results_json)
    except (TypeError, ValueError):
        return False
    if not isinstance(parsed, dict):
        return False
    return any(isinstance(value, dict) and value.get("result") == "cancelled" for value in parsed.values())


def build_alert_html(workflow_name: str, *, cancelled: bool = False) -> str:
    repo = os.getenv("GITHUB_REPOSITORY", "unknown/unknown")
    ref = os.getenv("GITHUB_REF_NAME", "unknown")
    run_url = build_run_url()
    link_html = (
        f'<p style="margin:18px 0 0 0;"><a href="{run_url}" style="color:#5ab8ff;font-weight:600;">View the failed run &rarr;</a></p>'
        if run_url
        else ""
    )
    if cancelled:
        badge_text = "&#9201; Job cancelled / timed out"
        message = (
            "One or more jobs in this scheduled workflow were <strong>cancelled</strong>, most likely because they "
            "exceeded a job time limit (either an explicit <code>timeout-minutes</code> guardrail or GitHub's hard "
            "6h job limit). Unlike a normal failure, this silently degrades today's results: the affected shard(s) "
            "contribute no screener/profit-opportunities picks and no prediction-tracker history entries for this "
            "run. Check the run logs for the cancelled job and consider re-running it manually."
        )
    else:
        badge_text = "&#9888; Workflow failed"
        message = "One or more jobs in this scheduled workflow failed. Check the run logs for details."
    return (
        "<html><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width, initial-scale=1"></head>'
        "<body style=\"font-family:Arial,Helvetica,sans-serif;background:#06111d;color:#ecf5ff;margin:0;padding:24px;\">"
        '<div style="max-width:640px;margin:0 auto;background:#0f1f33;border:1px solid #1f4f46;border-radius:18px;padding:28px;">'
        '<span style="display:inline-block;padding:3px 10px;border-radius:999px;background:#3a1414;color:#ff6b6b;'
        f'font-size:12px;font-weight:700;letter-spacing:0.04em;text-transform:uppercase;">{badge_text}</span>'
        f'<h2 style="color:#f4fff8;margin:14px 0 6px 0;">{workflow_name}</h2>'
        f'<p style="color:#9eb3cf;margin:0;">Repository: <strong style="color:#ecf5ff;">{repo}</strong><br>'
        f'Branch/ref: <strong style="color:#ecf5ff;">{ref}</strong></p>'
        f"{link_html}"
        f'<p style="color:#9eb3cf;margin:18px 0 0 0;">{message}</p>'
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


def send_failure_alert(workflow_name: str, *, job_results_json: str | None = None) -> bool:
    """Best-effort send; returns True if the alert was sent to at least one recipient."""
    cancelled = any_job_cancelled(job_results_json)
    try:
        recipients = resolve_alert_recipients()
        subject_prefix = "\u23f1\ufe0f Workflow job cancelled/timed out" if cancelled else "\u26a0\ufe0f Workflow failed"
        sent = send_brevo_email(
            subject=f"{subject_prefix}: {workflow_name}",
            html_content=build_alert_html(workflow_name, cancelled=cancelled),
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
    parser.add_argument(
        "--job-results",
        default=None,
        help=(
            "Optional JSON-encoded GitHub Actions `needs` context (e.g. '${{ toJson(needs) }}'), used to detect a "
            "cancelled/timed-out job (as opposed to a normal failure) and adjust the alert wording accordingly."
        ),
    )
    args = parser.parse_args()

    send_failure_alert(args.workflow, job_results_json=args.job_results)
    # Always exit 0: this notify step's own success/failure should never
    # affect the workflow's overall status or hide the original failure.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
