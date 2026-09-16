from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts import notify_workflow_failure


class BuildRunUrlTests(unittest.TestCase):
    def test_builds_url_from_github_env_vars(self):
        env = {
            "GITHUB_SERVER_URL": "https://github.com",
            "GITHUB_REPOSITORY": "bkabwe/bk-fin-and-inv",
            "GITHUB_RUN_ID": "12345",
        }
        with patch.dict("os.environ", env, clear=False):
            url = notify_workflow_failure.build_run_url()

        self.assertEqual(url, "https://github.com/bkabwe/bk-fin-and-inv/actions/runs/12345")

    def test_returns_empty_string_when_env_vars_missing(self):
        with patch.dict(
            "os.environ",
            {"GITHUB_SERVER_URL": "", "GITHUB_REPOSITORY": "", "GITHUB_RUN_ID": ""},
            clear=False,
        ):
            url = notify_workflow_failure.build_run_url()

        self.assertEqual(url, "")


class BuildAlertHtmlTests(unittest.TestCase):
    def test_includes_workflow_name_repo_and_run_link(self):
        env = {
            "GITHUB_REPOSITORY": "bkabwe/bk-fin-and-inv",
            "GITHUB_REF_NAME": "main",
            "GITHUB_SERVER_URL": "https://github.com",
            "GITHUB_RUN_ID": "999",
        }
        with patch.dict("os.environ", env, clear=False):
            html_content = notify_workflow_failure.build_alert_html("Train LightGBM Batch")

        self.assertIn("Train LightGBM Batch", html_content)
        self.assertIn("bkabwe/bk-fin-and-inv", html_content)
        self.assertIn("main", html_content)
        self.assertIn("https://github.com/bkabwe/bk-fin-and-inv/actions/runs/999", html_content)

    def test_omits_run_link_when_context_missing(self):
        with patch.dict(
            "os.environ",
            {"GITHUB_SERVER_URL": "", "GITHUB_REPOSITORY": "", "GITHUB_RUN_ID": ""},
            clear=False,
        ):
            html_content = notify_workflow_failure.build_alert_html("Scan Email Short-Term")

        self.assertNotIn("<a href", html_content)


class SendFailureAlertTests(unittest.TestCase):
    @patch("scripts.notify_workflow_failure.send_brevo_email")
    def test_returns_true_when_email_sent(self, mock_send):
        mock_send.return_value = 2

        result = notify_workflow_failure.send_failure_alert("Scan Email Short-Term")

        self.assertTrue(result)
        mock_send.assert_called_once()
        self.assertIn("Scan Email Short-Term", mock_send.call_args.kwargs["subject"])

    @patch("scripts.notify_workflow_failure.send_brevo_email")
    def test_returns_false_when_no_recipients_sent(self, mock_send):
        mock_send.return_value = 0

        result = notify_workflow_failure.send_failure_alert("Scan Email Short-Term")

        self.assertFalse(result)

    @patch("scripts.notify_workflow_failure.send_brevo_email", side_effect=RuntimeError("BREVO_API_KEY is not configured"))
    def test_swallows_exceptions_and_returns_false(self, mock_send):
        result = notify_workflow_failure.send_failure_alert("Scan Email Short-Term")

        self.assertFalse(result)


class MainTests(unittest.TestCase):
    @patch("scripts.notify_workflow_failure.send_failure_alert", return_value=True)
    def test_main_always_returns_zero_on_success(self, mock_alert):
        with patch("sys.argv", ["notify_workflow_failure.py", "--workflow", "Train LightGBM Batch"]):
            exit_code = notify_workflow_failure.main()

        self.assertEqual(exit_code, 0)
        mock_alert.assert_called_once_with("Train LightGBM Batch")

    @patch("scripts.notify_workflow_failure.send_failure_alert", return_value=False)
    def test_main_always_returns_zero_even_when_alert_fails(self, mock_alert):
        with patch("sys.argv", ["notify_workflow_failure.py", "--workflow", "Train LightGBM Batch"]):
            exit_code = notify_workflow_failure.main()

        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
