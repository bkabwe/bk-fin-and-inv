from __future__ import annotations

import base64
import unittest
from unittest.mock import MagicMock, patch

from modules.email_reports import (
    BREVO_EMAIL_API_URL,
    build_sender,
    csv_attachment,
    parse_recipients,
    render_html_table,
    render_legend,
    render_metric_tiles,
    render_report_html,
    send_brevo_email,
)


class ParseRecipientsTests(unittest.TestCase):
    def test_parses_comma_separated_env_value(self):
        with patch.dict("os.environ", {"SCAN_EMAIL_RECIPIENTS": "a@x.com, b@y.com"}, clear=False):
            self.assertEqual(parse_recipients(), ["a@x.com", "b@y.com"])

    def test_explicit_raw_value_overrides_env(self):
        with patch.dict("os.environ", {"SCAN_EMAIL_RECIPIENTS": "env@x.com"}, clear=False):
            self.assertEqual(parse_recipients("explicit@x.com"), ["explicit@x.com"])

    def test_strips_whitespace_and_drops_empties(self):
        self.assertEqual(parse_recipients(" a@x.com ,, b@y.com ,"), ["a@x.com", "b@y.com"])

    def test_empty_raises_runtime_error(self):
        with (
            patch.dict("os.environ", {"SCAN_EMAIL_RECIPIENTS": ""}, clear=False),
            self.assertRaises(RuntimeError),
        ):
            parse_recipients()

    def test_none_and_missing_env_raises(self):
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(RuntimeError):
            parse_recipients(None)


class BuildSenderTests(unittest.TestCase):
    def test_returns_name_and_email(self):
        with patch.dict("os.environ", {"SCAN_EMAIL_FROM": "sender@x.com"}, clear=False):
            self.assertEqual(build_sender(), {"name": "BK Self", "email": "sender@x.com"})

    def test_missing_env_raises(self):
        with (
            patch.dict("os.environ", {"SCAN_EMAIL_FROM": ""}, clear=False),
            self.assertRaises(RuntimeError),
        ):
            build_sender()


class CsvAttachmentTests(unittest.TestCase):
    def test_encodes_content_as_base64(self):
        result = csv_attachment("report.csv", "a,b\n1,2\n")
        self.assertEqual(result["name"], "report.csv")
        self.assertEqual(base64.b64decode(result["content"]).decode("utf-8"), "a,b\n1,2\n")

    def test_strips_filename_whitespace(self):
        result = csv_attachment("  report.csv  ", "x")
        self.assertEqual(result["name"], "report.csv")


class RenderMetricTilesTests(unittest.TestCase):
    def test_renders_wrapping_tiles_with_labels_and_values(self):
        html_out = render_metric_tiles([{"label": "Score", "value": "87"}])
        self.assertIn("Score", html_out)
        self.assertIn("87", html_out)
        # Tiles use wrapping inline-block divs (not a non-wrapping <table>/<td>
        # row) so they don't overflow the container on desktop Gmail web.
        self.assertIn("display:inline-block", html_out)
        self.assertNotIn("<table", html_out)

    def test_missing_value_defaults_to_dash(self):
        html_out = render_metric_tiles([{"label": "Score"}])
        self.assertIn("—", html_out)

    def test_empty_iterable_returns_empty_wrapper(self):
        html_out = render_metric_tiles([])
        self.assertNotIn("display:inline-block", html_out)


class RenderHtmlTableTests(unittest.TestCase):
    def test_renders_headers_and_rows(self):
        html_out = render_html_table(["Ticker", "Score"], [{"Ticker": "AAPL", "Score": 90}])
        self.assertIn("Ticker", html_out)
        self.assertIn("AAPL", html_out)
        self.assertIn("90", html_out)

    def test_missing_cell_defaults_to_dash(self):
        html_out = render_html_table(["Ticker", "Score"], [{"Ticker": "AAPL"}])
        self.assertIn("—", html_out)

    def test_no_rows_renders_placeholder(self):
        html_out = render_html_table(["Ticker"], [])
        self.assertIn("No rows available.", html_out)

    def test_escapes_html_in_cell_values(self):
        html_out = render_html_table(["Ticker"], [{"Ticker": "<script>alert(1)</script>"}])
        self.assertNotIn("<script>alert(1)</script>", html_out)
        self.assertIn("&lt;script&gt;", html_out)

    def test_boolean_cell_renders_yes_no_badge(self):
        html_out = render_html_table(["Point Hit"], [{"Point Hit": True}, {"Point Hit": False}])
        self.assertIn(">Yes<", html_out)
        self.assertIn(">No<", html_out)

    def test_confidence_column_renders_badge(self):
        html_out = render_html_table(["Confidence"], [{"Confidence": "Full"}])
        self.assertIn("border-radius:999px", html_out)
        self.assertIn("Full", html_out)

    def test_score_column_renders_bar_cell_scaled_to_fixed_domain(self):
        html_out = render_html_table(["Score"], [{"Score": 50}])
        # Score's domain is fixed at 0-100, so a value of 50 fills half the bar.
        self.assertIn("width:50.0%", html_out)

    def test_dynamic_bar_column_scales_to_max_value_in_table(self):
        html_out = render_html_table(
            ["Projected Upside %"],
            [{"Projected Upside %": 10.0}, {"Projected Upside %": 20.0}],
        )
        self.assertIn("width:50.0%", html_out)
        self.assertIn("width:100.0%", html_out)


class RenderLegendTests(unittest.TestCase):
    def test_renders_badge_and_description_for_each_item(self):
        html_out = render_legend(
            [
                {"label": "Full", "description": "All models available."},
                {"label": "Limited", "description": "Partial model coverage."},
            ]
        )
        self.assertIn("Full", html_out)
        self.assertIn("All models available.", html_out)
        self.assertIn("Limited", html_out)
        self.assertIn("Partial model coverage.", html_out)


class RenderReportHtmlTests(unittest.TestCase):
    def test_includes_title_subtitle_metrics_and_sections(self):
        html_out = render_report_html(
            title="Weekly Scan",
            subtitle="Top picks",
            metrics=[{"label": "Count", "value": "5"}],
            sections=["<div>section-a</div>"],
        )
        self.assertIn("Weekly Scan", html_out)
        self.assertIn("Top picks", html_out)
        self.assertIn("Count", html_out)
        self.assertIn("section-a", html_out)

    def test_escapes_title_and_subtitle(self):
        html_out = render_report_html(
            title="<b>bold</b>",
            subtitle="<i>italic</i>",
            metrics=[],
            sections=[],
        )
        self.assertNotIn("<b>bold</b>", html_out)
        self.assertNotIn("<i>italic</i>", html_out)


class SendBrevoEmailTests(unittest.TestCase):
    def _env(self, **overrides):
        env = {
            "BREVO_API_KEY": "test-key",
            "SCAN_EMAIL_FROM": "sender@x.com",
            "SCAN_EMAIL_RECIPIENTS": "r1@x.com,r2@x.com",
        }
        env.update(overrides)
        return env

    def test_missing_api_key_raises(self):
        with (
            patch.dict("os.environ", {"BREVO_API_KEY": ""}, clear=False),
            self.assertRaises(RuntimeError),
        ):
            send_brevo_email(subject="s", html_content="<p>hi</p>")

    @patch("modules.email_reports.requests.Session")
    def test_sends_to_all_recipients_and_returns_count(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session.post.return_value = MagicMock(status_code=200, text="ok")
        mock_session_cls.return_value = mock_session

        with patch.dict("os.environ", self._env(), clear=False):
            sent = send_brevo_email(subject="Report", html_content="<p>hi</p>")

        self.assertEqual(sent, 2)
        self.assertEqual(mock_session.post.call_count, 2)
        called_url = mock_session.post.call_args_list[0].args[0]
        self.assertEqual(called_url, BREVO_EMAIL_API_URL)

    @patch("modules.email_reports.requests.Session")
    def test_includes_attachments_in_payload(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session.post.return_value = MagicMock(status_code=200, text="ok")
        mock_session_cls.return_value = mock_session
        attachment = csv_attachment("r.csv", "a,b")

        with patch.dict("os.environ", self._env(), clear=False):
            send_brevo_email(
                subject="Report",
                html_content="<p>hi</p>",
                attachments=[attachment],
                recipients=["only@x.com"],
            )

        payload = mock_session.post.call_args_list[0].kwargs["json"]
        self.assertEqual(payload["attachment"], [attachment])
        self.assertEqual(payload["to"], [{"email": "only@x.com"}])

    @patch("modules.email_reports.requests.Session")
    def test_raises_when_all_sends_fail(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session.post.return_value = MagicMock(status_code=500, text="server error")
        mock_session_cls.return_value = mock_session

        with (
            patch.dict("os.environ", self._env(), clear=False),
            self.assertRaises(RuntimeError),
        ):
            send_brevo_email(subject="Report", html_content="<p>hi</p>")

    @patch("modules.email_reports.requests.Session")
    def test_partial_failure_still_raises_but_reports_prior_successes(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session.post.side_effect = [
            MagicMock(status_code=200, text="ok"),
            MagicMock(status_code=400, text="bad request"),
        ]
        mock_session_cls.return_value = mock_session

        with (
            patch.dict("os.environ", self._env(), clear=False),
            self.assertRaises(RuntimeError) as ctx,
        ):
            send_brevo_email(subject="Report", html_content="<p>hi</p>")

        self.assertIn("r2@x.com", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
