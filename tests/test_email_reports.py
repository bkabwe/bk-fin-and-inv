from __future__ import annotations

import base64
import unittest
from unittest.mock import MagicMock, patch

from modules import email_reports
from modules.email_reports import (
    BREVO_EMAIL_API_URL,
    StyledSection,
    build_sender,
    csv_attachment,
    parse_recipients,
    render_callout,
    render_glossary,
    render_html_table,
    render_legend,
    render_metric_tiles,
    render_panel,
    render_paragraph,
    render_report_html,
    render_row_title,
    render_subheading,
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


class StatTileLayoutTests(unittest.TestCase):
    @staticmethod
    def _widths(html_out: str) -> set[str]:
        import re

        return set(re.findall(r"display:inline-block;vertical-align:top;width:([0-9.]+)%", html_out))

    def test_short_strips_do_not_stretch_into_half_width_panels(self):
        html_out = render_metric_tiles([{"label": "A", "value": "1"}, {"label": "B", "value": "2"}])
        self.assertEqual(self._widths(html_out), {"25.0"})

    def test_six_tiles_share_one_row(self):
        html_out = render_metric_tiles([{"label": str(i), "value": "1"} for i in range(6)])
        self.assertEqual(self._widths(html_out), {"16.66"})

    def test_seven_tiles_split_four_and_three_not_six_and_one(self):
        html_out = render_metric_tiles([{"label": str(i), "value": "1"} for i in range(7)])
        self.assertEqual(self._widths(html_out), {"25.0"})

    def test_tiles_keep_a_minimum_width_so_they_wrap_on_narrow_screens(self):
        self.assertIn("min-width:150px", render_metric_tiles([{"label": "A", "value": "1"}]))


class BarColorTests(unittest.TestCase):
    def test_score_bar_color_follows_the_recommendation_tiers(self):
        green, blue, yellow, orange, red = (
            email_reports._GREEN,
            email_reports._BLUE,
            email_reports._YELLOW,
            email_reports._ORANGE,
            email_reports._RED,
        )
        for score, color in [(95, green), (80, green), (79, blue), (65, blue), (64, yellow), (50, yellow), (49, orange), (35, orange), (34, red), (0, red)]:
            with self.subTest(score=score):
                self.assertIn(f"background:{color}", render_html_table(["Score"], [{"Score": score}]))

    def test_risk_adjusted_upside_bar_is_sized_against_a_zero_to_one_domain(self):
        html_out = render_html_table(["Risk-Adjusted Upside"], [{"Risk-Adjusted Upside": "0.50"}])
        self.assertIn("width:50.0%", html_out)

    def test_risk_adjusted_upside_bar_color_follows_the_rating_tiers(self):
        for value, color in [("0.95", email_reports._GREEN), ("0.60", email_reports._GREEN), ("0.45", email_reports._YELLOW), ("0.30", email_reports._YELLOW), ("0.10", email_reports._ORANGE)]:
            with self.subTest(value=value):
                html_out = render_html_table(["Risk-Adjusted Upside"], [{"Risk-Adjusted Upside": value}])
                self.assertIn(f"background:{color}", html_out)

    def test_projected_upside_bar_is_blue_for_gains_and_red_for_losses(self):
        gain = render_html_table(["Projected Upside %"], [{"Projected Upside %": 12.0}])
        loss = render_html_table(["Projected Upside %"], [{"Projected Upside %": -4.0}])
        self.assertIn(f"background:{email_reports._BLUE}", gain)
        self.assertIn(f"background:{email_reports._RED}", loss)

    def test_thresholds_stay_in_sync_with_the_scoring_code(self):
        from modules import profit_opportunities, scoring_engine

        self.assertEqual(email_reports._RISK_ADJUSTED_STRONG_MIN, profit_opportunities.RISK_ADJUSTED_STRONG_MIN)
        self.assertEqual(email_reports._RISK_ADJUSTED_MODERATE_MIN, profit_opportunities.RISK_ADJUSTED_MODERATE_MIN)
        emoji_to_color = {
            "\U0001f7e2": email_reports._GREEN,
            "\U0001f535": email_reports._BLUE,
            "\U0001f7e1": email_reports._YELLOW,
            "\U0001f7e0": email_reports._ORANGE,
            "\U0001f534": email_reports._RED,
            "\u26d4": email_reports._RED,
        }
        for score in range(0, 101):
            with self.subTest(score=score):
                tier_emoji = scoring_engine._recommendation(score)[0]
                self.assertEqual(email_reports._score_bar_color(score), emoji_to_color[tier_emoji])


class BadgeColumnTests(unittest.TestCase):
    def test_recommendation_column_renders_badge_despite_emoji_prefix(self):
        html_out = render_html_table(["Recommendation"], [{"Recommendation": "\U0001f7e2 STRONG BUY"}])
        self.assertIn("border-radius:999px", html_out)
        self.assertIn(f"color:{email_reports._GREEN}", html_out)

    def test_risk_adjusted_rating_badge_ignores_the_threshold_suffix(self):
        for text, fg in [
            ("Strong", email_reports._GREEN),
            ("Strong (0.60+)", email_reports._GREEN),
            ("Moderate (0.30 to <0.60)", email_reports._YELLOW),
            ("Weak (<0.30)", email_reports._ORANGE),
        ]:
            with self.subTest(text=text):
                html_out = render_html_table(["Risk-Adjusted Rating"], [{"Risk-Adjusted Rating": text}])
                self.assertIn("border-radius:999px", html_out)
                self.assertIn(f"color:{fg}", html_out)

    def test_unrecognized_badge_text_falls_back_to_plain_text(self):
        html_out = render_html_table(["Recommendation"], [{"Recommendation": "Something else"}])
        self.assertNotIn("border-radius:999px", html_out)
        self.assertIn("Something else", html_out)

    def test_column_hints_explain_the_range_and_direction(self):
        html_out = render_html_table(["Score", "Risk-Adjusted Upside"], [])
        self.assertIn("0–100 · higher = better", html_out)
        self.assertIn("0–1 · higher = better", html_out)

    def test_company_column_is_hidden_on_small_screens_only_via_class(self):
        html_out = render_html_table(["Ticker", "Company"], [{"Ticker": "AAA", "Company": "Alpha Inc"}])
        self.assertEqual(html_out.count("bk-hide-sm"), 2)  # one <th>, one <td>


class RenderTableScrollTests(unittest.TestCase):
    def test_table_scrolls_horizontally_instead_of_overflowing_the_page(self):
        html_out = render_html_table(["Ticker"], [{"Ticker": "AAA"}])
        self.assertIn("overflow-x:auto", html_out)


class RenderTextBlocksTests(unittest.TestCase):
    def test_paragraph_emphasizes_bold_markup_and_escapes_everything_else(self):
        html_out = render_paragraph("Range is **0 to 1** <script>alert(1)</script>")
        self.assertIn("<strong", html_out)
        self.assertIn(">0 to 1</strong>", html_out)
        self.assertNotIn("<script>", html_out)
        self.assertIn("&lt;script&gt;", html_out)

    def test_paragraph_cannot_smuggle_markup_through_the_bold_syntax(self):
        html_out = render_paragraph('**<img src=x onerror="boom">**')
        self.assertNotIn("<img", html_out)
        self.assertIn("&lt;img", html_out)

    def test_callout_and_subheading_escape_text(self):
        self.assertIn("&lt;b&gt;", render_callout("<b>x</b>"))
        self.assertIn("&lt;b&gt;", render_subheading("<b>x</b>"))


class RenderPanelTests(unittest.TestCase):
    def test_panel_renders_title_description_and_body(self):
        panel = render_panel("Top picks", "<table>body-html</table>", description="Why these")
        self.assertIsInstance(panel, StyledSection)
        self.assertIn("Top picks", panel)
        self.assertIn("Why these", panel)
        self.assertIn("<table>body-html</table>", panel)

    def test_panel_escapes_title_and_description_but_not_the_trusted_body(self):
        panel = render_panel("<b>t</b>", "<p>body</p>", description="<i>d</i>")
        self.assertIn("&lt;b&gt;t&lt;/b&gt;", panel)
        self.assertIn("&lt;i&gt;d&lt;/i&gt;", panel)
        self.assertIn("<p>body</p>", panel)

    def test_untitled_panel_has_no_title_bar(self):
        self.assertNotIn("font-weight:600", render_panel("", "<p>x</p>"))
        self.assertIn("font-weight:600", render_panel("Titled", "<p>x</p>"))

    def test_row_title_is_a_styled_section_and_escapes_text(self):
        row = render_row_title("Profit <opportunities>")
        self.assertIsInstance(row, StyledSection)
        self.assertIn("Profit &lt;opportunities&gt;", row)


class RenderLegendLayoutTests(unittest.TestCase):
    def test_all_items_share_one_table_so_descriptions_align(self):
        html_out = render_legend(
            [
                {"label": "Strong (0.60+)", "style_key": "strong", "description": "a"},
                {"label": "Weak (<0.30)", "style_key": "weak", "description": "b"},
            ]
        )
        self.assertEqual(html_out.count("<table"), 1)
        self.assertEqual(html_out.count("<tr>"), 2)

    def test_explicit_style_key_picks_the_badge_color(self):
        html_out = render_legend([{"label": "Strong (0.60+)", "style_key": "strong", "description": "a"}])
        self.assertIn(f"color:{email_reports._GREEN}", html_out)


class RenderGlossaryTests(unittest.TestCase):
    GROUPS = {
        "First group": [
            {"term": "Alpha", "definition": "First **bold** meaning"},
            {"term": "Beta", "definition": "Second meaning"},
        ],
        "Second group": [{"term": "Gamma", "definition": "Third meaning"}],
    }

    def test_renders_one_table_with_a_heading_per_group_and_a_row_per_entry(self):
        html_out = render_glossary(self.GROUPS)
        self.assertEqual(html_out.count("<table"), 1)
        self.assertIn('class="bk-gloss"', html_out)
        self.assertEqual(html_out.count("<tr"), 5)  # two group headings + three entries
        for expected in ("First group", "Second group", ">Alpha</th>", ">Beta</th>", ">Gamma</th>", "Third meaning"):
            self.assertIn(expected, html_out)
        self.assertLess(html_out.index("First group"), html_out.index(">Alpha</th>"))
        self.assertLess(html_out.index(">Beta</th>"), html_out.index("Second group"))
        self.assertLess(html_out.index("Second group"), html_out.index(">Gamma</th>"))

    def test_escapes_headings_terms_and_definitions_but_keeps_bold_markup(self):
        html_out = render_glossary({"<b>g</b>": [{"term": "<i>t</i>", "definition": "a **b** <script>x</script>"}]})
        for raw in ("<b>g</b>", "<i>t</i>", "<script>"):
            self.assertNotIn(raw, html_out)
        self.assertIn("&lt;i&gt;t&lt;/i&gt;", html_out)
        self.assertIn("&lt;b&gt;g&lt;/b&gt;", html_out)
        self.assertIn(">b</strong>", html_out)

    def test_alternate_rows_are_striped_and_only_the_first_term_cell_sets_the_column_width(self):
        html_out = render_glossary(self.GROUPS)
        self.assertEqual(html_out.count(f"background:{email_reports._ROW_ALT_BG}"), 1)
        self.assertEqual(html_out.count('width="150"'), 1)
        self.assertLess(html_out.index('width="150"'), html_out.index(">Alpha</th>"))

    def test_nothing_to_show_renders_nothing(self):
        self.assertEqual(render_glossary({}), "")
        self.assertEqual(render_glossary({"Empty": []}), "")

    def test_untitled_group_has_no_heading_row(self):
        html_out = render_glossary({"": [{"term": "A", "definition": "b"}]})
        self.assertEqual(html_out.count("<tr"), 1)

    def test_terms_are_inherited_from_the_table_so_every_row_stays_small(self):
        # Every glossary row repeats, and the section sits at the very end of an
        # email Gmail clips at ~102KB, so per-row markup has to stay minimal.
        rows = [{"term": f"T{i}", "definition": "d"} for i in range(40)]
        per_row = len(render_glossary({"G": rows}).encode("utf-8")) / len(rows)
        self.assertLess(per_row, 100)


class ResponsiveDashboardTests(unittest.TestCase):
    def _report(self, **overrides):
        kwargs = {
            "title": "Weekly Scan",
            "subtitle": "Run date 2026-10-02 · ticker pool 10",
            "metrics": [{"label": "Count", "value": "5"}],
            "sections": [render_row_title("Row"), render_panel("Panel", "<p>x</p>")],
        }
        kwargs.update(overrides)
        return render_report_html(**kwargs)

    def test_dashboard_is_wide_on_laptops_and_fluid_below_that(self):
        html_out = self._report()
        self.assertIn(f"max-width:{email_reports._DASHBOARD_MAX_WIDTH_PX}px", html_out)
        self.assertGreaterEqual(email_reports._DASHBOARD_MAX_WIDTH_PX, 1000)
        self.assertIn("width:100%", html_out)

    def test_declares_viewport_and_a_phone_media_query(self):
        html_out = self._report()
        self.assertIn('name="viewport"', html_out)
        self.assertIn("@media only screen and (max-width: 720px)", html_out)

    def test_layout_does_not_depend_on_the_style_block(self):
        # Clients that drop <style> (e.g. non-Google accounts in Gmail) still need
        # the fluid, inline-styled layout.
        import re

        html_out = re.sub(r"<style>.*?</style>", "", self._report(), flags=re.S)
        self.assertIn("max-width:", html_out)
        self.assertIn("display:inline-block", html_out)
        self.assertIn("overflow-x:auto", self._report(sections=[render_panel("T", render_html_table(["Ticker"], []))]))

    def test_styled_sections_pass_through_and_plain_strings_get_a_panel(self):
        styled = self._report(sections=[render_row_title("Only row")])
        plain = self._report(sections=["<div>plain-body</div>"])
        self.assertEqual(styled.count('class="bk-panel-body"'), 0)
        self.assertEqual(plain.count('class="bk-panel-body"'), 1)
        self.assertIn("plain-body", plain)

    def test_subtitle_parts_become_separate_chips_and_stay_escaped(self):
        html_out = self._report(subtitle="A · <i>B</i>")
        self.assertIn(">A</span>", html_out)
        self.assertIn("&lt;i&gt;B&lt;/i&gt;", html_out)
        self.assertNotIn("<i>B</i>", html_out)

    def test_document_is_a_complete_html_page_for_email_clients(self):
        html_out = self._report()
        self.assertTrue(html_out.startswith("<!DOCTYPE html>"))
        self.assertIn('<meta charset="utf-8">', html_out)

    def test_phones_stack_each_glossary_definition_under_its_term(self):
        phone_css = email_reports._RESPONSIVE_CSS.split("@media only screen", 1)[1]
        self.assertIn(".bk-gloss", phone_css)
        self.assertIn("display: block", phone_css.split(".bk-gloss", 1)[1])


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
