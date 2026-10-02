from __future__ import annotations

import unittest
from datetime import date

from scripts import grading_report


def _row(ticker: str, **overrides):
    row = {
        "Ticker": ticker,
        "Company": f"{ticker} Inc",
        "Scan Price": 0.124,
        "Target Price": 0.2,
        "Projected Upside %": 61.29,
        "Target Date": "2026-10-30",
        "Actual @ Target": 0.18,
        "Actual Return %": 45.16,
        "Point Hit": False,
        "Band Hit": True,
        "Max Price Since Scan": 0.21,
        "Max Return %": 69.35,
        "Max Hit": True,
        "Max Band Hit": True,
        "Abs Error %": 16.13,
        "Status": "resolved",
    }
    row.update(overrides)
    return row


class BuildGradingReportTests(unittest.TestCase):
    def _report(self, rows=None, summary=None, score_validation=None, upside_validation=None, horizon="short_term"):
        rows = [_row("AAA"), _row("BBB", **{"Actual Return %": -3.2})] if rows is None else rows
        summary = summary if summary is not None else grading_report.compute_batch_summary(rows)
        return grading_report.build_grading_report(
            horizon,
            date(2026, 9, 4),
            date(2026, 10, 2),
            rows,
            summary,
            score_validation,
            upside_validation,
        )

    def test_renders_the_dashboard_with_batch_highlights_and_graded_table(self):
        html_out = self._report()
        self.assertIn("Short-Term (1–4 weeks) grading report", html_out)
        self.assertIn("Batch highlights", html_out)
        self.assertIn("Graded predictions", html_out)
        self.assertIn("Point hit rate", html_out)
        self.assertIn("Best performer:", html_out)
        self.assertIn("max-width:1200px", html_out)
        self.assertEqual(html_out.count('class="bk-table"'), 1)

    def test_does_not_use_the_legacy_light_text_palette(self):
        html_out = self._report()
        for legacy in ("#f4fff8", "#a9bdd7", "#dce8f7"):
            self.assertNotIn(legacy, html_out)

    def test_sub_dollar_prices_keep_their_extra_decimals(self):
        html_out = self._report()
        self.assertIn("$0.1240", html_out)
        self.assertIn("$0.2000", html_out)

    def test_validation_sections_only_appear_when_there_is_data(self):
        without = self._report()
        self.assertNotIn("Score validation", without)
        self.assertNotIn("Upside forecast validation", without)

        with_stats = self._report(
            score_validation={"overall": {"count": 40, "information_coefficient": 0.123, "precision_at_top_third": 61.0, "precision_at_bottom_third": 38.0}},
            upside_validation={
                "overall": {
                    "count": 40,
                    "information_coefficient": 0.05,
                    "mean_bias_pct": -4.2,
                    "median_bias_pct": -3.1,
                    "overoptimism_rate": 64.0,
                }
            },
        )
        self.assertIn("Score validation (all-time, this horizon)", with_stats)
        self.assertIn("Upside forecast validation (all-time, this horizon)", with_stats)
        self.assertIn("0.123", with_stats)
        self.assertIn("Over-optimism rate", with_stats)
        self.assertIn("resolved Short-Term (1–4 weeks) predictions to date", with_stats)

    def test_escapes_ticker_text_in_the_best_and_worst_performer_line(self):
        rows = [_row("<b>X</b>", **{"Actual Return %": 80.0}), _row("YYY", **{"Actual Return %": -5.0})]
        html_out = self._report(rows=rows)
        self.assertNotIn("<b>X</b>", html_out)
        self.assertIn("&lt;b&gt;X&lt;/b&gt;", html_out)

    def test_handles_an_empty_batch(self):
        html_out = self._report(rows=[], summary=grading_report.compute_batch_summary([]))
        self.assertIn("No rows available.", html_out)
        self.assertNotIn("Best performer:", html_out)


if __name__ == "__main__":
    unittest.main()
