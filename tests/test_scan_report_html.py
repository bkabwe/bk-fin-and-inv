from __future__ import annotations

import unittest

import pandas as pd

from scripts import scan_email_report as ser

# Gmail clips a message's HTML at ~102KB ("[Message clipped]"), which would hide
# the explanations at the bottom of the report.
GMAIL_CLIP_LIMIT_BYTES = 102_000


def _rating(rau: float) -> str:
    if rau >= 0.60:
        return "Strong (0.60+)"
    if rau >= 0.30:
        return "Moderate (0.30 to <0.60)"
    return "Weak (<0.30)"


def _profit_frame(*, medium_term: bool, rows: int = 75, company_length: int = 24) -> pd.DataFrame:
    records = []
    for i in range(rows):
        upside = 200.0 - i
        rau = round(max(0.05, 1.0 - i / 100), 4)
        price = 0.2083 if i == 0 else 12.5 + i
        record = {
            "Ticker": f"TK{i:03d}",
            "Company": (f"Company {i} " + "X" * company_length)[:company_length],
            "Score": 90 - (i % 40),
            "Current Price": price,
            "Target Price": round(price * (1 + upside / 100), 4),
            "Target Low": round(price * 0.9, 4),
            "Target High": round(price * (1 + upside / 100) * 1.05, 4),
            "Projected Upside %": upside,
            "Risk-Adjusted Upside": rau,
            "Forecast Range %": round(upside / rau, 2),
            "Risk-Adjusted Rating": _rating(rau),
            "Sector": "Technology",
            "Basis": "Ensemble: Trend(37%) + LightGBM(43%) + Resistance(20%)",
        }
        if medium_term:
            record["Confidence"] = "Full" if i % 3 == 0 else "Limited"
        records.append(record)
    return pd.DataFrame(records)


def _screener_frame(rows: int = 75, company_length: int = 24) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "Ticker": f"SC{i:03d}",
                "Company": (f"Screener Co {i} " + "Y" * company_length)[:company_length],
                "Score": 92 - (i % 40),
                "Recommendation": "\U0001f7e2 STRONG BUY" if i % 2 else "\U0001f535 BUY",
                "Current Price": 10.0 + i,
                "Target Price": 15.0 + i,
            }
            for i in range(rows)
        ]
    )
    frame.attrs.update({"fully_analyzed_count": 412, "fast_filtered_count": 3321})
    return frame


def _build(horizon: str, *, rows: int = 75, company_length: int = 24, empty: bool = False) -> str:
    medium_term = horizon == "medium_term"
    if empty:
        screener = pd.DataFrame(columns=["Ticker", "Company", "Score", "Recommendation", "Current Price", "Target Price"])
        profit_empty = _profit_frame(medium_term=medium_term, rows=0)
        by_upside = by_score = by_rau = profit_empty
    else:
        screener = _screener_frame(rows, company_length)
        by_upside = _profit_frame(medium_term=medium_term, rows=rows, company_length=company_length)
        by_score = by_upside.sort_values("Score", ascending=False).reset_index(drop=True)
        by_rau = by_upside.sort_values("Risk-Adjusted Upside", ascending=False).reset_index(drop=True)
    manifest = {"current_run": {"trained_tickers": ["A"] * 12, "carried_forward_tickers": ["B"] * 30, "dropped_stale_tickers": ["C"]}}
    stats = {
        "passed_fast_screen_count": 412,
        "recorded_count": 75,
        "lightgbm_unconfirmed_count": 31,
        "thin_history_dropped_count": 18,
    }
    return ser.build_scan_report(manifest, ["A"] * 42, screener, by_upside, by_score, by_rau, stats, horizon, "2026-10-02")


class ScanReportContentTests(unittest.TestCase):
    def test_explains_risk_adjusted_upside_range_and_what_high_and_low_mean(self):
        html_out = _build("short_term")
        self.assertIn("Risk-adjusted upside: how to read it", html_out)
        self.assertIn("Projected Upside % ÷ Forecast Range %", html_out)
        self.assertIn("Possible range: 0 to 1", html_out)
        self.assertIn("can never exceed 1.00", html_out)
        for tier in ("Strong (0.60+)", "Moderate (0.30 to &lt;0.60)", "Weak (&lt;0.30)"):
            self.assertIn(tier, html_out)

    def test_explains_why_short_term_resistance_is_always_twenty_percent(self):
        html_out = _build("short_term")
        self.assertIn("How the short-term target is built", html_out)
        self.assertIn("Why Resistance is always 20%", html_out)
        self.assertIn("fixed 20%", html_out)

    def test_medium_term_report_describes_its_own_weights_and_adds_the_confidence_guide(self):
        html_out = _build("medium_term")
        self.assertIn("How the medium-term target is built", html_out)
        self.assertNotIn("Why Resistance is always 20%", html_out)
        self.assertIn("Fixed 30% weight", html_out)
        self.assertIn("Fixed 20% weight", html_out)
        self.assertIn("Confidence: how to read it", html_out)
        self.assertIn("Technical Only", html_out)

    def test_short_term_report_omits_the_confidence_column_and_guide(self):
        html_out = _build("short_term")
        self.assertNotIn("Confidence: how to read it", html_out)
        self.assertNotIn(">Confidence<", html_out)

    def test_only_the_risk_adjusted_ranking_shows_its_components_and_tier(self):
        html_out = _build("short_term")
        self.assertEqual(html_out.count("Forecast Range %<div"), 1)
        self.assertEqual(html_out.count("Risk-Adjusted Rating</th>"), 1)
        # ...while all three profit rankings show the number itself.
        self.assertEqual(html_out.count("Risk-Adjusted Upside<div"), 3)

    def test_renders_one_screener_and_three_profit_tables_of_the_top_ten(self):
        html_out = _build("short_term")
        self.assertEqual(html_out.count('class="bk-table"'), 4)
        self.assertEqual(html_out.count("TK010"), 0)  # only the top PREVIEW_LIMIT rows are shown
        self.assertIn("TK000", html_out)
        self.assertEqual(html_out.count("Top 10 shown; the full top 75 is attached as CSV."), 4)

    def test_uses_the_dashboard_layout_not_the_legacy_light_text_palette(self):
        html_out = _build("short_term")
        self.assertNotIn("#f4fff8", html_out)
        self.assertNotIn("#a9bdd7", html_out)
        self.assertIn("max-width:1200px", html_out)
        self.assertIn("@media only screen", html_out)

    def test_sub_dollar_prices_keep_their_extra_decimals(self):
        html_out = _build("short_term")
        self.assertIn("$0.2083", html_out)

    def test_empty_results_still_render_every_section_without_errors(self):
        for horizon in ("short_term", "medium_term"):
            with self.subTest(horizon=horizon):
                html_out = _build(horizon, empty=True)
                self.assertIn("No rows available.", html_out)
                self.assertIn("Risk-adjusted upside: how to read it", html_out)

    def test_keeps_the_original_stat_tile_figures(self):
        html_out = _build("short_term")
        for expected in ("Tickers scanned", "Newly trained", "Carried forward", "Dropped stale", "Avg / median score", "Avg / median upside"):
            self.assertIn(expected, html_out)


class ScanReportSizeTests(unittest.TestCase):
    def test_stays_under_the_gmail_clipping_limit_even_with_long_company_names(self):
        for horizon in ("short_term", "medium_term"):
            with self.subTest(horizon=horizon):
                size = len(_build(horizon, company_length=60).encode("utf-8"))
                self.assertLess(size, GMAIL_CLIP_LIMIT_BYTES, f"{horizon} report is {size} bytes")


class PreviewRowFormattingTests(unittest.TestCase):
    def test_formats_numbers_and_shortens_the_rating_for_the_email_badge(self):
        frame = pd.DataFrame(
            [
                {
                    "Ticker": "AAA",
                    "Score": 81.234,
                    "Current Price": 0.12345,
                    "Target Price": 190.5,
                    "Projected Upside %": 12.3456,
                    "Forecast Range %": 20.0,
                    "Risk-Adjusted Upside": 0.61765,
                    "Risk-Adjusted Rating": "Strong (0.60+)",
                }
            ]
        )
        row = ser._preview_rows(frame, list(frame.columns))[0]
        self.assertEqual(row["Score"], "81.23")
        self.assertEqual(row["Current Price"], "$0.1235")
        self.assertEqual(row["Target Price"], "$190.50")
        self.assertEqual(row["Projected Upside %"], "12.35%")
        self.assertEqual(row["Forecast Range %"], "20.00%")
        self.assertEqual(row["Risk-Adjusted Upside"], "0.62")
        self.assertEqual(row["Risk-Adjusted Rating"], "Strong")

    def test_missing_values_render_as_dashes(self):
        frame = pd.DataFrame([{"Ticker": "AAA", "Current Price": None, "Risk-Adjusted Rating": None}])
        row = ser._preview_rows(frame, list(frame.columns))[0]
        self.assertEqual(row["Current Price"], "—")
        self.assertEqual(row["Risk-Adjusted Upside"], "—")
        self.assertIsNone(row["Risk-Adjusted Rating"])


if __name__ == "__main__":
    unittest.main()
