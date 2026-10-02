from __future__ import annotations

import html
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from modules import macro_regime, scoring_engine, sec_edgar_client
from modules.backtester import _compute_walk_forward_starts, get_walk_forward_window_config
from modules.fundamental_analysis import analyze_fundamentals, classify_market_cap_tier
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


GLOSSARY_ROW_TITLE = "&nbsp;Glossary</div>"


def _glossary_html(report_html: str) -> str:
    """Everything from the Glossary row title to the end of the report."""
    return report_html[report_html.index(GLOSSARY_ROW_TITLE):]


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
        self.assertIn("weighted blend of three inputs", html_out)
        self.assertIn("Trend(40%) + LightGBM(40%) + DCF+Comps(20%)", html_out)
        self.assertIn("Trend and LightGBM together share 80% of the weight", html_out)
        self.assertNotIn("Fundamental(", html_out)
        self.assertNotIn("Fixed 30% weight", html_out)
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


class ScanReportGlossaryTests(unittest.TestCase):
    def test_glossary_closes_the_report_after_the_how_to_read_guides(self):
        for horizon, guide in (
            ("short_term", "How the short-term target is built"),
            ("medium_term", "How the medium-term target is built"),
        ):
            with self.subTest(horizon=horizon):
                html_out = _build(horizon)
                self.assertEqual(html_out.count(GLOSSARY_ROW_TITLE), 1)
                glossary_at = html_out.index(GLOSSARY_ROW_TITLE)
                self.assertLess(html_out.index("Risk-adjusted upside: how to read it"), glossary_at)
                self.assertLess(html_out.index(guide), glossary_at)
                # One compact table, not a table per term (and nothing after it).
                self.assertEqual(html_out[glossary_at:].count("<table"), 1)

    def test_defines_the_scores_models_ensembles_and_counts_in_both_reports(self):
        terms = (
            "Score",
            "Fundamentals",
            "Sentiment",
            "Recommendation",
            "Time Horizon",
            "Macro regime",
            "Sector Trend",
            "Market Cap Tier",
            "Long-Term Stage",
            "Ensemble",
            "Trend",
            "LightGBM",
            "Walk-forward backtest",
            "Model weights",
            "Target Price",
            "Projected Upside %",
            "Target Low / High",
            "Forecast Range %",
            "Confidence Score",
            "Newly trained, Carried forward, Dropped stale",
            "Passed fast-screen, Fast-filtered",
            "Dropped: no LightGBM, thin history",
            "Recorded predictions",
            "High-confidence picks",
            "Entry Price, Stop Loss, % from Entry",
        )
        for horizon in ("short_term", "medium_term"):
            glossary = _glossary_html(_build(horizon))
            for term in terms:
                with self.subTest(horizon=horizon, term=term):
                    self.assertIn(f">{term}</th>", glossary)

    def test_short_term_glossary_explains_resistance_but_not_the_medium_term_components(self):
        glossary = _glossary_html(_build("short_term"))
        self.assertIn(">Resistance</th>", glossary)
        self.assertIn("Fixed 20% weight", glossary)
        self.assertNotIn(">Fundamental</th>", glossary)
        self.assertNotIn(">DCF+Comps</th>", glossary)
        self.assertNotIn("Full / Limited", glossary)
        self.assertNotIn(">Confidence</th>", glossary)

    def test_medium_term_glossary_explains_dcf_comps_but_not_resistance_or_a_fundamental_component(self):
        glossary = _glossary_html(_build("medium_term"))
        self.assertIn(">DCF+Comps</th>", glossary)
        self.assertNotIn(">Fundamental</th>", glossary)
        self.assertIn(">Fundamentals</th>", glossary)  # the 0-100 grade inside the Score is a different thing
        self.assertNotIn(">Resistance</th>", glossary)
        self.assertIn("Not the Full / Limited label above", glossary)
        self.assertIn("Profit lists: Confidence = Full", glossary)

    def test_no_longer_presents_eps_times_pe_as_a_fundamental_component(self):
        # The SEC feed has no forward EPS/P/E and derives trailing P/E from the latest close, so "EPS x P/E"
        # is just today's price. It used to sit in the medium/long-term ensembles as a hidden "no change"
        # anchor labelled "Fundamental"; neither the ensemble nor the guides carry it any more. If a real
        # valuation source is ever wired in, bring the component (and its guide entries) back deliberately.
        facts = {"facts": {"us-gaap": {"EarningsPerShareDiluted": {"units": {"USD/shares": [{"end": "2024-12-31", "val": 5, "form": "10-K"}]}}}}}
        info = sec_edgar_client.build_fundamentals_info_adapter(facts, current_price=100)
        self.assertIsNone(info["forwardEps"])
        self.assertIsNone(info["forwardPE"])
        self.assertAlmostEqual(info["trailingEps"] * info["trailingPE"], 100.0)

        self.assertFalse(hasattr(scoring_engine, "MEDIUM_TERM_FUNDAMENTAL_WEIGHT"))
        self.assertFalse(hasattr(scoring_engine, "LONG_TERM_FUNDAMENTAL_WEIGHT"))
        for horizon in ("short_term", "medium_term"):
            with self.subTest(horizon=horizon):
                html_out = html.unescape(_build(horizon))
                self.assertNotIn(">Fundamental</th>", html_out)
                self.assertNotIn("Fundamental(", html_out)
                self.assertNotIn("EPS × P/E", html_out)
                self.assertNotIn("“no change” anchor", html_out)

    def test_describes_the_garch_band_as_volatility_summed_over_the_horizon(self):
        for horizon, days in (("short_term", 30), ("medium_term", 180)):
            glossary = html.unescape(_glossary_html(_build(horizon)))
            with self.subTest(horizon=horizon):
                self.assertIn(f"summed over the {days} days that gives the horizon's σ", glossary)
                self.assertIn("e^(±1.96σ)", glossary)
                self.assertNotIn("daily volatility σ", glossary)

    def test_only_explains_counts_that_the_report_actually_shows(self):
        for horizon in ("short_term", "medium_term"):
            html_out = _build(horizon)
            report = html_out[: html_out.index(GLOSSARY_ROW_TITLE)]
            for label in (
                "Newly trained",
                "Carried forward",
                "Dropped stale",
                "Passed fast-screen",
                "Fast-filtered",
                "Recorded predictions",
                "Dropped: no LightGBM",
                "Dropped: thin history",
                "High-confidence picks",
            ):
                with self.subTest(horizon=horizon, label=label):
                    self.assertIn(label, report)

    def test_renders_even_when_every_result_set_is_empty(self):
        for horizon in ("short_term", "medium_term"):
            with self.subTest(horizon=horizon):
                self.assertIn(">Ensemble</th>", _glossary_html(_build(horizon, empty=True)))


class GlossaryFiguresStayInSyncTests(unittest.TestCase):
    """The glossary restates figures that live in the scoring code. Like the
    email's bar-color thresholds (see BarColorTests), they are duplicated on
    purpose -- the scoring modules must not depend on the email module -- so
    these tests fail when the two drift apart."""

    def test_score_point_budgets_match_the_composite_score(self):
        for horizon in ("short_term", "medium_term"):
            fundamental, sentiment = scoring_engine._fundamental_sentiment_weights(horizon)
            glossary = _glossary_html(_build(horizon))
            with self.subTest(horizon=horizon):
                self.assertIn(f"fundamentals up to {fundamental}, news sentiment up to {sentiment}", glossary)
                self.assertIn(f"scaled to {fundamental} points", glossary)
                self.assertIn(f"becomes 0–{sentiment} points. Neutral ({sentiment // 2})", glossary)

    def test_score_adjustments_match_the_scoring_code(self):
        glossary = _glossary_html(_build("short_term"))
        micro = scoring_engine._MARKET_CAP_RISK_SCORES["Micro Cap"]
        small = scoring_engine._MARKET_CAP_RISK_SCORES["Small Cap"]
        self.assertIn(f"sector trend (±{scoring_engine._SECTOR_MOMENTUM_SCORE})", glossary)
        self.assertIn(f"(−{abs(small)} Small, −{abs(micro)} Micro)", glossary)

    def test_recommendation_tiers_match_the_scoring_code(self):
        glossary = _glossary_html(_build("short_term"))
        self.assertIn(
            "80+ Strong Buy, 65–79 Buy, 50–64 Take Small Position, 35–49 Monitor, 20–34 Do Not Buy, under 20 Avoid",
            glossary,
        )
        for score, label in [
            (80, "STRONG BUY"),
            (79, "BUY"),
            (65, "BUY"),
            (64, "TAKE SMALL POSITION"),
            (50, "TAKE SMALL POSITION"),
            (49, "MONITOR"),
            (35, "MONITOR"),
            (34, "DO NOT BUY"),
            (20, "DO NOT BUY"),
            (19, "AVOID"),
        ]:
            with self.subTest(score=score):
                self.assertEqual(scoring_engine._recommendation(score).split(" ", 1)[1], label)

    def test_market_cap_tiers_match_the_fundamentals_code(self):
        glossary = _glossary_html(_build("short_term"))
        self.assertIn("Micro under $300M, Small to $2B, Mid to $10B, Large to $200B, Mega above.", glossary)
        for market_cap, tier in [
            (299_999_999, "Micro Cap"),
            (300_000_000, "Small Cap"),
            (1_999_999_999, "Small Cap"),
            (2_000_000_000, "Mid Cap"),
            (9_999_999_999, "Mid Cap"),
            (10_000_000_000, "Large Cap"),
            (199_999_999_999, "Large Cap"),
            (200_000_000_000, "Mega Cap"),
        ]:
            with self.subTest(market_cap=market_cap):
                self.assertEqual(classify_market_cap_tier(market_cap), tier)

    def test_model_weight_split_and_lightgbm_cap_match_the_ensemble_code(self):
        # LightGBM dominates the inverse-RMSE split, so its share is whatever the cap allows.
        backtest = {"n_windows": 6, "trend_rmse": 10.0, "lightgbm_rmse": 0.1}
        for horizon, horizon_days, budget, cap_text in (
            ("short_term", 30, 0.80, "70% (six or more)"),
            ("medium_term", 180, 0.80, "66% (five, the most the 180-day backtest can fit)"),
        ):
            glossary = _glossary_html(_build(horizon))
            window = get_walk_forward_window_config(horizon_days)
            # Only counts the backtester can actually produce: the 180d one tops out at 5 windows, so never 70%.
            max_windows = len(_compute_walk_forward_starts(window.max_history_rows, window))
            max_cap = scoring_engine._lightgbm_adaptive_share_cap(max_windows)
            with self.subTest(horizon=horizon):
                self.assertIn(f"share {round(budget * 100)}% of the weight", glossary)
                self.assertIn(f"capped at 50% (one backtest window), rising to {cap_text}", glossary)
                self.assertTrue(cap_text.startswith(f"{round(max_cap * 100)}%"))
                for windows, cap in ((1, 0.50), (max_windows, max_cap)):
                    weights = scoring_engine._projection_model_weights(
                        horizon_days, {**backtest, "lightgbm_windows": windows}, include_lightgbm=True
                    )
                    self.assertAlmostEqual(sum(weights.values()), budget)
                    self.assertAlmostEqual(weights["lightgbm"] / budget, cap)

    def test_fixed_medium_term_weights_match_the_ensemble_code(self):
        glossary = _glossary_html(_build("medium_term"))
        self.assertIn(f"Fixed {round(scoring_engine.MEDIUM_TERM_DCF_WEIGHT * 100)}% weight", glossary)

    def test_garch_band_matches_the_scoring_code(self):
        # A flat 4 (%^2) daily variance (2% daily volatility) accumulates to sqrt(4 x days) % over the horizon.
        forecast = SimpleNamespace(variance=pd.DataFrame([[4.0] * 720]))
        for horizon, days in (("short_term", 30), ("medium_term", 180)):
            with self.subTest(horizon=horizon), patch.object(scoring_engine, "ARCH_AVAILABLE", True):
                self.assertIn(f"summed over the {days} days", _glossary_html(_build(horizon)))
                low, high = scoring_engine._garch_confidence_from_returns(100.0, np.zeros(100), days, forecast=forecast)
                sigma = math.sqrt(4.0 * days) / 100.0
                self.assertAlmostEqual(low, 100.0 * math.exp(-1.96 * sigma))
                self.assertAlmostEqual(high, 100.0 * math.exp(1.96 * sigma))

    def test_walk_forward_windows_match_the_backtester(self):
        for horizon, horizon_days in (("short_term", 30), ("medium_term", 180)):
            window = get_walk_forward_window_config(horizon_days)
            with self.subTest(horizon=horizon):
                self.assertIn(
                    f"({window.train_len} days to train, {window.test_len} to test)", _glossary_html(_build(horizon))
                )

    def test_dcf_and_comparables_formula_matches_the_fundamentals_code(self):
        glossary = _glossary_html(_build("medium_term"))
        self.assertIn("EPS × (8.5 + 2 × growth %) × 4.4 ÷ the 10-year yield %", glossary)
        # EPS 5, 10% growth, 8.8% yield: 5 x (8.5 + 2 x 10) x (4.4 / 8.8) = 71.25.
        metrics = analyze_fundamentals(
            {"trailingEps": 5.0, "earningsGrowth": 0.10, "sector": "Technology"},
            current_price=100.0,
            risk_free_rate=0.088,
        )["metrics"]
        self.assertAlmostEqual(metrics["dcf_estimate"], 71.25, places=2)
        self.assertAlmostEqual(metrics["comparable_value_estimate"], 5.0 * metrics["sector_benchmark_pe"], delta=0.05)
        self.assertAlmostEqual(
            metrics["valuation_estimate"], (metrics["dcf_estimate"] + metrics["comparable_value_estimate"]) / 2, delta=0.05
        )

    def test_confidence_figures_match_the_scoring_code(self):
        glossary = _glossary_html(_build("medium_term"))
        spread = round(scoring_engine._CONFIDENCE_AGREEMENT_CV_SCALE * 100)
        self.assertIn(f"standard deviation reaches {spread}% of the blended forecast", glossary)
        self.assertEqual(scoring_engine._ensemble_agreement_score([100.0, 100.0], 100.0), 1.0)
        # Components whose standard deviation is exactly 20% of the forecast are fully distrusted.
        self.assertAlmostEqual(scoring_engine._ensemble_agreement_score([80.0, 120.0], 100.0), 1.0 - 0.20 / 0.20)
        self.assertIn(f"LightGBM backtest windows out of {scoring_engine.LIGHTGBM_ADAPTIVE_CAP_FULL_EVIDENCE_WINDOWS}", glossary)
        # Half of the raw move survives even at zero confidence.
        self.assertAlmostEqual(scoring_engine._apply_confidence_shrinkage(110.0, 100.0, 0), 105.0)
        self.assertIn("half the move is kept at zero confidence", glossary)

    def test_target_range_padding_matches_the_scoring_code(self):
        glossary = _glossary_html(_build("short_term"))
        micro = round(scoring_engine._MARKET_CAP_CONFIDENCE_PADDING["Micro Cap"] * 100)
        small = round(scoring_engine._MARKET_CAP_CONFIDENCE_PADDING["Small Cap"] * 100)
        self.assertIn(f"widened by {micro}% (Micro) or {small}% (Small) of the price each side", glossary)

    def test_macro_regime_thresholds_match_the_macro_regime_code(self):
        glossary = _glossary_html(_build("short_term"))
        self.assertIn("VIX under 15 = risk-on, over 25 = risk-off", glossary)
        self.assertIn("4+ of SPY and five sector ETFs", glossary)
        for vix, bullish_etfs, expected in [
            (14.99, 0, "risk_on"),
            (25.01, 6, "risk_off"),
            (15.0, 3, "neutral"),
            (25.0, 3, "neutral"),
            (20.0, 4, "risk_on"),
            (20.0, 3, "neutral"),
            (20.0, 2, "risk_off"),
        ]:
            with self.subTest(vix=vix, bullish_etfs=bullish_etfs):
                self.assertEqual(self._market_regime(vix, bullish_etfs), expected)

    @staticmethod
    def _market_regime(vix: float, bullish_etfs: int) -> str:
        """get_macro_regime() with the given VIX and this many of its six ETFs uptrending."""
        etfs = ["SPY", "XLK", "XLF", "XLE", "XLV", "XLI"]
        uptrending = set(etfs[:bullish_etfs])

        def _prices(symbol, **kwargs):
            closes = np.linspace(100.0, 200.0, 120) if symbol in uptrending else np.linspace(200.0, 100.0, 120)
            return pd.DataFrame({"Close": closes})

        def _clear_cache():
            clear = getattr(macro_regime.get_macro_regime, "clear", None)
            if callable(clear):
                clear()

        _clear_cache()
        try:
            with (
                patch("modules.macro_regime.get_vix_observations", return_value=pd.DataFrame({"Close": [vix]})),
                patch("modules.macro_regime.get_series_observations", return_value=pd.DataFrame({"value": [4.0]})),
                patch("modules.macro_regime.get_stock_data_polygon", side_effect=_prices),
            ):
                return macro_regime.get_macro_regime()["market_regime"]
        finally:
            _clear_cache()


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
