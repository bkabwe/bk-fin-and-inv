from __future__ import annotations

import unittest
from unittest.mock import patch

from modules import profit_opportunities


def _make_analysis(**overrides):
    analysis = {
        "company": "Apple Inc.",
        "score": 78,
        "current_price": 190.0,
        "sector_trend": "bullish",
        "market_cap_tier": "large",
        "longterm_stage": "markup",
        "technical": {"indicators": {"rsi": 55.0}},
        "projections": {
            "current_price": 190.0,
            "short_term_target": 200.0,
            "short_term_low": 195.0,
            "short_term_high": 205.0,
            "short_term_upside": 5.26,
            "short_term_basis": "ARIMA",
            "data_quality": "High",
        },
        "score_breakdown": {
            "trend": 15,
            "momentum": 8,
            "relative_strength": 5,
            "breakout": 3,
            "volume_quality": 2,
        },
    }
    analysis.update(overrides)
    return analysis


class AnalyzeTickerForHorizonSubscoreTests(unittest.TestCase):
    """analyze_ticker_for_horizon's row dict should expose the technical
    sub-score breakdown as internal "_*_score" fields, so
    modules.prediction_tracker can record + correlate them (see
    SUBSCORE_ROW_FIELDS)."""

    def test_row_includes_subscore_fields_from_score_breakdown(self):
        with patch.object(profit_opportunities, "analyze_stock", return_value=_make_analysis()):
            outcome = profit_opportunities.analyze_ticker_for_horizon("AAPL", "short_term", use_fast_screen=False)

        row = outcome["row"]
        self.assertIsNotNone(row)
        self.assertEqual(row["_trend_score"], 15)
        self.assertEqual(row["_momentum_score"], 8)
        self.assertEqual(row["_rs_score"], 5)
        self.assertEqual(row["_breakout_score"], 3)
        self.assertEqual(row["_volume_quality_score"], 2)
        self.assertEqual(row["_rsi"], 55.0)

    def test_row_subscore_fields_are_none_when_score_breakdown_missing(self):
        analysis = _make_analysis()
        analysis.pop("score_breakdown")
        with patch.object(profit_opportunities, "analyze_stock", return_value=analysis):
            outcome = profit_opportunities.analyze_ticker_for_horizon("AAPL", "short_term", use_fast_screen=False)

        row = outcome["row"]
        self.assertIsNotNone(row)
        for field in ("_trend_score", "_momentum_score", "_rs_score", "_breakout_score", "_volume_quality_score"):
            self.assertIsNone(row[field])


class AnalyzeTickerForHorizonLightgbmBacktestedTests(unittest.TestCase):
    """analyze_ticker_for_horizon's row dict should expose the per-horizon
    LightGBM-backtested flag computed by modules.scoring_engine, so callers
    (e.g. the scheduled scan report) can filter on genuinely backtest-
    confirmed LightGBM picks."""

    def test_row_exposes_true_lightgbm_backtested_flag_for_short_term(self):
        analysis = _make_analysis()
        analysis["projections"]["short_term_lightgbm_backtested"] = True
        with patch.object(profit_opportunities, "analyze_stock", return_value=analysis):
            outcome = profit_opportunities.analyze_ticker_for_horizon("AAPL", "short_term", use_fast_screen=False)

        self.assertIs(outcome["row"]["_lightgbm_backtested"], True)

    def test_row_exposes_false_lightgbm_backtested_flag_when_absent(self):
        analysis = _make_analysis()
        with patch.object(profit_opportunities, "analyze_stock", return_value=analysis):
            outcome = profit_opportunities.analyze_ticker_for_horizon("AAPL", "medium_term", use_fast_screen=False)

        self.assertIs(outcome["row"]["_lightgbm_backtested"], False)

    def test_row_lightgbm_backtested_flag_is_none_for_long_term(self):
        # long_term has no LightGBM component at all (see
        # modules.scoring_engine), so HORIZON_SETTINGS deliberately omits a
        # lightgbm_backtested_key for it and the row field stays None rather
        # than filtering out every long-term result.
        analysis = _make_analysis()
        with patch.object(profit_opportunities, "analyze_stock", return_value=analysis):
            outcome = profit_opportunities.analyze_ticker_for_horizon("AAPL", "long_term", use_fast_screen=False)

        self.assertIsNone(outcome["row"]["_lightgbm_backtested"])


class AnalyzeTickerForHorizonConfidenceTests(unittest.TestCase):
    """analyze_ticker_for_horizon's row dict should expose the per-horizon
    projection confidence, thin-history flag, sector, and a risk-adjusted
    upside ranking metric -- see modules.scoring_engine's
    _projection_confidence/_is_thin_history and the Phase 2 action-plan
    items these support."""

    def test_row_exposes_confidence_score_and_thin_history(self):
        analysis = _make_analysis()
        analysis["projections"]["short_term_confidence"] = 62
        analysis["projections"]["short_term_thin_history"] = True
        with patch.object(profit_opportunities, "analyze_stock", return_value=analysis):
            outcome = profit_opportunities.analyze_ticker_for_horizon("AAPL", "short_term", use_fast_screen=False)

        row = outcome["row"]
        self.assertEqual(row["Confidence Score"], 62)
        self.assertIs(row["_thin_history"], True)

    def test_row_defaults_confidence_score_to_zero_when_absent(self):
        analysis = _make_analysis()
        with patch.object(profit_opportunities, "analyze_stock", return_value=analysis):
            outcome = profit_opportunities.analyze_ticker_for_horizon("AAPL", "short_term", use_fast_screen=False)

        self.assertEqual(outcome["row"]["Confidence Score"], 0)
        self.assertIs(outcome["row"]["_thin_history"], False)

    def test_row_exposes_sector_from_fundamentals_metrics(self):
        analysis = _make_analysis(fundamentals={"metrics": {"sector": "Technology"}})
        with patch.object(profit_opportunities, "analyze_stock", return_value=analysis):
            outcome = profit_opportunities.analyze_ticker_for_horizon("AAPL", "short_term", use_fast_screen=False)

        self.assertEqual(outcome["row"]["Sector"], "Technology")

    def test_row_sector_defaults_to_unknown_when_absent(self):
        analysis = _make_analysis()
        with patch.object(profit_opportunities, "analyze_stock", return_value=analysis):
            outcome = profit_opportunities.analyze_ticker_for_horizon("AAPL", "short_term", use_fast_screen=False)

        self.assertEqual(outcome["row"]["Sector"], "Unknown")

    def test_risk_adjusted_upside_is_upside_over_band_width(self):
        # current=190, target_low=195, target_high=205 -> band width =
        # (205-195)/190*100 = 5.263...%; upside=5.26 -> ratio ~= 1.0.
        analysis = _make_analysis()
        with patch.object(profit_opportunities, "analyze_stock", return_value=analysis):
            outcome = profit_opportunities.analyze_ticker_for_horizon("AAPL", "short_term", use_fast_screen=False)

        row = outcome["row"]
        band_width_pct = (row["Target High"] - row["Target Low"]) / row["Current Price"] * 100
        self.assertAlmostEqual(row["Risk-Adjusted Upside"], row["Projected Upside %"] / band_width_pct, places=2)

    def test_risk_adjusted_upside_floors_band_width_to_avoid_divide_by_near_zero(self):
        analysis = _make_analysis()
        # A near-zero band width (target_low == target_high == target) must
        # not blow the ratio up toward infinity.
        analysis["projections"]["short_term_low"] = 200.0
        analysis["projections"]["short_term_high"] = 200.0
        with patch.object(profit_opportunities, "analyze_stock", return_value=analysis):
            outcome = profit_opportunities.analyze_ticker_for_horizon("AAPL", "short_term", use_fast_screen=False)

        row = outcome["row"]
        self.assertAlmostEqual(
            row["Risk-Adjusted Upside"],
            row["Projected Upside %"] / profit_opportunities.DEFAULT_MIN_BAND_WIDTH_PCT,
            places=2,
        )


def _row_for(projections_overrides: dict, *, horizon: str = "short_term", **analysis_overrides):
    analysis = _make_analysis(**analysis_overrides)
    analysis["projections"].update(projections_overrides)
    return profit_opportunities.profit_row_from_analysis("TEST", horizon, analysis)


class RiskAdjustedUpsideBandEdgeTests(unittest.TestCase):
    """Regression tests for the "8.02 / 8.00 then 0.83" Risk-Adjusted Upside
    artifact: a forecast-band edge that was legitimately 0.0 was mistaken for
    "missing" and replaced by a fabricated +/-5% band around the target, which
    is far narrower than the move it must contain and inflated the ratio."""

    def test_zero_lower_edge_is_a_real_edge_not_a_missing_one(self):
        # Price $0.004 -> target $0.012 (+200%, the cap) with a lower edge of
        # exactly $0.00 and an upper edge of $0.020. The honest band is
        # (0.020 - 0.0) / 0.004 = 500% wide, so the ratio is 200 / 500 = 0.4.
        # Treating 0.0 as "missing" gave a ~215% band and a ratio of ~0.93.
        row = _row_for(
            {
                "current_price": 0.004,
                "short_term_target": 0.012,
                "short_term_low": 0.0,
                "short_term_high": 0.020,
                "short_term_upside": 200.0,
            },
            current_price=0.004,
        )

        self.assertEqual(row["Target Low"], 0.0)
        self.assertAlmostEqual(row["Forecast Range %"], 500.0, places=2)
        self.assertAlmostEqual(row["Risk-Adjusted Upside"], 0.4, places=4)

    def test_missing_edges_fall_back_to_band_that_brackets_price_and_target(self):
        # A fabricated +/-5%-of-target band around a +200% target never
        # contained the current price (a 30%-wide band for a 200% move), which
        # yielded a ratio of ~6.7. The fallback band must bracket both.
        analysis = _make_analysis(current_price=100.0)
        analysis["projections"].update(
            {"current_price": 100.0, "short_term_target": 300.0, "short_term_upside": 200.0}
        )
        analysis["projections"].pop("short_term_low")
        analysis["projections"].pop("short_term_high")

        row = profit_opportunities.profit_row_from_analysis("TEST", "short_term", analysis)

        self.assertEqual(row["Target Low"], 100.0)
        self.assertEqual(row["Target High"], 315.0)
        self.assertAlmostEqual(row["Risk-Adjusted Upside"], 200.0 / 215.0, places=4)
        self.assertLessEqual(row["Risk-Adjusted Upside"], 1.0)

    def test_non_finite_edges_are_treated_as_missing(self):
        row = _row_for(
            {
                "current_price": 100.0,
                "short_term_target": 130.0,
                "short_term_low": float("nan"),
                "short_term_high": float("inf"),
                "short_term_upside": 30.0,
            },
            current_price=100.0,
        )

        self.assertEqual(row["Target Low"], 100.0)
        self.assertEqual(row["Target High"], 136.5)
        self.assertLessEqual(row["Risk-Adjusted Upside"], 1.0)

    def test_real_band_containing_price_and_target_never_exceeds_one(self):
        row = _row_for(
            {
                "current_price": 10.0,
                "short_term_target": 12.0,
                "short_term_low": 9.0,
                "short_term_high": 14.0,
                "short_term_upside": 20.0,
            },
            current_price=10.0,
        )

        # band = (14 - 9) / 10 = 50% wide; upside 20% -> 0.4.
        self.assertAlmostEqual(row["Forecast Range %"], 50.0, places=2)
        self.assertAlmostEqual(row["Risk-Adjusted Upside"], 0.4, places=4)

    def test_sub_cent_prices_keep_enough_decimals_in_the_row(self):
        row = _row_for(
            {
                "current_price": 0.003217,
                "short_term_target": 0.006434,
                "short_term_low": 0.002901,
                "short_term_high": 0.007212,
                "short_term_upside": 100.0,
            },
            current_price=0.003217,
        )

        self.assertEqual(row["Current Price"], 0.003217)
        self.assertEqual(row["Target Price"], 0.006434)
        self.assertEqual(row["Target Low"], 0.002901)
        self.assertEqual(row["Target High"], 0.007212)

    def test_normal_prices_keep_four_decimals_in_the_row(self):
        row = _row_for({"short_term_target": 200.123456}, current_price=190.0)

        self.assertEqual(row["Target Price"], 200.1235)


class RiskAdjustedRatingTests(unittest.TestCase):
    def test_buckets_follow_documented_thresholds(self):
        rating = profit_opportunities.risk_adjusted_rating

        self.assertEqual(rating(1.0), "Strong (0.60+)")
        self.assertEqual(rating(0.60), "Strong (0.60+)")
        self.assertEqual(rating(0.5999), "Moderate (0.30 to <0.60)")
        self.assertEqual(rating(0.30), "Moderate (0.30 to <0.60)")
        self.assertEqual(rating(0.2999), "Weak (<0.30)")
        self.assertEqual(rating(0.0), "Weak (<0.30)")
        self.assertEqual(rating(-0.2), "Weak (<0.30)")

    def test_missing_or_non_finite_value_has_no_rating(self):
        self.assertEqual(profit_opportunities.risk_adjusted_rating(None), "")
        self.assertEqual(profit_opportunities.risk_adjusted_rating(float("nan")), "")

    def test_row_exposes_forecast_range_and_self_describing_rating(self):
        # current=190, band=(205-195)/190 = 5.26% wide, upside 5.26% -> ratio 1.0.
        row = _row_for({})

        self.assertAlmostEqual(row["Forecast Range %"], 5.26, places=2)
        self.assertEqual(row["Risk-Adjusted Rating"], "Strong (0.60+)")


if __name__ == "__main__":
    unittest.main()
