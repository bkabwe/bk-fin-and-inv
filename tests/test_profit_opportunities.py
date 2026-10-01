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


if __name__ == "__main__":
    unittest.main()
