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


if __name__ == "__main__":
    unittest.main()
