from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from modules import scoring_engine


def _sample_projection_data(length: int) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=length)
    close = np.linspace(100.0, 130.0, num=length)
    return pd.DataFrame(
        {
            "Close": close,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Volume": np.full(length, 1_000.0),
        },
        index=index,
    )


class _FakeLinearRegression:
    def fit(self, x, y):
        return self

    def predict(self, future):
        return np.full(len(future), 4.7)


class RequiredHistoryDaysTests(unittest.TestCase):
    def test_matches_backtester_window_configs(self):
        # 60 train + 30 test, 360 train + 180 test, 450 train + 720 test --
        # see modules.backtester._WALK_FORWARD_WINDOW_CONFIGS. Projection
        # confidence intentionally reuses these thresholds rather than
        # inventing separate ones, so "enough evidence" means the same thing
        # in both places.
        self.assertEqual(scoring_engine._required_history_days(30), 90)
        self.assertEqual(scoring_engine._required_history_days(180), 540)
        self.assertEqual(scoring_engine._required_history_days(720), 1170)

    def test_falls_back_to_horizon_length_for_unsupported_horizon(self):
        self.assertEqual(scoring_engine._required_history_days(45), 45)


class IsThinHistoryTests(unittest.TestCase):
    def test_true_when_history_short_of_required_window(self):
        self.assertTrue(scoring_engine._is_thin_history(30, 50))

    def test_false_when_history_meets_required_window(self):
        self.assertFalse(scoring_engine._is_thin_history(30, 90))
        self.assertFalse(scoring_engine._is_thin_history(30, 200))


class EnsembleAgreementScoreTests(unittest.TestCase):
    def test_tight_agreement_scores_near_one(self):
        score = scoring_engine._ensemble_agreement_score([100.0, 101.0, 99.5], 100.0)
        self.assertGreater(score, 0.9)

    def test_wide_disagreement_scores_near_zero(self):
        score = scoring_engine._ensemble_agreement_score([50.0, 150.0, 100.0], 100.0)
        self.assertEqual(score, 0.0)

    def test_fewer_than_two_values_returns_neutral_default(self):
        self.assertEqual(
            scoring_engine._ensemble_agreement_score([100.0], 100.0),
            scoring_engine._CONFIDENCE_SINGLE_COMPONENT_AGREEMENT,
        )
        self.assertEqual(
            scoring_engine._ensemble_agreement_score([], 100.0),
            scoring_engine._CONFIDENCE_SINGLE_COMPONENT_AGREEMENT,
        )

    def test_none_projection_returns_neutral_default(self):
        self.assertEqual(
            scoring_engine._ensemble_agreement_score([100.0, 101.0], None),
            scoring_engine._CONFIDENCE_SINGLE_COMPONENT_AGREEMENT,
        )


class EvidenceDepthScoreTests(unittest.TestCase):
    def test_full_history_and_windows_scores_one(self):
        score = scoring_engine._evidence_depth_score(30, history_days=90, lightgbm_windows=6, includes_lightgbm=True)
        self.assertAlmostEqual(score, 1.0, places=6)

    def test_zero_history_and_windows_scores_zero(self):
        score = scoring_engine._evidence_depth_score(30, history_days=0, lightgbm_windows=0, includes_lightgbm=True)
        self.assertEqual(score, 0.0)

    def test_excludes_window_ratio_when_lightgbm_not_included(self):
        # Long-term (720d) never includes LightGBM, so evidence depth there
        # is purely a function of price-history depth.
        score = scoring_engine._evidence_depth_score(720, history_days=1170, lightgbm_windows=None, includes_lightgbm=False)
        self.assertAlmostEqual(score, 1.0, places=6)

    def test_partial_history_scores_between_zero_and_one(self):
        score = scoring_engine._evidence_depth_score(30, history_days=45, lightgbm_windows=0, includes_lightgbm=True)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)


class ProjectionConfidenceTests(unittest.TestCase):
    def test_none_projection_returns_zero(self):
        self.assertEqual(
            scoring_engine._projection_confidence(
                [], None, horizon_days=30, history_days=90, lightgbm_windows=6, includes_lightgbm=True
            ),
            0,
        )

    def test_strong_agreement_and_evidence_yields_high_confidence(self):
        confidence = scoring_engine._projection_confidence(
            [100.0, 101.0, 99.5],
            100.0,
            horizon_days=30,
            history_days=200,
            lightgbm_windows=6,
            includes_lightgbm=True,
        )
        self.assertGreaterEqual(confidence, 80)

    def test_thin_history_and_disagreement_yields_low_confidence(self):
        confidence = scoring_engine._projection_confidence(
            [60.0, 140.0],
            100.0,
            horizon_days=30,
            history_days=10,
            lightgbm_windows=0,
            includes_lightgbm=True,
        )
        self.assertLess(confidence, 30)

    def test_confidence_is_bounded_0_to_100(self):
        confidence = scoring_engine._projection_confidence(
            [100.0, 100.0],
            100.0,
            horizon_days=30,
            history_days=10_000,
            lightgbm_windows=100,
            includes_lightgbm=True,
        )
        self.assertLessEqual(confidence, 100)
        self.assertGreaterEqual(confidence, 0)


class ApplyConfidenceShrinkageTests(unittest.TestCase):
    def test_full_confidence_passes_through_unchanged(self):
        shrunk = scoring_engine._apply_confidence_shrinkage(150.0, 100.0, 100)
        self.assertAlmostEqual(shrunk, 150.0, places=6)

    def test_zero_confidence_retains_minimum_fraction_of_move(self):
        shrunk = scoring_engine._apply_confidence_shrinkage(150.0, 100.0, 0)
        expected = 100.0 + (150.0 - 100.0) * scoring_engine._SHRINKAGE_MIN_RETAINED_MOVE
        self.assertAlmostEqual(shrunk, expected, places=6)

    def test_never_fully_erases_direction_even_at_zero_confidence(self):
        shrunk = scoring_engine._apply_confidence_shrinkage(200.0, 100.0, 0)
        self.assertGreater(shrunk, 100.0)

    def test_partial_confidence_is_between_full_and_minimum_retention(self):
        full = scoring_engine._apply_confidence_shrinkage(150.0, 100.0, 100)
        zero = scoring_engine._apply_confidence_shrinkage(150.0, 100.0, 0)
        partial = scoring_engine._apply_confidence_shrinkage(150.0, 100.0, 50)
        self.assertLess(zero, partial)
        self.assertLess(partial, full)


class PriceProjectionsConfidenceIntegrationTests(unittest.TestCase):
    """End-to-end coverage of the new confidence/shrinkage/thin-history
    fields through `_get_price_projections_core`, exercising them together
    rather than only via their individual helper functions above."""

    def _run(self, data: pd.DataFrame) -> dict:
        info = {"exchange": "NASDAQ", "marketCap": 5_000_000_000, "currentPrice": float(data["Close"].iloc[-1])}
        technical = {"resistance_levels": [135.0], "indicators": {"bb_high": 134.0}}
        with (
            patch("modules.scoring_engine.LIGHTGBM_AVAILABLE", False),
            patch("modules.scoring_engine.load_return_models", return_value={}),
            patch(
                "modules.scoring_engine.run_walk_forward",
                return_value={
                    "arima_rmse": 2.0,
                    "trend_rmse": 1.0,
                    "lightgbm_rmse": None,
                    "n_windows": 0,
                    "arima_windows": 0,
                    "trend_windows": 0,
                    "lightgbm_windows": 0,
                },
            ),
            patch("modules.scoring_engine.LinearRegression", _FakeLinearRegression),
            patch("modules.scoring_engine._garch_confidence_from_returns", return_value=(None, None)),
            patch("modules.scoring_engine.analyze_technical", return_value=technical),
            patch(
                "modules.scoring_engine.get_macro_regime",
                return_value={"risk_free_rate": 0.045, "bullish_sectors": [], "bearish_sectors": [], "market_regime": "neutral"},
            ),
            patch("modules.scoring_engine.analyze_fundamentals", return_value={"metrics": {}, "fundamental_score": 50}),
        ):
            return scoring_engine._get_price_projections_core("AAPL", info=info, data=data)

    def test_ample_history_short_term_is_not_thin_and_has_nonzero_confidence(self):
        result = self._run(_sample_projection_data(260))
        self.assertFalse(result["short_term_thin_history"])
        self.assertGreater(result["short_term_confidence"], 0)
        self.assertIsInstance(result["short_term_confidence"], int)

    def test_thin_history_ticker_is_flagged_and_has_lower_confidence_than_ample_history(self):
        # 50 rows is short of the 30d horizon's 90-row (60 train + 30 test)
        # walk-forward requirement -- e.g. a recently-listed ticker. Model
        # agreement alone can still be high with a small, clean ensemble, so
        # assert the evidence-depth penalty relative to the ample-history
        # case rather than an absolute confidence threshold.
        thin_result = self._run(_sample_projection_data(50))
        ample_result = self._run(_sample_projection_data(260))
        self.assertTrue(thin_result["short_term_thin_history"])
        self.assertLess(thin_result["short_term_confidence"], ample_result["short_term_confidence"])

    def test_medium_and_long_term_thin_history_for_short_history_ticker(self):
        # 260 rows clears the 30d requirement (90 rows) but falls short of
        # both the 180d (540 rows) and 720d (1170 rows) requirements.
        result = self._run(_sample_projection_data(260))
        self.assertTrue(result["medium_term_thin_history"])
        self.assertTrue(result["long_term_thin_history"])

    def test_no_price_data_result_includes_zeroed_confidence_fields(self):
        result = scoring_engine._get_price_projections_core(
            "AAPL", info={"currentPrice": 0}, data=pd.DataFrame({"Close": []})
        )
        self.assertEqual(result["short_term_confidence"], 0)
        self.assertEqual(result["medium_term_confidence"], 0)
        self.assertEqual(result["long_term_confidence"], 0)
        self.assertTrue(result["short_term_thin_history"])


if __name__ == "__main__":
    unittest.main()
