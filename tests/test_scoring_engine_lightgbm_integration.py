from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from modules import scoring_engine


def _sample_projection_data(length: int = 260) -> pd.DataFrame:
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


def _sample_feature_table(data: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "technical_feature_a": np.linspace(0.1, 0.9, num=len(data)),
            "macro_feature_b": np.linspace(1.0, 1.5, num=len(data)),
        },
        index=data.index,
    )


class _FakeArimaFit:
    def forecast(self, steps: int) -> pd.Series:
        return pd.Series(np.linspace(110.0, 140.0, num=steps))


class _FakeLinearRegression:
    def fit(self, x, y):
        return self

    def predict(self, future):
        return np.log(np.full(len(future), 120.0))


class ScoringEngineLightGBMIntegrationTests(unittest.TestCase):
    def setUp(self):
        if hasattr(scoring_engine._load_live_lightgbm_models, "clear"):
            scoring_engine._load_live_lightgbm_models.clear()

    def test_short_horizon_weights_include_lightgbm_component(self):
        backtest = {"arima_rmse": 2.0, "trend_rmse": 1.0, "n_windows": 4}

        base_weights = scoring_engine._inverse_rmse_weights(backtest)
        model_weights = scoring_engine._projection_model_weights(30, backtest, include_lightgbm=True)
        projection, basis, _ = scoring_engine._weighted_ensemble(
            [
                ("ARIMA", 100.0, model_weights["arima"]),
                ("Trend", 110.0, model_weights["trend"]),
                ("LightGBM", 120.0, model_weights["lightgbm"]),
                ("Resistance", 130.0, 0.20),
            ]
        )

        self.assertIsNotNone(base_weights)
        self.assertAlmostEqual(sum(base_weights.values()), 1.0, places=6)
        self.assertAlmostEqual(model_weights["lightgbm"], scoring_engine.LIGHTGBM_WEIGHT_30D, places=6)
        self.assertAlmostEqual(sum(model_weights.values()), 0.80, places=6)
        self.assertIn("LightGBM(15%)", basis)
        expected = (
            (100.0 * model_weights["arima"])
            + (110.0 * model_weights["trend"])
            + (120.0 * model_weights["lightgbm"])
            + (130.0 * 0.20)
        )
        self.assertAlmostEqual(float(projection), expected, places=6)

    def test_medium_horizon_weights_include_smaller_lightgbm_component(self):
        backtest = {"arima_rmse": 2.0, "trend_rmse": 1.0, "n_windows": 4}

        model_weights = scoring_engine._projection_model_weights(180, backtest, include_lightgbm=True)
        projection, basis, _ = scoring_engine._weighted_ensemble(
            [
                ("ARIMA", 150.0, model_weights["arima"]),
                ("Trend", 180.0, model_weights["trend"]),
                ("LightGBM", 210.0, model_weights["lightgbm"]),
                ("Fundamental", 200.0, 0.20),
                ("DCF", 205.0, 0.10),
                ("Analyst x0.65", 195.0, 0.15),
            ]
        )

        self.assertLess(scoring_engine.LIGHTGBM_WEIGHT_180D, scoring_engine.LIGHTGBM_WEIGHT_30D)
        self.assertAlmostEqual(model_weights["lightgbm"], scoring_engine.LIGHTGBM_WEIGHT_180D, places=6)
        self.assertAlmostEqual(sum(model_weights.values()), 0.55, places=6)
        self.assertIn("LightGBM(8%)", basis)
        expected = (
            (150.0 * model_weights["arima"])
            + (180.0 * model_weights["trend"])
            + (210.0 * model_weights["lightgbm"])
            + (200.0 * 0.20)
            + (205.0 * 0.10)
            + (195.0 * 0.15)
        )
        self.assertAlmostEqual(float(projection), expected, places=6)

    def test_long_horizon_weights_explicitly_exclude_lightgbm(self):
        weights = scoring_engine._projection_model_weights(
            720,
            {"arima_rmse": 2.0, "trend_rmse": 1.0, "n_windows": 4},
            include_lightgbm=True,
        )
        self.assertNotIn("lightgbm", weights)
        self.assertEqual(set(weights.keys()), {"arima", "trend"})
        self.assertAlmostEqual(sum(weights.values()), 0.50, places=6)

    def test_live_projection_falls_back_when_no_saved_lightgbm_model_exists(self):
        data = _sample_projection_data()
        info = {"exchange": "NASDAQ", "marketCap": 5_000_000_000, "currentPrice": float(data["Close"].iloc[-1])}
        technical = {
            "resistance_levels": [135.0],
            "indicators": {"bb_high": 134.0},
        }

        with (
            patch("modules.scoring_engine.LIGHTGBM_AVAILABLE", True),
            patch("modules.scoring_engine.load_return_models", return_value={}) as load_mock,
            patch("modules.scoring_engine.predict_forward_return") as predict_mock,
            patch("modules.scoring_engine.build_feature_table", return_value=_sample_feature_table(data)),
            patch("modules.scoring_engine.run_walk_forward", return_value={"arima_rmse": 2.0, "trend_rmse": 1.0, "n_windows": 4}),
            patch("modules.scoring_engine._select_arima_order", return_value=(1, 1, 0)),
            patch("modules.scoring_engine.fit_arima_with_hardening", return_value=_FakeArimaFit()),
            patch("modules.scoring_engine.LinearRegression", _FakeLinearRegression),
            patch("modules.scoring_engine._garch_confidence_from_returns", return_value=(None, None)),
            patch("modules.scoring_engine.analyze_technical", return_value=technical),
            patch(
                "modules.scoring_engine.get_macro_regime",
                return_value={"risk_free_rate": 0.045, "bullish_sectors": [], "bearish_sectors": [], "market_regime": "neutral"},
            ),
            patch("modules.scoring_engine.analyze_fundamentals", return_value={"metrics": {}, "fundamental_score": 50}),
        ):
            result = scoring_engine._get_price_projections_core("AAPL", info=info, data=data)

        load_mock.assert_called_once()
        predict_mock.assert_not_called()
        self.assertNotIn("LightGBM", result["short_term_basis"])
        self.assertNotIn("LightGBM", result["medium_term_basis"])
        self.assertIn("ARIMA", result["short_term_basis"])
        self.assertTrue(any(msg.startswith("LightGBM: no saved return models under") for msg in result["models_skipped"]))

    def test_live_projection_uses_only_30d_and_180d_lightgbm_models(self):
        data = _sample_projection_data()
        info = {"exchange": "NASDAQ", "marketCap": 5_000_000_000, "currentPrice": float(data["Close"].iloc[-1])}
        technical = {
            "resistance_levels": [135.0],
            "indicators": {"bb_high": 134.0},
        }
        model_30 = object()
        model_180 = object()
        model_720 = object()

        def _predict(model, _row):
            if model is model_30:
                return 0.10
            if model is model_180:
                return 0.20
            if model is model_720:
                return 0.90
            raise AssertionError("unexpected model")

        with (
            patch("modules.scoring_engine.LIGHTGBM_AVAILABLE", True),
            patch("modules.scoring_engine.load_return_models", return_value={30: model_30, 180: model_180, 720: model_720}) as load_mock,
            patch("modules.scoring_engine.predict_forward_return", side_effect=_predict) as predict_mock,
            patch("modules.scoring_engine.build_feature_table", return_value=_sample_feature_table(data)),
            patch("modules.scoring_engine.run_walk_forward", return_value={"arima_rmse": 2.0, "trend_rmse": 1.0, "n_windows": 4}),
            patch("modules.scoring_engine._select_arima_order", return_value=(1, 1, 0)),
            patch("modules.scoring_engine.fit_arima_with_hardening", return_value=_FakeArimaFit()),
            patch("modules.scoring_engine.LinearRegression", _FakeLinearRegression),
            patch("modules.scoring_engine._garch_confidence_from_returns", return_value=(None, None)),
            patch("modules.scoring_engine.analyze_technical", return_value=technical),
            patch(
                "modules.scoring_engine.get_macro_regime",
                return_value={"risk_free_rate": 0.045, "bullish_sectors": [], "bearish_sectors": [], "market_regime": "neutral"},
            ),
            patch("modules.scoring_engine.analyze_fundamentals", return_value={"metrics": {}, "fundamental_score": 50}),
        ):
            result = scoring_engine._get_price_projections_core("AAPL", info=info, data=data)

        self.assertEqual(load_mock.call_args.kwargs["horizons"], (30, 180))
        self.assertEqual(predict_mock.call_count, 2)
        self.assertIn("LightGBM", result["short_term_basis"])
        self.assertIn("LightGBM", result["medium_term_basis"])
        self.assertNotIn("LightGBM", result["long_term_basis"])


if __name__ == "__main__":
    unittest.main()
