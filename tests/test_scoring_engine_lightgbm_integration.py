from __future__ import annotations

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from modules import scoring_engine


def _sample_projection_data(length: int = 260, start: float = 100.0, end: float = 130.0) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=length)
    close = np.linspace(start, end, num=length)
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


class _FakeLinearRegression:
    def fit(self, x, y):
        return self

    def predict(self, future):
        return np.log(np.full(len(future), 120.0))


class ScoringEngineLightGBMIntegrationTests(unittest.TestCase):
    def setUp(self):
        if hasattr(scoring_engine._load_live_lightgbm_models, "clear"):
            scoring_engine._load_live_lightgbm_models.clear()
        if hasattr(scoring_engine._load_live_lightgbm_manifest, "clear"):
            scoring_engine._load_live_lightgbm_manifest.clear()
        if hasattr(scoring_engine._load_live_lightgbm_batch, "clear"):
            scoring_engine._load_live_lightgbm_batch.clear()
        if hasattr(scoring_engine._load_live_lightgbm_shard_batch, "clear"):
            scoring_engine._load_live_lightgbm_shard_batch.clear()

    def test_short_horizon_weights_include_lightgbm_component(self):
        backtest = {"arima_rmse": 2.0, "trend_rmse": 1.0, "n_windows": 4}

        base_weights = scoring_engine._inverse_rmse_weights(backtest)
        model_weights = scoring_engine._projection_model_weights(30, backtest, include_lightgbm=True)
        projection, basis, _ = scoring_engine._weighted_ensemble(
            [
                ("Trend", 110.0, model_weights["trend"]),
                ("LightGBM", 120.0, model_weights["lightgbm"]),
                ("Resistance", 130.0, 0.20),
            ]
        )

        self.assertIsNotNone(base_weights)
        self.assertNotIn("arima", base_weights)
        self.assertAlmostEqual(sum(base_weights.values()), 1.0, places=6)
        self.assertNotIn("arima", model_weights)
        self.assertAlmostEqual(model_weights["lightgbm"], scoring_engine.LIGHTGBM_WEIGHT_30D, places=6)
        self.assertAlmostEqual(sum(model_weights.values()), 0.80, places=6)
        self.assertIn("LightGBM(15%)", basis)
        expected = (
            (110.0 * model_weights["trend"])
            + (120.0 * model_weights["lightgbm"])
            + (130.0 * 0.20)
        )
        self.assertAlmostEqual(float(projection), expected, places=6)

    def test_medium_horizon_weights_include_smaller_lightgbm_component(self):
        backtest = {"arima_rmse": 2.0, "trend_rmse": 1.0, "n_windows": 4}

        model_weights = scoring_engine._projection_model_weights(180, backtest, include_lightgbm=True)
        projection, basis, _ = scoring_engine._weighted_ensemble(
            [
                ("Trend", 180.0, model_weights["trend"]),
                ("LightGBM", 210.0, model_weights["lightgbm"]),
                ("DCF", 205.0, scoring_engine.MEDIUM_TERM_DCF_WEIGHT),
            ]
        )

        self.assertLess(scoring_engine.LIGHTGBM_WEIGHT_180D, scoring_engine.LIGHTGBM_WEIGHT_30D)
        self.assertAlmostEqual(model_weights["lightgbm"], scoring_engine.LIGHTGBM_WEIGHT_180D, places=6)
        self.assertNotIn("arima", model_weights)
        # Trend + LightGBM share everything DCF+Comps does not take.
        self.assertAlmostEqual(sum(model_weights.values()), 1.0 - scoring_engine.MEDIUM_TERM_DCF_WEIGHT, places=6)
        self.assertIn("LightGBM(8%)", basis)
        expected = (
            (180.0 * model_weights["trend"])
            + (210.0 * model_weights["lightgbm"])
            + (205.0 * scoring_engine.MEDIUM_TERM_DCF_WEIGHT)
        )
        self.assertAlmostEqual(float(projection), expected, places=6)

    def test_short_and_medium_basis_derive_adaptive_lightgbm_percent_from_backtest_evidence(self):
        # trend_rmse=1.0, lightgbm_rmse=0.9 -> LightGBM has the lower RMSE of
        # the pair, with a raw inverse-RMSE share of 10/19 (~52.6%). With
        # lightgbm_windows=4, the dynamic cap (_lightgbm_adaptive_share_cap)
        # is 0.62, so this modest win is under the cap and passes through
        # unclipped rather than being forced to an even 50/50 split.
        backtest = {
            "arima_rmse": 2.0,
            "trend_rmse": 1.0,
            "lightgbm_rmse": 0.9,
            "n_windows": 4,
            "lightgbm_windows": 4,
        }
        self.assertAlmostEqual(scoring_engine._lightgbm_adaptive_share_cap(4), 0.62, places=6)

        short_weights = scoring_engine._projection_model_weights(30, backtest, include_lightgbm=True)
        medium_weights = scoring_engine._projection_model_weights(180, backtest, include_lightgbm=True)
        medium_budget = 1.0 - scoring_engine.MEDIUM_TERM_DCF_WEIGHT
        self.assertAlmostEqual(sum(short_weights.values()), 0.80, places=6)
        self.assertAlmostEqual(sum(medium_weights.values()), medium_budget, places=6)
        self.assertGreater(short_weights["lightgbm"], scoring_engine.LIGHTGBM_WEIGHT_30D)
        self.assertGreater(medium_weights["lightgbm"], scoring_engine.LIGHTGBM_WEIGHT_180D)
        _, short_basis, _ = scoring_engine._weighted_ensemble(
            [
                ("Trend", 110.0, short_weights["trend"]),
                ("LightGBM", 120.0, short_weights["lightgbm"]),
            ]
        )
        _, medium_basis, _ = scoring_engine._weighted_ensemble(
            [
                ("Trend", 180.0, medium_weights["trend"]),
                ("LightGBM", 210.0, medium_weights["lightgbm"]),
            ]
        )

        # short budget (0.80) * raw share (10/19) = 8/19 (~42.1%) -> "42%".
        self.assertIn("LightGBM(42%)", short_basis)
        # Both horizons now keep a fixed 20% for their one non-trend input (Resistance / DCF+Comps),
        # so the medium budget (0.80) gives the same 8/19 (~42.1%) -> "42%".
        self.assertAlmostEqual(medium_budget, 0.80, places=6)
        self.assertIn("LightGBM(42%)", medium_basis)

    def test_adaptive_lightgbm_share_is_capped_and_excess_redistributed(self):
        # LightGBM's raw inverse-RMSE share here would be ~97.6% of the
        # (trend, lightgbm) pair (rmse of 0.1 vs. 4.0). With
        # lightgbm_windows=4, the dynamic cap is 0.62, so the excess above
        # that is redistributed back to trend (the only remaining
        # non-LightGBM component now that ARIMA has been removed from the
        # ensemble), landing on a 62/38 split rather than an even 50/50 one.
        backtest = {
            "trend_rmse": 4.0,
            "lightgbm_rmse": 0.1,
            "n_windows": 4,
            "lightgbm_windows": 4,
        }
        cap = scoring_engine._lightgbm_adaptive_share_cap(4)
        self.assertAlmostEqual(cap, 0.62, places=6)

        weights = scoring_engine._inverse_rmse_weights(backtest, include_lightgbm=True)

        self.assertIsNotNone(weights)
        self.assertNotIn("arima", weights)
        self.assertAlmostEqual(weights["lightgbm"], cap, places=6)
        self.assertAlmostEqual(weights["trend"], 1.0 - cap, places=6)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)

    def test_lightgbm_adaptive_share_cap_floors_ceilings_and_interpolates(self):
        # Sparse evidence (<=1 window) keeps the conservative 50% floor;
        # evidence scales linearly up to the 70% ceiling at >=6 windows.
        self.assertEqual(scoring_engine._lightgbm_adaptive_share_cap(None), scoring_engine.LIGHTGBM_MIN_ADAPTIVE_SHARE_CAP)
        self.assertEqual(scoring_engine._lightgbm_adaptive_share_cap(0), scoring_engine.LIGHTGBM_MIN_ADAPTIVE_SHARE_CAP)
        self.assertEqual(scoring_engine._lightgbm_adaptive_share_cap(1), scoring_engine.LIGHTGBM_MIN_ADAPTIVE_SHARE_CAP)
        self.assertAlmostEqual(scoring_engine._lightgbm_adaptive_share_cap(2), 0.54, places=6)
        self.assertAlmostEqual(scoring_engine._lightgbm_adaptive_share_cap(3), 0.58, places=6)
        self.assertAlmostEqual(scoring_engine._lightgbm_adaptive_share_cap(6), scoring_engine.LIGHTGBM_MAX_ADAPTIVE_SHARE_CAP)
        self.assertEqual(
            scoring_engine._lightgbm_adaptive_share_cap(10),
            scoring_engine.LIGHTGBM_MAX_ADAPTIVE_SHARE_CAP,
        )

    def test_long_horizon_weights_explicitly_exclude_lightgbm(self):
        weights = scoring_engine._projection_model_weights(
            720,
            {"arima_rmse": 2.0, "trend_rmse": 1.0, "n_windows": 4},
            include_lightgbm=True,
        )
        self.assertNotIn("lightgbm", weights)
        self.assertEqual(set(weights.keys()), {"trend"})
        self.assertAlmostEqual(sum(weights.values()), 1.0 - scoring_engine.LONG_TERM_DCF_WEIGHT, places=6)

    def test_live_projection_falls_back_when_no_saved_lightgbm_model_exists(self):
        data = _sample_projection_data()
        info = {"exchange": "NASDAQ", "marketCap": 5_000_000_000, "currentPrice": float(data["Close"].iloc[-1])}
        technical = {
            "resistance_levels": [135.0],
            "indicators": {"bb_high": 134.0},
        }

        with (
            patch("modules.scoring_engine.LIGHTGBM_AVAILABLE", True),
            # `_load_live_lightgbm_models` checks the release-backed manifest/batch
            # before falling back to `load_return_models`. Without this mock, the
            # unpatched manifest lookup makes a real network call to GitHub
            # Releases (and can download a multi-GB production model batch).
            patch("modules.scoring_engine._load_live_lightgbm_manifest", return_value={}),
            patch("modules.scoring_engine.load_return_models", return_value={}) as load_mock,
            patch("modules.scoring_engine.predict_forward_return") as predict_mock,
            patch("modules.scoring_engine.build_feature_table", return_value=_sample_feature_table(data)),
            patch(
                "modules.scoring_engine.run_walk_forward",
                return_value={
                    "arima_rmse": 2.0,
                    "trend_rmse": 1.0,
                    "lightgbm_rmse": 0.9,
                    "n_windows": 4,
                    "arima_windows": 4,
                    "trend_windows": 4,
                    "lightgbm_windows": 4,
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
            result = scoring_engine._get_price_projections_core("AAPL", info=info, data=data)

        load_mock.assert_called_once()
        predict_mock.assert_not_called()
        self.assertNotIn("LightGBM", result["short_term_basis"])
        self.assertNotIn("LightGBM", result["medium_term_basis"])
        self.assertIn("Trend", result["short_term_basis"])
        self.assertTrue(any(msg.startswith("LightGBM: no saved return models available from live manifest/batch") for msg in result["models_skipped"]))

    def _project_with_valuation(self, data, info, valuation_estimate, *, projection_confidence=None):
        """Run the projection core on `data`/`info` with every external dependency stubbed and no
        saved LightGBM model, so the medium/long-term blends hold only Trend and DCF+Comps.
        `projection_confidence` (0-100) pins the confidence that shrinks targets toward the price."""
        technical = {"resistance_levels": [135.0], "indicators": {"bb_high": 134.0}}
        metrics = {"valuation_estimate": valuation_estimate} if valuation_estimate is not None else {}
        pinned_confidence = (
            patch("modules.scoring_engine._projection_confidence", return_value=projection_confidence)
            if projection_confidence is not None
            else contextlib.nullcontext()
        )
        with (
            patch("modules.scoring_engine.LIGHTGBM_AVAILABLE", True),
            patch("modules.scoring_engine._load_live_lightgbm_manifest", return_value={}),
            patch("modules.scoring_engine.load_return_models", return_value={}),
            patch("modules.scoring_engine.build_feature_table", return_value=_sample_feature_table(data)),
            patch(
                "modules.scoring_engine.run_walk_forward",
                return_value={"trend_rmse": 1.0, "n_windows": 4, "trend_windows": 4, "lightgbm_windows": 0},
            ),
            patch("modules.scoring_engine._garch_confidence_from_returns", return_value=(None, None)),
            patch("modules.scoring_engine.analyze_technical", return_value=technical),
            patch(
                "modules.scoring_engine.get_macro_regime",
                return_value={"risk_free_rate": 0.045, "bullish_sectors": [], "bearish_sectors": [], "market_regime": "neutral"},
            ),
            patch("modules.scoring_engine.analyze_fundamentals", return_value={"metrics": metrics, "fundamental_score": 50}),
            pinned_confidence,
        ):
            return scoring_engine._get_price_projections_core("AAPL", info=info, data=data)

    @staticmethod
    def _sec_style_info(price, **overrides):
        # The SEC feed supplies trailing EPS only and derives trailing P/E from the latest close, so
        # EPS x P/E is just the price (see modules/sec_edgar_client.build_fundamentals_info_adapter).
        return {
            "exchange": "NASDAQ",
            "marketCap": 5_000_000_000,
            "currentPrice": price,
            "trailingEps": 5.0,
            "trailingPE": price / 5.0,
            **overrides,
        }

    def test_medium_and_long_term_blends_have_no_price_anchored_fundamental_slot(self):
        data = _sample_projection_data()
        info = self._sec_style_info(float(data["Close"].iloc[-1]))

        result = self._project_with_valuation(data, info, valuation_estimate=160.0)

        self.assertNotIn("Fundamental Fair Value", result["models_used"])
        self.assertNotIn("Fundamental", result["medium_term_basis"])
        self.assertNotIn("Fundamental", result["long_term_basis"])
        medium_trend = round((1.0 - scoring_engine.MEDIUM_TERM_DCF_WEIGHT) * 100)
        medium_dcf = round(scoring_engine.MEDIUM_TERM_DCF_WEIGHT * 100)
        long_trend = round((1.0 - scoring_engine.LONG_TERM_DCF_WEIGHT) * 100)
        long_dcf = round(scoring_engine.LONG_TERM_DCF_WEIGHT * 100)
        self.assertEqual(result["medium_term_basis"], f"Ensemble: Trend({medium_trend}%) + DCF+Comps({medium_dcf}%)")
        self.assertEqual(result["long_term_basis"], f"Ensemble: Trend({long_trend}%) + DCF+Comps({long_dcf}%)")

    def test_medium_and_long_term_targets_are_the_plain_blend_of_trend_and_dcf_comps(self):
        # A constant-growth price series makes the log-linear and log-polynomial trends coincide, so the
        # trend forecasts are known exactly. With confidence pinned to 100 (no shrinkage toward the
        # price) the targets are just the fixed-weight blends; a price-valued third input (the old
        # EPS x P/E term) would drag both back toward the current price.
        data = _sample_projection_data()
        growth = np.log(1.25) / 180  # the 180-day trend lands 25% above the last close
        data["Close"] = 100.0 * np.exp(growth * np.arange(len(data)))
        data["High"] = data["Close"] + 1.0
        data["Low"] = data["Close"] - 1.0
        price = float(data["Close"].iloc[-1])
        info = self._sec_style_info(price)
        dcf = price * 1.5

        result = self._project_with_valuation(data, info, valuation_estimate=dcf, projection_confidence=100.0)

        trend_180 = price * 1.25
        trend_720 = price * 1.25**4
        medium_dcf = scoring_engine.MEDIUM_TERM_DCF_WEIGHT
        long_dcf = scoring_engine.LONG_TERM_DCF_WEIGHT
        expected_medium = (1.0 - medium_dcf) * trend_180 + medium_dcf * dcf
        expected_long = (1.0 - long_dcf) * trend_720 + long_dcf * dcf
        self.assertAlmostEqual(result["medium_term_target"], expected_medium, delta=0.01)
        self.assertAlmostEqual(result["long_term_target"], expected_long, delta=0.01)
        # Under the old 50/30/20 and 40/35/25 splits these were ~1.225x and ~1.70x the price.
        self.assertGreater(result["medium_term_target"], price * 1.29)
        self.assertGreater(result["long_term_target"], price * 2.0)

    def test_bearish_outlook_is_reachable_when_every_model_forecasts_below_the_price(self):
        # A falling stock whose DCF+Comps estimate is also below the price. The old EPS x P/E term equalled
        # the price exactly, which kept "every model is below the price" from ever being true.
        data = _sample_projection_data(start=130.0, end=100.0)
        price = float(data["Close"].iloc[-1])
        self.assertAlmostEqual(price, 100.0)
        info = self._sec_style_info(price)
        self.assertAlmostEqual(info["trailingEps"] * info["trailingPE"], price)

        result = self._project_with_valuation(data, info, valuation_estimate=80.0)

        self.assertEqual(result["outlook"], "Bearish")

    def test_outlook_is_not_bearish_when_dcf_comps_sits_above_the_price(self):
        data = _sample_projection_data(start=130.0, end=100.0)
        info = self._sec_style_info(float(data["Close"].iloc[-1]))

        result = self._project_with_valuation(data, info, valuation_estimate=140.0)

        self.assertNotEqual(result["outlook"], "Bearish")

    def test_data_quality_follows_the_dcf_comps_estimate_and_a_trend_model(self):
        data = _sample_projection_data()
        price = float(data["Close"].iloc[-1])
        no_eps = {"exchange": "NASDAQ", "marketCap": 5_000_000_000, "currentPrice": price}
        otc_no_eps = {**no_eps, "exchange": "OTC"}
        otc_with_eps = self._sec_style_info(price, exchange="OTC")

        cases = [
            ("EPS-backed listed stock", self._sec_style_info(price), 160.0, "Full"),
            ("listed stock without EPS", no_eps, None, "Limited"),
            ("OTC stock with EPS", otc_with_eps, 160.0, "Full"),
            ("OTC stock without EPS", otc_no_eps, None, "Technical Only"),
        ]
        for label, info, valuation_estimate, expected in cases:
            with self.subTest(label):
                result = self._project_with_valuation(data, info, valuation_estimate)
                self.assertEqual(result["data_quality"], expected)
                self.assertEqual("DCF+Comps Estimate" in result["models_used"], valuation_estimate is not None)

    def test_otc_stock_without_eps_gets_the_technical_trend_only_recommendation(self):
        data = _sample_projection_data()
        info = {"exchange": "OTC", "marketCap": 50_000_000, "currentPrice": float(data["Close"].iloc[-1])}

        result = self._project_with_valuation(data, info, valuation_estimate=None)

        self.assertIn("technical trend only", result["recommendation_to_sell_at"])

    def test_live_model_loader_prefers_release_batch_manifest(self):
        model_30 = object()
        model_180 = object()
        with (
            patch("modules.scoring_engine.LIGHTGBM_AVAILABLE", True),
            patch("modules.scoring_engine._load_live_lightgbm_manifest", return_value={"latest_batch": {"asset_url": "https://example/batch.joblib"}}),
            patch("modules.scoring_engine._load_live_lightgbm_batch", return_value={"models": {"AAPL": {30: model_30, 180: model_180}}}) as batch_mock,
            patch("modules.scoring_engine.load_return_models") as legacy_mock,
        ):
            models = scoring_engine._load_live_lightgbm_models("AAPL")

        batch_mock.assert_called_once_with(("https://example/batch.joblib",))
        legacy_mock.assert_not_called()
        self.assertEqual(models, {30: model_30, 180: model_180})

    def test_live_model_loader_prefers_ticker_shard_over_combined_batch(self):
        model_30 = object()
        model_180 = object()
        manifest = {
            "tickers": {"AAPL": {"shard_index": 1}},
            "latest_batch": {
                "asset_url": "https://example/batch.joblib",
                "shards": [
                    {"shard_index": 0, "asset_urls": ["https://example/shard0.joblib"]},
                    {"shard_index": 1, "asset_urls": ["https://example/shard1.joblib"]},
                ],
            },
        }
        with (
            patch("modules.scoring_engine.LIGHTGBM_AVAILABLE", True),
            patch("modules.scoring_engine._load_live_lightgbm_manifest", return_value=manifest),
            patch(
                "modules.scoring_engine._load_live_lightgbm_shard_batch",
                return_value={"models": {"AAPL": {30: model_30, 180: model_180}}},
            ) as shard_mock,
            patch("modules.scoring_engine._load_live_lightgbm_batch") as batch_mock,
            patch("modules.scoring_engine.load_return_models") as legacy_mock,
        ):
            models = scoring_engine._load_live_lightgbm_models("AAPL")

        shard_mock.assert_called_once_with(("https://example/shard1.joblib",))
        batch_mock.assert_not_called()
        legacy_mock.assert_not_called()
        self.assertEqual(models, {30: model_30, 180: model_180})

    def test_live_model_loader_falls_back_to_combined_batch_when_shard_lookup_misses(self):
        model_30 = object()
        model_180 = object()
        manifest = {
            "tickers": {"AAPL": {"shard_index": 0}},
            "latest_batch": {
                "asset_url": "https://example/batch.joblib",
                "shards": [{"shard_index": 0, "asset_urls": ["https://example/shard0.joblib"]}],
            },
        }
        with (
            patch("modules.scoring_engine.LIGHTGBM_AVAILABLE", True),
            patch("modules.scoring_engine._load_live_lightgbm_manifest", return_value=manifest),
            patch("modules.scoring_engine._load_live_lightgbm_shard_batch", return_value={}) as shard_mock,
            patch(
                "modules.scoring_engine._load_live_lightgbm_batch",
                return_value={"models": {"AAPL": {30: model_30, 180: model_180}}},
            ) as batch_mock,
            patch("modules.scoring_engine.load_return_models") as legacy_mock,
        ):
            models = scoring_engine._load_live_lightgbm_models("AAPL")

        shard_mock.assert_called_once_with(("https://example/shard0.joblib",))
        batch_mock.assert_called_once_with(("https://example/batch.joblib",))
        legacy_mock.assert_not_called()
        self.assertEqual(models, {30: model_30, 180: model_180})

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
            patch("modules.scoring_engine._load_live_lightgbm_manifest", return_value={}),
            patch("modules.scoring_engine.load_return_models", return_value={30: model_30, 180: model_180, 720: model_720}) as load_mock,
            patch("modules.scoring_engine.predict_forward_return", side_effect=_predict) as predict_mock,
            patch("modules.scoring_engine.build_feature_table", return_value=_sample_feature_table(data)),
            patch(
                "modules.scoring_engine.run_walk_forward",
                return_value={
                    "arima_rmse": 2.0,
                    "trend_rmse": 1.0,
                    "lightgbm_rmse": 0.9,
                    "n_windows": 4,
                    "arima_windows": 4,
                    "trend_windows": 4,
                    "lightgbm_windows": 4,
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
            result = scoring_engine._get_price_projections_core("AAPL", info=info, data=data)

        self.assertEqual(load_mock.call_args.kwargs["horizons"], (30, 180))
        self.assertEqual(predict_mock.call_count, 2)
        self.assertIn("LightGBM", result["short_term_basis"])
        self.assertIn("LightGBM", result["medium_term_basis"])
        self.assertNotIn("LightGBM", result["long_term_basis"])
        # Both horizons have a live model AND real backtest evidence
        # (lightgbm_windows=4, lightgbm_rmse=0.9) in this fixture.
        self.assertTrue(result["short_term_lightgbm_backtested"])
        self.assertTrue(result["medium_term_lightgbm_backtested"])

    def test_live_projection_forwards_info_sector_to_feature_builder(self):
        data = _sample_projection_data()
        info = {
            "exchange": "NASDAQ",
            "marketCap": 5_000_000_000,
            "currentPrice": float(data["Close"].iloc[-1]),
            "sector": "Technology",
        }
        technical = {
            "resistance_levels": [135.0],
            "indicators": {"bb_high": 134.0},
        }

        with (
            patch("modules.scoring_engine.LIGHTGBM_AVAILABLE", True),
            patch("modules.scoring_engine._load_live_lightgbm_manifest", return_value={}),
            patch("modules.scoring_engine.load_return_models", return_value={30: object(), 180: object()}),
            patch("modules.scoring_engine.predict_forward_return", return_value=0.05),
            patch("modules.scoring_engine.build_feature_table", return_value=_sample_feature_table(data)) as feature_mock,
            patch(
                "modules.scoring_engine.run_walk_forward",
                return_value={
                    "arima_rmse": 2.0,
                    "trend_rmse": 1.0,
                    "lightgbm_rmse": 0.9,
                    "n_windows": 4,
                    "arima_windows": 4,
                    "trend_windows": 4,
                    "lightgbm_windows": 4,
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
            scoring_engine._get_price_projections_core("AAPL", info=info, data=data)

        feature_mock.assert_called_once()
        self.assertEqual(feature_mock.call_args.kwargs["sector"], "Technology")

    def test_live_projection_uses_fixed_fallback_weight_when_only_one_horizon_has_backtest_evidence(self):
        data = _sample_projection_data()
        info = {"exchange": "NASDAQ", "marketCap": 5_000_000_000, "currentPrice": float(data["Close"].iloc[-1])}
        technical = {
            "resistance_levels": [135.0],
            "indicators": {"bb_high": 134.0},
        }
        model_30 = object()
        model_180 = object()

        def _predict(model, _row):
            return 0.10 if model is model_30 else 0.20

        def _walk_forward(_ticker, _data, horizon=30, **_kwargs):
            # 30d backtest has no LightGBM windows (e.g. too little history for
            # that window count); 180d's separate backtest does. Each horizon's
            # gate must reflect its own backtest, not a shared/reused one: 30d
            # falls back to the small fixed LIGHTGBM_WEIGHT_30D weight (a live
            # model exists but this horizon has no evidence for it), while
            # 180d derives an evidence-based adaptive weight.
            if int(horizon) == 30:
                return {
                    "arima_rmse": 2.0,
                    "trend_rmse": 1.0,
                    "lightgbm_rmse": None,
                    "n_windows": 4,
                    "arima_windows": 4,
                    "trend_windows": 4,
                    "lightgbm_windows": 0,
                }
            return {
                "arima_rmse": 2.0,
                "trend_rmse": 1.0,
                "lightgbm_rmse": 0.9,
                "n_windows": 4,
                "arima_windows": 4,
                "trend_windows": 4,
                "lightgbm_windows": 4,
            }

        with (
            patch("modules.scoring_engine.LIGHTGBM_AVAILABLE", True),
            patch("modules.scoring_engine._load_live_lightgbm_manifest", return_value={}),
            patch("modules.scoring_engine.load_return_models", return_value={30: model_30, 180: model_180}),
            patch("modules.scoring_engine.predict_forward_return", side_effect=_predict),
            patch("modules.scoring_engine.build_feature_table", return_value=_sample_feature_table(data)),
            patch("modules.scoring_engine.run_walk_forward", side_effect=_walk_forward),
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

        # A live model exists for both horizons, so both should now contribute
        # to the basis even though only 180d has per-horizon backtest evidence:
        # 30d falls back to the fixed LIGHTGBM_WEIGHT_30D share instead of
        # being dropped from the ensemble entirely.
        self.assertIn(f"LightGBM({scoring_engine.LIGHTGBM_WEIGHT_30D * 100:.0f}%)", result["short_term_basis"])
        self.assertIn("LightGBM", result["medium_term_basis"])
        # The "backtested" flag distinguishes real per-horizon evidence from
        # the fixed-fallback weight above: 30d has no lightgbm_windows so it's
        # False despite LightGBM still appearing in the basis; 180d has real
        # evidence so it's True.
        self.assertFalse(result["short_term_lightgbm_backtested"])
        self.assertTrue(result["medium_term_lightgbm_backtested"])

    def test_lightgbm_backtested_flag_is_false_without_a_live_model_even_with_backtest_evidence(self):
        # Backtest evidence alone isn't enough: if there's no live per-ticker
        # LightGBM model for a horizon (so it never enters the ensemble at
        # all), the "backtested" flag must stay False for that horizon.
        data = _sample_projection_data()
        info = {"exchange": "NASDAQ", "marketCap": 5_000_000_000, "currentPrice": float(data["Close"].iloc[-1])}
        technical = {"resistance_levels": [135.0], "indicators": {"bb_high": 134.0}}
        model_180 = object()

        with (
            patch("modules.scoring_engine.LIGHTGBM_AVAILABLE", True),
            patch("modules.scoring_engine._load_live_lightgbm_manifest", return_value={}),
            patch("modules.scoring_engine.load_return_models", return_value={180: model_180}),
            patch("modules.scoring_engine.predict_forward_return", return_value=0.20),
            patch("modules.scoring_engine.build_feature_table", return_value=_sample_feature_table(data)),
            patch(
                "modules.scoring_engine.run_walk_forward",
                return_value={
                    "arima_rmse": 2.0,
                    "trend_rmse": 1.0,
                    "lightgbm_rmse": 0.9,
                    "n_windows": 4,
                    "arima_windows": 4,
                    "trend_windows": 4,
                    "lightgbm_windows": 4,
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
            result = scoring_engine._get_price_projections_core("AAPL", info=info, data=data)

        self.assertFalse(result["short_term_lightgbm_backtested"])
        self.assertTrue(result["medium_term_lightgbm_backtested"])


class GarchSharedForecastTests(unittest.TestCase):
    @unittest.skipUnless(scoring_engine.ARCH_AVAILABLE, "arch not installed")
    def test_shared_forecast_matches_independent_per_horizon_fits(self):
        # `_get_price_projections_core` fits GARCH(1,1) once (at the longest
        # 720d horizon) and reuses that forecast for the 30d/180d/720d
        # confidence intervals instead of refitting per horizon. GARCH
        # multi-step variance forecasts are a deterministic recursion from
        # the fitted parameters, so this must produce bit-identical results
        # to the old independent-fit-per-horizon behavior.
        rng = np.random.default_rng(7)
        log_returns = rng.normal(0, 0.01, 400)
        current_price = 100.0

        independent = {
            horizon: scoring_engine._garch_confidence_from_returns(current_price, log_returns, horizon)
            for horizon in (30, 180, 720)
        }

        shared_forecast = scoring_engine._fit_garch_forecast(log_returns, max_horizon=720)
        shared = {
            horizon: scoring_engine._garch_confidence_from_returns(current_price, log_returns, horizon, forecast=shared_forecast)
            for horizon in (30, 180, 720)
        }

        for horizon in (30, 180, 720):
            self.assertAlmostEqual(independent[horizon][0], shared[horizon][0], places=9)
            self.assertAlmostEqual(independent[horizon][1], shared[horizon][1], places=9)

    def test_falls_back_to_independent_fit_when_no_shared_forecast_given(self):
        with patch.object(scoring_engine, "ARCH_AVAILABLE", False):
            low, high = scoring_engine._garch_confidence_from_returns(100.0, np.zeros(100), 30)
        self.assertIsNone(low)
        self.assertIsNone(high)


def _fake_garch_forecast(step_variances):
    # arch's forecast.variance is a one-row frame holding the conditional variance of each day ahead.
    return SimpleNamespace(variance=pd.DataFrame([step_variances]))


class GarchCumulativeBandTests(unittest.TestCase):
    """The GARCH band must use the variance accumulated over the whole horizon, not one day's variance."""

    def _band(self, horizon_days, step_variances, current_price=100.0):
        with patch.object(scoring_engine, "ARCH_AVAILABLE", True):
            return scoring_engine._garch_confidence_from_returns(
                current_price,
                np.full(100, 0.01),  # only its length is checked when a forecast is supplied
                horizon_days,
                forecast=_fake_garch_forecast(step_variances),
            )

    def test_band_is_sized_by_the_summed_daily_variances_over_the_horizon(self):
        # 2% daily volatility = 4 (%^2) per day; over 25 days the variance is 100, so sigma is 10%.
        low, high = self._band(25, [4.0] * 40)

        sigma = np.sqrt(4.0 * 25) / 100.0
        self.assertAlmostEqual(low, 100.0 * np.exp(-1.96 * sigma))
        self.assertAlmostEqual(high, 100.0 * np.exp(1.96 * sigma))

    def test_band_is_wider_than_the_one_day_band_it_used_to_be(self):
        low, high = self._band(30, [4.0] * 30)

        one_day_low = 100.0 * np.exp(-1.96 * 0.02)
        one_day_high = 100.0 * np.exp(1.96 * 0.02)
        self.assertLess(low, one_day_low)
        self.assertGreater(high, one_day_high)
        # sqrt(30) times wider in log terms for a flat daily variance.
        self.assertAlmostEqual(np.log(high / 100.0) / np.log(one_day_high / 100.0), np.sqrt(30))

    def test_band_widens_with_the_horizon_and_is_symmetric_in_log_space(self):
        variances = [4.0] * 720
        bands = {horizon: self._band(horizon, variances) for horizon in (30, 180, 720)}

        for shorter, longer in ((30, 180), (180, 720)):
            self.assertLess(bands[longer][0], bands[shorter][0])
            self.assertGreater(bands[longer][1], bands[shorter][1])
        for low, high in bands.values():
            self.assertAlmostEqual(np.log(100.0 / low), np.log(high / 100.0))

    def test_uses_only_the_first_horizon_steps_of_a_longer_forecast(self):
        # Rising variance: the 3-day band must ignore the later (larger) steps.
        low, high = self._band(3, [1.0, 2.0, 3.0, 400.0, 400.0])

        sigma = np.sqrt(1.0 + 2.0 + 3.0) / 100.0
        self.assertAlmostEqual(high, 100.0 * np.exp(1.96 * sigma))
        self.assertAlmostEqual(low, 100.0 * np.exp(-1.96 * sigma))

    def test_extends_a_forecast_shorter_than_the_horizon_with_its_last_day(self):
        low, high = self._band(5, [1.0, 2.0, 4.0])

        sigma = np.sqrt(1.0 + 2.0 + 4.0 + 4.0 + 4.0) / 100.0
        self.assertAlmostEqual(high, 100.0 * np.exp(1.96 * sigma))
        self.assertAlmostEqual(low, 100.0 * np.exp(-1.96 * sigma))

    def test_returns_no_band_for_an_unusable_forecast(self):
        with (
            patch.object(scoring_engine, "ARCH_AVAILABLE", True),
            self.assertLogs(scoring_engine.logger, level="WARNING"),
        ):
            empty = scoring_engine._garch_confidence_from_returns(
                100.0, np.full(100, 0.01), 30, forecast=_fake_garch_forecast([])
            )
        self.assertEqual(empty, (None, None))

    @unittest.skipUnless(scoring_engine.ARCH_AVAILABLE, "arch not installed")
    def test_real_garch_fit_gives_a_band_wider_than_one_day_of_volatility(self):
        rng = np.random.default_rng(11)
        log_returns = rng.normal(0, 0.02, 500)
        forecast = scoring_engine._fit_garch_forecast(log_returns, max_horizon=30)

        low, high = scoring_engine._garch_confidence_from_returns(100.0, log_returns, 30, forecast=forecast)

        variance = forecast.variance.values[-1]
        self.assertAlmostEqual(high, 100.0 * np.exp(1.96 * np.sqrt(variance[:30].sum()) / 100.0))
        one_day_high = 100.0 * np.exp(1.96 * np.sqrt(variance[0]) / 100.0)
        self.assertGreater(np.log(high / 100.0), 3 * np.log(one_day_high / 100.0))
        self.assertLess(low, 100.0)


class FillMissing52WeekRangeTests(unittest.TestCase):
    def test_fills_range_when_keys_are_explicitly_none(self):
        # Regression test: modules.polygon_client.build_info_adapter always
        # sets these keys (to None, since Polygon's overview endpoint has no
        # 52-week range data) rather than omitting them, so the fallback
        # must check the *value*, not just key membership.
        data = _sample_projection_data(300)
        info = {"fiftyTwoWeekHigh": None, "fiftyTwoWeekLow": None}

        filled = scoring_engine._fill_missing_52w_range(info, data)

        cutoff = pd.to_datetime(data.index).max() - pd.Timedelta(days=365)
        trailing = data.loc[pd.to_datetime(data.index) >= cutoff]
        self.assertAlmostEqual(filled["fiftyTwoWeekHigh"], float(trailing["High"].max()), places=6)
        self.assertAlmostEqual(filled["fiftyTwoWeekLow"], float(trailing["Low"].min()), places=6)

    def test_does_not_overwrite_existing_values(self):
        data = _sample_projection_data(300)
        info = {"fiftyTwoWeekHigh": 999.0, "fiftyTwoWeekLow": 1.0}

        filled = scoring_engine._fill_missing_52w_range(info, data)

        self.assertEqual(filled["fiftyTwoWeekHigh"], 999.0)
        self.assertEqual(filled["fiftyTwoWeekLow"], 1.0)

    def test_handles_empty_data_without_raising(self):
        info = {"fiftyTwoWeekHigh": None, "fiftyTwoWeekLow": None}

        filled = scoring_engine._fill_missing_52w_range(info, pd.DataFrame())

        self.assertIsNone(filled["fiftyTwoWeekHigh"])
        self.assertIsNone(filled["fiftyTwoWeekLow"])


if __name__ == "__main__":
    unittest.main()
