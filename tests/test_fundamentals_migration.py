from __future__ import annotations

import unittest
import sys
import types
from unittest.mock import Mock, patch

import requests

try:
    import pandas  # noqa: F401
except Exception:
    fake_pandas = types.ModuleType("pandas")

    class _DataFrame:  # pragma: no cover - import-time shim only
        pass

    fake_pandas.DataFrame = _DataFrame
    fake_pandas.to_datetime = lambda *args, **kwargs: None  # type: ignore[assignment]
    fake_pandas.isna = lambda value: False  # type: ignore[assignment]
    sys.modules["pandas"] = fake_pandas

from modules import fred_client, macro_regime, polygon_client, sec_edgar_client
from modules import backtester, scoring_engine


def _http_error(status_code: int, retry_after: str | None = None) -> requests.exceptions.HTTPError:
    response = requests.Response()
    response.status_code = status_code
    response.url = "https://example.test/resource"
    if retry_after is not None:
        response.headers["Retry-After"] = retry_after
    return requests.exceptions.HTTPError(f"status={status_code}", response=response)


class RetryBehaviorTests(unittest.TestCase):
    def test_polygon_does_not_retry_non_retryable_403(self):
        attempts = {"count": 0}

        def _fetch():
            attempts["count"] += 1
            raise _http_error(403)

        with patch("modules.polygon_client.time.sleep") as sleep_mock:
            with self.assertRaises(requests.exceptions.HTTPError):
                polygon_client._fetch_with_retry(_fetch, max_retries=3, base_delay=0.01)
        self.assertEqual(attempts["count"], 1)
        sleep_mock.assert_not_called()

    def test_polygon_retries_429_with_retry_after(self):
        attempts = {"count": 0}

        def _fetch():
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise _http_error(429, retry_after="2")
            return {"ok": True}

        with patch("modules.polygon_client.time.sleep") as sleep_mock:
            payload = polygon_client._fetch_with_retry(_fetch, max_retries=3, base_delay=0.01)
        self.assertEqual(payload, {"ok": True})
        self.assertEqual(attempts["count"], 2)
        sleep_mock.assert_called_once_with(2.0)

    def test_fred_does_not_retry_non_retryable_403(self):
        attempts = {"count": 0}

        def _fetch():
            attempts["count"] += 1
            raise _http_error(403)

        with patch("modules.fred_client.time.sleep") as sleep_mock:
            with self.assertRaises(requests.exceptions.HTTPError):
                fred_client._fetch_with_retry(_fetch, max_retries=3, base_delay=0.01)
        self.assertEqual(attempts["count"], 1)
        sleep_mock.assert_not_called()

    def test_fred_retries_429_with_retry_after(self):
        attempts = {"count": 0}

        def _fetch():
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise _http_error(429, retry_after="2")
            return {"ok": True}

        with patch("modules.fred_client.time.sleep") as sleep_mock:
            payload = fred_client._fetch_with_retry(_fetch, max_retries=3, base_delay=0.01)
        self.assertEqual(payload, {"ok": True})
        self.assertEqual(attempts["count"], 2)
        sleep_mock.assert_called_once_with(2.0)


class SecFundamentalsAdapterTests(unittest.TestCase):
    def test_sec_adapter_builds_expected_ratio_fields(self):
        facts = {
            "facts": {
                "us-gaap": {
                    "SalesRevenueNet": {
                        "units": {
                            "USD": [
                                {"end": "2024-12-31", "val": 1200, "form": "10-K", "filed": "2025-02-01"},
                                {"end": "2023-12-31", "val": 1000, "form": "10-K", "filed": "2024-02-01"},
                            ]
                        }
                    },
                    "NetIncomeLoss": {
                        "units": {
                            "USD": [
                                {"end": "2024-12-31", "val": 200, "form": "10-K", "filed": "2025-02-01"},
                                {"end": "2023-12-31", "val": 160, "form": "10-K", "filed": "2024-02-01"},
                            ]
                        }
                    },
                    "Liabilities": {"units": {"USD": [{"end": "2024-12-31", "val": 500, "form": "10-K"}]}},
                    "StockholdersEquity": {"units": {"USD": [{"end": "2024-12-31", "val": 1000, "form": "10-K"}]}},
                    "EarningsPerShareDiluted": {"units": {"USD/shares": [{"end": "2024-12-31", "val": 5, "form": "10-K"}]}},
                    "CommonStockSharesOutstanding": {"units": {"shares": [{"end": "2024-12-31", "val": 40, "form": "10-K"}]}},
                }
            }
        }

        info = sec_edgar_client.build_fundamentals_info_adapter(facts, current_price=100)

        self.assertAlmostEqual(info["trailingPE"], 20.0)
        self.assertAlmostEqual(info["trailingEps"], 5.0)
        self.assertAlmostEqual(info["returnOnEquity"], 0.2)
        self.assertAlmostEqual(info["debtToEquity"], 50.0)
        self.assertAlmostEqual(info["revenueGrowth"], 0.2)
        self.assertAlmostEqual(info["earningsGrowth"], 0.25)
        self.assertAlmostEqual(info["sharesOutstanding"], 40.0)
        self.assertIsNone(info["forwardPE"])
        self.assertIsNone(info["forwardEps"])


class FredMacroRegimeTests(unittest.TestCase):
    def test_fred_vix_observations_filter_missing_values(self):
        payload = {
            "observations": [
                {"date": "2026-09-01", "value": "16.12"},
                {"date": "2026-09-02", "value": "."},
                {"date": "2026-09-03", "value": "18.45"},
            ]
        }

        response = Mock()
        response.json.return_value = payload
        response.raise_for_status.return_value = None

        with patch.dict("os.environ", {"FRED_API_KEY": "fred-key"}, clear=False):
            with patch("modules.fred_client.requests.get") as get_mock:
                get_mock.return_value = response

                df = fred_client.get_vix_observations("2026-09-01", "2026-09-03")

        self.assertEqual(list(df.columns), ["Close"])
        self.assertEqual(len(df), 2)
        self.assertAlmostEqual(float(df["Close"].iloc[-1]), 18.45)

    def test_macro_regime_returns_default_when_fred_not_configured(self):
        with patch("modules.macro_regime.get_vix_observations", side_effect=fred_client.FredNotConfiguredError("FRED_API_KEY is not configured")):
            result = macro_regime.get_macro_regime()

        self.assertEqual(
            result,
            {
                "vix": None,
                "vix_regime": "medium",
                "yield_10y": None,
                "risk_free_rate": 0.045,
                "bullish_sectors": [],
                "bearish_sectors": [],
                "market_regime": "neutral",
            },
        )


class ForecastingEnsembleTests(unittest.TestCase):
    def test_inverse_rmse_weights_use_arima_and_trend(self):
        weights = scoring_engine._inverse_rmse_weights(
            {"arima_rmse": 2.0, "trend_rmse": 1.0, "n_windows": 4, "unused_rmse": 0.0001}
        )
        self.assertIsNotNone(weights)
        self.assertEqual(set(weights.keys()), {"arima", "trend"})
        self.assertGreater(weights["trend"], weights["arima"])
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)

    def test_walk_forward_default_shape(self):
        pd_mod = __import__("pandas")
        result = backtester.run_walk_forward("AAPL", pd_mod.DataFrame())
        self.assertEqual(
            set(result.keys()),
            {
                "arima_rmse",
                "trend_rmse",
                "lightgbm_rmse",
                "n_windows",
                "arima_windows",
                "trend_windows",
                "lightgbm_windows",
            },
        )

    def test_confidence_bounds_fallback_without_garch(self):
        low, high = scoring_engine._confidence_bounds(
            target=110.0,
            values=[108.0, 112.0],
            current_price=100.0,
            cap_value=None,
            garch_low=None,
            garch_high=None,
        )
        self.assertEqual(low, 100.0)
        self.assertEqual(high, 120.0)

    def test_confidence_bounds_fallback_for_missing_garch_edge(self):
        low, high = scoring_engine._confidence_bounds(
            target=110.0,
            values=[108.0, 112.0],
            current_price=100.0,
            cap_value=None,
            garch_low=95.0,
            garch_high=None,
        )
        self.assertEqual(low, 95.0)
        self.assertEqual(high, 120.0)

    def test_confidence_bounds_fallback_low_never_above_current_price(self):
        low, _ = scoring_engine._confidence_bounds(
            target=150.0,
            values=[145.0, 155.0],
            current_price=100.0,
            cap_value=None,
            garch_low=None,
            garch_high=170.0,
        )
        self.assertLessEqual(low, 100.0)

    def test_confidence_bounds_fallback_high_never_below_current_price(self):
        _, high = scoring_engine._confidence_bounds(
            target=70.0,
            values=[72.0, 75.0],
            current_price=100.0,
            cap_value=None,
            garch_low=60.0,
            garch_high=None,
        )
        self.assertGreaterEqual(high, 100.0)


if __name__ == "__main__":
    unittest.main()
