from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from modules import macro_regime
from modules.fred_client import FRED_10Y_TREASURY_SERIES_ID, FredNotConfiguredError
from modules.polygon_client import PolygonNotConfiguredError

DEFAULT_REGIME = {
    "vix": None,
    "vix_regime": "medium",
    "yield_10y": None,
    "risk_free_rate": 0.045,
    "bullish_sectors": [],
    "bearish_sectors": [],
    "market_regime": "neutral",
}


def _vix_frame(*closes: float) -> pd.DataFrame:
    return pd.DataFrame({"Close": list(closes)})


def _yield_frame(*values: float) -> pd.DataFrame:
    return pd.DataFrame({"value": list(values)})


def _price_frame(*, uptrend: bool) -> pd.DataFrame:
    closes = np.linspace(100.0, 200.0, 120) if uptrend else np.linspace(200.0, 100.0, 120)
    return pd.DataFrame({"Close": closes})


class GetMacroRegimeTests(unittest.TestCase):
    def setUp(self):
        self._clear_cache()
        self.addCleanup(self._clear_cache)

    @staticmethod
    def _clear_cache():
        clear = getattr(macro_regime.get_macro_regime, "clear", None)
        if callable(clear):
            clear()

    def test_uses_free_fred_dgs10_for_the_10y_yield_and_never_requests_polygon_tnx(self):
        polygon_symbols: list[str] = []

        def _polygon(symbol, **kwargs):
            polygon_symbols.append(symbol)
            return _price_frame(uptrend=True)

        with (
            patch("modules.macro_regime.get_vix_observations", return_value=_vix_frame(17.0, 18.456)),
            patch("modules.macro_regime.get_series_observations", return_value=_yield_frame(4.1, 4.253)) as series_mock,
            patch("modules.macro_regime.get_stock_data_polygon", side_effect=_polygon),
        ):
            result = macro_regime.get_macro_regime()

        self.assertEqual(series_mock.call_args.args[0], FRED_10Y_TREASURY_SERIES_ID)
        self.assertNotIn("I:TNX", polygon_symbols)
        self.assertEqual(result["yield_10y"], 4.25)
        self.assertAlmostEqual(result["risk_free_rate"], 0.04253, places=5)
        self.assertEqual(result["vix"], 18.46)
        self.assertEqual(result["vix_regime"], "medium")
        self.assertEqual(sorted(polygon_symbols), ["SPY", "XLE", "XLF", "XLI", "XLK", "XLV"])
        # Six uptrending ETFs while VIX is in the medium band => breadth decides.
        self.assertEqual(result["market_regime"], "risk_on")
        self.assertEqual(len(result["bullish_sectors"]), 6)

    def test_missing_10y_yield_keeps_vix_and_sector_trends(self):
        with (
            patch("modules.macro_regime.get_vix_observations", return_value=_vix_frame(12.0)),
            patch("modules.macro_regime.get_series_observations", side_effect=RuntimeError("FRED unavailable")),
            patch("modules.macro_regime.get_stock_data_polygon", return_value=_price_frame(uptrend=False)),
        ):
            result = macro_regime.get_macro_regime()

        self.assertIsNone(result["yield_10y"])
        self.assertEqual(result["risk_free_rate"], 0.045)
        self.assertEqual(result["vix"], 12.0)
        self.assertEqual(result["vix_regime"], "low")
        self.assertEqual(result["market_regime"], "risk_on")
        self.assertEqual(len(result["bearish_sectors"]), 6)

    def test_missing_vix_keeps_10y_yield_and_sector_trends(self):
        with (
            patch("modules.macro_regime.get_vix_observations", side_effect=RuntimeError("VIX unavailable")),
            patch("modules.macro_regime.get_series_observations", return_value=_yield_frame(3.9)),
            patch("modules.macro_regime.get_stock_data_polygon", return_value=_price_frame(uptrend=False)),
        ):
            result = macro_regime.get_macro_regime()

        self.assertIsNone(result["vix"])
        self.assertEqual(result["vix_regime"], "medium")
        self.assertEqual(result["yield_10y"], 3.9)
        self.assertAlmostEqual(result["risk_free_rate"], 0.039)
        self.assertEqual(len(result["bearish_sectors"]), 6)
        self.assertEqual(result["market_regime"], "risk_off")

    def test_one_failing_sector_etf_does_not_discard_the_others(self):
        def _polygon(symbol, **kwargs):
            if symbol == "XLE":
                raise RuntimeError("403 Forbidden")
            return _price_frame(uptrend=True)

        with (
            patch("modules.macro_regime.get_vix_observations", return_value=_vix_frame(20.0)),
            patch("modules.macro_regime.get_series_observations", return_value=_yield_frame(4.0)),
            patch("modules.macro_regime.get_stock_data_polygon", side_effect=_polygon),
        ):
            result = macro_regime.get_macro_regime()

        self.assertNotIn("Energy", result["bullish_sectors"] + result["bearish_sectors"])
        self.assertEqual(sorted(result["bullish_sectors"]), ["Financials", "Healthcare", "Industrials", "SPY", "Technology"])
        self.assertEqual(result["yield_10y"], 4.0)

    def test_returns_defaults_when_every_source_fails(self):
        with (
            patch("modules.macro_regime.get_vix_observations", side_effect=RuntimeError("FRED down")),
            patch("modules.macro_regime.get_series_observations", side_effect=RuntimeError("FRED down")),
            patch("modules.macro_regime.get_stock_data_polygon", side_effect=RuntimeError("Polygon down")),
        ):
            result = macro_regime.get_macro_regime()

        self.assertEqual(result, DEFAULT_REGIME)

    def test_fred_not_configured_returns_defaults_without_querying_other_sources(self):
        with (
            patch("modules.macro_regime.get_vix_observations", side_effect=FredNotConfiguredError("FRED_API_KEY is not configured")),
            patch("modules.macro_regime.get_series_observations", return_value=_yield_frame(4.0)) as series_mock,
            patch("modules.macro_regime.get_stock_data_polygon", return_value=_price_frame(uptrend=True)) as polygon_mock,
        ):
            result = macro_regime.get_macro_regime()

        self.assertEqual(result, DEFAULT_REGIME)
        series_mock.assert_not_called()
        polygon_mock.assert_not_called()

    def test_polygon_not_configured_returns_defaults(self):
        with (
            patch("modules.macro_regime.get_vix_observations", return_value=_vix_frame(18.0)),
            patch("modules.macro_regime.get_series_observations", return_value=_yield_frame(4.0)),
            patch("modules.macro_regime.get_stock_data_polygon", side_effect=PolygonNotConfiguredError("POLYGON_API_KEY is not configured")),
        ):
            result = macro_regime.get_macro_regime()

        self.assertEqual(result, DEFAULT_REGIME)


if __name__ == "__main__":
    unittest.main()
