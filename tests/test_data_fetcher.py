from __future__ import annotations

import unittest
from unittest.mock import patch

from modules import data_fetcher


class FilterTickersByLiquidityTests(unittest.TestCase):
    def test_drops_tickers_below_threshold(self):
        bars = {
            "AAPL": {"close": 150.0, "volume": 50_000_000.0},  # $7.5B/day
            "PENNY": {"close": 0.01, "volume": 1000.0},  # $10/day
        }
        with patch("modules.data_fetcher.get_latest_grouped_daily_bars", return_value=bars):
            result = data_fetcher.filter_tickers_by_liquidity(["AAPL", "PENNY"])

        self.assertEqual(result, ["AAPL"])

    def test_keeps_tickers_missing_from_grouped_bars(self):
        bars = {"AAPL": {"close": 150.0, "volume": 50_000_000.0}}
        with patch("modules.data_fetcher.get_latest_grouped_daily_bars", return_value=bars):
            result = data_fetcher.filter_tickers_by_liquidity(["AAPL", "NEWIPO"])

        self.assertEqual(set(result), {"AAPL", "NEWIPO"})

    def test_fails_open_when_grouped_bars_empty(self):
        with patch("modules.data_fetcher.get_latest_grouped_daily_bars", return_value={}):
            result = data_fetcher.filter_tickers_by_liquidity(["AAPL", "PENNY"])

        self.assertEqual(result, ["AAPL", "PENNY"])

    def test_fails_open_when_grouped_bars_raises(self):
        with patch("modules.data_fetcher.get_latest_grouped_daily_bars", side_effect=RuntimeError("boom")):
            result = data_fetcher.filter_tickers_by_liquidity(["AAPL", "PENNY"])

        self.assertEqual(result, ["AAPL", "PENNY"])

    def test_empty_input_returns_empty_list_without_fetching_bars(self):
        with patch("modules.data_fetcher.get_latest_grouped_daily_bars") as bars_mock:
            result = data_fetcher.filter_tickers_by_liquidity([])

        self.assertEqual(result, [])
        bars_mock.assert_not_called()

    def test_respects_custom_min_dollar_volume(self):
        bars = {"SMALLCAP": {"close": 5.0, "volume": 100_000.0}}  # $500K/day
        with patch("modules.data_fetcher.get_latest_grouped_daily_bars", return_value=bars):
            dropped = data_fetcher.filter_tickers_by_liquidity(["SMALLCAP"], min_dollar_volume=1_000_000.0)
            kept = data_fetcher.filter_tickers_by_liquidity(["SMALLCAP"], min_dollar_volume=100_000.0)

        self.assertEqual(dropped, [])
        self.assertEqual(kept, ["SMALLCAP"])


class UniverseFetcherLiquidityWiringTests(unittest.TestCase):
    def setUp(self):
        if hasattr(data_fetcher.get_nasdaq_tickers, "clear"):
            data_fetcher.get_nasdaq_tickers.clear()
        if hasattr(data_fetcher.get_nyseamerican_tickers, "clear"):
            data_fetcher.get_nyseamerican_tickers.clear()
        if hasattr(data_fetcher.get_otc_tickers, "clear"):
            data_fetcher.get_otc_tickers.clear()
        if hasattr(data_fetcher.get_sp500_tickers, "clear"):
            data_fetcher.get_sp500_tickers.clear()

    def test_get_nasdaq_tickers_applies_liquidity_filter(self):
        with (
            patch("modules.data_fetcher.is_polygon_configured", return_value=True),
            patch("modules.data_fetcher.list_active_tickers", return_value=["AAPL", "PENNY"]),
            patch(
                "modules.data_fetcher.filter_tickers_by_liquidity",
                return_value=["AAPL"],
            ) as filter_mock,
        ):
            result = data_fetcher.get_nasdaq_tickers()

        filter_mock.assert_called_once_with(["AAPL", "PENNY"])
        self.assertEqual(result, ["AAPL"])

    def test_get_nyseamerican_tickers_applies_liquidity_filter(self):
        with (
            patch("modules.data_fetcher.is_polygon_configured", return_value=True),
            patch("modules.data_fetcher.list_active_tickers", return_value=["AAPL", "PENNY"]),
            patch(
                "modules.data_fetcher.filter_tickers_by_liquidity",
                return_value=["AAPL"],
            ) as filter_mock,
        ):
            result = data_fetcher.get_nyseamerican_tickers()

        filter_mock.assert_called_once_with(["AAPL", "PENNY"])
        self.assertEqual(result, ["AAPL"])

    def test_get_otc_tickers_applies_lower_liquidity_floor(self):
        with (
            patch("modules.data_fetcher.is_polygon_configured", return_value=True),
            patch("modules.data_fetcher.list_active_tickers", return_value=["PINKX"]),
            patch(
                "modules.data_fetcher.filter_tickers_by_liquidity",
                return_value=["PINKX"],
            ) as filter_mock,
        ):
            result = data_fetcher.get_otc_tickers()

        filter_mock.assert_called_once_with(
            ["PINKX"], min_dollar_volume=data_fetcher.DEFAULT_MIN_AVG_DOLLAR_VOLUME_OTC
        )
        self.assertEqual(result, ["PINKX"])

    def test_get_sp500_tickers_applies_liquidity_filter(self):
        html = "<table><tr><th>Symbol</th></tr>" + "".join(f"<tr><td>T{i:04d}</td></tr>" for i in range(101)) + "</table>"

        class _FakeResponse:
            text = html

            def raise_for_status(self):
                return None

        with (
            patch("modules.data_fetcher.requests.get", return_value=_FakeResponse()),
            patch(
                "modules.data_fetcher.filter_tickers_by_liquidity",
                side_effect=lambda tickers: tickers[:5],
            ) as filter_mock,
        ):
            result = data_fetcher.get_sp500_tickers()

        filter_mock.assert_called_once()
        self.assertEqual(len(result), 5)


if __name__ == "__main__":
    unittest.main()
