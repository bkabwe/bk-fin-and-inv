from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from modules import polygon_client


class PolygonClientTests(unittest.TestCase):
    def test_get_aggregates_range_preserves_missing_values_instead_of_zero_filling(self):
        payload = {
            "results": [
                {"t": 1704067200000, "o": 10.0, "h": 11.0, "l": 9.0, "c": None, "v": 1000},
                {"t": 1704153600000, "o": None, "h": None, "l": None, "c": None, "v": None},
            ]
        }

        with patch("modules.polygon_client._request_json", return_value=payload):
            data = polygon_client.get_aggregates_range(
                "AIMTEST",
                multiplier=1,
                timespan="day",
                from_date="2024-01-01",
                to_date="2024-01-02",
                adjusted=True,
            )

        self.assertEqual(len(data), 1)
        self.assertTrue(pd.isna(data.iloc[0]["Close"]))
        self.assertEqual(float(data.iloc[0]["Open"]), 10.0)

    def test_list_active_ticker_details_sends_exchange_not_primary_exchange_param(self):
        # Regression test: Polygon's /v3/reference/tickers list endpoint only
        # recognizes the "exchange" query param for MIC-code filtering;
        # "primary_exchange" (a result-row field name, not a request param)
        # was previously sent instead and silently ignored by Polygon.
        payload = {"results": [{"ticker": "AAPL", "primary_exchange": "XNAS", "type": "CS"}]}

        with patch("modules.polygon_client._request_json", return_value=payload) as request_mock:
            tickers = polygon_client.list_active_ticker_details(primary_exchange="XNAS", otc=False)

        called_params = request_mock.call_args.args[1]
        self.assertEqual(called_params.get("exchange"), "XNAS")
        self.assertNotIn("primary_exchange", called_params)
        self.assertNotIn("otc", called_params)
        self.assertEqual([row["ticker"] for row in tickers], ["AAPL"])

    def test_list_active_ticker_details_excludes_otc_rows_when_otc_false(self):
        payload = {
            "results": [
                {"ticker": "AAPL", "primary_exchange": "XNAS", "type": "CS"},
                {"ticker": "PINKX", "primary_exchange": "OTC MARKETS", "type": "CS"},
            ]
        }

        with patch("modules.polygon_client._request_json", return_value=payload):
            tickers = polygon_client.list_active_ticker_details(otc=False)

        self.assertEqual([row["ticker"] for row in tickers], ["AAPL"])

    def test_list_active_ticker_details_keeps_only_otc_rows_when_otc_true(self):
        payload = {
            "results": [
                {"ticker": "AAPL", "primary_exchange": "XNAS", "type": "CS"},
                {"ticker": "PINKX", "primary_exchange": "OTC MARKETS", "type": "CS"},
            ]
        }

        with patch("modules.polygon_client._request_json", return_value=payload):
            tickers = polygon_client.list_active_ticker_details(otc=True)

        self.assertEqual([row["ticker"] for row in tickers], ["PINKX"])

    def test_get_grouped_daily_bars_maps_ticker_to_close_and_volume(self):
        payload = {
            "results": [
                {"T": "AAPL", "c": 150.0, "v": 50_000_000},
                {"T": "ZZZZ", "c": 1.0, "v": 500},
            ]
        }

        with patch("modules.polygon_client._request_json", return_value=payload) as request_mock:
            bars = polygon_client.get_grouped_daily_bars("2024-01-05")

        called_path = request_mock.call_args.args[0]
        self.assertIn("/v2/aggs/grouped/locale/us/market/stocks/2024-01-05", called_path)
        self.assertEqual(bars["AAPL"], {"close": 150.0, "volume": 50_000_000.0})
        self.assertEqual(bars["ZZZZ"], {"close": 1.0, "volume": 500.0})

    def test_get_grouped_daily_bars_returns_empty_dict_for_no_results(self):
        with patch("modules.polygon_client._request_json", return_value={"results": []}):
            bars = polygon_client.get_grouped_daily_bars("2024-01-06")

        self.assertEqual(bars, {})

    def test_get_latest_grouped_daily_bars_walks_back_to_first_non_empty_day(self):
        call_count = {"n": 0}

        def _fake_grouped(trading_date: str):
            call_count["n"] += 1
            if call_count["n"] < 3:
                return {}
            return {"AAPL": {"close": 150.0, "volume": 1000.0}}

        with patch("modules.polygon_client.get_grouped_daily_bars", side_effect=_fake_grouped):
            bars = polygon_client.get_latest_grouped_daily_bars(lookback_days=10)

        self.assertEqual(bars, {"AAPL": {"close": 150.0, "volume": 1000.0}})
        self.assertEqual(call_count["n"], 3)

    def test_get_latest_grouped_daily_bars_returns_empty_after_exhausting_lookback(self):
        with patch("modules.polygon_client.get_grouped_daily_bars", return_value={}) as grouped_mock:
            bars = polygon_client.get_latest_grouped_daily_bars(lookback_days=3)

        self.assertEqual(bars, {})
        self.assertEqual(grouped_mock.call_count, 3)


class BuildInfoAdapterSectorTests(unittest.TestCase):
    # Each test uses its own ticker symbol because build_info_adapter is
    # TTL-cached per ticker for the life of the process.
    def _adapter_info(self, ticker: str, overview: dict) -> dict:
        with (
            patch("modules.polygon_client.get_ticker_overview", return_value=overview),
            patch("modules.polygon_client.get_previous_close", return_value=100.0),
            patch("modules.polygon_client.get_fundamentals_info_adapter", return_value={}),
            patch("modules.polygon_client.get_reference_dividends", return_value=[]),
        ):
            return polygon_client.build_info_adapter(ticker)

    def test_sector_is_the_standard_name_mapped_from_the_sic_code(self):
        # Regression test: the raw SIC description used to be returned as the
        # sector, which matches none of the standard sector names, so the
        # sector P/E benchmark, sector momentum and sector-ETF features all
        # silently did nothing.
        info = self._adapter_info(
            "SICSOFTTEST",
            {"name": "Soft Co", "sic_code": "7372", "sic_description": "SERVICES-PREPACKAGED SOFTWARE"},
        )

        self.assertEqual(info["sector"], "Technology")
        self.assertEqual(info["industry"], "SERVICES-PREPACKAGED SOFTWARE")

    def test_sector_is_none_when_the_sic_code_is_missing_or_unmapped(self):
        for ticker, overview in (
            ("SICMISSINGTEST", {"name": "Fund", "sic_description": "SOME DESCRIPTION"}),
            ("SICUNMAPPEDTEST", {"name": "Shell", "sic_code": "9995", "sic_description": "NON-OPERATING ESTABLISHMENTS"}),
        ):
            with self.subTest(ticker=ticker):
                info = self._adapter_info(ticker, overview)
                self.assertIsNone(info["sector"])
                self.assertEqual(info["industry"], overview["sic_description"])

    def test_an_explicit_overview_sector_is_used_when_the_sic_code_has_no_mapping(self):
        info = self._adapter_info("SICFALLBACKTEST", {"name": "Odd Co", "sector": "Utilities"})

        self.assertEqual(info["sector"], "Utilities")


if __name__ == "__main__":
    unittest.main()
