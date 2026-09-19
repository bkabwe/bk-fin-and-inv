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


if __name__ == "__main__":
    unittest.main()
