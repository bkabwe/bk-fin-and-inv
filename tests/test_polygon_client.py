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


if __name__ == "__main__":
    unittest.main()
