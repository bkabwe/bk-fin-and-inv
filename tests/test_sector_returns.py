from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from modules import sector_returns


def _sample_sector_table(length: int = 120) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=length)
    xlk_close = pd.Series(np.linspace(100.0, 150.0, num=length), index=index)
    return pd.DataFrame(
        {
            "XLK_return_21d": xlk_close.pct_change(periods=21),
            "XLK_return_63d": xlk_close.pct_change(periods=63),
        },
        index=index,
    )


class SectorRelativeStrengthColumnsTests(unittest.TestCase):
    def test_sector_relative_strength_columns_are_uniform_regardless_of_sector(self):
        columns = sector_returns.sector_relative_strength_columns()
        self.assertEqual(
            columns,
            [
                "sector_return_21d",
                "sector_relative_return_21d",
                "sector_return_63d",
                "sector_relative_return_63d",
            ],
        )


class ComputeRelativeStrengthFeaturesTests(unittest.TestCase):
    def test_unresolvable_sector_returns_all_nan_without_fetching(self):
        index = pd.bdate_range("2024-01-02", periods=30)
        close = pd.Series(np.linspace(100.0, 110.0, num=30), index=index)

        with patch("modules.sector_returns.build_sector_return_table") as fetch_mock:
            features = sector_returns.compute_relative_strength_features(index, close, "Not A Real Sector")

        fetch_mock.assert_not_called()
        self.assertEqual(list(features.columns), sector_returns.sector_relative_strength_columns())
        self.assertTrue(features.isna().all().all())

    def test_none_sector_returns_all_nan_without_fetching(self):
        index = pd.bdate_range("2024-01-02", periods=30)
        close = pd.Series(np.linspace(100.0, 110.0, num=30), index=index)

        with patch("modules.sector_returns.build_sector_return_table") as fetch_mock:
            features = sector_returns.compute_relative_strength_features(index, close, None)

        fetch_mock.assert_not_called()
        self.assertTrue(features.isna().all().all())

    def test_empty_shared_table_returns_all_nan(self):
        index = pd.bdate_range("2024-01-02", periods=30)
        close = pd.Series(np.linspace(100.0, 110.0, num=30), index=index)

        features = sector_returns.compute_relative_strength_features(
            index, close, "Technology", shared_sector_return_table=pd.DataFrame()
        )

        self.assertTrue(features.isna().all().all())

    def test_computes_sector_return_and_relative_return_against_shared_table(self):
        sector_table = _sample_sector_table(120)
        index = sector_table.index
        close = pd.Series(np.linspace(200.0, 260.0, num=len(index)), index=index)

        features = sector_returns.compute_relative_strength_features(
            index, close, "Technology", shared_sector_return_table=sector_table
        )

        expected_sector_return_21d = sector_table["XLK_return_21d"].reindex(index).ffill()
        expected_stock_return_21d = close.pct_change(periods=21)
        expected_relative_21d = expected_stock_return_21d - expected_sector_return_21d

        pd.testing.assert_series_equal(
            features["sector_return_21d"], expected_sector_return_21d.astype("float64"), check_names=False
        )
        pd.testing.assert_series_equal(
            features["sector_relative_return_21d"], expected_relative_21d.astype("float64"), check_names=False
        )

    def test_sector_alias_resolves_to_same_etf_mapping(self):
        index = pd.bdate_range("2024-01-02", periods=120)
        xlf_close = pd.Series(np.linspace(30.0, 40.0, num=len(index)), index=index)
        sector_table = pd.DataFrame(
            {
                "XLF_return_21d": xlf_close.pct_change(periods=21),
                "XLF_return_63d": xlf_close.pct_change(periods=63),
            },
            index=index,
        )
        close = pd.Series(np.linspace(200.0, 260.0, num=len(index)), index=index)

        canonical = sector_returns.compute_relative_strength_features(
            index, close, "Financials", shared_sector_return_table=sector_table
        )
        # "Financial Services" is a documented normalize_sector_name alias
        # for "Financials" -- both should resolve to the same XLF mapping.
        aliased = sector_returns.compute_relative_strength_features(
            index, close, "Financial Services", shared_sector_return_table=sector_table
        )
        pd.testing.assert_frame_equal(canonical, aliased)
        self.assertFalse(canonical["sector_return_21d"].isna().all())

    def test_empty_daily_index_returns_empty_frame_with_expected_columns(self):
        index = pd.DatetimeIndex([])
        close = pd.Series(dtype="float64")

        features = sector_returns.compute_relative_strength_features(index, close, "Technology")

        self.assertEqual(list(features.columns), sector_returns.sector_relative_strength_columns())
        self.assertEqual(len(features), 0)


class BuildSectorReturnTableTests(unittest.TestCase):
    def setUp(self):
        if hasattr(sector_returns.build_sector_return_table, "clear"):
            sector_returns.build_sector_return_table.clear()

    def test_fetches_each_distinct_etf_exactly_once(self):
        index = pd.bdate_range("2024-01-02", periods=90)
        price_frame = pd.DataFrame({"Close": np.linspace(50.0, 80.0, num=len(index))}, index=index)

        with patch("modules.data_fetcher.get_stock_data", return_value=price_frame) as fetch_mock:
            table = sector_returns.build_sector_return_table()

        distinct_etfs = sorted(set(sector_returns.SECTOR_ETF_MAP.values()))
        self.assertEqual(fetch_mock.call_count, len(distinct_etfs))
        for etf in distinct_etfs:
            self.assertIn(f"{etf}_return_21d", table.columns)
            self.assertIn(f"{etf}_return_63d", table.columns)

    def test_returns_empty_frame_when_no_etf_data_available(self):
        with patch("modules.data_fetcher.get_stock_data", return_value=pd.DataFrame()):
            table = sector_returns.build_sector_return_table()

        self.assertTrue(table.empty)


if __name__ == "__main__":
    unittest.main()
