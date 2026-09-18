from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from modules.cache_fallback import ttl_cache


class TtlCacheDataFrameKeyingTests(unittest.TestCase):
    """The non-Streamlit `cache_data` fallback (PR 58 fixed a crash here, but
    left it bypassing the cache whenever a call's key included a DataFrame)
    must actually cache calls whose DataFrame arguments have equal content,
    while still invoking the wrapped function again for genuinely different
    data. This is the regression test for `run_walk_forward`'s duplicate
    backtest invocation across `run_screener`/`scan_and_filter_profit_opportunities`."""

    def test_two_calls_with_equal_dataframe_content_hit_the_cache(self):
        calls = []

        @ttl_cache(ttl=60)
        def f(ticker, data, horizon=30, evaluate_lightgbm=False):
            calls.append(ticker)
            return len(data)

        df1 = pd.DataFrame({"Close": [1.0, 2.0, 3.0]})
        df2 = pd.DataFrame({"Close": [1.0, 2.0, 3.0]})  # equal content, distinct object

        result1 = f("AAPL", df1, horizon=30, evaluate_lightgbm=True)
        result2 = f("AAPL", df2, horizon=30, evaluate_lightgbm=True)

        self.assertEqual(len(calls), 1)
        self.assertEqual(result1, result2)

    def test_two_calls_with_different_dataframe_content_both_execute(self):
        calls = []

        @ttl_cache(ttl=60)
        def f(ticker, data):
            calls.append(ticker)
            return len(data)

        df1 = pd.DataFrame({"Close": [1.0, 2.0, 3.0]})
        df2 = pd.DataFrame({"Close": [1.0, 2.0, 4.0]})  # one value differs

        f("AAPL", df1)
        f("AAPL", df2)

        self.assertEqual(len(calls), 2)

    def test_different_horizon_kwarg_is_not_cached_together(self):
        calls = []

        @ttl_cache(ttl=60)
        def f(ticker, data, horizon=30):
            calls.append(horizon)
            return horizon

        df = pd.DataFrame({"Close": [1.0, 2.0, 3.0]})
        f("AAPL", df, horizon=30)
        f("AAPL", df, horizon=180)

        self.assertEqual(calls, [30, 180])

    def test_different_shape_same_tail_values_is_not_cached_together(self):
        # A 1y window and a 5y window that happen to share the same trailing
        # values are still genuinely different calls (different shape/index
        # bounds) -- only truly identical inputs should share a cache entry.
        calls = []

        @ttl_cache(ttl=60)
        def f(ticker, data):
            calls.append(len(data))
            return len(data)

        short = pd.DataFrame({"Close": [1.0, 2.0, 3.0]})
        long = pd.DataFrame({"Close": [0.0, 1.0, 2.0, 3.0]})

        f("AAPL", short)
        f("AAPL", long)

        self.assertEqual(calls, [3, 4])

    def test_series_and_ndarray_arguments_are_hashable(self):
        calls = []

        @ttl_cache(ttl=60)
        def f(series, arr):
            calls.append(1)
            return float(series.sum()) + float(arr.sum())

        series1 = pd.Series([1.0, 2.0, 3.0])
        series2 = pd.Series([1.0, 2.0, 3.0])
        arr1 = np.array([1.0, 2.0])
        arr2 = np.array([1.0, 2.0])

        f(series1, arr1)
        f(series2, arr2)

        self.assertEqual(len(calls), 1)

    def test_scalar_only_calls_still_cache_as_before(self):
        calls = []

        @ttl_cache(ttl=60)
        def f(ticker, target_price, horizon, rsi=None):
            calls.append(ticker)
            return f"{ticker}-{target_price}-{horizon}-{rsi}"

        f("AAPL", 150.0, "short_term", rsi=55.0)
        f("AAPL", 150.0, "short_term", rsi=55.0)
        f("AAPL", 150.0, "medium_term", rsi=55.0)

        self.assertEqual(len(calls), 2)

    def test_unfreezable_argument_falls_back_to_uncached_execution(self):
        calls = []

        class Unhashable:
            __hash__ = None

        @ttl_cache(ttl=60)
        def f(value):
            calls.append(1)
            return id(value)

        obj = Unhashable()
        f(obj)
        f(obj)

        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
