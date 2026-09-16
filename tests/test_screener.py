from __future__ import annotations

import unittest
from unittest.mock import patch

from modules import screener


class RunScreenerTests(unittest.TestCase):
    def _stock_result(self, ticker: str, **overrides):
        result = {
            "ticker": ticker,
            "company": f"{ticker} Corp.",
            "score": 80,
            "recommendation": "Buy",
            "time_horizon": "30d",
            "sector_trend": "very_bullish",
            "market_cap_tier": "large_cap",
            "longterm_stage": "compounder",
            "entry_price": 100.0,
            "target_price": 120.0,
            "stop_loss": 90.0,
            "current_price": 110.0,
        }
        result.update(overrides)
        return result

    def test_run_screener_ranks_sorts_and_limits_results(self):
        tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
        fast_scores = {
            "AAA": (80, None),
            "BBB": (72, None),
            "CCC": (60, None),
            "DDD": (58, None),
            "EEE": (40, None),
        }
        analyzed = {
            "AAA": self._stock_result("AAA", score=92, current_price=112.0),
            "BBB": self._stock_result("BBB", score=71, current_price=106.0),
            "CCC": self._stock_result("CCC", score=85, current_price=108.0),
            "DDD": self._stock_result("DDD", score=65, current_price=95.0),
        }
        progress_updates: list[float] = []
        totals: list[int] = []
        processed: list[tuple[str, dict]] = []

        def fast_score_side_effect(ticker: str):
            return fast_scores[ticker]

        def analyze_side_effect(ticker: str):
            return analyzed[ticker]

        def on_processed(ticker: str, counts: dict):
            processed.append((ticker, dict(counts)))

        with (
            patch("modules.screener.get_sp500_tickers", return_value=tickers),
            patch("modules.screener.fast_screen_score", side_effect=fast_score_side_effect) as fast_mock,
            patch("modules.screener.analyze_stock", side_effect=analyze_side_effect) as analyze_mock,
        ):
            result = screener.run_screener(
                min_score=70,
                max_results=2,
                label="S&P 500",
                max_workers=1,
                progress_callback=progress_updates.append,
                on_ticker_processed=on_processed,
                on_total_known=totals.append,
            )

        self.assertEqual(fast_mock.call_count, 5)
        self.assertEqual([call.args[0] for call in analyze_mock.call_args_list], ["AAA", "BBB", "CCC", "DDD"])
        self.assertEqual(result["Ticker"].tolist(), ["AAA", "CCC"])
        self.assertEqual(result["Score"].tolist(), [92, 85])
        self.assertEqual(result["Index"].tolist(), ["S&P 500", "S&P 500"])
        self.assertEqual(result["% from Entry"].tolist(), [12.0, 8.0])
        self.assertEqual(progress_updates, [0.2, 0.4, 0.6, 0.8, 1.0])
        self.assertEqual(totals, [5])
        self.assertEqual([ticker for ticker, _counts in processed], tickers)
        self.assertEqual(
            processed[-1][1],
            {
                "screened": 5,
                "total": 5,
                "qualified": 3,
                "fast_filtered": 1,
                "fully_analyzed": 4,
                "failed_count": 0,
            },
        )
        self.assertFalse(result.attrs["stopped"])
        self.assertEqual(result.attrs["source_ticker_count"], 5)
        self.assertEqual(result.attrs["qualified_count"], 3)
        self.assertEqual(result.attrs["fast_filtered_count"], 1)
        self.assertEqual(result.attrs["fully_analyzed_count"], 4)
        self.assertEqual(result.attrs["failed_count"], 0)
        self.assertEqual(result.attrs["universe"], "sp500")

    def test_run_screener_custom_universe_sanitizes_input_and_handles_missing_fields(self):
        custom_input = [" msft ", "", "???", "msft"]
        analyzed = self._stock_result(
            "MSFT",
            sector_trend=None,
            market_cap_tier=None,
            longterm_stage=None,
            entry_price=None,
            current_price=150.0,
        )

        with (
            patch("modules.screener.sanitize_ticker_list", return_value=["MSFT"]) as sanitize_mock,
            patch("modules.screener.fast_screen_score") as fast_mock,
            patch("modules.screener.analyze_stock", return_value=analyzed) as analyze_mock,
        ):
            result = screener.run_screener(
                universe="CUSTOM",
                custom_tickers=custom_input,
                use_fast_screen=False,
                max_workers=1,
            )

        sanitize_mock.assert_called_once_with(custom_input)
        fast_mock.assert_not_called()
        analyze_mock.assert_called_once_with("MSFT")
        self.assertEqual(result["Ticker"].tolist(), ["MSFT"])
        self.assertNotIn("Index", result.columns)
        row = result.iloc[0]
        self.assertEqual(row["Sector Trend"], "Unknown")
        self.assertEqual(row["Market Cap Tier"], "unknown")
        self.assertEqual(row["Long-Term Stage"], "")
        self.assertIsNone(row["% from Entry"])
        self.assertEqual(result.attrs["source_ticker_count"], 1)
        self.assertEqual(result.attrs["fast_filtered_count"], 0)
        self.assertEqual(result.attrs["fully_analyzed_count"], 1)
        self.assertEqual(result.attrs["failed_count"], 0)

    def test_run_screener_returns_empty_dataframe_for_empty_universe(self):
        progress_updates: list[float] = []
        totals: list[int] = []

        with (
            patch("modules.screener.get_nasdaq_tickers", return_value=[]),
            patch("modules.screener.fast_screen_score") as fast_mock,
            patch("modules.screener.analyze_stock") as analyze_mock,
        ):
            result = screener.run_screener(
                universe="nasdaq",
                max_workers=1,
                progress_callback=progress_updates.append,
                on_total_known=totals.append,
            )

        fast_mock.assert_not_called()
        analyze_mock.assert_not_called()
        self.assertTrue(result.empty)
        self.assertEqual(result.attrs, {})
        self.assertEqual(progress_updates, [])
        self.assertEqual(totals, [])

    def test_run_screener_returns_empty_results_when_all_tickers_are_fast_filtered(self):
        tickers = ["OTC1", "OTC2"]

        def fast_score_side_effect(ticker: str):
            return {"OTC1": (10, None), "OTC2": (20, None)}[ticker]

        with (
            patch("modules.screener.get_otc_tickers", return_value=tickers),
            patch("modules.screener.fast_screen_score", side_effect=fast_score_side_effect),
            patch("modules.screener.analyze_stock") as analyze_mock,
        ):
            result = screener.run_screener(universe="otc", min_score=70, max_workers=1)

        analyze_mock.assert_not_called()
        self.assertTrue(result.empty)
        self.assertFalse(result.attrs["stopped"])
        self.assertEqual(result.attrs["source_ticker_count"], 2)
        self.assertEqual(result.attrs["qualified_count"], 0)
        self.assertEqual(result.attrs["fast_filtered_count"], 2)
        self.assertEqual(result.attrs["fully_analyzed_count"], 0)
        self.assertEqual(result.attrs["failed_count"], 0)
        self.assertEqual(result.attrs["failed_tickers"], [])

    def test_run_screener_records_fast_screen_and_analysis_failures(self):
        tickers = ["GOOD", "BROKEN", "FASTERR"]
        progress_updates: list[float] = []

        def fast_score_side_effect(ticker: str):
            return {
                "GOOD": (80, None),
                "BROKEN": (90, None),
                "FASTERR": (0, "polygon timeout"),
            }[ticker]

        def analyze_side_effect(ticker: str):
            if ticker == "BROKEN":
                raise ValueError("bad payload")
            return self._stock_result("GOOD", score=88)

        with (
            patch("modules.screener.get_nyseamerican_tickers", return_value=tickers),
            patch("modules.screener.fast_screen_score", side_effect=fast_score_side_effect),
            patch("modules.screener.analyze_stock", side_effect=analyze_side_effect) as analyze_mock,
        ):
            result = screener.run_screener(
                universe="nyseamerican",
                min_score=70,
                max_workers=1,
                progress_callback=progress_updates.append,
            )

        self.assertEqual([call.args[0] for call in analyze_mock.call_args_list], ["GOOD", "BROKEN"])
        self.assertEqual(result["Ticker"].tolist(), ["GOOD"])
        self.assertEqual(progress_updates, [1 / 3, 2 / 3, 1.0])
        self.assertEqual(result.attrs["fully_analyzed_count"], 1)
        self.assertEqual(result.attrs["failed_count"], 2)
        self.assertEqual(
            result.attrs["failed_tickers"],
            [
                {"ticker": "BROKEN", "reason": "bad payload"},
                {"ticker": "FASTERR", "reason": "fast-screen error: polygon timeout"},
            ],
        )

    def test_run_screener_wraps_universe_fetch_runtime_errors(self):
        with (
            patch("modules.screener.get_nasdaq_tickers", side_effect=RuntimeError("upstream outage")),
            self.assertRaisesRegex(
                RuntimeError,
                "Unable to fetch ticker universe 'nasdaq': upstream outage",
            ),
        ):
            screener.run_screener(universe="nasdaq", max_workers=1)


if __name__ == "__main__":
    unittest.main()
