from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules import portfolio


class PortfolioStorageTestCase(unittest.TestCase):
    """Base class that isolates portfolio.py's JSON storage to a temp directory."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp_dir = Path(self._tmp.name)
        self.portfolio_file = tmp_dir / "portfolio.json"
        self.watchlist_file = tmp_dir / "watchlist.json"

        patcher_dir = patch.object(portfolio, "DATA_DIR", tmp_dir)
        patcher_pf = patch.object(portfolio, "PORTFOLIO_FILE", self.portfolio_file)
        patcher_wl = patch.object(portfolio, "WATCHLIST_FILE", self.watchlist_file)
        patcher_dir.start()
        patcher_pf.start()
        patcher_wl.start()
        self.addCleanup(patcher_dir.stop)
        self.addCleanup(patcher_pf.stop)
        self.addCleanup(patcher_wl.stop)


class PortfolioHoldingsTests(PortfolioStorageTestCase):
    def test_get_portfolio_starts_empty(self):
        self.assertEqual(portfolio.get_portfolio(), [])

    def test_add_holding_creates_new_entry(self):
        portfolio.add_holding("aapl", 10, 150.0, "2024-01-01", notes="core position")
        holdings = portfolio.get_portfolio()
        self.assertEqual(len(holdings), 1)
        self.assertEqual(holdings[0]["ticker"], "AAPL")
        self.assertEqual(holdings[0]["shares"], 10.0)
        self.assertEqual(holdings[0]["avg_cost"], 150.0)
        self.assertEqual(holdings[0]["notes"], "core position")

    def test_add_holding_updates_existing_entry(self):
        portfolio.add_holding("AAPL", 10, 150.0, "2024-01-01")
        portfolio.add_holding("AAPL", 20, 160.0, "2024-02-01", notes="added more")
        holdings = portfolio.get_portfolio()
        self.assertEqual(len(holdings), 1)
        self.assertEqual(holdings[0]["shares"], 20.0)
        self.assertEqual(holdings[0]["avg_cost"], 160.0)
        self.assertEqual(holdings[0]["notes"], "added more")

    def test_add_holding_sanitizes_ticker(self):
        portfolio.add_holding("  aapl  ", 5, 100.0, "2024-01-01")
        holdings = portfolio.get_portfolio()
        self.assertEqual(holdings[0]["ticker"], "AAPL")

    def test_add_holding_invalid_ticker_raises(self):
        with self.assertRaises(ValueError):
            portfolio.add_holding("AA/PL", 5, 100.0, "2024-01-01")

    def test_remove_holding(self):
        portfolio.add_holding("AAPL", 10, 150.0, "2024-01-01")
        portfolio.add_holding("MSFT", 5, 300.0, "2024-01-01")
        portfolio.remove_holding("aapl")
        holdings = portfolio.get_portfolio()
        self.assertEqual([h["ticker"] for h in holdings], ["MSFT"])

    def test_remove_holding_nonexistent_is_noop(self):
        portfolio.add_holding("AAPL", 10, 150.0, "2024-01-01")
        portfolio.remove_holding("MSFT")
        self.assertEqual(len(portfolio.get_portfolio()), 1)

    def test_update_holding_changes_shares_and_cost(self):
        portfolio.add_holding("AAPL", 10, 150.0, "2024-01-01")
        portfolio.update_holding("aapl", 15, 155.0)
        holdings = portfolio.get_portfolio()
        self.assertEqual(holdings[0]["shares"], 15.0)
        self.assertEqual(holdings[0]["avg_cost"], 155.0)

    def test_update_holding_nonexistent_is_noop(self):
        portfolio.update_holding("AAPL", 15, 155.0)
        self.assertEqual(portfolio.get_portfolio(), [])

    def test_save_portfolio_persists_across_loads(self):
        portfolio.save_portfolio([{"ticker": "NVDA", "shares": 1, "avg_cost": 500}])
        self.assertEqual(portfolio.get_portfolio()[0]["ticker"], "NVDA")


class WatchlistTests(PortfolioStorageTestCase):
    def test_get_watchlist_starts_empty(self):
        self.assertEqual(portfolio.get_watchlist(), [])

    def test_add_to_watchlist(self):
        portfolio.add_to_watchlist("aapl")
        self.assertEqual(portfolio.get_watchlist(), ["AAPL"])

    def test_add_to_watchlist_dedupes(self):
        portfolio.add_to_watchlist("AAPL")
        portfolio.add_to_watchlist("aapl")
        self.assertEqual(portfolio.get_watchlist(), ["AAPL"])

    def test_add_to_watchlist_invalid_ticker_raises(self):
        with self.assertRaises(ValueError):
            portfolio.add_to_watchlist("AA/PL")

    def test_remove_from_watchlist(self):
        portfolio.add_to_watchlist("AAPL")
        portfolio.add_to_watchlist("MSFT")
        portfolio.remove_from_watchlist("aapl")
        self.assertEqual(portfolio.get_watchlist(), ["MSFT"])

    def test_save_watchlist_sorts_and_dedupes(self):
        portfolio.save_watchlist(["msft", "AAPL", "aapl", "goog"])
        self.assertEqual(portfolio.get_watchlist(), ["AAPL", "GOOG", "MSFT"])


class DetectSplitSincePurchaseTests(unittest.TestCase):
    @patch("modules.portfolio.get_reference_splits")
    def test_no_splits_returns_default_result(self, mock_splits):
        mock_splits.return_value = []
        result = portfolio.detect_split_since_purchase("AAPL", "2024-01-01")
        self.assertFalse(result["split_detected"])
        self.assertIsNone(result["split_factor"])
        self.assertEqual(result["events"], [])
        self.assertIsNone(result["warning"])

    @patch("modules.portfolio.get_reference_splits")
    def test_forward_split_detected(self, mock_splits):
        mock_splits.return_value = [
            {"split_from": 1, "split_to": 4, "execution_date": "2024-06-01"}
        ]
        result = portfolio.detect_split_since_purchase("AAPL", "2024-01-01")
        self.assertTrue(result["split_detected"])
        self.assertAlmostEqual(result["split_factor"], 4.0)
        self.assertIn("forward split", result["warning"])

    @patch("modules.portfolio.get_reference_splits")
    def test_reverse_split_detected(self, mock_splits):
        mock_splits.return_value = [
            {"split_from": 4, "split_to": 1, "execution_date": "2024-06-01"}
        ]
        result = portfolio.detect_split_since_purchase("AAPL", "2024-01-01")
        self.assertTrue(result["split_detected"])
        self.assertAlmostEqual(result["split_factor"], 0.25)
        self.assertIn("reverse split", result["warning"])

    @patch("modules.portfolio.get_reference_splits")
    def test_cumulative_multiple_splits(self, mock_splits):
        mock_splits.return_value = [
            {"split_from": 1, "split_to": 2, "execution_date": "2024-03-01"},
            {"split_from": 1, "split_to": 2, "execution_date": "2024-09-01"},
        ]
        result = portfolio.detect_split_since_purchase("AAPL", "2024-01-01")
        self.assertAlmostEqual(result["split_factor"], 4.0)
        self.assertEqual(len(result["events"]), 2)

    @patch("modules.portfolio.get_reference_splits")
    def test_invalid_split_ratios_are_skipped(self, mock_splits):
        mock_splits.return_value = [{"split_from": 0, "split_to": 0, "execution_date": "2024-06-01"}]
        result = portfolio.detect_split_since_purchase("AAPL", "2024-01-01")
        self.assertFalse(result["split_detected"])

    @patch("modules.portfolio.get_reference_splits")
    def test_exception_from_api_is_caught_and_returns_default(self, mock_splits):
        mock_splits.side_effect = RuntimeError("network down")
        result = portfolio.detect_split_since_purchase("AAPL", "2024-01-01")
        self.assertFalse(result["split_detected"])
        self.assertIsNone(result["warning"])

    @patch("modules.portfolio.get_reference_splits")
    def test_empty_date_purchased_is_handled(self, mock_splits):
        mock_splits.return_value = []
        result = portfolio.detect_split_since_purchase("AAPL", "")
        self.assertFalse(result["split_detected"])


class AnalyzePortfolioHoldingsTests(PortfolioStorageTestCase):
    @patch("modules.portfolio.detect_split_since_purchase")
    @patch("modules.portfolio.get_price_projections")
    @patch("modules.portfolio.analyze_stock")
    def test_computes_pnl_and_recommendation_fields(
        self, mock_analyze, mock_projections, mock_split
    ):
        portfolio.add_holding("AAPL", 10, 100.0, "2024-01-01")
        mock_analyze.return_value = {
            "current_price": 120.0,
            "score": 85,
            "recommendation": "Buy",
            "sell_recommendation": None,
            "entry_price": 100.0,
            "target_price": 140.0,
            "projections": {
                "models_used": ["arima"],
                "data_quality": "Good",
                "recommendation_to_sell_at": "$150",
                "short_term_target": 130.0,
                "short_term_low": 120.0,
                "short_term_high": 140.0,
                "short_term_upside": 0.1,
                "short_term_basis": "arima",
                "medium_term_target": None,
                "medium_term_low": None,
                "medium_term_high": None,
                "medium_term_upside": None,
                "medium_term_basis": None,
                "long_term_target": None,
                "long_term_low": None,
                "long_term_high": None,
                "long_term_upside": None,
                "long_term_basis": None,
            },
        }
        mock_split.return_value = {"warning": None}

        rows = portfolio.analyze_portfolio_holdings()

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["Ticker"], "AAPL")
        self.assertEqual(row["Current Value"], 1200.0)
        self.assertEqual(row["Cost Basis"], 1000.0)
        self.assertEqual(row["P&L $"], 200.0)
        self.assertEqual(row["P&L %"], 20.0)
        self.assertEqual(row["Score"], 85)
        self.assertEqual(row["recommendation_to_sell_at"], "$150")
        mock_projections.assert_not_called()

    @patch("modules.portfolio.detect_split_since_purchase")
    @patch("modules.portfolio.get_price_projections")
    @patch("modules.portfolio.analyze_stock")
    def test_holds_for_breakeven_when_underwater(
        self, mock_analyze, mock_projections, mock_split
    ):
        portfolio.add_holding("AAPL", 10, 150.0, "2024-01-01")
        mock_analyze.return_value = {
            "current_price": 100.0,
            "score": 40,
            "recommendation": "Hold",
            "sell_recommendation": None,
            "entry_price": 150.0,
            "target_price": None,
            "projections": {"models_used": [], "data_quality": "Limited", "recommendation_to_sell_at": None},
        }
        mock_split.return_value = {"warning": None}

        rows = portfolio.analyze_portfolio_holdings()

        self.assertIn("break-even", rows[0]["recommendation_to_sell_at"])

    @patch("modules.portfolio.detect_split_since_purchase")
    @patch("modules.portfolio.get_price_projections")
    @patch("modules.portfolio.analyze_stock")
    def test_falls_back_to_get_price_projections_when_missing(
        self, mock_analyze, mock_projections, mock_split
    ):
        portfolio.add_holding("AAPL", 10, 100.0, "2024-01-01")
        mock_analyze.return_value = {
            "current_price": 110.0,
            "score": 70,
            "recommendation": "Buy",
            "sell_recommendation": None,
            "entry_price": 100.0,
            "target_price": 120.0,
            "projections": None,
        }
        mock_projections.return_value = {"models_used": ["trend"], "data_quality": "Limited"}
        mock_split.return_value = {"warning": None}

        rows = portfolio.analyze_portfolio_holdings()

        mock_projections.assert_called_once_with("AAPL")
        self.assertEqual(rows[0]["Projection Models"], "trend")

    @patch("modules.portfolio.detect_split_since_purchase")
    @patch("modules.portfolio.get_price_projections")
    @patch("modules.portfolio.analyze_stock")
    def test_analysis_exception_produces_error_row(
        self, mock_analyze, mock_projections, mock_split
    ):
        portfolio.add_holding("AAPL", 10, 100.0, "2024-01-01")
        mock_analyze.side_effect = RuntimeError("boom")

        rows = portfolio.analyze_portfolio_holdings()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Ticker"], "AAPL")
        self.assertEqual(rows[0]["Recommendation"], "⚠️ Analysis Failed")
        self.assertIn("boom", rows[0]["Error"])
        mock_split.assert_not_called()

    def test_empty_portfolio_returns_empty_list(self):
        self.assertEqual(portfolio.analyze_portfolio_holdings(), [])


if __name__ == "__main__":
    unittest.main()
