from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd

from scripts import scan_email_report


def _manifest_with_shards(shard_count: int, ticker_shards: dict[str, int]) -> dict:
    tickers = {
        ticker: {"shard_index": shard_index, "last_trained": "2026-01-01"}
        for ticker, shard_index in ticker_shards.items()
    }
    latest_batch = {
        "batch_id": "batch-1",
        "asset_url": "https://example.com/batch.joblib",
        "asset_urls": ["https://example.com/batch.joblib"],
    }
    if shard_count > 1:
        latest_batch["shard_count"] = shard_count
        latest_batch["shards"] = [
            {
                "shard_index": index,
                "asset_name": f"batch-shard-{index}.joblib",
                "asset_urls": [f"https://example.com/shard-{index}.joblib"],
            }
            for index in range(shard_count)
        ]
    return {"tickers": tickers, "latest_batch": latest_batch}


class TickersForShardTests(unittest.TestCase):
    def test_single_shard_returns_all_manifest_tickers_sorted(self):
        manifest = _manifest_with_shards(1, {"MSFT": 0, "AAPL": 0})
        self.assertEqual(scan_email_report.tickers_for_shard(manifest, 0, 1), ["AAPL", "MSFT"])

    def test_multi_shard_filters_by_assigned_shard_index(self):
        manifest = _manifest_with_shards(2, {"AAPL": 0, "MSFT": 1, "GOOG": 0})
        self.assertEqual(scan_email_report.tickers_for_shard(manifest, 0, 2), ["AAPL", "GOOG"])
        self.assertEqual(scan_email_report.tickers_for_shard(manifest, 1, 2), ["MSFT"])

    def test_ticker_missing_shard_index_defaults_to_shard_zero(self):
        manifest = {"tickers": {"NFLX": {}}, "latest_batch": {"shard_count": 2}}
        self.assertEqual(scan_email_report.tickers_for_shard(manifest, 0, 2), ["NFLX"])
        self.assertEqual(scan_email_report.tickers_for_shard(manifest, 1, 2), [])


class ManifestHelperTests(unittest.TestCase):
    def test_manifest_shard_count_defaults_to_one_when_absent(self):
        self.assertEqual(scan_email_report.manifest_shard_count({"latest_batch": {}}), 1)
        self.assertEqual(scan_email_report.manifest_shard_count(None), 1)

    def test_manifest_shard_count_reads_latest_batch(self):
        manifest = _manifest_with_shards(4, {"AAPL": 0})
        self.assertEqual(scan_email_report.manifest_shard_count(manifest), 4)

    def test_manifest_tickers_sorted_and_uppercased(self):
        manifest = {"tickers": {"msft": {}, "AAPL": {}, "": {}}}
        self.assertEqual(scan_email_report.manifest_tickers(manifest), ["AAPL", "MSFT"])


class ShardBatchAssetUrlsTests(unittest.TestCase):
    def test_uses_shard_urls_when_available(self):
        manifest = _manifest_with_shards(2, {"AAPL": 0})
        urls = scan_email_report._shard_batch_asset_urls(manifest, 0, 2)
        self.assertEqual(urls, ["https://example.com/shard-0.joblib"])

    def test_falls_back_to_combined_batch_when_shard_lookup_empty(self):
        manifest = _manifest_with_shards(1, {"AAPL": 0})
        with patch("scripts.scan_email_report.shard_asset_urls_by_index", return_value=[]) as mock_lookup:
            urls = scan_email_report._shard_batch_asset_urls(manifest, 0, 1)
        mock_lookup.assert_not_called()
        self.assertEqual(urls, ["https://example.com/batch.joblib"])

    def test_falls_back_to_combined_batch_when_shard_count_gt_one_but_no_shard_urls(self):
        manifest = _manifest_with_shards(1, {"AAPL": 0})
        manifest["latest_batch"]["shard_count"] = 3
        with patch("scripts.scan_email_report.shard_asset_urls_by_index", return_value=[]):
            urls = scan_email_report._shard_batch_asset_urls(manifest, 0, 3)
        self.assertEqual(urls, ["https://example.com/batch.joblib"])


class ScanAndFilterFinalizeSplitTests(unittest.TestCase):
    def test_scan_and_filter_does_not_record_or_strip_internal_fields(self):
        scanned = pd.DataFrame({"Ticker": ["AAPL"], "_rsi": [55.0], "_trend_score": [10]})
        scanned.attrs.update({"scanned_count": 1, "fast_filtered_count": 0, "fully_analyzed_count": 1, "failed_count": 0})
        filtered = pd.DataFrame({"Ticker": ["AAPL"], "_rsi": [55.0], "_trend_score": [10]})

        with (
            patch("scripts.scan_email_report.scan_profit_opportunities", return_value=scanned) as mock_scan,
            patch("scripts.scan_email_report.filter_by_upside", return_value=filtered) as mock_filter,
            patch("scripts.scan_email_report.record_predictions_from_scan") as mock_record,
        ):
            results, stats_base = scan_email_report.scan_and_filter_profit_opportunities(["AAPL"], "short_term")

        mock_record.assert_not_called()
        mock_filter.assert_called_once()
        self.assertEqual(mock_filter.call_args.kwargs.get("max_results"), None)
        self.assertIn("_rsi", results.columns)
        self.assertEqual(stats_base["scanned_count"], 1)
        self.assertEqual(stats_base["lightgbm_unconfirmed_count"], 0)
        self.assertNotIn("qualified_count", stats_base)
        self.assertNotIn("recorded_count", stats_base)
        mock_scan.assert_called_once()

    def test_scan_and_filter_drops_rows_lacking_lightgbm_backtest_evidence(self):
        scanned = pd.DataFrame(
            {
                "Ticker": ["AAPL", "MSFT"],
                "_lightgbm_backtested": [True, False],
                "Projected Upside %": [20.0, 30.0],
            }
        )
        scanned.attrs.update({"scanned_count": 2, "fast_filtered_count": 0, "fully_analyzed_count": 2, "failed_count": 0})

        with (
            patch("scripts.scan_email_report.scan_profit_opportunities", return_value=scanned),
            patch("scripts.scan_email_report.record_predictions_from_scan"),
        ):
            results, stats_base = scan_email_report.scan_and_filter_profit_opportunities(["AAPL", "MSFT"], "short_term")

        self.assertEqual(list(results["Ticker"]), ["AAPL"])
        self.assertEqual(stats_base["lightgbm_unconfirmed_count"], 1)

    def test_finalize_strips_internal_fields_and_records_once_when_requested(self):
        results = pd.DataFrame({"Ticker": ["MSFT", "AAPL"], "_rsi": [60.0, 55.0], "_trend_score": [1, 2]})

        with patch("scripts.scan_email_report.record_predictions_from_scan", return_value=2) as mock_record:
            display, stats = scan_email_report.finalize_profit_results(
                results, "short_term", record=True, stats_base={"scanned_count": 5}
            )

        mock_record.assert_called_once()
        self.assertEqual(mock_record.call_args.kwargs["horizon"], "short_term")
        self.assertNotIn("_rsi", display.columns)
        self.assertNotIn("_trend_score", display.columns)
        self.assertEqual(stats["recorded_count"], 2)
        self.assertEqual(stats["qualified_count"], 2)
        self.assertEqual(stats["scanned_count"], 5)

    def test_finalize_drops_confidence_and_lightgbm_backtested_for_short_term_only(self):
        results = pd.DataFrame(
            {
                "Ticker": ["AAPL"],
                "Confidence": ["Full"],
                "_lightgbm_backtested": [True],
            }
        )

        with patch("scripts.scan_email_report.record_predictions_from_scan"):
            short_display, _ = scan_email_report.finalize_profit_results(results, "short_term", record=False)
            medium_display, _ = scan_email_report.finalize_profit_results(results, "medium_term", record=False)

        self.assertNotIn("Confidence", short_display.columns)
        self.assertNotIn("_lightgbm_backtested", short_display.columns)
        self.assertIn("Confidence", medium_display.columns)
        self.assertNotIn("_lightgbm_backtested", medium_display.columns)

    def test_finalize_does_not_record_when_record_false(self):
        results = pd.DataFrame({"Ticker": ["MSFT"], "_rsi": [60.0]})
        with patch("scripts.scan_email_report.record_predictions_from_scan") as mock_record:
            _display, stats = scan_email_report.finalize_profit_results(results, "short_term", record=False)
        mock_record.assert_not_called()
        self.assertEqual(stats["recorded_count"], 0)

    def test_finalize_handles_empty_results(self):
        with patch("scripts.scan_email_report.record_predictions_from_scan") as mock_record:
            display, stats = scan_email_report.finalize_profit_results(pd.DataFrame(), "short_term", record=True)
        mock_record.assert_not_called()
        self.assertTrue(display.empty)
        self.assertEqual(stats["recorded_count"], 0)
        self.assertEqual(stats["qualified_count"], 0)


class RequireLightgbmBacktestedTests(unittest.TestCase):
    def test_keeps_only_confirmed_rows_and_counts_dropped(self):
        results = pd.DataFrame({"Ticker": ["AAPL", "MSFT", "GOOG"], "_lightgbm_backtested": [True, False, True]})
        filtered, dropped = scan_email_report._require_lightgbm_backtested(results)
        self.assertEqual(list(filtered["Ticker"]), ["AAPL", "GOOG"])
        self.assertEqual(dropped, 1)

    def test_no_op_when_column_missing(self):
        results = pd.DataFrame({"Ticker": ["AAPL"]})
        filtered, dropped = scan_email_report._require_lightgbm_backtested(results)
        pd.testing.assert_frame_equal(filtered, results)
        self.assertEqual(dropped, 0)

    def test_no_op_when_results_empty(self):
        filtered, dropped = scan_email_report._require_lightgbm_backtested(pd.DataFrame())
        self.assertTrue(filtered.empty)
        self.assertEqual(dropped, 0)



def _fake_analysis(
    ticker: str,
    *,
    score: int = 80,
    current_price: float = 100.0,
    upside_pct: float = 20.0,
    lightgbm_backtested: bool = True,
) -> dict:
    """A minimal but shape-complete `analyze_stock()`-style result, covering
    every field `screener_row_from_analysis`/`profit_row_from_analysis` read."""
    return {
        "ticker": ticker,
        "company": f"{ticker} Inc.",
        "current_price": current_price,
        "score": score,
        "recommendation": "Buy",
        "sector_trend": "bullish",
        "market_cap_tier": "large",
        "time_horizon": "Short-Term Trade",
        "longterm_stage": "",
        "entry_price": current_price,
        "target_price": current_price * 1.1,
        "stop_loss": current_price * 0.9,
        "technical": {"indicators": {"rsi": 55.0}},
        "score_breakdown": {"trend": 1, "momentum": 2, "relative_strength": 3, "breakout": 4, "volume_quality": 5},
        "projections": {
            "current_price": current_price,
            "short_term_target": current_price * (1 + upside_pct / 100),
            "short_term_low": current_price * 1.05,
            "short_term_high": current_price * 1.15,
            "short_term_upside": upside_pct,
            "short_term_basis": "ARIMA",
            "short_term_lightgbm_backtested": lightgbm_backtested,
            "data_quality": "Full",
        },
    }


class RunScanShardTests(unittest.TestCase):
    def test_empty_shard_skips_batch_download_and_scan(self):
        manifest = _manifest_with_shards(2, {"AAPL": 1})
        with (
            patch("scripts.scan_email_report.load_return_model_batch_from_urls") as mock_load_batch,
            patch("scripts.scan_email_report.analyze_stock") as mock_analyze,
        ):
            partial = scan_email_report.run_scan_shard("short_term", manifest, pd.DataFrame(), 0, 2)

        mock_load_batch.assert_not_called()
        mock_analyze.assert_not_called()
        self.assertEqual(partial["shard_index"], 0)
        self.assertTrue(partial["screener"].empty)
        self.assertTrue(partial["profit"].empty)

    def test_non_empty_shard_calls_analyze_stock_exactly_once_per_ticker(self):
        # Regression test for the PR-58 shard-timeout root cause: `analyze_stock`
        # (and the 30d/180d walk-forward backtests it triggers) must run exactly
        # once per ticker per shard, not once via a screener pass and again via
        # a profit-opportunities pass.
        manifest = _manifest_with_shards(2, {"AAPL": 0, "MSFT": 0})
        fake_price_data = pd.DataFrame({"Close": [1.0, 2.0]})

        with (
            patch(
                "scripts.scan_email_report.load_return_model_batch_from_urls",
                return_value={"models": {"AAPL": {}, "MSFT": {}}},
            ) as mock_load_batch,
            patch("scripts.scan_email_report.live_scoring_context") as mock_context,
            patch("scripts.scan_email_report.get_stock_data", return_value=fake_price_data),
            patch("scripts.scan_email_report.fast_screen_score", return_value=(90, None)),
            patch("scripts.scan_email_report.analyze_stock", side_effect=lambda ticker, **kw: _fake_analysis(ticker)) as mock_analyze,
        ):
            mock_context.return_value.__enter__ = lambda self: None
            mock_context.return_value.__exit__ = lambda self, *exc: None
            partial = scan_email_report.run_scan_shard("short_term", manifest, pd.DataFrame(), 0, 2)

        mock_load_batch.assert_called_once_with(["https://example.com/shard-0.joblib"])
        self.assertEqual(mock_analyze.call_count, 2)
        self.assertEqual(sorted(call.args[0] for call in mock_analyze.call_args_list), ["AAPL", "MSFT"])
        # Both the screener-style and profit-opportunities rows must be
        # derivable from that single shared analysis per ticker.
        self.assertEqual(sorted(partial["screener"]["Ticker"]), ["AAPL", "MSFT"])
        self.assertEqual(sorted(partial["profit"]["Ticker"]), ["AAPL", "MSFT"])
        self.assertEqual(partial["screener_attrs"]["fully_analyzed_count"], 2)
        self.assertEqual(partial["screener_attrs"]["fast_filtered_count"], 0)
        self.assertEqual(partial["screener_attrs"]["failed_count"], 0)
        self.assertEqual(partial["screener_attrs"]["source_ticker_count"], 2)
        self.assertEqual(partial["profit_stats_base"]["scanned_count"], 2)
        self.assertEqual(partial["profit_stats_base"]["passed_fast_screen_count"], 2)
        self.assertEqual(partial["profit_stats_base"]["lightgbm_unconfirmed_count"], 0)

    def test_ticker_failing_both_fast_screens_skips_analyze_stock(self):
        # Skipping analyze_stock entirely when a ticker fails *both* the
        # screener's (35) and profit-opportunities' (15) thresholds is the
        # other half of the Phase 2 optimization -- verify it doesn't
        # silently start calling analyze_stock for clearly-unqualified tickers.
        manifest = _manifest_with_shards(2, {"AAPL": 0})
        with (
            patch("scripts.scan_email_report.load_return_model_batch_from_urls", return_value={"models": {"AAPL": {}}}),
            patch("scripts.scan_email_report.live_scoring_context") as mock_context,
            patch("scripts.scan_email_report.get_stock_data", return_value=pd.DataFrame({"Close": [1.0]})),
            patch("scripts.scan_email_report.fast_screen_score", return_value=(5, None)),
            patch("scripts.scan_email_report.analyze_stock") as mock_analyze,
        ):
            mock_context.return_value.__enter__ = lambda self: None
            mock_context.return_value.__exit__ = lambda self, *exc: None
            partial = scan_email_report.run_scan_shard("short_term", manifest, pd.DataFrame(), 0, 2)

        mock_analyze.assert_not_called()
        self.assertTrue(partial["screener"].empty)
        self.assertTrue(partial["profit"].empty)
        self.assertEqual(partial["screener_attrs"]["fast_filtered_count"], 1)
        self.assertEqual(partial["profit_stats_base"]["fast_filtered_count"], 1)

    def test_ticker_passing_only_profit_threshold_builds_profit_row_only(self):
        # A fast-score of 20 clears the profit-opportunities threshold (15)
        # but not the stricter screener threshold (35) -- both are
        # independent, so only the profit row should be built even though
        # analyze_stock still only runs once.
        manifest = _manifest_with_shards(2, {"AAPL": 0})
        with (
            patch("scripts.scan_email_report.load_return_model_batch_from_urls", return_value={"models": {"AAPL": {}}}),
            patch("scripts.scan_email_report.live_scoring_context") as mock_context,
            patch("scripts.scan_email_report.get_stock_data", return_value=pd.DataFrame({"Close": [1.0]})),
            patch("scripts.scan_email_report.fast_screen_score", return_value=(20, None)),
            patch("scripts.scan_email_report.analyze_stock", side_effect=lambda ticker, **kw: _fake_analysis(ticker)) as mock_analyze,
        ):
            mock_context.return_value.__enter__ = lambda self: None
            mock_context.return_value.__exit__ = lambda self, *exc: None
            partial = scan_email_report.run_scan_shard("short_term", manifest, pd.DataFrame(), 0, 2)

        mock_analyze.assert_called_once()
        self.assertTrue(partial["screener"].empty)
        self.assertEqual(list(partial["profit"]["Ticker"]), ["AAPL"])
        self.assertEqual(partial["screener_attrs"]["fast_filtered_count"], 1)
        self.assertEqual(partial["screener_attrs"]["fully_analyzed_count"], 0)
        self.assertEqual(partial["profit_stats_base"]["fast_filtered_count"], 0)
        self.assertEqual(partial["profit_stats_base"]["passed_fast_screen_count"], 1)

    def test_raises_when_batch_is_empty(self):
        manifest = _manifest_with_shards(2, {"AAPL": 0})
        with (
            patch("scripts.scan_email_report.load_return_model_batch_from_urls", return_value={"models": {}}),
            self.assertRaises(RuntimeError),
        ):
            scan_email_report.run_scan_shard("short_term", manifest, pd.DataFrame(), 0, 2)


class MergeShardOutputsTests(unittest.TestCase):
    def test_merges_frames_and_sums_stats_across_partial_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            partial_dir = Path(tmp_dir)
            joblib.dump(
                {
                    "shard_index": 0,
                    "screener": pd.DataFrame({"Ticker": ["AAPL"], "Score": [80]}),
                    "screener_attrs": {"fast_filtered_count": 1, "fully_analyzed_count": 1, "failed_count": 0, "source_ticker_count": 2},
                    "profit": pd.DataFrame({"Ticker": ["AAPL"], "_rsi": [55.0]}),
                    "profit_stats_base": {"scanned_count": 2, "fast_filtered_count": 1, "passed_fast_screen_count": 1, "failed_count": 0},
                },
                partial_dir / "shard-0.joblib",
            )
            joblib.dump(
                {
                    "shard_index": 1,
                    "screener": pd.DataFrame({"Ticker": ["MSFT"], "Score": [90]}),
                    "screener_attrs": {"fast_filtered_count": 0, "fully_analyzed_count": 1, "failed_count": 1, "source_ticker_count": 2},
                    "profit": pd.DataFrame(),
                    "profit_stats_base": {"scanned_count": 2, "fast_filtered_count": 0, "passed_fast_screen_count": 1, "failed_count": 1},
                },
                partial_dir / "shard-1.joblib",
            )

            merged = scan_email_report.merge_shard_outputs(partial_dir)

        self.assertEqual(len(merged["screener_frames"]), 2)
        self.assertEqual(len(merged["profit_frames"]), 1)
        self.assertEqual(
            merged["screener_attrs_totals"],
            {"fast_filtered_count": 1, "fully_analyzed_count": 2, "failed_count": 1, "source_ticker_count": 4},
        )
        self.assertEqual(
            merged["profit_stats_totals"],
            {"scanned_count": 4, "fast_filtered_count": 1, "passed_fast_screen_count": 2, "failed_count": 1, "lightgbm_unconfirmed_count": 0},
        )

    def test_raises_when_no_partial_files_present(self):
        with tempfile.TemporaryDirectory() as tmp_dir, self.assertRaises(RuntimeError):
            scan_email_report.merge_shard_outputs(Path(tmp_dir))


class ReduceScanShardsTests(unittest.TestCase):
    def test_truncates_merged_frames_records_once_and_sends_single_email(self):
        manifest = {"tickers": {"AAPL": {}, "MSFT": {}}, "latest_batch": {}}
        screener_frame = pd.DataFrame({"Ticker": ["AAPL", "MSFT"], "Score": [70, 90]})
        profit_frame = pd.DataFrame(
            {
                "Ticker": ["AAPL", "MSFT"],
                "Score": [60, 80],
                "Projected Upside %": [10.0, 20.0],
                "_rsi": [55.0, 60.0],
            }
        )
        merged = {
            "screener_frames": [screener_frame],
            "screener_attrs_totals": {"fast_filtered_count": 1, "fully_analyzed_count": 2, "failed_count": 0, "source_ticker_count": 2},
            "profit_frames": [profit_frame],
            "profit_stats_totals": {
                "scanned_count": 2,
                "fast_filtered_count": 1,
                "passed_fast_screen_count": 2,
                "failed_count": 0,
                "lightgbm_unconfirmed_count": 0,
            },
        }
        finalized_display = profit_frame.drop(columns=["_rsi"])
        finalized_stats = {"scanned_count": 2, "recorded_count": 2, "qualified_count": 2}

        with (
            patch("scripts.scan_email_report.merge_shard_outputs", return_value=merged) as mock_merge,
            patch(
                "scripts.scan_email_report.finalize_profit_results",
                return_value=(finalized_display, finalized_stats),
            ) as mock_finalize,
            patch("scripts.scan_email_report.build_scan_report", return_value="<html></html>") as mock_build_report,
            patch("scripts.scan_email_report.send_brevo_email", return_value=3) as mock_send_email,
        ):
            exit_code = scan_email_report.reduce_scan_shards("short_term", manifest, Path("/tmp/does-not-matter"), "2026-01-01")

        self.assertEqual(exit_code, 0)
        mock_merge.assert_called_once()
        # finalize_profit_results is called three times: once (record=True) on
        # the deduplicated union of both cuts to record predictions exactly
        # once per ticker, and once each (record=False) for the by-upside and
        # by-score display/CSV frames.
        self.assertEqual(mock_finalize.call_count, 3)
        record_flags = [call.kwargs["record"] for call in mock_finalize.call_args_list]
        self.assertEqual(record_flags.count(True), 1)
        self.assertEqual(record_flags.count(False), 2)
        mock_build_report.assert_called_once()
        mock_send_email.assert_called_once()
        self.assertIn("2026-01-01", mock_send_email.call_args.kwargs["subject"])
        attachments = mock_send_email.call_args.kwargs["attachments"]
        self.assertEqual(len(attachments), 3)
        attachment_names = [attachment["name"] for attachment in attachments]
        self.assertIn("screener_top75_2026-01-01.csv", attachment_names)
        self.assertIn("profit_opportunities_by_upside_top75_2026-01-01.csv", attachment_names)
        self.assertIn("profit_opportunities_by_score_top75_2026-01-01.csv", attachment_names)

    def test_handles_all_empty_shards_without_raising(self):
        manifest = {"tickers": {}, "latest_batch": {}}
        merged = {
            "screener_frames": [],
            "screener_attrs_totals": {"fast_filtered_count": 0, "fully_analyzed_count": 0, "failed_count": 0, "source_ticker_count": 0},
            "profit_frames": [],
            "profit_stats_totals": {
                "scanned_count": 0,
                "fast_filtered_count": 0,
                "passed_fast_screen_count": 0,
                "failed_count": 0,
                "lightgbm_unconfirmed_count": 0,
            },
        }
        with (
            patch("scripts.scan_email_report.merge_shard_outputs", return_value=merged),
            patch("scripts.scan_email_report.finalize_profit_results", return_value=(pd.DataFrame(), {"recorded_count": 0})),
            patch("scripts.scan_email_report.build_scan_report", return_value="<html></html>"),
            patch("scripts.scan_email_report.send_brevo_email", return_value=0) as mock_send_email,
        ):
            exit_code = scan_email_report.reduce_scan_shards("short_term", manifest, Path("/tmp/does-not-matter"), "2026-01-01")
        self.assertEqual(exit_code, 0)
        mock_send_email.assert_called_once()


class DiscoverScanInputsTests(unittest.TestCase):
    def test_writes_manifest_and_macro_and_returns_shard_info(self):
        manifest = _manifest_with_shards(4, {"AAPL": 0, "MSFT": 1})
        macro_table = pd.DataFrame({"vix": [15.0]})

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_manifest = Path(tmp_dir) / "manifest.json"
            output_macro = Path(tmp_dir) / "macro.joblib"
            with (
                patch("scripts.scan_email_report.fetch_live_manifest", return_value=manifest),
                patch("scripts.scan_email_report.resolve_release_repository", return_value="owner/repo"),
                patch("scripts.scan_email_report.fetch_shared_macro_table", return_value=macro_table),
            ):
                info = scan_email_report.discover_scan_inputs(None, output_manifest, output_macro)

            self.assertEqual(info, {"tickers": ["AAPL", "MSFT"], "shard_count": 4})
            self.assertEqual(json.loads(output_manifest.read_text(encoding="utf-8")), manifest)
            loaded_macro = joblib.load(output_macro)
        pd.testing.assert_frame_equal(loaded_macro, macro_table)

    def test_raises_when_manifest_has_no_tickers(self):
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch("scripts.scan_email_report.fetch_live_manifest", return_value={"tickers": {}}),
            patch("scripts.scan_email_report.resolve_release_repository", return_value="owner/repo"),
            self.assertRaises(RuntimeError),
        ):
            scan_email_report.discover_scan_inputs(
                None, Path(tmp_dir) / "manifest.json", Path(tmp_dir) / "macro.joblib"
            )


class BuildParserSubcommandTests(unittest.TestCase):
    def test_discover_subcommand_parses(self):
        args = scan_email_report.build_parser().parse_args(
            ["discover", "--output-manifest", "/tmp/m.json", "--output-macro", "/tmp/m.joblib"]
        )
        self.assertEqual(args.command, "discover")

    def test_scan_shard_subcommand_parses(self):
        args = scan_email_report.build_parser().parse_args(
            [
                "scan-shard",
                "--horizon",
                "short_term",
                "--manifest-file",
                "/tmp/m.json",
                "--macro-file",
                "/tmp/m.joblib",
                "--shard-index",
                "1",
                "--shard-count",
                "4",
                "--output",
                "/tmp/out.joblib",
            ]
        )
        self.assertEqual(args.command, "scan-shard")
        self.assertEqual(args.shard_index, 1)
        self.assertEqual(args.shard_count, 4)

    def test_reduce_subcommand_parses(self):
        args = scan_email_report.build_parser().parse_args(
            [
                "reduce",
                "--horizon",
                "medium_term",
                "--manifest-file",
                "/tmp/m.json",
                "--partial-dir",
                "/tmp/partials",
            ]
        )
        self.assertEqual(args.command, "reduce")
        self.assertIsNone(args.run_date)

    def test_main_dispatches_to_reduce_handler(self):
        with (
            patch("scripts.scan_email_report._run_reduce", return_value=0) as mock_reduce,
            patch("scripts.scan_email_report._run_discover") as mock_discover,
            patch("scripts.scan_email_report._run_scan_shard") as mock_scan_shard,
        ):
            exit_code = scan_email_report.main(
                ["reduce", "--horizon", "short_term", "--manifest-file", "/tmp/m.json", "--partial-dir", "/tmp/p"]
            )
        self.assertEqual(exit_code, 0)
        mock_reduce.assert_called_once()
        mock_discover.assert_not_called()
        mock_scan_shard.assert_not_called()


if __name__ == "__main__":
    unittest.main()
