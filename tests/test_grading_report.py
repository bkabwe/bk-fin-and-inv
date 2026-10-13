from __future__ import annotations

import base64
import csv
import io
import unittest
from datetime import date, timedelta
from unittest.mock import patch

from scripts import grading_report

# Gmail clips messages larger than ~102KB (same limit as tests/test_scan_report_html.py).
GMAIL_CLIP_LIMIT_BYTES = 102_000


def _row(ticker: str, **overrides):
    row = {
        "Ticker": ticker,
        "Company": f"{ticker} Inc",
        "Scan Price": 0.124,
        "Target Price": 0.2,
        "Projected Upside %": 61.29,
        "Target Date": "2026-10-30",
        "Actual @ Target": 0.18,
        "Actual Return %": 45.16,
        "Point Hit": False,
        "Band Hit": True,
        "Max Price Since Scan": 0.21,
        "Max Return %": 69.35,
        "Max Hit": True,
        "Max Band Hit": True,
        "Abs Error %": 16.13,
        "Status": "resolved",
    }
    row.update(overrides)
    return row


def _score_stats(count: int = 40, **overrides):
    stats = {"count": count, "information_coefficient": 0.123, "precision_at_top_third": 61.0, "precision_at_bottom_third": 38.0}
    stats.update(overrides)
    return stats


def _upside_stats(count: int = 40, **overrides):
    stats = {
        "count": count,
        "information_coefficient": 0.05,
        "mean_bias_pct": -4.2,
        "median_bias_pct": -3.1,
        "overoptimism_rate": 64.0,
    }
    stats.update(overrides)
    return stats


class BuildGradingReportTests(unittest.TestCase):
    def _report(
        self,
        rows=None,
        summary=None,
        score_validation=None,
        upside_validation=None,
        horizon="short_term",
        validation_since=None,
    ):
        rows = [_row("AAA"), _row("BBB", **{"Actual Return %": -3.2})] if rows is None else rows
        summary = summary if summary is not None else grading_report.compute_batch_summary(rows)
        return grading_report.build_grading_report(
            horizon,
            date(2026, 9, 4),
            date(2026, 10, 2),
            rows,
            summary,
            score_validation,
            upside_validation,
            validation_since=validation_since,
        )

    def test_renders_the_dashboard_with_batch_highlights_and_graded_table(self):
        html_out = self._report()
        self.assertIn("Short-Term (1–4 weeks) grading report", html_out)
        self.assertIn("Batch highlights", html_out)
        self.assertIn("Graded predictions", html_out)
        self.assertIn("Point hit rate", html_out)
        self.assertIn("Best performer:", html_out)
        self.assertIn("max-width:1200px", html_out)
        self.assertEqual(html_out.count('class="bk-table"'), 1)

    def test_does_not_use_the_legacy_light_text_palette(self):
        html_out = self._report()
        for legacy in ("#f4fff8", "#a9bdd7", "#dce8f7"):
            self.assertNotIn(legacy, html_out)

    def test_sub_dollar_prices_keep_their_extra_decimals(self):
        html_out = self._report()
        self.assertIn("$0.1240", html_out)
        self.assertIn("$0.2000", html_out)

    def test_validation_sections_only_appear_when_there_is_data(self):
        without = self._report()
        self.assertNotIn("Score validation", without)
        self.assertNotIn("Upside forecast validation", without)

        with_stats = self._report(
            score_validation={"by_horizon": {"short_term": _score_stats()}},
            upside_validation={"by_horizon": {"short_term": _upside_stats()}},
        )
        self.assertIn("Score validation (all-time, this horizon)", with_stats)
        self.assertIn("Upside forecast validation (all-time, this horizon)", with_stats)
        self.assertIn("0.123", with_stats)
        self.assertIn("Over-optimism rate", with_stats)
        self.assertIn("resolved Short-Term (1–4 weeks) predictions to date", with_stats)

    def test_validation_panels_use_the_reports_own_horizon_not_the_blended_overall(self):
        score = {
            "overall": _score_stats(count=99, information_coefficient=0.999),
            "by_horizon": {
                "short_term": _score_stats(count=40, information_coefficient=0.123),
                "medium_term": _score_stats(count=7, information_coefficient=0.456),
            },
        }
        upside = {
            "overall": _upside_stats(count=98),
            "by_horizon": {"short_term": _upside_stats(count=41), "medium_term": _upside_stats(count=8)},
        }
        html_out = self._report(score_validation=score, upside_validation=upside)
        self.assertIn("all 40 resolved", html_out)
        self.assertIn("all 41 resolved", html_out)
        self.assertIn("0.123", html_out)
        for other_horizon_text in ("0.999", "0.456", "all 99 resolved", "all 98 resolved", "all 7 resolved", "all 8 resolved"):
            self.assertNotIn(other_horizon_text, html_out)

    def test_no_validation_panel_when_the_report_horizon_has_no_resolved_rows(self):
        score = {"overall": _score_stats(), "by_horizon": {"short_term": _score_stats(), "medium_term": _score_stats(count=0)}}
        upside = {"overall": _upside_stats(), "by_horizon": {"short_term": _upside_stats(), "medium_term": _upside_stats(count=0)}}
        html_out = self._report(horizon="medium_term", score_validation=score, upside_validation=upside)
        self.assertNotIn("Score validation", html_out)
        self.assertNotIn("Upside forecast validation", html_out)

    def test_validation_panels_name_the_scan_cutoff_when_one_is_given(self):
        html_out = self._report(
            score_validation={"by_horizon": {"short_term": _score_stats()}},
            upside_validation={"by_horizon": {"short_term": _upside_stats()}},
            validation_since=date(2026, 10, 2),
        )
        self.assertIn("Score validation (since 2026-10-02, this horizon)", html_out)
        self.assertIn("Upside forecast validation (since 2026-10-02, this horizon)", html_out)
        self.assertIn("predictions scanned since 2026-10-02 (the current scoring model)", html_out)
        self.assertNotIn("all-time", html_out)

    def test_table_previews_the_first_rows_and_points_to_the_csv(self):
        rows = [_row(f"AAA{i:02d}") for i in range(25)]
        html_out = self._report(rows=rows)
        self.assertIn("AAA09", html_out)
        self.assertNotIn("AAA10", html_out)
        self.assertIn("Showing the first 10 of 25 graded predictions; the full list is attached as CSV.", html_out)
        self.assertIn("tracked batch size 25", html_out)

    def test_batches_within_the_preview_limit_are_shown_in_full(self):
        rows = [_row(f"BBB{i:02d}") for i in range(grading_report.PREVIEW_LIMIT)]
        html_out = self._report(rows=rows)
        self.assertIn("BBB09", html_out)
        self.assertIn("The full list is also attached as CSV.", html_out)
        self.assertNotIn("Showing the first", html_out)

    def test_a_full_batch_report_fits_under_the_gmail_clip_limit(self):
        rows = [_row(f"TK{i:03d}", **{"Actual Return %": 123.456 - i}) for i in range(grading_report.GRADED_BATCH_LIMIT)]
        html_out = self._report(
            rows=rows,
            score_validation={"by_horizon": {"short_term": _score_stats()}},
            upside_validation={"by_horizon": {"short_term": _upside_stats()}},
            validation_since=date(2026, 10, 2),
        )
        self.assertLess(len(html_out.encode("utf-8")), GMAIL_CLIP_LIMIT_BYTES)

    def test_escapes_ticker_text_in_the_best_and_worst_performer_line(self):
        rows = [_row("<b>X</b>", **{"Actual Return %": 80.0}), _row("YYY", **{"Actual Return %": -5.0})]
        html_out = self._report(rows=rows)
        self.assertNotIn("<b>X</b>", html_out)
        self.assertIn("&lt;b&gt;X&lt;/b&gt;", html_out)

    def test_handles_an_empty_batch(self):
        html_out = self._report(rows=[], summary=grading_report.compute_batch_summary([]))
        self.assertIn("No rows available.", html_out)
        self.assertNotIn("Best performer:", html_out)
        self.assertNotIn("attached as CSV", html_out)


class BuildGradedCsvTests(unittest.TestCase):
    def test_includes_every_row_with_raw_values(self):
        rows = [_row("AAA"), _row("BBB", **{"Actual Return %": None, "Status": "pending"})]
        parsed = list(csv.DictReader(io.StringIO(grading_report.build_graded_csv(rows))))
        self.assertEqual([row["Ticker"] for row in parsed], ["AAA", "BBB"])
        self.assertEqual(list(parsed[0]), list(rows[0]))
        self.assertEqual(parsed[0]["Scan Price"], "0.124")
        self.assertEqual(parsed[1]["Actual Return %"], "")
        self.assertEqual(parsed[1]["Status"], "pending")

    def test_is_empty_without_rows(self):
        self.assertEqual(grading_report.build_graded_csv([]), "")


class BuildGradingRowsTests(unittest.TestCase):
    def _max_price_end_date(self, target_date, grading_date):
        record = {
            "ticker": "AAA",
            "scan_date": "2026-10-05",
            "target_date": target_date,
            "current_price_at_scan": 100.0,
            "target_price": 110.0,
            "status": "resolved",
        }
        with patch.object(grading_report, "compute_max_price_since_scan", return_value=105.0) as max_price:
            grading_report.build_grading_rows([record], grading_date)
        ticker, scan_date, end_date = max_price.call_args.args
        self.assertEqual((ticker, scan_date), ("AAA", "2026-10-05"))
        return end_date

    def test_max_price_window_stops_at_the_target_date_when_graded_late(self):
        # Medium-term batches are graded weeks after their target date.
        self.assertEqual(self._max_price_end_date("2027-01-30", date(2027, 3, 1)), date(2027, 1, 30))

    def test_max_price_window_stops_at_the_grading_date_when_the_target_is_later(self):
        self.assertEqual(self._max_price_end_date("2026-10-26", date(2026, 10, 20)), date(2026, 10, 20))

    def test_max_price_window_falls_back_to_the_grading_date_without_a_usable_target_date(self):
        for target_date in (None, "", "not-a-date"):
            with self.subTest(target_date=target_date):
                self.assertEqual(self._max_price_end_date(target_date, date(2026, 10, 26)), date(2026, 10, 26))


class ScanWindowTests(unittest.TestCase):
    def test_short_term_window_is_the_week_of_scans_that_just_reached_their_horizon(self):
        self.assertEqual(grading_report._scan_window("short_term", date(2026, 11, 2)), (date(2026, 10, 6), date(2026, 10, 12)))

    def test_window_never_reaches_back_before_the_current_model(self):
        self.assertEqual(grading_report._scan_window("short_term", date(2026, 10, 26)), (date(2026, 10, 5), date(2026, 10, 5)))
        self.assertEqual(grading_report._scan_window("medium_term", date(2027, 3, 1)), (date(2026, 10, 5), date(2026, 12, 1)))

    def test_windows_that_only_cover_earlier_models_are_empty(self):
        self.assertIsNone(grading_report._scan_window("short_term", date(2026, 10, 19)))  # scans 09-22..09-28
        self.assertIsNone(grading_report._scan_window("medium_term", date(2026, 11, 1)))  # scans up to 08-03
        self.assertIsNone(grading_report._scan_window("medium_term", date(2027, 1, 1)))  # scans 08-03..10-03


class SchedulingTests(unittest.TestCase):
    """The grading and scan workflows share their crons, so the lookup window
    has to grade every scheduled scan exactly once."""

    @staticmethod
    def _mondays(start: date, end: date):
        day = start + timedelta(days=(7 - start.weekday()) % 7)
        while day <= end:
            yield day
            day += timedelta(days=7)

    @staticmethod
    def _first_of_odd_months(start: date, end: date):
        for year in range(start.year, end.year + 1):
            for month in (1, 3, 5, 7, 9, 11):
                if start <= date(year, month, 1) <= end:
                    yield date(year, month, 1)

    def _times_each_scan_is_graded(self, horizon, scans, grading_days):
        graded = dict.fromkeys(scans, 0)
        for grading_day in grading_days:
            window = grading_report._scan_window(horizon, grading_day)
            in_window = [scan for scan in scans if window and window[0] <= scan <= window[1]]
            if in_window:
                graded[max(in_window)] += 1  # find_grading_batch picks the newest scan
        return graded

    def test_every_scheduled_short_term_scan_is_graded_once(self):
        scans = list(self._mondays(date(2026, 10, 5), date(2028, 12, 25)))
        graded = self._times_each_scan_is_graded("short_term", scans, self._mondays(date(2026, 10, 5), date(2029, 3, 31)))
        self.assertEqual(set(graded.values()), {1})

    def test_every_scheduled_medium_term_scan_is_graded_once(self):
        scans = list(self._first_of_odd_months(date(2026, 11, 1), date(2028, 11, 1)))
        graded = self._times_each_scan_is_graded(
            "medium_term", scans, self._first_of_odd_months(date(2026, 11, 1), date(2029, 6, 1))
        )
        self.assertEqual(set(graded.values()), {1})


def _prediction(scan_date, ticker, horizon="short_term", source="profit_opportunities", **overrides):
    record = {
        "ticker": ticker,
        "company": f"{ticker} Inc",
        "scan_date": scan_date,
        "horizon": horizon,
        "source": source,
        "current_price_at_scan": 100.0,
        "target_price": 110.0,
        "target_low": 104.0,
        "projected_upside_pct": 10.0,
        "target_date": "2026-10-26",
        "actual_price_at_target_date": 108.0,
        "actual_return_pct": 8.0,
        "hit_target": False,
        "hit_band": True,
        "status": "resolved",
    }
    record.update(overrides)
    return record


class FindGradingBatchTests(unittest.TestCase):
    def _find(self, records, horizon, today):
        with patch.object(grading_report, "get_all_predictions", return_value=records):
            return grading_report.find_grading_batch(horizon, today)

    def test_picks_the_newest_batch_in_the_window(self):
        records = [_prediction("2026-10-13", "SCHEDULED"), _prediction("2026-10-15", "MANUAL1"), _prediction("2026-10-15", "MANUAL2")]
        scan_date, batch = self._find(records, "short_term", date(2026, 11, 9))  # window 10-13..10-19
        self.assertEqual(scan_date, date(2026, 10, 15))
        self.assertEqual([record["ticker"] for record in batch], ["MANUAL1", "MANUAL2"])

    def test_ignores_other_horizons_sources_and_batches_outside_the_window(self):
        records = [
            _prediction("2026-10-05", "MEDIUM", horizon="medium_term"),
            _prediction("2026-10-05", "SCREENER", source="screener"),
            _prediction("2026-09-28", "LEGACY"),  # before the current model
            _prediction("2026-10-02", "MANUAL"),  # manual run from before the cutoff
            _prediction("2026-10-12", "NOTYET"),  # not at its horizon on 10-26
        ]
        self.assertIsNone(self._find(records, "short_term", date(2026, 10, 26)))

    def test_leaves_batches_an_earlier_run_already_covered_alone(self):
        records = [_prediction("2026-10-06", "AAA")]
        self.assertIsNone(self._find(records, "short_term", date(2026, 11, 9)))  # window 10-13..10-19
        self.assertIsNotNone(self._find(records, "short_term", date(2026, 11, 2)))  # window 10-06..10-12

    def test_matches_scan_dates_that_carry_a_time(self):
        scan_date, batch = self._find([_prediction("2026-10-05T14:30:00+00:00", "AAA")], "short_term", date(2026, 10, 26))
        self.assertEqual(scan_date, date(2026, 10, 5))
        self.assertEqual(len(batch), 1)

    def test_medium_term_picks_the_scan_that_matured_since_the_last_run(self):
        records = [_prediction("2026-11-01", "FIRST", horizon="medium_term"), _prediction("2027-01-01", "SECOND", horizon="medium_term")]
        self.assertIsNone(self._find(records, "medium_term", date(2027, 1, 1)))
        self.assertEqual(self._find(records, "medium_term", date(2027, 3, 1))[0], date(2026, 11, 1))
        self.assertEqual(self._find(records, "medium_term", date(2027, 5, 1))[0], date(2027, 1, 1))


class GenerateGradingReportTests(unittest.TestCase):
    def _run(self, records, horizon, today, send_side_effect=None):
        with (
            patch.object(grading_report, "resolve_pending_predictions", return_value={"resolved": 0}) as resolve,
            patch.object(grading_report, "get_all_predictions", return_value=records),
            patch.object(grading_report, "compute_max_price_since_scan", return_value=112.0),
            patch.object(grading_report, "compute_score_validation_stats", return_value={}) as score_stats,
            patch.object(grading_report, "compute_upside_validation_stats", return_value={}) as upside_stats,
            patch.object(grading_report, "send_brevo_email", return_value=2, side_effect=send_side_effect) as send,
        ):
            code = grading_report.generate_grading_report(horizon, today=today)
        return code, resolve, send, score_stats, upside_stats

    def test_is_a_no_op_when_the_only_matured_scans_predate_the_current_model(self):
        records = [
            _prediction("2026-09-21", "LEGACY1"),
            _prediction("2026-09-28", "LEGACY2"),
            _prediction("2026-10-05", "NOTYET"),
            _prediction("2026-10-12", "NOTYET2"),
        ]
        code, resolve, send, _score, _upside = self._run(records, "short_term", date(2026, 10, 19))
        self.assertEqual(code, 0)
        send.assert_not_called()
        resolve.assert_called_once_with()  # pending predictions are still resolved

    def test_is_a_no_op_when_the_window_has_no_recorded_scan(self):
        code, _resolve, send, _score, _upside = self._run([_prediction("2026-10-05", "AAA")], "short_term", date(2026, 11, 9))
        self.assertEqual(code, 0)
        send.assert_not_called()

    def test_medium_term_is_a_no_op_until_a_current_model_scan_has_matured(self):
        records = [_prediction("2026-10-02", "MANUAL", horizon="medium_term")]
        for today in (date(2026, 11, 1), date(2027, 1, 1)):
            with self.subTest(today=today):
                code, _resolve, send, _score, _upside = self._run(records, "medium_term", today)
                self.assertEqual(code, 0)
                send.assert_not_called()

    def test_grades_the_matured_scan_and_attaches_every_graded_row_as_csv(self):
        records = (
            [_prediction("2026-10-05", f"T{i:02d}") for i in range(12)]
            + [_prediction("2026-10-02", "MANUAL")]
            + [_prediction("2026-10-12", "NOTYET")]
        )
        code, _resolve, send, _score, _upside = self._run(records, "short_term", date(2026, 10, 26))
        self.assertEqual(code, 0)
        send.assert_called_once()
        kwargs = send.call_args.kwargs
        self.assertIn("2026-10-05", kwargs["subject"])
        self.assertIn("Showing the first 10 of 12 graded predictions", kwargs["html_content"])
        self.assertEqual(len(kwargs["attachments"]), 1)
        attachment = kwargs["attachments"][0]
        self.assertEqual(attachment["name"], "graded_predictions_short_term_2026-10-05.csv")
        parsed = list(csv.DictReader(io.StringIO(base64.b64decode(attachment["content"]).decode("utf-8"))))
        self.assertEqual([row["Ticker"] for row in parsed], [f"T{i:02d}" for i in range(12)])

    def test_grades_at_most_the_first_75_recorded_rows(self):
        records = [_prediction("2026-10-05", f"T{i:02d}") for i in range(80)]
        _code, _resolve, send, _score, _upside = self._run(records, "short_term", date(2026, 10, 26))
        parsed = list(csv.DictReader(io.StringIO(base64.b64decode(send.call_args.kwargs["attachments"][0]["content"]).decode("utf-8"))))
        self.assertEqual(len(parsed), grading_report.GRADED_BATCH_LIMIT)
        self.assertEqual(parsed[-1]["Ticker"], "T74")

    def test_validation_stats_only_count_scans_from_the_current_model(self):
        records = [_prediction("2026-10-05", "AAA")]
        _code, _resolve, _send, score_stats, upside_stats = self._run(records, "short_term", date(2026, 10, 26))
        score_stats.assert_called_once_with(since=grading_report.CURRENT_MODEL_START_DATE)
        upside_stats.assert_called_once_with(since=grading_report.CURRENT_MODEL_START_DATE)

    def test_email_failures_still_fail_the_run(self):
        with self.assertRaises(RuntimeError):
            self._run([_prediction("2026-10-05", "AAA")], "short_term", date(2026, 10, 26), send_side_effect=RuntimeError("brevo down"))


if __name__ == "__main__":
    unittest.main()
