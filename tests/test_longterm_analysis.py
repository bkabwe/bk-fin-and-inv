from __future__ import annotations

import unittest
from unittest.mock import call, patch

import numpy as np
import pandas as pd

from modules import longterm_analysis


def _history_frame(
    close: np.ndarray | list[float],
    volume: np.ndarray | list[float] | None = None,
    *,
    start: str = "2020-01-02",
    freq: str = "B",
) -> pd.DataFrame:
    index = pd.date_range(start, periods=len(close), freq=freq)
    frame = {"Close": np.asarray(close, dtype=float)}
    if volume is not None:
        frame["Volume"] = np.asarray(volume, dtype=float)
    return pd.DataFrame(frame, index=index)


def _benchmark_series(index: pd.Index, start: float, end: float) -> pd.Series:
    return pd.Series(np.linspace(start, end, num=len(index)), index=index, dtype=float)


class LongTermAnalysisHelperTests(unittest.TestCase):
    def test_safe_last_returns_last_non_nan_value(self):
        self.assertEqual(
            longterm_analysis._safe_last(pd.Series([np.nan, 2.5, np.nan, 3.5])),
            3.5,
        )
        self.assertIsNone(longterm_analysis._safe_last(pd.Series(dtype=float)))
        self.assertIsNone(longterm_analysis._safe_last(None))

    def test_relative_strength_ratio_handles_outperformance_and_flat_benchmark(self):
        index = pd.date_range("2024-01-01", periods=100, freq="D")
        stock = pd.Series(np.linspace(100.0, 130.0, num=len(index)), index=index)
        benchmark = pd.Series(np.linspace(100.0, 110.0, num=len(index)), index=index)
        flat_benchmark = pd.Series(np.full(len(index), 100.0), index=index)

        ratio = longterm_analysis._relative_strength_ratio(stock, benchmark)

        self.assertAlmostEqual(ratio, (1.30 / 1.10), places=6)
        self.assertIsNone(
            longterm_analysis._relative_strength_ratio(stock, flat_benchmark)
        )

    def test_primary_trend_and_stage_classification_cover_major_regimes(self):
        cases = [
            {
                "name": "bull",
                "close": pd.Series(
                    np.linspace(50.0, 200.0, num=600),
                    index=pd.bdate_range("2020-01-02", periods=600),
                ),
                "rs_spy": 1.2,
                "rs_sector": 1.1,
                "trend": "primary_bull",
                "stage": "Stage 2 (Advancing)",
            },
            {
                "name": "bear",
                "close": pd.Series(
                    np.linspace(200.0, 50.0, num=600),
                    index=pd.bdate_range("2020-01-02", periods=600),
                ),
                "rs_spy": 0.8,
                "rs_sector": None,
                "trend": "primary_bear",
                "stage": "Stage 4 (Declining)",
            },
            {
                "name": "basing",
                "close": pd.Series(
                    np.full(600, 100.0),
                    index=pd.bdate_range("2020-01-02", periods=600),
                ),
                "rs_spy": None,
                "rs_sector": None,
                "trend": "consolidation_basing",
                "stage": "Stage 1 (Basing)",
            },
        ]

        for case in cases:
            with self.subTest(case=case["name"]):
                close = case["close"]
                sma150 = float(close.rolling(150).mean().iloc[-1])
                sma200 = float(close.rolling(200).mean().iloc[-1])

                self.assertEqual(
                    longterm_analysis._primary_trend(close, sma150, sma200),
                    case["trend"],
                )
                self.assertEqual(
                    longterm_analysis._classify_stage(
                        close,
                        sma150,
                        sma200,
                        case["rs_spy"],
                        case["rs_sector"],
                    ),
                    case["stage"],
                )

    def test_volume_confirmation_identifies_accumulation_and_distribution(self):
        index = pd.bdate_range("2024-01-02", periods=126)
        rising_close = pd.Series(np.linspace(100.0, 130.0, num=126), index=index)
        falling_close = pd.Series(np.linspace(130.0, 100.0, num=126), index=index)
        rising_volume = pd.Series(
            np.concatenate([np.full(63, 1_000_000.0), np.full(63, 1_100_000.0)]),
            index=index,
        )

        self.assertEqual(
            longterm_analysis._volume_confirmation(rising_close, rising_volume),
            "accumulation",
        )
        self.assertEqual(
            longterm_analysis._volume_confirmation(falling_close, rising_volume),
            "distribution",
        )

    def test_drawdown_resilience_reports_drawdown_and_recovery_days(self):
        close = pd.Series(
            [100.0, 120.0, 90.0, 95.0, 121.0],
            index=pd.date_range("2024-01-01", periods=5, freq="D"),
        )

        max_drawdown_pct, recovery_days = longterm_analysis._drawdown_resilience(close)

        self.assertAlmostEqual(max_drawdown_pct, -25.0)
        self.assertEqual(recovery_days, 2)

    def test_cross_history_counts_local_golden_and_death_crosses(self):
        close = pd.Series(
            np.concatenate(
                [
                    np.full(220, 200.0),
                    np.linspace(200.0, 80.0, 80),
                    np.full(40, 80.0),
                    np.linspace(80.0, 220.0, 120),
                    np.full(80, 220.0),
                    np.linspace(220.0, 120.0, 100),
                ]
            ),
            index=pd.bdate_range("2020-01-02", periods=640),
        )

        with patch(
            "modules.longterm_analysis.get_indicator_series",
            return_value=pd.DataFrame(),
        ) as mock_indicator_series:
            golden_count, death_count, follow_through = (
                longterm_analysis._cross_history(close, "CROSS")
            )

        mock_indicator_series.assert_not_called()
        self.assertEqual(golden_count, 1)
        self.assertEqual(death_count, 2)
        self.assertEqual(follow_through, 100.0)


class AnalyzeLongTermTechnicalScoreTests(unittest.TestCase):
    def test_analyze_longterm_score_clamps_strong_uptrend_to_twenty(self):
        history = _history_frame(
            np.linspace(50.0, 200.0, num=600),
            np.concatenate([np.full(537, 1_000_000.0), np.full(63, 1_200_000.0)]),
        )
        spy_close = _benchmark_series(history.index, 100.0, 140.0)
        sector_close = _benchmark_series(history.index, 100.0, 130.0)

        def _benchmark(symbol: str) -> pd.Series:
            return spy_close if symbol == "SPY" else sector_close

        with (
            patch(
                "modules.longterm_analysis._get_benchmark_close",
                side_effect=_benchmark,
            ) as mock_benchmark,
            patch(
                "modules.longterm_analysis.get_indicator_series",
                return_value=pd.DataFrame(),
            ) as mock_indicator_series,
        ):
            result = longterm_analysis.analyze_longterm_technical_score(
                "UP",
                sector="Technology",
                data=history,
            )

        self.assertIsInstance(result, dict)
        self.assertEqual(result["longterm_technical_score"], 20)
        self.assertEqual(result["primary_trend"], "primary_bull")
        self.assertEqual(result["longterm_stage"], "Stage 2 (Advancing)")
        self.assertEqual(result["volume_trend_confirmation"], "accumulation")
        self.assertEqual(
            mock_benchmark.call_args_list,
            [call("SPY"), call("XLK")],
        )
        mock_indicator_series.assert_not_called()

    def test_analyze_longterm_score_fetches_history_and_skips_unknown_sector_etf(self):
        history = _history_frame(
            np.linspace(200.0, 50.0, num=600),
            np.concatenate([np.full(537, 1_000_000.0), np.full(63, 1_200_000.0)]),
        )
        spy_close = _benchmark_series(history.index, 100.0, 120.0)

        with (
            patch(
                "modules.longterm_analysis.get_stock_data",
                return_value=history,
            ) as mock_get_stock_data,
            patch(
                "modules.longterm_analysis._get_benchmark_close",
                return_value=spy_close,
            ) as mock_benchmark,
            patch(
                "modules.longterm_analysis.get_indicator_series",
                return_value=pd.DataFrame(),
            ) as mock_indicator_series,
        ):
            result = longterm_analysis.analyze_longterm_technical_score(
                "BEAR",
                sector="Unknown Sector",
            )

        mock_get_stock_data.assert_called_once_with(
            "BEAR",
            period="5y",
            interval="1d",
        )
        mock_benchmark.assert_called_once_with("SPY")
        mock_indicator_series.assert_not_called()
        self.assertEqual(result["longterm_technical_score"], 0)
        self.assertEqual(result["primary_trend"], "primary_bear")
        self.assertEqual(result["longterm_stage"], "Stage 4 (Declining)")
        self.assertEqual(result["volume_trend_confirmation"], "distribution")

    def test_analyze_longterm_score_flat_history_without_volume_is_basing(self):
        history = _history_frame(np.full(600, 100.0))

        with (
            patch(
                "modules.longterm_analysis._get_benchmark_close",
                return_value=pd.Series(dtype=float),
            ) as mock_benchmark,
            patch(
                "modules.longterm_analysis.get_indicator_series",
                return_value=pd.DataFrame(),
            ) as mock_indicator_series,
        ):
            result = longterm_analysis.analyze_longterm_technical_score(
                "FLAT",
                data=history,
            )

        mock_benchmark.assert_called_once_with("SPY")
        mock_indicator_series.assert_not_called()
        self.assertEqual(result["longterm_technical_score"], 11)
        self.assertEqual(result["primary_trend"], "consolidation_basing")
        self.assertEqual(result["longterm_stage"], "Stage 1 (Basing)")
        self.assertEqual(
            result["volume_trend_confirmation"],
            "insufficient_data",
        )

    def test_analyze_longterm_score_returns_default_payload_for_empty_inputs(self):
        cases = [
            pd.DataFrame(),
            pd.DataFrame({"Volume": [1_000_000.0, 1_100_000.0]}),
        ]

        for history in cases:
            with self.subTest(columns=tuple(history.columns)):
                with (
                    patch(
                        "modules.longterm_analysis._get_benchmark_close"
                    ) as mock_benchmark,
                    patch(
                        "modules.longterm_analysis.get_indicator_series"
                    ) as mock_indicator_series,
                ):
                    result = longterm_analysis.analyze_longterm_technical_score(
                        "EMPTY",
                        data=history,
                    )

                self.assertEqual(result["longterm_technical_score"], 0)
                self.assertEqual(result["longterm_stage"], "Stage Unknown")
                self.assertEqual(result["primary_trend"], "insufficient_data")
                self.assertEqual(
                    result["volume_trend_confirmation"],
                    "insufficient_data",
                )
                self.assertEqual(result["golden_cross_count"], 0)
                self.assertEqual(result["death_cross_count"], 0)
                mock_benchmark.assert_not_called()
                mock_indicator_series.assert_not_called()

    def test_analyze_longterm_score_short_history_uses_indicator_fallback(self):
        history = _history_frame(
            np.linspace(100.0, 40.0, num=100),
            np.full(100, 1_000_000.0),
        )

        with (
            patch(
                "modules.longterm_analysis._get_benchmark_close",
                return_value=pd.Series(dtype=float),
            ) as mock_benchmark,
            patch(
                "modules.longterm_analysis.get_indicator_series",
                return_value=pd.DataFrame(),
            ) as mock_indicator_series,
        ):
            result = longterm_analysis.analyze_longterm_technical_score(
                "SHORT",
                data=history,
            )

        mock_benchmark.assert_called_once_with("SPY")
        self.assertEqual(
            mock_indicator_series.call_args_list,
            [
                call("sma", "SHORT", window=150),
                call("sma", "SHORT", window=200),
            ],
        )
        self.assertEqual(result["longterm_technical_score"], 0)
        self.assertEqual(result["longterm_stage"], "Stage Unknown")
        self.assertEqual(result["primary_trend"], "insufficient_data")
        self.assertEqual(result["golden_cross_count"], 0)
        self.assertEqual(result["death_cross_count"], 0)

    def test_analyze_longterm_score_nan_heavy_close_series_returns_zero(self):
        close = np.full(260, np.nan)
        close[-10:] = np.linspace(100.0, 40.0, num=10)
        history = _history_frame(close, np.full(260, 1_000_000.0))

        with (
            patch(
                "modules.longterm_analysis._get_benchmark_close",
                return_value=pd.Series(dtype=float),
            ) as mock_benchmark,
            patch(
                "modules.longterm_analysis.get_indicator_series",
                return_value=pd.DataFrame(),
            ) as mock_indicator_series,
        ):
            result = longterm_analysis.analyze_longterm_technical_score(
                "NAN",
                data=history,
            )

        mock_benchmark.assert_called_once_with("SPY")
        self.assertEqual(
            mock_indicator_series.call_args_list,
            [
                call("sma", "NAN", window=150),
                call("sma", "NAN", window=200),
            ],
        )
        self.assertEqual(result["longterm_technical_score"], 0)
        self.assertEqual(result["longterm_stage"], "Stage Unknown")
        self.assertEqual(result["primary_trend"], "insufficient_data")
        self.assertEqual(result["volume_trend_confirmation"], "insufficient_data")
        self.assertEqual(result["max_drawdown_pct"], -60.0)


if __name__ == "__main__":
    unittest.main()
