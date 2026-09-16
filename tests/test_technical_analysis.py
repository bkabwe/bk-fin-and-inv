from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from modules import technical_analysis


def _make_oscillating_trend_frame(length: int = 120, *, break_last_bar: bool = False) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=length, freq="D")
    x = np.arange(length, dtype=float)
    close = 100.0 + (0.4 * x) + (4.0 * np.sin(x / 4.0))
    open_ = close + (0.2 * np.cos(x / 3.0))
    high = np.maximum(open_, close) + 1.5
    low = np.minimum(open_, close) - 1.5
    volume = 1000.0 + (200.0 * np.sin(x / 5.0)) + (5.0 * x)

    if break_last_bar:
        close[-1] = close[-2] - 12.0
        open_[-1] = close[-1] + 1.0
        high[-1] = open_[-1] + 1.0
        low[-1] = close[-1] - 3.0
        volume[-1] = float(np.mean(volume[-20:]) * 2.0)

    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=index,
    )


def _make_gap_frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=30, freq="D")
    close = np.linspace(50.0, 60.0, 30)
    open_ = close - 0.2
    high = close + 0.5
    low = close - 0.5
    volume = np.full(30, 1000.0)

    close[23] = 57.5
    open_[23] = 57.3
    high[23] = 58.0
    low[23] = 57.0

    open_[24] = 60.5
    high[24] = 61.0
    low[24] = 60.0
    close[24] = 60.8
    volume[24] = 1400.0

    for i in range(25, 30):
        close[i] = 61.0 + (0.2 * (i - 24))
        open_[i] = close[i] - 0.2
        high[i] = close[i] + 0.5
        low[i] = close[i] - 0.5

    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=index,
    )


def _make_breakout_frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=40, freq="D")
    close = np.array(
        [
            95.0,
            95.5,
            96.0,
            96.5,
            97.0,
            97.5,
            98.0,
            98.5,
            99.0,
            99.5,
            100.0,
            99.8,
            100.1,
            99.9,
            100.0,
            100.2,
            100.1,
            99.9,
            100.3,
            100.2,
            100.1,
            100.4,
            100.3,
            100.2,
            100.5,
            100.6,
            100.5,
            100.4,
            100.6,
            100.7,
            100.6,
            100.5,
            100.7,
            100.8,
            100.7,
            100.9,
            100.8,
            100.9,
            101.0,
            104.0,
        ],
        dtype=float,
    )
    open_ = close - 0.4
    high = close + 0.8
    low = close - 0.8
    volume = np.full(40, 1000.0)
    volume[-10:] = np.linspace(800.0, 700.0, 10)
    volume[-1] = 3500.0

    open_[-1] = 101.5
    high[-1] = 105.5
    low[-1] = 101.0
    high[-2] = 101.2
    low[-2] = 100.2

    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=index,
    )


def _make_flat_frame(length: int = 60, price: float = 50.0) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=length, freq="D")
    return pd.DataFrame(
        {
            "Open": np.full(length, price),
            "High": np.full(length, price + 1.0),
            "Low": np.full(length, price - 1.0),
            "Close": np.full(length, price),
            "Volume": np.full(length, 1000.0),
        },
        index=index,
    )


def _make_relative_strength_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.date_range("2024-01-01", periods=120, freq="D")
    x = np.arange(120, dtype=float)
    stock_close = 100.0 + (0.2 * x)
    stock_close[-21:] += np.linspace(0.0, 12.0, 21)
    spy_close = 100.0 + (0.15 * x)
    return (
        pd.DataFrame({"Close": stock_close}, index=index),
        pd.DataFrame({"Close": spy_close}, index=index),
    )


class TechnicalAnalysisPatternAndGapTests(unittest.TestCase):
    def test_detect_chart_patterns_finds_bullish_and_bearish_setups(self):
        patterns = technical_analysis.detect_chart_patterns(_make_oscillating_trend_frame())

        self.assertGreaterEqual(len(patterns), 3)
        names = {pattern["name"] for pattern in patterns}
        implications = {pattern["implication"] for pattern in patterns}
        self.assertIn("Double Top", names)
        self.assertIn("Double Bottom", names)
        self.assertIn("bullish", implications)
        self.assertIn("bearish", implications)
        self.assertTrue(any(pattern["target"] is not None for pattern in patterns))

    def test_short_frames_return_safe_defaults_for_chart_gap_and_trend_helpers(self):
        short_frame = _make_flat_frame(length=20)

        self.assertEqual(technical_analysis.detect_chart_patterns(short_frame), [])
        self.assertEqual(technical_analysis.detect_gaps(short_frame), [])
        self.assertEqual(
            technical_analysis.detect_trendlines(short_frame),
            {
                "uptrend_line": {"slope": None, "intercept": None, "current_value": None, "broken": False},
                "downtrend_line": {"slope": None, "intercept": None, "current_value": None, "broken": False},
                "channel_upper": None,
                "channel_lower": None,
                "trendline_break_signal": None,
            },
        )
        self.assertEqual(
            technical_analysis.analyze_volume_quality(short_frame),
            {
                "breakout_volume_confirmed": False,
                "climactic_volume": False,
                "volume_divergence": False,
                "consolidation_volume_declining": False,
            },
        )
        self.assertEqual(
            technical_analysis.analyze_dow_theory(short_frame),
            {
                "primary_trend": "sideways",
                "last_swing_high": None,
                "last_swing_low": None,
                "trend_confirmed": False,
                "hh_hl": False,
                "lh_ll": False,
            },
        )

    def test_detect_gaps_identifies_runaway_gap_and_gap_metadata(self):
        gaps = technical_analysis.detect_gaps(_make_gap_frame())

        self.assertEqual(len(gaps), 1)
        gap = gaps[0]
        self.assertEqual(gap["type"], "Runaway gap")
        self.assertEqual(gap["direction"], "up")
        self.assertEqual(gap["significance"], "medium")
        self.assertEqual(gap["date"], "2024-01-25")
        self.assertAlmostEqual(gap["gap_pct"], 3.48, places=2)
        self.assertAlmostEqual(gap["level"], 59.0, places=2)


class TechnicalAnalysisTrendAndStrengthTests(unittest.TestCase):
    def test_detect_trendlines_flags_bearish_break_below_uptrend(self):
        trendlines = technical_analysis.detect_trendlines(_make_oscillating_trend_frame(break_last_bar=True))

        self.assertGreater(trendlines["uptrend_line"]["slope"], 0.0)
        self.assertTrue(trendlines["uptrend_line"]["broken"])
        self.assertIsNotNone(trendlines["uptrend_line"]["current_value"])
        self.assertIsNotNone(trendlines["channel_upper"])
        self.assertEqual(trendlines["trendline_break_signal"], "bearish_break")

    def test_relative_strength_vs_spy_handles_nans_and_short_252_day_history(self):
        stock, spy = _make_relative_strength_frames()
        stock.iloc[0, 0] = np.nan
        stock.iloc[10, 0] = np.nan
        spy.iloc[2, 0] = np.nan

        result = technical_analysis.relative_strength_vs_spy(stock, spy)

        self.assertGreater(result["rs_20d"], 1.0)
        self.assertGreater(result["rs_50d"], 1.0)
        self.assertIsNone(result["rs_252d"])
        self.assertEqual(result["rs_trend"], "improving")

    def test_relative_strength_vs_spy_returns_neutral_ratio_for_flat_spy(self):
        stock, _ = _make_relative_strength_frames()
        flat_spy = pd.DataFrame({"Close": np.full(len(stock), 100.0)}, index=stock.index)

        result = technical_analysis.relative_strength_vs_spy(stock, flat_spy)

        self.assertEqual(result["rs_20d"], 1.0)
        self.assertEqual(result["rs_50d"], 1.0)
        self.assertIsNone(result["rs_252d"])
        self.assertEqual(result["rs_trend"], "stable")

    def test_analyze_dow_theory_detects_higher_highs_and_higher_lows(self):
        result = technical_analysis.analyze_dow_theory(_make_oscillating_trend_frame())

        self.assertEqual(result["primary_trend"], "uptrend")
        self.assertTrue(result["trend_confirmed"])
        self.assertTrue(result["hh_hl"])
        self.assertFalse(result["lh_ll"])
        self.assertGreater(result["last_swing_high"], result["last_swing_low"])


class TechnicalAnalysisVolumeAndBreakoutTests(unittest.TestCase):
    def test_analyze_volume_quality_detects_breakout_and_climactic_volume(self):
        result = technical_analysis.analyze_volume_quality(_make_breakout_frame())

        self.assertTrue(result["breakout_volume_confirmed"])
        self.assertTrue(result["climactic_volume"])
        self.assertFalse(result["volume_divergence"])
        self.assertFalse(result["consolidation_volume_declining"])

    def test_analyze_volume_quality_detects_declining_consolidation_volume(self):
        result = technical_analysis.analyze_volume_quality(_make_oscillating_trend_frame())

        self.assertFalse(result["breakout_volume_confirmed"])
        self.assertFalse(result["climactic_volume"])
        self.assertFalse(result["volume_divergence"])
        self.assertTrue(result["consolidation_volume_declining"])

    def test_check_breakout_returns_strong_signal_when_volume_and_throwback_align(self):
        result = technical_analysis.check_breakout(_make_breakout_frame(), [101.0])

        self.assertTrue(result["breakout_detected"])
        self.assertEqual(result["breakout_level"], 101.0)
        self.assertTrue(result["volume_confirmed"])
        self.assertTrue(result["throwback_confirmed"])
        self.assertEqual(result["signal_strength"], "strong")

    def test_check_breakout_returns_default_when_no_level_is_broken(self):
        result = technical_analysis.check_breakout(_make_breakout_frame(), [200.0])

        self.assertEqual(
            result,
            {
                "breakout_detected": False,
                "breakout_level": None,
                "volume_confirmed": False,
                "throwback_confirmed": False,
                "signal_strength": None,
            },
        )


class TechnicalAnalysisIntegrationTests(unittest.TestCase):
    def test_get_round_number_levels_scales_by_price_zone(self):
        self.assertEqual(technical_analysis.get_round_number_levels(12.3), [11.0, 12.0, 13.0, 14.0])
        self.assertEqual(technical_analysis.get_round_number_levels(78.0), [70.0, 75.0, 80.0, 85.0])
        self.assertEqual(technical_analysis.get_round_number_levels(310.0), [250.0, 300.0, 350.0, 400.0])
        self.assertEqual(technical_analysis.get_round_number_levels(0.0), [])

    def test_analyze_technical_returns_uptrend_structure_for_representative_data(self):
        frame = _make_oscillating_trend_frame()
        result = technical_analysis.analyze_technical(frame)

        self.assertEqual(result["trend"], "uptrend")
        self.assertEqual(result["signals"]["dow_theory_trend"], "uptrend")
        self.assertGreaterEqual(result["indicators"]["rsi"], 0.0)
        self.assertLessEqual(result["indicators"]["rsi"], 100.0)
        self.assertIsNone(result["indicators"]["sma200"])
        self.assertLessEqual(result["entry_price"], float(frame["Close"].iloc[-1]))
        self.assertGreaterEqual(result["target_price"], float(frame["Close"].iloc[-1]))
        self.assertEqual(result["support_levels"], sorted(set(result["support_levels"])))
        self.assertEqual(result["resistance_levels"], sorted(set(result["resistance_levels"])))
        self.assertGreater(len(result["patterns"]), 0)
        self.assertEqual(len(result["data"]), len(frame))

    def test_analyze_technical_handles_nans_without_crashing(self):
        frame = _make_oscillating_trend_frame(length=80)
        frame.iloc[10, frame.columns.get_loc("Close")] = np.nan
        frame.iloc[11, frame.columns.get_loc("High")] = np.nan
        frame.iloc[12, frame.columns.get_loc("Low")] = np.nan
        frame.iloc[13, frame.columns.get_loc("Volume")] = np.nan

        result = technical_analysis.analyze_technical(frame)

        self.assertEqual(len(result["data"]), len(frame))
        self.assertTrue(pd.isna(result["data"].iloc[10]["Close"]))
        self.assertIsNotNone(result["indicators"]["rsi"])
        self.assertIsNotNone(result["signals"]["stochastic"])
        self.assertIsNone(result["indicators"]["sma200"])

    def test_analyze_technical_keeps_flat_series_indicators_flat(self):
        result = technical_analysis.analyze_technical(_make_flat_frame())

        self.assertEqual(result["trend"], "sideways")
        self.assertAlmostEqual(result["indicators"]["sma20"], 50.0, places=6)
        self.assertAlmostEqual(result["indicators"]["sma50"], 50.0, places=6)
        self.assertAlmostEqual(result["indicators"]["ema12"], 50.0, places=6)
        self.assertAlmostEqual(result["indicators"]["ema26"], 50.0, places=6)
        self.assertAlmostEqual(result["indicators"]["bb_low"], 50.0, places=6)
        self.assertAlmostEqual(result["indicators"]["bb_high"], 50.0, places=6)
        self.assertAlmostEqual(result["atr"], 2.0, places=6)
        self.assertTrue(result["signals"]["bollinger_squeeze"])

    def test_analyze_technical_returns_default_payload_for_empty_input(self):
        empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        result = technical_analysis.analyze_technical(empty)

        self.assertEqual(result["trend"], "unknown")
        self.assertEqual(result["signals"], {})
        self.assertEqual(result["support_levels"], [])
        self.assertEqual(result["resistance_levels"], [])
        self.assertEqual(result["patterns"], [])
        self.assertEqual(result["gaps"], [])
        self.assertEqual(result["indicators"], {})
        self.assertEqual(result["volume_trend"], "unknown")
        self.assertFalse(result["breakout"]["breakout_detected"])


if __name__ == "__main__":
    unittest.main()
