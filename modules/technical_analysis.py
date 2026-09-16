from __future__ import annotations

import math

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator

from modules.logger import get_logger

logger = get_logger(__name__)


def _safe(value):
    try:
        return None if pd.isna(value) else float(value)
    except Exception:
        return None


def _pivot_points(series: pd.Series, window: int = 5, mode: str = "high") -> list[tuple[int, float]]:
    if series is None or series.empty or len(series) < (2 * window + 1):
        return []
    points: list[tuple[int, float]] = []
    values = series.astype(float).values
    for i in range(window, len(values) - window):
        chunk = values[i - window : i + window + 1]
        center = values[i]
        if mode == "high" and center >= np.max(chunk):
            points.append((i, float(center)))
        if mode == "low" and center <= np.min(chunk):
            points.append((i, float(center)))
    return points


def _line_from_points(p1: tuple[int, float], p2: tuple[int, float]) -> tuple[float, float]:
    x1, y1 = p1
    x2, y2 = p2
    slope = (y2 - y1) / max((x2 - x1), 1e-9)
    intercept = y1 - slope * x1
    return float(slope), float(intercept)


def detect_chart_patterns(df: pd.DataFrame) -> list[dict[str, float | str | None]]:
    if df is None or df.empty or len(df) < 40:
        return []

    close = df["Close"].tail(180).astype(float)
    high = df["High"].tail(180).astype(float)
    low = df["Low"].tail(180).astype(float)
    patterns: list[dict[str, float | str | None]] = []

    def add(name: str, implication: str, description: str, target: float | None = None):
        patterns.append(
            {
                "name": name,
                "implication": implication,
                "description": description,
                "target": round(float(target), 2) if target is not None and np.isfinite(target) else None,
            }
        )

    hi = high.nlargest(3).sort_index()
    lo = low.nsmallest(3).sort_index()

    if len(hi) >= 2 and abs(hi.iloc[-1] - hi.iloc[-2]) / max(abs(hi.iloc[-2]), 1e-9) < 0.02:
        breakdown = float(low.tail(30).min())
        top = float(np.mean([hi.iloc[-1], hi.iloc[-2]]))
        add("Double Top", "bearish", "Two peaks near same level.", breakdown - (top - breakdown))

    if len(lo) >= 2 and abs(lo.iloc[-1] - lo.iloc[-2]) / max(abs(lo.iloc[-2]), 1e-9) < 0.02:
        breakout = float(high.tail(30).max())
        bottom = float(np.mean([lo.iloc[-1], lo.iloc[-2]]))
        add("Double Bottom", "bullish", "Two troughs near same level.", breakout + (breakout - bottom))

    if len(hi) == 3 and hi.iloc[1] > hi.iloc[0] and hi.iloc[1] > hi.iloc[2]:
        neckline = float(low.loc[hi.index.min() : hi.index.max()].mean())
        head = float(hi.iloc[1])
        add("Head and Shoulders", "bearish", "Center peak above shoulders.", neckline - (head - neckline))

    if len(lo) == 3 and lo.iloc[1] < lo.iloc[0] and lo.iloc[1] < lo.iloc[2]:
        neckline = float(high.loc[lo.index.min() : lo.index.max()].mean())
        head = float(lo.iloc[1])
        add("Inverse Head and Shoulders", "bullish", "Center trough below shoulders.", neckline + (neckline - head))

    x = np.arange(len(close))
    hs = np.polyfit(x, high.tail(len(x)), 1)[0]
    ls = np.polyfit(x, low.tail(len(x)), 1)[0]
    widest_part = float(high.max() - low.min())
    breakout_point = float(close.iloc[-1])
    if hs < 0 < ls:
        add("Symmetrical Triangle", "neutral", "Converging highs and lows.", breakout_point + widest_part)
    elif abs(hs) < 1e-3 and ls > 0:
        add("Ascending Triangle", "bullish", "Flat top and rising lows.", breakout_point + widest_part)
    elif hs < 0 and abs(ls) < 1e-3:
        add("Descending Triangle", "bearish", "Falling highs and flat lows.", breakout_point - widest_part)

    move = close.iloc[-20] - close.iloc[-40]
    if abs(move) > close.std() * 0.8:
        flagpole_height = float(abs(close.iloc[-20] - close.iloc[-40]))
        add(
            "Flag / Pennant",
            "bullish" if move > 0 else "bearish",
            "Impulse then consolidation.",
            breakout_point + flagpole_height if move > 0 else breakout_point - flagpole_height,
        )

    if close.iloc[-1] > close.rolling(20).max().iloc[-1] * 0.98:
        add("Cup and Handle", "bullish", "Rounded recovery and handle breakout setup.", close.iloc[-1] * 1.08)

    # Extended Schabacker-style library
    if len(hi) == 3 and abs(hi.iloc[0] - hi.iloc[1]) / max(abs(hi.iloc[1]), 1e-9) < 0.02 and abs(hi.iloc[1] - hi.iloc[2]) / max(abs(hi.iloc[2]), 1e-9) < 0.02:
        add("Triple Top", "bearish", "Three peaks near the same level.", float(low.tail(40).min()))

    if len(lo) == 3 and abs(lo.iloc[0] - lo.iloc[1]) / max(abs(lo.iloc[1]), 1e-9) < 0.02 and abs(lo.iloc[1] - lo.iloc[2]) / max(abs(lo.iloc[2]), 1e-9) < 0.02:
        add("Triple Bottom", "bullish", "Three troughs near the same level.", float(high.tail(40).max()))

    if hs > 0 and ls > 0 and hs < ls:
        add("Rising Wedge", "bearish", "Both trendlines rise with narrowing spread.", close.iloc[-1] * 0.94)
    if hs < 0 and ls < 0 and abs(ls) < abs(hs):
        add("Falling Wedge", "bullish", "Both trendlines fall with narrowing spread.", close.iloc[-1] * 1.06)

    lows_rm = low.tail(60).rolling(7, min_periods=3).mean().dropna()
    if len(lows_rm) >= 20:
        xv = np.arange(len(lows_rm))
        quad = np.polyfit(xv, lows_rm.values, 2)
        if quad[0] > 0 and lows_rm.iloc[0] > lows_rm.min() and lows_rm.iloc[-1] > lows_rm.min():
            add("Rounding Bottom", "bullish", "Saucer-like base pattern.", close.iloc[-1] * 1.1)

    # Island reversal
    if len(df) >= 4:
        prev = df.iloc[-3]
        mid = df.iloc[-2]
        cur = df.iloc[-1]
        if mid["Low"] > prev["High"] and cur["High"] < mid["Low"]:
            add("Island Reversal", "bearish", "Gap up then gap down isolates price island.", close.iloc[-1] * 0.95)
        elif mid["High"] < prev["Low"] and cur["Low"] > mid["High"]:
            add("Island Reversal", "bullish", "Gap down then gap up isolates price island.", close.iloc[-1] * 1.05)

    # Key reversal
    recent_high = float(high.tail(20).max())
    recent_low = float(low.tail(20).min())
    if high.iloc[-1] >= recent_high and close.iloc[-1] < close.iloc[-2]:
        add("Key Reversal", "bearish", "New high intraday but bearish close.", close.iloc[-1] * 0.96)
    if low.iloc[-1] <= recent_low and close.iloc[-1] > close.iloc[-2]:
        add("Key Reversal", "bullish", "New low intraday but bullish close.", close.iloc[-1] * 1.04)

    if hs > 0 and ls < 0:
        add("Broadening Formation", "bearish", "Highs rise while lows fall.", close.iloc[-1] * 0.93)

    logger.info("Detected %d chart patterns", len(patterns))
    return patterns


def detect_gaps(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty or len(df) < 25:
        return []

    gaps: list[dict] = []
    avg_vol20 = df["Volume"].rolling(20).mean()
    high_52w = df["High"].tail(252).max()
    low_52w = df["Low"].tail(252).min()

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        cur = df.iloc[i]

        direction = None
        gap_size = 0.0
        if cur["Low"] > prev["High"]:
            direction = "up"
            gap_size = float(cur["Low"] - prev["High"])
        elif cur["High"] < prev["Low"]:
            direction = "down"
            gap_size = float(prev["Low"] - cur["High"])

        if not direction:
            continue

        price_ref = max(float(prev["Close"]), 1e-9)
        gap_pct = (gap_size / price_ref) * 100.0
        vol = float(cur["Volume"])
        vol_avg = float(avg_vol20.iloc[i]) if pd.notna(avg_vol20.iloc[i]) else vol

        # Context checks
        roll_high = float(df["High"].iloc[max(0, i - 20) : i].max()) if i > 0 else float(prev["High"])
        roll_low = float(df["Low"].iloc[max(0, i - 20) : i].min()) if i > 0 else float(prev["Low"])
        at_sr = abs(float(prev["High"]) - roll_high) / max(abs(roll_high), 1e-9) < 0.01 or abs(float(prev["Low"]) - roll_low) / max(abs(roll_low), 1e-9) < 0.01

        win = df.iloc[max(0, i - 60) : i + 1]
        moved_from_low = (float(cur["Close"]) - float(win["Low"].min())) / max(abs(float(win["Low"].min())), 1e-9)
        moved_from_high = (float(win["High"].max()) - float(cur["Close"])) / max(abs(float(win["High"].max())), 1e-9)
        mid_trend = moved_from_low >= 0.10 or moved_from_high >= 0.10

        near_extreme = (abs(float(cur["High"]) - float(high_52w)) / max(abs(float(high_52w)), 1e-9) < 0.02) or (
            abs(float(cur["Low"]) - float(low_52w)) / max(abs(float(low_52w)), 1e-9) < 0.02
        )

        gap_type = "Common gap"
        significance = "low"
        if gap_pct >= 1.5 and vol >= 2.0 * vol_avg and near_extreme:
            gap_type = "Exhaustion gap"
            significance = "high"
        elif gap_pct >= 1.5 and vol >= 1.5 * vol_avg and at_sr:
            gap_type = "Breakaway gap"
            significance = "high"
        elif gap_pct >= 1.0 and mid_trend:
            gap_type = "Runaway gap"
            significance = "medium"

        gaps.append(
            {
                "type": gap_type,
                "date": str(df.index[i].date()) if hasattr(df.index[i], "date") else str(df.index[i]),
                "gap_pct": round(float(gap_pct), 2),
                "direction": direction,
                "significance": significance,
                "level": round(float((float(cur["Open"]) + float(prev["Close"])) / 2.0), 2),
            }
        )

    logger.info("Detected %d gaps", len(gaps))
    return gaps[-30:]


def detect_trendlines(df: pd.DataFrame) -> dict:
    default = {
        "uptrend_line": {"slope": None, "intercept": None, "current_value": None, "broken": False},
        "downtrend_line": {"slope": None, "intercept": None, "current_value": None, "broken": False},
        "channel_upper": None,
        "channel_lower": None,
        "trendline_break_signal": None,
    }
    if df is None or df.empty or len(df) < 30:
        return default

    lows = _pivot_points(df["Low"].astype(float), window=5, mode="low")
    highs = _pivot_points(df["High"].astype(float), window=5, mode="high")
    if len(lows) >= 2:
        s, b = _line_from_points(lows[-2], lows[-1])
        cur_val = (s * (len(df) - 1)) + b
        default["uptrend_line"] = {"slope": round(s, 6), "intercept": round(b, 6), "current_value": round(float(cur_val), 2), "broken": False}

    if len(highs) >= 2:
        s, b = _line_from_points(highs[-2], highs[-1])
        cur_val = (s * (len(df) - 1)) + b
        default["downtrend_line"] = {
            "slope": round(s, 6),
            "intercept": round(b, 6),
            "current_value": round(float(cur_val), 2),
            "broken": False,
        }

    close = float(df["Close"].iloc[-1])
    vol = float(df["Volume"].iloc[-1])
    avg_vol = float(df["Volume"].tail(20).mean()) if len(df) >= 20 else vol

    up_val = default["uptrend_line"].get("current_value")
    down_val = default["downtrend_line"].get("current_value")

    if up_val and close < up_val * 0.995 and vol > avg_vol:
        default["uptrend_line"]["broken"] = True
        default["trendline_break_signal"] = "bearish_break"
    if down_val and close > down_val * 1.005 and vol > avg_vol:
        default["downtrend_line"]["broken"] = True
        default["trendline_break_signal"] = "bullish_break"

    if up_val and len(highs) >= 5:
        default["channel_upper"] = round(float(max(h[1] for h in highs[-5:])), 2)
    if down_val and len(lows) >= 5:
        default["channel_lower"] = round(float(min(l[1] for l in lows[-5:])), 2)

    logger.info("Trendline break signal: %s", default["trendline_break_signal"])
    return default


def relative_strength_vs_spy(ticker_data: pd.DataFrame, spy_data: pd.DataFrame) -> dict:
    default = {"rs_20d": None, "rs_50d": None, "rs_252d": None, "rs_trend": "stable"}
    if ticker_data is None or spy_data is None or ticker_data.empty or spy_data.empty:
        return default

    stock = ticker_data["Close"].astype(float).dropna()
    spy = spy_data["Close"].astype(float).dropna()
    merged = pd.concat([stock.rename("stock"), spy.rename("spy")], axis=1).dropna()
    if merged.empty:
        return default

    def rs(period: int):
        if len(merged) <= period:
            return None
        sret = (merged["stock"].iloc[-1] / merged["stock"].iloc[-period - 1]) - 1
        mret = (merged["spy"].iloc[-1] / merged["spy"].iloc[-period - 1]) - 1
        if abs(float(mret)) < 1e-9:
            return 1.0
        return float(sret / mret)

    rs20 = rs(20)
    rs50 = rs(50)
    rs252 = rs(252)

    trend = "stable"
    if rs20 is not None and rs50 is not None:
        if rs20 > rs50 * 1.05:
            trend = "improving"
        elif rs20 < rs50 * 0.95:
            trend = "deteriorating"

    return {
        "rs_20d": round(rs20, 3) if rs20 is not None and np.isfinite(rs20) else None,
        "rs_50d": round(rs50, 3) if rs50 is not None and np.isfinite(rs50) else None,
        "rs_252d": round(rs252, 3) if rs252 is not None and np.isfinite(rs252) else None,
        "rs_trend": trend,
    }


def analyze_volume_quality(df: pd.DataFrame) -> dict:
    default = {
        "breakout_volume_confirmed": False,
        "climactic_volume": False,
        "volume_divergence": False,
        "consolidation_volume_declining": False,
    }
    if df is None or df.empty or len(df) < 25:
        return default

    avg_vol20 = float(df["Volume"].tail(20).mean())
    last_vol = float(df["Volume"].iloc[-1])
    last_close = float(df["Close"].iloc[-1])
    prev_20_high = float(df["High"].iloc[-21:-1].max()) if len(df) >= 21 else float(df["High"].max())
    breakout = last_close > prev_20_high
    breakout_volume_confirmed = breakout and last_vol >= 1.5 * avg_vol20

    tr = (df["High"] - df["Low"]).astype(float)
    large_bar = float(tr.iloc[-1]) >= float(tr.tail(20).mean()) * 1.5 if len(tr) >= 20 else False
    climactic_volume = last_vol >= 3.0 * avg_vol20 and large_bar

    obv = OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()
    prev_high_idx = df["Close"].iloc[-21:-1].idxmax() if len(df) >= 21 else df["Close"].idxmax()
    prev_high_pos = df.index.get_loc(prev_high_idx)
    volume_divergence = breakout and float(obv.iloc[-1]) < float(obv.iloc[prev_high_pos])

    recent10 = float(df["Volume"].tail(10).mean())
    prior10 = float(df["Volume"].iloc[-20:-10].mean()) if len(df) >= 20 else recent10
    consolidation_volume_declining = recent10 < prior10

    return {
        "breakout_volume_confirmed": bool(breakout_volume_confirmed),
        "climactic_volume": bool(climactic_volume),
        "volume_divergence": bool(volume_divergence),
        "consolidation_volume_declining": bool(consolidation_volume_declining),
    }


def analyze_dow_theory(df: pd.DataFrame) -> dict:
    default = {
        "primary_trend": "sideways",
        "last_swing_high": None,
        "last_swing_low": None,
        "trend_confirmed": False,
        "hh_hl": False,
        "lh_ll": False,
    }
    if df is None or df.empty or len(df) < 30:
        return default

    highs = _pivot_points(df["High"].astype(float), window=5, mode="high")
    lows = _pivot_points(df["Low"].astype(float), window=5, mode="low")
    if len(highs) < 2 or len(lows) < 2:
        return default

    # keep only significant swings (>=3% move)
    sig_highs = [highs[0]]
    for point in highs[1:]:
        if abs(point[1] - sig_highs[-1][1]) / max(abs(sig_highs[-1][1]), 1e-9) >= 0.03:
            sig_highs.append(point)

    sig_lows = [lows[0]]
    for point in lows[1:]:
        if abs(point[1] - sig_lows[-1][1]) / max(abs(sig_lows[-1][1]), 1e-9) >= 0.03:
            sig_lows.append(point)

    if len(sig_highs) < 2 or len(sig_lows) < 2:
        return default

    hh = sig_highs[-1][1] > sig_highs[-2][1]
    hl = sig_lows[-1][1] > sig_lows[-2][1]
    lh = sig_highs[-1][1] < sig_highs[-2][1]
    ll = sig_lows[-1][1] < sig_lows[-2][1]

    trend = "sideways"
    if hh and hl:
        trend = "uptrend"
    elif lh and ll:
        trend = "downtrend"

    trend_confirmed = bool(hh or ll)
    return {
        "primary_trend": trend,
        "last_swing_high": round(float(sig_highs[-1][1]), 2),
        "last_swing_low": round(float(sig_lows[-1][1]), 2),
        "trend_confirmed": trend_confirmed,
        "hh_hl": bool(hh and hl),
        "lh_ll": bool(lh and ll),
    }


def get_round_number_levels(current_price: float) -> list[float]:
    if current_price is None or current_price <= 0:
        return []

    if current_price < 20:
        step = 1
    elif current_price < 100:
        step = 5
    elif current_price < 500:
        step = 50
    else:
        step = 100

    anchor = math.floor(current_price / step) * step
    levels = sorted({max(step, anchor - step), anchor, anchor + step, anchor + 2 * step})
    return [round(float(x), 2) for x in levels]


def check_breakout(df: pd.DataFrame, resistance_levels: list[float]) -> dict:
    result = {
        "breakout_detected": False,
        "breakout_level": None,
        "volume_confirmed": False,
        "throwback_confirmed": False,
        "signal_strength": None,
    }
    if df is None or df.empty or not resistance_levels:
        return result

    close = float(df["Close"].iloc[-1])
    avg_vol20 = float(df["Volume"].tail(20).mean()) if len(df) >= 20 else float(df["Volume"].iloc[-1])
    last_vol = float(df["Volume"].iloc[-1])

    broken_levels = [lvl for lvl in resistance_levels if close > float(lvl) * 1.005]
    if not broken_levels:
        return result

    level = max(float(lvl) for lvl in broken_levels)
    volume_confirmed = last_vol >= 1.3 * avg_vol20

    window = df.tail(20)
    near_level_touch = (window["Low"].astype(float) <= level * 1.01).any()
    held_above = (window["Close"].astype(float).tail(5) >= level * 0.995).all()
    throwback_confirmed = bool(near_level_touch and held_above)

    strength = "weak"
    if volume_confirmed and throwback_confirmed:
        strength = "strong"
    elif volume_confirmed or throwback_confirmed:
        strength = "moderate"

    result.update(
        {
            "breakout_detected": True,
            "breakout_level": round(level, 2),
            "volume_confirmed": bool(volume_confirmed),
            "throwback_confirmed": bool(throwback_confirmed),
            "signal_strength": strength,
        }
    )
    return result


def analyze_technical(data: pd.DataFrame) -> dict:
    if data is None or data.empty:
        return {
            "trend": "unknown",
            "signals": {},
            "support_levels": [],
            "resistance_levels": [],
            "entry_price": None,
            "target_price": None,
            "indicators": {},
            "patterns": [],
            "volume_confirmation": False,
            "volume_trend": "unknown",
            "atr": None,
            "gaps": [],
            "trendlines": {
                "uptrend_line": {"slope": None, "intercept": None, "current_value": None, "broken": False},
                "downtrend_line": {"slope": None, "intercept": None, "current_value": None, "broken": False},
                "channel_upper": None,
                "channel_lower": None,
                "trendline_break_signal": None,
            },
            "volume_quality": {
                "breakout_volume_confirmed": False,
                "climactic_volume": False,
                "volume_divergence": False,
                "consolidation_volume_declining": False,
            },
            "dow_theory": {
                "primary_trend": "sideways",
                "last_swing_high": None,
                "last_swing_low": None,
                "trend_confirmed": False,
                "hh_hl": False,
                "lh_ll": False,
            },
            "breakout": {
                "breakout_detected": False,
                "breakout_level": None,
                "volume_confirmed": False,
                "throwback_confirmed": False,
                "signal_strength": None,
            },
        }

    df = data.copy()
    df["sma20"] = SMAIndicator(df["Close"], window=20).sma_indicator()
    df["sma50"] = SMAIndicator(df["Close"], window=50).sma_indicator()
    df["sma200"] = SMAIndicator(df["Close"], window=200).sma_indicator()
    df["ema12"] = EMAIndicator(df["Close"], window=12).ema_indicator()
    df["ema26"] = EMAIndicator(df["Close"], window=26).ema_indicator()

    rsi = RSIIndicator(df["Close"], window=14).rsi()
    macd = MACD(df["Close"], 26, 12, 9)
    macd_line = macd.macd()
    macd_signal = macd.macd_signal()
    stoch = StochasticOscillator(df["High"], df["Low"], df["Close"], window=14, smooth_window=3).stoch()
    bb = BollingerBands(df["Close"], window=20, window_dev=2)
    bb_high, bb_low = bb.bollinger_hband(), bb.bollinger_lband()
    bb_mid = bb.bollinger_mavg()
    atr = AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()
    obv = OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()

    df["rsi"] = rsi
    df["macd"] = macd_line
    df["macd_signal"] = macd_signal
    df["bb_high"] = bb_high
    df["bb_low"] = bb_low
    df["bb_mid"] = bb_mid

    last = df.iloc[-1]
    trend = "sideways"
    if last["sma20"] > last["sma50"] > last["sma200"]:
        trend = "uptrend"
    elif last["sma20"] < last["sma50"] < last["sma200"]:
        trend = "downtrend"

    recent = df.tail(20)
    prior = df.tail(40).head(20)
    vol_trend = "increasing" if recent["Volume"].mean() >= (prior["Volume"].mean() if not prior.empty else recent["Volume"].mean()) else "decreasing"
    volume_confirmation = vol_trend == "increasing"

    support = sorted(df["Low"].tail(90).nsmallest(3).round(2).tolist())
    resistance = sorted(df["High"].tail(90).nlargest(3).round(2).tolist())
    current_price = _safe(last["Close"]) or 0.0

    round_levels = get_round_number_levels(current_price)
    support.extend([lvl for lvl in round_levels if lvl <= current_price])
    resistance.extend([lvl for lvl in round_levels if lvl >= current_price])

    patterns = detect_chart_patterns(df)
    for pattern in patterns:
        target = pattern.get("target")
        if target is None:
            continue
        target_val = float(target)
        if target_val >= current_price:
            resistance.append(round(target_val, 2))
        else:
            support.append(round(target_val, 2))

    gaps = detect_gaps(df)
    for gap in gaps:
        level = gap.get("level")
        if level is None:
            continue
        if gap.get("direction") == "up":
            support.append(float(level))
        else:
            resistance.append(float(level))

    support = sorted({round(float(x), 2) for x in support if x is not None})
    resistance = sorted({round(float(x), 2) for x in resistance if x is not None})

    trendlines = detect_trendlines(df)
    volume_quality = analyze_volume_quality(df)
    dow_theory = analyze_dow_theory(df)
    breakout = check_breakout(df, resistance)

    # cross-check trend with Dow theory
    if dow_theory.get("primary_trend") in {"uptrend", "downtrend"} and trend == "sideways":
        trend = dow_theory["primary_trend"]

    entry = round(
        min(
            [
                x
                for x in [
                    max([s for s in support if s <= current_price], default=current_price),
                    _safe(bb_low.iloc[-1]),
                    current_price,
                ]
                if x is not None
            ]
        ),
        2,
    )
    target = round(
        max(
            [
                x
                for x in [
                    min([r for r in resistance if r >= current_price], default=current_price),
                    _safe(bb_high.iloc[-1]),
                    current_price,
                ]
                if x is not None
            ]
        ),
        2,
    )

    return {
        "trend": trend,
        "signals": {
            "rsi_signal": "overbought" if (rsi.iloc[-1] or 0) > 70 else "oversold" if (rsi.iloc[-1] or 0) < 30 else "neutral",
            "macd_signal": "bullish" if (macd_line.iloc[-1] or 0) > (macd_signal.iloc[-1] or 0) else "bearish",
            "stochastic": _safe(stoch.iloc[-1]),
            "bollinger_squeeze": ((bb_high.iloc[-1] - bb_low.iloc[-1]) / max(current_price, 1e-9)) < 0.08,
            "trendline_break_signal": trendlines.get("trendline_break_signal"),
            "dow_theory_trend": dow_theory.get("primary_trend"),
        },
        "support_levels": support,
        "resistance_levels": resistance,
        "entry_price": entry,
        "target_price": target,
        "indicators": {
            "rsi": _safe(rsi.iloc[-1]),
            "macd": _safe(macd_line.iloc[-1]),
            "macd_signal": _safe(macd_signal.iloc[-1]),
            "obv": _safe(obv.iloc[-1]),
            "atr": _safe(atr.iloc[-1]),
            "sma20": _safe(last["sma20"]),
            "sma50": _safe(last["sma50"]),
            "sma200": _safe(last["sma200"]),
            "ema12": _safe(last["ema12"]),
            "ema26": _safe(last["ema26"]),
            "bb_low": _safe(bb_low.iloc[-1]),
            "bb_high": _safe(bb_high.iloc[-1]),
        },
        "patterns": patterns,
        "volume_confirmation": volume_confirmation,
        "volume_trend": vol_trend,
        "atr": _safe(atr.iloc[-1]),
        "gaps": [{k: v for k, v in gap.items() if k != "level"} for gap in gaps],
        "trendlines": trendlines,
        "volume_quality": volume_quality,
        "dow_theory": dow_theory,
        "breakout": breakout,
        "data": df,
    }
