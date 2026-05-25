from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator


def _safe(value):
    try:
        return None if pd.isna(value) else float(value)
    except Exception:
        return None


def detect_chart_patterns(df: pd.DataFrame) -> list[dict[str, str]]:
    if df.empty or len(df) < 40:
        return []
    close = df["Close"].tail(120)
    high = df["High"].tail(120)
    low = df["Low"].tail(120)
    patterns = []

    hi = high.nlargest(3).sort_index()
    lo = low.nsmallest(3).sort_index()
    if len(hi) >= 2 and abs(hi.iloc[-1] - hi.iloc[-2]) / max(hi.iloc[-2], 1e-9) < 0.02:
        patterns.append({"name": "Double Top", "implication": "bearish", "description": "Two peaks near same level."})
    if len(lo) >= 2 and abs(lo.iloc[-1] - lo.iloc[-2]) / max(lo.iloc[-2], 1e-9) < 0.02:
        patterns.append({"name": "Double Bottom", "implication": "bullish", "description": "Two troughs near same level."})

    if len(hi) == 3 and hi.iloc[1] > hi.iloc[0] and hi.iloc[1] > hi.iloc[2]:
        patterns.append({"name": "Head and Shoulders", "implication": "bearish", "description": "Center peak above shoulders."})
    if len(lo) == 3 and lo.iloc[1] < lo.iloc[0] and lo.iloc[1] < lo.iloc[2]:
        patterns.append({"name": "Inverse Head and Shoulders", "implication": "bullish", "description": "Center trough below shoulders."})

    x = np.arange(len(close))
    hs = np.polyfit(x, high.tail(len(x)), 1)[0]
    ls = np.polyfit(x, low.tail(len(x)), 1)[0]
    if hs < 0 < ls:
        patterns.append({"name": "Symmetrical Triangle", "implication": "neutral", "description": "Converging highs and lows."})
    elif abs(hs) < 1e-3 and ls > 0:
        patterns.append({"name": "Ascending Triangle", "implication": "bullish", "description": "Flat top and rising lows."})
    elif hs < 0 and abs(ls) < 1e-3:
        patterns.append({"name": "Descending Triangle", "implication": "bearish", "description": "Falling highs and flat lows."})

    move = close.iloc[-20] - close.iloc[-40]
    if abs(move) > close.std() * 0.8:
        patterns.append(
            {"name": "Flag / Pennant", "implication": "bullish" if move > 0 else "bearish", "description": "Impulse then consolidation."}
        )
    if close.iloc[-1] > close.rolling(20).max().iloc[-1] * 0.98:
        patterns.append({"name": "Cup and Handle", "implication": "bullish", "description": "Rounded recovery and handle breakout setup."})
    return patterns


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
    entry = round(min([x for x in [max([s for s in support if s <= current_price], default=current_price), _safe(bb_low.iloc[-1]), current_price] if x is not None]), 2)
    target = round(max([x for x in [min([r for r in resistance if r >= current_price], default=current_price), _safe(bb_high.iloc[-1]), current_price] if x is not None]), 2)

    return {
        "trend": trend,
        "signals": {
            "rsi_signal": "overbought" if (rsi.iloc[-1] or 0) > 70 else "oversold" if (rsi.iloc[-1] or 0) < 30 else "neutral",
            "macd_signal": "bullish" if (macd_line.iloc[-1] or 0) > (macd_signal.iloc[-1] or 0) else "bearish",
            "stochastic": _safe(stoch.iloc[-1]),
            "bollinger_squeeze": ((bb_high.iloc[-1] - bb_low.iloc[-1]) / max(current_price, 1e-9)) < 0.08,
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
        "patterns": detect_chart_patterns(df),
        "volume_confirmation": volume_confirmation,
        "volume_trend": vol_trend,
        "atr": _safe(atr.iloc[-1]),
        "data": df,
    }
