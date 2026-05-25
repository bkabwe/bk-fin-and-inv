from __future__ import annotations

import numpy as np

from modules.data_fetcher import get_stock_data, get_stock_info
from modules.fundamental_analysis import analyze_fundamentals
from modules.sentiment_analysis import analyze_sentiment
from modules.technical_analysis import analyze_technical


def _recommendation(score: int) -> str:
    if score >= 80:
        return "🟢 STRONG BUY"
    if score >= 65:
        return "🔵 BUY"
    if score >= 50:
        return "🟡 TAKE SMALL POSITION"
    if score >= 35:
        return "🟠 MONITOR"
    if score >= 20:
        return "🔴 DO NOT BUY"
    return "⛔ AVOID"


def analyze_stock(ticker: str, period: str = "1y", interval: str = "1d") -> dict:
    data = get_stock_data(ticker, period=period, interval=interval)
    info = get_stock_info(ticker)
    technical = analyze_technical(data)
    current_price = float(data["Close"].iloc[-1]) if not data.empty else info.get("currentPrice")
    fundamentals = analyze_fundamentals(info, current_price=current_price)
    sentiment = analyze_sentiment(ticker)

    trend_score = 15 if technical["trend"] == "uptrend" else 8 if technical["trend"] == "sideways" else 2
    rsi, macd, signal = (
        technical["indicators"].get("rsi"),
        technical["indicators"].get("macd"),
        technical["indicators"].get("macd_signal"),
    )
    momentum_score = (8 if rsi is not None and 40 <= rsi <= 65 else 4 if rsi is not None and 30 <= rsi <= 75 else 1) + (
        7 if macd is not None and signal is not None and macd > signal else 2
    )
    volume_score = 10 if technical["volume_confirmation"] else 5 if technical["volume_trend"] == "increasing" else 2
    bullish = sum(1 for p in technical["patterns"] if p["implication"] == "bullish")
    bearish = sum(1 for p in technical["patterns"] if p["implication"] == "bearish")
    pattern_score = max(0, min(10, 5 + (bullish - bearish) * 2))

    technical_total = max(0, min(50, trend_score + momentum_score + volume_score + pattern_score))
    fundamental_total = round((fundamentals["fundamental_score"] / 100) * 30)
    sentiment_total = round(((sentiment["sentiment_score"] + 1) / 2) * 20)
    total = max(0, min(100, int(technical_total + fundamental_total + sentiment_total)))

    horizon = "Medium-Term Setup"
    if macd is not None and signal is not None and macd > signal and technical["trend"] != "uptrend":
        horizon = "Short-Term Opportunity"
    if technical["trend"] == "uptrend" and fundamentals["fundamental_score"] >= 65:
        horizon = "Long-Term Hold"

    entry, target = technical.get("entry_price"), technical.get("target_price")
    atr = technical.get("atr") or 0
    stop_loss = round((entry or current_price or 0) - (1.5 * atr), 2) if (entry or current_price) else None

    sell = None
    near_resistance = bool(target and current_price and current_price >= target * 0.97)
    high52 = fundamentals["metrics"].get("52w_high")
    if total < 35 and near_resistance:
        sell = "SELL"
    elif rsi and rsi > 75 and high52 and current_price and current_price >= high52 * 0.98:
        sell = "CONSIDER SELLING"
    elif macd and signal and macd < signal and technical["indicators"].get("sma50") and current_price < technical["indicators"]["sma50"]:
        sell = "REDUCE POSITION"

    return {
        "ticker": ticker.upper(),
        "company": info.get("shortName") or info.get("longName") or ticker.upper(),
        "current_price": current_price,
        "score": total,
        "recommendation": _recommendation(total),
        "time_horizon": horizon,
        "entry_price": entry,
        "target_price": target,
        "stop_loss": stop_loss,
        "technical": technical,
        "fundamentals": fundamentals,
        "sentiment": sentiment,
        "score_breakdown": {
            "technical": technical_total,
            "fundamental": fundamental_total,
            "sentiment": sentiment_total,
            "trend": trend_score,
            "momentum": momentum_score,
            "volume": volume_score,
            "patterns": pattern_score,
        },
        "sell_recommendation": sell,
    }


def _safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        if np.isnan(out):
            return None
        return out
    except Exception:
        return None


def _round_price(value: float | None) -> float:
    return round(max(value or 0.0, 0.01), 2)


def get_price_projections(ticker: str) -> dict:
    """
    Returns projected price targets based on technical analysis and fundamental metrics.
    """
    data = get_stock_data(ticker, period="1y", interval="1d")
    info = get_stock_info(ticker)
    technical = analyze_technical(data)

    current_price = _safe_float(data["Close"].iloc[-1] if not data.empty else None) or _safe_float(
        info.get("currentPrice") or info.get("regularMarketPrice")
    )
    if not current_price:
        return {
            "short_term_target": 0.0,
            "short_term_upside": 0.0,
            "short_term_basis": "Insufficient market data.",
            "medium_term_target": 0.0,
            "medium_term_upside": 0.0,
            "medium_term_basis": "Insufficient market data.",
            "long_term_target": 0.0,
            "long_term_upside": 0.0,
            "long_term_basis": "Insufficient market data.",
            "current_price": 0.0,
            "recommendation_to_sell_at": "Insufficient data to generate a sell recommendation.",
        }

    indicators = technical.get("indicators", {})
    resistance_levels = [x for x in technical.get("resistance_levels", []) if _safe_float(x)]
    resistance_above = sorted([x for x in resistance_levels if x > current_price])
    nearest_resistance = resistance_above[0] if resistance_above else None
    major_resistance = resistance_above[-1] if resistance_above else None

    rsi = _safe_float(indicators.get("rsi"))
    macd = _safe_float(indicators.get("macd"))
    macd_signal = _safe_float(indicators.get("macd_signal"))
    bb_high = _safe_float(indicators.get("bb_high"))
    atr = _safe_float(indicators.get("atr") or technical.get("atr")) or 0.0

    bullish_momentum = bool(macd is not None and macd_signal is not None and macd > macd_signal and (rsi is None or rsi < 65))
    if nearest_resistance and bb_high and bullish_momentum:
        short_target = min(nearest_resistance, bb_high)
        short_basis = "Nearest resistance blended with bullish MACD momentum and upper Bollinger Band."
    elif nearest_resistance and bb_high:
        short_target = min(nearest_resistance, bb_high)
        short_basis = "Nearest resistance constrained by upper Bollinger Band."
    elif nearest_resistance:
        short_target = nearest_resistance
        short_basis = "Nearest technical resistance above current price."
    elif bb_high:
        short_target = bb_high
        short_basis = "Upper Bollinger Band projection."
    else:
        short_target = current_price * 1.05
        short_basis = "Fallback 5% move due to missing resistance levels."

    analyst_target = _safe_float(info.get("targetMeanPrice"))
    atr_ratio = atr / current_price if current_price else 0.0
    technical_projection = current_price * (1 + atr_ratio * 3)
    if analyst_target:
        medium_target = analyst_target * 0.6 + technical_projection * 0.4
        medium_basis = "60% analyst target + 40% ATR technical projection."
    else:
        medium_target = major_resistance or (current_price * 1.15)
        medium_basis = (
            "Next major resistance level."
            if major_resistance
            else "Fallback 15% move due to missing analyst target and resistance."
        )

    sector_avg_pe = {
        "Technology": 28.0,
        "Financial Services": 14.0,
        "Healthcare": 22.0,
        "Consumer Cyclical": 22.0,
        "Communication Services": 20.0,
        "Industrials": 19.0,
        "Energy": 13.0,
        "Utilities": 17.0,
        "Real Estate": 18.0,
        "Basic Materials": 16.0,
        "Consumer Defensive": 20.0,
    }
    sector = str(info.get("sector") or "")
    sector_pe = sector_avg_pe.get(sector, 20.0)
    eps_forward = _safe_float(info.get("forwardEps")) or _safe_float(info.get("trailingEps"))
    forward_pe = _safe_float(info.get("forwardPE"))
    fair_value_estimate = eps_forward * sector_pe if eps_forward and forward_pe else None

    technical_df = technical.get("data")

    is_otc_or_penny = ticker.upper().endswith("Y") or ticker.upper().endswith("F") or current_price < 5 or info.get("exchange") == "PNK"
    if fair_value_estimate:
        long_target = max(fair_value_estimate, current_price * 1.25)
        long_basis = f"Fundamental fair value from EPS and sector P/E ({sector_pe:.1f}) with 25% floor."
    else:
        if technical_df is not None and not technical_df.empty and "sma200" in technical_df:
            sma200 = technical_df["sma200"].dropna()
            if len(sma200) >= 20:
                x = np.arange(len(sma200))
                slope = np.polyfit(x, sma200.values, 1)[0]
                months = 378 if is_otc_or_penny else 252
                long_target = current_price + (slope * months)
            else:
                long_target = current_price * 1.25
        else:
            long_target = current_price * 1.25
        long_basis = (
            "Trend-based 200-day SMA projection over 18 months for limited-fundamental OTC/penny stock."
            if is_otc_or_penny
            else "Trend-based 200-day SMA projection over 12 months."
        )
        if long_target <= 0:
            long_target = current_price * 1.25
            long_basis = "Fallback long-term projection using 25% appreciation floor."
        long_target = max(long_target, current_price * 1.05)

    short_target = _round_price(short_target)
    medium_target = _round_price(medium_target)
    long_target = _round_price(long_target)
    current_price = _round_price(current_price)

    short_upside = round(((short_target - current_price) / current_price) * 100, 2)
    medium_upside = round(((medium_target - current_price) / current_price) * 100, 2)
    long_upside = round(((long_target - current_price) / current_price) * 100, 2)

    if current_price >= medium_target:
        sell_text = "Stock may be near fair value. Consider taking profits."
    elif short_upside <= 0:
        sell_text = f"Hold for recovery to break-even at ${medium_target:.2f} before considering sell."
    else:
        sell_text = (
            f"Consider selling 50% of position near ${short_target:.2f} (short-term target). "
            f"Hold remainder for medium-term target of ${medium_target:.2f}."
        )

    return {
        "short_term_target": short_target,
        "short_term_upside": short_upside,
        "short_term_basis": short_basis,
        "medium_term_target": medium_target,
        "medium_term_upside": medium_upside,
        "medium_term_basis": medium_basis,
        "long_term_target": long_target,
        "long_term_upside": long_upside,
        "long_term_basis": long_basis,
        "current_price": current_price,
        "recommendation_to_sell_at": sell_text,
    }
