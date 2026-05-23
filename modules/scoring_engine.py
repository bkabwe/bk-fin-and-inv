from __future__ import annotations

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
