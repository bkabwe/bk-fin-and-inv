from __future__ import annotations


def analyze_fundamentals(info: dict, current_price: float | None = None) -> dict:
    if not info:
        return {"fundamental_score": 0, "flags": ["missing data"], "metrics": {}}
    trailing_pe = info.get("trailingPE")
    forward_pe = info.get("forwardPE")
    eps = info.get("trailingEps")
    growth = info.get("revenueGrowth")
    dte = info.get("debtToEquity")
    roe = info.get("returnOnEquity")
    target = info.get("targetMeanPrice")
    rec = info.get("recommendationKey")
    current_price = current_price or info.get("currentPrice") or info.get("regularMarketPrice")

    score, flags = 50, []
    if trailing_pe and trailing_pe < 20:
        score += 8
        flags.append("undervalued")
    elif trailing_pe and trailing_pe > 35:
        score -= 8
        flags.append("overvalued")
    if eps and eps > 0:
        score += 6
    if growth and growth > 0.1:
        score += 6
        flags.append("strong growth")
    elif growth and growth < 0:
        score -= 6
    if dte is not None and dte < 100:
        score += 5
        flags.append("strong balance sheet")
    elif dte is not None and dte > 250:
        score -= 6
    if roe and roe > 0.15:
        score += 6
    elif roe and roe < 0.05:
        score -= 4
    if target and current_price:
        score += 5 if target > current_price else -3
    if rec in {"strong_buy", "buy"}:
        score += 6
    elif rec in {"underperform", "sell"}:
        score -= 6
    score = max(0, min(100, int(score)))

    high, low = info.get("fiftyTwoWeekHigh"), info.get("fiftyTwoWeekLow")
    rel = None
    if high and low and current_price and high > low:
        rel = (current_price - low) / (high - low)
    return {
        "fundamental_score": score,
        "flags": sorted(set(flags)),
        "metrics": {
            "trailing_pe": trailing_pe,
            "forward_pe": forward_pe,
            "eps": eps,
            "revenue_growth": growth,
            "debt_to_equity": dte,
            "roe": roe,
            "market_cap": info.get("marketCap"),
            "sector": info.get("sector"),
            "52w_high": high,
            "52w_low": low,
            "current_price": current_price,
            "price_relative_to_52w_range": rel,
            "dividend_yield": info.get("dividendYield"),
            "analyst_target_price": target,
            "analyst_recommendation": rec,
        },
    }
