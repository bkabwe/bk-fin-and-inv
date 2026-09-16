from __future__ import annotations

from modules.logger import get_logger

logger = get_logger(__name__)

SECTOR_BENCHMARK_PE = {
    "Technology": 28,
    "Healthcare": 22,
    "Financials": 14,
    "Energy": 12,
    "Consumer Staples": 18,
    "Consumer Discretionary": 25,
    "Industrials": 20,
    "Materials": 16,
    "Utilities": 16,
    "Real Estate": 35,
    "Communication Services": 22,
}


def _safe_float(value):
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


_SECTOR_ALIASES = {
    "Financial Services": "Financials",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Basic Materials": "Materials",
}


def normalize_sector_name(sector: str | None) -> str | None:
    if not sector:
        return None
    clean = str(sector).strip()
    return _SECTOR_ALIASES.get(clean, clean) or None


def classify_market_cap_tier(market_cap: float | int | None) -> str:
    market_cap_value = _safe_float(market_cap)
    if market_cap_value is None or market_cap_value <= 0:
        return "unknown"

    # Widely used market-cap buckets for US equities. Keep thresholds explicit so
    # they are easy to adjust if the app's risk model changes later.
    if market_cap_value < 300_000_000:
        return "Micro Cap"
    if market_cap_value < 2_000_000_000:
        return "Small Cap"
    if market_cap_value < 10_000_000_000:
        return "Mid Cap"
    if market_cap_value < 200_000_000_000:
        return "Large Cap"
    return "Mega Cap"


def analyze_fundamentals(info: dict, current_price: float | None = None, risk_free_rate: float = 0.045) -> dict:
    if not info:
        return {"fundamental_score": 0, "flags": ["missing data"], "metrics": {"market_cap_tier": "unknown"}}

    trailing_pe = _safe_float(info.get("trailingPE"))
    forward_pe = _safe_float(info.get("forwardPE"))
    pe_used = forward_pe if (forward_pe and forward_pe > 0) else trailing_pe
    eps = _safe_float(info.get("trailingEps"))
    forward_eps = _safe_float(info.get("forwardEps"))
    growth = _safe_float(info.get("earningsGrowth"))
    if growth is None:
        growth = _safe_float(info.get("revenueGrowth"))
    dte = _safe_float(info.get("debtToEquity"))
    roe = _safe_float(info.get("returnOnEquity"))
    target = _safe_float(info.get("targetMeanPrice"))
    rec = info.get("recommendationKey")
    current_price = current_price or _safe_float(info.get("currentPrice")) or _safe_float(info.get("regularMarketPrice"))
    sector = info.get("sector")
    normalized_sector = normalize_sector_name(sector)
    market_cap = info.get("marketCap")
    market_cap_tier = classify_market_cap_tier(market_cap)

    score, flags = 50, []

    sector_benchmark_pe = float(SECTOR_BENCHMARK_PE.get(str(normalized_sector), 20.0))
    sector_pe_relative = (pe_used / sector_benchmark_pe) if pe_used and sector_benchmark_pe > 0 else None
    if sector_pe_relative is not None:
        if sector_pe_relative < 0.9:
            score += 8
            flags.append("sector-undervalued")
        elif sector_pe_relative > 1.25:
            score -= 8
            flags.append("sector-overvalued")

    growth_rate_pct = (growth * 100.0) if growth is not None else None
    peg_ratio = None
    if pe_used and growth_rate_pct and growth_rate_pct > 0:
        peg_ratio = pe_used / growth_rate_pct
        if peg_ratio < 1:
            score += 6
            flags.append("peg-undervalued")
        elif peg_ratio > 2:
            score -= 6
            flags.append("peg-expensive")

    eps_for_dcf = forward_eps if (forward_eps and forward_eps > 0) else eps
    dcf_estimate = None
    if eps_for_dcf and eps_for_dcf > 0 and growth_rate_pct is not None and risk_free_rate and risk_free_rate > 0:
        # Graham's revised intrinsic-value formula expects Y as a *percentage number*
        # (e.g. 4.4 meaning 4.4%), but risk_free_rate is a decimal fraction (e.g. 0.045)
        # everywhere else in this codebase (see modules/macro_regime.py). Multiply by 100
        # to convert to the percentage-number units the formula expects; otherwise the
        # 4.4/Y term inflates ~100x and blows past the ensemble's price-target safety cap.
        risk_free_rate_pct = risk_free_rate * 100.0
        dcf_estimate = eps_for_dcf * (8.5 + 2 * growth_rate_pct) * (4.4 / risk_free_rate_pct)

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
    high, low = _safe_float(info.get("fiftyTwoWeekHigh")), _safe_float(info.get("fiftyTwoWeekLow"))
    rel = None
    if high and low and current_price and high > low:
        rel = (current_price - low) / (high - low)

    logger.info("Fundamental analysis complete for sector=%s peg=%s dcf=%s", sector, peg_ratio, dcf_estimate)

    return {
        "fundamental_score": score,
        "flags": sorted(set(flags)),
        "metrics": {
            "trailing_pe": trailing_pe,
            "forward_pe": forward_pe,
            "eps": eps,
            "revenue_growth": _safe_float(info.get("revenueGrowth")),
            "earnings_growth": growth,
            "debt_to_equity": dte,
            "roe": roe,
            "market_cap": market_cap,
            "market_cap_tier": market_cap_tier,
            "sector": sector,
            "52w_high": high,
            "52w_low": low,
            "current_price": current_price,
            "price_relative_to_52w_range": rel,
            "dividend_yield": _safe_float(info.get("dividendYield")),
            "analyst_target_price": target,
            "analyst_recommendation": rec,
            "peg_ratio": round(float(peg_ratio), 3) if peg_ratio is not None else None,
            "dcf_estimate": round(float(dcf_estimate), 2) if dcf_estimate is not None else None,
            "sector_benchmark_pe": round(sector_benchmark_pe, 2),
            "sector_pe_relative": round(float(sector_pe_relative), 3) if sector_pe_relative is not None else None,
        },
    }
