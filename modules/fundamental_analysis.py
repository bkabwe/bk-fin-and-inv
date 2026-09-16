from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from modules.logger import get_logger

logger = get_logger(__name__)

# Static fallback sector P/E benchmarks, used whenever a fresher value isn't
# available from data/sector_benchmark_pe.json (see get_sector_benchmark_pe
# and scripts/refresh_sector_pe.py below). Sector multiples drift with rate
# cycles, so treat these as a last-resort default, not a source of truth --
# refresh the JSON file periodically (e.g. quarterly) by re-running
# `python scripts/refresh_sector_pe.py`.
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

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SECTOR_BENCHMARK_PE_FILE = DATA_DIR / "sector_benchmark_pe.json"


@lru_cache(maxsize=4)
def _load_refreshed_sector_pe(mtime: float) -> dict[str, float]:
    """Parse data/sector_benchmark_pe.json, keyed by its mtime so edits/refreshes
    (see scripts/refresh_sector_pe.py) are picked up without a process restart,
    while repeated calls within the same file version are served from cache."""
    try:
        payload = json.loads(SECTOR_BENCHMARK_PE_FILE.read_text(encoding="utf-8"))
        sectors = payload.get("sectors", {})
        return {str(name): float(pe) for name, pe in sectors.items() if pe is not None}
    except Exception as exc:
        logger.warning("Failed to parse %s: %s", SECTOR_BENCHMARK_PE_FILE, exc)
        return {}


def get_sector_benchmark_pe(sector: str | None, default: float = 20.0) -> float:
    """Sector P/E benchmark used for sector-relative valuation.

    Prefers a refreshed value computed from the current ticker universe (see
    scripts/refresh_sector_pe.py, written to data/sector_benchmark_pe.json),
    falling back to the static SECTOR_BENCHMARK_PE defaults above, and finally
    to `default` for an unrecognized sector. This is the refresh mechanism
    for the previously-static-only benchmark: run the script periodically
    (e.g. quarterly) to keep values current; the app works unmodified if the
    file is absent or stale.
    """
    key = str(sector)
    try:
        mtime = SECTOR_BENCHMARK_PE_FILE.stat().st_mtime
    except OSError:
        return float(SECTOR_BENCHMARK_PE.get(key, default))
    refreshed = _load_refreshed_sector_pe(mtime)
    if key in refreshed:
        return float(refreshed[key])
    return float(SECTOR_BENCHMARK_PE.get(key, default))


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

    sector_benchmark_pe = get_sector_benchmark_pe(normalized_sector)
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
        #
        # KNOWN LIMITATION (methodology, independent of the unit fix above): this is
        # still Graham's unmodified 1962 heuristic (8.5 + 2*growth), which is known to
        # be unreliable for high-growth, negative-earnings, or cyclical names. It is
        # kept as one ensemble input among several rather than a standalone valuation;
        # a multi-stage DCF with an explicit discount rate would be a more defensible
        # (but materially more involved) replacement if this weight ever needs to grow.
        risk_free_rate_pct = risk_free_rate * 100.0
        dcf_estimate = eps_for_dcf * (8.5 + 2 * growth_rate_pct) * (4.4 / risk_free_rate_pct)

    # Comparables-based valuation: EPS x current sector peer P/E multiple.
    # Supplements (rather than replaces) the Graham DCF heuristic above with
    # an independent cross-check that doesn't depend on the growth-rate input
    # DCF is most sensitive to, so the two can disagree usefully on
    # high-growth/negative-earnings names where Graham's formula is weakest.
    comparable_value_estimate = None
    if eps_for_dcf and eps_for_dcf > 0 and sector_benchmark_pe > 0:
        comparable_value_estimate = eps_for_dcf * sector_benchmark_pe

    # Blended valuation_estimate is what the scoring ensemble consumes (see
    # modules/scoring_engine.py) in place of the raw DCF number, so the
    # "DCF" ensemble slot now reflects both approaches rather than Graham's
    # heuristic alone.
    valuation_inputs = [v for v in (dcf_estimate, comparable_value_estimate) if v is not None]
    valuation_estimate = (sum(valuation_inputs) / len(valuation_inputs)) if valuation_inputs else None

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

    logger.info(
        "Fundamental analysis complete for sector=%s peg=%s dcf=%s comps=%s",
        sector,
        peg_ratio,
        dcf_estimate,
        comparable_value_estimate,
    )

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
            "comparable_value_estimate": round(float(comparable_value_estimate), 2)
            if comparable_value_estimate is not None
            else None,
            "valuation_estimate": round(float(valuation_estimate), 2) if valuation_estimate is not None else None,
            "sector_benchmark_pe": round(sector_benchmark_pe, 2),
            "sector_pe_relative": round(float(sector_pe_relative), 3) if sector_pe_relative is not None else None,
        },
    }
