from __future__ import annotations

from functools import lru_cache

import pandas as pd
import yfinance as yf

from modules.logger import get_logger

logger = get_logger(__name__)

try:
    import streamlit as st

    cache_data = st.cache_data
except Exception:  # pragma: no cover
    def cache_data(ttl: int | None = None):
        def decorator(func):
            return func

        return decorator


@cache_data(ttl=3600)
def get_macro_regime() -> dict:
    default = {
        "vix": None,
        "vix_regime": "medium",
        "yield_10y": None,
        "risk_free_rate": 0.045,
        "bullish_sectors": [],
        "bearish_sectors": [],
        "market_regime": "neutral",
    }
    try:
        vix_df = yf.download("^VIX", period="1mo", interval="1d", progress=False, auto_adjust=False)
        tnx_df = yf.download("^TNX", period="1mo", interval="1d", progress=False, auto_adjust=False)

        vix = float(vix_df["Close"].dropna().iloc[-1]) if not vix_df.empty and "Close" in vix_df else None
        y10 = float(tnx_df["Close"].dropna().iloc[-1]) if not tnx_df.empty and "Close" in tnx_df else None

        sector_map = {
            "SPY": "SPY",
            "XLK": "Technology",
            "XLF": "Financials",
            "XLE": "Energy",
            "XLV": "Healthcare",
            "XLI": "Industrials",
        }
        bullish: list[str] = []
        bearish: list[str] = []

        for symbol, label in sector_map.items():
            data = yf.download(symbol, period="6mo", interval="1d", progress=False, auto_adjust=False)
            if data.empty or "Close" not in data:
                continue
            close = data["Close"].astype(float)
            sma20 = close.rolling(20).mean().iloc[-1]
            sma50 = close.rolling(50).mean().iloc[-1]
            if pd.isna(sma20) or pd.isna(sma50):
                continue
            if sma20 > sma50:
                bullish.append(label)
            else:
                bearish.append(label)

        vix_regime = "medium"
        market_regime = "neutral"
        if vix is not None:
            if vix < 15:
                vix_regime = "low"
                market_regime = "risk_on"
            elif vix > 25:
                vix_regime = "high"
                market_regime = "risk_off"

        # Slightly refine with sector breadth when VIX is medium
        if vix_regime == "medium":
            if len(bullish) >= 4:
                market_regime = "risk_on"
            elif len(bearish) >= 4:
                market_regime = "risk_off"

        risk_free_rate = (y10 / 100.0) if y10 is not None else 0.045

        result = {
            "vix": round(vix, 2) if vix is not None else None,
            "vix_regime": vix_regime,
            "yield_10y": round(y10, 2) if y10 is not None else None,
            "risk_free_rate": round(float(risk_free_rate), 5),
            "bullish_sectors": bullish,
            "bearish_sectors": bearish,
            "market_regime": market_regime,
        }
        logger.info("Macro regime computed: %s", result.get("market_regime"))
        return result
    except Exception as exc:
        logger.warning("Macro regime calculation failed: %s", exc)
        return default
