from __future__ import annotations

import threading

import pandas as pd

from modules.fred_client import FredNotConfiguredError, get_vix_observations
from modules.logger import get_logger
from modules.polygon_client import PolygonNotConfiguredError, get_stock_data_polygon

logger = get_logger(__name__)

try:
    import streamlit as st

    cache_data = st.cache_data
except Exception:  # pragma: no cover
    import threading as _threading
    import time as _time

    def cache_data(ttl: int | None = None):  # type: ignore[misc]
        """TTL-aware, stampede-safe cache fallback for non-Streamlit deployments."""

        def decorator(func):
            _cache: dict = {}
            _inflight: dict = {}
            _lock = _threading.Lock()

            def wrapper(*args, **kwargs):
                key = (args, tuple(sorted(kwargs.items())))
                while True:
                    now = _time.monotonic()
                    with _lock:
                        entry = _cache.get(key)
                        if entry is not None:
                            value, ts = entry
                            if ttl is None or (now - ts) < ttl:
                                return value
                        event = _inflight.get(key)
                        if event is None:
                            ev = _threading.Event()
                            _inflight[key] = ev
                            break
                    event.wait(timeout=300)

                try:
                    result = func(*args, **kwargs)
                    with _lock:
                        _cache[key] = (result, _time.monotonic())
                    return result
                finally:
                    with _lock:
                        ev = _inflight.pop(key, None)
                    if ev is not None:
                        ev.set()

            return wrapper

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
        lookback_end = pd.Timestamp.now(tz="UTC").normalize()
        lookback_start = lookback_end - pd.Timedelta(days=30)
        vix_df = get_vix_observations(lookback_start.date(), lookback_end.date())
        tnx_df = get_stock_data_polygon("I:TNX", period="1mo", interval="1d")

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
            data = get_stock_data_polygon(symbol, period="6mo", interval="1d")
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
    except (FredNotConfiguredError, PolygonNotConfiguredError) as exc:
        logger.warning("Macro regime unavailable: %s", exc)
        return default
    except Exception as exc:
        logger.warning("Macro regime calculation failed: %s", exc)
        return default
