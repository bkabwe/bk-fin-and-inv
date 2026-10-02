from __future__ import annotations

import pandas as pd

from modules.fred_client import (
    FRED_10Y_TREASURY_SERIES_ID,
    FredNotConfiguredError,
    get_series_observations,
    get_vix_observations,
)
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


# A missing API key is a deployment misconfiguration rather than a transient
# upstream failure, so it still collapses the whole regime to its defaults
# instead of being swallowed per component below.
_NOT_CONFIGURED_ERRORS = (FredNotConfiguredError, PolygonNotConfiguredError)

_SECTOR_ETF_LABELS = {
    "SPY": "SPY",
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLI": "Industrials",
}


def _latest_value(frame: pd.DataFrame | None, column: str) -> float | None:
    if frame is None or frame.empty or column not in frame:
        return None
    values = frame[column].dropna()
    return float(values.iloc[-1]) if not values.empty else None


def _fetch_vix(start: pd.Timestamp, end: pd.Timestamp) -> float | None:
    try:
        return _latest_value(get_vix_observations(start.date(), end.date()), "Close")
    except _NOT_CONFIGURED_ERRORS:
        raise
    except Exception as exc:
        logger.warning("Macro regime: VIX unavailable: %s", exc)
        return None


def _fetch_10y_yield(start: pd.Timestamp, end: pd.Timestamp) -> float | None:
    # FRED's DGS10 (10-Year Treasury Constant Maturity, in percent) is free
    # with the same FRED_API_KEY already used for VIX. It replaces Polygon's
    # "I:TNX" index ticker, which needs a separate paid Indices subscription
    # and returned HTTP 403 on the Stocks-only plan.
    try:
        return _latest_value(
            get_series_observations(FRED_10Y_TREASURY_SERIES_ID, start.date(), end.date()),
            "value",
        )
    except _NOT_CONFIGURED_ERRORS:
        raise
    except Exception as exc:
        logger.warning("Macro regime: 10Y Treasury yield unavailable: %s", exc)
        return None


def _fetch_sector_trends() -> tuple[list[str], list[str]]:
    bullish: list[str] = []
    bearish: list[str] = []
    for symbol, label in _SECTOR_ETF_LABELS.items():
        try:
            data = get_stock_data_polygon(symbol, period="6mo", interval="1d")
        except _NOT_CONFIGURED_ERRORS:
            raise
        except Exception as exc:
            logger.warning("Macro regime: %s sector trend unavailable: %s", symbol, exc)
            continue
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
    return bullish, bearish


@cache_data(ttl=3600)
def get_macro_regime() -> dict:
    """Summarize the market regime from VIX, the 10Y yield and sector-ETF trends.

    Each input is fetched independently, so a transient failure in one (e.g.
    a FRED outage, or a single ETF's Polygon request failing) leaves the
    others intact: the unavailable pieces keep their neutral defaults instead
    of the whole regime collapsing back to the all-default result. A missing
    API key (FRED or Polygon) is not transient, so it returns the defaults.
    """
    default = {
        "vix": None,
        "vix_regime": "medium",
        "yield_10y": None,
        "risk_free_rate": 0.045,
        "bullish_sectors": [],
        "bearish_sectors": [],
        "market_regime": "neutral",
    }
    lookback_end = pd.Timestamp.now(tz="UTC").normalize()
    lookback_start = lookback_end - pd.Timedelta(days=30)

    try:
        vix = _fetch_vix(lookback_start, lookback_end)
        y10 = _fetch_10y_yield(lookback_start, lookback_end)
        bullish, bearish = _fetch_sector_trends()
    except _NOT_CONFIGURED_ERRORS as exc:
        logger.warning("Macro regime unavailable: %s", exc)
        return default

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
