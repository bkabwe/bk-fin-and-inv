from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from modules.data_fetcher import get_stock_data
from modules.fundamental_analysis import normalize_sector_name
from modules.polygon_client import get_indicator_series


@dataclass
class LongTermSignals:
    primary_trend: str
    longterm_stage: str
    volume_trend_confirmation: str
    max_drawdown_pct: float | None
    recovery_days: int | None
    golden_cross_count: int
    death_cross_count: int
    favorable_cross_follow_through_pct: float | None
    longterm_technical_score: int



try:
    import streamlit as st

    cache_data = st.cache_data
except Exception:  # pragma: no cover
    from functools import lru_cache

    def cache_data(ttl: int | None = None):
        def decorator(func):
            return lru_cache(maxsize=128)(func)

        return decorator


@cache_data(ttl=3600)
def _get_benchmark_close(symbol: str) -> pd.Series:
    df = get_stock_data(symbol, period="5y", interval="1d")
    if df.empty or "Close" not in df:
        return pd.Series(dtype=float)
    return df["Close"].astype(float).dropna()


_STAGE_MA_SPREAD_THRESHOLD = 0.01
_STAGE_PRICE_DISTANCE_THRESHOLD = 0.06
_STAGE_RS_BASELINE = 1.0

_SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Industrials": "XLI",
}


def _safe_last(value: pd.Series) -> float | None:
    if value is None or value.empty:
        return None
    try:
        return float(value.dropna().iloc[-1])
    except Exception:
        return None


def _latest_sma_from_polygon(ticker: str, window: int, close: pd.Series) -> float | None:
    non_null_close = close.dropna()
    if len(non_null_close) >= window:
        local = non_null_close.rolling(window).mean().iloc[-1]
        if not pd.isna(local):
            return float(local)
    series = get_indicator_series("sma", ticker, window=window)
    if not series.empty and "value" in series:
        return _safe_last(series["value"])
    return None


def _sma_history_from_polygon(ticker: str, window: int, close: pd.Series, *, min_local_history: int | None = None) -> pd.Series:
    non_null_close = close.dropna()
    threshold = max(window, int(min_local_history or 0))
    if len(non_null_close) >= threshold:
        return non_null_close.rolling(window).mean().astype(float)
    series = get_indicator_series("sma", ticker, window=window)
    if not series.empty and "value" in series:
        return series["value"].astype(float)
    return non_null_close.rolling(window).mean().astype(float)


def _primary_trend(close: pd.Series, sma150: float | None, sma200: float | None) -> str:
    if close.empty:
        return "insufficient_data"
    current = float(close.iloc[-1])
    if sma150 is None or sma200 is None:
        return "insufficient_data"

    last_252 = close.tail(min(252, len(close)))
    prev_252 = close.iloc[-504:-252] if len(close) >= 504 else close.head(max(len(close) // 2, 1))
    high_break = not prev_252.empty and float(last_252.max()) > float(prev_252.max())
    low_break = not prev_252.empty and float(last_252.min()) > float(prev_252.min())

    if current > sma150 > sma200 and high_break and low_break:
        return "primary_bull"
    if current < sma150 < sma200 and not low_break:
        return "primary_bear"
    return "consolidation_basing"


def _relative_strength_ratio(stock: pd.Series, benchmark: pd.Series, lookback: int = 126) -> float | None:
    merged = pd.concat([stock.rename("s"), benchmark.rename("b")], axis=1).dropna()
    if merged.empty:
        return None

    trailing_days = max(30, int(round((lookback / 252.0) * 365.0)))
    try:
        cutoff = pd.to_datetime(merged.index).max() - pd.Timedelta(days=trailing_days)
        merged = merged.loc[pd.to_datetime(merged.index) >= cutoff]
    except Exception:
        pass

    if len(merged) < 2:
        return None
    sret = (merged["s"].iloc[-1] / merged["s"].iloc[0]) - 1
    bret = (merged["b"].iloc[-1] / merged["b"].iloc[0]) - 1
    if abs(float(bret)) < 1e-9:
        return None
    return float((1 + sret) / (1 + bret))


def _classify_stage(
    close: pd.Series,
    sma150: float | None,
    sma200: float | None,
    rs_spy: float | None,
    rs_sector: float | None,
) -> str:
    if close.empty or sma150 is None or sma200 is None:
        return "Stage Unknown"
    current = float(close.iloc[-1])

    ma_spread = (sma150 - sma200) / max(abs(sma200), 1e-9)
    sector_rs_ok = rs_sector is None or rs_sector >= 1.0

    if current > sma150 > sma200 and ma_spread > _STAGE_MA_SPREAD_THRESHOLD and (rs_spy or 0) >= _STAGE_RS_BASELINE and sector_rs_ok:
        return "Stage 2 (Advancing)"
    if current < sma150 < sma200 and ma_spread < -_STAGE_MA_SPREAD_THRESHOLD and (rs_spy or _STAGE_RS_BASELINE) < _STAGE_RS_BASELINE:
        return "Stage 4 (Declining)"
    if abs(ma_spread) <= _STAGE_MA_SPREAD_THRESHOLD and abs((current - sma200) / max(abs(sma200), 1e-9)) < _STAGE_PRICE_DISTANCE_THRESHOLD:
        return "Stage 1 (Basing)"
    return "Stage 3 (Topping)"


def _volume_confirmation(close: pd.Series, volume: pd.Series) -> str:
    if len(close) < 126 or len(volume) < 126:
        return "insufficient_data"
    recent_price_ret = (close.iloc[-1] / close.iloc[-63]) - 1
    recent_vol = float(volume.tail(63).mean())
    prior_vol = float(volume.iloc[-126:-63].mean())
    vol_delta = (recent_vol / prior_vol) - 1 if prior_vol > 0 else 0

    if recent_price_ret > 0 and vol_delta > 0.05:
        return "accumulation"
    if recent_price_ret > 0 and vol_delta < -0.05:
        return "weakening_uptrend"
    if recent_price_ret < 0 and vol_delta > 0.05:
        return "distribution"
    return "neutral"


def _drawdown_resilience(close: pd.Series) -> tuple[float | None, int | None]:
    if close.empty:
        return None, None
    running_max = close.cummax()
    drawdowns = (close / running_max) - 1.0
    max_dd = float(drawdowns.min())

    trough_idx = int(drawdowns.values.argmin()) if len(drawdowns) else 0
    trough_price = float(close.iloc[trough_idx]) if len(close) else 0.0
    prior_peak = float(running_max.iloc[trough_idx]) if len(running_max) else 0.0
    recovery_days = None
    if prior_peak > 0 and trough_idx < len(close) - 1:
        recovered = close.iloc[trough_idx + 1 :] >= prior_peak
        if bool(recovered.any()):
            first = recovered[recovered].index[0]
            recovery_days = int((first - close.index[trough_idx]).days)

    _ = trough_price
    return max_dd * 100.0, recovery_days


def _cross_history(close: pd.Series, ticker: str) -> tuple[int, int, float | None]:
    if len(close) < 220:
        return 0, 0, None

    sma50 = _sma_history_from_polygon(ticker, 50, close, min_local_history=220)
    sma200 = _sma_history_from_polygon(ticker, 200, close, min_local_history=220)
    merged = pd.concat([close.rename("close"), sma50.rename("sma50"), sma200.rename("sma200")], axis=1).dropna()
    if len(merged) < 220:
        return 0, 0, None

    prev = (merged["sma50"].shift(1) - merged["sma200"].shift(1)).dropna()
    cur = (merged["sma50"] - merged["sma200"]).loc[prev.index]

    golden = ((prev <= 0) & (cur > 0))
    death = ((prev >= 0) & (cur < 0))

    golden_idx = list(merged.loc[golden].index)
    death_idx = list(merged.loc[death].index)

    favorable = 0
    total_eval = 0
    for idx in golden_idx:
        loc = merged.index.get_loc(idx)
        end = min(len(merged) - 1, loc + 63)
        if end <= loc:
            continue
        total_eval += 1
        if float(merged["close"].iloc[end]) > float(merged["close"].iloc[loc]):
            favorable += 1

    follow_through = (favorable / total_eval) * 100.0 if total_eval else None
    return int(golden.sum()), int(death.sum()), follow_through


def analyze_longterm_technical_score(ticker: str, sector: str | None = None, data: pd.DataFrame | None = None) -> dict:
    history = data if data is not None else get_stock_data(ticker, period="5y", interval="1d")
    if history is None or history.empty or "Close" not in history:
        return {
            "longterm_technical_score": 0,
            "longterm_stage": "Stage Unknown",
            "primary_trend": "insufficient_data",
            "volume_trend_confirmation": "insufficient_data",
            "max_drawdown_pct": None,
            "recovery_days": None,
            "golden_cross_count": 0,
            "death_cross_count": 0,
            "favorable_cross_follow_through_pct": None,
        }

    close = history["Close"].astype(float).dropna()
    volume = history["Volume"].astype(float).fillna(0) if "Volume" in history else pd.Series(dtype=float)

    sma150 = _latest_sma_from_polygon(ticker, 150, close)
    sma200 = _latest_sma_from_polygon(ticker, 200, close)

    trend = _primary_trend(close, sma150, sma200)

    spy_close = _get_benchmark_close("SPY")
    rs_spy = _relative_strength_ratio(close, spy_close) if not spy_close.empty else None

    sector_symbol = _SECTOR_ETF_MAP.get(normalize_sector_name(sector))
    rs_sector = None
    if sector_symbol:
        sector_close = _get_benchmark_close(sector_symbol)
        if not sector_close.empty:
            rs_sector = _relative_strength_ratio(close, sector_close)

    stage = _classify_stage(close, sma150, sma200, rs_spy, rs_sector)
    vol_signal = _volume_confirmation(close, volume)
    max_dd, recovery_days = _drawdown_resilience(close)
    golden_count, death_count, follow_pct = _cross_history(close, ticker)

    score = 0
    score += 8 if trend == "primary_bull" else 2 if trend == "consolidation_basing" else 0
    score += 6 if stage.startswith("Stage 2") else 3 if stage.startswith("Stage 1") else 1 if stage.startswith("Stage 3") else 0
    score += 3 if vol_signal == "accumulation" else 1 if vol_signal == "neutral" else 0
    if max_dd is not None:
        score += 4 if max_dd >= -25 else 2 if max_dd >= -40 else 0
    if recovery_days is not None:
        score += 2 if recovery_days <= 252 else 1 if recovery_days <= 504 else 0
    if follow_pct is not None:
        score += 3 if follow_pct >= 60 else 1 if follow_pct >= 45 else 0
    if death_count > golden_count:
        score -= 2

    score = max(0, min(20, int(round(score))))

    return {
        "longterm_technical_score": score,
        "longterm_stage": stage,
        "primary_trend": trend,
        "volume_trend_confirmation": vol_signal,
        "max_drawdown_pct": round(float(max_dd), 2) if max_dd is not None else None,
        "recovery_days": recovery_days,
        "golden_cross_count": golden_count,
        "death_cross_count": death_count,
        "favorable_cross_follow_through_pct": round(float(follow_pct), 2) if follow_pct is not None else None,
    }
