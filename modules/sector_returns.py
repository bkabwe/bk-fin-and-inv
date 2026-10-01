"""Sector relative-strength feature table — a cross-sectional LightGBM input.

Computes each ticker's trailing return relative to its own GICS-style sector
(via the standard SPDR Select Sector ETFs), complementing the purely
single-ticker technical/fundamental/macro features in
modules/feature_engineering.py with a cross-sectional signal: "is this stock
outperforming its own sector lately, or lagging it?" -- something none of
the existing per-ticker features can express on their own.

Mirrors the existing modules/fred_client.py macro-feature-table pattern:
every sector ETF is fetched ONCE per run via build_sector_return_table() and
can be shared across an entire batch (passed as `shared_sector_return_table`,
the same way modules/feature_engineering.py's `shared_macro_table` is shared
across a batch run) instead of being re-fetched per ticker.
"""
from __future__ import annotations

import threading
import time

import pandas as pd

from modules.fundamental_analysis import normalize_sector_name
from modules.logger import get_logger

logger = get_logger(__name__)

try:
    import streamlit as st

    cache_data = st.cache_data
except Exception:  # pragma: no cover

    def cache_data(ttl: int | None = None):  # type: ignore[misc]
        """TTL-aware, stampede-safe cache fallback for non-Streamlit usage."""

        def decorator(func):
            _cache: dict = {}
            _inflight: dict = {}
            _lock = threading.Lock()

            def wrapper(*args, **kwargs):
                key = (args, tuple(sorted(kwargs.items())))
                while True:
                    now = time.monotonic()
                    with _lock:
                        entry = _cache.get(key)
                        if entry is not None:
                            value, ts = entry
                            if ttl is None or (now - ts) < ttl:
                                return value
                        event = _inflight.get(key)
                        if event is None:
                            ev = threading.Event()
                            _inflight[key] = ev
                            break
                    event.wait(timeout=300)

                try:
                    result = func(*args, **kwargs)
                    with _lock:
                        _cache[key] = (result, time.monotonic())
                    return result
                finally:
                    with _lock:
                        ev = _inflight.pop(key, None)
                    if ev is not None:
                        ev.set()

            return wrapper

        return decorator


# Standard SPDR Select Sector ETF mapping, covering every sector in
# modules.fundamental_analysis.SECTOR_BENCHMARK_PE (post normalize_sector_name()
# aliasing, e.g. "Financial Services" -> "Financials").
SECTOR_ETF_MAP: dict[str, str] = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Energy": "XLE",
    "Consumer Staples": "XLP",
    "Consumer Discretionary": "XLY",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

# Trailing-return lookback windows (trading days), matching the two
# intermediate-horizon windows commonly used elsewhere for momentum features.
RELATIVE_STRENGTH_WINDOWS: tuple[int, ...] = (21, 63)

# Sector ETF prices move far more slowly than a once-per-scan TTL needs to
# chase -- 4h keeps a single scan run's repeated lookups cheap without
# serving badly stale data across days.
_SECTOR_RETURN_CACHE_TTL_SECONDS = 4 * 60 * 60


def _normalize_daily_index(index_like) -> pd.DatetimeIndex:
    """Local copy of feature_engineering.normalize_daily_index's logic --
    duplicated rather than imported to avoid a circular import between this
    module and modules.feature_engineering (which imports from here)."""
    index = pd.DatetimeIndex(pd.to_datetime(index_like, errors="coerce"))
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    return index.normalize()


def sector_relative_strength_columns() -> list[str]:
    """Feature column names produced by compute_relative_strength_features,
    uniform across every ticker regardless of its actual sector (the value
    is always "this ticker's own-sector ETF", so a trained model sees a
    consistent column meaning for every row)."""
    columns: list[str] = []
    for window in RELATIVE_STRENGTH_WINDOWS:
        columns.append(f"sector_return_{window}d")
        columns.append(f"sector_relative_return_{window}d")
    return columns


@cache_data(ttl=_SECTOR_RETURN_CACHE_TTL_SECONDS)
def build_sector_return_table(period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Fetch every mapped SPDR sector ETF exactly once and compute trailing
    rolling returns for each, for use as a `shared_sector_return_table`
    shared across every ticker in a batch run -- never fetched per-ticker.

    Returns a DataFrame indexed by date with columns ``{ETF}_return_{window}d``
    for every ETF in SECTOR_ETF_MAP.values() and window in
    RELATIVE_STRENGTH_WINDOWS. Empty DataFrame if no ETF data is available
    (e.g. Polygon not configured).
    """
    from modules.data_fetcher import get_stock_data

    closes: dict[str, pd.Series] = {}
    for etf in sorted(set(SECTOR_ETF_MAP.values())):
        try:
            data = get_stock_data(etf, period=period, interval=interval)
            if data is None or data.empty or "Close" not in data:
                logger.warning("Sector ETF fetch returned no data for %s", etf)
                continue
            closes[etf] = data["Close"].astype(float)
        except Exception as exc:
            logger.warning("Sector ETF fetch failed for %s: %s", etf, exc)

    if not closes:
        return pd.DataFrame()

    close_table = pd.DataFrame(closes)
    close_table.index = _normalize_daily_index(close_table.index)
    close_table = close_table[~close_table.index.isna()].sort_index()
    close_table = close_table[~close_table.index.duplicated(keep="last")]

    returns = pd.DataFrame(index=close_table.index)
    for etf in close_table.columns:
        for window in RELATIVE_STRENGTH_WINDOWS:
            returns[f"{etf}_return_{window}d"] = close_table[etf].pct_change(periods=window)
    return returns


def compute_relative_strength_features(
    daily_index: pd.DatetimeIndex,
    close: pd.Series,
    sector: str | None,
    shared_sector_return_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-ticker sector relative-strength features, aligned to daily_index.

    ``close`` is the ticker's own daily Close series (same index convention
    as daily_index). ``sector`` is the ticker's raw sector label (e.g. from
    Polygon/Yahoo ticker info) -- normalized via
    modules.fundamental_analysis.normalize_sector_name and mapped to its
    SPDR ETF. If ``sector`` doesn't resolve to a known ETF, or ETF data is
    unavailable, returns an all-NaN frame (LightGBM handles NaN features
    natively) rather than raising or skipping the column entirely, so the
    output schema stays identical regardless of whether sector data was
    resolvable for this particular ticker.

    ``shared_sector_return_table`` should be the output of
    build_sector_return_table(), typically fetched once and shared across a
    whole batch run. When omitted, this function self-fetches via
    build_sector_return_table() (itself TTL-cached), matching
    modules.feature_engineering._macro_daily_features's self-fetch-when-no-
    shared-table fallback.
    """
    columns = sector_relative_strength_columns()
    features = pd.DataFrame(index=daily_index, columns=columns, dtype="float64")
    if len(daily_index) == 0:
        return features

    etf = SECTOR_ETF_MAP.get(normalize_sector_name(sector) or "")
    if etf is None:
        return features

    sector_table = shared_sector_return_table if shared_sector_return_table is not None else build_sector_return_table()
    if sector_table is None or sector_table.empty:
        return features

    aligned_close = close.reindex(daily_index).astype(float) if close is not None else pd.Series(dtype="float64", index=daily_index)

    for window in RELATIVE_STRENGTH_WINDOWS:
        etf_column = f"{etf}_return_{window}d"
        if etf_column not in sector_table.columns:
            continue
        sector_return = sector_table[etf_column].reindex(daily_index).ffill()
        stock_return = aligned_close.pct_change(periods=window)
        features[f"sector_return_{window}d"] = sector_return
        features[f"sector_relative_return_{window}d"] = stock_return - sector_return

    return features.apply(pd.to_numeric, errors="coerce").astype("float64")
