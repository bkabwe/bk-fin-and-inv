from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd

from modules import fred_client, sec_edgar_client
from modules.logger import get_logger
from modules.validators import sanitize_ticker

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

            def _stable_value(value):
                if isinstance(value, float) and np.isnan(value):
                    return "__nan__"
                if isinstance(value, tuple):
                    return tuple(_stable_value(item) for item in value)
                if isinstance(value, list):
                    return tuple(_stable_value(item) for item in value)
                if isinstance(value, dict):
                    return tuple(sorted((k, _stable_value(v)) for k, v in value.items()))
                return value

            def wrapper(*args, **kwargs):
                key = (_stable_value(args), _stable_value(tuple(sorted(kwargs.items()))))
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


_FEATURE_CACHE_TTL_SECONDS = 21600
_FUNDAMENTAL_COLUMNS = [
    "fundamental_gross_margin",
    "fundamental_debt_to_equity",
    "fundamental_revenue_growth",
]
_MACRO_PREFIXES = ("dgs10", "cpiaucsl", "fedfunds")


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(float(denominator)) <= 1e-12:
        return None
    return float(numerator) / float(denominator)


def _safe_float_series(values: tuple[float | int | None, ...]) -> pd.Series:
    return pd.Series(pd.to_numeric(list(values), errors="coerce"), dtype="float64")


def normalize_daily_index(index_like) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(index_like, errors="coerce"))
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    return index.normalize()


def _normalize_price_frame(price_data: pd.DataFrame) -> pd.DataFrame:
    if price_data is None or price_data.empty:
        return pd.DataFrame(columns=["Close", "High", "Low", "Volume"])
    frame = price_data.copy()
    for col in ["Close", "High", "Low", "Volume"]:
        if col not in frame.columns:
            frame[col] = np.nan
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame.index = normalize_daily_index(frame.index)
    frame = frame[~frame.index.isna()].sort_index()
    return frame[["Close", "High", "Low", "Volume"]]


def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    valid = avg_gain.notna() & avg_loss.notna()
    rsi = rsi.where(~(valid & (avg_loss == 0) & (avg_gain > 0)), 100.0)
    rsi = rsi.where(~(valid & (avg_gain == 0) & (avg_loss > 0)), 0.0)
    rsi = rsi.where(~(valid & (avg_gain == 0) & (avg_loss == 0)), 50.0)
    return rsi


def _technical_features(price_frame: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=price_frame.index.copy())
    close = price_frame["Close"].astype(float)
    high = price_frame["High"].astype(float)
    low = price_frame["Low"].astype(float)
    volume = price_frame["Volume"].astype(float)
    returns = close.pct_change()

    features["technical_volatility_10d"] = returns.rolling(10).std()
    features["technical_volatility_30d"] = returns.rolling(30).std()
    features["technical_ema_12d"] = close.ewm(span=12, adjust=False, min_periods=12).mean()
    features["technical_ema_26d"] = close.ewm(span=26, adjust=False, min_periods=26).mean()
    features["technical_rsi_14d"] = _compute_rsi(close, window=14)

    # Daily-bar VWAP approximation: cumulative typical-price*volume / cumulative volume.
    # True VWAP is intraday; this approximation is used because the app currently works with daily OHLCV bars.
    typical_price = (high + low + close) / 3.0
    cumulative_volume = volume.cumsum().replace(0, np.nan)
    features["technical_vwap_daily_approx"] = (typical_price * volume).cumsum() / cumulative_volume

    rolling_vol_avg_20 = volume.rolling(20).mean()
    features["technical_volume_vs_avg_20d"] = volume / rolling_vol_avg_20.replace(0, np.nan)
    return features


def _select_concept_series(company_facts: dict, concept_name: str) -> pd.Series:
    entries = sec_edgar_client.get_concept_entries(company_facts, concept_name)
    if not entries:
        return pd.Series(dtype="float64")
    rows: list[dict[str, float | pd.Timestamp]] = []
    for entry in sorted(entries, key=lambda item: (item["filed"], item["end"])):
        filed = pd.to_datetime(entry.get("filed"), errors="coerce")
        if pd.isna(filed):
            continue
        value = entry.get("value", entry.get("val"))
        if value is None:
            continue
        rows.append({"filed": filed.normalize(), "value": float(value)})
    if not rows:
        return pd.Series(dtype="float64")
    df = pd.DataFrame(rows).sort_values("filed")
    df = df.drop_duplicates(subset=["filed"], keep="last")
    return pd.Series(df["value"].values, index=pd.DatetimeIndex(df["filed"]))


def _fundamental_daily_features(ticker: str, daily_index: pd.DatetimeIndex) -> pd.DataFrame:
    features = pd.DataFrame(index=daily_index, columns=_FUNDAMENTAL_COLUMNS, dtype="float64")
    try:
        company_facts = sec_edgar_client.get_company_facts(ticker)
    except Exception as exc:
        logger.warning("Fundamental features unavailable for %s: %s", ticker, exc)
        return features

    if not company_facts:
        logger.warning("Fundamental features unavailable for %s: no SEC company facts", ticker)
        return features

    revenue = _select_concept_series(company_facts, "revenue")
    liabilities = _select_concept_series(company_facts, "liabilities")
    equity = _select_concept_series(company_facts, "equity")
    gross_profit = _select_concept_series(company_facts, "gross_profit")

    filed_features = pd.DataFrame(index=revenue.index.union(liabilities.index).union(equity.index).union(gross_profit.index).sort_values())
    filed_features["gross_margin"] = [
        _safe_div(gp, rev) for gp, rev in zip(gross_profit.reindex(filed_features.index), revenue.reindex(filed_features.index), strict=False)
    ]
    debt_to_equity_values: list[float] = []
    for liab, eq in zip(liabilities.reindex(filed_features.index), equity.reindex(filed_features.index), strict=False):
        ratio = _safe_div(liab, eq)
        # Keep SEC adapter compatibility: debtToEquity is represented as percent points (ratio * 100).
        debt_to_equity_values.append((ratio * 100.0) if ratio is not None else np.nan)
    filed_features["debt_to_equity"] = debt_to_equity_values
    filed_features["revenue_growth"] = revenue.reindex(filed_features.index).pct_change(periods=1)

    if filed_features.empty:
        return features

    expanded_index = daily_index.union(filed_features.index).sort_values()
    daily_filled = filed_features.reindex(expanded_index).ffill().reindex(daily_index)
    features["fundamental_gross_margin"] = daily_filled["gross_margin"]
    features["fundamental_debt_to_equity"] = daily_filled["debt_to_equity"]
    features["fundamental_revenue_growth"] = daily_filled["revenue_growth"]
    return features


def _macro_columns() -> list[str]:
    columns: list[str] = []
    for prefix in _MACRO_PREFIXES:
        columns.append(f"macro_{prefix}_level")
        for window in (5, 30):
            columns.append(f"macro_{prefix}_delta_{window}d")
            columns.append(f"macro_{prefix}_pct_change_{window}d")
    return columns


def prepare_shared_macro_feature_table(macro_table: pd.DataFrame | None) -> pd.DataFrame:
    if macro_table is None or macro_table.empty:
        return pd.DataFrame(columns=_macro_columns())
    prepared = macro_table.copy()
    prepared.index = normalize_daily_index(prepared.index)
    prepared = prepared[~prepared.index.isna()].sort_index()
    prepared = prepared[~prepared.index.duplicated(keep="last")]
    prepared = prepared.rename(columns={col: f"macro_{col}" for col in prepared.columns if not str(col).startswith("macro_")})
    return prepared


def _macro_daily_features(daily_index: pd.DatetimeIndex, shared_macro_table: pd.DataFrame | None = None) -> pd.DataFrame:
    feature_columns = _macro_columns()
    features = pd.DataFrame(index=daily_index, columns=feature_columns, dtype="float64")
    if len(daily_index) == 0:
        return features
    if shared_macro_table is not None:
        prepared = prepare_shared_macro_feature_table(shared_macro_table)
        if prepared.empty:
            return features
        return prepared.reindex(daily_index).ffill().reindex(columns=feature_columns)
    start_date = daily_index.min().date()
    end_date = daily_index.max().date()
    try:
        macro = fred_client.get_macro_feature_table(start_date, end_date, realtime_end=end_date)
    except Exception as exc:
        logger.warning("Macro features unavailable: %s", exc)
        return features
    prepared = prepare_shared_macro_feature_table(macro)
    if prepared.empty:
        return features
    return prepared.reindex(daily_index).ffill().reindex(columns=feature_columns)


@cache_data(ttl=_FEATURE_CACHE_TTL_SECONDS)
def _build_feature_table_cached(
    ticker: str,
    lookback_days: int,
    dates: tuple[str, ...],
    close_values: tuple[float | int | None, ...],
    high_values: tuple[float | int | None, ...],
    low_values: tuple[float | int | None, ...],
    volume_values: tuple[float | int | None, ...],
) -> pd.DataFrame:
    index = pd.DatetimeIndex(pd.to_datetime(list(dates), errors="coerce")).dropna().sort_values()
    if len(index) == 0:
        return pd.DataFrame()

    price_frame = pd.DataFrame(index=index)
    price_frame["Close"] = _safe_float_series(close_values).values
    price_frame["High"] = _safe_float_series(high_values).values
    price_frame["Low"] = _safe_float_series(low_values).values
    price_frame["Volume"] = _safe_float_series(volume_values).values

    table = _technical_features(price_frame)
    table = table.join(_fundamental_daily_features(ticker, table.index), how="left")
    table = table.join(_macro_daily_features(table.index), how="left")
    table = table.apply(pd.to_numeric, errors="coerce").astype("float64")
    table.index.name = "Date"

    if lookback_days > 0 and len(table) > lookback_days:
        table = table.tail(int(lookback_days))
    return table


def _build_feature_table_uncached(
    ticker: str,
    price_data: pd.DataFrame,
    *,
    lookback_days: int,
    shared_macro_table: pd.DataFrame | None,
) -> pd.DataFrame:
    normalized = _normalize_price_frame(price_data)
    if normalized.empty:
        logger.warning("Feature engineering skipped for %s: empty price data", ticker)
        return pd.DataFrame()
    table = _technical_features(normalized)
    table = table.join(_fundamental_daily_features(ticker, table.index), how="left")
    table = table.join(_macro_daily_features(table.index, shared_macro_table=shared_macro_table), how="left")
    table = table.apply(pd.to_numeric, errors="coerce").astype("float64")
    table.index.name = "Date"
    if lookback_days > 0 and len(table) > lookback_days:
        table = table.tail(int(lookback_days))
    return table


def build_feature_table(
    ticker: str,
    price_data: pd.DataFrame,
    lookback_days: int = 252,
    *,
    shared_macro_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    try:
        clean_ticker = sanitize_ticker(ticker)
    except Exception:
        clean_ticker = str(ticker or "").strip().upper()
        logger.warning("Feature engineering received invalid ticker format: %s", ticker)

    normalized = _normalize_price_frame(price_data)
    if normalized.empty:
        logger.warning("Feature engineering skipped for %s: empty price data", clean_ticker or ticker)
        return pd.DataFrame()

    if shared_macro_table is not None:
        return _build_feature_table_uncached(
            clean_ticker,
            normalized,
            lookback_days=int(lookback_days),
            shared_macro_table=shared_macro_table,
        )

    dates = tuple(pd.DatetimeIndex(normalized.index).strftime("%Y-%m-%d").tolist())
    return _build_feature_table_cached(
        clean_ticker,
        int(lookback_days),
        dates,
        tuple(normalized["Close"].tolist()),
        tuple(normalized["High"].tolist()),
        tuple(normalized["Low"].tolist()),
        tuple(normalized["Volume"].tolist()),
    )
