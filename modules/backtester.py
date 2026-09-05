from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from modules.logger import get_logger

logger = get_logger(__name__)

try:
    from statsmodels.tsa.arima.model import ARIMA

    ARIMA_AVAILABLE = True
except Exception:  # pragma: no cover
    ARIMA_AVAILABLE = False

try:
    from sklearn.linear_model import LinearRegression

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    SKLEARN_AVAILABLE = False

try:
    import streamlit as st

    cache_data = st.cache_data
except Exception:  # pragma: no cover
    import threading as _threading
    import time as _time

    def cache_data(ttl: int | None = None):  # type: ignore[misc]
        """TTL-aware, stampede-safe cache fallback for non-Streamlit (FastAPI) deployments."""

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


def _rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    if len(actual) == 0 or len(pred) == 0:
        return float("inf")
    n = min(len(actual), len(pred))
    return float(np.sqrt(np.mean((actual[:n] - pred[:n]) ** 2)))


def _select_arima_order(series: pd.Series) -> tuple[int, int, int]:
    if not ARIMA_AVAILABLE:
        return (5, 1, 0)
    best_order = (5, 1, 0)
    best_aic = float("inf")
    for p in range(6):
        for d in range(2):
            for q in range(3):
                try:
                    fit = ARIMA(series, order=(p, d, q)).fit()
                    if fit.aic < best_aic:
                        best_aic = float(fit.aic)
                        best_order = (p, d, q)
                except Exception:
                    continue
    return best_order


@cache_data(ttl=86400)
def run_walk_forward(ticker: str, data: pd.DataFrame) -> dict:
    default = {"arima_rmse": 1.0, "trend_rmse": 1.0, "n_windows": 0}
    try:
        if data is None or data.empty or "Close" not in data or len(data) < 120:
            return default

        close = data["Close"].dropna().astype(float).tail(252)
        if len(close) < 120:
            return default

        train_len = 60
        test_len = 30
        stride = 30

        starts = list(range(0, max(1, len(close) - (train_len + test_len) + 1), stride))[:10]

        arima_errors: list[float] = []
        trend_errors: list[float] = []

        for start in starts:
            train = close.iloc[start : start + train_len]
            test = close.iloc[start + train_len : start + train_len + test_len]
            if len(train) < train_len or len(test) < test_len:
                continue

            test_vals = test.values.astype(float)

            if ARIMA_AVAILABLE:
                try:
                    order = _select_arima_order(train)
                    pred = ARIMA(train, order=order).fit().forecast(steps=test_len)
                    arima_errors.append(_rmse(test_vals, np.array(pred.values, dtype=float)))
                except Exception:
                    pass

            if SKLEARN_AVAILABLE:
                try:
                    y = np.log(np.maximum(train.values.astype(float), 1e-9))
                    x = np.arange(len(y)).reshape(-1, 1)
                    model = LinearRegression().fit(x, y)
                    x_future = np.arange(len(y), len(y) + test_len).reshape(-1, 1)
                    pred = np.exp(model.predict(x_future))
                    trend_errors.append(_rmse(test_vals, pred))
                except Exception:
                    pass

        n_windows = max(len(arima_errors), len(trend_errors))
        if n_windows == 0:
            return default

        result = {
            "arima_rmse": round(float(np.mean(arima_errors)) if arima_errors else 1.0, 6),
            "trend_rmse": round(float(np.mean(trend_errors)) if trend_errors else 1.0, 6),
            "n_windows": int(n_windows),
        }
        logger.info("Walk-forward backtest complete for %s: %s", ticker.upper(), result)
        return result
    except Exception as exc:
        logger.warning("Walk-forward failed for %s: %s", ticker.upper(), exc)
        return default
