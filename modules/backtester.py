from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from modules.arima_hardening import ARIMA_AVAILABLE, fit_arima_with_hardening
from modules.feature_engineering import build_feature_table, normalize_daily_index
from modules.lightgbm_model import (
    LIGHTGBM_AVAILABLE,
    build_return_training_examples,
    predict_forward_return,
    train_return_models,
)
from modules.logger import get_logger

logger = get_logger(__name__)

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


# Keep synchronized with rolling/min_period windows in modules.feature_engineering._technical_features.
_LIGHTGBM_FEATURE_WARMUP_DAYS = max((10, 30, 12, 26, 14, 20))
DEFAULT_WALK_FORWARD_HORIZON = 30


@dataclass(frozen=True)
class WalkForwardWindowConfig:
    horizon: int
    train_len: int
    test_len: int
    stride: int
    max_history_rows: int
    default_period: str
    max_windows: int = 10


_WALK_FORWARD_WINDOW_CONFIGS: dict[int, WalkForwardWindowConfig] = {
    30: WalkForwardWindowConfig(
        horizon=30,
        train_len=60,
        test_len=30,
        stride=30,
        max_history_rows=252,
        default_period="2y",
    ),
    180: WalkForwardWindowConfig(
        horizon=180,
        train_len=360,
        test_len=180,
        stride=180,
        max_history_rows=1260,
        default_period="5y",
    ),
    # Polygon Starter history is capped at ~5y, and real daily histories usually land a bit
    # under 1260 rows once exchange holidays are excluded. Keep enough slack to guarantee one
    # clean non-overlapping 720d holdout window without changing the 5y fetch default.
    720: WalkForwardWindowConfig(
        horizon=720,
        train_len=450,
        test_len=720,
        stride=720,
        max_history_rows=1260,
        default_period="5y",
    ),
}


def get_walk_forward_window_config(horizon: int = DEFAULT_WALK_FORWARD_HORIZON) -> WalkForwardWindowConfig:
    config = _WALK_FORWARD_WINDOW_CONFIGS.get(int(horizon))
    if config is None:
        supported = ", ".join(str(value) for value in sorted(_WALK_FORWARD_WINDOW_CONFIGS))
        raise ValueError(f"Unsupported walk-forward horizon: {horizon}. Supported horizons: {supported}")
    return config


def _compute_walk_forward_starts(close_len: int, config: WalkForwardWindowConfig) -> list[int]:
    required_rows = int(config.train_len + config.test_len)
    if int(close_len) < required_rows:
        return []
    last_start = int(close_len) - required_rows + 1
    return list(range(0, last_start, int(config.stride)))[: int(config.max_windows)]


def _normalize_lightgbm_price_frame(price_frame: pd.DataFrame) -> pd.DataFrame:
    normalized = price_frame.copy()
    normalized.index = normalize_daily_index(normalized.index)
    normalized = normalized[~normalized.index.isna()].sort_index()
    normalized = normalized[~normalized.index.duplicated(keep="last")]
    return normalized


def _select_arima_order(series: pd.Series) -> tuple[int, int, int]:
    if not ARIMA_AVAILABLE:
        return (5, 1, 0)
    best_order = (5, 1, 0)
    best_aic = float("inf")
    for p in range(6):
        for d in range(2):
            for q in range(3):
                try:
                    fit = fit_arima_with_hardening(series, order=(p, d, q), logger=logger)
                    if fit.aic < best_aic:
                        best_aic = float(fit.aic)
                        best_order = (p, d, q)
                except Exception:
                    continue
    return best_order


@cache_data(ttl=86400)
def run_walk_forward(
    ticker: str,
    data: pd.DataFrame,
    horizon: int = DEFAULT_WALK_FORWARD_HORIZON,
    evaluate_lightgbm: bool = False,
    evaluate_naive_baseline: bool = False,
    lightgbm_diagnostics: bool = False,
) -> dict:
    default = {
        "arima_rmse": 1.0,
        "trend_rmse": 1.0,
        "lightgbm_rmse": None,
        "n_windows": 0,
        "arima_windows": 0,
        "trend_windows": 0,
        "lightgbm_windows": 0,
    }
    if evaluate_naive_baseline:
        default["naive_rmse"] = 1.0
        default["naive_windows"] = 0
    if lightgbm_diagnostics:
        default["lightgbm_diagnostics"] = {"per_window": [], "prediction_stats": None}
    try:
        config = get_walk_forward_window_config(horizon)
        minimum_required_rows = max(120, int(config.train_len + config.test_len))
        if data is None or data.empty or "Close" not in data or len(data) < minimum_required_rows:
            return default

        close = data["Close"].dropna().astype(float).tail(int(config.max_history_rows))
        if len(close) < minimum_required_rows:
            return default
        price_frame = data.copy().reindex(close.index)
        lightgbm_price_frame = _normalize_lightgbm_price_frame(price_frame)
        starts = _compute_walk_forward_starts(len(close), config)

        arima_errors: list[float] = []
        trend_errors: list[float] = []
        lightgbm_errors: list[float] = []
        naive_errors: list[float] = []
        lightgbm_diagnostic_rows: list[dict[str, float | int]] = []
        lightgbm_horizon = int(config.horizon)

        for start in starts:
            train = close.iloc[start : start + int(config.train_len)]
            test = close.iloc[start + int(config.train_len) : start + int(config.train_len) + int(config.test_len)]
            if len(train) < int(config.train_len) or len(test) < int(config.test_len):
                continue

            test_vals = test.values.astype(float)

            if ARIMA_AVAILABLE:
                try:
                    order = _select_arima_order(train)
                    pred = fit_arima_with_hardening(train, order=order, logger=logger).forecast(steps=int(config.test_len))
                    arima_errors.append(_rmse(test_vals, np.array(pred.values, dtype=float)))
                except Exception as exc:
                    logger.warning(
                        "ARIMA backtest window failed for %s at start=%d: %s: %s",
                        ticker.upper(),
                        start,
                        type(exc).__name__,
                        exc,
                    )

            if SKLEARN_AVAILABLE:
                try:
                    y = np.log(np.maximum(train.values.astype(float), 1e-9))
                    x = np.arange(len(y)).reshape(-1, 1)
                    model = LinearRegression().fit(x, y)
                    x_future = np.arange(len(y), len(y) + int(config.test_len)).reshape(-1, 1)
                    pred = np.exp(model.predict(x_future))
                    trend_errors.append(_rmse(test_vals, pred))
                except Exception:
                    pass

            if evaluate_naive_baseline:
                start_price = float(train.iloc[-1])
                naive_pred = np.full(int(config.test_len), start_price, dtype=float)
                naive_errors.append(_rmse(test_vals, naive_pred))

            if evaluate_lightgbm and LIGHTGBM_AVAILABLE:
                try:
                    train_price = lightgbm_price_frame.iloc[start : start + int(config.train_len)]
                    warmup_start = max(0, start - _LIGHTGBM_FEATURE_WARMUP_DAYS)
                    train_end = start + int(config.train_len)
                    lightgbm_history = lightgbm_price_frame.iloc[warmup_start : train_end + int(lightgbm_horizon)]
                    features = build_feature_table(
                        ticker,
                        lightgbm_history,
                        lookback_days=len(lightgbm_history),
                    )
                    datasets = build_return_training_examples(
                        ticker=ticker,
                        price_data=lightgbm_history,
                        feature_table=features,
                        lookback_days=len(lightgbm_history),
                        horizons=(lightgbm_horizon,),
                    )
                    if lightgbm_horizon not in datasets:
                        logger.warning(
                            "LightGBM backtest skipped for %s at start=%d: no labeled rows for horizon %sd",
                            ticker.upper(),
                            start,
                            lightgbm_horizon,
                        )
                        continue
                    x_train_all, y_train_all = datasets[lightgbm_horizon]
                    x_train = x_train_all.reindex(train_price.index).dropna(how="all")
                    y_train = y_train_all.reindex(x_train.index).dropna()
                    x_train = x_train.reindex(y_train.index)
                    if len(y_train) == 0:
                        logger.warning(
                            "LightGBM backtest skipped for %s at start=%d: no usable rows remain after training-window alignment",
                            ticker.upper(),
                            start,
                        )
                        continue
                    dynamic_min_rows = min(50, max(10, int(config.train_len / 3)))
                    models = train_return_models(
                        {lightgbm_horizon: (x_train, y_train)},
                        min_rows_per_horizon=dynamic_min_rows,
                    )
                    if not models or lightgbm_horizon not in models:
                        logger.warning(
                            "LightGBM backtest skipped for %s at start=%d: no trained model for horizon %sd",
                            ticker.upper(),
                            start,
                            lightgbm_horizon,
                        )
                        continue
                    anchor_idx = train_price.index[-1]
                    if anchor_idx not in features.index:
                        logger.warning(
                            "LightGBM backtest skipped for %s at start=%d: anchor date %s missing from feature table",
                            ticker.upper(),
                            start,
                            anchor_idx,
                        )
                        continue
                    latest_row = features.loc[anchor_idx]
                    if latest_row.isna().all():
                        logger.warning(
                            "LightGBM backtest skipped for %s at start=%d: latest feature row is entirely NaN",
                            ticker.upper(),
                            start,
                        )
                        continue
                    predicted_return = predict_forward_return(models[lightgbm_horizon], latest_row)
                    if predicted_return is None:
                        logger.warning(
                            "LightGBM backtest skipped for %s at start=%d: inference returned no prediction",
                            ticker.upper(),
                            start,
                        )
                        continue
                    start_price = float(train.iloc[-1])
                    if start_price <= 0:
                        logger.warning(
                            "LightGBM backtest skipped for %s at start=%d: non-positive start price %.6f",
                            ticker.upper(),
                            start,
                            start_price,
                        )
                        continue
                    realized_return = (float(test_vals[-1]) - start_price) / start_price
                    if lightgbm_diagnostics:
                        diagnostic_row = {
                            "start": int(start),
                            "predicted_return": float(predicted_return),
                            "realized_return": float(realized_return),
                        }
                        lightgbm_diagnostic_rows.append(diagnostic_row)
                        logger.info(
                            "LightGBM diagnostic for %s start=%d predicted_return=%.8f realized_return=%.8f",
                            ticker.upper(),
                            start,
                            float(predicted_return),
                            float(realized_return),
                        )
                    final_price = start_price * (1.0 + float(predicted_return))
                    if final_price <= 0:
                        logger.warning(
                            "LightGBM backtest skipped for %s at start=%d: predicted final price %.6f is non-positive",
                            ticker.upper(),
                            start,
                            final_price,
                        )
                        continue
                    growth = np.exp(np.log(final_price / start_price) / int(config.test_len))
                    pred_path = start_price * np.power(growth, np.arange(1, int(config.test_len) + 1))
                    lightgbm_errors.append(_rmse(test_vals, pred_path.astype(float)))
                except Exception as exc:
                    logger.warning(
                        "LightGBM backtest window failed for %s at start=%d horizon=%sd: %s: %s",
                        ticker.upper(),
                        start,
                        lightgbm_horizon,
                        type(exc).__name__,
                        exc,
                    )

        n_windows = max(len(arima_errors), len(trend_errors), len(lightgbm_errors), len(naive_errors))
        if n_windows == 0:
            return default

        result = {
            "arima_rmse": round(float(np.mean(arima_errors)) if arima_errors else 1.0, 6),
            "trend_rmse": round(float(np.mean(trend_errors)) if trend_errors else 1.0, 6),
            "lightgbm_rmse": round(float(np.mean(lightgbm_errors)), 6) if lightgbm_errors else None,
            "n_windows": int(n_windows),
            "arima_windows": int(len(arima_errors)),
            "trend_windows": int(len(trend_errors)),
            "lightgbm_windows": int(len(lightgbm_errors)),
        }
        if evaluate_naive_baseline:
            result["naive_rmse"] = round(float(np.mean(naive_errors)) if naive_errors else 1.0, 6)
            result["naive_windows"] = int(len(naive_errors))
        if lightgbm_diagnostics:
            prediction_stats = None
            if lightgbm_diagnostic_rows:
                predicted = np.array([float(row["predicted_return"]) for row in lightgbm_diagnostic_rows], dtype=float)
                prediction_stats = {
                    "count": int(len(predicted)),
                    "min": round(float(np.min(predicted)), 8),
                    "max": round(float(np.max(predicted)), 8),
                    "mean": round(float(np.mean(predicted)), 8),
                    "std": round(float(np.std(predicted)), 8),
                }
            result["lightgbm_diagnostics"] = {
                "per_window": lightgbm_diagnostic_rows,
                "prediction_stats": prediction_stats,
            }
        logger.info("Walk-forward backtest complete for %s: %s", ticker.upper(), result)
        return result
    except Exception as exc:
        logger.warning("Walk-forward failed for %s: %s", ticker.upper(), exc)
        return default
