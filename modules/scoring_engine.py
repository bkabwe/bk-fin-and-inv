from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from modules.arima_hardening import ARIMA_AVAILABLE, fit_arima_with_hardening
from modules.backtester import run_walk_forward
from modules.data_fetcher import get_stock_data, get_stock_info
from modules.feature_engineering import build_feature_table, normalize_daily_index
from modules.fundamental_analysis import analyze_fundamentals, classify_market_cap_tier, normalize_sector_name
from modules.lightgbm_model import LIGHTGBM_AVAILABLE, load_return_models, predict_forward_return
from modules.logger import get_logger
from modules.longterm_analysis import analyze_longterm_technical_score
from modules.macro_regime import get_macro_regime
from modules.sentiment_analysis import analyze_sentiment
from modules.technical_analysis import analyze_technical, relative_strength_vs_spy

logger = get_logger(__name__)

_SECTOR_MOMENTUM_SCORE = 3
_MARKET_CAP_RISK_SCORES = {
    "Micro Cap": -5,
    "Small Cap": -3,
}
_MARKET_CAP_CONFIDENCE_PADDING = {
    "Micro Cap": 0.10,
    "Small Cap": 0.05,
}
_REPO_ROOT = Path(__file__).resolve().parents[1]
LIGHTGBM_LIVE_MODEL_DIR = _REPO_ROOT / "data" / "lightgbm_return_models"

# Conservative live rollout based on honestly-reported walk-forward evidence:
# - 30d: 89 tickers across two disjoint random samples, 6 windows/ticker, LightGBM beat
#   ARIMA, trend, and naive on 100% of tickers. Give it a real weight, but still below
#   ARIMA/trend parity while the model's production track record remains newer.
# - 180d: 30 tickers, 4 windows/ticker, LightGBM again beat ARIMA, trend, and naive on
#   100% of tickers, but with thinner replication and higher prediction variance for some
#   tickers. Keep the live weight real but smaller than 30d.
LIGHTGBM_WEIGHT_30D = 0.15
LIGHTGBM_WEIGHT_180D = 0.08

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


STATSMODELS_AVAILABLE = ARIMA_AVAILABLE

try:  # pragma: no cover
    from sklearn.linear_model import LinearRegression

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    SKLEARN_AVAILABLE = False

try:  # pragma: no cover
    from arch import arch_model

    ARCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    ARCH_AVAILABLE = False


@cache_data(ttl=86400)
def _select_arima_order(ticker: str, series_values: tuple[float, ...]) -> tuple[int, int, int]:
    if not STATSMODELS_AVAILABLE:
        return (5, 1, 0)
    series = np.array(series_values, dtype=float)
    if len(series) < 60:
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
    logger.info("Selected ARIMA order for %s: %s (aic=%.2f)", ticker.upper(), best_order, best_aic)
    return best_order


def _recommendation(score: int) -> str:
    if score >= 80:
        return "🟢 STRONG BUY"
    if score >= 65:
        return "🔵 BUY"
    if score >= 50:
        return "🟡 TAKE SMALL POSITION"
    if score >= 35:
        return "🟠 MONITOR"
    if score >= 20:
        return "🔴 DO NOT BUY"
    return "⛔ AVOID"


def _sector_momentum_adjustment(sector: str | None, macro_regime: dict) -> tuple[str, int]:
    normalized_sector = normalize_sector_name(sector)
    if not normalized_sector:
        return "unknown", 0

    bullish = {s for s in (normalize_sector_name(value) for value in macro_regime.get("bullish_sectors", [])) if s}
    bearish = {s for s in (normalize_sector_name(value) for value in macro_regime.get("bearish_sectors", [])) if s}
    if normalized_sector in bullish:
        return "bullish", _SECTOR_MOMENTUM_SCORE
    if normalized_sector in bearish:
        return "bearish", -_SECTOR_MOMENTUM_SCORE
    return "neutral", 0


def _normalize_market_cap_tier(market_cap_tier: str | None) -> str:
    clean = " ".join(str(market_cap_tier or "").strip().lower().split())
    if clean == "micro cap":
        return "Micro Cap"
    if clean == "small cap":
        return "Small Cap"
    if clean == "mid cap":
        return "Mid Cap"
    if clean == "large cap":
        return "Large Cap"
    if clean == "mega cap":
        return "Mega Cap"
    return "unknown"


def _market_cap_score_adjustment(market_cap_tier: str | None) -> int:
    return _MARKET_CAP_RISK_SCORES.get(_normalize_market_cap_tier(market_cap_tier), 0)


def _resolved_market_cap_tier(market_cap: float | int | None) -> str:
    return _normalize_market_cap_tier(classify_market_cap_tier(market_cap))


def _apply_market_cap_confidence_padding(
    low: float,
    high: float,
    current_price: float,
    market_cap_tier: str,
) -> tuple[float, float]:
    padding_ratio = _MARKET_CAP_CONFIDENCE_PADDING.get(_normalize_market_cap_tier(market_cap_tier), 0.0)
    if padding_ratio <= 0 or current_price <= 0:
        return low, high

    padding = current_price * padding_ratio
    widened_low = max(0.0, low - padding)
    widened_high = high + padding
    return round(widened_low, 2), round(max(widened_low, widened_high), 2)


def _weighted_ensemble(components: list[tuple[str, float | None, float]]) -> tuple[float | None, str, list[float]]:
    available = [(name, float(value), weight) for name, value, weight in components if value is not None and float(weight) > 0.0]
    if not available:
        return None, "No projection models available", []
    lightgbm_weight = next((float(weight) for name, _, weight in available if name == "LightGBM"), None)
    other_total = sum(float(weight) for name, _, weight in available if name != "LightGBM")
    normalized_weights: dict[str, float] = {}
    if lightgbm_weight is not None and other_total > 0 and lightgbm_weight < 1.0:
        remaining = max(1.0 - lightgbm_weight, 0.0)
        for name, _, weight in available:
            if name == "LightGBM":
                normalized_weights[name] = lightgbm_weight
            else:
                normalized_weights[name] = remaining * (float(weight) / other_total)
    else:
        total_weight = sum(float(weight) for _, _, weight in available)
        normalized_weights = {name: (float(weight) / total_weight) for name, _, weight in available}
    projection = sum(value * normalized_weights[name] for name, value, _ in available)
    basis = "Ensemble: " + " + ".join([f"{name}({normalized_weights[name] * 100:.0f}%)" for name, _, _ in available])
    return projection, basis, [value for _, value, _ in available]


def _confidence_bounds(
    target: float,
    values: list[float],
    current_price: float,
    cap_value: float | None,
    garch_low: float | None = None,
    garch_high: float | None = None,
) -> tuple[float, float]:
    lows = [target, current_price, *values]
    highs = [target, current_price, *values]
    if garch_low is not None:
        lows.append(garch_low)
    if garch_high is not None:
        highs.append(garch_high)

    # Fallback widening when one or both GARCH confidence edges are unavailable.
    if garch_low is None or garch_high is None:
        spread_values = np.array([target, current_price, *values], dtype=float)
        spread = float(np.std(spread_values)) if len(spread_values) > 1 else 0.0
        floor = max(current_price * 0.08, abs(target - current_price))
        band = max(floor, spread * 1.96)
        if garch_low is None:
            lows.append(min(target - band, current_price))
        if garch_high is None:
            highs.append(max(target + band, current_price))

    low = max(0.0, min(lows))
    high = max(highs)
    if cap_value is not None:
        low = min(low, cap_value)
        high = min(high, cap_value)
    return round(low, 2), round(max(low, high), 2)


def _cap_target(value: float, current_price: float, cap_value: float | None) -> float:
    target = max(float(value), current_price)
    if cap_value is not None:
        target = min(target, cap_value)
    return round(target, 2)


def _inverse_rmse_weights(backtest: dict) -> dict[str, float] | None:
    try:
        rmses = {
            "arima": float(backtest.get("arima_rmse") or 0),
            "trend": float(backtest.get("trend_rmse") or 0),
        }
        if int(backtest.get("n_windows") or 0) <= 0:
            return None
        inv = {k: (1.0 / max(v, 1e-9)) for k, v in rmses.items() if v > 0}
        total = sum(inv.values())
        if total <= 0:
            return None
        return {k: inv[k] / total for k in inv}
    except Exception:
        return None


def _blend_lightgbm_weight(base_weights: dict[str, float], lightgbm_weight: float = 0.0) -> dict[str, float]:
    available = {name: float(weight) for name, weight in base_weights.items() if float(weight) > 0.0}
    if lightgbm_weight <= 0.0 or not available:
        return available
    total = sum(available.values())
    if total <= 0.0:
        return available
    reserved = min(float(lightgbm_weight), total)
    remaining_total = max(total - reserved, 0.0)
    blended = {name: remaining_total * (weight / total) for name, weight in available.items()}
    blended["lightgbm"] = reserved
    return blended


def _projection_model_weights(
    horizon_days: int,
    backtest: dict | None,
    *,
    include_lightgbm: bool,
) -> dict[str, float]:
    if int(horizon_days) == 30:
        base_weights = _inverse_rmse_weights(backtest or {}) if backtest else {"arima": 0.55, "trend": 0.45}
        scaled = {
            "arima": 0.80 * float((base_weights or {}).get("arima", 0.55)),
            "trend": 0.80 * float((base_weights or {}).get("trend", 0.45)),
        }
        return _blend_lightgbm_weight(scaled, LIGHTGBM_WEIGHT_30D if include_lightgbm else 0.0)

    if int(horizon_days) == 180:
        if backtest:
            base_weights = _inverse_rmse_weights(backtest)
            if base_weights:
                scaled = {
                    "arima": 0.55 * float(base_weights.get("arima", 0.0)),
                    "trend": 0.55 * float(base_weights.get("trend", 0.0)),
                }
            else:
                scaled = {"arima": 0.0, "trend": 0.55}
        else:
            scaled = {"arima": 0.0, "trend": 0.55}
        return _blend_lightgbm_weight(scaled, LIGHTGBM_WEIGHT_180D if include_lightgbm else 0.0)

    # Keep 720d LightGBM weight at zero for now. The current validation has only one
    # non-overlapping backtest window per ticker at this horizon, so the 20/30 vs.
    # ARIMA/trend and 22/30 vs. naive win rates are promising but not yet actionable.
    # Revisit once more historical data accrues naturally and yields additional 720d windows.
    if backtest:
        base_weights = _inverse_rmse_weights(backtest)
        if base_weights:
            return {
                "arima": 0.50 * float(base_weights.get("arima", 0.0)),
                "trend": 0.50 * float(base_weights.get("trend", 0.0)),
            }
    return {"arima": 0.0, "trend": 0.50}


def _has_lightgbm_backtest_support(backtest: dict | None) -> bool:
    if not backtest:
        return False
    return int(backtest.get("lightgbm_windows") or 0) > 0 and backtest.get("lightgbm_rmse") is not None


@cache_data(ttl=3600)
def _load_live_lightgbm_models(ticker: str) -> dict[int, object]:
    if not LIGHTGBM_AVAILABLE:
        return {}
    return load_return_models(LIGHTGBM_LIVE_MODEL_DIR / str(ticker).upper(), horizons=(30, 180))


def _live_lightgbm_price_projections(
    ticker: str,
    data: pd.DataFrame | None,
    current_price: float,
) -> tuple[dict[int, float], list[str], list[str]]:
    if not LIGHTGBM_AVAILABLE:
        return {}, [], ["LightGBM: package not installed"]
    if data is None or data.empty or current_price <= 0:
        return {}, [], ["LightGBM: insufficient price history"]

    models = _load_live_lightgbm_models(ticker)
    if not models:
        return {}, [], [f"LightGBM: no saved return models under {LIGHTGBM_LIVE_MODEL_DIR / str(ticker).upper()}"]

    try:
        feature_table = build_feature_table(ticker, data, lookback_days=len(data))
    except Exception as exc:
        return {}, [], [f"LightGBM: feature table unavailable ({exc})"]
    if feature_table is None or feature_table.empty:
        return {}, [], ["LightGBM: feature table unavailable"]

    features = feature_table.copy()
    features.index = normalize_daily_index(features.index)
    features = features[~features.index.isna()].sort_index()
    features = features[~features.index.duplicated(keep="last")]
    if features.empty:
        return {}, [], ["LightGBM: feature table unavailable"]

    latest_feature_rows = features.dropna(how="all")
    if latest_feature_rows.empty:
        return {}, [], ["LightGBM: latest feature row unavailable"]
    latest_feature_row = latest_feature_rows.iloc[-1]

    projections: dict[int, float] = {}
    used_models: list[str] = []
    skipped_models: list[str] = []
    for horizon in (30, 180):
        model = models.get(horizon)
        if model is None:
            skipped_models.append(f"LightGBM {horizon}d: saved model unavailable")
            continue
        try:
            predicted_return = predict_forward_return(model, latest_feature_row)
        except Exception as exc:
            skipped_models.append(f"LightGBM {horizon}d: inference failed ({exc})")
            continue
        if predicted_return is None:
            skipped_models.append(f"LightGBM {horizon}d: inference returned no prediction")
            continue
        projected_price = current_price * (1.0 + float(predicted_return))
        if projected_price <= 0:
            skipped_models.append(f"LightGBM {horizon}d: projected price was non-positive")
            continue
        projections[horizon] = float(projected_price)
        used_models.append(f"LightGBM Return {horizon}d")
    return projections, used_models, skipped_models


def _garch_confidence_from_returns(current_price: float, log_returns: np.ndarray, horizon_days: int) -> tuple[float | None, float | None]:
    if not ARCH_AVAILABLE or len(log_returns) < 60 or current_price <= 0:
        return None, None
    try:
        model = arch_model(log_returns * 100, vol="Garch", p=1, q=1, rescale=False)
        fit = model.fit(disp="off")
        fcast = fit.forecast(horizon=max(horizon_days, 1), reindex=False)
        var = float(fcast.variance.values[-1, min(horizon_days - 1, fcast.variance.shape[1] - 1)])
        sigma = np.sqrt(max(var, 1e-9)) / 100.0
        z = 1.96
        low = current_price * np.exp(-z * sigma)
        high = current_price * np.exp(z * sigma)
        return float(low), float(high)
    except Exception as exc:
        logger.warning("GARCH confidence fallback used: %s", exc)
        return None, None


def _get_price_projections_core(
    ticker: str,
    *,
    info: dict | None = None,
    data: pd.DataFrame | None = None,
    avg_cost: float | None = None,
) -> dict:
    info = info or get_stock_info(ticker)
    data = data if data is not None else get_stock_data(ticker, period="2y", interval="1d")
    technical = analyze_technical(data)
    macro = get_macro_regime()
    risk_free_rate = float(macro.get("risk_free_rate") or 0.045)

    close = data["Close"].dropna() if data is not None and not data.empty and "Close" in data else None
    current_price = float(close.iloc[-1]) if close is not None and not close.empty else float(info.get("currentPrice") or 0)
    if current_price <= 0:
        logger.warning("Price projections skipped for %s due to missing current price data", ticker)
        return {
            "current_price": 0.0,
            "short_term_target": 0.0,
            "short_term_low": 0.0,
            "short_term_high": 0.0,
            "short_term_upside": 0.0,
            "short_term_basis": "No price data available",
            "short_term_horizon": "1–4 weeks",
            "medium_term_target": 0.0,
            "medium_term_low": 0.0,
            "medium_term_high": 0.0,
            "medium_term_upside": 0.0,
            "medium_term_basis": "No price data available",
            "medium_term_horizon": "1–6 months",
            "long_term_target": 0.0,
            "long_term_low": 0.0,
            "long_term_high": 0.0,
            "long_term_upside": 0.0,
            "long_term_basis": "No price data available",
            "long_term_horizon": "6–24 months",
            "models_used": [],
            "models_skipped": ["No historical price data available"],
            "outlook": "Neutral",
            "recommendation_to_sell_at": "Insufficient data to build projections.",
            "data_quality": "Limited",
        }

    models_used: list[str] = []
    models_skipped: list[str] = []
    exchange = str(info.get("exchange") or info.get("fullExchangeName") or "").upper()
    is_speculative_otc = "OTC" in exchange
    market_cap_tier = _resolved_market_cap_tier(info.get("marketCap"))
    cap_value = None if is_speculative_otc else (3 * current_price)

    lightgbm_projections, lightgbm_models_used, lightgbm_models_skipped = _live_lightgbm_price_projections(
        ticker,
        data,
        current_price,
    )
    models_used.extend(lightgbm_models_used)
    models_skipped.extend(lightgbm_models_skipped)

    # adaptive model weights from walk-forward
    backtest = None
    model_weights = None
    try:
        backtest = run_walk_forward(ticker, data)
        model_weights = _inverse_rmse_weights(backtest)
        if model_weights:
            models_used.append("Adaptive Weights")
        else:
            models_skipped.append("Adaptive Weights: backtest unavailable")
    except Exception as exc:
        models_skipped.append(f"Adaptive Weights: {exc}")

    short_backtest = backtest if model_weights else None
    short_model_weights = _projection_model_weights(
        30,
        short_backtest,
        include_lightgbm=30 in lightgbm_projections and _has_lightgbm_backtest_support(short_backtest),
    )
    medium_model_weights = _projection_model_weights(
        180,
        backtest if model_weights else None,
        include_lightgbm=180 in lightgbm_projections,
    )
    long_model_weights = _projection_model_weights(720, backtest if model_weights else None, include_lightgbm=False)

    arima_30 = arima_180 = arima_720 = None
    if STATSMODELS_AVAILABLE:
        try:
            if close is None or len(close) < 60:
                raise ValueError("not enough history")
            series = close.tail(252).astype(float)
            order = _select_arima_order(ticker.upper(), tuple(np.round(series.values, 6).tolist()))
            arima_forecast = fit_arima_with_hardening(series, order=order, logger=logger).forecast(steps=720)
            arima_30 = float(arima_forecast.iloc[29]) if len(arima_forecast) >= 30 else float(arima_forecast.iloc[-1])
            arima_180 = float(arima_forecast.iloc[179]) if len(arima_forecast) >= 180 else float(arima_forecast.iloc[-1])
            arima_720 = float(arima_forecast.iloc[719]) if len(arima_forecast) >= 720 else float(arima_forecast.iloc[-1])
            models_used.append("ARIMA")
        except Exception as exc:
            models_skipped.append(f"ARIMA: {exc}")
    else:
        models_skipped.append("ARIMA: package not installed")

    trend_30 = trend_180 = trend_720 = None
    if SKLEARN_AVAILABLE:
        try:
            if close is None or len(close) < 30:
                raise ValueError("not enough history")
            trend_series = close.tail(200).astype(float).reset_index(drop=True)
            y_log = np.log(np.maximum(trend_series.values, 1e-9))
            x = np.arange(len(trend_series)).reshape(-1, 1)
            model = LinearRegression()
            model.fit(x, y_log)
            trend_30 = float(np.exp(model.predict(np.array([[len(trend_series) + 29]]))[0]))
            trend_180 = float(np.exp(model.predict(np.array([[len(trend_series) + 179]]))[0]))
            trend_720 = float(np.exp(model.predict(np.array([[len(trend_series) + 719]]))[0]))
            models_used.append("Log Linear Trend")
        except Exception as exc:
            models_skipped.append(f"Linear Regression: {exc}")
    else:
        models_skipped.append("Linear Regression: package not installed")

    poly_30 = poly_180 = poly_720 = None
    try:
        if close is None or len(close) < 30:
            raise ValueError("not enough history")
        poly_series = close.tail(200).astype(float).reset_index(drop=True)
        x = np.arange(len(poly_series))
        y_log = np.log(np.maximum(poly_series.values, 1e-9))
        coeffs = np.polyfit(x, y_log, deg=2 if len(poly_series) >= 60 else 1)
        poly = np.poly1d(coeffs)
        poly_30 = float(np.exp(poly(len(poly_series) + 29)))
        poly_180 = float(np.exp(poly(len(poly_series) + 179)))
        poly_720 = float(np.exp(poly(len(poly_series) + 719)))
        models_used.append("Log Polynomial Trend")
    except Exception as exc:
        models_skipped.append(f"NumPy Polynomial Trend: {exc}")

    if trend_30 is None and poly_30 is not None:
        trend_30, trend_180, trend_720 = poly_30, poly_180, poly_720
    elif trend_30 is not None and poly_30 is not None:
        trend_30 = (trend_30 + poly_30) / 2
        trend_180 = (trend_180 + poly_180) / 2 if trend_180 is not None and poly_180 is not None else trend_180
        trend_720 = (trend_720 + poly_720) / 2 if trend_720 is not None and poly_720 is not None else trend_720

    fundamentals = analyze_fundamentals(info, current_price=current_price, risk_free_rate=risk_free_rate)
    metrics = fundamentals.get("metrics", {})

    fair_value = None
    forward_eps = info.get("forwardEps")
    trailing_eps = info.get("trailingEps")
    forward_pe = info.get("forwardPE")
    trailing_pe = info.get("trailingPE")
    if forward_eps and forward_eps > 0 and forward_pe and forward_pe > 0:
        fair_value = float(forward_eps) * float(forward_pe)
        models_used.append("Fundamental Fair Value")
    elif trailing_eps and trailing_eps > 0 and trailing_pe and trailing_pe > 0:
        fair_value = float(trailing_eps) * float(trailing_pe)
        models_used.append("Fundamental Fair Value")
    elif (forward_eps and forward_eps > 0) or (trailing_eps and trailing_eps > 0):
        fair_value = float(forward_eps or trailing_eps) * 20.0
        models_used.append("Fundamental Fair Value")
    else:
        models_skipped.append("Fundamental Fair Value: no EPS data")

    dcf_estimate = metrics.get("dcf_estimate")
    if dcf_estimate:
        models_used.append("DCF Estimate")
    else:
        models_skipped.append("DCF Estimate: unavailable")

    resistance_levels = sorted([float(x) for x in technical.get("resistance_levels", []) if x is not None])
    nearest_resistance = next((value for value in resistance_levels if value >= current_price), None)
    bb_upper = technical.get("indicators", {}).get("bb_high")
    technical_resistance = max([x for x in [nearest_resistance, bb_upper, current_price] if x is not None])
    models_used.append("Technical Resistance")

    analyst_target = info.get("targetMeanPrice")
    analyst_medium = analyst_long = None
    if analyst_target and analyst_target > 0:
        analyst_medium = float(analyst_target) * 0.65
        analyst_long = float(analyst_target)
        models_used.append("Analyst Target")
    else:
        models_skipped.append("Analyst Target: unavailable")

    short_projection, short_basis, short_values = _weighted_ensemble(
        [
            ("ARIMA", arima_30, short_model_weights.get("arima", 0.0)),
            ("Trend", trend_30, short_model_weights.get("trend", 0.0)),
            ("LightGBM", lightgbm_projections.get(30), short_model_weights.get("lightgbm", 0.0)),
            ("Resistance", technical_resistance, 0.20),
        ]
    )
    medium_projection, medium_basis, medium_values = _weighted_ensemble(
        [
            ("ARIMA", arima_180, medium_model_weights.get("arima", 0.0)),
            ("Trend", trend_180, medium_model_weights.get("trend", 0.0)),
            ("LightGBM", lightgbm_projections.get(180), medium_model_weights.get("lightgbm", 0.0)),
            ("Fundamental", fair_value, 0.20),
            ("DCF", dcf_estimate, 0.10),
            ("Analyst x0.65", analyst_medium, 0.15),
        ]
    )
    long_projection, long_basis, long_values = _weighted_ensemble(
        [
            ("ARIMA", arima_720, long_model_weights.get("arima", 0.0)),
            ("Trend", trend_720, long_model_weights.get("trend", 0.0)),
            ("Fundamental", fair_value, 0.20),
            ("DCF", dcf_estimate, 0.10),
            ("Analyst", analyst_long, 0.20),
        ]
    )

    short_projection = _cap_target(short_projection or current_price, current_price, cap_value)
    medium_projection = _cap_target(medium_projection or short_projection, current_price, cap_value)
    long_projection = _cap_target(long_projection or medium_projection, current_price, cap_value)

    directional_models = [
        x
        for x in [
            arima_30,
            trend_30,
            arima_180,
            trend_180,
            arima_720,
            trend_720,
            lightgbm_projections.get(30),
            lightgbm_projections.get(180),
            fair_value,
            dcf_estimate,
        ]
        if x is not None
    ]
    bearish = bool(directional_models) and all(x < current_price for x in directional_models)
    outlook = "Bearish" if bearish else "Bullish" if long_projection > current_price * 1.10 else "Neutral"

    garch_low_30 = garch_high_30 = garch_low_180 = garch_high_180 = garch_low_720 = garch_high_720 = None
    if close is not None and len(close) > 80:
        log_returns = np.log(close / close.shift(1)).dropna().values.astype(float)
        garch_low_30, garch_high_30 = _garch_confidence_from_returns(current_price, log_returns, 30)
        garch_low_180, garch_high_180 = _garch_confidence_from_returns(current_price, log_returns, 180)
        garch_low_720, garch_high_720 = _garch_confidence_from_returns(current_price, log_returns, 720)
        if garch_low_30 is not None:
            models_used.append("GARCH(1,1) Confidence")

    short_low, short_high = _confidence_bounds(
        short_projection,
        short_values,
        current_price,
        cap_value,
        garch_low=garch_low_30,
        garch_high=garch_high_30,
    )
    short_low, short_high = _apply_market_cap_confidence_padding(short_low, short_high, current_price, market_cap_tier)
    medium_low, medium_high = _confidence_bounds(
        medium_projection,
        medium_values,
        current_price,
        cap_value,
        garch_low=garch_low_180,
        garch_high=garch_high_180,
    )
    medium_low, medium_high = _apply_market_cap_confidence_padding(medium_low, medium_high, current_price, market_cap_tier)
    long_low, long_high = _confidence_bounds(
        long_projection,
        long_values,
        current_price,
        cap_value,
        garch_low=garch_low_720,
        garch_high=garch_high_720,
    )
    long_low, long_high = _apply_market_cap_confidence_padding(long_low, long_high, current_price, market_cap_tier)

    near_resistance = current_price >= (technical_resistance * 0.97) if technical_resistance else False
    if is_speculative_otc and "Fundamental Fair Value: no EPS data" in models_skipped:
        recommendation_to_sell_at = (
            "Limited data available for OTC stock. Projection based on technical trend only. "
            "High risk — size position accordingly."
        )
    elif bearish:
        recommendation_to_sell_at = (
            f"Models indicate downside risk. Consider reducing position. Stop-loss at ${current_price * 0.92:.2f}."
        )
    elif avg_cost and avg_cost > current_price and short_projection < avg_cost:
        recommendation_to_sell_at = (
            f"Currently at a loss. Hold for recovery to break-even at ${avg_cost:.2f}. "
            f"Short-term technical target: ${short_projection:.2f}."
        )
    elif avg_cost and avg_cost < current_price and near_resistance:
        recommendation_to_sell_at = (
            f"Consider taking 50% profit near ${short_projection:.2f} (short-term target). "
            f"Hold remainder for medium-term target of ${medium_projection:.2f}."
        )
    else:
        recommendation_to_sell_at = (
            f"Take partial profits near ${short_projection:.2f}, then review momentum toward ${medium_projection:.2f}."
        )

    if is_speculative_otc and "Fundamental Fair Value" not in models_used:
        data_quality = "Technical Only"
    elif {"ARIMA", "Fundamental Fair Value"}.issubset(set(models_used)) and (
        "Log Linear Trend" in models_used or "Log Polynomial Trend" in models_used
    ):
        data_quality = "Full"
    else:
        data_quality = "Limited"

    logger.info(
        "Price projections for %s | used=%s | skipped=%s",
        ticker.upper(),
        sorted(set(models_used)),
        models_skipped,
    )

    return {
        "current_price": round(current_price, 2),
        "short_term_target": short_projection,
        "short_term_low": short_low,
        "short_term_high": short_high,
        "short_term_upside": round(((short_projection - current_price) / current_price) * 100, 2),
        "short_term_basis": short_basis,
        "short_term_horizon": "1–4 weeks",
        "medium_term_target": medium_projection,
        "medium_term_low": medium_low,
        "medium_term_high": medium_high,
        "medium_term_upside": round(((medium_projection - current_price) / current_price) * 100, 2),
        "medium_term_basis": medium_basis,
        "medium_term_horizon": "1–6 months",
        "long_term_target": long_projection,
        "long_term_low": long_low,
        "long_term_high": long_high,
        "long_term_upside": round(((long_projection - current_price) / current_price) * 100, 2),
        "long_term_basis": long_basis,
        "long_term_horizon": "6–24 months",
        "models_used": sorted(set(models_used)),
        "models_skipped": models_skipped,
        "outlook": outlook,
        "recommendation_to_sell_at": recommendation_to_sell_at,
        "data_quality": data_quality,
    }


@cache_data(ttl=3600)
def _get_cached_price_projections(ticker: str, avg_cost: float | None = None) -> dict:
    return _get_price_projections_core(ticker, avg_cost=avg_cost)


def get_price_projections(
    ticker: str,
    avg_cost: float | None = None,
    *,
    data_override: pd.DataFrame | None = None,
    info_override: dict | None = None,
) -> dict:
    if data_override is None and info_override is None:
        return _get_cached_price_projections(ticker, avg_cost=avg_cost)
    return _get_price_projections_core(ticker, info=info_override, data=data_override, avg_cost=avg_cost)


def analyze_stock(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    avg_cost: float | None = None,
    *,
    data_override: pd.DataFrame | None = None,
    info_override: dict | None = None,
    projection_data_override: pd.DataFrame | None = None,
) -> dict:
    data = data_override if data_override is not None else get_stock_data(ticker, period=period, interval=interval)
    info = info_override if info_override is not None else get_stock_info(ticker)
    projection_data = projection_data_override if projection_data_override is not None else data_override

    info = dict(info or {})
    trailing_52w = pd.DataFrame()
    if not data.empty:
        try:
            max_date = pd.to_datetime(data.index).max()
            cutoff = max_date - pd.Timedelta(days=365)
            trailing_52w = data.loc[pd.to_datetime(data.index) >= cutoff]
        except Exception:
            trailing_52w = data.tail(252)
    if "fiftyTwoWeekHigh" not in info and not trailing_52w.empty and "High" in trailing_52w:
        try:
            info["fiftyTwoWeekHigh"] = float(trailing_52w["High"].astype(float).max())
        except Exception:
            pass
    if "fiftyTwoWeekLow" not in info and not trailing_52w.empty and "Low" in trailing_52w:
        try:
            info["fiftyTwoWeekLow"] = float(trailing_52w["Low"].astype(float).min())
        except Exception:
            pass

    technical = analyze_technical(data)
    if not data.empty:
        current_price = float(data["Close"].iloc[-1])
    else:
        _raw_price = info.get("currentPrice")
        try:
            current_price = float(_raw_price) if _raw_price is not None else None
        except Exception:
            current_price = None

    macro_regime = get_macro_regime()
    risk_free_rate = float(macro_regime.get("risk_free_rate") or 0.045)

    fundamentals = analyze_fundamentals(info, current_price=current_price, risk_free_rate=risk_free_rate)
    sentiment = analyze_sentiment(ticker)

    spy_data = get_stock_data("SPY", period=period, interval=interval)
    relative_strength = relative_strength_vs_spy(data, spy_data)

    trend_score = 15 if technical["trend"] == "uptrend" else 8 if technical["trend"] == "sideways" else 2
    rsi, macd, signal = (
        technical["indicators"].get("rsi"),
        technical["indicators"].get("macd"),
        technical["indicators"].get("macd_signal"),
    )
    momentum_score = (8 if rsi is not None and 40 <= rsi <= 65 else 4 if rsi is not None and 30 <= rsi <= 75 else 1) + (
        7 if macd is not None and signal is not None and macd > signal else 2
    )
    volume_score = 10 if technical["volume_confirmation"] else 5 if technical["volume_trend"] == "increasing" else 2
    bullish = sum(1 for p in technical["patterns"] if p["implication"] == "bullish")
    bearish = sum(1 for p in technical["patterns"] if p["implication"] == "bearish")
    pattern_score = max(0, min(10, 5 + (bullish - bearish) * 2))

    trendline_score = 0
    trendline_break = technical.get("signals", {}).get("trendline_break_signal")
    if trendline_break == "bullish_break":
        trendline_score = 5
    elif trendline_break == "bearish_break":
        trendline_score = -5

    rs_score = 0
    rs20 = relative_strength.get("rs_20d")
    if rs20 is not None and rs20 > 1.1:
        rs_score = 5
    elif rs20 is not None and rs20 < 0.9:
        rs_score = -3

    breakout_score = 0
    breakout = technical.get("breakout", {})
    if breakout.get("signal_strength") == "strong":
        breakout_score = 7

    volume_quality_score = 0
    vq = technical.get("volume_quality", {})
    if vq.get("breakout_volume_confirmed"):
        volume_quality_score += 3
    if vq.get("climactic_volume") and vq.get("volume_divergence"):
        volume_quality_score -= 3

    macro_score = 0
    if macro_regime.get("market_regime") == "risk_off":
        macro_score = -5
    elif macro_regime.get("market_regime") == "risk_on":
        macro_score = 3

    sector_trend, sector_momentum_score = _sector_momentum_adjustment(info.get("sector"), macro_regime)
    market_cap_tier = _resolved_market_cap_tier(info.get("marketCap"))
    market_cap_score = _market_cap_score_adjustment(market_cap_tier)

    technical_total = max(
        0,
        min(
            50,
            trend_score + momentum_score + volume_score + pattern_score + trendline_score + rs_score + breakout_score + volume_quality_score,
        ),
    )
    fundamental_total = round((fundamentals["fundamental_score"] / 100) * 30)
    sentiment_total = round(((sentiment["sentiment_score"] + 1) / 2) * 20)
    base_total = technical_total + fundamental_total + sentiment_total + macro_score + sector_momentum_score + market_cap_score

    horizon = "Medium-Term Setup"
    if macd is not None and signal is not None and macd > signal and technical["trend"] != "uptrend":
        horizon = "Short-Term Opportunity"
    if technical["trend"] == "uptrend" and fundamentals["fundamental_score"] >= 65:
        horizon = "Long-Term Hold"

    longterm = {
        "longterm_technical_score": 0,
        "longterm_stage": None,
        "primary_trend": None,
        "volume_trend_confirmation": None,
        "max_drawdown_pct": None,
        "recovery_days": None,
        "golden_cross_count": 0,
        "death_cross_count": 0,
        "favorable_cross_follow_through_pct": None,
    }
    if horizon == "Long-Term Hold":
        longterm_history = data if data is not None and not data.empty else projection_data
        current_len = len(longterm_history) if longterm_history is not None else 0
        if projection_data is not None and not projection_data.empty and len(projection_data) > current_len:
            longterm_history = projection_data
        if longterm_history is None or longterm_history.empty or len(longterm_history) < 900:
            longterm_history = get_stock_data(ticker, period="5y", interval="1d")
        longterm = analyze_longterm_technical_score(ticker, sector=info.get("sector"), data=longterm_history)

    total = max(0, min(100, int(base_total + int(longterm.get("longterm_technical_score") or 0))))

    entry = technical.get("entry_price")
    projections = get_price_projections(
        ticker,
        avg_cost=avg_cost,
        data_override=projection_data,
        info_override=info,
    )
    target = projections.get("short_term_target") or technical.get("target_price")
    atr = technical.get("atr") or 0
    stop_loss = round((entry or current_price or 0) - (1.5 * atr), 2) if (entry or current_price) else None

    sell = projections.get("recommendation_to_sell_at")
    near_resistance = bool(target and current_price and current_price >= target * 0.97)
    high52 = fundamentals["metrics"].get("52w_high")
    if total < 35 and near_resistance and not sell:
        sell = "SELL"
    elif rsi and rsi > 75 and high52 and current_price and current_price >= high52 * 0.98 and not sell:
        sell = "CONSIDER SELLING"
    elif (
        macd
        and signal
        and macd < signal
        and technical["indicators"].get("sma50")
        and current_price < technical["indicators"]["sma50"]
        and not sell
    ):
        sell = "REDUCE POSITION"

    return {
        "ticker": ticker.upper(),
        "company": info.get("shortName") or info.get("longName") or ticker.upper(),
        "current_price": current_price,
        "score": total,
        "recommendation": _recommendation(total),
        "sector_trend": sector_trend,
        "market_cap_tier": market_cap_tier,
        "time_horizon": horizon,
        "longterm_stage": longterm.get("longterm_stage"),
        "primary_trend": longterm.get("primary_trend"),
        "volume_trend_confirmation": longterm.get("volume_trend_confirmation"),
        "max_drawdown_pct": longterm.get("max_drawdown_pct"),
        "recovery_days": longterm.get("recovery_days"),
        "golden_cross_count": longterm.get("golden_cross_count"),
        "death_cross_count": longterm.get("death_cross_count"),
        "favorable_cross_follow_through_pct": longterm.get("favorable_cross_follow_through_pct"),
        "entry_price": entry,
        "target_price": target,
        "stop_loss": stop_loss,
        "projections": projections,
        "technical": technical,
        "fundamentals": fundamentals,
        "sentiment": sentiment,
        "macro_regime": macro_regime,
        "relative_strength": relative_strength,
        "score_breakdown": {
            "technical": technical_total,
            "fundamental": fundamental_total,
            "sentiment": sentiment_total,
            "macro": macro_score,
            "sector_momentum": sector_momentum_score,
            "market_cap": market_cap_score,
            **({"longterm_technical": int(longterm.get("longterm_technical_score") or 0)} if horizon == "Long-Term Hold" else {}),
            "trend": trend_score,
            "momentum": momentum_score,
            "volume": volume_score,
            "patterns": pattern_score,
            "trendline": trendline_score,
            "relative_strength": rs_score,
            "breakout": breakout_score,
            "volume_quality": volume_quality_score,
        },
        "sell_recommendation": sell,
    }


# ---------------------------------------------------------------------------
# Fast-screen helper
# ---------------------------------------------------------------------------
# Maximum possible contribution from non-technical components:
#   fundamental (0-30) + sentiment (0-20) + macro (-5 to +3) + sector (±3)
#   + market-cap risk (0 to -5, penalty-only) => up to 56
_MAX_NON_TECHNICAL_SCORE = 56


def fast_screen_score(ticker: str, period: str = "1y", interval: str = "1d") -> tuple[int, str | None]:
    """Compute only the technical subscore for a ticker (fast, cheap).

    Returns ``(technical_score_out_of_100, error_reason)`` where ``error_reason``
    is ``None`` on success.  The returned score is the technical subscore
    normalised to a 0-100 scale (``technical_total / 50 * 100``) so it can be
    compared directly against ``min_score``.

    This is a genuine partial computation of the scoring logic in
    ``analyze_stock`` — it uses the same ``analyze_technical`` call and the
    same formula for ``technical_total``.  It intentionally omits the
    expensive steps (ARIMA/GARCH forecasting and walk-forward
    backtesting) to serve as a cheap first-pass filter.
    """
    try:
        data = get_stock_data(ticker, period=period, interval=interval)
        if data is None or data.empty:
            return 0, "no data"
        technical = analyze_technical(data)

        trend_score = 15 if technical["trend"] == "uptrend" else 8 if technical["trend"] == "sideways" else 2
        rsi = technical["indicators"].get("rsi")
        macd = technical["indicators"].get("macd")
        signal = technical["indicators"].get("macd_signal")
        momentum_score = (8 if rsi is not None and 40 <= rsi <= 65 else 4 if rsi is not None and 30 <= rsi <= 75 else 1) + (
            7 if macd is not None and signal is not None and macd > signal else 2
        )
        volume_score = 10 if technical["volume_confirmation"] else 5 if technical["volume_trend"] == "increasing" else 2
        bullish = sum(1 for p in technical["patterns"] if p["implication"] == "bullish")
        bearish = sum(1 for p in technical["patterns"] if p["implication"] == "bearish")
        pattern_score = max(0, min(10, 5 + (bullish - bearish) * 2))

        trendline_break = technical.get("signals", {}).get("trendline_break_signal")
        trendline_score = 5 if trendline_break == "bullish_break" else -5 if trendline_break == "bearish_break" else 0

        breakout_score = 7 if technical.get("breakout", {}).get("signal_strength") == "strong" else 0

        vq = technical.get("volume_quality", {})
        volume_quality_score = 0
        if vq.get("breakout_volume_confirmed"):
            volume_quality_score += 3
        if vq.get("climactic_volume") and vq.get("volume_divergence"):
            volume_quality_score -= 3

        technical_total = max(0, min(50, trend_score + momentum_score + volume_score + pattern_score + trendline_score + breakout_score + volume_quality_score))
        # Normalize to 0-100 so callers can compare directly against min_score.
        normalized = int(round(technical_total / 50 * 100))
        return normalized, None
    except Exception as exc:
        return 0, str(exc)
