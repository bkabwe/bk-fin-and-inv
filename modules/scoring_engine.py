from __future__ import annotations

from functools import lru_cache

import numpy as np

from modules.data_fetcher import get_stock_data, get_stock_info
from modules.fundamental_analysis import analyze_fundamentals
from modules.logger import get_logger
from modules.sentiment_analysis import analyze_sentiment
from modules.technical_analysis import analyze_technical

logger = get_logger(__name__)

try:
    import streamlit as st

    cache_data = st.cache_data
except Exception:  # pragma: no cover
    def cache_data(ttl: int | None = None):
        def decorator(func):
            return lru_cache(maxsize=128)(func)

        return decorator


try:  # pragma: no cover
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:  # pragma: no cover
    PROPHET_AVAILABLE = False

try:  # pragma: no cover
    from statsmodels.tsa.arima.model import ARIMA

    STATSMODELS_AVAILABLE = True
except ImportError:  # pragma: no cover
    STATSMODELS_AVAILABLE = False

try:  # pragma: no cover
    from sklearn.linear_model import LinearRegression

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    SKLEARN_AVAILABLE = False


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


def _weighted_ensemble(components: list[tuple[str, float | None, float]]) -> tuple[float | None, str, list[float]]:
    available = [(name, float(value), weight) for name, value, weight in components if value is not None]
    if not available:
        return None, "No projection models available", []
    total_weight = sum(weight for _, _, weight in available)
    projection = sum(value * (weight / total_weight) for _, value, weight in available)
    basis = "Ensemble: " + " + ".join([f"{name}({weight / total_weight * 100:.0f}%)" for name, _, weight in available])
    return projection, basis, [value for _, value, _ in available]


def _confidence_bounds(
    target: float,
    values: list[float],
    prophet_low: float | None,
    prophet_high: float | None,
    current_price: float,
    cap_value: float | None,
) -> tuple[float, float]:
    lows = [target, current_price, *values]
    highs = [target, current_price, *values]
    if prophet_low is not None:
        lows.append(prophet_low)
    if prophet_high is not None:
        highs.append(prophet_high)
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


@cache_data(ttl=3600)
def get_price_projections(ticker: str, avg_cost: float | None = None) -> dict:
    """
    Multi-model price projection using:
    1. Prophet (Facebook) time series forecasting
    2. ARIMA via statsmodels
    3. Linear regression trend via scikit-learn
    4. Fundamental fair value (P/E based)
    5. Technical resistance levels

    Returns ensemble projections for short (1-4 weeks), medium (1-6 months),
    and long term (6-24 months).
    """
    info = get_stock_info(ticker)
    data = get_stock_data(ticker, period="2y", interval="1d")
    technical = analyze_technical(data)
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
    cap_value = None if is_speculative_otc else (3 * current_price)

    prophet_30 = prophet_180 = prophet_720 = None
    prophet_low_30 = prophet_low_180 = prophet_low_720 = None
    prophet_high_30 = prophet_high_180 = prophet_high_720 = None
    if PROPHET_AVAILABLE:
        try:
            if close is None or len(close) < 60:
                raise ValueError("not enough history")
            prophet_df = close.reset_index()
            date_col = prophet_df.columns[0]
            prophet_df = prophet_df.rename(columns={date_col: "ds", "Close": "y"})[["ds", "y"]]
            prophet_df["ds"] = prophet_df["ds"].dt.tz_localize(None)
            model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
            model.fit(prophet_df)
            forecast = model.predict(model.make_future_dataframe(periods=720, freq="D"))
            base_idx = len(prophet_df) - 1
            p30 = forecast.iloc[min(base_idx + 30, len(forecast) - 1)]
            p180 = forecast.iloc[min(base_idx + 180, len(forecast) - 1)]
            p720 = forecast.iloc[min(base_idx + 720, len(forecast) - 1)]
            prophet_30, prophet_180, prophet_720 = float(p30["yhat"]), float(p180["yhat"]), float(p720["yhat"])
            prophet_low_30, prophet_low_180, prophet_low_720 = float(p30["yhat_lower"]), float(p180["yhat_lower"]), float(
                p720["yhat_lower"]
            )
            prophet_high_30, prophet_high_180, prophet_high_720 = float(p30["yhat_upper"]), float(p180["yhat_upper"]), float(
                p720["yhat_upper"]
            )
            models_used.append("Prophet")
        except Exception as exc:
            models_skipped.append(f"Prophet: {exc}")
    else:
        models_skipped.append("Prophet: package not installed")

    arima_30 = arima_180 = None
    if STATSMODELS_AVAILABLE:
        try:
            if close is None or len(close) < 60:
                raise ValueError("not enough history")
            series = close.tail(252).astype(float)
            arima_forecast = ARIMA(series, order=(5, 1, 0)).fit().forecast(steps=180)
            arima_30 = float(arima_forecast.iloc[29]) if len(arima_forecast) >= 30 else float(arima_forecast.iloc[-1])
            arima_180 = float(arima_forecast.iloc[179]) if len(arima_forecast) >= 180 else float(arima_forecast.iloc[-1])
            models_used.append("ARIMA")
        except Exception as exc:
            models_skipped.append(f"ARIMA: {exc}")
    else:
        models_skipped.append("ARIMA: package not installed")

    trend_30 = trend_180 = trend_720 = None
    poly_30 = poly_180 = poly_720 = None
    if SKLEARN_AVAILABLE:
        try:
            if close is None or len(close) < 30:
                raise ValueError("not enough history")
            trend_series = close.tail(200).astype(float).reset_index(drop=True)
            x = np.arange(len(trend_series)).reshape(-1, 1)
            model = LinearRegression()
            model.fit(x, trend_series.values)
            trend_30 = float(model.predict(np.array([[len(trend_series) + 29]]))[0])
            trend_180 = float(model.predict(np.array([[len(trend_series) + 179]]))[0])
            trend_720 = float(model.predict(np.array([[len(trend_series) + 719]]))[0])
            models_used.append("Linear Regression Trend")
        except Exception as exc:
            models_skipped.append(f"Linear Regression: {exc}")
    else:
        models_skipped.append("Linear Regression: package not installed")

    try:
        if close is None or len(close) < 30:
            raise ValueError("not enough history")
        poly_series = close.tail(200).astype(float).reset_index(drop=True)
        x = np.arange(len(poly_series))
        coeffs = np.polyfit(x, poly_series.values, deg=2 if len(poly_series) >= 60 else 1)
        poly = np.poly1d(coeffs)
        poly_30 = float(poly(len(poly_series) + 29))
        poly_180 = float(poly(len(poly_series) + 179))
        poly_720 = float(poly(len(poly_series) + 719))
        models_used.append("NumPy Polynomial Trend")
    except Exception as exc:
        models_skipped.append(f"NumPy Polynomial Trend: {exc}")

    if trend_30 is None and poly_30 is not None:
        trend_30, trend_180, trend_720 = poly_30, poly_180, poly_720
    elif trend_30 is not None and poly_30 is not None:
        trend_30 = (trend_30 + poly_30) / 2
        trend_180 = (trend_180 + poly_180) / 2 if trend_180 is not None and poly_180 is not None else trend_180
        trend_720 = (trend_720 + poly_720) / 2 if trend_720 is not None and poly_720 is not None else trend_720

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

    resistance_levels = sorted([float(x) for x in technical.get("resistance_levels", []) if x is not None])
    nearest_resistance = next((value for value in resistance_levels if value >= current_price), None)
    bb_upper = technical.get("indicators", {}).get("bb_high")
    technical_resistance = max([x for x in [nearest_resistance, bb_upper, current_price] if x is not None])
    models_used.append("Technical Resistance")

    analyst_target_half = None
    analyst_target = info.get("targetMeanPrice")
    if analyst_target and analyst_target > 0:
        analyst_target_half = float(analyst_target) * 0.5
        models_used.append("Analyst Target (50%)")
    else:
        models_skipped.append("Analyst Target: unavailable")

    short_projection, short_basis, short_values = _weighted_ensemble(
        [
            ("Prophet", prophet_30, 0.35),
            ("ARIMA", arima_30, 0.25),
            ("Trend", trend_30, 0.20),
            ("Resistance", technical_resistance, 0.20),
        ]
    )
    medium_projection, medium_basis, medium_values = _weighted_ensemble(
        [
            ("Prophet", prophet_180, 0.30),
            ("Trend", trend_180, 0.25),
            ("Fundamental", fair_value, 0.30),
            ("Analyst x0.5", analyst_target_half, 0.15),
        ]
    )
    long_projection, long_basis, long_values = _weighted_ensemble(
        [
            ("Prophet", prophet_720, 0.25),
            ("Trend", trend_720, 0.25),
            ("Fundamental", fair_value, 0.50),
        ]
    )

    short_projection = _cap_target(short_projection or current_price, current_price, cap_value)
    medium_projection = _cap_target(medium_projection or short_projection, current_price, cap_value)
    long_projection = _cap_target(long_projection or medium_projection, current_price, cap_value)

    directional_models = [
        x
        for x in [prophet_30, arima_30, trend_30, prophet_180, arima_180, trend_180, prophet_720, trend_720, fair_value]
        if x is not None
    ]
    bearish = bool(directional_models) and all(x < current_price for x in directional_models)
    outlook = "Bearish" if bearish else "Bullish" if long_projection > current_price * 1.10 else "Neutral"

    short_low, short_high = _confidence_bounds(
        short_projection,
        short_values,
        prophet_low_30,
        prophet_high_30,
        current_price,
        cap_value,
    )
    medium_low, medium_high = _confidence_bounds(
        medium_projection,
        medium_values,
        prophet_low_180,
        prophet_high_180,
        current_price,
        cap_value,
    )
    long_low, long_high = _confidence_bounds(
        long_projection,
        long_values,
        prophet_low_720,
        prophet_high_720,
        current_price,
        cap_value,
    )

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
    elif {"Prophet", "ARIMA", "Linear Regression Trend", "Fundamental Fair Value"}.issubset(set(models_used)):
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


def analyze_stock(ticker: str, period: str = "1y", interval: str = "1d", avg_cost: float | None = None) -> dict:
    data = get_stock_data(ticker, period=period, interval=interval)
    info = get_stock_info(ticker)
    technical = analyze_technical(data)
    current_price = float(data["Close"].iloc[-1]) if not data.empty else info.get("currentPrice")
    fundamentals = analyze_fundamentals(info, current_price=current_price)
    sentiment = analyze_sentiment(ticker)

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

    technical_total = max(0, min(50, trend_score + momentum_score + volume_score + pattern_score))
    fundamental_total = round((fundamentals["fundamental_score"] / 100) * 30)
    sentiment_total = round(((sentiment["sentiment_score"] + 1) / 2) * 20)
    total = max(0, min(100, int(technical_total + fundamental_total + sentiment_total)))

    horizon = "Medium-Term Setup"
    if macd is not None and signal is not None and macd > signal and technical["trend"] != "uptrend":
        horizon = "Short-Term Opportunity"
    if technical["trend"] == "uptrend" and fundamentals["fundamental_score"] >= 65:
        horizon = "Long-Term Hold"

    entry = technical.get("entry_price")
    projections = get_price_projections(ticker, avg_cost=avg_cost)
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
        "time_horizon": horizon,
        "entry_price": entry,
        "target_price": target,
        "stop_loss": stop_loss,
        "projections": projections,
        "technical": technical,
        "fundamentals": fundamentals,
        "sentiment": sentiment,
        "score_breakdown": {
            "technical": technical_total,
            "fundamental": fundamental_total,
            "sentiment": sentiment_total,
            "trend": trend_score,
            "momentum": momentum_score,
            "volume": volume_score,
            "patterns": pattern_score,
        },
        "sell_recommendation": sell,
    }
