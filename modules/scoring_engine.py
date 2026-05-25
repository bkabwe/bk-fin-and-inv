from __future__ import annotations

from functools import lru_cache

import numpy as np

from modules.data_fetcher import get_stock_data, get_stock_info
from modules.fundamental_analysis import analyze_fundamentals
from modules.sentiment_analysis import analyze_sentiment
from modules.technical_analysis import analyze_technical

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


def _safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        if np.isnan(out):
            return None
        return out
    except Exception:
        return None


def _round_price(value: float | None) -> float:
    return round(max(value or 0.0, 0.01), 2)


def get_price_projections(ticker: str) -> dict:
    """
    Returns projected price targets based on technical analysis and fundamental metrics.
    """
    data = get_stock_data(ticker, period="1y", interval="1d")
    info = get_stock_info(ticker)
    technical = analyze_technical(data)

    current_price = _safe_float(data["Close"].iloc[-1] if not data.empty else None) or _safe_float(
        info.get("currentPrice") or info.get("regularMarketPrice")
    )
    if not current_price:
        return {
            "short_term_target": 0.0,
            "short_term_upside": 0.0,
            "short_term_basis": "Insufficient market data.",
            "medium_term_target": 0.0,
            "medium_term_upside": 0.0,
            "medium_term_basis": "Insufficient market data.",
            "long_term_target": 0.0,
            "long_term_upside": 0.0,
            "long_term_basis": "Insufficient market data.",
            "current_price": 0.0,
            "recommendation_to_sell_at": "Insufficient data to generate a sell recommendation.",
        }

    indicators = technical.get("indicators", {})
    resistance_levels = [x for x in technical.get("resistance_levels", []) if _safe_float(x)]
    resistance_above = sorted([x for x in resistance_levels if x > current_price])
    nearest_resistance = resistance_above[0] if resistance_above else None
    major_resistance = resistance_above[-1] if resistance_above else None

    rsi = _safe_float(indicators.get("rsi"))
    macd = _safe_float(indicators.get("macd"))
    macd_signal = _safe_float(indicators.get("macd_signal"))
    bb_high = _safe_float(indicators.get("bb_high"))
    atr = _safe_float(indicators.get("atr") or technical.get("atr")) or 0.0

    bullish_momentum = bool(macd is not None and macd_signal is not None and macd > macd_signal and (rsi is None or rsi < 65))
    if nearest_resistance and bb_high and bullish_momentum:
        short_target = min(nearest_resistance, bb_high)
        short_basis = "Nearest resistance blended with bullish MACD momentum and upper Bollinger Band."
    elif nearest_resistance and bb_high:
        short_target = min(nearest_resistance, bb_high)
        short_basis = "Nearest resistance constrained by upper Bollinger Band."
    elif nearest_resistance:
        short_target = nearest_resistance
        short_basis = "Nearest technical resistance above current price."
    elif bb_high:
        short_target = bb_high
        short_basis = "Upper Bollinger Band projection."
    else:
        short_target = current_price * 1.05
        short_basis = "Fallback 5% move due to missing resistance levels."

    analyst_target = _safe_float(info.get("targetMeanPrice"))
    atr_ratio = atr / current_price if current_price else 0.0
    technical_projection = current_price * (1 + atr_ratio * 3)
    if analyst_target:
        medium_target = analyst_target * 0.6 + technical_projection * 0.4
        medium_basis = "60% analyst target + 40% ATR technical projection."
    else:
        medium_target = major_resistance or (current_price * 1.15)
        medium_basis = (
            "Next major resistance level."
            if major_resistance
            else "Fallback 15% move due to missing analyst target and resistance."
        )

    sector_avg_pe = {
        "Technology": 28.0,
        "Financial Services": 14.0,
        "Healthcare": 22.0,
        "Consumer Cyclical": 22.0,
        "Communication Services": 20.0,
        "Industrials": 19.0,
        "Energy": 13.0,
        "Utilities": 17.0,
        "Real Estate": 18.0,
        "Basic Materials": 16.0,
        "Consumer Defensive": 20.0,
    }
    sector = str(info.get("sector") or "")
    sector_pe = sector_avg_pe.get(sector, 20.0)
    eps_forward = _safe_float(info.get("forwardEps")) or _safe_float(info.get("trailingEps"))
    forward_pe = _safe_float(info.get("forwardPE"))
    fair_value_estimate = eps_forward * sector_pe if eps_forward and forward_pe else None

    technical_df = technical.get("data")

    is_otc_or_penny = ticker.upper().endswith("Y") or ticker.upper().endswith("F") or current_price < 5 or info.get("exchange") == "PNK"
    if fair_value_estimate:
        long_target = max(fair_value_estimate, current_price * 1.25)
        long_basis = f"Fundamental fair value from EPS and sector P/E ({sector_pe:.1f}) with 25% floor."
    else:
        if technical_df is not None and not technical_df.empty and "sma200" in technical_df:
            sma200 = technical_df["sma200"].dropna()
            if len(sma200) >= 20:
                x = np.arange(len(sma200))
                slope = np.polyfit(x, sma200.values, 1)[0]
                months = 378 if is_otc_or_penny else 252
                long_target = current_price + (slope * months)
            else:
                long_target = current_price * 1.25
        else:
            long_target = current_price * 1.25
        long_basis = (
            "Trend-based 200-day SMA projection over 18 months for limited-fundamental OTC/penny stock."
            if is_otc_or_penny
            else "Trend-based 200-day SMA projection over 12 months."
        )
        if long_target <= 0:
            long_target = current_price * 1.25
            long_basis = "Fallback long-term projection using 25% appreciation floor."
        long_target = max(long_target, current_price * 1.05)

    short_target = _round_price(short_target)
    medium_target = _round_price(medium_target)
    long_target = _round_price(long_target)
    current_price = _round_price(current_price)

    short_upside = round(((short_target - current_price) / current_price) * 100, 2)
    medium_upside = round(((medium_target - current_price) / current_price) * 100, 2)
    long_upside = round(((long_target - current_price) / current_price) * 100, 2)

    if current_price >= medium_target:
        sell_text = "Stock may be near fair value. Consider taking profits."
    elif short_upside <= 0:
        sell_text = f"Hold for recovery to break-even at ${medium_target:.2f} before considering sell."
    else:
        sell_text = (
            f"Consider selling 50% of position near ${short_target:.2f} (short-term target). "
            f"Hold remainder for medium-term target of ${medium_target:.2f}."
        )

    return {
        "short_term_target": short_target,
        "short_term_upside": short_upside,
        "short_term_basis": short_basis,
        "medium_term_target": medium_target,
        "medium_term_upside": medium_upside,
        "medium_term_basis": medium_basis,
        "long_term_target": long_target,
        "long_term_upside": long_upside,
        "long_term_basis": long_basis,
        "current_price": current_price,
        "recommendation_to_sell_at": sell_text,
    }
