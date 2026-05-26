from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from modules.data_fetcher import get_nasdaq_tickers, get_nyseamerican_tickers, get_otc_tickers, get_sp500_tickers, get_stock_data
from modules.portfolio import get_portfolio
from modules.scoring_engine import analyze_stock

try:  # pragma: no cover
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:  # pragma: no cover
    PROPHET_AVAILABLE = False

try:  # pragma: no cover
    from statsmodels.tsa.arima.model import ARIMA

    ARIMA_AVAILABLE = True
except ImportError:  # pragma: no cover
    ARIMA_AVAILABLE = False

try:  # pragma: no cover
    from sklearn.linear_model import LinearRegression

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    SKLEARN_AVAILABLE = False


st.title("💰 Profit Opportunities")
st.caption("Identifies stocks with the highest projected profit potential based on multi-model predictive analysis.")

horizon = st.radio(
    "Time Horizon",
    ["Short-Term (1–4 weeks)", "Medium-Term (1–6 months)", "Long-Term (6–24 months)"],
)

selected_universes = st.multiselect(
    "Universe",
    ["S&P 500", "NASDAQ", "NYSE American", "OTC", "My Portfolio"],
    default=["S&P 500", "NASDAQ"],
)
min_upside_pct = st.slider("Min Upside %", min_value=5, max_value=100, value=15)
st.info(
    "ℹ️ This page analyses **every stock** in your selected universes and surfaces "
    "the highest projected profit opportunities. Larger universes will take longer."
)
large = [u for u in selected_universes if u in ("NASDAQ", "OTC")]
if large:
    st.warning(
        f"⚠️ {' and '.join(large)} contain thousands of stocks. "
        "A full scan can take 30–60 minutes. Consider starting with S&P 500 or NYSE American."
    )


@st.cache_data(ttl=1800)
def estimate_target_date(ticker: str, target_price: float, horizon_value: str, rsi: float | None = None) -> tuple[str, str]:
    """
    Returns (estimated_date_range, confidence) for when ticker may reach target_price.
    """

    def _format_date_range(center_date: date, confidence: str) -> str:
        windows = {
            "Short-Term (1–4 weeks)": {"High": 4, "Medium": 6, "Low": 8},
            "Medium-Term (1–6 months)": {"High": 14, "Medium": 21, "Low": 30},
            "Long-Term (6–24 months)": {"High": 30, "Medium": 45, "Low": 60},
        }
        span = windows[horizon_value][confidence]
        start = center_date - timedelta(days=span)
        end = center_date + timedelta(days=span)
        return f"~{start.strftime('%b %d, %Y')} – {end.strftime('%b %d, %Y')}"

    today = date.today()
    if horizon_value == "Short-Term (1–4 weeks)":
        days = 21
        if rsi is not None:
            if rsi < 50:
                days = 28
            elif rsi > 60:
                days = 14
        est = today + timedelta(days=days)
        return _format_date_range(est, "Medium"), "Medium"

    fallback_days = 90 if horizon_value == "Medium-Term (1–6 months)" else 365
    max_days = 180 if horizon_value == "Medium-Term (1–6 months)" else 720

    try:
        data = get_stock_data(ticker, period="2y", interval="1d")
    except Exception:
        data = pd.DataFrame()

    if data.empty or "Close" not in data:
        est = today + timedelta(days=fallback_days)
        return _format_date_range(est, "Low"), "Low"

    close = data["Close"].dropna().astype(float)
    if close.empty:
        est = today + timedelta(days=fallback_days)
        return _format_date_range(est, "Low"), "Low"

    prophet_day: int | None = None
    arima_hit = False
    trend_day: int | None = None

    if PROPHET_AVAILABLE and len(close) >= 60:
        try:
            prophet_df = close.reset_index().rename(columns={close.index.name or "Date": "ds", "Close": "y"})
            if "ds" not in prophet_df.columns:
                prophet_df = prophet_df.rename(columns={prophet_df.columns[0]: "ds"})
            prophet_df["ds"] = pd.to_datetime(prophet_df["ds"]).dt.tz_localize(None)
            prophet_df = prophet_df[["ds", "y"]]

            model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
            model.fit(prophet_df)
            forecast = model.predict(model.make_future_dataframe(periods=max_days, freq="D"))
            future_only = forecast.tail(max_days)
            crossing = future_only[future_only["yhat"] >= float(target_price)]
            if not crossing.empty:
                cross_ds = pd.to_datetime(crossing.iloc[0]["ds"]).date()
                prophet_day = max(1, (cross_ds - today).days)
        except Exception:
            prophet_day = None

    if ARIMA_AVAILABLE and len(close) >= 60:
        try:
            arima_forecast = ARIMA(close.tail(252), order=(5, 1, 0)).fit().forecast(steps=max_days)
            arima_hit = bool((arima_forecast >= float(target_price)).any())
        except Exception:
            arima_hit = False

    if len(close) >= 30:
        try:
            series = close.tail(200).reset_index(drop=True)
            x = np.arange(len(series)).reshape(-1, 1)
            if SKLEARN_AVAILABLE:
                model = LinearRegression()
                model.fit(x, series.values)
                slope = float(model.coef_[0])
                intercept = float(model.intercept_)
            else:
                slope, intercept = np.polyfit(np.arange(len(series)), series.values, 1)
            if slope > 0:
                day_x = int(round((float(target_price) - intercept) / slope))
                if day_x > len(series):
                    trend_day = day_x - len(series)
        except Exception:
            trend_day = None

    if prophet_day is not None:
        est = today + timedelta(days=min(max_days, prophet_day))
        confidence = "High" if arima_hit else "Medium"
        return _format_date_range(est, confidence), confidence

    if trend_day is not None:
        est = today + timedelta(days=min(max_days, max(1, trend_day)))
        confidence = "Medium" if arima_hit else "Low"
        return _format_date_range(est, confidence), confidence

    est = today + timedelta(days=fallback_days)
    confidence = "Medium" if arima_hit else "Low"
    return _format_date_range(est, confidence), confidence


def _collect_tickers(universes: list[str]) -> tuple[list[str], dict[str, str]]:
    universe_fetchers = {
        "S&P 500": get_sp500_tickers,
        "NASDAQ": get_nasdaq_tickers,
        "NYSE American": get_nyseamerican_tickers,
        "OTC": get_otc_tickers,
    }

    source_map: dict[str, set[str]] = {}
    ordered: list[str] = []

    for universe_name in universes:
        if universe_name == "My Portfolio":
            holdings = get_portfolio() or []
            if not holdings:
                st.info("No portfolio holdings found.")
                continue
            values = [str(x.get("ticker", "")).strip().upper() for x in holdings if x.get("ticker")]
        else:
            fetcher = universe_fetchers[universe_name]
            try:
                values = fetcher()
            except Exception as exc:
                st.warning(f"Could not load {universe_name} universe: {exc}")
                continue

        for ticker in values:
            clean = str(ticker).strip().upper().replace(".", "-")
            if not clean:
                continue
            if clean not in source_map:
                source_map[clean] = set()
                ordered.append(clean)
            source_map[clean].add(universe_name)

    return ordered, {k: ", ".join(sorted(v)) for k, v in source_map.items() if k in ordered}


if st.button("Run Analysis"):
    tickers, source_labels = _collect_tickers(selected_universes)
    if not tickers:
        st.warning("No tickers available for analysis.")
    else:
        progress = st.progress(0.0)
        rows: list[dict] = []
        for i, ticker in enumerate(tickers, start=1):
            try:
                analysis = analyze_stock(ticker)
                projections = analysis.get("projections") or {}
                current_price = float(projections.get("current_price") or analysis.get("current_price") or 0)
                if current_price <= 0:
                    continue
                rows.append(
                    {
                        "ticker": ticker,
                        "company": analysis.get("company") or ticker,
                        "index": source_labels.get(ticker, ""),
                        "score": int(analysis.get("score") or 0),
                        "current_price": current_price,
                        "technical": analysis.get("technical") or {},
                        "projections": projections,
                    }
                )
            except Exception:
                continue
            progress.progress(i / len(tickers))
        st.session_state["profit_opportunities_scan"] = rows
        st.session_state["profit_opportunities_scanned_total"] = len(tickers)

scan_rows = st.session_state.get("profit_opportunities_scan", [])
total_scanned = st.session_state.get("profit_opportunities_scanned_total", 0)

if total_scanned:
    max_results = st.slider("Max Results to Display", 5, 200, 25)
    horizon_map = {
        "Short-Term (1–4 weeks)": ("short_term_target", "short_term_upside", "short_term_basis"),
        "Medium-Term (1–6 months)": ("medium_term_target", "medium_term_upside", "medium_term_basis"),
        "Long-Term (6–24 months)": ("long_term_target", "long_term_upside", "long_term_basis"),
    }
    target_key, upside_key, basis_key = horizon_map[horizon]

    display_rows = []
    for row in scan_rows:
        projections = row.get("projections") or {}
        target_price = float(projections.get(target_key) or 0)
        upside = float(projections.get(upside_key) or 0)
        if target_price <= 0 or upside < min_upside_pct or upside <= 0:
            continue
        rsi = row.get("technical", {}).get("indicators", {}).get("rsi")
        est_date, confidence = estimate_target_date(row["ticker"], target_price, horizon, rsi)
        display_rows.append(
            {
                "Ticker": row["ticker"],
                "Company": row["company"],
                "Index": row["index"],
                "Score": row["score"],
                "Current Price": row["current_price"],
                "Target Price": target_price,
                "Projected Upside %": upside,
                "Est. Target Date": est_date,
                "Confidence": confidence,
                "Basis": projections.get(basis_key, ""),
            }
        )

    if not display_rows:
        st.warning("No stocks met the minimum upside threshold. Try lowering the Min Upside %.")
        st.caption(f"✅ Scanned **{total_scanned}** stocks → **0** met the upside threshold")
    else:
        results_df = pd.DataFrame(display_rows).sort_values("Projected Upside %", ascending=False).head(max_results)
        st.caption(
            f"✅ Scanned **{total_scanned}** stocks → "
            f"**{len(display_rows)}** met the upside threshold → "
            f"Showing top **{len(results_df)}**"
        )

        styled = results_df.style.format(
            {
                "Current Price": "${:,.2f}",
                "Target Price": "${:,.2f}",
                "Projected Upside %": "{:.2f}%",
            }
        )
        st.dataframe(styled, use_container_width=True)

        chart_df = results_df[["Ticker", "Projected Upside %", "Confidence"]].copy()
        fig = px.bar(
            chart_df,
            x="Ticker",
            y="Projected Upside %",
            color="Confidence",
            color_discrete_map={"High": "#2ca02c", "Medium": "#ffbf00", "Low": "#ff8c00"},
            title=f"Projected Upside by Stock ({horizon})",
        )
        st.plotly_chart(fig, use_container_width=True)

        top = results_df.iloc[0]
        st.success(
            f"🏆 Top Pick: **{top['Ticker']}** ({top['Company']}) — "
            f"Projected {top['Projected Upside %']:.1f}% upside by {top['Est. Target Date']} "
            f"(Confidence: {top['Confidence']})"
        )

        st.download_button(
            "Export to CSV",
            results_df.to_csv(index=False),
            file_name="profit_opportunities.csv",
            mime="text/csv",
        )

st.caption("⚠️ Projected dates and upside figures are model estimates based on historical data. They are not guarantees of future performance.")
