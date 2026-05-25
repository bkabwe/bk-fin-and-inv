from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from ta.momentum import RSIIndicator
from ta.trend import MACD

from modules.data_fetcher import get_stock_data
from modules.scoring_engine import analyze_stock
from modules.validators import sanitize_ticker

st.title("📊 Deep Dive Stock Analysis")
ticker_input = st.text_input("Ticker", value="AAPL")
period_map = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "2Y": "2y", "5Y": "5y"}
period = st.selectbox("Time Period", list(period_map.keys()), index=3)

if ticker_input:
    try:
        ticker = sanitize_ticker(ticker_input)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    analysis = analyze_stock(ticker, period=period_map[period], interval="1d")
    df = get_stock_data(ticker, period=period_map[period], interval="1d")
    if df.empty:
        st.error("No price data available.")
        st.stop()
    tech = analysis["technical"]["data"]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="OHLC"), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", opacity=0.2), row=1, col=1)
    for col, name in [("sma20", "SMA 20"), ("sma50", "SMA 50"), ("sma200", "SMA 200")]:
        if col in tech:
            fig.add_trace(go.Scatter(x=tech.index, y=tech[col], mode="lines", name=name), row=1, col=1)
    for lvl in analysis["technical"].get("support_levels", []):
        fig.add_hline(y=lvl, line_dash="dash", line_color="green", row=1, col=1)
    for lvl in analysis["technical"].get("resistance_levels", []):
        fig.add_hline(y=lvl, line_dash="dash", line_color="orange", row=1, col=1)
    if "bb_high" in tech.columns:
        fig.add_trace(
            go.Scatter(
                x=tech.index,
                y=tech["bb_high"],
                mode="lines",
                name="BB Upper",
                line=dict(color="gray", dash="dot"),
            ),
            row=1,
            col=1,
        )
    if "bb_low" in tech.columns:
        fig.add_trace(
            go.Scatter(
                x=tech.index,
                y=tech["bb_low"],
                mode="lines",
                name="BB Lower",
                line=dict(color="gray", dash="dot"),
                fill="tonexty",
                fillcolor="rgba(128,128,128,0.1)",
            ),
            row=1,
            col=1,
        )
    for value, color in [
        (analysis.get("entry_price"), "green"),
        (analysis.get("target_price"), "blue"),
        (analysis.get("stop_loss"), "red"),
    ]:
        if value:
            fig.add_hline(y=value, line_dash="dot", line_color=color, row=1, col=1)

    rsi = tech["rsi"] if "rsi" in tech.columns else RSIIndicator(df["Close"], window=14).rsi()
    if "macd" in tech.columns and "macd_signal" in tech.columns:
        macd = tech["macd"]
        signal = tech["macd_signal"]
    else:
        macd_calc = MACD(df["Close"], 26, 12, 9)
        macd = macd_calc.macd()
        signal = macd_calc.macd_signal()

    fig.add_trace(go.Scatter(x=rsi.index, y=rsi, mode="lines", name="RSI"), row=2, col=1)
    fig.add_trace(go.Scatter(x=macd.index, y=macd, mode="lines", name="MACD"), row=3, col=1)
    fig.add_trace(go.Scatter(x=signal.index, y=signal, mode="lines", name="Signal"), row=3, col=1)
    for i, p in enumerate(analysis["technical"].get("patterns", [])[:3], start=1):
        fig.add_annotation(x=df.index[-1], y=float(df["Close"].iloc[-1]) * (1 + i * 0.02), text=p["name"], showarrow=True)
    fig.update_layout(height=900, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    expected = None
    if analysis.get("entry_price") and analysis.get("target_price"):
        expected = (analysis["target_price"] - analysis["entry_price"]) / analysis["entry_price"] * 100
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Score", analysis["score"])
    c2.metric("Recommendation", analysis["recommendation"])
    c3.metric("Time Horizon", analysis["time_horizon"])
    c4.metric("Entry / Target", f"{analysis.get('entry_price')} / {analysis.get('target_price')}")
    c5.metric("Stop / Expected %", f"{analysis.get('stop_loss')} / {expected:.2f}%" if expected is not None else str(analysis.get("stop_loss")))

    t1, t2, t3, t4 = st.tabs(["Technical Analysis", "Fundamental Analysis", "Detected Patterns", "News & Sentiment"])
    with t1:
        st.json(analysis["technical"])
    with t2:
        st.table(pd.DataFrame([analysis["fundamentals"]["metrics"]]).T)
    with t3:
        st.table(pd.DataFrame(analysis["technical"].get("patterns", [])))
    with t4:
        st.write(f"Sentiment: {analysis['sentiment']['sentiment_label']} ({analysis['sentiment']['sentiment_score']})")
        st.table(pd.DataFrame(analysis["sentiment"].get("headlines", [])))
    st.bar_chart(
        pd.DataFrame(
            [
                {"Component": "Technical", "Points": analysis["score_breakdown"]["technical"]},
                {"Component": "Fundamental", "Points": analysis["score_breakdown"]["fundamental"]},
                {"Component": "Sentiment", "Points": analysis["score_breakdown"]["sentiment"]},
            ]
        ).set_index("Component")
    )
