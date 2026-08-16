from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from ta.momentum import RSIIndicator
from ta.trend import MACD

from modules.data_fetcher import get_stock_data
from modules.prediction_tracker import record_prediction
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

    # Record projections for the track-record feature (one per horizon).
    try:
        _proj = analysis.get("projections") or {}
        _current_price = float(_proj.get("current_price") or analysis.get("current_price") or 0)
        _company = analysis.get("company") or ticker
        _score = int(analysis.get("score") or 0)
        _dq = str(_proj.get("data_quality") or "Limited")
        for _horizon_key, _target_k, _low_k, _high_k, _upside_k, _basis_k in [
            ("short_term", "short_term_target", "short_term_low", "short_term_high", "short_term_upside", "short_term_basis"),
            ("medium_term", "medium_term_target", "medium_term_low", "medium_term_high", "medium_term_upside", "medium_term_basis"),
            ("long_term", "long_term_target", "long_term_low", "long_term_high", "long_term_upside", "long_term_basis"),
        ]:
            _tp = float(_proj.get(_target_k) or 0)
            _up = float(_proj.get(_upside_k) or 0)
            if _current_price > 0 and _tp > 0:
                record_prediction(
                    ticker=ticker,
                    company=_company,
                    horizon=_horizon_key,
                    current_price=_current_price,
                    target_price=_tp,
                    target_low=float(_proj.get(_low_k) or _tp * 0.95),
                    target_high=float(_proj.get(_high_k) or _tp * 1.05),
                    projected_upside_pct=_up,
                    score=_score,
                    data_quality=_dq,
                    models_used=str(_proj.get(_basis_k) or ""),
                    source="stock_analysis",
                )
    except Exception:
        pass  # Never block UI for tracking failures

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

    t1, t2, t3, t4, t5, t6, t7, t8, t9, t10 = st.tabs(
        [
            "Technical Analysis",
            "Fundamental Analysis",
            "Schabacker Patterns",
            "Gaps",
            "Trendlines",
            "Relative Strength",
            "Macro Regime",
            "Breakout",
            "News & Sentiment",
            "Score Breakdown",
        ]
    )

    with t1:
        st.json(analysis["technical"])

    with t2:
        st.table(pd.DataFrame([analysis["fundamentals"]["metrics"]]).T)

    with t3:
        st.table(pd.DataFrame(analysis["technical"].get("patterns", [])))

    with t4:
        gaps_df = pd.DataFrame(analysis["technical"].get("gaps", []))
        if gaps_df.empty:
            st.info("No recent gaps detected.")
        else:
            st.dataframe(gaps_df[["type", "date", "direction", "gap_pct", "significance"]], use_container_width=True)

    with t5:
        st.json(analysis["technical"].get("trendlines", {}))

    with t6:
        st.table(pd.DataFrame([analysis.get("relative_strength", {})]))

    with t7:
        macro = analysis.get("macro_regime", {})
        st.metric("VIX", macro.get("vix"))
        st.metric("10Y Yield", macro.get("yield_10y"))
        st.metric("Market Regime", macro.get("market_regime"))
        c_bull, c_bear = st.columns(2)
        c_bull.write("**Bullish Sectors**")
        c_bull.write(", ".join(macro.get("bullish_sectors", [])) or "None")
        c_bear.write("**Bearish Sectors**")
        c_bear.write(", ".join(macro.get("bearish_sectors", [])) or "None")

    with t8:
        st.json(analysis["technical"].get("breakout", {}))

    with t9:
        st.write(f"Sentiment: {analysis['sentiment']['sentiment_label']} ({analysis['sentiment']['sentiment_score']})")
        st.table(pd.DataFrame(analysis["sentiment"].get("headlines", [])))

    with t10:
        breakdown_df = pd.DataFrame([{"Component": k, "Points": v} for k, v in analysis["score_breakdown"].items()]).set_index("Component")
        st.bar_chart(breakdown_df)
