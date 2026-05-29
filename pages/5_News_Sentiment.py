from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.sentiment_analysis import analyze_sentiment
from modules.validators import sanitize_ticker

st.title("📰 News & Sentiment")
ticker_input = st.text_input("Ticker", value="AAPL")
if ticker_input:
    try:
        ticker = sanitize_ticker(ticker_input)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    result = analyze_sentiment(ticker)
    st.write(f"Overall Sentiment: **{result['sentiment_label']}** ({result['sentiment_score']})")

    m1, m2, m3 = st.columns(3)
    m1.metric("Short Interest Ratio", result.get("short_ratio"))
    m2.metric("Short % of Float", result.get("short_pct_float"))
    m3.metric("Put/Call Ratio", result.get("put_call_ratio"))
    st.caption(f"Options Sentiment: {result.get('options_sentiment', 'Neutral')}")

    headlines = pd.DataFrame(result.get("headlines", []))
    if not headlines.empty:
        st.dataframe(headlines, use_container_width=True)
    st.plotly_chart(
        go.Figure(go.Indicator(mode="gauge+number", value=result["sentiment_score"], gauge={"axis": {"range": [-1, 1]}})),
        use_container_width=True,
    )
    if not headlines.empty:
        trend = headlines[["score"]].copy()
        trend["n"] = range(1, len(trend) + 1)
        st.line_chart(trend.set_index("n")["score"])

st.subheader("Market-wide sentiment")
market = []
for symbol in ["SPY", "QQQ", "DIA", "IWM"]:
    sentiment = analyze_sentiment(symbol)
    market.append(
        {
            "Ticker": symbol,
            "Sentiment": sentiment["sentiment_label"],
            "Score": sentiment["sentiment_score"],
            "Put/Call": sentiment.get("put_call_ratio"),
        }
    )
st.table(pd.DataFrame(market))
