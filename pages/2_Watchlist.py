from __future__ import annotations

import pandas as pd
import streamlit as st

from modules.data_fetcher import get_stock_data
from modules.portfolio import add_holding, add_to_watchlist, get_watchlist, remove_from_watchlist
from modules.scoring_engine import analyze_stock
from modules.validators import sanitize_ticker

st.title("👀 Watchlist")

c1, c2 = st.columns([3, 1])
new_ticker = c1.text_input("Add to Watchlist", value="").upper()
if c2.button("Add") and new_ticker:
    try:
        add_to_watchlist(sanitize_ticker(new_ticker))
        st.rerun()
    except ValueError as exc:
        st.error(str(exc))

watchlist = get_watchlist()
if not watchlist:
    st.info("Watchlist is empty.")
    st.stop()

sort_by = st.selectbox("Sort by", ["score", "alphabetical", "% change"])
rows = []
for ticker in watchlist:
    try:
        analysis = analyze_stock(ticker)
        prices = get_stock_data(ticker, period="5d", interval="1d")
        change = None
        if not prices.empty and len(prices) > 1:
            change = ((prices["Close"].iloc[-1] - prices["Close"].iloc[-2]) / prices["Close"].iloc[-2]) * 100
        rows.append(
            {
                "Ticker": ticker,
                "Current Price": analysis.get("current_price"),
                "Daily %": round(change, 2) if change is not None else None,
                "Score": analysis.get("score"),
                "Recommendation": analysis.get("recommendation"),
            }
        )
    except Exception:
        continue

frame = pd.DataFrame(rows)
if sort_by == "score":
    frame = frame.sort_values("Score", ascending=False)
elif sort_by == "alphabetical":
    frame = frame.sort_values("Ticker")
else:
    frame = frame.sort_values("Daily %", ascending=False)
st.dataframe(frame, use_container_width=True)

selected = st.selectbox("Select ticker", frame["Ticker"].tolist())
b1, b2 = st.columns(2)
if b1.button("Remove from Watchlist"):
    remove_from_watchlist(selected)
    st.rerun()
if b2.button("Move to Portfolio"):
    current = float(frame[frame["Ticker"] == selected]["Current Price"].iloc[0] or 0)
    add_holding(selected, 1, current, str(pd.Timestamp.now().date()))
    remove_from_watchlist(selected)
    st.rerun()
