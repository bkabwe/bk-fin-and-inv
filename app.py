from __future__ import annotations

import pandas as pd
import streamlit as st

from modules.env import load_environment

load_environment()

from modules.data_fetcher import get_stock_data
from modules.notifications import get_notifications, get_unread_count, mark_all_read
from modules.portfolio import analyze_portfolio_holdings, get_portfolio
from modules.scoring_engine import analyze_stock
from modules.screener import run_screener
from modules.validators import sanitize_ticker

st.set_page_config(page_title="BK Stock Market Analyzer", page_icon="📈", layout="wide")
st.title("📈 BK Stock Market Analyzer")

if not st.session_state.get("disclaimer_shown"):
    st.warning(
        "⚠️ **Disclaimer**: This application is for informational and educational purposes only. "
        "It does not constitute financial advice. Always consult a qualified financial advisor "
        "before making investment decisions. Past performance is not indicative of future results.",
        icon="⚠️",
    )
    if st.button("I understand — Continue"):
        st.session_state["disclaimer_shown"] = True
        st.rerun()
    st.stop()

_, bell_col = st.columns([8, 2])
with bell_col:
    if st.button(f"🔔 {get_unread_count()}"):
        st.session_state["show_notifications"] = not st.session_state.get("show_notifications", False)

if st.session_state.get("show_notifications", False):
    with st.expander("Notifications", expanded=True):
        for note in get_notifications()[:20]:
            icon = {"buy_signal": "🟢", "sell_signal": "🔴", "alert": "🟠", "info": "ℹ️"}.get(note.get("type"), "ℹ️")
            st.write(f"{icon} {note.get('timestamp', '')} | {note.get('ticker', '')} | {note.get('message', '')}")
        if st.button("Mark all read"):
            mark_all_read()
            st.rerun()

st.sidebar.header("Navigation")
st.sidebar.info("Use the built-in Pages menu to open Portfolio, Watchlist, Screener, Stock Analysis, and News Sentiment.")
quick_ticker = st.sidebar.text_input("Quick Analyze", value="AAPL")
if st.sidebar.button("Run Quick Analyze"):
    try:
        safe_ticker = sanitize_ticker(quick_ticker)
        quick = analyze_stock(safe_ticker)
        st.sidebar.success(f"{quick['score']} - {quick['recommendation']}")
    except ValueError as exc:
        st.sidebar.error(str(exc))

st.subheader("Market Overview")
rows = []
for symbol in ["SPY", "QQQ", "DIA", "IWM"]:
    df = get_stock_data(symbol, period="5d", interval="1d")
    if df.empty or len(df) < 2:
        continue
    pct = (df["Close"].iloc[-1] - df["Close"].iloc[-2]) / df["Close"].iloc[-2] * 100
    rows.append({"Index": symbol, "Last": round(df["Close"].iloc[-1], 2), "Daily %": round(pct, 2)})
if rows:
    market_df = pd.DataFrame(rows)
    st.dataframe(market_df, use_container_width=True)
    avg = market_df["Daily %"].mean()
    st.metric("Market Sentiment", "Bullish" if avg > 0 else "Bearish", f"{avg:.2f}%")

st.subheader("Top 5 Portfolio Holdings")
if get_portfolio():
    summary = analyze_portfolio_holdings()[:5]
    if summary:
        st.dataframe(pd.DataFrame(summary)[["Ticker", "Current Value", "P&L %", "Score", "Recommendation"]], use_container_width=True)
else:
    st.caption("No holdings yet.")

st.subheader("🏆 Top 15 Picks of the Day")
if st.button("Load Top Picks"):
    picks = pd.DataFrame()
    with st.spinner("Running screener..."):
        universes = [
            ("sp500", "S&P 500"),
            ("nasdaq", "NASDAQ"),
            ("nyseamerican", "NYSE American"),
            ("otc", "OTC"),
        ]
        frames = []
        for universe, label in universes:
            try:
                frame = run_screener(universe, min_score=65, max_results=20, label=label)
                if not frame.empty:
                    frames.append(frame)
            except RuntimeError as exc:
                st.warning(f"Could not load {label} universe: {exc}")
        picks = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not picks.empty:
            picks = picks.sort_values("Score", ascending=False).head(15)
    if picks.empty:
        st.warning("No picks found.")
    else:
        display_cols = [
            "Ticker",
            "Company",
            "Index",
            "Score",
            "Recommendation",
            "Time Horizon",
            "Entry Price",
            "Target Price",
        ]
        st.dataframe(picks[[c for c in display_cols if c in picks.columns]], use_container_width=True)

st.subheader("Recent Notifications")
notes = get_notifications()[:5]
if notes:
    st.table(pd.DataFrame(notes)[["timestamp", "ticker", "type", "message"]])
else:
    st.caption("No notifications yet.")
