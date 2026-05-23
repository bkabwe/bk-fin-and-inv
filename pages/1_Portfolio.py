from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.notifications import add_notification
from modules.portfolio import add_holding, analyze_portfolio_holdings, get_portfolio, remove_holding, update_holding

st.title("💼 Portfolio Tracker")

with st.form("add_holding_form"):
    c1, c2, c3 = st.columns(3)
    ticker = c1.text_input("Ticker Symbol").upper()
    shares = c2.number_input("Number of Shares", min_value=0.0, value=1.0, step=1.0)
    avg_cost = c3.number_input("Average Cost", min_value=0.0, value=1.0)
    c4, c5 = st.columns(2)
    date_purchased = c4.date_input("Date Purchased")
    notes = c5.text_input("Notes")
    if st.form_submit_button("Add Holding") and ticker:
        add_holding(ticker, shares, avg_cost, str(date_purchased), notes)
        st.success(f"Added {ticker}")

portfolio = get_portfolio()
if not portfolio:
    st.info("No holdings yet.")
    st.stop()

df = pd.DataFrame(portfolio)
st.dataframe(df, use_container_width=True)

selected = st.selectbox("Select holding to edit/remove", df["ticker"].tolist())
selected_row = df[df["ticker"] == selected].iloc[0]
c1, c2, c3 = st.columns(3)
new_shares = c1.number_input("Update Shares", min_value=0.0, value=float(selected_row["shares"]))
new_cost = c2.number_input("Update Avg Cost", min_value=0.0, value=float(selected_row["avg_cost"]))
if c1.button("Update"):
    update_holding(selected, new_shares, new_cost)
    st.rerun()
if c3.button("Remove"):
    remove_holding(selected)
    st.rerun()

if st.button("Analyze All Holdings"):
    with st.spinner("Analyzing holdings..."):
        rows = analyze_portfolio_holdings()
    if rows:
        result_df = pd.DataFrame(rows)
        st.dataframe(result_df, use_container_width=True)
        for _, row in result_df.iterrows():
            st.markdown(f"### {row['Ticker']}: {row['Recommendation']} ({row['Score']})")
            st.write(f"Entry paid vs current vs target: {row['Avg Cost']} / {row['Current Price']} / {row['Target Price']}")
            if row.get("Sell Signal"):
                st.error(f"Sell signal: {row['Sell Signal']}")
                add_notification("Sell signal", row["Sell Signal"], row["Ticker"], "sell_signal")
        st.plotly_chart(px.pie(result_df, values="Current Value", names="Ticker", title="Portfolio Allocation"), use_container_width=True)
        st.plotly_chart(px.bar(result_df, x="Ticker", y="P&L $", color="P&L $", title="P&L by Holding"), use_container_width=True)
