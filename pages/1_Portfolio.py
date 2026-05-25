from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
        st.session_state["portfolio_analysis_rows"] = analyze_portfolio_holdings()
    st.session_state["portfolio_just_analyzed"] = True

rows = st.session_state.get("portfolio_analysis_rows", [])
just_analyzed = st.session_state.pop("portfolio_just_analyzed", False)
if rows:
    result_df = pd.DataFrame(rows)
    st.dataframe(result_df, use_container_width=True)
    for _, row in result_df.iterrows():
        st.markdown(f"### {row['Ticker']}: {row['Recommendation']} ({row['Score']})")
        st.write(f"Entry paid vs current vs target: {row['Avg Cost']} / {row['Current Price']} / {row['Target Price']}")
        if row.get("Sell Signal"):
            st.error(f"Sell signal: {row['Sell Signal']}")
            if just_analyzed:
                add_notification("Sell signal", row["Sell Signal"], row["Ticker"], "sell_signal")
    st.plotly_chart(px.pie(result_df, values="Current Value", names="Ticker", title="Portfolio Allocation"), use_container_width=True)
    st.plotly_chart(px.bar(result_df, x="Ticker", y="P&L $", color="P&L $", title="P&L by Holding"), use_container_width=True)

    st.subheader("📈 Portfolio Price Projections")
    for _, row in result_df.iterrows():
        ticker = row["Ticker"]
        current_price = float(row.get("Current Price") or 0)
        avg_cost = float(row.get("Avg Cost") or 0)
        short_target = float(row.get("short_term_target") or 0)
        medium_target = float(row.get("medium_term_target") or 0)
        long_target = float(row.get("long_term_target") or 0)

        if current_price > long_target:
            status = "🔴 Near or above fair value"
        elif medium_target <= current_price <= long_target:
            status = "🟡 Approaching target"
        else:
            status = "🟢 Below all targets, upside available"

        with st.expander(f"{ticker} — ${current_price:.2f} | {status}", expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.markdown(
                f"**Short-Term (1-4 weeks)**  \nTarget: ${short_target:.2f}  \nUpside: {row.get('short_term_upside', 0):+.2f}%  \nBasis: {row.get('short_term_basis', '')}"
            )
            c2.markdown(
                f"**Medium-Term (1-6 months)**  \nTarget: ${medium_target:.2f}  \nUpside: {row.get('medium_term_upside', 0):+.2f}%  \nBasis: {row.get('medium_term_basis', '')}"
            )
            c3.markdown(
                f"**Long-Term (6-24 months)**  \nTarget: ${long_target:.2f}  \nUpside: {row.get('long_term_upside', 0):+.2f}%  \nBasis: {row.get('long_term_basis', '')}"
            )

            rec = row.get("recommendation_to_sell_at", "")
            if current_price >= avg_cost:
                st.success(f"Sell Recommendation: {rec}")
            else:
                st.warning(f"Sell Recommendation: {rec}")

            fig = go.Figure()
            x = [0, 1]
            fig.add_trace(go.Scatter(x=x, y=[current_price, current_price], mode="lines", name="Current Price", line=dict(color="black", width=3)))
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=[avg_cost, avg_cost],
                    mode="lines",
                    name="Entry / Avg Cost",
                    line=dict(color="gray", dash="dash"),
                )
            )
            fig.add_trace(go.Scatter(x=x, y=[short_target, short_target], mode="lines", name="Short Target", line=dict(color="orange")))
            fig.add_trace(go.Scatter(x=x, y=[medium_target, medium_target], mode="lines", name="Medium Target", line=dict(color="blue")))
            fig.add_trace(go.Scatter(x=x, y=[long_target, long_target], mode="lines", name="Long Target", line=dict(color="green")))
            fig.update_layout(
                title=f"{ticker} Price Targets",
                xaxis=dict(showticklabels=False, title=""),
                yaxis_title="Price ($)",
                height=320,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
