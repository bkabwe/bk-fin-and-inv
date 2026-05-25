from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.notifications import add_notification
from modules.portfolio import add_holding, analyze_portfolio_holdings, get_portfolio, remove_holding, update_holding


def _fmt_money(value) -> str:
    try:
        return f"\\${float(value):.2f}"
    except Exception:
        return "\\$0.00"


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
    with st.spinner("Running predictive models..."):
        rows = analyze_portfolio_holdings()
    if rows:
        result_df = pd.DataFrame(rows)
        st.dataframe(result_df, use_container_width=True)
        for _, row in result_df.iterrows():
            projection = row.get("Projection") or {}
            models_used = projection.get("models_used", [])
            data_quality = projection.get("data_quality", "Limited")
            quality_badge = {"Full": "🟢 Full Data", "Limited": "🟡 Limited Data", "Technical Only": "🔴 Technical Only"}.get(
                data_quality,
                "🟡 Limited Data",
            )

            st.markdown(f"### {row['Ticker']}: {row['Recommendation']} ({row['Score']})")
            st.write(f"Data quality: {quality_badge}")
            st.write(
                f"Short target: **{_fmt_money(projection.get('short_term_target', row['Target Price']))}** "
                f"*(range: {_fmt_money(projection.get('short_term_low', row['Target Price']))} – "
                f"{_fmt_money(projection.get('short_term_high', row['Target Price']))})*"
            )
            st.write(
                f"Medium target: **{_fmt_money(projection.get('medium_term_target', row['Target Price']))}** "
                f"*(range: {_fmt_money(projection.get('medium_term_low', row['Target Price']))} – "
                f"{_fmt_money(projection.get('medium_term_high', row['Target Price']))})*"
            )
            st.write(
                f"Long target: **{_fmt_money(projection.get('long_term_target', row['Target Price']))}** "
                f"*(range: {_fmt_money(projection.get('long_term_low', row['Target Price']))} – "
                f"{_fmt_money(projection.get('long_term_high', row['Target Price']))})*"
            )
            if models_used:
                st.caption("Models used: " + ", ".join(models_used))
            if data_quality == "Technical Only":
                st.warning(
                    "⚠️ Projections based on technical analysis only due to limited fundamental data for this OTC stock."
                )
            if row.get("Sell Signal"):
                st.error(f"Sell signal: {row['Sell Signal']}")
                add_notification("Sell signal", row["Sell Signal"], row["Ticker"], "sell_signal")

            points = ["Current", "Short", "Medium", "Long"]
            lows = [
                row["Current Price"],
                projection.get("short_term_low", row["Current Price"]),
                projection.get("medium_term_low", row["Current Price"]),
                projection.get("long_term_low", row["Current Price"]),
            ]
            highs = [
                row["Current Price"],
                projection.get("short_term_high", row["Current Price"]),
                projection.get("medium_term_high", row["Current Price"]),
                projection.get("long_term_high", row["Current Price"]),
            ]
            targets = [
                row["Current Price"],
                projection.get("short_term_target", row["Current Price"]),
                projection.get("medium_term_target", row["Current Price"]),
                projection.get("long_term_target", row["Current Price"]),
            ]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=points, y=highs, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
            fig.add_trace(
                go.Scatter(
                    x=points,
                    y=lows,
                    mode="lines",
                    fill="tonexty",
                    name="Confidence band",
                    fillcolor="rgba(0, 123, 255, 0.15)",
                    line=dict(width=0),
                )
            )
            fig.add_trace(go.Scatter(x=points, y=targets, mode="lines+markers", name="Projection targets", line=dict(color="#1f77b4")))
            fig.add_hline(y=row["Current Price"], line_dash="dash", line_color="black", annotation_text="Current price")
            fig.add_hline(y=row["Avg Cost"], line_dash="dot", line_color="gray", annotation_text="Avg cost / break-even")
            fig.add_hline(y=projection.get("short_term_target", row["Current Price"]), line_dash="dot", line_color="orange")
            fig.add_hline(y=projection.get("medium_term_target", row["Current Price"]), line_dash="dot", line_color="blue")
            fig.add_hline(y=projection.get("long_term_target", row["Current Price"]), line_dash="dot", line_color="green")
            fig.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20), showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

        st.plotly_chart(px.pie(result_df, values="Current Value", names="Ticker", title="Portfolio Allocation"), use_container_width=True)
        st.plotly_chart(px.bar(result_df, x="Ticker", y="P&L $", color="P&L $", title="P&L by Holding"), use_container_width=True)
