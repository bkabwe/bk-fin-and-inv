from __future__ import annotations

import streamlit as st

from modules.screener import run_screener

st.title("🔍 Stock Screener")

label = st.selectbox("Universe", ["S&P 500", "NASDAQ 100", "Russell 2000", "OTC", "Custom"])
map_universe = {"S&P 500": "sp500", "NASDAQ 100": "nasdaq100", "Russell 2000": "russell2000", "OTC": "otc", "Custom": "custom"}
custom = []
if label == "Custom":
    txt = st.text_area("Custom tickers (comma separated)", "AAPL,MSFT,NLST")
    custom = [x.strip().upper() for x in txt.split(",") if x.strip()]

min_score = st.slider("Min Score", 0, 100, 60)
max_results = st.slider("Max Results", 5, 100, 25)
batch_size = st.slider(
    "Max stocks to screen",
    min_value=10,
    max_value=500,
    value=50,
    help="Limit the number of stocks screened at once. Lower = faster. S&P 500 full scan = ~503 stocks.",
)
time_filter = st.multiselect("Time Horizon", ["Short-Term Opportunity", "Medium-Term Setup", "Long-Term Hold"])
sector_filter = st.text_input("Sector filter (optional)").strip().lower()

if st.button("Run Screener"):
    progress = st.progress(0)

    def _cb(v: float):
        progress.progress(min(1.0, max(0.0, v)))

    with st.spinner("Screening..."):
        results = run_screener(
            map_universe[label],
            min_score=min_score,
            max_results=max_results,
            batch_size=batch_size,
            custom_tickers=custom,
            progress_callback=_cb,
        )
    if results.empty:
        st.warning("No results.")
    else:
        if time_filter:
            results = results[results["Time Horizon"].isin(time_filter)]
        if sector_filter:
            results = results[results["Company"].fillna("").str.lower().str.contains(sector_filter)]
        st.dataframe(results, use_container_width=True)
        st.download_button("Export to CSV", results.to_csv(index=False), "screener_results.csv", "text/csv")

st.markdown(
    """
- 80–100: 🟢 STRONG BUY  
- 65–79: 🔵 BUY  
- 50–64: 🟡 TAKE SMALL POSITION  
- 35–49: 🟠 MONITOR  
- 20–34: 🔴 DO NOT BUY  
- 0–19: ⛔ AVOID
"""
)
