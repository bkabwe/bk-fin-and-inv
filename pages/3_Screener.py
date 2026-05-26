from __future__ import annotations

import streamlit as st

from modules.screener import run_screener
from modules.validators import sanitize_ticker_list

st.title("🔍 Stock Screener")

label = st.selectbox("Universe", ["S&P 500", "NASDAQ", "NYSE American", "OTC", "Custom"])
map_universe = {
    "S&P 500": "sp500",
    "NASDAQ": "nasdaq",
    "NYSE American": "nyseamerican",
    "OTC": "otc",
    "Custom": "custom",
}
custom = []
if label == "Custom":
    txt = st.text_area("Custom tickers (comma separated)", "AAPL,MSFT,NLST")
    custom = sanitize_ticker_list([x.strip() for x in txt.split(",") if x.strip()])

min_score = st.slider("Min Score", 0, 100, 60)
max_results = st.slider("Max Results to Display", 5, 500, 25)
st.info(
    "ℹ️ The screener analyses **every stock** in the selected universe, then displays "
    "the top results ranked by score. Larger universes (e.g. NASDAQ with 3,000+ stocks or OTC) "
    "will take longer to complete."
)
if map_universe[label] in ("nasdaq", "otc"):
    st.warning(
        f"⚠️ {label} contains thousands of stocks. A full scan may take a long time. "
        "Consider using a higher Min Score (e.g. 70+) to focus on the best candidates."
    )
time_filter = st.multiselect("Time Horizon", ["Short-Term Opportunity", "Medium-Term Setup", "Long-Term Hold"])
sector_filter = st.text_input("Sector filter (optional)").strip().lower()

if st.button("Run Screener"):
    progress = st.progress(0)

    def _cb(v: float):
        progress.progress(min(1.0, max(0.0, v)))

    try:
        with st.spinner("Screening..."):
            results = run_screener(
                map_universe[label],
                min_score=min_score,
                max_results=max_results,
                custom_tickers=custom,
                progress_callback=_cb,
            )
    except RuntimeError as e:
        st.error(str(e))
        st.stop()
    if results.attrs.get("fallback_used"):
        count = results.attrs.get("source_ticker_count", 0)
        st.warning(
            f"⚠️ Could not fetch live ticker list. Results based on fallback list of {count} stocks. "
            "Check your internet connection."
        )
    if map_universe[label] in ("nasdaq", "otc", "nyseamerican"):
        count = int(results.attrs.get("source_ticker_count", 0))
        if count and count < 50:
            st.warning(f"⚠️ {label} scrape returned only {count} tickers. Results may be incomplete.")
    if results.empty:
        st.warning("No results.")
    else:
        if time_filter:
            results = results[results["Time Horizon"].isin(time_filter)]
        if sector_filter:
            results = results[results["Company"].fillna("").str.lower().str.contains(sector_filter)]
        st.dataframe(results, use_container_width=True)
        total_scanned = results.attrs.get("source_ticker_count", 0)
        total_qualified = results.attrs.get("qualified_count", len(results))
        st.caption(
            f"✅ Scanned **{total_scanned}** stocks → "
            f"**{total_qualified}** scored ≥{min_score} → "
            f"Showing top **{len(results)}** by score"
        )
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
