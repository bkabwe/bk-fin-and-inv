from __future__ import annotations

import threading
import time

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.polygon_client import is_polygon_configured
from modules.portfolio import get_portfolio
from modules.prediction_tracker import record_predictions_from_scan
from modules.profit_opportunities import (
    DEFAULT_FAST_SCREEN_MARGIN,
    HORIZON_LABEL_TO_KEY,
    add_estimated_dates,
    collect_universe_tickers,
    filter_by_upside,
    scan_profit_opportunities,
)

st.title("💰 Profit Opportunities")
st.caption("Identifies stocks with the highest projected profit potential based on multi-model predictive analysis.")
if not is_polygon_configured():
    st.warning("POLYGON_API_KEY is not configured. Configure it to fetch market and fundamental data.")

horizon = st.radio(
    "Time Horizon",
    ["Short-Term (1–4 weeks)", "Medium-Term (1–6 months)", "Long-Term (6–24 months)"],
)
horizon_key = HORIZON_LABEL_TO_KEY[horizon]

selected_universes = st.multiselect(
    "Universe",
    ["S&P 500", "NASDAQ", "NYSE American", "OTC", "My Portfolio"],
    default=["S&P 500", "NASDAQ"],
)
min_upside_pct = st.slider("Min Upside %", min_value=5, max_value=100, value=15)

# ---------------------------------------------------------------------------
# Scan-mode selector
# ---------------------------------------------------------------------------
scan_mode = st.radio(
    "Scan Mode",
    ["Fast (Recommended)", "Thorough (Original, Slower)"],
    help=(
        "**Fast**: Parallel thread-pool execution + optional fast-screen pre-filter "
        "(technical subscore proxy). Significantly quicker on large universes. "
        "May very rarely miss a borderline candidate near the upside threshold.\n\n"
        "**Thorough**: Original sequential behaviour — every ticker gets full "
        "ARIMA/trend/GARCH analysis. Slower but no pre-filtering shortcuts."
    ),
)
_fast_mode = scan_mode.startswith("Fast")

use_fast_screen = False
if _fast_mode:
    use_fast_screen = st.checkbox(
        "Enable fast-screen pre-filter",
        value=True,
        help=(
            "Before running the expensive full analysis, compute a cheap "
            "technical subscore. Tickers whose subscore falls well below the "
            "proxy threshold are skipped (saves significant time on large "
            "universes). Disable for exhaustive coverage."
        ),
    )

st.info(
    "ℹ️ This page analyses **every stock** in your selected universes. "
    "Larger universes will take longer to scan."
    + (" Fast mode uses parallel processing to reduce runtime." if _fast_mode else "")
)
large = [u for u in selected_universes if u in ("NASDAQ", "OTC")]
if large:
    if _fast_mode:
        st.warning(
            f"⚠️ {' and '.join(large)} contain thousands of stocks. "
            "Fast mode (parallel + pre-filter) should reduce runtime significantly, "
            "but a full scan may still take several minutes."
        )
    else:
        st.warning(
            f"⚠️ {' and '.join(large)} contain thousands of stocks. "
            "A full scan in Thorough mode can take 30–60 minutes. "
            "Consider starting with S&P 500 or NYSE American."
        )


if st.button("Run Analysis"):
    portfolio_tickers = [str(h.get("ticker", "")).strip().upper() for h in (get_portfolio() or []) if h.get("ticker")]
    if "My Portfolio" in selected_universes and not portfolio_tickers:
        st.info("No portfolio holdings found.")
    tickers, source_labels = collect_universe_tickers(selected_universes, portfolio_tickers=portfolio_tickers)

    if not tickers:
        st.warning("No tickers available for analysis.")
    else:
        progress = st.progress(0.0)

        # IMPORTANT: Streamlit widgets (including st.progress) are only
        # supported when called from the main script-execution thread, so
        # scan_profit_opportunities() (which may run its per-ticker work on a
        # ThreadPoolExecutor) must never touch st.* from its callbacks. The
        # on_ticker_processed callback below only updates a plain dict; the
        # actual progress.progress(...) calls happen here, on the main
        # thread, via a polling loop around the (blocking) scan call.
        _progress_state = {"screened": 0, "total": len(tickers)}
        _state_lock = threading.Lock()

        def _on_ticker_processed(ticker: str, counts: dict) -> None:
            with _state_lock:
                _progress_state.update(counts)

        _scan_result: dict[str, pd.DataFrame] = {}

        def _run_scan() -> None:
            _scan_result["df"] = scan_profit_opportunities(
                tickers,
                horizon_key,
                use_fast_screen=use_fast_screen,
                fast_screen_margin=DEFAULT_FAST_SCREEN_MARGIN,
                parallel=_fast_mode,
                source_labels=source_labels,
                on_ticker_processed=_on_ticker_processed,
            )

        worker = threading.Thread(target=_run_scan, daemon=True)
        worker.start()
        while worker.is_alive():
            with _state_lock:
                screened = _progress_state["screened"]
                total = _progress_state["total"] or 1
            progress.progress(min(1.0, screened / total))
            time.sleep(0.2)
        worker.join()
        progress.progress(1.0)

        scanned_df = _scan_result.get("df", pd.DataFrame())
        st.session_state["profit_opportunities_scan"] = scanned_df
        st.session_state["profit_opportunities_scan_horizon"] = horizon_key
        st.session_state["profit_opportunities_scanned_total"] = len(tickers)
        if _fast_mode:
            st.session_state["profit_opportunities_fast_filtered"] = int(scanned_df.attrs.get("fast_filtered_count", 0))
            st.session_state["profit_opportunities_fully_analyzed"] = int(scanned_df.attrs.get("fully_analyzed_count", 0))
            st.session_state["profit_opportunities_failed"] = scanned_df.attrs.get("failed_tickers", [])
        else:
            # Clear fast-mode stats when running thorough
            st.session_state.pop("profit_opportunities_fast_filtered", None)
            st.session_state.pop("profit_opportunities_fully_analyzed", None)
            st.session_state.pop("profit_opportunities_failed", None)

scanned_df = st.session_state.get("profit_opportunities_scan")
scan_horizon_key = st.session_state.get("profit_opportunities_scan_horizon")
total_scanned = st.session_state.get("profit_opportunities_scanned_total", 0)

_DISPLAY_COLUMNS = [
    "Ticker",
    "Company",
    "Index",
    "Score",
    "Sector Trend",
    "Market Cap Tier",
    "Long-Term Stage",
    "Current Price",
    "Target Price",
    "Projected Upside %",
    "Est. Target Date",
    "Confidence",
    "Basis",
]

if scanned_df is not None and not scanned_df.empty:
    if scan_horizon_key != horizon_key:
        st.info("Horizon changed since the last scan — click **Run Analysis** to rescan for the new horizon.")
    else:
        max_results = st.slider("Max Results to Display", 5, 200, 25)

        # filter_by_upside() applies the min-upside threshold at *display*
        # time (not scan time) so users can adjust the slider without
        # triggering a full re-scan.
        qualifying = filter_by_upside(scanned_df, horizon_key, min_upside_pct=min_upside_pct)
        top_rows = qualifying.head(max_results)
        display_rows = add_estimated_dates(top_rows.to_dict("records"), horizon_key) if not top_rows.empty else []
        for row in display_rows:
            row.pop("_rsi", None)

        # Record predictions for the track-record feature.
        if display_rows:
            try:
                _recorded = record_predictions_from_scan(display_rows, horizon=horizon_key, source="profit_opportunities")
                if _recorded:
                    st.toast(f"📌 Recorded {_recorded} new prediction(s) for tracking.", icon="📌")
            except Exception:
                pass  # Never block UI for tracking failures

        if not display_rows:
            st.warning("No stocks met the minimum upside threshold. Try lowering the Min Upside %.")
            if total_scanned:
                st.caption(f"✅ Scanned **{total_scanned}** stocks → **0** met the upside threshold")
        else:
            results_df = pd.DataFrame(display_rows)
            results_df = results_df[[c for c in _DISPLAY_COLUMNS if c in results_df.columns]]
            _fast_filtered = st.session_state.get("profit_opportunities_fast_filtered")
            _fully_analyzed = st.session_state.get("profit_opportunities_fully_analyzed")
            if total_scanned:
                caption_parts = [f"✅ Scanned **{total_scanned}** stocks"]
                if _fast_filtered is not None and _fully_analyzed is not None:
                    caption_parts.append(
                        f"fast-filtered **{_fast_filtered}** · fully analysed **{_fully_analyzed}**"
                    )
                caption_parts.append(
                    f"**{len(qualifying)}** met the upside threshold → Showing top **{len(results_df)}**"
                )
                st.caption(" → ".join(caption_parts))
            formatters = {
                "Current Price": "${:,.2f}",
                "Target Price": "${:,.2f}",
                "Projected Upside %": "{:.2f}%",
            }
            styled = results_df.style.format({k: v for k, v in formatters.items() if k in results_df.columns})
            st.dataframe(styled, use_container_width=True)

            chart_df = results_df[["Ticker", "Projected Upside %", "Confidence"]].copy()
            fig = px.bar(
                chart_df,
                x="Ticker",
                y="Projected Upside %",
                color="Confidence",
                color_discrete_map={"High": "#2ca02c", "Medium": "#ffbf00", "Low": "#ff8c00"},
                title=f"Projected Upside by Stock ({horizon})",
            )
            st.plotly_chart(fig, use_container_width=True)

            top = results_df.iloc[0]
            st.success(
                f"🏆 Top Pick: **{top['Ticker']}** ({top['Company']}) — "
                f"Projected {top['Projected Upside %']:.1f}% upside by {top['Est. Target Date']} "
                f"(Confidence: {top['Confidence']})"
            )

            st.download_button(
                "Export to CSV",
                results_df.to_csv(index=False),
                file_name="profit_opportunities.csv",
                mime="text/csv",
            )

st.caption("⚠️ Projected dates and upside figures are model estimates based on historical data. They are not guarantees of future performance.")
