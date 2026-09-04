from __future__ import annotations

import concurrent.futures
import threading
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from modules.data_fetcher import get_nasdaq_tickers, get_nyseamerican_tickers, get_otc_tickers, get_sp500_tickers, get_stock_data
from modules.portfolio import get_portfolio
from modules.prediction_tracker import record_predictions_from_scan
from modules.scoring_engine import analyze_stock, fast_screen_score
from modules.tiingo_client import (
    TIINGO_REVALIDATION_DEFAULT_TOP_N,
    TIINGO_REVALIDATION_MAX_TOP_N,
    is_tiingo_configured,
    revalidate_profit_rows,
    summarize_revalidation,
)

try:  # pragma: no cover
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:  # pragma: no cover
    PROPHET_AVAILABLE = False

try:  # pragma: no cover
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:  # pragma: no cover
    ARIMA_AVAILABLE = False

try:  # pragma: no cover
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    SKLEARN_AVAILABLE = False


st.title("💰 Profit Opportunities")
st.caption("Identifies stocks with the highest projected profit potential based on multi-model predictive analysis.")

horizon = st.radio(
    "Time Horizon",
    ["Short-Term (1–4 weeks)", "Medium-Term (1–6 months)", "Long-Term (6–24 months)"],
)

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
        "Prophet/ARIMA/GARCH analysis. Slower but no pre-filtering shortcuts."
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

revalidate_with_tiingo = st.checkbox(
    "Revalidate Top Results with Tiingo (free tier)",
    value=False,
    help=(
        "After the normal yfinance scan/ranking finishes, re-run only the top ranked "
        "results against Tiingo's official API as a confirmation pass. Requires "
        "TIINGO_API_KEY and adds latency."
    ),
)
revalidate_top_n = TIINGO_REVALIDATION_DEFAULT_TOP_N
if revalidate_with_tiingo:
    revalidate_top_n = st.number_input(
        "Top results to revalidate",
        min_value=1,
        max_value=TIINGO_REVALIDATION_MAX_TOP_N,
        value=TIINGO_REVALIDATION_DEFAULT_TOP_N,
        step=1,
    )
    if not is_tiingo_configured():
        st.info("Set the TIINGO_API_KEY environment variable to enable Tiingo revalidation.")

# Shared constants matching modules/screener.py defaults
_MAX_WORKERS = 8
_FAST_SCREEN_MARGIN = 15
# Conservative proxy threshold: tickers whose fast technical subscore is below
# this value are very unlikely to produce a high projected upside, so we skip
# them.  We use a fixed low-bar proxy base (30 out of 100) with a safety
# margin rather than the upside % directly, because fast_screen_score() returns
# a normalized technical subscore (not an upside %) — a score this low makes
# any meaningful upside projection unlikely.
_FAST_SCREEN_BASE = 30  # base proxy cutoff (out of 100)
_FAST_SCREEN_PROXY_THRESHOLD = _FAST_SCREEN_BASE - _FAST_SCREEN_MARGIN  # = 15

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


@st.cache_data(ttl=1800)
def estimate_target_date(ticker: str, target_price: float, horizon_value: str, rsi: float | None = None) -> tuple[str, str]:
    """Returns (estimated_date_range, confidence) for when ticker may reach target_price."""

    def _format_date_range(center_date: date, confidence: str) -> str:
        windows = {
            "Short-Term (1–4 weeks)": {"High": 4, "Medium": 6, "Low": 8},
            "Medium-Term (1–6 months)": {"High": 14, "Medium": 21, "Low": 30},
            "Long-Term (6–24 months)": {"High": 30, "Medium": 45, "Low": 60},
        }
        span = windows[horizon_value][confidence]
        start = center_date - timedelta(days=span)
        end = center_date + timedelta(days=span)
        return f"~{start.strftime('%b %d, %Y')} – {end.strftime('%b %d, %Y')}"

    today = date.today()
    if horizon_value == "Short-Term (1–4 weeks)":
        days = 21
        if rsi is not None:
            if rsi < 50:
                days = 28
            elif rsi > 60:
                days = 14
        est = today + timedelta(days=days)
        return _format_date_range(est, "Medium"), "Medium"

    fallback_days = 90 if horizon_value == "Medium-Term (1–6 months)" else 365
    max_days = 180 if horizon_value == "Medium-Term (1–6 months)" else 720

    try:
        data = get_stock_data(ticker, period="2y", interval="1d")
    except Exception:
        data = pd.DataFrame()

    if data.empty or "Close" not in data:
        est = today + timedelta(days=fallback_days)
        return _format_date_range(est, "Low"), "Low"

    close = data["Close"].dropna().astype(float)
    if close.empty:
        est = today + timedelta(days=fallback_days)
        return _format_date_range(est, "Low"), "Low"

    prophet_day: int | None = None
    arima_hit = False
    trend_day: int | None = None

    if PROPHET_AVAILABLE and len(close) >= 60:
        try:
            prophet_df = close.reset_index().rename(columns={close.index.name or "Date": "ds", "Close": "y"})
            if "ds" not in prophet_df.columns:
                prophet_df = prophet_df.rename(columns={prophet_df.columns[0]: "ds"})
            prophet_df["ds"] = pd.to_datetime(prophet_df["ds"]).dt.tz_localize(None)
            prophet_df = prophet_df[["ds", "y"]]
            model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
            model.fit(prophet_df)
            forecast = model.predict(model.make_future_dataframe(periods=max_days, freq="D"))
            future_only = forecast.tail(max_days)
            crossing = future_only[future_only["yhat"] >= float(target_price)]
            if not crossing.empty:
                cross_ds = pd.to_datetime(crossing.iloc[0]["ds"]).date()
                prophet_day = max(1, (cross_ds - today).days)
        except Exception:
            prophet_day = None

    if ARIMA_AVAILABLE and len(close) >= 60:
        try:
            arima_forecast = ARIMA(close.tail(252), order=(5, 1, 0)).fit().forecast(steps=max_days)
            arima_hit = bool((arima_forecast >= float(target_price)).any())
        except Exception:
            arima_hit = False

    if len(close) >= 30:
        try:
            series = close.tail(200).reset_index(drop=True)
            x = np.arange(len(series)).reshape(-1, 1)
            if SKLEARN_AVAILABLE:
                model = LinearRegression()
                model.fit(x, series.values)
                slope = float(model.coef_[0])
                intercept = float(model.intercept_)
            else:
                slope, intercept = np.polyfit(np.arange(len(series)), series.values, 1)
            if slope > 0:
                day_x = int(round((float(target_price) - intercept) / slope))
                if day_x > len(series):
                    trend_day = day_x - len(series)
        except Exception:
            trend_day = None

    if prophet_day is not None:
        est = today + timedelta(days=min(max_days, prophet_day))
        confidence = "High" if arima_hit else "Medium"
        return _format_date_range(est, confidence), confidence

    if trend_day is not None:
        est = today + timedelta(days=min(max_days, max(1, trend_day)))
        confidence = "Medium" if arima_hit else "Low"
        return _format_date_range(est, confidence), confidence

    est = today + timedelta(days=fallback_days)
    confidence = "Medium" if arima_hit else "Low"
    return _format_date_range(est, confidence), confidence


def _collect_tickers(universes: list[str]) -> tuple[list[str], dict[str, str]]:
    universe_fetchers = {
        "S&P 500": get_sp500_tickers,
        "NASDAQ": get_nasdaq_tickers,
        "NYSE American": get_nyseamerican_tickers,
        "OTC": get_otc_tickers,
    }

    source_map: dict[str, set[str]] = {}
    ordered: list[str] = []

    for universe_name in universes:
        if universe_name == "My Portfolio":
            holdings = get_portfolio() or []
            if not holdings:
                st.info("No portfolio holdings found.")
                continue
            values = [str(x.get("ticker", "")).strip().upper() for x in holdings if x.get("ticker")]
        else:
            fetcher = universe_fetchers[universe_name]
            try:
                values = fetcher()
            except Exception as exc:
                st.warning(f"Could not load {universe_name} universe: {exc}")
                continue

        for ticker in values:
            clean = str(ticker).strip().upper().replace(".", "-")
            if not clean:
                continue
            if clean not in source_map:
                source_map[clean] = set()
                ordered.append(clean)
            source_map[clean].add(universe_name)

    return ordered, {k: ", ".join(sorted(v)) for k, v in source_map.items() if k in ordered}


if st.button("Run Analysis"):
    tickers, source_labels = _collect_tickers(selected_universes)
    if not tickers:
        st.warning("No tickers available for analysis.")
    else:
        progress = st.progress(0.0)
        rows: list[dict] = []

        if _fast_mode:
            # ------------------------------------------------------------------
            # FAST MODE: parallel thread pool + optional fast-screen pre-filter
            # ------------------------------------------------------------------
            # IMPORTANT: Streamlit widgets (including st.progress) are only
            # supported when called from the main script-execution thread.
            # Calling progress.progress(...) from a ThreadPoolExecutor worker
            # thread either silently no-ops or fails to trigger a UI re-render
            # because Streamlit's ScriptRunContext is not attached to worker
            # threads. To keep the progress bar working correctly, worker
            # threads below ONLY update shared, thread-safe counters — the
            # actual `progress.progress(...)` call happens on the main thread,
            # inside the `as_completed` loop, once per completed future.
            failed_tickers: list[dict] = []
            # NOTE: This page executes as top-level Streamlit script code (not
            # inside a function), so `_process_ticker` below has no enclosing
            # *function* scope — only module/global scope. `nonlocal` requires
            # an enclosing function scope and raises SyntaxError here
            # ("no binding for nonlocal ... found"). Use a mutable dict instead
            # so counters can be updated in place without `nonlocal`/`global`.
            _counters = {"fast_filtered": 0, "fully_analyzed": 0, "processed": 0}
            _lock = threading.Lock()
            total = len(tickers)

            def _process_ticker(ticker: str) -> None:
                try:
                    if use_fast_screen:
                        fast_score, err = fast_screen_score(ticker)
                        if err is not None:
                            with _lock:
                                failed_tickers.append({"ticker": ticker, "reason": f"fast-screen error: {err}"})
                            return
                        # Conservative proxy: skip tickers whose cheap technical
                        # subscore is well below a low threshold, since a low
                        # technical subscore makes any meaningful projected upside
                        # very unlikely.  The margin keeps borderline tickers safe.
                        if fast_score < _FAST_SCREEN_PROXY_THRESHOLD:
                            with _lock:
                                _counters["fast_filtered"] += 1
                            return

                    analysis = analyze_stock(ticker)
                    with _lock:
                        _counters["fully_analyzed"] += 1
                    projections = analysis.get("projections") or {}
                    current_price = float(projections.get("current_price") or analysis.get("current_price") or 0)
                    if current_price <= 0:
                        return
                    with _lock:
                        rows.append(
                            {
                                "ticker": ticker,
                                "company": analysis.get("company") or ticker,
                                "index": source_labels.get(ticker, ""),
                                "score": int(analysis.get("score") or 0),
                                "current_price": current_price,
                                "technical": analysis.get("technical") or {},
                                "projections": projections,
                            }
                        )
                except Exception as exc:
                    with _lock:
                        failed_tickers.append({"ticker": ticker, "reason": str(exc) or type(exc).__name__})
                finally:
                    # NOTE: Do NOT call progress.progress(...) here — this runs
                    # on a worker thread. Only update the shared counter; the
                    # main thread reads `processed` and updates the widget.
                    with _lock:
                        _counters["processed"] += 1

            with concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
                futures = {executor.submit(_process_ticker, t): t for t in tickers}
                # This loop runs on the main script thread, so it's the correct
                # place to update Streamlit widgets as each future resolves.
                for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                    try:
                        future.result()
                    except Exception:
                        pass
                    progress.progress(completed / total)

            st.session_state["profit_opportunities_fast_filtered"] = _counters["fast_filtered"]
            st.session_state["profit_opportunities_fully_analyzed"] = _counters["fully_analyzed"]
            st.session_state["profit_opportunities_failed"] = failed_tickers

        else:
            # ------------------------------------------------------------------
            # THOROUGH MODE: original sequential loop, no pre-filter (unchanged)
            # ------------------------------------------------------------------
            for i, ticker in enumerate(tickers, start=1):
                try:
                    analysis = analyze_stock(ticker)
                    projections = analysis.get("projections") or {}
                    current_price = float(projections.get("current_price") or analysis.get("current_price") or 0)
                    if current_price <= 0:
                        continue
                    rows.append(
                        {
                            "ticker": ticker,
                            "company": analysis.get("company") or ticker,
                            "index": source_labels.get(ticker, ""),
                            "score": int(analysis.get("score") or 0),
                            "current_price": current_price,
                            "technical": analysis.get("technical") or {},
                            "projections": projections,
                        }
                    )
                except Exception:
                    continue
                progress.progress(i / len(tickers))
            # Clear fast-mode stats when running thorough
            st.session_state.pop("profit_opportunities_fast_filtered", None)
            st.session_state.pop("profit_opportunities_fully_analyzed", None)
            st.session_state.pop("profit_opportunities_failed", None)

        st.session_state["profit_opportunities_scan"] = rows
        st.session_state["profit_opportunities_scanned_total"] = len(tickers)

scan_rows = st.session_state.get("profit_opportunities_scan", [])
total_scanned = st.session_state.get("profit_opportunities_scanned_total", 0)

if scan_rows:
    max_results = st.slider("Max Results to Display", 5, 200, 25)
    horizon_map = {
        "Short-Term (1–4 weeks)": ("short_term_target", "short_term_upside", "short_term_basis"),
        "Medium-Term (1–6 months)": ("medium_term_target", "medium_term_upside", "medium_term_basis"),
        "Long-Term (6–24 months)": ("long_term_target", "long_term_upside", "long_term_basis"),
    }
    target_key, upside_key, basis_key = horizon_map[horizon]

    display_rows = []
    for row in scan_rows:
        projections = row.get("projections") or {}
        target_price = float(projections.get(target_key) or 0)
        upside = float(projections.get(upside_key) or 0)
        if target_price <= 0 or upside < min_upside_pct or upside <= 0:
            continue
        rsi = row.get("technical", {}).get("indicators", {}).get("rsi")
        est_date, confidence = estimate_target_date(row["ticker"], target_price, horizon, rsi)
        display_rows.append(
            {
                "Ticker": row["ticker"],
                "Company": row["company"],
                "Index": row["index"],
                "Score": row["score"],
                "Current Price": row["current_price"],
                "Target Price": target_price,
                "Projected Upside %": upside,
                "Est. Target Date": est_date,
                "Confidence": confidence,
                "Basis": projections.get(basis_key, ""),
            }
        )

    # Record predictions for the track-record feature.
    if display_rows:
        horizon_key_map = {
            "Short-Term (1–4 weeks)": "short_term",
            "Medium-Term (1–6 months)": "medium_term",
            "Long-Term (6–24 months)": "long_term",
        }
        _h = horizon_key_map.get(horizon, "short_term")
        try:
            _recorded = record_predictions_from_scan(display_rows, horizon=_h, source="profit_opportunities")
            if _recorded:
                st.toast(f"📌 Recorded {_recorded} new prediction(s) for tracking.", icon="📌")
        except Exception:
            pass  # Never block UI for tracking failures

    if not display_rows:
        st.warning("No stocks met the minimum upside threshold. Try lowering the Min Upside %.")
        if total_scanned:
            st.caption(f"✅ Scanned **{total_scanned}** stocks → **0** met the upside threshold")
    else:
        ranked_rows = pd.DataFrame(display_rows).sort_values("Projected Upside %", ascending=False).reset_index(drop=True)
        if revalidate_with_tiingo:
            horizon_key_map = {
                "Short-Term (1–4 weeks)": "short_term",
                "Medium-Term (1–6 months)": "medium_term",
                "Long-Term (6–24 months)": "long_term",
            }
            ranked_rows = pd.DataFrame(
                revalidate_profit_rows(
                    ranked_rows.to_dict(orient="records"),
                    horizon_key=horizon_key_map.get(horizon, "short_term"),
                    top_n=int(revalidate_top_n),
                )
            )
        results_df = ranked_rows.head(max_results)
        revalidation_summary = summarize_revalidation(ranked_rows.to_dict(orient="records"))
        _fast_filtered = st.session_state.get("profit_opportunities_fast_filtered")
        _fully_analyzed = st.session_state.get("profit_opportunities_fully_analyzed")
        if total_scanned:
            caption_parts = [f"✅ Scanned **{total_scanned}** stocks"]
            if _fast_filtered is not None and _fully_analyzed is not None:
                caption_parts.append(
                    f"fast-filtered **{_fast_filtered}** · fully analysed **{_fully_analyzed}**"
                )
            caption_parts.append(
                f"**{len(display_rows)}** met the upside threshold → Showing top **{len(results_df)}**"
            )
            st.caption(" → ".join(caption_parts))
        if revalidate_with_tiingo:
            if revalidation_summary["status"] == "unavailable" and not is_tiingo_configured():
                st.warning("Tiingo revalidation was requested but TIINGO_API_KEY was not configured.")
            elif revalidation_summary["status"] == "unavailable":
                st.warning("Tiingo revalidation was requested, but no rows could be verified.")
            else:
                st.caption(
                    f"Tiingo verified **{int(revalidation_summary['verified'])}** ranked result(s)"
                    + (
                        f"; **{int(revalidation_summary['unavailable'])}** were unavailable."
                        if int(revalidation_summary["unavailable"])
                        else "."
                    )
                )

        styled = results_df.style.format(
            {
                "Current Price": "${:,.2f}",
                "Target Price": "${:,.2f}",
                "Projected Upside %": "{:.2f}%",
                "Verified Current Price": "${:,.2f}",
                "Verified Target Price": "${:,.2f}",
                "Verified Projected Upside %": "{:.2f}%",
                "Price Difference %": "{:.2f}%",
                "Target Difference %": "{:.2f}%",
            }
        )
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
