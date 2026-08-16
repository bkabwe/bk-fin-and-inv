# 📈 BK Stock Market Analyzer

Comprehensive US stock analysis toolkit inspired by Richard W. Schabacker's technical analysis principles.

## Installation

```bash
pip install -r requirements.txt
```

> Note: Installing `prophet` may take a few minutes on macOS because native dependencies are compiled during `pip install prophet`.

## Run the dashboard

```bash
streamlit run app.py
```

## Run the CLI analyzer

```bash
python analyze_stock.py AAPL
```

## Feature Overview

- Streamlit dashboard + multipage app
- Manual portfolio tracker with sell-signal checks
- Watchlist management
- Screener for S&P 500, NASDAQ 100, Russell 2000, OTC, and custom tickers
- Notification bell with unread count + mark-all-read
- Technical analysis (SMA/EMA, RSI, MACD, Stochastic, Bollinger, ATR, OBV, support/resistance, classic patterns)
- Unified score + recommendation engine with entry/target/stop-loss suggestions
- Multi-model projected price targets (Prophet/ARIMA/regression/fair-value/technical ensemble)

## Data Sources

- S&P 500 list: Wikipedia
- NASDAQ 100 list: stockanalysis.com
- Russell 2000 list: chartmill.com (`https://www.chartmill.com/stock/markets/usa/index/russell-2000`)
- OTC list: stockanalysis.com
- Market/news/quote data: yfinance

## Security Notes

- A startup disclaimer is shown once per Streamlit session.
- Ticker input is sanitized/validated across all app entry points.
- Portfolio/watchlist/notification JSON writes use atomic temp-file replacement.
- Personal financial JSON files are ignored by `.gitignore` and should never be committed.
- Streamlit production config is in `.streamlit/config.toml` (XSRF protection enabled).

## Score legend

- 80–100: 🟢 **STRONG BUY**
- 65–79: 🔵 **BUY**
- 50–64: 🟡 **TAKE SMALL POSITION**
- 35–49: 🟠 **MONITOR**
- 20–34: 🔴 **DO NOT BUY**
- 0–19: ⛔ **AVOID**

## Troubleshooting

- **macOS SSL/cert issues**: run Python from an environment with updated certs and retry `pip install -r requirements.txt`.
- **Prophet install fails**: try `pip install pystan==2.19.1.1` then `pip install prophet`.
- **yfinance rate limits (HTTP 429)**: the app retries with backoff for quote/info fetches; wait briefly and retry.

## Disclaimer

This project is for educational/research use only and is **not financial advice**.

## Recent improvements

### Split/dividend-adjusted price data
All historical price data fetched via `yfinance` now uses `auto_adjust=True`,
so OHLC series are continuously back-adjusted for stock splits, reverse splits,
and dividends.  This ensures that technical indicators, walk-forward backtests,
Prophet/ARIMA/GARCH forecasts, relative-strength calculations, and charts do
**not** show artificial price discontinuities caused by corporate actions.

The latest real-time quote used for "Current Price" display is sourced from
`yfinance`'s `info["currentPrice"]` field (unadjusted) and is clearly
documented as such in the code.

### TTL-aware caching in API (non-Streamlit) mode
When the app runs as a FastAPI/Celery backend, `st.cache_data` is unavailable.
Previously the fallback was a plain `functools.lru_cache` that ignored the
`ttl` argument, so cached data could go stale indefinitely.  The fallback is
now a lightweight TTL-aware decorator that actually expires entries after the
configured TTL (e.g. 1 hour for stock data, 24 hours for backtests).

### Failed-ticker visibility in screener results
Exceptions during per-ticker analysis are no longer silently swallowed.
Screener results (Streamlit and API) now track and surface:
- **Failed ticker count** — how many tickers could not be analyzed (and why).
- **Failed ticker list** — the ticker symbols and error reasons, visible in
  the API `ProgressResponse` and logged as warnings in Streamlit.

### Two-tier fast-screen pre-filter (screener performance)
The screener now uses a two-tier approach to dramatically cut runtime across
large universes (S&P 500, NASDAQ, Russell 2000) while keeping results accurate:

1. **Fast tier** — every ticker in the universe is evaluated with a cheap
   technical-subscore pass (data fetch + `analyze_technical` only, no
   Prophet/ARIMA/GARCH/backtest).  The subscore is normalized to 0–100.
2. **Full tier** — only tickers whose fast-tier score ≥ `min_score − margin`
   proceed to the expensive full analysis (forecasting + walk-forward backtest).

Key safeguards:
- Every ticker in the universe is still analyzed at least at the fast-tier
  level — none are silently excluded from consideration.
- The default safety margin is **15 points**, making the pre-filter
  deliberately conservative to minimize false negatives (candidates that
  would have qualified after full analysis but get cut early).
- The margin is configurable via the `fast_screen_margin` parameter.
- The pre-filter can be disabled entirely with `use_fast_screen=False`
  (useful for small custom universes where speed matters less).
- Screener results include transparency counts:
  - `fast_filtered_count` — tickers eliminated by the fast tier.
  - `fully_analyzed_count` — tickers that went through full analysis.

Additionally, per-ticker fetches are now parallelized with a thread pool
(`max_workers=8` by default, tunable) since yfinance calls are I/O-bound,
subject to conservative concurrency limits to avoid HTTP 429 rate-limiting.

### Portfolio split/reverse-split detection
The portfolio view now detects whether any held ticker has undergone a
split or reverse split since the recorded purchase date.  When a split is
detected, a visible warning is shown in the portfolio card:

> ⚠️ TICKER has undergone a Nx forward split since YYYY-MM-DD.
> Your share count and cost basis may be outdated — please update this holding.

Share count and cost basis are **not** auto-adjusted (to avoid silent
corruption of user data), but the warning makes it clear that manual
review is needed.

---

## Prediction Track Record

### Overview
Every price projection shown to the user is automatically persisted so that,
once the estimated target date has passed, the app can verify whether the
prediction actually hit — giving you an empirical track record of the
Prophet/ARIMA/GARCH/trend ensemble's accuracy.

### What gets tracked
Predictions are recorded from three sources:

| Source | Trigger |
|--------|---------|
| **Profit Opportunities** (Streamlit) | Whenever a scan produces qualifying rows |
| **Profit Opportunities** (API/Celery) | At the end of each background scan task |
| **Stock Analysis** | Each time a ticker is opened on the Deep Dive page |

Each prediction stores: ticker, company, horizon (`short_term` / `medium_term` /
`long_term`), scan date, estimated target date, current price at scan, target
price, target low/high band, projected upside %, score, confidence tier,
model basis string, and source.

Predictions are deduplicated by `(ticker, horizon, scan_date)` so re-running
the same scan on the same day does not create duplicate entries.

### How predictions resolve
When the **Track Record** page loads, the app automatically checks all
`pending` predictions whose `target_date` has already passed.  For each one it:

1. Fetches the split-adjusted historical closing price on/near the target date
   via `get_stock_data` (the same adjusted-price path used app-wide).
2. Computes `actual_return_pct` and marks the prediction as either:
   - **`resolved`** — actual price data was available.
   - **`unresolved_no_data`** — ticker was delisted or no price data found.
3. Records two hit definitions:
   - **Hit (strict)** — actual price ≥ projected target price.
   - **Hit (band)** — actual price fell within the projected low–high band.

No new infrastructure is required.  Resolution runs on-demand when you visit
the Track Record page; a Celery task is not needed.

### Track Record page (`7_Track_Record`)
The new **📈 Track Record** page (sidebar item 7) shows:

- **Summary metrics** — total predictions, resolved count, pending count,
  overall hit rate (strict and band definitions).
- **Mean / Median Absolute % Error** — average gap between projected upside
  and actual return.
- **Breakdown tables** — hit rate and MAE split by horizon, by source, and by
  confidence tier (calibration view).
- **Calibration chart** — bar chart of hit rate by confidence tier, making
  miscalibration visible at a glance.
- **Resolved predictions table** — filterable/sortable by horizon, source, and
  hit/miss result; exportable to CSV.
- **Projected vs Actual scatter** — visual comparison of projected upside vs
  actual return per resolved prediction.
- **Pending predictions** — collapsible list of predictions still awaiting
  their target date.

### API endpoints
Two read-only endpoints are available for programmatic access:

```
GET /api/track-record/summary
```
Returns the same summary stats (overall hit rates, MAE, breakdown by horizon/
source/confidence) as a JSON object.

```
GET /api/track-record/predictions?status=resolved
```
Returns the full predictions list, optionally filtered by `status`
(`pending`, `resolved`, `unresolved_no_data`).

### Storage
Predictions are persisted to `data/predictions.json` using the same
atomic-write pattern (tempfile + `os.replace`) used by the portfolio and
watchlist modules.  The file is excluded from git via `.gitignore`
(`data/*.json`).
