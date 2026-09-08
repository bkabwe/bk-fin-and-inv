# 📈 BK Stock Market Analyzer

Comprehensive US stock analysis toolkit inspired by Richard W. Schabacker's technical analysis principles.

## Installation

```bash
pip install -r requirements.txt
```

### macOS note for LightGBM

On macOS, `lightgbm` also requires the OpenMP runtime (`libomp`) at the
system level. If `import lightgbm` fails with a missing `libomp.dylib`
error, install it with Homebrew:

```bash
brew install libomp
```

## Optional environment variables

- `POLYGON_API_KEY` — required for Polygon.io (Massive) market data, ticker
  reference, indicators, splits/dividends, and news endpoints used across the app.
- `FRED_API_KEY` — optional FRED (Federal Reserve Economic Data) API key used to
  fetch macro series (`VIXCLS`, `DGS10`, `CPIAUCSL`, `FEDFUNDS`) used for regime
  context and feature engineering, replacing Polygon's separate paid Indices add-on
  requirement for `I:VIX`. Get a free key at
  https://fred.stlouisfed.org/docs/api/api_key.html
- `SEC_EDGAR_CONTACT_EMAIL` — optional contact email embedded in SEC EDGAR
  `User-Agent` headers. If unset, the app uses a placeholder and logs a warning;
  setting a real contact is recommended by SEC API guidance.

You can provide it either as a normal shell environment variable:

```bash
export POLYGON_API_KEY=your_polygon_api_key
export FRED_API_KEY=your_fred_api_key
```

Or by creating a `.env` file in the project root (automatically loaded at startup via `python-dotenv`):

```dotenv
POLYGON_API_KEY=your_polygon_api_key
FRED_API_KEY=your_fred_api_key
SEC_EDGAR_CONTACT_EMAIL=you@example.com
```

> Starter-plan behavior used by this app: unlimited API calls, up to 5 years of
> historical lookback for aggregates, and "current price" sourced from
> previous-day close (`/v2/aggs/ticker/{ticker}/prev`) because live snapshots
> are not included on Starter.

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
- Multi-model projected price targets (ARIMA + log-linear/log-polynomial trend + LightGBM return models for 30d/180d when saved models exist + fundamental fair value + DCF + technical resistance + analyst target, with GARCH confidence bounds)
- Prophet was removed from the ensemble after walk-forward backtests consistently showed materially higher RMSE on stock series (which generally lack the strong recurring seasonality Prophet is designed for).

## Data Sources

- S&P 500 list: Wikipedia (index-constituent source)
- NASDAQ / NYSE American / OTC active universes: Polygon `/v3/reference/tickers`
- Market OHLCV data: Polygon aggregates `/v2/aggs/...` with `adjusted=true`
- Current price proxy: Polygon previous-day close `/v2/aggs/ticker/{ticker}/prev`
- VIX / macro regime volatility context: FRED `VIXCLS` daily observations API
- Additional macro feature series: FRED `DGS10` (10Y Treasury), `CPIAUCSL` (CPI),
  and `FEDFUNDS` (Fed Funds Rate), including 5-day/30-day deltas and percent changes
- Reference/profile fields (name/sector/market-cap): Polygon ticker overview
- Fundamentals/ratios (P/E, EPS, ROE, debt-to-equity, growth): SEC EDGAR
  Company Facts XBRL API (10-K/10-Q filing data)
- Splits/dividends: Polygon `/v3/reference/splits` and `/v3/reference/dividends`
- News: Polygon `/v2/reference/news`

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

## Scoring methodology

The composite stock score remains a 0–100 scale built from technicals, fundamentals,
sentiment, and macro context, with all final scores clamped to that range.

- **Technical score** contributes up to 50 points from trend, momentum, volume,
  patterns, breakout quality, and relative strength.
- **Fundamental score** contributes up to 30 points after valuation, growth,
  balance-sheet, and analyst-sentiment checks.
- **Sentiment score** contributes up to 20 points from recent news tone.
- **Overall macro regime** adds a modest overlay (`risk_on` / `risk_off`) of
  roughly +3 / -5 points.
- **Sector momentum** now adds a small stock-specific overlay of **+3** when the
  stock's sector is currently in the macro model's bullish ETF basket and **-3**
  when that sector is in the bearish basket.
- **Market-cap risk tier** now classifies stocks as **Micro Cap** (<$300M),
  **Small Cap** ($300M to <$2B), **Mid Cap** ($2B to <$10B), **Large Cap**
  ($10B to <$200B), or **Mega Cap** (≥$200B). Micro/small caps receive only a modest risk
  adjustment (up to **-5** points) and can use wider projection confidence bands,
  reflecting higher volatility without overriding the core valuation logic.
- **Long-Term-only technical overlay** adds up to 20 points (`longterm_technical_score`)
  from 5-year signals: secular trend (SMA150/SMA200), Weinstein stage analysis,
  volume confirmation, drawdown/recovery resilience, and golden/death-cross history.


### Long-Term methodology details

When `time_horizon` is **Long-Term Hold**, analysis now adds dedicated 5-year
signals and exposes:

- `score_breakdown.longterm_technical`
- `longterm_stage` (Stage 1/2/3/4 classification)
- supporting durability signals (`primary_trend`, volume confirmation,
  drawdown/recovery stats, golden/death-cross history)

Short-Term and Medium-Term scoring logic/weights are unchanged.

## Troubleshooting

- **macOS SSL/cert issues**: run Python from an environment with updated certs and retry `pip install -r requirements.txt`.
- **POLYGON_API_KEY missing**: either export it in your shell or add it to a project-root `.env` file before launching Streamlit/API workers.
- **Polygon data unavailable for a ticker**: retry shortly; the app handles missing responses gracefully and skips unavailable symbols.
- **Fundamental metric timing**: SEC EDGAR fundamentals update on filing cadence
  (10-Q/10-K), not daily like market-price feeds.

## Disclaimer

This project is for educational/research use only and is **not financial advice**.

## Recent improvements

### Split-adjusted price data
All historical OHLCV data is now sourced from Polygon aggregates with
`adjusted=true`, so indicator, backtest, and forecast inputs remain continuous
across splits/reverse-splits. Polygon's aggregate-bar adjustment flag is
split-focused; dividend and other corporate-action reference data are fetched
from separate Polygon endpoints when needed.

Current-price display uses Polygon previous-day close (`/v2/aggs/.../prev`),
which matches Starter-plan availability (no live snapshot feed).

### TTL-aware caching in API (non-Streamlit) mode
When the app runs as a FastAPI/Celery backend, `st.cache_data` is unavailable.
Previously the fallback was a plain `functools.lru_cache` that ignored the
`ttl` argument, so cached data could go stale indefinitely.  The fallback is
now a lightweight TTL-aware decorator that actually expires entries after the
configured TTL (e.g. 1 hour for stock data, 24 hours for backtests).

### Shared feature-engineering pipeline (Phase 2 prep layer)
The app now includes `modules/feature_engineering.py`, which builds a daily-indexed
feature table per ticker as a data-preparation layer for a planned future
feature-based forecasting model (not yet wired into scoring/projections in this
phase). The table combines:
- Technical features (rolling volatility, EMA, RSI, daily-bar VWAP approximation,
  and volume-vs-average)
- SEC EDGAR fundamentals (`fundamental_revenue_growth`,
  `fundamental_debt_to_equity` (percent points, matching existing SEC adapter
  scaling), and `fundamental_gross_margin`), forward-filled from filing dates
  across daily rows
- Shared macro features from FRED (`DGS10`, `CPIAUCSL`, `FEDFUNDS`) with level and
  5-day/30-day delta and percent-change metrics

Design principle: feature assembly is TTL-cached incrementally and decoupled from
scan cadence so repeated scans avoid unnecessary recomputation/refetching.

### LightGBM return models (live for 30d/180d, gated off for 720d)
The app now includes `modules/lightgbm_model.py` as an explicit Phase 2 candidate
model module. It predicts **forward return** (not raw price) for the existing
30/180/720-day horizons using the shared engineered feature table and historical
close prices. Training/inference are intentionally decoupled via save/load helpers
so scan-time inference does not require retraining.

Live scoring now uses saved per-ticker LightGBM models in the ensemble for:
- **30d** with `LIGHTGBM_WEIGHT_30D = 0.15`
- **180d** with `LIGHTGBM_WEIGHT_180D = 0.08`

Both weights are intentionally below ARIMA/trend parity as a conservative rollout:
30d has the strongest validation evidence (89 tickers across two disjoint random
samples, 6 windows/ticker, 100% LightGBM win rate vs ARIMA/trend/naive), while
180d also won 100% of tickers but on only 30 tickers with 4 windows/ticker and
higher observed prediction variance. **720d remains intentionally excluded from the
live ensemble for now** because current validation has only one non-overlapping
window per ticker, which is promising but not yet actionable.

To train and persist the live 30d/180d models for a ticker before scoring:

```bash
python scripts/train_lightgbm_return_models.py AAPL
```

This writes per-ticker joblib artifacts under `data/lightgbm_return_models/<TICKER>/`.
If no saved model is available, live scoring falls back to the existing ARIMA/trend
ensemble behavior for that ticker/horizon.

### Walk-forward validation gate for LightGBM
`modules/backtester.run_walk_forward()` now supports optional LightGBM RMSE
evaluation alongside ARIMA and trend over the same rolling windows, for
validation/decision-gate analysis. The live ensemble now consumes that evidence
conservatively at 30d/180d only; 720d remains gated off until more natural history
accrues and yields more than one non-overlapping validation window per ticker.
Because LightGBM is trained as fixed-horizon return models (`30/180/720` days),
the walk-forward path maps each test window to the closest available horizon
(e.g., 30-day test windows use the 30-day model) and converts the predicted
return into a daily price path ending at that horizon for RMSE comparison.
An earlier validation bug made this path report all-zero LightGBM window counts:
macro-feature columns could remain object-typed when FRED data was unavailable,
causing every per-window LightGBM fit to fail under the old silent `except`.
Feature tables are now coerced back to numeric dtypes before training, and any
remaining per-window LightGBM failure is logged with ticker/window context.

To reproduce aggregate comparisons across a representative ticker sample:

```bash
python scripts/compare_lightgbm_backtest.py --sample-size 30
```

The script prints mean/median RMSE per model, LightGBM win percentage vs both
ARIMA and trend, and ticker/window evaluation counts per model.

### ARIMA convergence hardening
ARIMA fitting paths in both backtesting and scoring now use stronger convergence
settings: higher optimizer iteration budget, smarter initialization, and a
single fallback retry with an alternate optimizer when convergence warnings occur.
The ARIMA order-grid candidates and AIC selection logic are unchanged, preserving
comparability with prior walk-forward results while reducing non-convergence risk.
Those ARIMA paths now also attach an explicit supported business-day frequency to
trading-day price series before fitting/predicting, so statsmodels keeps using
calendar-aware forecast indexes instead of warning that the date index will be
ignored in future releases.

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
   ARIMA/trend/GARCH/backtest).  The subscore is normalized to 0–100.
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
(`max_workers=8` by default, tunable) since Polygon HTTP requests are I/O-bound.

### Profit Opportunities scan mode (Fast vs Thorough)
The Profit Opportunities page now offers a **Scan Mode** selector:

- **Fast (Recommended)** — runs per-ticker analysis in a parallel thread pool
  (`max_workers=8`) and optionally applies a cheap technical-subscore
  pre-filter before the expensive ARIMA/trend/GARCH full analysis.  The
  pre-filter uses the same safeguarded two-tier approach as the Screener
  (conservative threshold + 15-point safety margin) and is toggle-able via
  the "Enable fast-screen pre-filter" checkbox.  Significantly faster on large
  universes.  Transparency counts (fast-filtered / fully-analysed / failed)
  are shown in the results summary.

- **Thorough (Original, Slower)** — the original sequential loop: every ticker
  in the selected universe goes through full analysis with no pre-filtering.
  Use this when you want a guaranteed exhaustive baseline or are scanning a
  small universe.

The same `scan_mode`, `use_fast_screen`, and `fast_screen_margin` options are
available in the API/Celery path via `ProfitRequest` fields.

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
ARIMA/trend/GARCH-enhanced ensemble's accuracy.

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
