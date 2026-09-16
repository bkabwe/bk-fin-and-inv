# 📈 BK Stock Market Analyzer

Comprehensive US stock analysis toolkit inspired by Richard W. Schabacker's technical analysis principles.

## Installation

```bash
pip install -r requirements.txt
```

`requirements.txt` is the full set needed to run the Streamlit dashboard
locally (adds `streamlit`, `plotly`, `Pillow` for the UI on top of the core
analysis stack). The scheduled GitHub Actions workflows (scan/grading email
reports, LightGBM batch training) never render the Streamlit UI, so they
install the leaner `requirements-workflows.txt` instead — see
`.github/workflows/*.yml`. The FastAPI/Celery backend under `api/` uses
`requirements-api.txt` in addition to one of the above (see
[Optional FastAPI + React frontend](#optional-fastapi--react-frontend)).

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
- `BREVO_API_KEY` — required for the scheduled scan/grading workflows that send
  transactional HTML email reports through Brevo's HTTPS API.
- `SCAN_EMAIL_RECIPIENTS` — comma-separated email recipients for scheduled scan
  and grading reports. Whitespace around addresses is ignored.
- `SCAN_EMAIL_FROM` — verified Brevo sender email address used with the fixed
  sender display name `BK Self`.

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
BREVO_API_KEY=your_brevo_api_key
SCAN_EMAIL_RECIPIENTS=you@example.com,second@example.com
SCAN_EMAIL_FROM=you@example.com
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

## Optional FastAPI + React frontend

Everything in this app is available through the Streamlit dashboard above.
A second, optional interface exists under `api/` (FastAPI + Celery/Redis job
queue) and `frontend/` (React + Vite SPA) for anyone who wants a
non-blocking, async version of the Profit Opportunities scan (with a stop
button) instead of Streamlit's blocking scan. It duplicates — rather than
replaces — the Streamlit feature set and is not used by any of the scheduled
GitHub Actions workflows, which call `scripts/*.py` against `modules/`
directly.

Requirements: `pip install -r requirements-workflows.txt -r requirements-api.txt`
(the API/Celery layer doesn't render the Streamlit UI, so it doesn't need
`requirements.txt`'s `streamlit`/`plotly`/`Pillow`) and a local Redis
instance (`redis://localhost:6379/0` by default; see `api/worker.py`).

```bash
./start-api.sh
```

This starts Redis (via `brew services`, macOS-only), the Celery worker
(`celery -A api.worker worker`), the FastAPI backend (`uvicorn api.main:app`
on port 8000, docs at `/docs`), and the Vite dev server (`npm run dev` under
`frontend/`, port 5173) together. Start services individually if you're not
on macOS or don't use Homebrew's `redis` service.

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
- **Fundamental score** (valuation, growth, balance-sheet, and analyst-sentiment
  checks) and **Sentiment score** (recent news tone) together contribute a
  combined 50 points, split 30/20 by default. When callers pass
  `investment_horizon` (`short_term`/`medium_term`/`long_term`, the same
  canonical keys used by the Profit Opportunities scan), sentiment's share
  decays for longer horizons and fundamentals pick up the difference: 30/20
  (short_term, default), 36/14 (medium_term), 42/8 (long_term). This mirrors
  how price-projection ensemble weights already shrink LightGBM's influence at
  longer horizons. Callers that don't pass a horizon keep the original flat
  30/20 split. Sentiment headline scoring uses FinBERT (`ProsusAI/finbert`)
  via the `transformers` + `torch` dependencies.
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

## Known modeling limitations & caveats

These are documented, audited findings that are intentionally **not** being
force-fixed without stronger evidence or a clearer cost/benefit case. They are
tracked here so future changes can address them with real data/evidence
rather than guesswork:

- **Correlated technical sub-scores**: `trend_score`, `momentum_score`
  (RSI+MACD), `rs_score`, `breakout_score`, and `volume_quality_score` largely
  fire off the same underlying "uptrend + volume" signal, potentially
  over-crediting a single pattern several times within the 0–50 technical
  total. Confirming and correcting for this would require empirically
  measuring correlation among these sub-scores across a real historical
  ticker universe.
- **DCF methodology is dated**: independent of the unit-conversion fix already
  applied, the DCF estimate is Graham's unmodified 1962 heuristic
  (`8.5 + 2×growth`), known to be unreliable for high-growth,
  negative-earnings, or cyclical names. It remains one ensemble input among
  several rather than a standalone valuation.
- **Static sector P/E benchmarks**: `SECTOR_BENCHMARK_PE` in
  `modules/fundamental_analysis.py` is a hardcoded dict with no refresh
  mechanism, so sector multiples can drift out of date with rate cycles.
- **Survivorship bias in the ticker universe**: `get_sp500_tickers()` scrapes
  the *current* Wikipedia membership table, so LightGBM training and
  walk-forward backtests only ever see tickers still in the index today,
  which can inflate apparent historical accuracy versus a true point-in-time
  historical membership list.
- **RMSE-metric geometry favors terminal-point accuracy**: the LightGBM
  walk-forward comparison converts every model's forecast into a smooth
  geometric curve toward one terminal-return guess before computing RMSE,
  which structurally advantages models optimized directly for terminal
  return (like LightGBM) over general-purpose extrapolators (ARIMA/trend). A
  supplementary path-level metric would give a fuller comparison.
- **No transaction costs or slippage in backtest RMSE**: backtests measure
  price-forecast RMSE, not net-of-cost tradeable returns. Treat backtest wins
  as directional-accuracy evidence only, not proof of after-cost
  profitability.

## Testing & CI

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

`.github/workflows/ci.yml` runs on every push to `main` and every pull
request:
- **test**: installs `requirements.txt` + `requirements-api.txt`, runs the
  full `unittest` suite.
- **lint**: `ruff check --select E9,F .` (Python syntax errors + pyflakes
  only — not full style/complexity linting), then `frontend/`'s `tsc
  --noEmit` type-check and `npm test` (Vitest).

Separately, `.github/workflows/train-lightgbm-batch.yml`,
`scan-email-short-term.yml`, `scan-email-medium-term.yml`,
`grading-report-short-term.yml`, and `grading-report-medium-term.yml` run on
schedules to retrain/promote LightGBM models and send scan/grading email
reports. They install `requirements-workflows.txt` (not the full
`requirements.txt`, since they only run `scripts/*.py` against `modules/`
headlessly) and don't touch `api/` or `frontend/`.

## Troubleshooting

- **macOS SSL/cert issues**: run Python from an environment with updated certs and retry `pip install -r requirements.txt`.
- **POLYGON_API_KEY missing**: either export it in your shell or add it to a project-root `.env` file before launching Streamlit/API workers.
- **Polygon data unavailable for a ticker**: retry shortly; the app handles missing responses gracefully and skips unavailable symbols.
- **Fundamental metric timing**: SEC EDGAR fundamentals update on filing cadence
  (10-Q/10-K), not daily like market-price feeds.

## Disclaimer

This project is for educational/research use only and is **not financial advice**.

## Recent improvements

### Scheduled scan and grading email workflows

The repository now includes four scheduled GitHub Actions workflows that reuse
the promoted LightGBM `latest.json` manifest universe and email finance-themed
HTML reports via Brevo:

- `.github/workflows/scan-email-short-term.yml`
- `.github/workflows/scan-email-medium-term.yml`
- `.github/workflows/grading-report-short-term.yml`
- `.github/workflows/grading-report-medium-term.yml`

The scan workflows run `scripts/scan_email_report.py` for short-term or
medium-term profit-opportunity horizons, attach screener/profit-opportunity top
75 CSVs, and record the profit-opportunity picks into `data/predictions.json`
for later grading. The grading workflows run `scripts/grading_report.py` to
evaluate the exact prior recorded scan batch using both point-in-time resolution
and max-price-since-scan excursion checks.

### Track-record bugfix: Stock Analysis no longer auto-records predictions

Viewing a ticker on the Stock Analysis page no longer writes passive
`stock_analysis` predictions into `data/predictions.json`. Track-record entries
are now created only from deliberate profit-opportunity scans (interactive or
scheduled), which keeps Track Record totals free of browse-noise.

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
watchlist modules. Unlike other `data/*.json` files (which hold personal
portfolio/watchlist data and stay untracked), `data/predictions.json` is
explicitly un-ignored in `.gitignore` and is committed back to the repository
by the `scan-email-*` and `grading-report-*` GitHub Actions workflows after
each run, so prediction history survives across ephemeral CI filesystems.
