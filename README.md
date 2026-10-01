# 📈 BK Stock Market Analyzer

Comprehensive US stock analysis toolkit inspired by Richard W. Schabacker's technical analysis principles.

## Installation

```bash
pip install -r requirements.txt
```

`requirements.txt` is the full set needed to run the Streamlit dashboard
locally (adds `streamlit`, `plotly`, `Pillow` for the UI on top of the core
analysis stack). The scheduled GitHub Actions workflows never render the
Streamlit UI, so they install a leaner file instead — see
`.github/workflows/*.yml`. Scan/grading email report workflows install
`requirements-workflows.txt`; `train-lightgbm-batch.yml` installs the even
leaner `requirements-lightgbm-batch.txt` (same core stack minus
`torch`/`transformers`, since the batch pipeline never needs FinBERT
sentiment). The FastAPI/Celery backend under `api/` uses
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
- `WORKFLOW_ALERT_RECIPIENTS` — optional comma-separated email recipients for
  workflow-failure alerts only (see "Workflow-failure alerts" below). If
  unset, alerts fall back to `SCAN_EMAIL_FROM` alone, so by default only the
  maintainer is paged rather than the full `SCAN_EMAIL_RECIPIENTS` list.

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
- Multi-model projected price targets (log-linear/log-polynomial trend + LightGBM return models for 30d/180d when saved models exist + fundamental fair value + DCF + technical resistance + analyst target, with GARCH confidence bounds)
- Prophet was removed from the ensemble after walk-forward backtests consistently showed materially higher RMSE on stock series (which generally lack the strong recurring seasonality Prophet is designed for). ARIMA was later removed too: on its own it isn't reliable for the dynamism of stock markets, and live scoring's adaptive weighting already favored trend/LightGBM over it. `modules/backtester.py`'s walk-forward comparison tooling (`scripts/compare_lightgbm_backtest.py`) still supports an opt-in ARIMA evaluation for offline research.

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
  total. These five sub-scores are now recorded per scan alongside each
  prediction (see `modules/prediction_tracker.py`'s `SUBSCORE_ROW_FIELDS`),
  and `compute_subscore_correlation_stats()` empirically measures their
  pairwise correlation across recorded scan history once enough predictions
  have accumulated — no down-weighting/consolidation has been applied yet,
  pending that analysis.
- **DCF methodology is dated**: the DCF estimate is still Graham's unmodified
  1962 heuristic (`8.5 + 2×growth`), known to be unreliable for high-growth,
  negative-earnings, or cyclical names. `analyze_fundamentals()` now blends it
  with a comparables-based estimate (trailing EPS × sector-benchmark P/E) into
  a `valuation_estimate` used by the scoring ensemble, so a single dated
  heuristic no longer drives that ensemble slot alone — but neither input is
  a modern multi-stage DCF, so this remains a documented simplification.
- **Sector P/E benchmarks now have a manual refresh path**: `SECTOR_BENCHMARK_PE`
  in `modules/fundamental_analysis.py` remains a hardcoded static fallback,
  but `scripts/refresh_sector_pe.py` can be run periodically (manually, or via
  a scheduled workflow) to compute fresh per-sector median trailing P/Es from
  live ticker data and write them to `data/sector_benchmark_pe.json`
  (gitignored, local/derived data); `get_sector_benchmark_pe()` prefers that
  refreshed file over the static dict whenever it exists.
- **Survivorship bias in the ticker universe**: `get_sp500_tickers()` scrapes
  the *current* Wikipedia membership table, so LightGBM training and
  walk-forward backtests only ever see tickers still in the index today,
  which can inflate apparent historical accuracy versus a true point-in-time
  historical membership list. A survivorship-bias-aware ticker universe
  (e.g. a point-in-time membership snapshot) remains a follow-up data
  project rather than a quick fix — see the docstring on
  `modules/data_fetcher.get_sp500_tickers()` for the full caveat.
- **RMSE-metric geometry favors terminal-point accuracy**: the LightGBM
  walk-forward comparison converts every model's forecast into a smooth
  geometric curve toward one terminal-return guess before computing RMSE,
  which structurally advantages models optimized directly for terminal
  return (like LightGBM) over general-purpose extrapolators (ARIMA/trend).
  `run_walk_forward(..., evaluate_path_rmse=True)` now adds a supplementary
  **path-level RMSE** (`arima_path_rmse`/`trend_path_rmse`/
  `lightgbm_path_rmse`) that compares day-over-day log returns instead of
  price levels, directly penalizing LightGBM's artificially smooth
  interpolated path against the noisier real walk — see `_path_rmse()` in
  `modules/backtester.py`. It is opt-in (off by default) since it adds
  per-window computation; `scripts/compare_lightgbm_backtest.py --path-rmse`
  enables it from the CLI.
- **Window counts are small — raw RMSE deltas aren't statistical proof**:
  per-ticker walk-forward window counts can be as low as 1-4, too few to
  trust a raw "LightGBM RMSE is lower" claim as evidence of real skill
  rather than noise. `modules.backtest_comparison.diebold_mariano_test()`
  now pools one (RMSE² difference) observation per ticker across the
  comparison and runs a paired t-test against zero, reporting a p-value and
  significance flag; `summarize_backtest_results()` includes a
  `significance_tests` list (LightGBM vs. ARIMA/Trend/naive) and
  `format_comparison_summary()` prints it. Needs at least 2 tickers with
  overlapping coverage to estimate variance.
- **No transaction costs or slippage in backtest RMSE**: backtests measure
  price-forecast RMSE, not net-of-cost tradeable returns. Treat backtest wins
  as directional-accuracy evidence only, not proof of after-cost
  profitability. A proper fix is a distinct, larger-scope evaluation
  framework (simulating net-of-cost tradeable returns from a trading rule),
  not a tweak to the existing RMSE comparison, so it remains a documented
  gap rather than a quick change.
- **LightGBM hyperparameters (`n_estimators=250, learning_rate=0.05,
  num_leaves=31`) are fixed, not tuned**: a proper sweep is constrained by
  the same small training windows noted above (as few as ~30 embargoed rows
  at the 30d horizon after the purge window), so any grid search would risk
  overfitting hyperparameters to a single backtest run — the exact anti-
  pattern the Diebold-Mariano significance testing above was added to catch.
  Revisiting these values is better done once enough accumulated
  `modules/experiment_tracker.py` run history exists to compare candidate
  configs across many runs rather than one.
- **Long-horizon projections do not account for dividends**: price history
  from Polygon is split-adjusted but **not** dividend-adjusted (see "Data
  Sources" below), so projected upside for dividend-paying names is a pure
  price-return estimate. This caveat compounds at longer horizons — a 720d
  projection accumulates roughly 2 years of foregone/ignored dividend
  compounding, whereas a 30d projection barely feels it. No total-return
  adjustment is applied; treat long-horizon upside numbers for
  high-dividend-yield names as a price-only lower bound on total return.
- **No earnings-date/event-risk awareness**: forecasts do not flag or
  discount confidence for tickers with an earnings release inside the
  projection horizon, even though earnings prints are a major source of
  short-horizon price-path discontinuity that a smooth technical/LightGBM
  projection cannot anticipate. This is a known gap rather than an
  oversight: Polygon's earnings-calendar endpoint
  (`/vX/reference/earnings`) requires an Advanced-tier subscription, one
  level above the Starter plan this app is built against (see "Optional
  environment variables" above), so this remains blocked on a data-source
  upgrade rather than an engineering task.

## Testing & CI

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

`.github/workflows/ci.yml` runs on every push to `main` and every pull
request:
- **test**: installs `requirements.txt` + `requirements-api.txt`, runs the
  full `unittest` suite.
- **lint**: `ruff check --select E9,F,B,SIM .` (syntax errors, pyflakes,
  flake8-bugbear, and flake8-simplify — still not full style/complexity
  linting), then `frontend/`'s `tsc --noEmit` type-check and `npm test`
  (Vitest).
- **typecheck**: `mypy` (config in `pyproject.toml`), scoped to `modules/`.
  This job is advisory only (`continue-on-error: true`) and will not fail a
  PR — it surfaces type issues incrementally as coverage improves. It runs
  on Python 3.12 (rather than the `PYTHON_VERSION` used by `test`/`lint`)
  because numpy's bundled type stubs need 3.12+ to parse.

Separately, `.github/workflows/train-lightgbm-batch.yml`,
`scan-email-short-term.yml`, `scan-email-medium-term.yml`,
`grading-report-short-term.yml`, and `grading-report-medium-term.yml` run on
schedules to retrain/promote LightGBM models and send scan/grading email
reports. They don't touch `api/` or `frontend/`, and none of them install the
full `requirements.txt` since they only run `scripts/*.py` against `modules/`
headlessly. `train-lightgbm-batch.yml` and the scan workflows
(`scan-email-short-term.yml`/`scan-email-medium-term.yml`) share the same
discover -> shard (matrix) -> reduce job structure so a single run can process
the full trained-ticker universe without any one job loading everything into
memory at once or exceeding the 6-hour per-job limit. Only each scan
workflow's `scan-shard` job — which calls `analyze_stock` and therefore needs
FinBERT sentiment — installs `requirements-workflows.txt`; every other job
(`discover`, `reduce`, and all of `train-lightgbm-batch.yml`'s jobs) installs
the leaner `requirements-lightgbm-batch.txt`, since those steps never call
`analyze_sentiment` and skipping the multi-GB `torch`/`transformers` install
leaves more disk headroom for the batch-artifact merge steps. The
grading-report workflows remain single-job and still install
`requirements-workflows.txt` directly.

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
medium-term profit-opportunity horizons, split across `discover` (fetches the
promoted manifest and shared FRED macro table once), `scan-shard` (a matrix
job — one per ticker-partitioned shard declared in the manifest, or a single
job for older/unsharded manifests — that scans/filters its slice of tickers
without truncating or recording), and `reduce` (merges every shard's partial
results, truncates to the final top 75, records the profit-opportunity picks
into `data/predictions.db` for later grading exactly once, attaches
screener/profit-opportunity top 75 CSVs, and emails the report). The grading
workflows run `scripts/grading_report.py` to evaluate the exact prior recorded
scan batch using both point-in-time resolution and max-price-since-scan
excursion checks.

Shard count is manifest-driven, set by `--scan-shard-count` at promote time
(`scripts/lightgbm_batch_pipeline.py`'s `DEFAULT_SCAN_SHARD_COUNT`, currently
16, chosen to fit within a single wave under the account's GitHub Actions
concurrent-job cap) — a separate concept from `train-lightgbm-batch.yml`'s own `MATRIX_JOBS`
training-compute parallelism. FRED is fetched once in `discover` and shared
via artifact with every live scoring call inside a shard job — including each
ticker's walk-forward backtest windows in `modules/backtester.py`, not just
the live LightGBM projection path — so per-ticker analysis never re-fetches
macro data live from FRED. SEC EDGAR fundamentals, however, are fetched
per-ticker inside every `scan-shard` job, and
`modules/sec_edgar_client.py` only self-throttles each job's own process to
`SEC_EDGAR_MAX_REQUESTS_PER_SECOND` (default/max 10 req/s) — it can't
coordinate across the parallel shards on its own. Each scan workflow's "Scan
shard" step divides that per-shard cap by the live shard count
(`10.0 / needs.discover.outputs.shard_count`), the same pattern
`train-lightgbm-batch.yml`'s `train` job already uses for its own
`MATRIX_JOBS`, so the *aggregate* SEC EDGAR call rate across all shards stays
bounded at ~10 req/s regardless of how many shards run concurrently. Within
each shard job, tickers are analyzed concurrently via a `ThreadPoolExecutor`
(`scripts/scan_email_report.py::run_scan_shard`, `DEFAULT_SCAN_SHARD_MAX_WORKERS = 8`,
mirroring `modules/screener.py::run_screener`'s existing pattern), since each
ticker's analysis is independent and I/O-bound.

### Workflow-failure alerts

All 5 scheduled workflows (`train-lightgbm-batch.yml`, `scan-email-short-term.yml`,
`scan-email-medium-term.yml`, `grading-report-short-term.yml`, and
`grading-report-medium-term.yml`) include a `notify-on-failure` job
(`if: failure()`, depending on every other job in the workflow) that emails a
short failure alert — repo, branch, and a link to the failed run — via the same
Brevo API integration used for scan/grading reports
(`scripts/notify_workflow_failure.py`, reusing `modules/email_reports.py`'s
`send_brevo_email`). This surfaces pipeline breaks (e.g. a reduce/promote job
failure) proactively instead of only being noticed by chance. The notifier only
installs `requests` (not each workflow's full dependency set), and any
failure while sending the alert itself is swallowed so it never masks or
replaces the original job failure it's reporting on.

These alerts intentionally go to a **separate, narrower** recipient list than
the scan/grading reports: `WORKFLOW_ALERT_RECIPIENTS` if set, otherwise just
`SCAN_EMAIL_FROM` (the maintainer's own address) — never the full
`SCAN_EMAIL_RECIPIENTS` distribution list, since a CI/pipeline failure is an
operational concern for whoever maintains the workflows, not every scan-report
recipient.

### Track-record bugfix: Stock Analysis no longer auto-records predictions

Viewing a ticker on the Stock Analysis page no longer writes passive
`stock_analysis` predictions into `data/predictions.db`. Track-record entries
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
phase). All features are intentionally stationary (ratios, percent changes, and
returns rather than raw price/level values) so a tree-based model like LightGBM
never has to extrapolate outside the numeric range it was trained on. The table
combines:
- Technical features: rolling volatility, close-to-EMA and close-to-daily-VWAP-
  approximation ratios, RSI, volume-vs-average, lagged daily returns (AR-style:
  1d/2d/5d), and rolling mean returns (MA-style: 5d/20d/50d)
- SEC EDGAR fundamentals (`fundamental_revenue_growth`,
  `fundamental_debt_to_equity` (percent points, matching existing SEC adapter
  scaling), and `fundamental_gross_margin`), forward-filled from filing dates
  across daily rows
- Shared macro features from FRED (`DGS10`, `CPIAUCSL`, `FEDFUNDS`) as 5-day/
  30-day delta and percent-change metrics only (raw levels are used internally
  to compute these but are not exposed as features)

Design principle: feature assembly is TTL-cached incrementally and decoupled from
scan cadence so repeated scans avoid unnecessary recomputation/refetching.

### LightGBM return models (live for 30d/180d, gated off for 720d)
The app now includes `modules/lightgbm_model.py` as an explicit Phase 2 candidate
model module. It predicts **forward return** (not raw price) for the existing
30/180/720-day horizons using the shared engineered feature table and historical
close prices. Training/inference are intentionally decoupled via save/load helpers
so scan-time inference does not require retraining. LightGBM's own features
(`modules/feature_engineering.build_feature_table`) already include forward-filled
SEC EDGAR fundamentals alongside technical indicators and shared FRED macro
features, so it is not a purely technical model.

Live scoring now runs a dedicated per-ticker walk-forward backtest at scan time
for each in-scope horizon (`run_walk_forward(..., horizon=30, evaluate_lightgbm=True,
evaluate_arima=False)` and the same at `horizon=180`) and derives LightGBM's
ensemble weight **adaptively** from that evidence: `_inverse_rmse_weights()`
splits each horizon's weight budget across trend/LightGBM in proportion to
their inverse walk-forward RMSE, instead of reserving a fixed percentage for
LightGBM. A **dynamic cap** (`_lightgbm_adaptive_share_cap()`) limits
LightGBM's share of that split so a noisy, small-sample per-ticker backtest
can't crowd out trend entirely; any excess above the cap is redistributed
back to trend. The cap itself scales with how many walk-forward windows back
the ticker's `lightgbm_rmse` estimate: it floors at
`LIGHTGBM_MIN_ADAPTIVE_SHARE_CAP` (50%) for sparse evidence (≤1 window) and
rises linearly to `LIGHTGBM_MAX_ADAPTIVE_SHARE_CAP` (70%) once evidence
reaches `LIGHTGBM_ADAPTIVE_CAP_FULL_EVIDENCE_WINDOWS` (6) windows, so
better-replicated per-ticker backtests (e.g. the 30d horizon's typical 6
windows) can earn a larger adaptive share than thinner ones (e.g. 180d's
typical 4 windows, landing at a 62% cap). When a ticker's backtest can't
produce a valid `lightgbm_rmse` (e.g. too little history), live scoring
falls back to the previous fixed weights, `LIGHTGBM_WEIGHT_30D = 0.15` and
`LIGHTGBM_WEIGHT_180D = 0.08`, which remain in the code as that fallback.
**720d remains intentionally excluded from the live ensemble for now**
because current validation has only one non-overlapping window per ticker,
which is promising but not yet actionable.

ARIMA was removed from the live ensemble entirely (it isn't reliable on its
own for the dynamism of stock markets, and adaptive weighting already
favored trend/LightGBM over it in practice); `evaluate_arima=False` above
skips the expensive per-ticker ARIMA order search during live scans.
`modules/backtester.run_walk_forward()` still supports an opt-in
`evaluate_arima=True` (the default) for offline research/comparison via
`modules/backtest_comparison.py`.

Because the 180d backtest needs at least 540 rows of history (a 360-row train
window + 180-row test window), the live price-projection path now fetches 5
years of daily history by default (up from 2y) so that backtest can actually
run instead of silently reporting zero windows; every downstream consumer of
that history already truncates to its own required window, so the larger
fetch is safe.

To train and persist the live 30d/180d models for a ticker before scoring:

```bash
python scripts/train_lightgbm_return_models.py AAPL
```

This writes per-ticker joblib artifacts under `data/lightgbm_return_models/<TICKER>/`.
If no saved model is available, live scoring falls back to the trend-only
ensemble behavior for that ticker/horizon.

### Walk-forward validation gate for LightGBM
`modules/backtester.run_walk_forward()` now supports optional LightGBM RMSE
evaluation alongside trend (and, when `evaluate_arima=True`, ARIMA) over the
same rolling windows, for validation/decision-gate analysis. The live
ensemble consumes that evidence at 30d/180d only, via its own per-horizon
backtest call at each of those two horizons with `evaluate_arima=False` (see
above); 720d remains gated off until more natural history accrues and yields
more than one non-overlapping validation window per ticker.
Because LightGBM is trained as fixed-horizon return models (`30/180/720` days),
the walk-forward path maps each test window to the closest available horizon
(e.g., 30-day test windows use the 30-day model) and converts the predicted
return into a daily price path ending at that horizon for RMSE comparison.
An earlier validation bug made this path report all-zero LightGBM window counts:
macro-feature columns could remain object-typed when FRED data was unavailable,
causing every per-window LightGBM fit to fail under the old silent `except`.
Feature tables are now coerced back to numeric dtypes before training, and any
remaining per-window LightGBM failure is logged with ticker/window context.

LightGBM's training rows are also now **embargoed**: the last `horizon` rows of
each raw training window are excluded before fitting, because their
forward-return label's target close falls inside that window's held-out test
period (a purged/embargoed split, preventing label leakage across the
train/test boundary). This shrinks the usable 30d-horizon training window from
60 to 30 rows; 180d/720d windows are large enough that the embargo removes a
proportionally smaller share.

To reproduce aggregate comparisons across a representative ticker sample:

```bash
python scripts/compare_lightgbm_backtest.py --sample-size 30
```

The script prints mean/median RMSE per model, LightGBM win percentage vs both
ARIMA and trend, and ticker/window evaluation counts per model.

### ARIMA convergence hardening
ARIMA is no longer part of the live price-projection ensemble (see above), but
`modules/arima_hardening.py` still backs ARIMA fitting in the walk-forward
backtest-comparison tooling (`modules/backtest_comparison.py`, opt-in via
`evaluate_arima=True`) and the independent "days to target price" estimate in
`modules/profit_opportunities.py`. Both paths use stronger convergence
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
   trend/GARCH/backtest).  The subscore is normalized to 0–100.
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
  pre-filter before the expensive trend/GARCH full analysis.  The
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


### Point-in-time universe/sentiment snapshots

The repo now includes `scripts/snapshot_universe.py` plus a daily
`.github/workflows/snapshot-universe.yml` job that appends the current
`get_sp500_tickers()` membership list to
`data/snapshots/universe_membership.jsonl`. This does **not** remove the
existing survivorship-bias caveat overnight — historical backtests still only
know today's membership until enough dated snapshots accumulate — but it starts
building the point-in-time index-membership history that a future
survivorship-bias-aware training/backtest path can consume instead of
retroactively scraping Wikipedia's current table.

The same script also has an opt-in `--with-sentiment` mode that writes one
JSONL summary per ticker under `data/snapshots/sentiment/YYYY-MM-DD.jsonl`,
using `analyze_sentiment()`'s FinBERT + market-signal output. The scheduled
workflow intentionally leaves that heavier path off for now so the daily run
stays fast, but the plumbing is now in place to begin accumulating a dated
sentiment archive instead of only re-computing whatever headlines happen to be
available at scan time.

### Experiment tracking for backtest runs

The repo now includes `modules/experiment_tracker.py`, a lightweight SQLite
store at `data/experiments.db` for recording walk-forward/backtest comparison
runs: run name, horizon, optional explicit git commit, serialized config,
serialized summary metrics, and optional sample tickers. This addresses the
current gap where research runs are easy to print once but awkward to compare
later without ad-hoc notes or copy/paste.

`list_experiment_runs()` surfaces recent runs as plain dicts, and
`compare_latest_two_runs()` computes per-model numeric deltas such as
`models.lightgbm.mean_rmse` between the two most recent runs for the same
`(run_name, horizon)` pair. Unlike `data/predictions.db`, this database is
intentionally local/derived developer state for iterative experimentation, so
it is gitignored rather than committed back by CI.

### Per-horizon projection confidence, shrinkage, and thin-history flags

Each of the short/medium/long-term price projections now carries its own
0-100 **projection confidence** (`short_term_confidence`,
`medium_term_confidence`, `long_term_confidence` in
`modules/scoring_engine.py`), computed from ensemble model agreement/
dispersion and backtest-window evidence depth — deliberately kept separate
from the composite 0-100 `Score`, since a stock can score well on
fundamentals/technicals while its *projection* for a given horizon is
backed by thin evidence (e.g. a recent IPO, or a horizon with only 1-2
backtest windows).

That confidence number does two things:

- **Shrinkage**: `_apply_confidence_shrinkage()` pulls each horizon's point
  projection back toward the current price in proportion to
  `(1 - confidence/100)`, so a low-confidence extreme forecast (large
  magnitude, thin evidence) is damped rather than passed through at full
  size, while a high-confidence forecast is left mostly intact. This is
  separate from (and on top of) the existing GARCH/market-cap confidence
  **band**-widening, which affects the displayed range, not the point
  estimate itself.
- **Thin-history flag**: `_is_thin_history(horizon_days, history_days)`
  marks `short_term_thin_history`/`medium_term_thin_history`/
  `long_term_thin_history` as `True` when available price history is too
  short relative to the horizon to trust that horizon's projection at all.

### Risk-adjusted upside and diversification cap

The Profit Opportunities scan (`modules/profit_opportunities.py`) now
surfaces a **Confidence Score** column (the relevant horizon's projection
confidence above) and a **Risk-Adjusted Upside** column
(`projected upside % ÷ confidence-band width %`, floored at a minimum band
width to avoid dividing by a near-zero band), rewarding upside that comes
with a tight projection range over the same upside wrapped in a wide one.
Both the scheduled scan report (`scripts/scan_email_report.py`) and
`modules/screener.py` also now carry a **Sector** field used by a new
`_apply_diversification_cap()` (max `DEFAULT_MAX_PER_SECTOR = 15` per
sector) applied to every ranked cut, and `_exclude_thin_history()` drops
thin-history rows from the qualifying pool before ranking. The scheduled
report is now split into **three** top-75 rankings over the same qualifying
pool: by projected upside, by overall score, and by risk-adjusted upside —
not just the original "by upside"/"by score" pair.

### Upside-forecast validation (beyond the composite Score)

`compute_score_validation_stats()` in `modules/prediction_tracker.py` only
ever tested the composite 0-100 `Score`'s rank-ordering power against
realized outcomes. The new `compute_upside_validation_stats()` instead
tests the raw `projected_upside_pct` forecast number itself against
`actual_return_pct` once predictions resolve — reporting an information
coefficient (Pearson correlation between projected and realized return),
mean/median bias (a large negative gap flags systematic over-optimism in
large upside projections), and an over-optimism rate. `scripts/
grading_report.py` now surfaces this alongside the existing score
validation stats.

### Liquidity/tradability filter for scan universes

`get_sp500_tickers()`, `get_nasdaq_tickers()`, `get_nyseamerican_tickers()`,
and `get_otc_tickers()` in `modules/data_fetcher.py` previously had no
liquidity floor, so thinly-traded names with little realistic tradability
could enter the scan universe. `filter_tickers_by_liquidity()` now drops
tickers whose trailing average dollar volume falls below a floor
(`DEFAULT_MIN_AVG_DOLLAR_VOLUME = $1M/day` for the major exchanges,
`DEFAULT_MIN_AVG_DOLLAR_VOLUME_OTC = $100K/day` for OTC, since OTC names are
structurally much lower-volume and a $1M floor would eliminate nearly the
whole OTC universe). To avoid thousands of per-ticker API calls, the filter
is backed by a new bulk `get_grouped_daily_bars()` / `get_latest_grouped_
daily_bars()` pair in `modules/polygon_client.py`, which fetches close +
volume for **every** US ticker on the latest trading day in a single
Polygon call (`/v2/aggs/grouped/locale/us/market/stocks/{date}`), walking
back up to 10 days to skip weekends/holidays. The filter fails open (keeps
the unfiltered list) on any fetch error or empty response, and keeps (does
not drop) tickers missing from the bars response (e.g. very recent IPOs),
treating "no data" as not evidence of illiquidity.

### LightGBM bagging/model-averaging (opt-in)

`modules/lightgbm_model.train_return_models()` gained an opt-in
`n_bagged_estimators` parameter (default `1`, i.e. unchanged single-model
behavior). When set above 1, it trains that many row-bootstrapped,
differently-seeded `LGBMRegressor`s per horizon (each fit on a
`bagging_fraction` resample of the training rows) and wraps them in a new
`BaggedLGBMRegressor` that averages their predictions — a variance-reduction
technique suited to the existing small training windows (as few as ~50-60
rows at the 30d horizon) without touching the feature set or objective. The
wrapper mirrors the plain model's `.predict()`/`.feature_name_` interface,
so it drops into the existing `predict_forward_return()`/`save_return_
models()`/`load_return_models()` plumbing unchanged, and is also exposed via
`scripts/train_lightgbm_return_models.py --n-bagged-estimators`. The
scheduled batch-training pipeline (`scripts/lightgbm_batch_pipeline.py`)
still calls `train_return_models()` with the implicit default (bagging
off), since enabling it for live models is a methodology change that should
follow a dedicated walk-forward comparison first, not ship as a default.

### LightGBM quantile regression / prediction intervals (not yet live)

`modules/lightgbm_model.train_return_quantile_models()` trains per-horizon,
per-quantile LightGBM models (`objective="quantile"`) to produce forward-
return prediction intervals (default an 80% interval via
`DEFAULT_RETURN_QUANTILES = (0.1, 0.9)`), alongside new
`predict_forward_return_quantiles()`/`save_return_quantile_models()`/
`load_return_quantile_models()` helpers mirroring the point-estimate API.
This is deliberately **not yet wired into** the live scoring ensemble's
GARCH-based confidence bands: doing so would change several already-
validated downstream fields (market-cap band padding, per-horizon
projection confidence, risk-adjusted upside), so it ships here as a
tested, standalone capability for offline evaluation first, pending a
backtest comparing it against the current band-construction approach.

---

## Prediction Track Record

### Overview
Every price projection shown to the user is automatically persisted so that,
once the estimated target date has passed, the app can verify whether the
prediction actually hit — giving you an empirical track record of the
trend/LightGBM/GARCH-enhanced ensemble's accuracy.

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
model basis string, source, and (when available) a `sub_scores` breakdown of
the five technical sub-scores (`trend`, `momentum`, `relative_strength`,
`breakout`, `volume_quality`) used to empirically check their correlation —
see `compute_subscore_correlation_stats()` in `modules/prediction_tracker.py`.

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
Predictions are persisted to `data/predictions.db`, a SQLite database (see
`modules/prediction_tracker.py`). Unlike other `data/*.json` files (which
hold personal portfolio/watchlist data and stay untracked), `data/predictions.db`
is explicitly un-ignored in `.gitignore` and is committed back to the
repository by the `scan-email-*` and `grading-report-*` GitHub Actions
workflows after each run, so prediction history survives across ephemeral CI
filesystems. `.gitattributes` marks it binary so git never attempts a
line-based diff/3-way merge on it; the workflows' push-retry loop runs
`git rebase --abort` (instead of blindly retrying) if a rebase ever fails,
since a genuine concurrent-write conflict on a binary file can't be
auto-resolved the way a JSON conflict sometimes could.

Earlier versions of this module stored predictions as a flat
`data/predictions.json` file, with every write (even recording one new
prediction) loading and re-serializing the entire, ever-growing history.
SQLite lets `record_prediction`/`resolve_pending_predictions` issue targeted
INSERT/UPDATE statements against only the rows they touch, and lets the
read-heavy validation stats (`compute_score_validation_stats`,
`compute_upside_validation_stats`, `compute_subscore_correlation_stats`)
`SELECT` only the (resolved, or sub-score-bearing) rows they need instead of
materializing the full history in Python — so these all stay cheap as
prediction history keeps growing. The historical `data/predictions.json` file
is kept in the repo as a frozen snapshot (no longer written to); a one-time
`scripts/migrate_predictions_to_sqlite.py` backfilled it into
`data/predictions.db` and can be re-run safely (it's idempotent, keyed on
each record's existing `id`).
