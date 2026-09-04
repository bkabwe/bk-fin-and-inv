# BK Fin API + React Frontend

## Prerequisites
1. Homebrew
2. Node.js 18+
3. Python 3.11+
4. Redis

## Installation
```bash
brew install redis node
pip3 install -r requirements-api.txt
cd frontend && npm install
```

## Running
```bash
chmod +x start-api.sh && ./start-api.sh
```

## Running Streamlit instead
```bash
streamlit run app.py
```

## API docs
http://localhost:8000/docs

## Troubleshooting
- Redis not running: `brew services start redis`
- Port conflicts: free ports 5173/8000/6379
- Celery not connecting: verify Redis on localhost:6379

## Recent changes

### Split-adjusted price data
All `yfinance` calls in the backend now use `auto_adjust=True`.  Historical
OHLC series used for indicators, backtests, and forecasting are continuously
adjusted for splits/dividends — eliminating artificial price jumps in API
responses.

### TTL-aware cache in non-Streamlit mode
The `@cache_data(ttl=...)` fallback used by the FastAPI/Celery backend now
honours the configured TTL (previously it used `lru_cache` which ignored TTL,
causing potentially stale data to be served indefinitely).

### Screener API changes (`/screener`)

**New request fields** (`POST /screener/start`):

| Field | Type | Default | Description |
|---|---|---|---|
| `use_fast_screen` | bool | `true` | Enable two-tier fast-screen pre-filter |
| `fast_screen_margin` | int | `15` | Safety margin (points) for fast-screen cutoff |
| `revalidate_with_tiingo` | bool | `false` | Optionally re-check final top results with Tiingo |
| `revalidate_top_n` | int | `50` | Number of ranked results to revalidate (1-100) |

**New progress fields** (`GET /screener/{job_id}/progress`):

| Field | Type | Description |
|---|---|---|
| `fast_filtered` | int | Tickers eliminated by the fast-screen tier |
| `fully_analyzed` | int | Tickers that went through full analysis |
| `failed_count` | int | Tickers that raised an exception during analysis |
| `failed_tickers` | list | `[{"ticker": "X", "reason": "..."}]` objects |
| `revalidation_status` | string | `not_requested`, `pending`, `running`, `complete`, or `unavailable` |

The same `failed_count` / `failed_tickers` fields are also available on the
`/profit/{job_id}/progress` endpoint.

### Chart accuracy
The `/analysis/{ticker}/price` endpoint now returns split-adjusted OHLC data
so the React `PriceChart` component no longer displays artificial price jumps.
Missing OHLC values in the chart are filtered out client-side rather than
coerced to `0`, eliminating false collapse-to-zero artifacts.
