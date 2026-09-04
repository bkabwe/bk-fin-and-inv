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
Backend historical OHLCV now comes from Polygon aggregates with `adjusted=true`,
which keeps indicator/backtest/forecast series continuous across corporate actions.
Current price is sourced from Polygon previous-day close (Starter-plan compatible).

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

**New progress fields** (`GET /screener/{job_id}/progress`):

| Field | Type | Description |
|---|---|---|
| `fast_filtered` | int | Tickers eliminated by the fast-screen tier |
| `fully_analyzed` | int | Tickers that went through full analysis |
| `failed_count` | int | Tickers that raised an exception during analysis |
| `failed_tickers` | list | `[{"ticker": "X", "reason": "..."}]` objects |
| `revalidation_status` | string | `not_requested` |

The same `failed_count` / `failed_tickers` / `revalidation_status` fields are
also available on the `/profit/{job_id}/progress` endpoint.

### Chart accuracy
The `/analysis/{ticker}/price` endpoint now returns split-adjusted OHLC data
so the React `PriceChart` component no longer displays artificial price jumps.
Missing OHLC values in the chart are filtered out client-side rather than
coerced to `0`, eliminating false collapse-to-zero artifacts.
