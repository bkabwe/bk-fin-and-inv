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
