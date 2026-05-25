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
- Fundamental analysis scoring
- News sentiment scoring
- Unified score + recommendation engine with entry/target/stop-loss suggestions

## Score legend

- 80–100: 🟢 **STRONG BUY**
- 65–79: 🔵 **BUY**
- 50–64: 🟡 **TAKE SMALL POSITION**
- 35–49: 🟠 **MONITOR**
- 20–34: 🔴 **DO NOT BUY**
- 0–19: ⛔ **AVOID**

## Notes

- NLST and other OTC symbols are supported (data may be limited).
- Portfolio holdings are manually entered (no broker integration).
- Designed for a moderate-risk profile.
- Uses free data only (yfinance).
- Domain is optional; deploy on Streamlit Cloud for a free public URL.

## Disclaimer

This project is for educational/research use only and is **not financial advice**.
