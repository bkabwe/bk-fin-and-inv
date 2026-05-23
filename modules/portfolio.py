from __future__ import annotations

import json
from pathlib import Path

from modules.scoring_engine import analyze_stock

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PORTFOLIO_FILE = DATA_DIR / "portfolio.json"
WATCHLIST_FILE = DATA_DIR / "watchlist.json"


def _ensure():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not PORTFOLIO_FILE.exists():
        PORTFOLIO_FILE.write_text("[]", encoding="utf-8")
    if not WATCHLIST_FILE.exists():
        WATCHLIST_FILE.write_text("[]", encoding="utf-8")


def _load(path: Path):
    _ensure()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save(path: Path, payload):
    _ensure()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def get_portfolio() -> list[dict]:
    return _load(PORTFOLIO_FILE)


def save_portfolio(portfolio: list[dict]):
    _save(PORTFOLIO_FILE, portfolio)


def add_holding(ticker: str, shares: float, avg_cost: float, date_purchased: str, notes: str = ""):
    ticker = ticker.upper().strip()
    portfolio = get_portfolio()
    match = next((x for x in portfolio if x.get("ticker") == ticker), None)
    data = {"ticker": ticker, "shares": float(shares), "avg_cost": float(avg_cost), "date_purchased": date_purchased, "notes": notes}
    if match:
        match.update(data)
    else:
        portfolio.append(data)
    save_portfolio(portfolio)


def remove_holding(ticker: str):
    ticker = ticker.upper().strip()
    save_portfolio([x for x in get_portfolio() if x.get("ticker") != ticker])


def update_holding(ticker: str, shares: float, avg_cost: float):
    ticker = ticker.upper().strip()
    portfolio = get_portfolio()
    for h in portfolio:
        if h.get("ticker") == ticker:
            h["shares"] = float(shares)
            h["avg_cost"] = float(avg_cost)
    save_portfolio(portfolio)


def get_watchlist() -> list[str]:
    return [x.upper() for x in _load(WATCHLIST_FILE)]


def save_watchlist(watchlist: list[str]):
    _save(WATCHLIST_FILE, sorted(set([x.upper() for x in watchlist])))


def add_to_watchlist(ticker: str):
    ticker = ticker.upper().strip()
    watchlist = get_watchlist()
    if ticker and ticker not in watchlist:
        watchlist.append(ticker)
        save_watchlist(watchlist)


def remove_from_watchlist(ticker: str):
    ticker = ticker.upper().strip()
    save_watchlist([x for x in get_watchlist() if x != ticker])


def analyze_portfolio_holdings() -> list[dict]:
    rows = []
    for holding in get_portfolio():
        try:
            analysis = analyze_stock(holding["ticker"])
            shares = float(holding["shares"])
            avg = float(holding["avg_cost"])
            current = analysis.get("current_price") or 0
            value, basis = shares * current, shares * avg
            pnl = value - basis
            rows.append(
                {
                    "Ticker": holding["ticker"],
                    "Shares": shares,
                    "Avg Cost": avg,
                    "Current Price": round(current, 2),
                    "Current Value": round(value, 2),
                    "Cost Basis": round(basis, 2),
                    "P&L $": round(pnl, 2),
                    "P&L %": round((pnl / basis * 100), 2) if basis else 0,
                    "Score": analysis["score"],
                    "Recommendation": analysis["recommendation"],
                    "Sell Signal": analysis.get("sell_recommendation"),
                    "Entry Price": analysis.get("entry_price"),
                    "Target Price": analysis.get("target_price"),
                }
            )
        except Exception:
            continue
    return rows
