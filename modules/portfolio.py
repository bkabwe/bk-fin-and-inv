from __future__ import annotations

import json
from pathlib import Path

from modules.scoring_engine import analyze_stock, get_price_projections

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
            projections = get_price_projections(holding["ticker"])
            shares = float(holding["shares"])
            avg = float(holding["avg_cost"])
            current = analysis.get("current_price") or 0
            value, basis = shares * current, shares * avg
            pnl = value - basis
            recommendation_to_sell_at = projections.get("recommendation_to_sell_at")
            if current and avg and current < avg:
                recommendation_to_sell_at = f"Hold for recovery to break-even at ${avg:.2f} before considering sell."
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
                    "short_term_target": projections.get("short_term_target"),
                    "short_term_upside": projections.get("short_term_upside"),
                    "short_term_basis": projections.get("short_term_basis"),
                    "medium_term_target": projections.get("medium_term_target"),
                    "medium_term_upside": projections.get("medium_term_upside"),
                    "medium_term_basis": projections.get("medium_term_basis"),
                    "long_term_target": projections.get("long_term_target"),
                    "long_term_upside": projections.get("long_term_upside"),
                    "long_term_basis": projections.get("long_term_basis"),
                    "recommendation_to_sell_at": recommendation_to_sell_at,
                }
            )
        except Exception:
            continue
    return rows
