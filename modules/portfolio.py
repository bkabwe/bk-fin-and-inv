from __future__ import annotations

import json
import os
import tempfile
from datetime import date, timedelta
from pathlib import Path

from modules.logger import get_logger
from modules.polygon_client import get_reference_splits
from modules.scoring_engine import analyze_stock, get_price_projections
from modules.validators import sanitize_ticker

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PORTFOLIO_FILE = DATA_DIR / "portfolio.json"
WATCHLIST_FILE = DATA_DIR / "watchlist.json"
logger = get_logger(__name__)


def _ensure():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not PORTFOLIO_FILE.exists():
        _atomic_write(PORTFOLIO_FILE, [])
    if not WATCHLIST_FILE.exists():
        _atomic_write(WATCHLIST_FILE, [])


def _atomic_write(path: Path, data) -> None:
    """Write JSON atomically using temp file + rename to prevent corruption."""
    dir_path = path.parent
    dir_path.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=dir_path,
        delete=False,
        suffix=".tmp",
    ) as tmp:
        json.dump(data, tmp, indent=2)
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def _load(path: Path):
    _ensure()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save(path: Path, payload):
    _ensure()
    _atomic_write(path, payload)


def get_portfolio() -> list[dict]:
    return _load(PORTFOLIO_FILE)


def save_portfolio(portfolio: list[dict]):
    _save(PORTFOLIO_FILE, portfolio)


def add_holding(ticker: str, shares: float, avg_cost: float, date_purchased: str, notes: str = ""):
    ticker = sanitize_ticker(ticker)
    portfolio = get_portfolio()
    match = next((x for x in portfolio if x.get("ticker") == ticker), None)
    data = {"ticker": ticker, "shares": float(shares), "avg_cost": float(avg_cost), "date_purchased": date_purchased, "notes": notes}
    if match:
        match.update(data)
        logger.info("Updated holding %s in portfolio", ticker)
    else:
        portfolio.append(data)
        logger.info("Added holding %s to portfolio", ticker)
    save_portfolio(portfolio)


def remove_holding(ticker: str):
    ticker = sanitize_ticker(ticker)
    save_portfolio([x for x in get_portfolio() if x.get("ticker") != ticker])
    logger.info("Removed holding %s from portfolio", ticker)


def update_holding(ticker: str, shares: float, avg_cost: float):
    ticker = sanitize_ticker(ticker)
    portfolio = get_portfolio()
    for h in portfolio:
        if h.get("ticker") == ticker:
            h["shares"] = float(shares)
            h["avg_cost"] = float(avg_cost)
    save_portfolio(portfolio)
    logger.info("Updated holding %s shares/cost in portfolio", ticker)


def get_watchlist() -> list[str]:
    return [x.upper() for x in _load(WATCHLIST_FILE)]


def save_watchlist(watchlist: list[str]):
    _save(WATCHLIST_FILE, sorted(set([x.upper() for x in watchlist])))


def add_to_watchlist(ticker: str):
    ticker = sanitize_ticker(ticker)
    watchlist = get_watchlist()
    if ticker and ticker not in watchlist:
        watchlist.append(ticker)
        save_watchlist(watchlist)


def remove_from_watchlist(ticker: str):
    ticker = sanitize_ticker(ticker)
    save_watchlist([x for x in get_watchlist() if x != ticker])


def detect_split_since_purchase(ticker: str, date_purchased: str) -> dict:
    """Detect whether a stock split or reverse split has occurred since the purchase date.

    Returns a dict with:
    - ``split_detected`` (bool): True if any split event occurred after purchase.
    - ``split_factor`` (float | None): Cumulative split factor since purchase (>1 = forward, <1 = reverse).
    - ``events`` (list[dict]): Individual split events with date and ratio.
    - ``warning`` (str | None): Human-readable warning if a split was detected.
    """
    result: dict = {"split_detected": False, "split_factor": None, "events": [], "warning": None}
    try:
        execution_date_gte = None
        if date_purchased:
            try:
                purchased_day = date.fromisoformat(str(date_purchased).split("T", 1)[0])
                execution_date_gte = (purchased_day + timedelta(days=1)).isoformat()
            except Exception:
                raw = str(date_purchased).split("T", 1)[0]
                try:
                    fallback_day = date.fromisoformat(raw)
                    execution_date_gte = (fallback_day + timedelta(days=1)).isoformat()
                except Exception:
                    execution_date_gte = raw
        splits = get_reference_splits(ticker, execution_date_gte=execution_date_gte)
        if not splits:
            return result
        cumulative = 1.0
        events = []
        for event in splits:
            split_from = float(event.get("split_from") or 0)
            split_to = float(event.get("split_to") or 0)
            if split_from <= 0 or split_to <= 0:
                continue
            ratio = split_to / split_from
            cumulative *= ratio
            events.append({"date": str(event.get("execution_date") or ""), "ratio": float(ratio)})
        if not events:
            return result
        result["split_detected"] = True
        result["split_factor"] = round(cumulative, 6)
        result["events"] = events
        if cumulative > 1:
            result["warning"] = (
                f"⚠️ {ticker} has undergone a {cumulative:.4g}x forward split since {date_purchased}. "
                "Your share count and cost basis may be outdated — please update this holding."
            )
        else:
            result["warning"] = (
                f"⚠️ {ticker} has undergone a {cumulative:.4g}x reverse split since {date_purchased}. "
                "Your share count and cost basis may be outdated — please update this holding."
            )
    except Exception as exc:
        logger.warning("Split detection failed for %s: %s", ticker, exc)
    return result


def analyze_portfolio_holdings() -> list[dict]:
    rows = []
    for holding in get_portfolio():
        try:
            analysis = analyze_stock(holding["ticker"], avg_cost=float(holding["avg_cost"]))
            projections = analysis.get("projections") or get_price_projections(holding["ticker"])
            shares = float(holding["shares"])
            avg = float(holding["avg_cost"])
            current = analysis.get("current_price") or 0
            value, basis = shares * current, shares * avg
            pnl = value - basis

            # Detect splits/reverse splits since purchase date and warn user.
            split_info = detect_split_since_purchase(holding["ticker"], holding.get("date_purchased", ""))
            split_warning = split_info.get("warning")

            recommendation_to_sell_at = projections.get("recommendation_to_sell_at")
            if current and avg and current < avg:
                recommendation_to_sell_at = (
                    f"Hold for recovery to break-even at ${avg:.2f} before considering sell."
                )

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
                    # Full projection dict for in-card rendering
                    "Projection": projections,
                    "Projection Models": ", ".join(projections.get("models_used", [])),
                    "Projection Data Quality": projections.get("data_quality", "Limited"),
                    # Flat projection keys used by the expander section
                    "short_term_target": projections.get("short_term_target"),
                    "short_term_low": projections.get("short_term_low"),
                    "short_term_high": projections.get("short_term_high"),
                    "short_term_upside": projections.get("short_term_upside"),
                    "short_term_basis": projections.get("short_term_basis"),
                    "medium_term_target": projections.get("medium_term_target"),
                    "medium_term_low": projections.get("medium_term_low"),
                    "medium_term_high": projections.get("medium_term_high"),
                    "medium_term_upside": projections.get("medium_term_upside"),
                    "medium_term_basis": projections.get("medium_term_basis"),
                    "long_term_target": projections.get("long_term_target"),
                    "long_term_low": projections.get("long_term_low"),
                    "long_term_high": projections.get("long_term_high"),
                    "long_term_upside": projections.get("long_term_upside"),
                    "long_term_basis": projections.get("long_term_basis"),
                    "recommendation_to_sell_at": recommendation_to_sell_at,
                    "split_warning": split_warning,
                    "split_info": split_info,
                }
            )
        except Exception as exc:
            ticker = str(holding.get("ticker", "UNKNOWN"))
            logger.error("Failed to analyze holding %s", ticker, exc_info=True)
            rows.append(
                {
                    "Ticker": ticker,
                    "Shares": float(holding.get("shares", 0) or 0),
                    "Avg Cost": float(holding.get("avg_cost", 0) or 0),
                    "Current Price": None,
                    "Current Value": 0.0,
                    "Cost Basis": 0.0,
                    "P&L $": 0.0,
                    "P&L %": 0.0,
                    "Score": 0,
                    "Recommendation": "⚠️ Analysis Failed",
                    "Error": str(exc),
                    "Sell Signal": "⚠️ Analysis Failed",
                    "Entry Price": None,
                    "Target Price": None,
                    "Projection": {},
                    "Projection Models": "",
                    "Projection Data Quality": "Limited",
                    "short_term_target": None,
                    "short_term_low": None,
                    "short_term_high": None,
                    "short_term_upside": None,
                    "short_term_basis": None,
                    "medium_term_target": None,
                    "medium_term_low": None,
                    "medium_term_high": None,
                    "medium_term_upside": None,
                    "medium_term_basis": None,
                    "long_term_target": None,
                    "long_term_low": None,
                    "long_term_high": None,
                    "long_term_upside": None,
                    "long_term_basis": None,
                    "recommendation_to_sell_at": "Unable to compute recommendation due to analysis error.",
                    "split_warning": None,
                    "split_info": {},
                }
            )
    return rows
