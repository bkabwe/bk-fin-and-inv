from __future__ import annotations

from functools import lru_cache
from statistics import mean

from textblob import TextBlob

from modules.data_fetcher import get_news, get_stock_info
from modules.logger import get_logger

logger = get_logger(__name__)

try:
    import streamlit as st

    cache_data = st.cache_data
except Exception:  # pragma: no cover

    def cache_data(ttl: int | None = None):
        def decorator(func):
            return lru_cache(maxsize=128)(func)

        return decorator


@cache_data(ttl=3600)
def _get_market_sentiment_signals(ticker: str) -> dict:
    short_ratio = None
    short_pct_float = None
    put_call_ratio = None

    try:
        # Polygon Starter does not include legacy options-chain fields used
        # previously; we keep these fields nullable so sentiment scoring degrades
        # gracefully while preserving return-shape compatibility.
        info = get_stock_info(ticker) or {}
        short_ratio = info.get("shortRatio")
        short_pct_float = info.get("shortPercentOfFloat")
    except Exception as exc:
        logger.warning("Unable to fetch market sentiment signals for %s: %s", ticker, exc)

    return {
        "short_ratio": round(float(short_ratio), 3) if short_ratio is not None else None,
        "short_pct_float": round(float(short_pct_float), 3) if short_pct_float is not None else None,
        "put_call_ratio": round(float(put_call_ratio), 3) if put_call_ratio is not None else None,
    }


def analyze_sentiment(ticker: str) -> dict:
    scored = []
    for article in get_news(ticker):
        title = article.get("title") or article.get("content", {}).get("title")
        if not title:
            continue
        score = TextBlob(title).sentiment.polarity
        scored.append(
            {
                "headline": title,
                "score": round(score, 3),
                "label": "Positive" if score > 0.1 else "Negative" if score < -0.1 else "Neutral",
                "publisher": article.get("publisher") or article.get("source") or article.get("content", {}).get("provider", {}).get("displayName"),
                "link": article.get("article_url") or article.get("link") or article.get("content", {}).get("canonicalUrl", {}).get("url"),
            }
        )

    sentiment_score = float(mean([x["score"] for x in scored])) if scored else 0.0
    extra = _get_market_sentiment_signals(ticker)

    put_call_ratio = extra.get("put_call_ratio")
    short_ratio = extra.get("short_ratio")

    if put_call_ratio is not None:
        if put_call_ratio > 1.5:
            sentiment_score -= 0.1
        elif put_call_ratio < 0.7:
            sentiment_score += 0.1

    if short_ratio is not None and short_ratio > 10:
        sentiment_score -= 0.1

    sentiment_score = max(-1.0, min(1.0, sentiment_score))
    label = "Positive" if sentiment_score > 0.1 else "Negative" if sentiment_score < -0.1 else "Neutral"

    options_interpretation = "Neutral"
    if put_call_ratio is not None:
        if put_call_ratio > 1.5:
            options_interpretation = "Bearish options positioning"
        elif put_call_ratio < 0.7:
            options_interpretation = "Bullish options positioning"

    logger.info(
        "Sentiment analysis complete for %s | score=%.3f | put_call=%s | short_ratio=%s",
        ticker.upper(),
        sentiment_score,
        put_call_ratio,
        short_ratio,
    )

    return {
        "sentiment_score": round(float(sentiment_score), 3),
        "sentiment_label": label,
        "headlines": scored,
        "short_ratio": extra.get("short_ratio"),
        "short_pct_float": extra.get("short_pct_float"),
        "put_call_ratio": extra.get("put_call_ratio"),
        "options_sentiment": options_interpretation,
    }
