from __future__ import annotations

import threading
import time
from functools import lru_cache
from statistics import mean

from modules.data_fetcher import get_news, get_stock_info
from modules.logger import get_logger

logger = get_logger(__name__)

try:
    import streamlit as st

    cache_data = st.cache_data
    cache_resource = st.cache_resource
except Exception:  # pragma: no cover

    def cache_data(ttl: int | None = None):  # type: ignore[misc]
        """TTL-aware, stampede-safe cache fallback for non-Streamlit usage."""

        def decorator(func):
            _cache: dict = {}
            _inflight: dict = {}
            _lock = threading.Lock()

            def wrapper(*args, **kwargs):
                key = (args, tuple(sorted(kwargs.items())))
                while True:
                    now = time.monotonic()
                    with _lock:
                        entry = _cache.get(key)
                        if entry is not None:
                            value, ts = entry
                            if ttl is None or (now - ts) < ttl:
                                return value
                        event = _inflight.get(key)
                        if event is None:
                            ev = threading.Event()
                            _inflight[key] = ev
                            break
                    event.wait(timeout=300)

                try:
                    result = func(*args, **kwargs)
                    with _lock:
                        _cache[key] = (result, time.monotonic())
                    return result
                finally:
                    with _lock:
                        ev = _inflight.pop(key, None)
                    if ev is not None:
                        ev.set()

            return wrapper

        return decorator

    def cache_resource(func):
        return lru_cache(maxsize=1)(func)


@cache_resource
def _load_finbert_pipeline():
    try:
        from transformers import pipeline

        return pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except Exception as exc:
        logger.warning("Unable to load FinBERT model, falling back to neutral headlines: %s", exc)
        return None


@cache_data(ttl=3600)
def _get_market_sentiment_signals(ticker: str) -> dict:
    short_ratio = None
    short_pct_float = None
    put_call_ratio = None

    try:
        # The current Polygon-backed info adapter does not expose short-interest
        # or options-chain metrics yet, so these sentiment signals remain null
        # while preserving the existing return-shape contract.
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


def _analyze_sentiment_uncached(ticker: str) -> dict:
    scored = []
    scored_inputs = []
    for article in get_news(ticker):
        title = article.get("title") or article.get("content", {}).get("title")
        if not title:
            continue
        scored_inputs.append((article, title))

    finbert = _load_finbert_pipeline() if scored_inputs else None
    if finbert is None:
        model_outputs = [{"label": "neutral", "score": 0.0} for _ in scored_inputs]
    else:
        try:
            model_outputs = finbert([title for _, title in scored_inputs])
        except Exception as exc:
            logger.warning("FinBERT inference failed, falling back to neutral headlines: %s", exc)
            model_outputs = [{"label": "neutral", "score": 0.0} for _ in scored_inputs]

    for (article, title), model_output in zip(scored_inputs, model_outputs):
        model_label = str(model_output.get("label", "neutral")).lower()
        confidence = float(model_output.get("score", 0.0) or 0.0)
        score = confidence if model_label == "positive" else -confidence if model_label == "negative" else 0.0
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


# `analyze_sentiment`'s inputs (news headlines) are already cached upstream via
# `get_news` (ttl=900), but FinBERT inference itself re-ran on every call even
# when the underlying cached news hadn't changed. Cache the composed result
# here too (same 15-minute TTL as `get_news`) so repeated calls for the same
# ticker within that window skip both the news re-fetch *and* the FinBERT
# batch-inference pass.
analyze_sentiment = cache_data(ttl=900)(_analyze_sentiment_uncached)
