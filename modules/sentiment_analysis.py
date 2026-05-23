from __future__ import annotations

from statistics import mean

from textblob import TextBlob

from modules.data_fetcher import get_news


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
                "publisher": article.get("publisher") or article.get("content", {}).get("provider", {}).get("displayName"),
                "link": article.get("link") or article.get("content", {}).get("canonicalUrl", {}).get("url"),
            }
        )
    sentiment_score = mean([x["score"] for x in scored]) if scored else 0.0
    label = "Positive" if sentiment_score > 0.1 else "Negative" if sentiment_score < -0.1 else "Neutral"
    return {"sentiment_score": round(float(sentiment_score), 3), "sentiment_label": label, "headlines": scored}
