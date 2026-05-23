from __future__ import annotations

import argparse

from modules.scoring_engine import analyze_stock


def main():
    parser = argparse.ArgumentParser(description="Analyze a US stock ticker")
    parser.add_argument("ticker", help="Ticker symbol, e.g. AAPL")
    args = parser.parse_args()

    result = analyze_stock(args.ticker)
    print("=" * 80)
    print(f"{result['company']} ({result['ticker']})")
    print("=" * 80)
    print(f"Current Price: {result.get('current_price')}")
    print(f"Overall Score: {result['score']}/100")
    print(f"Recommendation: {result['recommendation']}")
    print(f"Time Horizon: {result['time_horizon']}")
    print(f"Entry: {result.get('entry_price')} | Target: {result.get('target_price')} | Stop: {result.get('stop_loss')}")
    tech = result["technical"]
    print("\nTechnical Signals")
    print("-" * 80)
    print(f"Trend: {tech['trend']}")
    print(f"RSI: {tech['indicators'].get('rsi')} ({tech['signals'].get('rsi_signal')})")
    print(f"MACD: {tech['indicators'].get('macd')} / Signal: {tech['indicators'].get('macd_signal')} ({tech['signals'].get('macd_signal')})")
    print("\nFundamentals")
    print("-" * 80)
    metrics = result["fundamentals"].get("metrics", {})
    print(f"Trailing P/E: {metrics.get('trailing_pe')} | Forward P/E: {metrics.get('forward_pe')}")
    print(f"EPS: {metrics.get('eps')} | Revenue Growth: {metrics.get('revenue_growth')}")
    print(f"Debt/Equity: {metrics.get('debt_to_equity')} | ROE: {metrics.get('roe')}")
    print("\nSentiment")
    print("-" * 80)
    sent = result["sentiment"]
    print(f"Sentiment Score: {sent['sentiment_score']} ({sent['sentiment_label']})")
    print("Top Headlines:")
    for headline in sent.get("headlines", [])[:3]:
        print(f"- [{headline['label']}] {headline['headline']}")
    print("\nDetected Patterns")
    print("-" * 80)
    for pattern in tech.get("patterns", []):
        print(f"- {pattern['name']} ({pattern['implication']}): {pattern['description']}")
    print("\nScore Breakdown")
    print("-" * 80)
    for k, v in result["score_breakdown"].items():
        print(f"{k.title()}: {v}")
    print("=" * 80)


if __name__ == "__main__":
    main()
