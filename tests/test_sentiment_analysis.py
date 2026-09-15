from __future__ import annotations

import importlib
import sys
import types
import unittest
from unittest.mock import Mock, patch


def _load_sentiment_analysis_module():
    fake_data_fetcher = types.SimpleNamespace(get_news=lambda _ticker: [], get_stock_info=lambda _ticker: {})
    with patch.dict(sys.modules, {"modules.data_fetcher": fake_data_fetcher}):
        module = importlib.import_module("modules.sentiment_analysis")
        return importlib.reload(module)


class SentimentAnalysisTests(unittest.TestCase):
    def test_finbert_label_mapping_and_batch_call(self):
        sentiment_analysis = _load_sentiment_analysis_module()
        news = [
            {"title": "Company beats expectations"},
            {"title": "Shares plunged after guidance cut"},
            {"title": "Analysts maintain outlook"},
        ]
        pipeline_mock = Mock(
            return_value=[
                {"label": "positive", "score": 0.91},
                {"label": "negative", "score": 0.82},
                {"label": "neutral", "score": 0.77},
            ]
        )

        with patch.object(sentiment_analysis, "get_news", return_value=news):
            with patch.object(sentiment_analysis, "_load_finbert_pipeline", return_value=pipeline_mock):
                with patch.object(
                    sentiment_analysis,
                    "_get_market_sentiment_signals",
                    return_value={"short_ratio": None, "short_pct_float": None, "put_call_ratio": None},
                ):
                    result = sentiment_analysis.analyze_sentiment("AAPL")

        pipeline_mock.assert_called_once_with([item["title"] for item in news])
        self.assertEqual([row["score"] for row in result["headlines"]], [0.91, -0.82, 0.0])
        self.assertEqual([row["label"] for row in result["headlines"]], ["Positive", "Negative", "Neutral"])
        self.assertEqual(result["sentiment_score"], 0.03)
        self.assertEqual(result["sentiment_label"], "Neutral")

    def test_sentiment_score_is_clamped_after_adjustments(self):
        sentiment_analysis = _load_sentiment_analysis_module()
        with patch.object(sentiment_analysis, "get_news", return_value=[{"title": "Bullish outlook"}]):
            with patch.object(
                sentiment_analysis,
                "_load_finbert_pipeline",
                return_value=Mock(return_value=[{"label": "positive", "score": 1.0}]),
            ):
                with patch.object(
                    sentiment_analysis,
                    "_get_market_sentiment_signals",
                    return_value={"short_ratio": None, "short_pct_float": None, "put_call_ratio": 0.6},
                ):
                    bullish = sentiment_analysis.analyze_sentiment("MSFT")

        self.assertEqual(bullish["sentiment_score"], 1.0)

        with patch.object(sentiment_analysis, "get_news", return_value=[{"title": "Bearish outlook"}]):
            with patch.object(
                sentiment_analysis,
                "_load_finbert_pipeline",
                return_value=Mock(return_value=[{"label": "negative", "score": 1.0}]),
            ):
                with patch.object(
                    sentiment_analysis,
                    "_get_market_sentiment_signals",
                    return_value={"short_ratio": 12.0, "short_pct_float": None, "put_call_ratio": 1.6},
                ):
                    bearish = sentiment_analysis.analyze_sentiment("TSLA")

        self.assertEqual(bearish["sentiment_score"], -1.0)

    def test_model_load_failure_falls_back_to_neutral(self):
        sentiment_analysis = _load_sentiment_analysis_module()
        news = [{"title": "Headline one"}, {"title": "Headline two"}]
        with patch.object(sentiment_analysis, "get_news", return_value=news):
            with patch.object(sentiment_analysis, "_load_finbert_pipeline", return_value=None):
                with patch.object(
                    sentiment_analysis,
                    "_get_market_sentiment_signals",
                    return_value={"short_ratio": None, "short_pct_float": None, "put_call_ratio": None},
                ):
                    result = sentiment_analysis.analyze_sentiment("NVDA")

        self.assertEqual(result["sentiment_score"], 0.0)
        self.assertEqual(result["sentiment_label"], "Neutral")
        self.assertEqual([row["score"] for row in result["headlines"]], [0.0, 0.0])
        self.assertEqual([row["label"] for row in result["headlines"]], ["Neutral", "Neutral"])


if __name__ == "__main__":
    unittest.main()
