from __future__ import annotations

import importlib
import unittest
from unittest.mock import Mock, patch


def _load_sentiment_analysis_module():
    # `modules.sentiment_analysis` is a process-wide singleton once imported;
    # each test below isolates `get_news`/`_load_finbert_pipeline`/
    # `_get_market_sentiment_signals` via `patch.object(...)` on the returned
    # module for the scope of that test, so no module reload is required.
    #
    # NOTE: an earlier version of this helper swapped a fake module into
    # `sys.modules["modules.data_fetcher"]` and called `importlib.reload()`
    # to get an "isolated" module reference. That combination triggers a
    # genuine bug in Streamlit's cache-write pickling path (a stale
    # `CachedResult` class reference) once `analyze_sentiment` itself became
    # `@cache_data`-wrapped, so it was replaced with a plain import.
    module = importlib.import_module("modules.sentiment_analysis")
    # `analyze_sentiment` is wrapped in a ttl-cache (see modules/sentiment_analysis.py);
    # clear any residual cached entries from a previous test so each test
    # starts from a clean cache.
    if hasattr(module.analyze_sentiment, "clear"):
        module.analyze_sentiment.clear()
    return module


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

        with (
            patch.object(sentiment_analysis, "get_news", return_value=news),
            patch.object(sentiment_analysis, "_load_finbert_pipeline", return_value=pipeline_mock),
            patch.object(
                sentiment_analysis,
                "_get_market_sentiment_signals",
                return_value={"short_ratio": None, "short_pct_float": None, "put_call_ratio": None},
            ),
        ):
            result = sentiment_analysis.analyze_sentiment("AAPL")

        pipeline_mock.assert_called_once_with([item["title"] for item in news])
        self.assertEqual([row["score"] for row in result["headlines"]], [0.91, -0.82, 0.0])
        self.assertEqual([row["label"] for row in result["headlines"]], ["Positive", "Negative", "Neutral"])
        self.assertEqual(result["sentiment_score"], 0.03)
        self.assertEqual(result["sentiment_label"], "Neutral")

    def test_sentiment_score_is_clamped_after_adjustments(self):
        sentiment_analysis = _load_sentiment_analysis_module()
        with (
            patch.object(sentiment_analysis, "get_news", return_value=[{"title": "Bullish outlook"}]),
            patch.object(
                sentiment_analysis,
                "_load_finbert_pipeline",
                return_value=Mock(return_value=[{"label": "positive", "score": 1.0}]),
            ),
            patch.object(
                sentiment_analysis,
                "_get_market_sentiment_signals",
                return_value={"short_ratio": None, "short_pct_float": None, "put_call_ratio": 0.6},
            ),
        ):
            bullish = sentiment_analysis.analyze_sentiment("MSFT")

        self.assertEqual(bullish["sentiment_score"], 1.0)

        with (
            patch.object(sentiment_analysis, "get_news", return_value=[{"title": "Bearish outlook"}]),
            patch.object(
                sentiment_analysis,
                "_load_finbert_pipeline",
                return_value=Mock(return_value=[{"label": "negative", "score": 1.0}]),
            ),
            patch.object(
                sentiment_analysis,
                "_get_market_sentiment_signals",
                return_value={"short_ratio": 12.0, "short_pct_float": None, "put_call_ratio": 1.6},
            ),
        ):
            bearish = sentiment_analysis.analyze_sentiment("TSLA")

        self.assertEqual(bearish["sentiment_score"], -1.0)

    def test_model_load_failure_falls_back_to_neutral(self):
        sentiment_analysis = _load_sentiment_analysis_module()
        news = [{"title": "Headline one"}, {"title": "Headline two"}]
        with (
            patch.object(sentiment_analysis, "get_news", return_value=news),
            patch.object(sentiment_analysis, "_load_finbert_pipeline", return_value=None),
            patch.object(
                sentiment_analysis,
                "_get_market_sentiment_signals",
                return_value={"short_ratio": None, "short_pct_float": None, "put_call_ratio": None},
            ),
        ):
            result = sentiment_analysis.analyze_sentiment("NVDA")

        self.assertEqual(result["sentiment_score"], 0.0)
        self.assertEqual(result["sentiment_label"], "Neutral")
        self.assertEqual([row["score"] for row in result["headlines"]], [0.0, 0.0])
        self.assertEqual([row["label"] for row in result["headlines"]], ["Neutral", "Neutral"])


class FinbertPipelineLoaderTests(unittest.TestCase):
    """Exercises `_load_finbert_pipeline` itself (unmocked), rather than
    stubbing it out as the tests in `SentimentAnalysisTests` do."""

    @classmethod
    def setUpClass(cls):
        # `transformers` exposes optional-backend attributes (like `pipeline`)
        # via a self-replacing lazy module: the *first* real access to
        # `transformers.pipeline` swaps `sys.modules["transformers"]` for a
        # new, stabilized module object. If `patch("transformers.pipeline", ...)`
        # runs before that first access, it patches the soon-to-be-discarded
        # pre-swap object, so a later `from transformers import pipeline`
        # (resolved against the new, swapped-in module) never sees the patch.
        # Force that one-time swap here, before any test below patches it.
        import transformers

        transformers.pipeline  # noqa: B018 -- access triggers transformers' lazy-module swap

    def setUp(self):
        self.sentiment_analysis = _load_sentiment_analysis_module()
        # `_load_finbert_pipeline` is wrapped in `cache_resource` (a
        # `st.cache_resource`/`lru_cache(maxsize=1)` singleton), so its result
        # from an earlier test would otherwise leak into this one. Clear
        # before and after so this test class doesn't affect, or get affected
        # by, other tests that exercise `analyze_sentiment`.
        self._clear_finbert_cache()
        self.addCleanup(self._clear_finbert_cache)

    def _clear_finbert_cache(self):
        loader = self.sentiment_analysis._load_finbert_pipeline
        if hasattr(loader, "clear"):
            loader.clear()
        elif hasattr(loader, "cache_clear"):
            loader.cache_clear()

    def test_real_load_failure_returns_none(self):
        # Let `_load_finbert_pipeline` run for real; only the underlying
        # `transformers.pipeline` constructor is forced to fail, exercising
        # the actual try/except fallback path instead of a mocked loader.
        with patch("transformers.pipeline", side_effect=RuntimeError("model download failed")) as pipeline_ctor:
            result = self.sentiment_analysis._load_finbert_pipeline()

        self.assertIsNone(result)
        pipeline_ctor.assert_called_once()

    def test_successful_load_is_reused_as_a_singleton(self):
        # A successful load should only construct the pipeline once; the
        # second call must reuse the cached instance rather than re-invoking
        # `transformers.pipeline` again.
        fake_pipeline = Mock(name="finbert_pipeline")
        with patch("transformers.pipeline", return_value=fake_pipeline) as pipeline_ctor:
            first = self.sentiment_analysis._load_finbert_pipeline()
            second = self.sentiment_analysis._load_finbert_pipeline()

        pipeline_ctor.assert_called_once()
        self.assertIs(first, fake_pipeline)
        self.assertIs(second, fake_pipeline)


if __name__ == "__main__":
    unittest.main()
