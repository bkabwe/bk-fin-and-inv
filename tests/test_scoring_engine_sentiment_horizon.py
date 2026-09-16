from __future__ import annotations

import unittest

from modules import scoring_engine


class FundamentalSentimentWeightsTests(unittest.TestCase):
    """Sentiment's ensemble weight should decay for longer investment
    horizons, with fundamentals picking up the freed-up weight so the
    combined fundamental+sentiment pool stays fixed at 50 points."""

    def test_default_and_short_term_use_the_original_flat_split(self):
        self.assertEqual(scoring_engine._fundamental_sentiment_weights(None), (30, 20))
        self.assertEqual(scoring_engine._fundamental_sentiment_weights("short_term"), (30, 20))

    def test_medium_term_decays_sentiment_and_grows_fundamental(self):
        fundamental_weight, sentiment_weight = scoring_engine._fundamental_sentiment_weights("medium_term")
        self.assertEqual((fundamental_weight, sentiment_weight), (36, 14))
        self.assertLess(sentiment_weight, 20)
        self.assertGreater(fundamental_weight, 30)

    def test_long_term_decays_sentiment_further_than_medium_term(self):
        medium_fundamental, medium_sentiment = scoring_engine._fundamental_sentiment_weights("medium_term")
        long_fundamental, long_sentiment = scoring_engine._fundamental_sentiment_weights("long_term")
        self.assertEqual((long_fundamental, long_sentiment), (42, 8))
        self.assertLess(long_sentiment, medium_sentiment)
        self.assertGreater(long_fundamental, medium_fundamental)

    def test_unknown_horizon_falls_back_to_default_split(self):
        self.assertEqual(scoring_engine._fundamental_sentiment_weights("unknown_horizon"), (30, 20))

    def test_pool_stays_fixed_at_50_points_across_all_horizons(self):
        for horizon in (None, "short_term", "medium_term", "long_term"):
            fundamental_weight, sentiment_weight = scoring_engine._fundamental_sentiment_weights(horizon)
            self.assertEqual(fundamental_weight + sentiment_weight, 50)


if __name__ == "__main__":
    unittest.main()
