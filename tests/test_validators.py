from __future__ import annotations

import unittest

from modules.validators import (
    MAX_CUSTOM_TICKERS,
    MAX_TICKER_LENGTH,
    sanitize_ticker,
    sanitize_ticker_list,
)


class SanitizeTickerTests(unittest.TestCase):
    def test_uppercases_and_strips_whitespace(self):
        self.assertEqual(sanitize_ticker("  aapl  "), "AAPL")

    def test_accepts_hyphen_and_dot(self):
        self.assertEqual(sanitize_ticker("brk-b"), "BRK-B")
        self.assertEqual(sanitize_ticker("brk.b"), "BRK.B")

    def test_accepts_alphanumeric(self):
        self.assertEqual(sanitize_ticker("bf.b1"), "BF.B1")

    def test_truncates_to_max_length(self):
        long_ticker = "a" * (MAX_TICKER_LENGTH + 5)
        result = sanitize_ticker(long_ticker)
        self.assertEqual(len(result), MAX_TICKER_LENGTH)
        self.assertEqual(result, "A" * MAX_TICKER_LENGTH)

    def test_empty_string_raises(self):
        with self.assertRaises(ValueError):
            sanitize_ticker("")

    def test_whitespace_only_raises(self):
        with self.assertRaises(ValueError):
            sanitize_ticker("   ")

    def test_rejects_invalid_characters(self):
        for bad in ["AAPL!", "AA PL", "AA/PL", "AA;DROP TABLE", "<script>"]:
            with self.assertRaises(ValueError):
                sanitize_ticker(bad)

    def test_non_string_input_is_coerced(self):
        self.assertEqual(sanitize_ticker(123), "123")

    def test_error_message_includes_offending_ticker(self):
        with self.assertRaises(ValueError) as ctx:
            sanitize_ticker("AA/PL")
        self.assertIn("AA/PL", str(ctx.exception))


class SanitizeTickerListTests(unittest.TestCase):
    def test_sanitizes_each_ticker(self):
        result = sanitize_ticker_list([" aapl ", "msft", "brk.b"])
        self.assertEqual(result, ["AAPL", "MSFT", "BRK.B"])

    def test_skips_invalid_tickers_silently(self):
        result = sanitize_ticker_list(["AAPL", "AA/PL", "", "MSFT"])
        self.assertEqual(result, ["AAPL", "MSFT"])

    def test_empty_list_returns_empty_list(self):
        self.assertEqual(sanitize_ticker_list([]), [])

    def test_all_invalid_returns_empty_list(self):
        self.assertEqual(sanitize_ticker_list(["", "!!!", "///"]), [])

    def test_truncates_to_max_custom_tickers(self):
        tickers = [f"T{i}" for i in range(MAX_CUSTOM_TICKERS + 50)]
        result = sanitize_ticker_list(tickers)
        self.assertLessEqual(len(result), MAX_CUSTOM_TICKERS)
        self.assertEqual(result, tickers[:MAX_CUSTOM_TICKERS])


if __name__ == "__main__":
    unittest.main()
