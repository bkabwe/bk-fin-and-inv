from __future__ import annotations

import unittest
import sys
import types
from unittest.mock import patch

import requests

if "pandas" not in sys.modules:
    fake_pandas = types.ModuleType("pandas")

    class _DataFrame:  # pragma: no cover - import-time shim only
        pass

    fake_pandas.DataFrame = _DataFrame
    fake_pandas.to_datetime = lambda *args, **kwargs: None  # type: ignore[assignment]
    fake_pandas.isna = lambda value: False  # type: ignore[assignment]
    sys.modules["pandas"] = fake_pandas

from modules import polygon_client, sec_edgar_client


def _http_error(status_code: int, retry_after: str | None = None) -> requests.exceptions.HTTPError:
    response = requests.Response()
    response.status_code = status_code
    response.url = "https://example.test/resource"
    if retry_after is not None:
        response.headers["Retry-After"] = retry_after
    return requests.exceptions.HTTPError(f"status={status_code}", response=response)


class RetryBehaviorTests(unittest.TestCase):
    def test_polygon_does_not_retry_non_retryable_403(self):
        attempts = {"count": 0}

        def _fetch():
            attempts["count"] += 1
            raise _http_error(403)

        with patch("modules.polygon_client.time.sleep") as sleep_mock:
            with self.assertRaises(requests.exceptions.HTTPError):
                polygon_client._fetch_with_retry(_fetch, max_retries=3, base_delay=0.01)
        self.assertEqual(attempts["count"], 1)
        sleep_mock.assert_not_called()

    def test_polygon_retries_429_with_retry_after(self):
        attempts = {"count": 0}

        def _fetch():
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise _http_error(429, retry_after="2")
            return {"ok": True}

        with patch("modules.polygon_client.time.sleep") as sleep_mock:
            payload = polygon_client._fetch_with_retry(_fetch, max_retries=3, base_delay=0.01)
        self.assertEqual(payload, {"ok": True})
        self.assertEqual(attempts["count"], 2)
        sleep_mock.assert_called_once_with(2.0)


class SecFundamentalsAdapterTests(unittest.TestCase):
    def test_sec_adapter_builds_expected_ratio_fields(self):
        facts = {
            "facts": {
                "us-gaap": {
                    "SalesRevenueNet": {
                        "units": {
                            "USD": [
                                {"end": "2024-12-31", "val": 1200, "form": "10-K", "filed": "2025-02-01"},
                                {"end": "2023-12-31", "val": 1000, "form": "10-K", "filed": "2024-02-01"},
                            ]
                        }
                    },
                    "NetIncomeLoss": {
                        "units": {
                            "USD": [
                                {"end": "2024-12-31", "val": 200, "form": "10-K", "filed": "2025-02-01"},
                                {"end": "2023-12-31", "val": 160, "form": "10-K", "filed": "2024-02-01"},
                            ]
                        }
                    },
                    "Liabilities": {"units": {"USD": [{"end": "2024-12-31", "val": 500, "form": "10-K"}]}},
                    "StockholdersEquity": {"units": {"USD": [{"end": "2024-12-31", "val": 1000, "form": "10-K"}]}},
                    "EarningsPerShareDiluted": {"units": {"USD/shares": [{"end": "2024-12-31", "val": 5, "form": "10-K"}]}},
                    "CommonStockSharesOutstanding": {"units": {"shares": [{"end": "2024-12-31", "val": 40, "form": "10-K"}]}},
                }
            }
        }

        info = sec_edgar_client.build_fundamentals_info_adapter(facts, current_price=100)

        self.assertAlmostEqual(info["trailingPE"], 20.0)
        self.assertAlmostEqual(info["trailingEps"], 5.0)
        self.assertAlmostEqual(info["returnOnEquity"], 0.2)
        self.assertAlmostEqual(info["debtToEquity"], 50.0)
        self.assertAlmostEqual(info["revenueGrowth"], 0.2)
        self.assertAlmostEqual(info["earningsGrowth"], 0.25)
        self.assertAlmostEqual(info["sharesOutstanding"], 40.0)
        self.assertIsNone(info["forwardPE"])
        self.assertIsNone(info["forwardEps"])


if __name__ == "__main__":
    unittest.main()
