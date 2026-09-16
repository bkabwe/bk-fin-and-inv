from __future__ import annotations

import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from modules import prediction_tracker
from scripts import scan_email_report

REPO_ROOT = Path(__file__).resolve().parents[1]
STOCK_ANALYSIS_PAGE = REPO_ROOT / "pages" / "4_Stock_Analysis.py"


class PredictionTrackerExcursionTests(unittest.TestCase):
    def test_compute_max_price_since_scan_uses_high_prices_within_requested_window(self):
        frame = pd.DataFrame(
            {
                "High": [100.0, 112.5, 108.0, 121.0],
                "Close": [99.0, 110.0, 107.0, 118.0],
            },
            index=pd.to_datetime(["2026-08-20", "2026-08-21", "2026-08-22", "2026-08-25"]),
        )

        with patch("modules.data_fetcher.get_stock_data", return_value=frame):
            value = prediction_tracker.compute_max_price_since_scan("AAPL", "2026-08-21", "2026-08-22")

        self.assertEqual(value, 112.5)


class ScanEmailReportTests(unittest.TestCase):
    def test_run_profit_opportunities_scan_records_predictions_for_ranked_results(self):
        sample_frame = pd.DataFrame({"Close": [10.0, 11.0], "High": [10.5, 11.5]}, index=pd.date_range("2026-01-01", periods=2, freq="D"))

        def _analysis_for(ticker, **_kwargs):
            upside = 25.0 if ticker == "MSFT" else 18.0
            target = 125.0 if ticker == "MSFT" else 118.0
            return {
                "company": f"{ticker} Inc.",
                "score": 82 if ticker == "MSFT" else 71,
                "current_price": 100.0,
                "projections": {
                    "current_price": 100.0,
                    "short_term_target": target,
                    "short_term_low": target * 0.95,
                    "short_term_high": target * 1.05,
                    "short_term_upside": upside,
                    "short_term_basis": "ARIMA, Trend",
                    "data_quality": "High" if ticker == "MSFT" else "Medium",
                },
            }

        # run_profit_opportunities_scan() delegates the actual per-ticker
        # fast-screen/analysis/data-fetch calls to modules.profit_opportunities
        # (shared with the Streamlit page and the FastAPI route), so patch
        # targets live there rather than on scripts.scan_email_report.
        with patch("modules.profit_opportunities.get_stock_data", return_value=sample_frame):
            with patch("modules.profit_opportunities.fast_screen_score", return_value=(42, None)):
                with patch("modules.profit_opportunities.analyze_stock", side_effect=_analysis_for):
                    with patch("scripts.scan_email_report.record_predictions_from_scan", return_value=2) as record_mock:
                        results, stats = scan_email_report.run_profit_opportunities_scan(["AAPL", "MSFT"], "short_term")

        record_mock.assert_called_once()
        self.assertEqual(record_mock.call_args.kwargs["horizon"], "short_term")
        self.assertEqual(record_mock.call_args.kwargs["source"], "profit_opportunities")
        recorded_rows = record_mock.call_args.args[0]
        self.assertEqual([row["Ticker"] for row in recorded_rows], ["MSFT", "AAPL"])
        self.assertEqual(list(results["Ticker"]), ["MSFT", "AAPL"])
        self.assertEqual(stats["recorded_count"], 2)


class _DummyContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def metric(self, *args, **kwargs):
        return None

    def write(self, *args, **kwargs):
        return None


class _DummyFigure:
    def add_trace(self, *args, **kwargs):
        return None

    def add_hline(self, *args, **kwargs):
        return None

    def add_annotation(self, *args, **kwargs):
        return None

    def update_layout(self, *args, **kwargs):
        return None


class _DummyStreamlit(types.SimpleNamespace):
    def __init__(self):
        super().__init__()
        self.session_state = {}

    def title(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def text_input(self, *args, **kwargs):
        return "AAPL"

    def selectbox(self, *args, **kwargs):
        return "1Y"

    def stop(self):
        raise AssertionError("st.stop should not be called in this render path")

    def plotly_chart(self, *args, **kwargs):
        return None

    def columns(self, count):
        return [_DummyContext() for _ in range(count)]

    def tabs(self, names):
        return [_DummyContext() for _ in names]

    def json(self, *args, **kwargs):
        return None

    def table(self, *args, **kwargs):
        return None

    def dataframe(self, *args, **kwargs):
        return None

    def metric(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def bar_chart(self, *args, **kwargs):
        return None

    def write(self, *args, **kwargs):
        return None


class StockAnalysisPageTests(unittest.TestCase):
    def test_page_render_does_not_record_predictions_passively(self):
        price_frame = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0],
                "High": [101.0, 102.0, 103.0],
                "Low": [99.0, 100.0, 101.0],
                "Close": [100.0, 101.0, 102.0],
                "Volume": [1000, 1200, 1100],
            },
            index=pd.date_range("2026-01-01", periods=3, freq="D"),
        )
        tech_frame = price_frame.copy()
        tech_frame["sma20"] = tech_frame["Close"]
        tech_frame["sma50"] = tech_frame["Close"]
        tech_frame["sma200"] = tech_frame["Close"]
        tech_frame["rsi"] = [55.0, 56.0, 57.0]
        tech_frame["macd"] = [0.1, 0.2, 0.3]
        tech_frame["macd_signal"] = [0.05, 0.1, 0.15]

        analysis = {
            "ticker": "AAPL",
            "company": "Apple Inc.",
            "score": 82,
            "recommendation": "BUY",
            "time_horizon": "Medium-Term Hold",
            "entry_price": 99.0,
            "target_price": 120.0,
            "stop_loss": 92.0,
            "current_price": 102.0,
            "technical": {
                "data": tech_frame,
                "trend": "uptrend",
                "support_levels": [],
                "resistance_levels": [],
                "patterns": [],
                "gaps": [],
                "trendlines": {},
                "breakout": {},
                "indicators": {"rsi": 57.0, "macd": 0.3, "macd_signal": 0.15},
            },
            "fundamentals": {"metrics": {"sector": "Technology"}},
            "relative_strength": {},
            "macro_regime": {"vix": 18.0, "yield_10y": 4.1, "market_regime": "risk_on", "bullish_sectors": [], "bearish_sectors": []},
            "sentiment": {"sentiment_label": "Positive", "sentiment_score": 0.5, "headlines": []},
            "score_breakdown": {"technical": 40, "fundamental": 25, "sentiment": 17},
            "projections": {
                "current_price": 102.0,
                "short_term_target": 108.0,
                "short_term_low": 104.0,
                "short_term_high": 112.0,
                "short_term_upside": 5.88,
                "short_term_basis": "ARIMA",
                "medium_term_target": 120.0,
                "medium_term_low": 114.0,
                "medium_term_high": 126.0,
                "medium_term_upside": 17.65,
                "medium_term_basis": "ARIMA, Trend",
                "long_term_target": 140.0,
                "long_term_low": 130.0,
                "long_term_high": 150.0,
                "long_term_upside": 37.25,
                "long_term_basis": "Trend",
                "data_quality": "High",
            },
        }

        fake_streamlit = _DummyStreamlit()
        fake_graph_objects = types.SimpleNamespace(Candlestick=lambda **kwargs: kwargs, Bar=lambda **kwargs: kwargs, Scatter=lambda **kwargs: kwargs)
        fake_plotly_subplots = types.SimpleNamespace(make_subplots=lambda **kwargs: _DummyFigure())
        fake_ta_momentum = types.SimpleNamespace(RSIIndicator=lambda *args, **kwargs: types.SimpleNamespace(rsi=lambda: pd.Series([57.0, 57.0, 57.0])))
        fake_ta_trend = types.SimpleNamespace(MACD=lambda *args, **kwargs: types.SimpleNamespace(macd=lambda: pd.Series([0.1, 0.2, 0.3]), macd_signal=lambda: pd.Series([0.05, 0.1, 0.15])))

        with patch.dict(
            sys.modules,
            {
                "streamlit": fake_streamlit,
                "plotly.graph_objects": fake_graph_objects,
                "plotly.subplots": fake_plotly_subplots,
                "ta.momentum": fake_ta_momentum,
                "ta.trend": fake_ta_trend,
            },
        ):
            with patch("modules.polygon_client.is_polygon_configured", return_value=True):
                with patch("modules.scoring_engine.analyze_stock", return_value=analysis):
                    with patch("modules.data_fetcher.get_stock_data", return_value=price_frame):
                        with patch("modules.prediction_tracker.record_prediction", side_effect=AssertionError("record_prediction should not be called")) as record_mock:
                            runpy.run_path(str(STOCK_ANALYSIS_PAGE), run_name="__main__")

        record_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
