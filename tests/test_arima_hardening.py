from __future__ import annotations

import unittest
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from modules import arima_hardening, backtester, scoring_engine


class ArimaHardeningTests(unittest.TestCase):
    def test_shared_fit_uses_hardened_parameters(self):
        model = MagicMock()
        init_result = MagicMock()
        init_result.params = np.array([0.1, 0.1, 0.1])
        final_result = MagicMock()
        model.fit.side_effect = [init_result, final_result]

        with patch("modules.arima_hardening.ARIMA", return_value=model):
            result = arima_hardening.fit_arima_with_hardening(
                pd.Series(np.linspace(100.0, 120.0, num=120)),
                order=(1, 1, 1),
                logger=MagicMock(),
            )

        self.assertIs(result, final_result)
        self.assertEqual(model.fit.call_args_list[0].kwargs.get("method"), "innovations_mle")
        self.assertEqual(model.fit.call_args_list[1].kwargs.get("method"), "statespace")
        self.assertEqual(model.fit.call_args_list[1].kwargs.get("method_kwargs", {}).get("maxiter"), 200)

    def test_shared_fit_retries_with_powell_on_convergence_warning(self):
        model = MagicMock()
        init_result = MagicMock()
        init_result.params = np.array([0.1, 0.1, 0.1])
        primary_result = MagicMock()
        retry_result = MagicMock()

        def _fit_side_effect(*args, **kwargs):
            method = kwargs.get("method")
            if method == "innovations_mle":
                return init_result
            method_kwargs = kwargs.get("method_kwargs", {})
            if method == "statespace" and method_kwargs.get("method") != "powell":
                warnings.warn("did not converge", arima_hardening.ConvergenceWarning)
                return primary_result
            return retry_result

        model.fit.side_effect = _fit_side_effect
        with patch("modules.arima_hardening.ARIMA", return_value=model):
            result = arima_hardening.fit_arima_with_hardening(
                pd.Series(np.linspace(100.0, 120.0, num=120)),
                order=(1, 1, 1),
                logger=MagicMock(),
            )

        self.assertIs(result, retry_result)
        self.assertEqual(model.fit.call_count, 3)
        self.assertEqual(model.fit.call_args_list[2].kwargs.get("method_kwargs", {}).get("method"), "powell")

    def test_shared_fit_retries_when_converged_flag_is_false(self):
        model = MagicMock()
        init_result = MagicMock()
        init_result.params = np.array([0.1, 0.1, 0.1])
        primary_result = MagicMock()
        primary_result.mle_retvals = {"converged": False}
        retry_result = MagicMock()
        model.fit.side_effect = [init_result, primary_result, retry_result]

        with patch("modules.arima_hardening.ARIMA", return_value=model):
            result = arima_hardening.fit_arima_with_hardening(
                pd.Series(np.linspace(100.0, 120.0, num=120)),
                order=(1, 1, 1),
                logger=MagicMock(),
            )

        self.assertIs(result, retry_result)
        self.assertEqual(model.fit.call_count, 3)
        self.assertEqual(model.fit.call_args_list[2].kwargs.get("method_kwargs", {}).get("method"), "powell")

    def test_backtester_order_selection_uses_shared_hardened_fit(self):
        fit_result = MagicMock()
        fit_result.aic = 1.0
        with patch("modules.backtester.fit_arima_with_hardening", return_value=fit_result) as fit_mock:
            order = backtester._select_arima_order(pd.Series(np.linspace(10.0, 20.0, num=80)))
        self.assertEqual(order, (0, 0, 0))
        self.assertTrue(fit_mock.called)

    def test_scoring_order_selection_uses_shared_hardened_fit(self):
        fit_result = MagicMock()
        fit_result.aic = 1.0
        values = tuple(np.linspace(10.0, 20.0, num=80).tolist())
        with patch("modules.scoring_engine.fit_arima_with_hardening", return_value=fit_result) as fit_mock:
            order = scoring_engine._select_arima_order("AAPL", values)
        self.assertEqual(order, (0, 0, 0))
        self.assertTrue(fit_mock.called)

    @unittest.skipUnless(arima_hardening.ARIMA_AVAILABLE, "statsmodels not installed")
    def test_fit_assigns_supported_frequency_without_missing_index_warning(self):
        business_days = pd.bdate_range("2024-01-02", periods=140)
        market_holidays = pd.DatetimeIndex(
            arima_hardening._NYSEHolidayCalendar().holidays(start=business_days.min(), end=business_days.max())
        )
        trading_days = business_days.difference(market_holidays)
        removed_positions = [9, 27, 54, 88]
        series_index = trading_days.delete(removed_positions)
        series = pd.Series(np.linspace(100.0, 120.0, num=len(series_index)), index=series_index)

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            forecast = arima_hardening.fit_arima_with_hardening(series, order=(1, 0, 0), logger=MagicMock()).forecast(steps=5)

        messages = [str(item.message).lower() for item in captured]
        self.assertFalse(any("no associated frequency information" in message for message in messages))
        self.assertFalse(any("no supported index is available" in message for message in messages))
        self.assertIsInstance(forecast.index, pd.DatetimeIndex)
        self.assertIsNotNone(forecast.index.freq)

    def test_supported_datetime_index_handles_non_holiday_gaps(self):
        business_days = pd.bdate_range("2024-01-02", periods=90)
        market_holidays = pd.DatetimeIndex(
            arima_hardening._NYSEHolidayCalendar().holidays(start=business_days.min(), end=business_days.max())
        )
        trading_days = business_days.difference(market_holidays)
        series_index = trading_days.delete([4, 12, 33])
        series = pd.Series(np.linspace(100.0, 105.0, num=len(series_index)), index=series_index)

        prepared = arima_hardening._with_supported_datetime_index(series)

        self.assertIsInstance(prepared.index, pd.DatetimeIndex)
        self.assertIsNotNone(prepared.index.freq)


if __name__ == "__main__":
    unittest.main()
