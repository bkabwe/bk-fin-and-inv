from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from modules import backtester, scoring_engine


class ArimaHardeningTests(unittest.TestCase):
    def test_backtester_fit_uses_hardened_parameters(self):
        model = MagicMock()
        init_result = MagicMock()
        init_result.params = np.array([0.1, 0.1, 0.1])
        final_result = MagicMock()
        model.fit.side_effect = [init_result, final_result]

        with patch("modules.backtester.ARIMA", return_value=model):
            result = backtester._fit_arima_with_hardening(pd.Series(np.linspace(100.0, 120.0, num=120)), order=(1, 1, 1))

        self.assertIs(result, final_result)
        self.assertEqual(model.fit.call_args_list[0].kwargs.get("method"), "innovations_mle")
        self.assertEqual(model.fit.call_args_list[1].kwargs.get("method"), "statespace")
        self.assertEqual(model.fit.call_args_list[1].kwargs.get("method_kwargs", {}).get("maxiter"), 200)

    def test_scoring_fit_uses_hardened_parameters(self):
        model = MagicMock()
        init_result = MagicMock()
        init_result.params = np.array([0.1, 0.1, 0.1])
        final_result = MagicMock()
        model.fit.side_effect = [init_result, final_result]

        with patch("modules.scoring_engine.ARIMA", return_value=model):
            result = scoring_engine._fit_arima_with_hardening(pd.Series(np.linspace(90.0, 130.0, num=140)), order=(2, 1, 0))

        self.assertIs(result, final_result)
        self.assertEqual(model.fit.call_args_list[0].kwargs.get("method"), "innovations_mle")
        self.assertEqual(model.fit.call_args_list[1].kwargs.get("method"), "statespace")
        self.assertEqual(model.fit.call_args_list[1].kwargs.get("method_kwargs", {}).get("maxiter"), 200)


if __name__ == "__main__":
    unittest.main()
