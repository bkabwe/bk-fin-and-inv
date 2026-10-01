from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from scripts import train_lightgbm_return_models as cli


class TrainLightgbmReturnModelsSectorWiringTests(unittest.TestCase):
    def _run_main(self, argv):
        with patch("sys.argv", ["train_lightgbm_return_models.py", *argv]):
            return cli.main()

    def test_fetches_sector_and_passes_it_to_training_example_builder(self):
        with (
            patch("scripts.train_lightgbm_return_models.get_stock_info", return_value={"sector": "Technology"}) as info_mock,
            patch(
                "scripts.train_lightgbm_return_models.build_return_training_examples_for_ticker", return_value={30: (MagicMock(), MagicMock())}
            ) as examples_mock,
            patch("scripts.train_lightgbm_return_models.train_return_models", return_value={30: MagicMock()}),
            patch("scripts.train_lightgbm_return_models.save_return_models", return_value=["/tmp/fake/model.joblib"]),
        ):
            exit_code = self._run_main(["AAPL", "--horizons", "30"])

        self.assertEqual(exit_code, 0)
        info_mock.assert_called_once_with("AAPL")
        self.assertEqual(examples_mock.call_args.kwargs["sector"], "Technology")

    def test_sector_fetch_failure_does_not_block_training(self):
        with (
            patch("scripts.train_lightgbm_return_models.get_stock_info", side_effect=RuntimeError("boom")),
            patch(
                "scripts.train_lightgbm_return_models.build_return_training_examples_for_ticker", return_value={30: (MagicMock(), MagicMock())}
            ) as examples_mock,
            patch("scripts.train_lightgbm_return_models.train_return_models", return_value={30: MagicMock()}),
            patch("scripts.train_lightgbm_return_models.save_return_models", return_value=["/tmp/fake/model.joblib"]),
        ):
            exit_code = self._run_main(["AAPL", "--horizons", "30"])

        self.assertEqual(exit_code, 0)
        self.assertIsNone(examples_mock.call_args.kwargs["sector"])


if __name__ == "__main__":
    unittest.main()
