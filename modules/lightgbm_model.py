from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from modules.data_fetcher import get_stock_data
from modules.feature_engineering import build_feature_table
from modules.logger import get_logger

logger = get_logger(__name__)

RETURN_HORIZONS = (30, 180, 720)
MIN_TRAINING_ROWS_PER_HORIZON = 50

try:  # pragma: no cover
    from lightgbm import LGBMRegressor

    LIGHTGBM_AVAILABLE = True
except Exception:  # pragma: no cover
    LGBMRegressor = None  # type: ignore[assignment]
    LIGHTGBM_AVAILABLE = False


def _normalize_close_series(price_data: pd.DataFrame) -> pd.Series:
    if price_data is None or price_data.empty or "Close" not in price_data:
        return pd.Series(dtype="float64", name="Close")
    close = pd.to_numeric(price_data["Close"], errors="coerce").astype("float64")
    index = pd.to_datetime(price_data.index, errors="coerce")
    frame = pd.DataFrame({"Close": close.values}, index=index)
    frame = frame[~frame.index.isna()].sort_index()
    if getattr(frame.index, "tz", None) is not None:
        frame.index = frame.index.tz_convert(None)
    return frame["Close"]


def build_return_training_examples(
    ticker: str,
    price_data: pd.DataFrame,
    feature_table: pd.DataFrame | None = None,
    lookback_days: int = 1260,
    horizons: tuple[int, ...] = RETURN_HORIZONS,
) -> dict[int, tuple[pd.DataFrame, pd.Series]]:
    """Build per-horizon (X, y) datasets where y is forward return."""
    features = feature_table if feature_table is not None else build_feature_table(ticker, price_data, lookback_days=lookback_days)
    if features is None or features.empty:
        logger.warning("LightGBM training examples skipped for %s: empty feature table", str(ticker).upper())
        return {}

    close = _normalize_close_series(price_data)
    if close.empty:
        logger.warning("LightGBM training examples skipped for %s: empty close-price history", str(ticker).upper())
        return {}

    aligned = features.copy().join(close.rename("Close"), how="left")
    feature_columns = [col for col in aligned.columns if col != "Close"]
    datasets: dict[int, tuple[pd.DataFrame, pd.Series]] = {}

    for horizon in horizons:
        target = (aligned["Close"].shift(-horizon) - aligned["Close"]) / aligned["Close"]
        mask = target.notna() & aligned["Close"].notna()
        if not mask.any():
            continue
        datasets[int(horizon)] = (aligned.loc[mask, feature_columns], target.loc[mask].astype("float64"))

    logger.info(
        "LightGBM training examples built for %s: feature_rows=%d horizons=%s",
        str(ticker).upper(),
        len(features),
        sorted(datasets.keys()),
    )
    return datasets


def build_return_training_examples_for_ticker(
    ticker: str,
    period: str = "5y",
    interval: str = "1d",
    lookback_days: int = 1260,
    horizons: tuple[int, ...] = RETURN_HORIZONS,
) -> dict[int, tuple[pd.DataFrame, pd.Series]]:
    """Build examples by reusing the existing Polygon-backed get_stock_data flow."""
    price_data = get_stock_data(ticker, period=period, interval=interval)
    if price_data is None or price_data.empty:
        logger.warning("LightGBM training examples skipped for %s: no historical price data", str(ticker).upper())
        return {}
    return build_return_training_examples(
        ticker=ticker,
        price_data=price_data,
        feature_table=None,
        lookback_days=lookback_days,
        horizons=horizons,
    )


def train_return_models(
    training_examples: dict[int, tuple[pd.DataFrame, pd.Series]],
    min_rows_per_horizon: int = MIN_TRAINING_ROWS_PER_HORIZON,
    random_state: int = 42,
) -> dict[int, LGBMRegressor] | None:
    """
    Train one LightGBMRegressor per horizon.
    NaN feature values are passed through intentionally (native LightGBM handling).
    """
    if not LIGHTGBM_AVAILABLE:
        logger.warning("LightGBM package not installed; skipping model training.")
        return None
    if not training_examples:
        logger.warning("No LightGBM training examples provided.")
        return None

    models: dict[int, LGBMRegressor] = {}
    for horizon, (x_train, y_train) in sorted(training_examples.items()):
        if len(y_train) < int(min_rows_per_horizon):
            logger.warning(
                "Skipping LightGBM horizon %sd: insufficient rows (%d < %d)",
                horizon,
                len(y_train),
                int(min_rows_per_horizon),
            )
            continue
        model = LGBMRegressor(
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            random_state=random_state,
        )
        logger.info(
            "Training LightGBM horizon %sd with %d rows and %d features",
            horizon,
            len(y_train),
            x_train.shape[1],
        )
        model.fit(x_train, y_train)
        models[int(horizon)] = model

    if not models:
        logger.warning("LightGBM model training skipped: no horizon met minimum row threshold.")
        return None
    logger.info("LightGBM training complete: trained horizons=%s", sorted(models.keys()))
    return models


def predict_forward_return(model: LGBMRegressor | None, feature_row: pd.Series | pd.DataFrame) -> float | None:
    if model is None:
        logger.warning("LightGBM inference skipped: model is None")
        return None
    if feature_row is None:
        logger.warning("LightGBM inference skipped: empty feature row")
        return None
    x = feature_row.to_frame().T if isinstance(feature_row, pd.Series) else feature_row.copy()
    if x.empty:
        logger.warning("LightGBM inference skipped: empty feature row")
        return None
    prediction = float(model.predict(x.iloc[[0]])[0])
    logger.info("LightGBM inference complete: predicted_forward_return=%.6f", prediction)
    return prediction


def save_return_models(models: dict[int, LGBMRegressor], directory: str | Path) -> list[Path]:
    base = Path(directory)
    base.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for horizon, model in sorted(models.items()):
        path = base / f"lightgbm_return_h{int(horizon)}.joblib"
        joblib.dump(model, path)
        saved_paths.append(path)
    logger.info("Saved %d LightGBM return model(s) to %s", len(saved_paths), str(base))
    return saved_paths


def load_return_models(directory: str | Path, horizons: tuple[int, ...] = RETURN_HORIZONS) -> dict[int, LGBMRegressor]:
    base = Path(directory)
    models: dict[int, LGBMRegressor] = {}
    for horizon in horizons:
        path = base / f"lightgbm_return_h{int(horizon)}.joblib"
        if not path.exists():
            continue
        models[int(horizon)] = joblib.load(path)
    if not models:
        logger.warning("No LightGBM return models found under %s", str(base))
    else:
        logger.info("Loaded LightGBM return model(s): %s", sorted(models.keys()))
    return models
