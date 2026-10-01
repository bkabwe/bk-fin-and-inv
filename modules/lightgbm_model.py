from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from modules.data_fetcher import get_stock_data
from modules.feature_engineering import build_feature_table, normalize_daily_index
from modules.logger import get_logger

logger = get_logger(__name__)

RETURN_HORIZONS = (30, 180, 720)
MIN_TRAINING_ROWS_PER_HORIZON = 50
MIN_FORWARD_RETURN_LABEL = -1.0
MAX_FORWARD_RETURN_LABEL_30D = 5.0
MAX_FORWARD_RETURN_LABEL_180D = 10.0
MAX_FORWARD_RETURN_LABEL_LONG = 20.0

# Default lower/upper quantile levels for the optional prediction-interval
# models below (an 80% forward-return interval). Not yet consumed by the
# live scoring ensemble's confidence bands -- see train_return_quantile_models.
DEFAULT_RETURN_QUANTILES = (0.1, 0.9)

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
    frame = pd.DataFrame({"Close": close.values}, index=normalize_daily_index(price_data.index))
    frame = frame[~frame.index.isna()].sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame["Close"]


def _label_sanity_bounds(horizon: int) -> tuple[float, float]:
    if int(horizon) <= 30:
        return MIN_FORWARD_RETURN_LABEL, MAX_FORWARD_RETURN_LABEL_30D
    if int(horizon) <= 180:
        return MIN_FORWARD_RETURN_LABEL, MAX_FORWARD_RETURN_LABEL_180D
    return MIN_FORWARD_RETURN_LABEL, MAX_FORWARD_RETURN_LABEL_LONG


def prepare_lightgbm_feature_frame(feature_table: pd.DataFrame | None) -> pd.DataFrame:
    if feature_table is None or feature_table.empty:
        return pd.DataFrame()
    features = feature_table.copy()
    features.index = normalize_daily_index(features.index)
    features = features[~features.index.isna()].sort_index()
    features = features[~features.index.duplicated(keep="last")]
    return features


def latest_lightgbm_feature_row(feature_table: pd.DataFrame | None) -> pd.Series | None:
    features = prepare_lightgbm_feature_frame(feature_table)
    if features.empty:
        return None
    latest_feature_rows = features.dropna(how="all")
    if latest_feature_rows.empty:
        return None
    return latest_feature_rows.iloc[-1]


def summarize_training_feature_ranges(x_train: pd.DataFrame) -> pd.DataFrame:
    if x_train is None or x_train.empty:
        return pd.DataFrame(columns=["train_min", "train_max", "train_mean", "train_std", "train_non_null_count"])
    numeric = x_train.apply(pd.to_numeric, errors="coerce")
    summary = pd.DataFrame(index=numeric.columns)
    summary["train_min"] = numeric.min()
    summary["train_max"] = numeric.max()
    summary["train_mean"] = numeric.mean()
    summary["train_std"] = numeric.std(ddof=0)
    summary["train_non_null_count"] = numeric.count().astype("int64")
    summary.index.name = "feature"
    return summary


def compare_feature_row_to_training_ranges(
    feature_row: pd.Series | pd.DataFrame | None,
    training_feature_ranges: pd.DataFrame,
    *,
    std_threshold: float = 5.0,
) -> pd.DataFrame:
    if feature_row is None:
        return pd.DataFrame(
            columns=[
                "live_value",
                "train_min",
                "train_max",
                "train_mean",
                "train_std",
                "train_non_null_count",
                "outside_training_range",
                "std_threshold",
                "abs_zscore",
                "beyond_std_threshold",
            ]
        )
    live_series = feature_row if isinstance(feature_row, pd.Series) else feature_row.iloc[0]
    live_series = pd.to_numeric(live_series, errors="coerce")
    comparison = training_feature_ranges.copy()
    comparison["live_value"] = live_series.reindex(comparison.index)
    comparison["outside_training_range"] = (
        comparison["live_value"].notna()
        & (
            comparison["live_value"].lt(comparison["train_min"])
            | comparison["live_value"].gt(comparison["train_max"])
        )
    )
    comparison["std_threshold"] = float(std_threshold)
    with np.errstate(divide="ignore", invalid="ignore"):
        comparison["abs_zscore"] = (
            (comparison["live_value"] - comparison["train_mean"]).abs() / comparison["train_std"].replace(0.0, np.nan)
        )
    comparison["beyond_std_threshold"] = comparison["abs_zscore"].gt(float(std_threshold)).fillna(False)
    ordered = [
        "live_value",
        "train_min",
        "train_max",
        "train_mean",
        "train_std",
        "train_non_null_count",
        "outside_training_range",
        "std_threshold",
        "abs_zscore",
        "beyond_std_threshold",
    ]
    return comparison.reindex(columns=ordered)


def build_return_training_examples(
    ticker: str,
    price_data: pd.DataFrame,
    feature_table: pd.DataFrame | None = None,
    lookback_days: int = 1260,
    horizons: tuple[int, ...] = RETURN_HORIZONS,
) -> dict[int, tuple[pd.DataFrame, pd.Series]]:
    """Build per-horizon (X, y) datasets where y is forward return."""
    raw_features = feature_table if feature_table is not None else build_feature_table(ticker, price_data, lookback_days=lookback_days)
    features = prepare_lightgbm_feature_frame(raw_features)
    if features.empty:
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
        future_close = aligned["Close"].shift(-horizon)
        future_dates = aligned.index.to_series().shift(-horizon)
        with np.errstate(divide="ignore", invalid="ignore"):
            target = (future_close - aligned["Close"]) / aligned["Close"]
        min_label, max_label = _label_sanity_bounds(int(horizon))
        price_mask = aligned["Close"].gt(0) & future_close.gt(0)
        finite_mask = pd.Series(np.isfinite(target.to_numpy(dtype="float64", copy=False)), index=target.index)
        bound_mask = target.between(min_label, max_label, inclusive="both")
        mask = target.notna() & price_mask & finite_mask & bound_mask
        excluded = target.loc[target.notna() & ~mask]
        for current_date, label in excluded.items():
            future_date = future_dates.get(current_date)
            current_close = aligned.at[current_date, "Close"]
            future_close_value = future_close.get(current_date)
            logger.warning(
                (
                    "Excluding LightGBM training label for %s horizon=%sd "
                    "current_date=%s future_date=%s current_close=%.6f future_close=%.6f forward_return=%.6f "
                    "allowed_range=[%.2f, %.2f]"
                ),
                str(ticker).upper(),
                int(horizon),
                pd.Timestamp(current_date).date().isoformat(),
                pd.Timestamp(future_date).date().isoformat() if pd.notna(future_date) else "n/a",
                float(current_close) if pd.notna(current_close) else float("nan"),
                float(future_close_value) if pd.notna(future_close_value) else float("nan"),
                float(label),
                float(min_label),
                float(max_label),
            )
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
    *,
    shared_macro_table: pd.DataFrame | None = None,
) -> dict[int, tuple[pd.DataFrame, pd.Series]]:
    """Build examples by reusing the existing Polygon-backed get_stock_data flow."""
    price_data = get_stock_data(ticker, period=period, interval=interval)
    if price_data is None or price_data.empty:
        logger.warning("LightGBM training examples skipped for %s: no historical price data", str(ticker).upper())
        return {}
    feature_table = build_feature_table(
        ticker,
        price_data,
        lookback_days=lookback_days,
        shared_macro_table=shared_macro_table,
    )
    return build_return_training_examples(
        ticker=ticker,
        price_data=price_data,
        feature_table=feature_table,
        lookback_days=lookback_days,
        horizons=horizons,
    )


class BaggedLGBMRegressor:
    """Averages predictions across several seeded/row-subsampled LGBMRegressors.

    Exists to reduce single-seed variance within the existing small
    walk-forward training windows (as few as ~50-60 rows at the 30d horizon)
    without changing the feature set, objective, or hyperparameters. Mirrors
    the plain `LGBMRegressor` interface used elsewhere in this module
    (`.predict()`, `.feature_name_`) so it is a drop-in replacement for
    `predict_forward_return()` and `joblib`-based save/load. Off by default:
    `train_return_models(..., n_bagged_estimators=1)` (the default) returns a
    single bare `LGBMRegressor` per horizon, identical to prior behavior.
    """

    def __init__(self, models: list[LGBMRegressor]):
        if not models:
            raise ValueError("BaggedLGBMRegressor requires at least one fitted model.")
        self.models = list(models)
        self.feature_name_ = list(getattr(self.models[0], "feature_name_", []) or [])

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        predictions = np.column_stack([model.predict(x) for model in self.models])
        return predictions.mean(axis=1)


def _fit_bagged_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_bagged_estimators: int,
    bagging_fraction: float,
    random_state: int,
) -> LGBMRegressor | BaggedLGBMRegressor:
    if n_bagged_estimators <= 1:
        model = LGBMRegressor(
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            random_state=random_state,
        )
        model.fit(x_train, y_train)
        return model

    fraction = min(1.0, max(0.1, float(bagging_fraction)))
    sample_size = max(1, int(round(len(y_train) * fraction)))
    rng = np.random.default_rng(random_state)
    bagged_models: list[LGBMRegressor] = []
    for seed_offset in range(int(n_bagged_estimators)):
        row_positions = rng.choice(len(y_train), size=sample_size, replace=True)
        x_sample = x_train.iloc[row_positions]
        y_sample = y_train.iloc[row_positions]
        model = LGBMRegressor(
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            random_state=random_state + seed_offset,
        )
        model.fit(x_sample, y_sample)
        bagged_models.append(model)
    return BaggedLGBMRegressor(bagged_models)


def train_return_models(
    training_examples: dict[int, tuple[pd.DataFrame, pd.Series]],
    min_rows_per_horizon: int = MIN_TRAINING_ROWS_PER_HORIZON,
    random_state: int = 42,
    n_bagged_estimators: int = 1,
    bagging_fraction: float = 0.8,
) -> dict[int, LGBMRegressor] | None:
    """
    Train one LightGBMRegressor per horizon.
    NaN feature values are passed through intentionally (native LightGBM handling).

    n_bagged_estimators controls optional bagging/model averaging (default 1 =
    disabled, preserving prior single-model behavior): when > 1, trains that
    many row-bootstrapped, differently-seeded LGBMRegressors per horizon
    (each fit on `bagging_fraction` of the training rows, sampled with
    replacement) and returns a BaggedLGBMRegressor that averages their
    predictions, trading a modest train-time cost for reduced prediction
    variance on the same small windows.
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
        logger.info(
            "Training LightGBM horizon %sd with %d rows, %d features, n_bagged_estimators=%d",
            horizon,
            len(y_train),
            x_train.shape[1],
            int(n_bagged_estimators),
        )
        models[int(horizon)] = _fit_bagged_model(
            x_train,
            y_train,
            n_bagged_estimators=int(n_bagged_estimators),
            bagging_fraction=float(bagging_fraction),
            random_state=random_state,
        )

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
    feature_order = list(getattr(model, "feature_name_", []) or [])
    if feature_order:
        for column in feature_order:
            if column not in x.columns:
                x[column] = np.nan
        x = x.reindex(columns=feature_order)
    prediction = float(model.predict(x.iloc[[0]])[0])
    logger.info("LightGBM inference complete: predicted_forward_return=%.6f", prediction)
    return prediction


def training_row_counts(training_examples: dict[int, tuple[pd.DataFrame, pd.Series]] | None) -> dict[int, int]:
    counts: dict[int, int] = {}
    for horizon, (_x_train, y_train) in sorted((training_examples or {}).items()):
        counts[int(horizon)] = int(len(y_train))
    return counts


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


def train_return_quantile_models(
    training_examples: dict[int, tuple[pd.DataFrame, pd.Series]],
    min_rows_per_horizon: int = MIN_TRAINING_ROWS_PER_HORIZON,
    random_state: int = 42,
    quantiles: tuple[float, ...] = DEFAULT_RETURN_QUANTILES,
) -> dict[int, dict[float, LGBMRegressor]] | None:
    """Train per-horizon, per-quantile LightGBM forward-return models
    (objective="quantile") to produce prediction intervals, complementing the
    point-estimate models from train_return_models.

    This is an additive, offline-evaluation capability: it reuses the same
    training examples/feature set as the point-estimate models and is not yet
    wired into the live scoring ensemble's GARCH-based confidence bands (see
    README's "Known modeling limitations" discussion of confidence bands) --
    integrating it there would change several already-validated downstream
    fields (market-cap band padding, projection confidence, risk-adjusted
    upside) and should follow a dedicated backtest comparison first.
    """
    if not LIGHTGBM_AVAILABLE:
        logger.warning("LightGBM package not installed; skipping quantile model training.")
        return None
    if not training_examples:
        logger.warning("No LightGBM training examples provided.")
        return None
    if not quantiles:
        logger.warning("No quantile levels provided; skipping quantile model training.")
        return None

    models: dict[int, dict[float, LGBMRegressor]] = {}
    for horizon, (x_train, y_train) in sorted(training_examples.items()):
        if len(y_train) < int(min_rows_per_horizon):
            logger.warning(
                "Skipping LightGBM quantile models for horizon %sd: insufficient rows (%d < %d)",
                horizon,
                len(y_train),
                int(min_rows_per_horizon),
            )
            continue
        horizon_models: dict[float, LGBMRegressor] = {}
        for quantile in quantiles:
            model = LGBMRegressor(
                objective="quantile",
                alpha=float(quantile),
                n_estimators=250,
                learning_rate=0.05,
                num_leaves=31,
                random_state=random_state,
            )
            model.fit(x_train, y_train)
            horizon_models[float(quantile)] = model
        logger.info(
            "Trained LightGBM quantile models for horizon %sd: quantiles=%s rows=%d",
            horizon,
            sorted(horizon_models.keys()),
            len(y_train),
        )
        models[int(horizon)] = horizon_models

    if not models:
        logger.warning("LightGBM quantile model training skipped: no horizon met minimum row threshold.")
        return None
    return models


def predict_forward_return_quantiles(
    quantile_models: dict[float, LGBMRegressor] | None,
    feature_row: pd.Series | pd.DataFrame,
) -> dict[float, float]:
    """Predict forward return at each trained quantile level for one feature row."""
    if not quantile_models:
        logger.warning("LightGBM quantile inference skipped: no quantile models provided")
        return {}
    predictions: dict[float, float] = {}
    for quantile, model in sorted(quantile_models.items()):
        prediction = predict_forward_return(model, feature_row)
        if prediction is not None:
            predictions[float(quantile)] = prediction
    return predictions


def save_return_quantile_models(
    quantile_models: dict[int, dict[float, LGBMRegressor]], directory: str | Path
) -> list[Path]:
    base = Path(directory)
    base.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for horizon, horizon_models in sorted(quantile_models.items()):
        for quantile, model in sorted(horizon_models.items()):
            path = base / f"lightgbm_return_h{int(horizon)}_q{quantile:.2f}.joblib"
            joblib.dump(model, path)
            saved_paths.append(path)
    logger.info("Saved %d LightGBM quantile model(s) to %s", len(saved_paths), str(base))
    return saved_paths


def load_return_quantile_models(
    directory: str | Path,
    horizons: tuple[int, ...] = RETURN_HORIZONS,
    quantiles: tuple[float, ...] = DEFAULT_RETURN_QUANTILES,
) -> dict[int, dict[float, LGBMRegressor]]:
    base = Path(directory)
    models: dict[int, dict[float, LGBMRegressor]] = {}
    for horizon in horizons:
        horizon_models: dict[float, LGBMRegressor] = {}
        for quantile in quantiles:
            path = base / f"lightgbm_return_h{int(horizon)}_q{float(quantile):.2f}.joblib"
            if not path.exists():
                continue
            horizon_models[float(quantile)] = joblib.load(path)
        if horizon_models:
            models[int(horizon)] = horizon_models
    if not models:
        logger.warning("No LightGBM quantile return models found under %s", str(base))
    else:
        logger.info("Loaded LightGBM quantile return model(s): %s", sorted(models.keys()))
    return models
