from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)
from pandas.tseries.offsets import CustomBusinessDay

try:  # pragma: no cover
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    ARIMA_AVAILABLE = True
except Exception:  # pragma: no cover
    ARIMA = None  # type: ignore[assignment]
    ConvergenceWarning = Warning  # type: ignore[assignment]
    ARIMA_AVAILABLE = False

# Keep warning suppression safety net in place even with improved convergence settings.
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# Higher than statsmodels defaults to reduce ARIMA optimizer non-convergence on noisier equity series.
ARIMA_MAXITER = 200


class _NYSEHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("NewYearsDay", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, observance=nearest_workday, start_date="2021-06-19"),
        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("ChristmasDay", month=12, day=25, observance=nearest_workday),
    ]


def _with_supported_datetime_index(series: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    if not isinstance(series, pd.Series):
        return series

    prepared = pd.to_numeric(series, errors="coerce").astype("float64").copy()
    if not isinstance(prepared.index, pd.DatetimeIndex) or len(prepared.index) < 2:
        return prepared

    index = pd.DatetimeIndex(pd.to_datetime(prepared.index, errors="coerce"))
    if index.hasnans:
        return prepared
    if getattr(index, "tz", None) is not None:
        index = index.tz_convert(None)
    if not index.is_monotonic_increasing:
        prepared = prepared.sort_index()
        index = pd.DatetimeIndex(prepared.index)
        if getattr(index, "tz", None) is not None:
            index = index.tz_convert(None)

    normalized = pd.DatetimeIndex(index.normalize())
    if normalized.has_duplicates:
        prepared.index = normalized
        prepared = prepared[~prepared.index.duplicated(keep="last")]
        normalized = pd.DatetimeIndex(prepared.index)
        if len(normalized) < 2:
            return prepared

    if getattr(normalized, "freq", None) is not None:
        prepared.index = normalized
        return prepared

    inferred = pd.infer_freq(normalized)
    if inferred:
        prepared.index = pd.DatetimeIndex(normalized, freq=inferred)
        return prepared

    missing_business_days = pd.bdate_range(normalized.min(), normalized.max()).difference(normalized)
    calendar = _NYSEHolidayCalendar()
    known_market_holidays = pd.DatetimeIndex(
        calendar.holidays(start=normalized.min(), end=normalized.max())
    ).normalize()
    if len(missing_business_days) == 0:
        prepared.index = pd.DatetimeIndex(index, freq="B")
        return prepared
    holiday_start = normalized.min() - pd.DateOffset(years=1)
    holiday_end = normalized.max() + pd.DateOffset(years=5)
    supported_holidays = pd.DatetimeIndex(calendar.holidays(start=holiday_start, end=holiday_end)).normalize()
    residual_missing_days = missing_business_days.difference(known_market_holidays)
    if len(residual_missing_days) > 0:
        supported_holidays = supported_holidays.union(residual_missing_days).normalize().unique().sort_values()
    business_day_freq = CustomBusinessDay(holidays=supported_holidays)
    try:
        # Preserve observed trading-day values/timestamps without inserting rows.
        prepared.index = pd.DatetimeIndex(normalized, freq=business_day_freq)
    except ValueError:
        expected = pd.date_range(start=normalized.min(), end=normalized.max(), freq=business_day_freq)
        if len(expected) == len(normalized) and expected.equals(normalized):
            idx_with_freq = pd.DatetimeIndex(normalized)
            idx_with_freq.freq = business_day_freq
            prepared.index = idx_with_freq
        else:
            prepared.index = normalized
    return prepared


def fit_arima_with_hardening(
    series: pd.Series | np.ndarray,
    order: tuple[int, int, int],
    logger: Any,
):
    """Use smarter initialization + higher maxiter; retry once with alternate optimizer on convergence warnings."""
    if not ARIMA_AVAILABLE:
        raise RuntimeError("statsmodels ARIMA is not available")

    model = ARIMA(_with_supported_datetime_index(series), order=order)
    start_params = None
    try:
        initial = model.fit(method="innovations_mle")
        start_params = np.asarray(initial.params, dtype=float)
    except Exception:
        start_params = None

    with warnings.catch_warnings(record=True) as primary_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        result = model.fit(
            method="statespace",
            start_params=start_params,
            method_kwargs={"maxiter": ARIMA_MAXITER, "disp": 0},
        )
    convergence_warning_seen = any(issubclass(w.category, ConvergenceWarning) for w in primary_warnings)
    mle_retvals = getattr(result, "mle_retvals", {}) or {}
    converged_flag = bool(mle_retvals.get("converged", True))
    if converged_flag and not convergence_warning_seen:
        return result

    logger.info("ARIMA fit did not fully converge for order=%s; retrying with Powell optimizer", order)
    return model.fit(
        method="statespace",
        start_params=start_params,
        method_kwargs={"maxiter": ARIMA_MAXITER, "disp": 0, "method": "powell"},
    )
