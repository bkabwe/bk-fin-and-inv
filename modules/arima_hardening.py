from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

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


def fit_arima_with_hardening(
    series: pd.Series | np.ndarray,
    order: tuple[int, int, int],
    logger: Any,
):
    """Use smarter initialization + higher maxiter; retry once with alternate optimizer on convergence warnings."""
    if not ARIMA_AVAILABLE:
        raise RuntimeError("statsmodels ARIMA is not available")

    model = ARIMA(series, order=order)
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
