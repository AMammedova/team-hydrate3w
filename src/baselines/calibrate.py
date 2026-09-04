"""Member 2, W2.4 — isotonic regression or Platt scaling on validation
folds; reliability diagrams before/after. Feeds src/eval/metrics.py's
expected_calibration_error() and src/eval/plots.py's reliability figure.
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def fit_calibrator(
    y_val_true: np.ndarray,
    y_val_prob: np.ndarray,
    method: str = "isotonic",
):
    """Fit a calibrator on validation-fold positive-score probabilities.

    Parameters
    ----------
    y_val_true : np.ndarray
        Binary labels (1 = positive event, 0 = normal). Shape (N,).
    y_val_prob : np.ndarray
        Positive-class probabilities from positive_score(). Shape (N,).
    method : str
        "isotonic" (default, non-parametric) or "platt" (logistic regression).

    Returns
    -------
    calibrator
        A fitted object with a .predict(y_prob) method that maps raw
        positive-class probabilities to calibrated probabilities.
    """
    y_val_true = np.asarray(y_val_true, dtype=np.int64)
    y_val_prob = np.asarray(y_val_prob, dtype=np.float64)

    if method == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(y_val_prob, y_val_true)
        # Wrap so the returned object always has a .predict() interface
        return cal

    if method == "platt":
        # Platt scaling: fit a logistic regression on the log-odds
        # of the raw score. Clip to avoid log(0).
        prob_clipped = np.clip(y_val_prob, 1e-7, 1 - 1e-7)
        log_odds = np.log(prob_clipped / (1 - prob_clipped)).reshape(-1, 1)
        cal = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
        cal.fit(log_odds, y_val_true)

        class _PlattWrapper:
            def __init__(self, model):
                self._model = model

            def predict(self, y_prob: np.ndarray) -> np.ndarray:
                p = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-7, 1 - 1e-7)
                lo = np.log(p / (1 - p)).reshape(-1, 1)
                return self._model.predict_proba(lo)[:, 1]

        return _PlattWrapper(cal)

    raise ValueError(f"method must be 'isotonic' or 'platt', got {method!r}")
