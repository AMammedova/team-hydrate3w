"""Probability calibration. Fit on VALIDATION, applied unchanged to that
fold's test wells (TEAM_5_MEMBERS.md 9.3).

Module 8 picks the threshold by sweeping a probability axis, so probabilities
must mean what they say; weighted training on a ~3% positive rate makes them
miscalibrated by construction.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class Calibrator:
    """Uniform .predict() so callers never branch on the method."""

    def __init__(self, method: str, model, *, degenerate: bool = False, constant: float = 0.0):
        self.method = method
        self._model = model
        self.degenerate = degenerate
        self._constant = float(constant)

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
        if self.degenerate:
            # Validation saw one class. Identity, not the base rate: collapsing
            # to a constant would erase the ranking threshold selection needs.
            return p
        if self.method == "isotonic":
            return np.clip(self._model.predict(p), 0.0, 1.0)
        lo = _logit(p)
        return self._model.predict_proba(lo.reshape(-1, 1))[:, 1]

    def __repr__(self) -> str:
        return f"Calibrator(method={self.method!r}, degenerate={self.degenerate})"


def _logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _fit_one(method: str, y_true: np.ndarray, y_prob: np.ndarray) -> Calibrator:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(y_prob, y_true)
        return Calibrator("isotonic", model)

    if method == "platt":
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        model.fit(_logit(y_prob).reshape(-1, 1), y_true)
        return Calibrator("platt", model)

    raise ValueError(f"method must be 'isotonic', 'platt' or 'auto', got {method!r}")


def fit_calibrator(
    y_val_true: np.ndarray,
    y_val_prob: np.ndarray,
    method: str = "platt",
    *,
    n_cv: int = 5,
    random_state: int = 42,
) -> Calibrator:
    """y_val_true must be BINARY -- pass (y != 0), not the 3-class label.

    Platt by default: a validation fold holds a handful of positive events,
    where isotonic fits a step function to noise. "auto" picks by CV ECE.
    """
    y_true = np.asarray(y_val_true)
    y_prob = np.asarray(y_val_prob, dtype=np.float64)

    if y_true.ndim != 1 or y_prob.ndim != 1 or len(y_true) != len(y_prob):
        raise ValueError(
            f"y_val_true {y_true.shape} and y_val_prob {y_prob.shape} must be "
            f"1-D and the same length"
        )
    if y_true.ndim == 1 and set(np.unique(y_true)) - {0, 1}:
        raise ValueError(
            "y_val_true must be BINARY (0/1). Pass (y != 0), not the 3-class label."
        )
    y_true = y_true.astype(np.int64)

    if len(np.unique(y_true)) < 2:
        logger.warning(
            "validation fold has a single class (%d rows, %d positive) -- calibration "
            "is undefined; falling back to the identity mapping",
            len(y_true), int(y_true.sum()),
        )
        return Calibrator("identity", None, degenerate=True)

    if method != "auto":
        return _fit_one(method, y_true, y_prob)

    from sklearn.model_selection import StratifiedKFold

    from src.eval.metrics import expected_calibration_error

    n_pos = int(y_true.sum())
    splits = min(n_cv, n_pos, len(y_true) - n_pos)
    if splits < 2:
        logger.warning(
            "only %d positive rows in validation -- too few to compare calibrators; "
            "defaulting to platt", n_pos,
        )
        return _fit_one("platt", y_true, y_prob)

    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    scores: dict[str, list[float]] = {"platt": [], "isotonic": []}
    for tr, te in skf.split(y_prob.reshape(-1, 1), y_true):
        if len(np.unique(y_true[tr])) < 2:
            continue
        for name in scores:
            cal = _fit_one(name, y_true[tr], y_prob[tr])
            scores[name].append(
                expected_calibration_error(y_true[te], cal.predict(y_prob[te]))
            )

    means = {k: float(np.mean(v)) for k, v in scores.items() if v}
    if not means:
        return _fit_one("platt", y_true, y_prob)
    winner = min(means, key=means.get)
    logger.info("calibrator auto-selection by CV ECE: %s -> %s", means, winner)
    return _fit_one(winner, y_true, y_prob)


def calibration_report(
    y_true: np.ndarray,
    y_prob_before: np.ndarray,
    y_prob_after: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Before/after ECE and Brier. Brier too, because predicting the base
    rate everywhere gives near-zero ECE and zero discriminative value."""
    from src.eval.metrics import expected_calibration_error

    y_true = np.asarray(y_true).astype(np.int64)
    out = {}
    for tag, p in (("before", y_prob_before), ("after", y_prob_after)):
        p = np.asarray(p, dtype=np.float64)
        out[f"ece_{tag}"] = float(expected_calibration_error(y_true, p, n_bins))
        out[f"brier_{tag}"] = float(np.mean((p - y_true) ** 2))
        out[f"mean_prob_{tag}"] = float(p.mean())
    out["base_rate"] = float(y_true.mean())
    out["ece_delta"] = out["ece_after"] - out["ece_before"]
    out["brier_delta"] = out["brier_after"] - out["brier_before"]
    return out
