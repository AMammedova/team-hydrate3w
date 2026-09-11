"""Hyperparameter search for XGBoostBaseline: training fold only, grouped
inner CV, scored on validation PR-AUC over positive_score().

Returns a SearchResult, not a flat dict: XGBClassifier silently accepts and
ignores unknown keys, so metadata mixed into hyperparameters goes unnoticed.
"""

from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_GRID = {
    "n_estimators": [300, 600],
    "max_depth": [3, 6],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
}


@dataclass
class SearchResult:
    best_params: dict          # model-ready kwargs only
    best_val_pr_auc: float
    n_configs_tried: int
    n_configs_scored: int
    seconds: float
    trials: pd.DataFrame = field(repr=False)

    def summary(self) -> str:
        return (
            f"best val PR-AUC {self.best_val_pr_auc:.4f} from {self.n_configs_scored}"
            f"/{self.n_configs_tried} scored configurations in {self.seconds:.1f}s; "
            f"best_params={self.best_params}"
        )


def _iter_configs(param_grid: dict, n_iter: int | None, rng: np.random.Generator):
    keys = list(param_grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(param_grid[k] for k in keys))]
    if n_iter is not None and n_iter < len(combos):
        pick = rng.choice(len(combos), size=n_iter, replace=False)
        return [combos[i] for i in sorted(pick)], len(combos)
    return combos, len(combos)


def search(
    X_train: np.ndarray,
    mask_train: np.ndarray | None,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    param_grid: dict | None = None,
    *,
    n_inner_splits: int = 3,
    n_iter: int | None = None,
    random_state: int = 42,
    device: str = "auto",
    base_params: dict | None = None,
    score_rows: np.ndarray | None = None,
) -> SearchResult:
    """X_train is the FEATURE matrix; mask_train is accepted and ignored.
    groups_train are well ids, so inner CV never splits a well.

    score_rows: rows eligible for inner SCORING (training still uses all).
    Needed for real_plus_sim, whose training fold is ~90% positive simulated
    windows against a 3% real rate.
    """
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import GroupKFold
    from xgboost import XGBClassifier

    from src.baselines.device import resolve_device
    from src.baselines.xgb_model import compute_sample_weight
    from src.eval.metrics import positive_score

    param_grid = param_grid or DEFAULT_GRID
    rng = np.random.default_rng(random_state)
    combos, n_total = _iter_configs(param_grid, n_iter, rng)

    y_train = np.asarray(y_train)
    groups_train = np.asarray(groups_train)
    y_bin = (y_train != 0).astype(np.int64)

    n_groups = len(np.unique(groups_train))
    n_splits = min(n_inner_splits, n_groups)
    if n_splits < 2:
        raise ValueError(
            f"inner CV needs at least 2 well groups in the training fold, got {n_groups}."
        )
    if n_splits < n_inner_splits:
        logger.warning(
            "training fold has only %d wells; inner CV reduced from %d to %d splits",
            n_groups, n_inner_splits, n_splits,
        )

    device_kw = resolve_device(device)
    fixed = dict(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        verbosity=0,
        random_state=random_state,
        **device_kw,
        **(base_params or {}),
    )

    logger.info(
        "XGBoost search: %d/%d configurations, %d-fold grouped inner CV, device=%s",
        len(combos), n_total, n_splits, device_kw["device"],
    )

    gkf = GroupKFold(n_splits=n_splits)
    inner = list(gkf.split(X_train, y_bin, groups=groups_train))

    t0 = time.perf_counter()
    rows: list[dict] = []
    best_score, best_params = -np.inf, None

    if score_rows is not None:
        score_rows = np.asarray(score_rows).astype(bool)
        if len(score_rows) != len(y_train):
            raise ValueError(
                f"score_rows has {len(score_rows)} entries but there are "
                f"{len(y_train)} training rows"
            )
        inner = [(tr_i, va_i[score_rows[va_i]]) for tr_i, va_i in inner]

    # A single-class inner TRAINING side does not raise: XGBoost sets
    # n_classes_=1, ignores num_class=3, and predict_proba returns a
    # transposed (3, 2n) array that fails much later. One real fold hits this.
    usable, dropped = [], []
    for tr_i, va_i in inner:
        if len(va_i) == 0 or len(np.unique(y_bin[va_i])) < 2:
            dropped.append("val single-class")
        elif len(np.unique(y_train[tr_i])) < 2:
            dropped.append("train single-class")
        else:
            usable.append((tr_i, va_i))
    if dropped:
        logger.warning(
            "inner CV: %d of %d splits unusable (%s)",
            len(dropped), len(inner), ", ".join(sorted(set(dropped))),
        )
    if not usable:
        raise RuntimeError(
            "no usable inner CV split: every split had a single-class training or "
            "validation side. This training fold's positives are concentrated in too "
            "few wells to tune on -- reduce the outer fold count, or pass --no-tune."
        )

    for combo in combos:
        scores = []
        for tr_i, va_i in usable:
            clf = XGBClassifier(**fixed, **combo)
            clf.fit(
                X_train[tr_i], y_train[tr_i],
                sample_weight=compute_sample_weight(y_train[tr_i]),
            )
            proba = clf.predict_proba(X_train[va_i])
            if proba.shape != (len(va_i), 3):
                raise AssertionError(
                    f"predict_proba returned {proba.shape}, expected {(len(va_i), 3)}"
                )
            scores.append(
                float(average_precision_score(y_bin[va_i], positive_score(proba)))
            )
        skipped = len(inner) - len(usable)

        mean = float(np.mean(scores)) if scores else float("nan")
        rows.append({**combo, "mean_val_pr_auc": mean,
                     "n_inner_scored": len(scores), "n_inner_skipped": skipped})
        if scores and mean > best_score:
            best_score, best_params = mean, dict(combo)

    seconds = time.perf_counter() - t0
    trials = pd.DataFrame(rows).sort_values("mean_val_pr_auc", ascending=False)
    n_scored = int(trials["mean_val_pr_auc"].notna().sum())

    if best_params is None:
        raise RuntimeError("no configuration could be scored")

    result = SearchResult(
        best_params=best_params,
        best_val_pr_auc=best_score,
        n_configs_tried=n_total,
        n_configs_scored=n_scored,
        seconds=seconds,
        trials=trials,
    )
    logger.info("%s", result.summary())
    return result
