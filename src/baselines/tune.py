"""Member 2, W2.3 — hyperparameter search for XGBoostBaseline, validation
folds only. Record how many configurations were tried (paper needs this
number alongside Member 3's, per W2.3/W3.9 -- comparable tuning budget).
"""

from __future__ import annotations

import itertools
import logging

import numpy as np

logger = logging.getLogger(__name__)


def search(
    X_train: np.ndarray,
    mask_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    param_grid: dict,
) -> dict:
    """Grid search over XGBoost hyperparameters on validation folds.

    Uses 3-fold cross-validation on the provided training data to select
    the best configuration, scored by validation PR-AUC on the positive class.

    Parameters
    ----------
    X_train : np.ndarray
        Feature matrix, shape (N, n_features). Use RollingFeatureExtractor first.
    mask_train : np.ndarray
        Presence mask — used only for feature extraction (already applied here).
    y_train : np.ndarray
        Labels (0=Normal, 1=Transient, 2=Established). Shape (N,).
    groups_train : np.ndarray
        Well group ids (N,) — used for GroupKFold to prevent well leakage.
    param_grid : dict
        Dict of param_name -> list of values to try.
        Example: {"n_estimators": [100, 300], "max_depth": [3, 6],
                  "learning_rate": [0.05, 0.1], "subsample": [0.8]}

    Returns
    -------
    dict
        Best hyperparameters found. Includes "n_configs_tried".
    """
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import GroupKFold
    from xgboost import XGBClassifier

    from src.eval.metrics import positive_score

    # Build binary labels for PR-AUC (positive = any hydrate phase)
    y_bin = (y_train != 0).astype(np.int64)

    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    logger.info("XGBoost hyperparameter search: %d configurations", len(combos))

    best_score = -1.0
    best_params: dict = {}

    gkf = GroupKFold(n_splits=3)

    for combo in combos:
        params = dict(zip(keys, combo))
        scores = []

        for train_idx, val_idx in gkf.split(X_train, y_bin, groups=groups_train):
            X_tr, X_va = X_train[train_idx], X_train[val_idx]
            y_tr_bin, y_va_bin = y_bin[train_idx], y_bin[val_idx]
            y_tr_full = y_train[train_idx]

            if y_va_bin.min() == y_va_bin.max():
                # Val fold has no positives — skip this split
                continue

            clf = XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                use_label_encoder=False,
                verbosity=0,
                random_state=42,
                **params,
            )
            clf.fit(X_tr, y_tr_full)
            proba = clf.predict_proba(X_va)  # shape (N, 3)
            pos_score = positive_score(proba)
            scores.append(average_precision_score(y_va_bin, pos_score))

        if not scores:
            continue
        mean_score = float(np.mean(scores))
        logger.debug("  params=%s  val_PR-AUC=%.4f", params, mean_score)

        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    best_params["n_configs_tried"] = len(combos)
    best_params["best_val_pr_auc"] = round(best_score, 4)
    logger.info(
        "Best XGBoost params: %s (val PR-AUC=%.4f, tried %d configs)",
        {k: v for k, v in best_params.items() if k not in ("n_configs_tried", "best_val_pr_auc")},
        best_score,
        len(combos),
    )
    return best_params
