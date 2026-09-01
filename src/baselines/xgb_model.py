"""
Module 4 (baseline) — see project statement section 7 (DL4.1-DL4.4) and
Addendum A.3 for the reconciled explicit-mask contract.

Class imbalance is handled via `sample_weight` in fit(), computed by the
caller (Member 2, W2.2) from the training fold's actual class counts --
NOT via `scale_pos_weight`. scale_pos_weight is a binary-classification-
only parameter in XGBoost (confirmed against XGBoost's own docs/forum);
with objective="multi:softprob" and num_class=3 it has no defined effect
and silently does nothing. An earlier version of this file exposed it as
a constructor parameter anyway, which was misleading -- removed.
"""

from __future__ import annotations

import numpy as np
import xgboost as xgb

from src.baselines.features import RollingFeatureExtractor


class XGBoostBaseline:
    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        random_state: int = 42,
        **xgb_params,
    ) -> None:
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            objective="multi:softprob",
            num_class=3,
            **xgb_params,
        )
        self.feature_extractor = RollingFeatureExtractor()

    def _featurize(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # X, mask: (n_windows, n_channels, window_size) -- channels-first.
        # RollingFeatureExtractor also folds in the presence-fraction from
        # mask as its own feature per channel (W2.1) -- a NaN and a real
        # zero are not the same thing to this baseline either.
        return self.feature_extractor.transform(X, mask)

    def fit(
        self, X: np.ndarray, mask: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> "XGBoostBaseline":
        X_feat = self._featurize(X, mask)
        self.model.fit(X_feat, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        X_feat = self._featurize(X, mask)
        return self.model.predict_proba(X_feat)

    def feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_


def compute_sample_weight(y: np.ndarray) -> np.ndarray:
    """
    Inverse-frequency sample weights from the TRAINING fold's class
    counts only (W2.2) -- pass the result straight into
    XGBoostBaseline.fit(..., sample_weight=...). This is the actual
    imbalance-handling mechanism for this model; scale_pos_weight is not
    used (see module docstring).
    """
    classes, counts = np.unique(y, return_counts=True)
    weight_per_class = {c: len(y) / (len(classes) * n) for c, n in zip(classes, counts)}
    return np.array([weight_per_class[label] for label in y])
