"""Member 3, W3.3 — hyperparameter search for XGBoostBaseline.

Runs on the TRAINING fold only, scored by grouped inner CV. The number of
configurations evaluated is recorded and reported, because the paper has to
show that the baseline and the deep models got a comparable tuning budget
(W2.3 / W3.9) -- a baseline that was tuned for 4 configurations while the TCN
got 40 is not a baseline, it is a strawman.

RETURN TYPE CHANGED (deliberately)
----------------------------------
search() used to return one flat dict that mixed hyperparameters with
metadata:

    {"n_estimators": 300, "max_depth": 5,
     "n_configs_tried": 12, "best_val_pr_auc": 0.41}

Splatting that into XGBClassifier(**best) is silently wrong. XGBoost accepts
the unknown keys, forwards them to the booster, and prints

    Parameters: { "best_val_pr_auc", "n_configs_tried" } are not used.

at WARNING level -- which nobody reads -- while the caller believes it
reproduced the tuned model. search() now returns a SearchResult whose
.best_params contains ONLY model-ready keyword arguments.

WHAT IS SCORED
--------------
Validation PR-AUC on positive_score() = P(Transient) + P(Established), the
same pre-registered reduction alarm.py and thresholds.py use. Tuning on
accuracy or on multiclass logloss would optimise a quantity nobody reports
(TEAM_5_MEMBERS.md §9 red line 6).

Sample weights are applied inside the search exactly as XGBoostBaseline.fit()
applies them at final fit time. Tuning an unweighted model and then shipping
a weighted one selects hyperparameters for a different problem.
"""

from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# A deliberately modest default grid: 2 x 2 x 2 x 2 = 16 configurations.
# Quoted in the report next to the deep models' budget.
DEFAULT_GRID = {
    "n_estimators": [300, 600],
    "max_depth": [3, 6],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
}


@dataclass
class SearchResult:
    """Outcome of one hyperparameter search.

    best_params
        Model-ready keyword arguments and NOTHING else -- safe to splat
        straight into XGBClassifier(**result.best_params).
    trials
        One row per configuration with its mean inner-CV PR-AUC, so the
        report can show the search actually explored something.
    """

    best_params: dict
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
    """Full grid, or `n_iter` configurations sampled from it without
    replacement when the grid is larger than the budget."""
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
    """Grouped inner-CV hyperparameter search on one training fold.

    Parameters
    ----------
    X_train
        FEATURE matrix (N, n_features) -- already through
        RollingFeatureExtractor.transform(). Not the raw (N, C, W) windows.
    mask_train
        Accepted and ignored: the mask has already been consumed by the
        feature extractor. Kept in the signature because the team's module
        contract lists it, and because dropping it silently would break the
        call sites in run_all.sh.
    y_train
        3-class window labels (0 Normal / 1 Transient / 2 Established).
    groups_train
        Well id per row. Inner CV is grouped on these, so a well never sits
        on both sides of an inner split -- the same defence the outer CV
        uses, for the same reason (DATA_FINDINGS.md §2).
    param_grid
        name -> list of values. Defaults to DEFAULT_GRID.
    n_iter
        Cap on configurations. None means the full grid.
    device
        "auto" (GPU when XGBoost can really see one), "cuda" or "cpu".
    score_rows
        Optional boolean mask over the training rows marking which ones may
        be SCORED in the inner CV. Inner training still uses everything.

        This exists for the `real_plus_sim` condition. That condition trains
        on real + simulated windows but is evaluated on real validation and
        test wells, and the simulated windows are about 90% positive against
        a 3% real positive rate. Scoring the inner CV on simulated rows would
        therefore pick the hyperparameters that best fit the simulator, then
        report them as the tuned baseline for real wells. Passing
        `score_rows=(is_sim == 0)` keeps inner scoring on the distribution
        the outer fold is actually judged on.

    Returns
    -------
    SearchResult
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
            f"inner CV needs at least 2 well groups in the training fold, got {n_groups}. "
            f"Reduce the outer fold count so training keeps more wells."
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

    # Restrict inner SCORING rows once, so every configuration is judged on
    # exactly the same rows (the comparison between configs stays paired).
    if score_rows is not None:
        score_rows = np.asarray(score_rows).astype(bool)
        if len(score_rows) != len(y_train):
            raise ValueError(
                f"score_rows has {len(score_rows)} entries but there are "
                f"{len(y_train)} training rows"
            )
        inner = [(tr_i, va_i[score_rows[va_i]]) for tr_i, va_i in inner]

    # Decide ONCE which inner splits are usable, so every configuration is
    # scored on exactly the same splits and the search stays paired.
    #
    # Two ways an inner split is unusable, and the second one bites hard:
    #
    #   * the inner VALIDATION side is single-class -> PR-AUC is undefined;
    #   * the inner TRAINING side is single-class -> XGBoost does not raise.
    #     It sets n_classes_=1, ignores num_class=3, and predict_proba
    #     returns a transposed nonsense array of shape (3, 2*n) whose first
    #     axis is not the sample axis at all. Feeding that to
    #     average_precision_score fails with an opaque length mismatch far
    #     from the cause. This is not hypothetical: with 3 outer folds, one
    #     of this dataset's training folds carries all of its positive
    #     windows in a SINGLE well, so grouped inner CV necessarily produces
    #     an inner training split with no positives at all.
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
            "few wells to tune on -- reduce the outer fold count so training keeps "
            "more positive wells, or pass --no-tune to use the default configuration."
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
                    f"predict_proba returned {proba.shape}, expected {(len(va_i), 3)}. "
                    f"XGBoost silently drops to n_classes_=1 on a single-class training "
                    f"split; the guard above should have caught this."
                )
            scores.append(
                float(average_precision_score(y_bin[va_i], positive_score(proba)))
            )
        skipped = len(inner) - len(usable)

        mean = float(np.mean(scores)) if scores else float("nan")
        rows.append({**combo, "mean_val_pr_auc": mean,
                     "n_inner_scored": len(scores), "n_inner_skipped": skipped})
        logger.debug("  %s -> %.4f (%d inner folds)", combo, mean, len(scores))
        if scores and mean > best_score:
            best_score, best_params = mean, dict(combo)

    seconds = time.perf_counter() - t0
    trials = pd.DataFrame(rows).sort_values("mean_val_pr_auc", ascending=False)
    n_scored = int(trials["mean_val_pr_auc"].notna().sum())

    if best_params is None:
        raise RuntimeError(
            "no configuration could be scored: every inner validation split was "
            "single-class. The training fold does not contain enough positive wells "
            "to tune on -- reduce the outer fold count or widen the channel set."
        )

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
