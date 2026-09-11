"""Feature importance.

gain is computed on TRAINING data and favours high-cardinality features;
permutation is out-of-sample against validation PR-AUC. Physical claims
should cite permutation.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.baselines.features import PRESENCE_SUFFIX

logger = logging.getLogger(__name__)


def _parse_name(name: str) -> tuple[str, str, float]:
    """Split "<channel>|<stat>|scale<f>" or "<channel>|presence_frac".
    Splitting on "_" silently mis-parsed every `last_diff` feature."""
    parts = name.split("|")
    if len(parts) == 2 and parts[1] == PRESENCE_SUFFIX:
        return parts[0], PRESENCE_SUFFIX, float("nan")
    if len(parts) == 3 and parts[2].startswith("scale"):
        try:
            scale = float(parts[2][len("scale"):])
        except ValueError:
            scale = float("nan")
        return parts[0], parts[1], scale
    logger.warning("feature name %r does not match the expected grammar", name)
    return name, "unknown", float("nan")


def _decorate(df: pd.DataFrame) -> pd.DataFrame:
    parsed = [_parse_name(n) for n in df["feature"]]
    df = df.copy()
    df["channel"] = [p[0] for p in parsed]
    df["stat"] = [p[1] for p in parsed]
    df["scale"] = [p[2] for p in parsed]
    df = df.reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df


def summarize_importances(model, feature_names: list) -> pd.DataFrame:
    """feature_names must come from the extractor that built the training
    matrix. Length is checked; order cannot be, so do not hand-build it."""
    if hasattr(model, "model") and hasattr(model.model, "feature_importances_"):
        model = model.model
    if not hasattr(model, "feature_importances_"):
        raise TypeError(
            f"{type(model).__name__} has no feature_importances_; pass a fitted "
            f"tree-based model or an XGBoostBaseline"
        )

    importances = np.asarray(model.feature_importances_, dtype=np.float64)
    if len(importances) != len(feature_names):
        raise ValueError(
            f"model has {len(importances)} importances but {len(feature_names)} feature "
            f"names were given -- a mismatched list renames every row silently."
        )

    df = pd.DataFrame({"feature": list(feature_names), "importance": importances})
    df = df.sort_values("importance", ascending=False)
    return _decorate(df)[["rank", "feature", "importance", "channel", "stat", "scale"]]


def permutation_importance_table(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list,
    *,
    n_repeats: int = 5,
    random_state: int = 42,
    scoring: str = "pr_auc",
) -> pd.DataFrame:
    """X_val is the FEATURE matrix, y_val the 3-class labels. Positive
    importance = load bearing."""
    from sklearn.metrics import average_precision_score

    from src.eval.metrics import positive_score

    X_val = np.asarray(X_val)
    y_val = np.asarray(y_val)
    if X_val.shape[1] != len(feature_names):
        raise ValueError(
            f"X_val has {X_val.shape[1]} columns but {len(feature_names)} names were given"
        )
    y_bin = (y_val != 0).astype(np.int64)
    if len(np.unique(y_bin)) < 2:
        raise ValueError(
            "validation fold is single-class -- permutation importance against PR-AUC "
            "is undefined here."
        )
    if scoring != "pr_auc":
        raise ValueError(f"only 'pr_auc' scoring is supported, got {scoring!r}")

    predict = model.predict_proba

    def _score(mat: np.ndarray) -> float:
        return float(average_precision_score(y_bin, positive_score(predict(mat))))

    base = _score(X_val)
    rng = np.random.default_rng(random_state)

    drops = np.zeros(X_val.shape[1], dtype=np.float64)
    stds = np.zeros(X_val.shape[1], dtype=np.float64)
    work = X_val.copy()
    for j in range(X_val.shape[1]):
        original = work[:, j].copy()
        reps = np.empty(n_repeats, dtype=np.float64)
        for r in range(n_repeats):
            work[:, j] = rng.permutation(original)
            reps[r] = base - _score(work)
        work[:, j] = original
        drops[j] = reps.mean()
        stds[j] = reps.std(ddof=1) if n_repeats > 1 else 0.0

    df = pd.DataFrame({
        "feature": list(feature_names),
        "importance": drops,
        "importance_std": stds,
    }).sort_values("importance", ascending=False)
    out = _decorate(df)
    out.attrs["baseline_pr_auc"] = base
    return out[["rank", "feature", "importance", "importance_std", "channel", "stat", "scale"]]


def _share(totals: pd.Series) -> pd.Series:
    """Fraction of total POSITIVE importance. Permutation values are signed,
    and a near-zero sum produces shares that are negative or above 1 -- exactly
    on no-signal folds. NaN means no attribution available."""
    pos = totals.clip(lower=0.0)
    denom = pos.sum()
    if not np.isfinite(denom) or denom <= 0:
        return pd.Series(np.nan, index=totals.index)
    return pos / denom


def channel_importance(df: pd.DataFrame) -> pd.DataFrame:
    """One row per physical sensor."""
    agg = (
        df.groupby("channel")["importance"]
        .agg(total="sum", mean="mean", max="max", n_features="size")
        .sort_values("total", ascending=False)
        .reset_index()
    )
    agg["share"] = _share(agg["total"])
    agg.insert(0, "rank", np.arange(1, len(agg) + 1))
    return agg


def stat_importance(df: pd.DataFrame) -> pd.DataFrame:
    """One row per statistic."""
    agg = (
        df.groupby("stat")["importance"]
        .agg(total="sum", mean="mean", n_features="size")
        .sort_values("total", ascending=False)
        .reset_index()
    )
    agg["share"] = _share(agg["total"])
    return agg


def scale_importance(df: pd.DataFrame) -> pd.DataFrame:
    """One row per timescale."""
    agg = (
        df.dropna(subset=["scale"])
        .groupby("scale")["importance"]
        .agg(total="sum", mean="mean", n_features="size")
        .sort_values("total", ascending=False)
        .reset_index()
    )
    agg["share"] = _share(agg["total"])
    return agg
