"""Member 3, W3.5 — feature importance: which sensors and which statistics
carry the signal.

This is where the project earns a genuine domain paragraph rather than
another table, so the analysis has to be defensible enough to hang a physical
claim on.

TWO IMPORTANCE MEASURES, AND WHY BOTH
-------------------------------------
gain (`summarize_importances`)
    XGBoost's own accounting: the average training-loss reduction across the
    splits that used a feature. Free -- it falls out of the fitted booster --
    but it is computed on TRAINING data and is well known to favour features
    with many distinct values, which here means the continuous pressure
    statistics over the coarse presence fractions. Fine for "what did the
    trees lean on", not evidence about physics.

permutation (`permutation_importance_table`)
    Shuffle one column of the VALIDATION features, re-score, and record how
    far validation PR-AUC falls. Measures out-of-sample contribution to the
    metric actually reported, and inherits none of gain's cardinality bias.
    This is the one the Discussion's physical argument should cite.

Both share the "|"-separated name grammar emitted by
RollingFeatureExtractor.feature_names(), so a row can always be traced back
to (channel, statistic, timescale).

READING THE RESULT AGAINST THE PHYSICS
--------------------------------------
The 3W descriptor's hydrate-in-service-line mechanism is a growing
restriction: pressure upstream of the production choke climbs while the
temperature downstream of it falls as flow drops and Joule-Thomson cooling
sets in. Against the main-arm channel set that predicts P-MON-CKP and
T-JUS-CKP carrying most of the signal, with P-ANULAR -- the annulus, which
does not see the service line -- carrying little. `channel_importance()`
aggregates to exactly that granularity so the prediction can be checked
instead of asserted.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.baselines.features import PRESENCE_SUFFIX

logger = logging.getLogger(__name__)


def _parse_name(name: str) -> tuple[str, str, float]:
    """Split a feature name into (channel, stat, scale).

    Grammar is "<channel>|<stat>|scale<f>" or "<channel>|presence_frac";
    see RollingFeatureExtractor.feature_names().

    The separator is "|" for a reason. The previous convention was
    "<channel>_<stat>_scale<f>" split on "_", which silently mis-parsed every
    `last_diff` feature -- 3W channel names contain "-" and the stat name
    contains "_", so a third of the rows came back with stat "last" and no
    scale. Nothing raised; the importance table just quietly described
    features that did not exist.
    """
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
    """Gain importance from a fitted tree ensemble, joined to feature names.

    Parameters
    ----------
    model
        Fitted estimator exposing `.feature_importances_` (XGBClassifier,
        XGBoostBaseline's wrapped model, sklearn forests). An XGBoostBaseline
        is also accepted directly.
    feature_names
        MUST come from the same extractor instance that produced the training
        matrix -- `extractor.feature_names(channels)`. Length is checked; the
        order cannot be, so do not hand-build this list.

    Returns
    -------
    pd.DataFrame with columns rank, feature, importance, channel, stat, scale,
    sorted by importance descending.
    """
    if hasattr(model, "model") and hasattr(model.model, "feature_importances_"):
        model = model.model            # XGBoostBaseline -> wrapped XGBClassifier
    if not hasattr(model, "feature_importances_"):
        raise TypeError(
            f"{type(model).__name__} has no feature_importances_; pass a fitted "
            f"tree-based model or an XGBoostBaseline"
        )

    importances = np.asarray(model.feature_importances_, dtype=np.float64)
    if len(importances) != len(feature_names):
        raise ValueError(
            f"model has {len(importances)} importances but {len(feature_names)} feature "
            f"names were given. These must come from the extractor that built the "
            f"training matrix -- a mismatched list renames every row silently."
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
    """Out-of-sample permutation importance against validation PR-AUC.

    For each column: shuffle it `n_repeats` times, re-score, and report the
    mean DROP in the metric. A positive value means the feature was load
    bearing; a value around zero (or negative) means the model could have
    done without it.

    X_val must be the FEATURE matrix, y_val the 3-class labels for the same
    rows. Scoring uses positive_score() so this measures contribution to the
    quantity the paper actually reports.
    """
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
            "is undefined here. Use a fold whose validation split contains positives."
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
        "importance": drops,          # mean drop in validation PR-AUC
        "importance_std": stds,
    }).sort_values("importance", ascending=False)
    out = _decorate(df)
    out.attrs["baseline_pr_auc"] = base
    return out[["rank", "feature", "importance", "importance_std", "channel", "stat", "scale"]]


def _share(totals: pd.Series) -> pd.Series:
    """Fraction of the total POSITIVE importance.

    Permutation importances are signed: permuting an irrelevant column can
    raise the score by chance, giving a small negative drop. Dividing by the
    plain sum then misbehaves badly -- when the positives and negatives
    nearly cancel, the denominator approaches zero and "shares" come out
    negative or above 1. That is exactly what happens on a fold with no
    measurable signal, which is precisely the fold where a reader most needs
    the table not to look authoritative.

    So shares are computed over the positive part only, and are NaN when
    nothing has positive importance -- an honest "no attribution available"
    rather than a fabricated ranking.
    """
    pos = totals.clip(lower=0.0)
    denom = pos.sum()
    if not np.isfinite(denom) or denom <= 0:
        return pd.Series(np.nan, index=totals.index)
    return pos / denom


def channel_importance(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a feature-level table to one row per physical sensor.

    This is the granularity the physical argument is made at -- "the model
    leans on the pressure upstream of the choke" is a claim about a sensor,
    not about `P-MON-CKP|slope|scale0.5`.
    """
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
    """Collapse to one row per statistic (mean/std/.../presence_frac).

    Answers "is the model reading levels or reading trends?" -- if `slope`
    and `last_diff` outrank `mean`, the baseline is using the shape of the
    ramp rather than the absolute sensor level, which is the behaviour
    per-instance normalisation was introduced to force.
    """
    agg = (
        df.groupby("stat")["importance"]
        .agg(total="sum", mean="mean", n_features="size")
        .sort_values("total", ascending=False)
        .reset_index()
    )
    total = agg["total"].sum()
    agg["share"] = agg["total"] / total if total else np.nan
    return agg


def scale_importance(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per timescale -- the evidence for or against the
    multi-timescale design in the feature ablation."""
    agg = (
        df.dropna(subset=["scale"])
        .groupby("scale")["importance"]
        .agg(total="sum", mean="mean", n_features="size")
        .sort_values("total", ascending=False)
        .reset_index()
    )
    total = agg["total"].sum()
    agg["share"] = agg["total"] / total if total else np.nan
    return agg
