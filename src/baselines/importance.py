"""Member 2, W2.5 — feature importance analysis: which features and
physical variables carry the signal. One of the few places this project
generates genuine domain insight -- write it up, don't just plot it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_importances(model, feature_names: list) -> "pd.DataFrame":
    """Extract and summarize feature importances from a fitted tree-based model.

    Works with XGBoost (feature_importances_ attr) and scikit-learn tree
    ensembles. Importance type is 'gain' for XGBoost (the default), which
    is the most meaningful for interpretability -- it measures the average
    training loss reduction across all splits using that feature.

    Parameters
    ----------
    model
        A fitted model with a .feature_importances_ attribute (XGBoost,
        RandomForest, GradientBoosting, etc.).
    feature_names : list
        Names corresponding to each importance value. Should be built by
        the feature extractor, e.g.:
            ["P-MON-CKP_mean_scale1.0", "P-MON-CKP_std_scale1.0", ...]

    Returns
    -------
    pd.DataFrame
        Columns: feature, importance, rank, channel, stat, scale
        Sorted descending by importance.
    """
    importances = np.asarray(model.feature_importances_, dtype=np.float64)

    if len(importances) != len(feature_names):
        raise ValueError(
            f"model has {len(importances)} importances but {len(feature_names)} "
            f"feature names were provided"
        )

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    # Parse the feature name convention used by RollingFeatureExtractor.
    # Convention: "<channel>_<stat>_scale<scale>" or "<channel>_presence_frac"
    channels, stats, scales = [], [], []
    for name in df["feature"]:
        parts = name.split("_")
        if "presence" in parts:
            # e.g. "P-MON-CKP_presence_frac"
            channels.append(parts[0])
            stats.append("presence_frac")
            scales.append(1.0)
        elif len(parts) >= 3:
            # e.g. "P-MON-CKP_mean_scale1.0"
            channels.append(parts[0])
            stats.append(parts[1])
            try:
                scales.append(float(parts[-1].replace("scale", "")))
            except ValueError:
                scales.append(float("nan"))
        else:
            channels.append(name)
            stats.append("unknown")
            scales.append(float("nan"))

    df["channel"] = channels
    df["stat"] = stats
    df["scale"] = scales

    return df[["rank", "feature", "importance", "channel", "stat", "scale"]]
