"""Device selection for the XGBoost baseline."""

from __future__ import annotations

import json

import numpy as np


def resolve_device(prefer: str = "auto") -> dict:
    """XGBoost kwargs for the device actually available.

    XGBoost 2.x does not raise when device="cuda" has no GPU -- it warns and
    trains on CPU. So probe, then read the device back off the fitted booster.
    prefer="cuda" raises rather than silently producing CPU numbers.
    """
    import xgboost as xgb

    if prefer not in ("auto", "cuda", "cpu"):
        raise ValueError(f"prefer must be 'auto', 'cuda' or 'cpu', got {prefer!r}")
    if prefer == "cpu":
        return {"device": "cpu", "tree_method": "hist"}

    try:
        probe = xgb.XGBClassifier(
            n_estimators=1, device="cuda", tree_method="hist", verbosity=0
        )
        probe.fit(np.zeros((4, 2), dtype=np.float32), np.array([0, 1, 0, 1]))
        resolved = json.loads(
            probe.get_booster().save_config()
        )["learner"]["generic_param"]["device"]
    except Exception:
        resolved = "cpu"

    if resolved.startswith("cuda"):
        return {"device": "cuda", "tree_method": "hist"}
    if prefer == "cuda":
        raise RuntimeError(
            "device='cuda' was requested but XGBoost resolved to CPU -- no visible GPU. "
            "Use --device auto to fall back deliberately."
        )
    return {"device": "cpu", "tree_method": "hist"}
