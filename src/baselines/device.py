"""Member 3 — honest device selection for the XGBoost baseline.

Kept in its own module so src/baselines/xgb_model.py, which the team doc
lists as already finished, does not have to change.
"""

from __future__ import annotations

import json

import numpy as np


def resolve_device(prefer: str = "auto") -> dict:
    """XGBoost keyword arguments for the device that is ACTUALLY available.

    Returns e.g. {"device": "cuda", "tree_method": "hist"} or
    {"device": "cpu", "tree_method": "hist"}.

    Why this is not a one-line `torch.cuda.is_available()` check:

    1. XGBoost ships its own CUDA runtime, so it can have a usable GPU when
       the installed torch is a CPU build, and vice versa. This machine is
       exactly that case in reverse -- torch is 2.x+cpu -- so asking torch
       answers a different question.
    2. XGBoost 2.x does NOT raise when `device="cuda"` is requested on a
       machine with no GPU. It logs

           No visible GPU is found, setting device to CPU.

       at WARNING level, which nothing surfaces, and trains on CPU anyway. A
       probe that only checks "did fit() raise?" therefore reports a GPU run
       that never happened -- and that claim would go straight into the
       report's reproducibility section. Verified against xgboost 2.0.3.

    So the probe fits a two-row model and then reads the device back out of
    the fitted booster's own config, which is the only answer XGBoost will
    not overstate.

    prefer="cuda" raises instead of falling back, so a run that was meant to
    be on the GPU fails loudly rather than quietly producing CPU numbers.
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
        config = json.loads(probe.get_booster().save_config())
        resolved = config["learner"]["generic_param"]["device"]
    except Exception:
        resolved = "cpu"

    if resolved.startswith("cuda"):
        return {"device": "cuda", "tree_method": "hist"}
    if prefer == "cuda":
        raise RuntimeError(
            "device='cuda' was requested but XGBoost resolved to CPU -- no visible GPU. "
            "Use --device auto to fall back deliberately, or run on the GPU machine."
        )
    return {"device": "cpu", "tree_method": "hist"}
