"""
Module 8 — threshold selection. Member 4, W4.3.

Thresholds are selected on validation folds ONLY, at a fixed false-alarm
budget, then applied unchanged to test folds -- this is what makes the
lead-time comparison across models meaningful (same operating point for
everyone). Addendum decision: target_far = 1 false alarm per 100
operating-hours is the reported headline point; also sweep the full
curve (select_threshold_curve) so other budgets stay visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.alarm import alarm_times


def _adaptive_grid(y_val_proba: dict, n_points: int = 200) -> np.ndarray:
    """
    Build a data-adaptive threshold grid from the observed score quantiles
    of the validation instances.

    After Platt calibration the positive scores no longer span [0, 1] --
    M3_FINDINGS.md section 7a measured that in fold 0 the calibrated scores
    live in [0.0000, 0.0021], so linspace(0, 1, 200) would place 0 candidate
    thresholds inside the score range and select_threshold() would always
    return 1.0 (the model can never fire).

    Using score quantiles instead keeps 200 resolution points where the
    scores actually live.  Platt scaling is monotone, so sweeping over
    score quantiles and sweeping over raw scores give identical alarm
    decisions wherever the fixed grid could resolve them -- they add
    resolution the fixed grid was missing.

    A small epsilon is subtracted from the minimum so the lowest threshold
    is strictly below every score (threshold < score means alarm fires)
    and a small epsilon is added above the maximum so the highest threshold
    is guaranteed to suppress all alarms.
    """
    arrays = [np.asarray(v, dtype=float).ravel() for v in y_val_proba.values()]
    if not arrays:
        return np.linspace(0.0, 1.0, n_points)
    all_scores = np.concatenate(arrays)
    if len(all_scores) == 0:
        return np.linspace(0.0, 1.0, n_points)
    lo = float(all_scores.min())
    hi = float(all_scores.max())
    eps = max(1e-9, (hi - lo) * 1e-4)
    return np.linspace(max(0.0, lo - eps), min(1.0, hi + eps), n_points)


def select_threshold(
    y_val_proba: dict,       # {instance_id: (n_windows,) positive_score()}
    y_val_time: dict,        # {instance_id: (n_windows,) window_end_time}
    val_normal_hours: float,
    smooth_window: int,
    min_duration: float,
    target_far: float = 1 / 100,   # 1 false alarm per 100 operating-hours
) -> float:
    """
    Sweep candidate thresholds; for each, count false-alarm onsets
    (alarm.alarm_times) on Normal-only validation instances, divide by
    val_normal_hours, and pick the threshold whose false-alarm rate is
    closest to target_far without exceeding it (prefer under-alarming
    over over-alarming when no exact match exists).

    The threshold grid is built from the observed score quantiles rather
    than a fixed linspace(0, 1, 200).  See _adaptive_grid() for why.
    """
    thresholds = _adaptive_grid(y_val_proba)
    best_threshold = float(thresholds[-1])   # safest default: suppress all alarms
    closest_far = -1.0

    for thresh in thresholds:
        total_alarms = 0
        for inst_id in y_val_proba:
            alarms = alarm_times(
                proba=y_val_proba[inst_id], t=y_val_time[inst_id],
                smooth_window=smooth_window, threshold=thresh,
                min_duration=min_duration,
            )
            total_alarms += len(alarms)

        far = total_alarms / val_normal_hours

        if far <= target_far and far > closest_far:
            closest_far = far
            best_threshold = thresh

    return float(best_threshold)


def select_threshold_curve(
    y_val_proba: dict, y_val_time: dict, val_normal_hours: float,
    smooth_window: int, min_duration: float,
) -> "pd.DataFrame":
    """
    Full lead-time-vs-false-alarm-rate curve (W4.3's "most informative
    figure") -- one row per swept threshold, with its resulting
    false-alarm rate. plots.py consumes this directly.

    Uses the same data-adaptive grid as select_threshold() so the curve
    has uniform resolution across the actual score range.
    """
    import pandas as pd
    thresholds = _adaptive_grid(y_val_proba)
    records = []

    for thresh in thresholds:
        total_alarms = 0
        for inst_id in y_val_proba:
            alarms = alarm_times(
                proba=y_val_proba[inst_id], t=y_val_time[inst_id],
                smooth_window=smooth_window, threshold=thresh,
                min_duration=min_duration,
            )
            total_alarms += len(alarms)

        far = total_alarms / val_normal_hours
        records.append({"threshold": thresh, "false_alarm_rate": far})

    return pd.DataFrame(records)
