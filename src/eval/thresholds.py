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
    """
    thresholds = np.linspace(0, 1, 200)
    best_threshold = 1.0
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
    """
    import pandas as pd
    thresholds = np.linspace(0, 1, 200)
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
