"""
Tests for Module 8 — thresholds.py
"""

from __future__ import annotations

import numpy as np

from src.eval.thresholds import select_threshold, select_threshold_curve


def test_select_threshold():
    # 2 validation instances, normal operating hours
    y_val_proba = {
        "inst1": np.array([0.1, 0.2, 0.6, 0.1, 0.1]),
        "inst2": np.array([0.1, 0.8, 0.9, 0.2, 0.1]),
    }
    y_val_time = {
        "inst1": np.array([0.0, 60.0, 120.0, 180.0, 240.0]),
        "inst2": np.array([0.0, 60.0, 120.0, 180.0, 240.0]),
    }
    
    # 2 instances * 5 mins = 10 mins = 0.166 hours
    # Target FAR = 1/100, meaning we want 0 alarms on this small sample
    val_normal_hours = 10.0 / 60.0
    
    thresh = select_threshold(
        y_val_proba=y_val_proba,
        y_val_time=y_val_time,
        val_normal_hours=val_normal_hours,
        smooth_window=1,
        min_duration=0,
        target_far=0.0, # We want 0 alarms
    )
    
    # Threshold must be higher than 0.9 to have 0 alarms
    assert thresh > 0.9


def test_select_threshold_curve():
    y_val_proba = {
        "inst1": np.array([0.1, 0.2, 0.6, 0.1, 0.1]),
        "inst2": np.array([0.1, 0.8, 0.9, 0.2, 0.1]),
    }
    y_val_time = {
        "inst1": np.array([0.0, 60.0, 120.0, 180.0, 240.0]),
        "inst2": np.array([0.0, 60.0, 120.0, 180.0, 240.0]),
    }
    
    val_normal_hours = 10.0 / 60.0
    
    df = select_threshold_curve(
        y_val_proba=y_val_proba,
        y_val_time=y_val_time,
        val_normal_hours=val_normal_hours,
        smooth_window=1,
        min_duration=0,
    )
    
    assert len(df) == 200
    assert "threshold" in df.columns
    assert "false_alarm_rate" in df.columns
    
    # Highest threshold should have 0 alarms
    assert df.loc[df["threshold"] == 1.0, "false_alarm_rate"].iloc[0] == 0.0
