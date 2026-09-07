"""
Tests for Module 8 — thresholds.py
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.thresholds import _adaptive_grid, select_threshold, select_threshold_curve


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
    # Target FAR = 0.0, meaning we want 0 alarms
    val_normal_hours = 10.0 / 60.0

    thresh = select_threshold(
        y_val_proba=y_val_proba,
        y_val_time=y_val_time,
        val_normal_hours=val_normal_hours,
        smooth_window=1,
        min_duration=0,
        target_far=0.0,  # We want 0 alarms
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

    # The adaptive grid stays within the score range; highest threshold suppresses all alarms
    assert df["false_alarm_rate"].iloc[-1] == 0.0


# --- Tests for the M3 §7a adaptive grid fix ---

def test_adaptive_grid_compressed_range():
    """M3 §7a: after Platt calibration scores live in [0, 0.002], not [0, 1].
    The adaptive grid must place all 200 points inside that narrow range,
    not outside it where they are useless."""
    # Simulate calibrated scores in fold 0: range [0.0000, 0.0021]
    compressed_proba = {
        "inst_normal_1": np.array([0.0001, 0.0005, 0.0010, 0.0015, 0.0021]),
        "inst_normal_2": np.array([0.0000, 0.0003, 0.0008, 0.0012, 0.0018]),
    }
    grid = _adaptive_grid(compressed_proba, n_points=200)
    assert len(grid) == 200
    # Every threshold must be within or just outside the compressed score range
    assert grid[0] <= 0.003, f"grid min {grid[0]} is above the score range"
    assert grid[-1] <= 0.003, f"grid max {grid[-1]} is above the score range"
    # At least some thresholds must be strictly below the max score (so alarms can fire)
    assert grid[0] < 0.0021


def test_adaptive_grid_wide_range():
    """For a model with scores spanning the full [0, 1] range, the adaptive
    grid should behave identically to linspace(0, 1, 200)."""
    wide_proba = {
        "inst1": np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
    }
    grid = _adaptive_grid(wide_proba, n_points=200)
    assert len(grid) == 200
    assert grid[0] <= 0.0
    assert grid[-1] >= 0.99


def test_adaptive_grid_empty_dict():
    """Empty proba dict must fall back gracefully to linspace(0, 1, 200)."""
    grid = _adaptive_grid({}, n_points=200)
    assert len(grid) == 200
    assert grid[0] == pytest.approx(0.0)
    assert grid[-1] == pytest.approx(1.0)


def test_compressed_scores_can_fire():
    """Regression: with linspace(0,1,200) and calibrated scores in [0,0.002],
    select_threshold returns 1.0 (model never fires). With the adaptive grid
    a threshold below the scores must be found and the model can fire."""
    # Simulate calibrated scores where the model should fire at FAR budget
    # Use a permissive FAR so it's easy to find a threshold that fires
    compressed_proba = {
        "inst_normal": np.array([0.0001, 0.0005, 0.0010, 0.0015, 0.0021]),
    }
    compressed_time = {
        "inst_normal": np.array([0.0, 60.0, 120.0, 180.0, 240.0]),
    }
    thresh = select_threshold(
        y_val_proba=compressed_proba,
        y_val_time=compressed_time,
        val_normal_hours=100.0,   # 100 h => target budget is 1 alarm
        smooth_window=1,
        min_duration=0,
        target_far=1 / 100,
    )
    # With the adaptive grid, the threshold must be within the score range
    # (not 1.0, which would mean no alarm can ever fire)
    assert thresh < 1.0, (
        f"select_threshold returned {thresh}: the model can never fire. "
        "This is the M3 §7a bug — switch to the adaptive grid."
    )
