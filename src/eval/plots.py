"""
Module 8 — the four figures. Member 4, W4.5.

The brief says one clear comparison plot and one honest results table
beat ten decorative ones (§9 of the DL brief). Resist adding a fifth.
"""

from __future__ import annotations

import pandas as pd


def plot_annotated_trace(instance_df: pd.DataFrame, alarms: dict, out_path: str) -> None:
    """Raw sensor trace with the transient period shaded and each model's
    alarm onset marked as a vertical line. Built on top of Member 1's
    plotting helpers in src/data/stats.py."""
    raise NotImplementedError


def plot_lead_time_vs_false_alarm_rate(curves: dict, out_path: str) -> None:
    """One curve per model, from thresholds.select_threshold_curve()."""
    raise NotImplementedError


def plot_per_well_lead_time_box(fold_metrics: pd.DataFrame, out_path: str) -> None:
    raise NotImplementedError


def plot_reliability_diagram(y_true, y_prob_before, y_prob_after, out_path: str) -> None:
    """Before/after calibration (Member 2's src/baselines/calibrate.py output)."""
    raise NotImplementedError
