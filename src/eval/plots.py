"""
Module 8 — the four figures. Member 4, W4.5.

The brief says one clear comparison plot and one honest results table
beat ten decorative ones (§9 of the DL brief). Resist adding a fifth.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.contract import EVENT_CODE
from src.data.stats import _shade_state_zones


def plot_annotated_trace(instance_df: pd.DataFrame, alarms: dict, out_path: str) -> None:
    """Raw sensor trace with the transient period shaded and each model's
    alarm onset marked as a vertical line. Built on top of Member 1's
    plotting helpers in src/data/stats.py.

    `alarms`: model_name -> alarm time, in seconds from `instance_df`'s first
    sample (the same convention as WindowBuilder.build_windows()'s
    window_end_time / alarm.py's alarm times). Empty dict draws the zones
    with no alarm lines.
    """
    columns = [c for c in instance_df.columns if c not in ("class", "state")]
    fig, ax = plt.subplots(figsize=(10, 4))
    _shade_state_zones(ax, instance_df, EVENT_CODE)
    for col in columns:
        ax.plot(instance_df.index, instance_df[col], linewidth=0.8, label=col)

    t0 = instance_df.index[0]
    colors = plt.get_cmap("tab10").colors
    for i, (model, alarm_seconds) in enumerate(alarms.items()):
        ax.axvline(
            t0 + pd.Timedelta(seconds=alarm_seconds),
            color=colors[i % len(colors)], linestyle="--", linewidth=1.5,
            label=f"{model} alarm",
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Sensor value")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_lead_time_vs_false_alarm_rate(curves: dict, out_path: str) -> None:
    """One curve per model, from thresholds.select_threshold_curve()."""
    raise NotImplementedError


def plot_per_well_lead_time_box(fold_metrics: pd.DataFrame, out_path: str) -> None:
    raise NotImplementedError


def plot_reliability_diagram(y_true, y_prob_before, y_prob_after, out_path: str) -> None:
    """Before/after calibration (Member 2's src/baselines/calibrate.py output)."""
    raise NotImplementedError
