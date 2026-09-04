"""
Module 8 — the four figures. Owner split per TEAM_5_MEMBERS.md:
    - plot_annotated_trace            → M1 (builds on src/data/stats.py)
    - plot_lead_time_vs_false_alarm_rate → M5
    - plot_per_well_lead_time_box       → M5
    - plot_reliability_diagram          → M3 (owns calibrate.py)

The brief says one clear comparison plot and one honest results table
beat ten decorative ones (§9 of the DL brief). Resist adding a fifth.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.contract import EVENT_CODE
from src.data.stats import _shade_state_zones


# ---------------------------------------------------------------------------
# M1-owned figure — plot_annotated_trace
# ---------------------------------------------------------------------------
def plot_annotated_trace(
    instance_df: pd.DataFrame,
    alarms: dict,
    out_path: str,
) -> None:
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


# ---------------------------------------------------------------------------
# M5-owned figure — lead time vs false alarm rate curve
# ---------------------------------------------------------------------------
def plot_lead_time_vs_false_alarm_rate(
    curves: dict[str, pd.DataFrame],
    out_path: str,
    target_far: float = 1 / 100,
    figsize: tuple[int, int] = (8, 5),
) -> None:
    """One curve per model, from thresholds.select_threshold_curve().

    This is W4.3's 'most informative figure' — it shows the trade-off
    between false alarm rate and the ability to detect events early.
    A vertical reference line at target_far marks the operating point.

    Parameters
    ----------
    curves : dict[str, pd.DataFrame]
        {model_name: DataFrame with columns 'threshold', 'false_alarm_rate'
         and optionally 'lead_time'}.
    out_path : str
        Where to save the figure (.png).
    target_far : float
        The reference false-alarm rate to mark on the plot (default 1/100 h).
    figsize : tuple
        Figure dimensions.
    """
    import matplotlib.ticker as ticker  # noqa: F401 — available for callers

    fig, ax = plt.subplots(figsize=figsize)

    # Color palette for up to 4 models
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    markers = ["o", "s", "^", "D"]

    for idx, (model_name, curve_df) in enumerate(curves.items()):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        if "lead_time" in curve_df.columns and "false_alarm_rate" in curve_df.columns:
            # Sort by false_alarm_rate for a clean curve
            sorted_df = curve_df.sort_values("false_alarm_rate")
            ax.plot(
                sorted_df["false_alarm_rate"],
                sorted_df["lead_time"],
                label=model_name,
                color=color,
                linewidth=2,
                marker=marker,
                markevery=max(1, len(sorted_df) // 10),
                markersize=5,
            )
        elif "false_alarm_rate" in curve_df.columns and "threshold" in curve_df.columns:
            # If only threshold + FAR available (before full eval), plot FAR vs threshold
            sorted_df = curve_df.sort_values("threshold")
            ax.plot(
                sorted_df["threshold"],
                sorted_df["false_alarm_rate"],
                label=model_name,
                color=color,
                linewidth=2,
            )

    # Reference line at target FAR
    ax.axvline(
        x=target_far, color="red", linestyle="--", linewidth=1, alpha=0.7,
        label=f"Target FAR = {target_far:.4f}/h",
    )

    ax.set_xlabel("False Alarms per Operating Hour", fontsize=11)
    ax.set_ylabel("Lead Time (seconds)", fontsize=11)
    ax.set_title("Lead Time vs False Alarm Rate", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# M5-owned figure — per-well lead time box plot
# ---------------------------------------------------------------------------
def plot_per_well_lead_time_box(
    fold_metrics: pd.DataFrame,
    out_path: str,
    figsize: tuple[int, int] = (10, 5),
) -> None:
    """Box plot of lead times grouped by well, optionally split by model.

    Parameters
    ----------
    fold_metrics : pd.DataFrame
        Must contain at least 'well_id' and 'lead_time' columns.
        Optionally 'model' for color-coded comparison.
    out_path : str
        Where to save the figure (.png).
    figsize : tuple
        Figure dimensions.
    """
    if "well_id" not in fold_metrics.columns or "lead_time" not in fold_metrics.columns:
        # Nothing to plot — create an empty figure with an explanation
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5, 0.5,
            "No per-well lead time data available yet.\n"
            "Requires completed model runs (M3/M4) and threshold selection (M5).",
            ha="center", va="center", fontsize=11, transform=ax.transAxes,
        )
        ax.set_title("Per-Well Lead Time Distribution", fontsize=12, fontweight="bold")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    # Filter out rows where lead_time is NaN (missed events)
    df = fold_metrics.dropna(subset=["lead_time"]).copy()
    if df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "All events missed — no lead times to plot.",
                ha="center", va="center", fontsize=11, transform=ax.transAxes)
        ax.set_title("Per-Well Lead Time Distribution", fontsize=12, fontweight="bold")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=figsize)

    # Convert lead time to minutes for readability
    df["lead_time_min"] = df["lead_time"] / 60.0

    hue_col = "model" if "model" in df.columns and df["model"].nunique() > 1 else None

    try:
        import seaborn as sns
        sns.boxplot(
            data=df, x="well_id", y="lead_time_min", hue=hue_col,
            ax=ax, palette="Set2", showfliers=True,
        )
    except ImportError:
        # Fallback without seaborn
        wells = sorted(df["well_id"].unique())
        data_per_well = [df[df["well_id"] == w]["lead_time_min"].values for w in wells]
        ax.boxplot(data_per_well, labels=wells)

    ax.set_xlabel("Well ID", fontsize=11)
    ax.set_ylabel("Lead Time (minutes)", fontsize=11)
    ax.set_title("Per-Well Lead Time Distribution", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", alpha=0.3)

    if hue_col:
        ax.legend(fontsize=9, title="Model", loc="best")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# M3-owned figure — reliability diagram
# ---------------------------------------------------------------------------
def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob_before: np.ndarray,
    y_prob_after: np.ndarray,
    out_path: str,
    n_bins: int = 10,
) -> None:
    """Before/after calibration reliability diagram.
    Uses M3's src/baselines/calibrate.py output.

    M3 implements this — it needs the before/after probability arrays
    from fit_calibrator().

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth (1 = positive event).
    y_prob_before : np.ndarray
        Predicted probabilities before calibration.
    y_prob_after : np.ndarray
        Predicted probabilities after calibration (isotonic/Platt).
    out_path : str
        Where to save the figure (.png).
    n_bins : int
        Number of calibration bins.
    """
    from sklearn.calibration import calibration_curve
    from src.eval.metrics import expected_calibration_error
    import matplotlib.pyplot as plt

    # Calculate ECE for both
    ece_before = expected_calibration_error(y_true, y_prob_before, n_bins)
    ece_after = expected_calibration_error(y_true, y_prob_after, n_bins)

    # Compute calibration curves
    prob_true_before, prob_pred_before = calibration_curve(y_true, y_prob_before, n_bins=n_bins, strategy='uniform')
    prob_true_after, prob_pred_after = calibration_curve(y_true, y_prob_after, n_bins=n_bins, strategy='uniform')

    fig, ax = plt.subplots(figsize=(7, 7))

    # Perfectly calibrated diagonal line
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")

    # Before calibration curve
    ax.plot(
        prob_pred_before, prob_true_before, 
        marker="s", color="#FF5722", label=f"Before Calibration (ECE = {ece_before:.3f})", linewidth=1.5
    )

    # After calibration curve
    ax.plot(
        prob_pred_after, prob_true_after, 
        marker="o", color="#4CAF50", label=f"After Calibration (ECE = {ece_after:.3f})", linewidth=2.0
    )

    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Fraction of Positives", fontsize=11)
    ax.set_title("Reliability Diagram (Calibration Curve)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
