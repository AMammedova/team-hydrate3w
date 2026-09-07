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
    title: str = "Reliability diagram - XGBoost baseline",
) -> None:
    """Before/after calibration reliability diagram, with the bin counts that
    make it readable. M3-owned; consumes src/baselines/calibrate.py's output.

    WHY THE SECOND PANEL EXISTS
    ---------------------------
    The positive rate on this task is about 3%, so almost every window scores
    low and the high-probability bins can hold a handful of rows out of tens
    of thousands. A bare reliability curve draws those bins the same size as
    a bin holding 30,000 rows, and a reader cannot tell a real miscalibration
    from three unlucky windows. The lower panel is the per-bin count on a log
    axis; marker area in the upper panel is scaled by the same count. Any
    claim made from this figure has to survive looking at both.

    Parameters
    ----------
    y_true
        Binary ground truth (1 = Transient or Established). Pass (y != 0).
    y_prob_before, y_prob_after
        positive_score() before and after the fitted Calibrator.
    """
    from src.eval.metrics import expected_calibration_error

    y_true = np.asarray(y_true).astype(int).ravel()
    before = np.clip(np.asarray(y_prob_before, dtype=float).ravel(), 0.0, 1.0)
    after = np.clip(np.asarray(y_prob_after, dtype=float).ravel(), 0.0, 1.0)
    if not (len(y_true) == len(before) == len(after)):
        raise ValueError(
            f"length mismatch: y_true={len(y_true)}, before={len(before)}, after={len(after)}"
        )

    fig, (ax, ax_hist) = plt.subplots(
        2, 1, figsize=(7, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    single_class = len(np.unique(y_true)) < 2
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    def _curve(p):
        """Per-bin (mean predicted, observed fraction, count). Computed here
        rather than via sklearn's calibration_curve because that drops empty
        bins silently, and which bins were empty is exactly what the reader
        needs to know."""
        idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
        xs, ys, ns = [], [], []
        for b in range(n_bins):
            sel = idx == b
            n = int(sel.sum())
            ns.append(n)
            if n:
                xs.append(float(p[sel].mean()))
                ys.append(float(y_true[sel].mean()))
            else:
                xs.append(np.nan)
                ys.append(np.nan)
        return np.array(xs), np.array(ys), np.array(ns)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1,
            label="perfectly calibrated", zorder=1)

    series = [
        ("before calibration", before, "#FF5722", "s"),
        ("after calibration", after, "#4CAF50", "o"),
    ]
    for label, p, color, marker in series:
        xs, ys, ns = _curve(p)
        ece = expected_calibration_error(y_true, p, n_bins)
        brier = float(np.mean((p - y_true) ** 2))
        ok = ~np.isnan(xs)
        # Marker area tracks bin population, so a bin of 3 cannot masquerade
        # as a bin of 30,000.
        sizes = 20.0 + 180.0 * (ns[ok] / max(ns.max(), 1)) ** 0.5
        ax.plot(xs[ok], ys[ok], color=color, linewidth=1.6, alpha=0.9, zorder=2,
                label=f"{label} (ECE={ece:.3f}, Brier={brier:.3f})")
        ax.scatter(xs[ok], ys[ok], s=sizes, color=color, marker=marker,
                   edgecolor="white", linewidth=0.6, zorder=3)
        ax_hist.step(edges[:-1], np.maximum(ns, 0.7), where="post",
                     color=color, linewidth=1.4, label=label)

    base_rate = float(y_true.mean())
    ax.axhline(base_rate, color="#1565C0", linestyle=":", linewidth=1.2,
               label=f"base rate = {base_rate:.3f}")

    ax.set_ylabel("observed fraction of positives", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    if single_class:
        ax.text(0.5, 0.5, "validation fold is single-class: calibration undefined",
                ha="center", va="center", fontsize=11, color="crimson",
                transform=ax.transAxes)

    ax_hist.set_yscale("log")
    ax_hist.set_xlabel("predicted probability  P(Transient) + P(Established)", fontsize=11)
    ax_hist.set_ylabel("windows/bin", fontsize=9)
    ax_hist.grid(True, alpha=0.3)
    ax_hist.tick_params(labelsize=8)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
