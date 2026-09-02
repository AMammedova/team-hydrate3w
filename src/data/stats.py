"""Member 1, W1.10 — dataset figures for the paper: transient-duration
histogram, an annotated raw multi-variable trace (the paper's opening
figure), variable-availability bar chart."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.contract import EVENT_CODE, TRANSIENT_OFFSET
from src.data.inventory import InstanceRecord

# The channels the primary cache is built on (DATA_FINDINGS.md sec 9) -- the
# sensible default for a figure, since they are the ones a model actually sees.
PLOT_CHANNELS = ("P-MON-CKP", "P-JUS-CKGL", "T-TPT", "T-JUS-CKP", "P-ANULAR")

# Shared with src/eval/plots.py:plot_annotated_trace -- same zones, same colors,
# so the "opening figure" (this module) and the "results figure" (eval/plots.py)
# read as one visual language.
_ZONE_COLOR = {
    "normal": "#ffffff",
    "transient": "#f5c15d",
    "established": "#e35d5d",
    "unlabeled": "#d9d9d9",
}


def _state_spans(df: pd.DataFrame, event_code: int = EVENT_CODE) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """
    Collapses the per-timestep raw `class` column into maximal (start, end,
    label) runs, label in {"normal", "transient", "established", "unlabeled"}.

    Deliberately reimplemented here rather than importing windowing's private
    `_raw_label_to_state`: that function is internal to the labeling pipeline,
    while this only needs the three raw-code comparisons for a plot legend --
    duplicating three comparisons is cheaper than exposing a private helper
    across module boundaries.
    """
    raw = df["class"].to_numpy(dtype="float64")
    label = np.where(
        np.isnan(raw), "unlabeled",
        np.where(raw == event_code, "established",
                 np.where(raw == TRANSIENT_OFFSET + event_code, "transient", "normal")),
    )
    n = len(label)
    if n == 0:
        return []
    change = np.flatnonzero(label[1:] != label[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, n]
    return [(df.index[s], df.index[e - 1], label[s]) for s, e in zip(starts, ends)]


def _shade_state_zones(ax, df: pd.DataFrame, event_code: int = EVENT_CODE) -> None:
    seen = set()
    for start, end, label in _state_spans(df, event_code):
        ax.axvspan(
            start, end, color=_ZONE_COLOR[label], alpha=0.35,
            label=label.capitalize() if label not in seen else None,
        )
        seen.add(label)


def _transient_durations(instances: list[InstanceRecord], event_code: int = EVENT_CODE) -> list[float]:
    """
    Seconds from the first to the last TRANSIENT-labeled sample, for every
    instance that has one. Real Event-9 instances have monotonic severity
    (DATA_FINDINGS.md sec 4 / windowing.is_monotonic_severity), so the
    transient span is a single contiguous run and "first to last" is the
    whole transient phase, not just one of several fragments.
    """
    durations = []
    for inst in instances:
        raw = inst.df["class"].to_numpy(dtype="float64")
        hit = np.flatnonzero(raw == TRANSIENT_OFFSET + event_code)
        if hit.size == 0:
            continue
        start, end = inst.df.index[hit[0]], inst.df.index[hit[-1]]
        durations.append((end - start).total_seconds())
    return durations


def transient_duration_histogram(
    instances: list[InstanceRecord],
    out_path: str,
    window_seconds: int = 1800,
) -> None:
    """
    Distribution of transient-phase durations, in hours.

    Annotated with the median and with `window_seconds` (the cache's window
    span, 60 decimated samples x 30 s = 30 min), because the comparison between
    those two is the argument the figure exists to make: the pre-measurement
    default of a 60-second window covered ~0.5% of a typical transient
    (DATA_FINDINGS.md sec 4), which is what forced the decimation change.
    """
    durations_h = [d / 3600.0 for d in _transient_durations(instances)]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    if durations_h:
        ax.hist(durations_h, bins=min(10, len(durations_h)), color="#4c72b0", edgecolor="white")
        median_h = float(np.median(durations_h))
        ax.axvline(median_h, color="#22303f", linestyle="--", linewidth=1.5,
                   label=f"median {median_h:.1f} h")
    ax.axvline(window_seconds / 3600.0, color="#e35d5d", linestyle=":", linewidth=1.8,
               label=f"window span {window_seconds / 60:.0f} min")
    ax.set_xlabel("Transient phase duration (hours)")
    ax.set_ylabel("Number of events")
    ax.set_title(f"Real transient phase duration (n={len(durations_h)} events)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def annotated_trace_figure(
    instance: InstanceRecord,
    out_path: str,
    columns: list[str] | None = None,
    max_channels: int = 5,
) -> None:
    """
    One real instance, raw trace + state-zone shading -- the paper's opening
    figure.

    One subplot per channel, sharing the x axis. Two reasons it is not a single
    overlaid axis: the channels differ by orders of magnitude (choke pressure in
    Pa against temperature in degrees C), which flattens everything but the
    largest onto the axis floor; and per-channel axes let a reader see which
    channel actually moves at onset, which is the whole point of the figure.

    `columns` defaults to the channels the cache is built on (PLOT_CHANNELS,
    those present in this instance), falling back to whichever channels vary at
    all. Taking the first N columns instead would pick the ABER-*/ESTADO-* valve
    states, which are constant for most instances and show nothing.
    """
    df = instance.df
    if columns is None:
        columns = [c for c in PLOT_CHANNELS if c in df.columns]
        if not columns:
            varying = [c for c in instance.variable_columns() if df[c].nunique(dropna=True) > 1]
            columns = varying[:max_channels]
    columns = columns[:max_channels]
    if not columns:
        raise ValueError(f"{instance.instance_id} has no plottable variable columns")

    fig, axes = plt.subplots(
        len(columns), 1, figsize=(10, 1.6 * len(columns) + 1), sharex=True, squeeze=False
    )
    for ax, col in zip(axes[:, 0], columns):
        _shade_state_zones(ax, df, instance.event_code)
        ax.plot(df.index, df[col], linewidth=0.8, color="#22303f")
        ax.set_ylabel(col, fontsize=8)
        ax.tick_params(labelsize=8)

    axes[0, 0].set_title(f"{instance.instance_id} ({instance.well_id})")
    # Zone legend once, on the top axis -- _shade_state_zones labels each zone
    # the first time it draws it, so the handles are already there.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, loc="upper left", fontsize=8, ncol=len(handles))
    axes[-1, 0].set_xlabel("Time")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
