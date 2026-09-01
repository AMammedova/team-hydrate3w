"""
Module 2 — Windowing, Labeling, and Missingness Handling.
See DL_Project_Statement_Hydrate3W.docx, section 5 (DL2.1-DL2.5), and
its Addendum A.5 for the label_rule default.

Interfaces are frozen; build_windows() is still TODO. Whoever picks this
up: read DL2.1b before hardcoding window_size=60 -- it's an unvalidated
default that must be checked against real transient durations once
Module 1's summary() has run. label_window() below IS implemented and
tested -- use it, don't reinvent it inline in build_windows().
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd

from src.contract import ESTABLISHED, EVENT_CODE, NORMAL, TRANSIENT, TRANSIENT_OFFSET
from src.data.inventory import InstanceRecord

_SEVERITY_ORDER = (NORMAL, TRANSIENT, ESTABLISHED)  # low -> high; index IS the severity


def _raw_label_to_state(raw_label: int, event_code: int = EVENT_CODE) -> int:
    """Map a raw 3W class code to one of NORMAL/TRANSIENT/ESTABLISHED for this event."""
    if raw_label == event_code:
        return ESTABLISHED
    if raw_label == TRANSIENT_OFFSET + event_code:
        return TRANSIENT
    return NORMAL  # 0 (true normal) and any unrelated event code both collapse to Normal here


def label_window(
    raw_labels: np.ndarray, event_code: int = EVENT_CODE, rule: str = "most_severe"
) -> int:
    """
    raw_labels: per-timestep raw 3W class codes within one window.
    Returns one of NORMAL / TRANSIENT / ESTABLISHED.

    rule="most_severe" (DEFAULT): label = the most severe state present
    anywhere in the window (Normal < Transient < Established). Fixes a
    real bug in naive "any_transient" labeling: a window that has
    already progressed to Established but also contains an earlier
    transient sample would get called "Transient" under any_transient,
    silently reporting a LESS advanced state than the window actually
    contains. most_severe can't make that mistake, since Established
    always outranks Transient in the max.

    IMPORTANT, corrected reasoning (an earlier version of this docstring
    claimed most_severe "flags onset immediately, unlike final_timestep"
    -- that claim was wrong and has been removed): under a causal
    sliding window, final_timestep ALSO flags a window as Transient the
    moment the window's last sample first becomes Transient -- it does
    not need to wait for the window to "fully advance" past onset. In
    fact, whenever a single instance's severity never decreases over
    time (Normal -> Transient -> Established, no reversion -- the
    expected case for one failure progression), most_severe and
    final_timestep are PROVABLY IDENTICAL for every window: the max of a
    non-decreasing sequence always equals its last element. See
    is_monotonic_severity() below -- run it during Module 1's inventory
    step on real Event-9 instances. If it's ever False, most_severe and
    final_timestep genuinely diverge for that instance and the
    difference is worth a closer look; if it's always True (the expected
    case), the choice between the two doesn't actually change any
    result, and most_severe stays the implemented default purely because
    it's a strict, free safety margin against that unverified edge case,
    not because of any onset-timing advantage.

    rule="final_timestep": label = state at the window's last sample.
    Clean causal "nowcast" framing; mathematically identical to
    most_severe whenever is_monotonic_severity() holds for the instance
    (see above). Kept as the standard sensitivity-sweep alternative.

    rule="majority": label = the state that occupies more than half the
    window. A middle ground; also kept as a sensitivity-sweep option.
    """
    if len(raw_labels) == 0:
        raise ValueError("label_window() called on an empty window")

    states = np.array([_raw_label_to_state(int(l), event_code) for l in raw_labels])

    if rule == "most_severe":
        return int(states.max())
    if rule == "final_timestep":
        return int(states[-1])
    if rule == "majority":
        counts = np.bincount(states, minlength=len(_SEVERITY_ORDER))
        return int(counts.argmax())
    raise ValueError(f"unknown label_rule {rule!r} -- expected 'most_severe', 'final_timestep', or 'majority'")


def is_monotonic_severity(raw_labels: np.ndarray, event_code: int = EVENT_CODE) -> bool:
    """
    True if this instance's per-timestep severity never decreases over
    time (Normal -> Transient -> Established, no reversion). Module 1
    should call this on every real Event-9 instance during the inventory
    step (W1.3) and report the fraction that pass. Under monotonic
    severity, label_window(rule="most_severe") and
    label_window(rule="final_timestep") are mathematically identical for
    every window drawn from that instance -- this is a decidable,
    checkable fact about the real data, not a design opinion, and it's
    what actually settles whether the label_rule choice matters at all
    for this event.
    """
    states = np.array([_raw_label_to_state(int(l), event_code) for l in raw_labels])
    return bool(np.all(np.diff(states) >= 0))


class VariableSelector:
    def __init__(self, max_missing_frac: float = 0.5, frozen_run_seconds: int = 60) -> None:
        self.max_missing_frac = max_missing_frac
        self.frozen_run_seconds = frozen_run_seconds
        self._kept_columns: list[str] | None = None

    def fit(self, instances: list[InstanceRecord]) -> "VariableSelector":
        # TODO: compute per-variable missing fraction across `instances`
        # (TRAINING instances only -- see DL2.1) and drop columns above
        # max_missing_frac. Store the kept column list on self.
        raise NotImplementedError

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: subset to self._kept_columns; detect frozen runs
        # (>= frozen_run_seconds consecutive identical readings) and
        # convert them to NaN rather than trusting them as real signal.
        raise NotImplementedError


def mask_missing(X: np.ndarray, fill: str = "ffill_then_mean") -> tuple[np.ndarray, np.ndarray]:
    """
    X: (n_windows, window_size, n_channels), NaNs present.
    Returns (X_filled, missing_mask), both same shape as X.
    """
    # TODO
    raise NotImplementedError


class WindowBuilder:
    def __init__(
        self,
        window_size: int = 60,   # seconds -- UNVALIDATED DEFAULT, see DL2.1b
        stride: int = 10,
        label_rule: str = "most_severe",   # "most_severe" | "final_timestep" | "majority"
        min_valid_frac: float = 0.5,
    ) -> None:
        self.window_size = window_size
        self.stride = stride
        self.label_rule = label_rule
        self.min_valid_frac = min_valid_frac

    def build_windows(
        self, instance: InstanceRecord
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            X: (n_windows, window_size, n_channels)
            y: (n_windows,) in {0: Normal, 1: Transient, 2: Established} -- see label_window()
            well_ids: (n_windows,) -- instance.well_id repeated
            window_end_time: (n_windows,) -- np.datetime64, required for
                the lead-time metric in Module 8
        """
        # TODO: slide a window of `window_size` seconds with `stride`
        # seconds step over instance.df; label each window by calling
        # label_window(raw_labels_in_window, event_code=instance.event_code,
        # rule=self.label_rule) -- do not reimplement the labeling logic
        # here. Drop windows below min_valid_frac valid (non-NaN) data.
        raise NotImplementedError
