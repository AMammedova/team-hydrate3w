"""
Module 8 — alarm definition and lead time. Member 4, W4.2.

THIS DEFINITION IS FROZEN BEFORE RESULTS ARE SEEN. Once real numbers
start coming in, do not touch alarm_times()'s logic to make a curve look
better -- that is exactly what the rubric penalizes as tuning after
seeing results. Commit this file, date the commit, and treat it as done.

CAUSALITY: smoothing here MUST be trailing (uses only samples at or
before the current timestep), never centered. An earlier version of this
file used np.convolve(..., mode="same") with a symmetric kernel, which
is a centered moving average -- the smoothed value at time t depended on
p[t+1], p[t+2], ... i.e. on the future. For a system whose entire output
is "how much lead time before failure," that's a real leakage bug: it
makes the measured lead time artificially optimistic, using information
an online system would not have yet. Fixed below via a trailing rolling
mean; test_alarm.py's causality test guards against this regressing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _trailing_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling mean: output[t] = mean(x[max(0, t-window+1) : t+1]).
    Never looks at x[t+1:]. min_periods=1 avoids NaN at the start instead
    of requiring a full window before the first alarm can even fire."""
    return pd.Series(x).rolling(window=window, min_periods=1).mean().to_numpy()


def alarm_times(
    proba: np.ndarray,
    t: np.ndarray,
    smooth_window: int,
    threshold: float,
    min_duration: float,
) -> list[float]:
    """
    proba: (n_windows,) positive_score() output for ONE instance, in time order.
    t: (n_windows,) timestamps (seconds) matching proba, i.e. window_end_time.

    Smooth the positive-class probability with a TRAILING rolling
    average (causal -- see module docstring). Declare an alarm when the
    smoothed value stays above `threshold` for at least `min_duration`
    continuous seconds. Return the onset timestamps (one per alarm, not
    one per sample above threshold -- counting samples instead of onsets
    is the W4.1 pitfall that inflates the false-alarm rate by orders of
    magnitude).
    """
    if len(proba) == 0:
        return []

    smoothed = _trailing_mean(proba, smooth_window)
    above = smoothed >= threshold

    onsets: list[float] = []
    run_start_idx: int | None = None
    for i, is_above in enumerate(above):
        if is_above and run_start_idx is None:
            run_start_idx = i
        elif not is_above and run_start_idx is not None:
            if t[i - 1] - t[run_start_idx] >= min_duration:
                onsets.append(float(t[run_start_idx]))
            run_start_idx = None
    if run_start_idx is not None and t[-1] - t[run_start_idx] >= min_duration:
        onsets.append(float(t[run_start_idx]))

    return onsets


def lead_time(failure_time: float, first_alarm_time: float | None) -> float | None:
    """failure_time - first_alarm_time, for alarms occurring before failure.
    None if there was no alarm before failure (a miss, not a lead time of 0)."""
    if first_alarm_time is None or np.isnan(failure_time):
        return None
    if first_alarm_time >= failure_time:
        return None
    return failure_time - first_alarm_time


def first_alarm_before(onsets: list[float], failure_time: float) -> float | None:
    """First alarm onset strictly before failure_time, or None if there wasn't one."""
    before = [o for o in onsets if o < failure_time]
    return min(before) if before else None
