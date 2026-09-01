"""
Unit tests for src.eval.alarm. The causality test is the important one:
it proves alarm_times() at time t cannot be changed by modifying
probabilities strictly after t -- guards against the centered-smoothing
bug (np.convolve(mode="same")) regressing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.alarm import alarm_times, first_alarm_before, lead_time


def test_basic_alarm_fires():
    t = np.arange(30, dtype=float)
    proba = np.array([0.05] * 10 + [0.9] * 10 + [0.05] * 10)
    onsets = alarm_times(proba, t, smooth_window=3, threshold=0.5, min_duration=3)
    assert len(onsets) == 1
    assert onsets[0] >= 10.0  # can't fire before the signal actually rises


def test_no_alarm_below_threshold():
    t = np.arange(20, dtype=float)
    proba = np.full(20, 0.1)
    assert alarm_times(proba, t, smooth_window=3, threshold=0.5, min_duration=3) == []


def test_min_duration_debounces_a_brief_spike():
    t = np.arange(20, dtype=float)
    proba = np.array([0.05] * 9 + [0.95] + [0.05] * 10)  # one-sample spike
    onsets = alarm_times(proba, t, smooth_window=1, threshold=0.5, min_duration=3)
    assert onsets == []  # too brief to count as a real alarm


def test_causality_future_values_cannot_change_past_alarm():
    """
    THE regression test for the causal-smoothing bug. Two probability
    sequences are IDENTICAL up through index `split - 1`, and differ
    only from `split` onward. A causal (trailing) smoother must produce
    identical onsets before `split` for both -- if smoothing looked into
    the future (the old np.convolve(mode="same") bug), the run that
    jumps high right at `split` would leak into the smoothed values just
    before it and could trigger an onset before the data actually
    changed. This exact configuration (smooth_window=11, threshold=0.4)
    was verified against the old buggy implementation to actually catch
    it (onset spuriously appears at t=39, one step before split=40) --
    this is not a test that just happens to pass either way.
    """
    n, split = 60, 40
    t = np.arange(n, dtype=float)

    proba_a = np.full(n, 0.05)          # never rises
    proba_b = np.full(n, 0.05)
    proba_b[split:] = 0.95              # jumps high exactly at split
    assert np.array_equal(proba_a[:split], proba_b[:split])  # identical before split, by construction

    onsets_a = alarm_times(proba_a, t, smooth_window=11, threshold=0.4, min_duration=1)
    onsets_b = alarm_times(proba_b, t, smooth_window=11, threshold=0.4, min_duration=1)

    onsets_a_before_split = [o for o in onsets_a if o < t[split]]
    onsets_b_before_split = [o for o in onsets_b if o < t[split]]
    assert onsets_a_before_split == onsets_b_before_split == [], (
        "an onset appeared before the split point using information that only "
        "exists after it -- smoothing is looking into the future"
    )


def test_lead_time_basic():
    assert lead_time(failure_time=100.0, first_alarm_time=60.0) == 40.0


def test_lead_time_none_if_alarm_after_failure():
    assert lead_time(failure_time=100.0, first_alarm_time=150.0) is None


def test_lead_time_none_if_no_alarm():
    assert lead_time(failure_time=100.0, first_alarm_time=None) is None


def test_first_alarm_before_picks_earliest_qualifying_onset():
    onsets = [80.0, 30.0, 95.0, 10.0]
    assert first_alarm_before(onsets, failure_time=100.0) == 10.0


def test_first_alarm_before_ignores_onsets_after_failure():
    onsets = [150.0, 200.0]
    assert first_alarm_before(onsets, failure_time=100.0) is None
