"""
Unit tests for src.data.windowing.label_window(). Covers the exact bug a
naive "any_transient" rule has: a window that's already progressed to
Established must never be reported as Transient just because an earlier
transient sample also appears in it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.contract import ESTABLISHED, NORMAL, TRANSIENT, TRANSIENT_OFFSET
from src.data.windowing import label_window

EVENT = 9
T = TRANSIENT_OFFSET + EVENT  # 109, raw transient code
E = EVENT                     # 9, raw established code
N = 0                          # raw normal code


def test_all_normal():
    assert label_window(np.array([N, N, N, N]), event_code=EVENT) == NORMAL


def test_most_severe_default_is_most_severe():
    # default rule, no explicit `rule=` kwarg
    assert label_window(np.array([N, N, T, T]), event_code=EVENT) == TRANSIENT


def test_most_severe_never_downgrades_established_to_transient():
    """
    The exact bug scenario: window is [Normal, Transient, Transient,
    Established, Established]. Naive any_transient would call this
    "Transient" even though the window has already reached Established --
    silently reporting a LESS advanced state than what's actually there.
    most_severe must return ESTABLISHED, never TRANSIENT, here.
    """
    raw = np.array([N, T, T, E, E])
    assert label_window(raw, event_code=EVENT, rule="most_severe") == ESTABLISHED


def test_most_severe_flags_onset_as_soon_as_it_appears():
    """most_severe correctly flags onset the moment a transient sample
    enters the window. Note: final_timestep does this too in this exact
    case (the transient sample IS the last one) -- see
    test_most_severe_equals_final_timestep_when_monotonic for the general
    proof that the two rules agree whenever severity is monotonic."""
    raw = np.array([N, N, N, N, T])
    assert label_window(raw, event_code=EVENT, rule="most_severe") == TRANSIENT


def test_final_timestep_rule():
    raw = np.array([N, T, T, E, E])
    assert label_window(raw, event_code=EVENT, rule="final_timestep") == ESTABLISHED
    raw2 = np.array([N, N, T, N, N])  # ends Normal even though it dipped into Transient
    assert label_window(raw2, event_code=EVENT, rule="final_timestep") == NORMAL


def test_majority_rule():
    raw = np.array([N, N, T, T, T])  # 3/5 Transient
    assert label_window(raw, event_code=EVENT, rule="majority") == TRANSIENT
    raw2 = np.array([N, N, N, T, T])  # 3/5 Normal
    assert label_window(raw2, event_code=EVENT, rule="majority") == NORMAL


def test_unrelated_event_code_collapses_to_normal():
    """A raw label belonging to a DIFFERENT event type (not this event's
    transient/established codes) is not this task's positive class."""
    other_event_transient = TRANSIENT_OFFSET + 3  # some other event, not 9
    raw = np.array([N, other_event_transient, N])
    assert label_window(raw, event_code=EVENT) == NORMAL


def test_unknown_rule_raises():
    with pytest.raises(ValueError):
        label_window(np.array([N, N]), event_code=EVENT, rule="not_a_real_rule")


def test_empty_window_raises():
    with pytest.raises(ValueError):
        label_window(np.array([]), event_code=EVENT)


def test_monotonic_severity_detection():
    from src.data.windowing import is_monotonic_severity
    assert is_monotonic_severity(np.array([N, N, T, T, E, E]), event_code=EVENT) is True
    assert is_monotonic_severity(np.array([N, N, N, N]), event_code=EVENT) is True
    assert is_monotonic_severity(np.array([N, T, N, T]), event_code=EVENT) is False  # reverts T->N
    assert is_monotonic_severity(np.array([N, E, T]), event_code=EVENT) is False  # reverts E->T


def test_most_severe_equals_final_timestep_when_monotonic():
    """
    THE regression test for the corrected reasoning: whenever an
    instance's severity is monotonic (the expected case -- see
    is_monotonic_severity), most_severe and final_timestep must produce
    IDENTICAL labels for every window drawn from it. This is not an
    approximation -- max of a non-decreasing sequence always equals its
    last element -- verified here over many random monotonic sequences
    and many random window slices of each.
    """
    from src.data.windowing import is_monotonic_severity

    rng = np.random.default_rng(0)
    raw_by_state = {NORMAL: N, TRANSIENT: T, ESTABLISHED: E}

    for _ in range(200):
        n = rng.integers(3, 40)
        # build a monotonic *state* sequence, then map back to raw codes
        state_seq = np.sort(rng.integers(0, 3, n))
        raw_seq = np.array([raw_by_state[s] for s in state_seq])
        assert is_monotonic_severity(raw_seq, event_code=EVENT) is True

        # slice random windows out of it and confirm the two rules agree on every one
        for _ in range(5):
            start = rng.integers(0, n - 1)
            end = rng.integers(start + 1, n + 1)
            window = raw_seq[start:end]
            most_severe = label_window(window, event_code=EVENT, rule="most_severe")
            final_ts = label_window(window, event_code=EVENT, rule="final_timestep")
            assert most_severe == final_ts, (
                f"diverged on a monotonic sequence: window={window.tolist()}, "
                f"most_severe={most_severe}, final_timestep={final_ts}"
            )


def test_most_severe_and_final_timestep_CAN_diverge_when_non_monotonic():
    """The one case where the two rules genuinely differ: label noise /
    reversion within an instance. Confirms the tests above aren't
    vacuously true because the rules always agree regardless of input."""
    raw = np.array([N, T, N])  # reverts from Transient back to Normal -- non-monotonic
    assert label_window(raw, event_code=EVENT, rule="most_severe") == TRANSIENT
    assert label_window(raw, event_code=EVENT, rule="final_timestep") == NORMAL
