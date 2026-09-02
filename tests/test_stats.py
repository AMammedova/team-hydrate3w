"""
Unit tests for src/data/stats.py (Module 1, W1.10).

Numeric logic (_state_spans, _transient_durations) is tested directly
against hand-worked expectations; the plotting functions get a smoke test
(file gets written, no exception) since pixel content isn't worth asserting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.contract import TRANSIENT_OFFSET
from src.data.inventory import InstanceRecord
from src.data.stats import (
    _state_spans,
    _transient_durations,
    annotated_trace_figure,
    transient_duration_histogram,
)

EVENT = 9
T = TRANSIENT_OFFSET + EVENT
E = EVENT
N = 0


def _make_instance(instance_id: str, class_sequence: list) -> InstanceRecord:
    n = len(class_sequence)
    idx = pd.date_range("2020-01-01", periods=n, freq="1s")
    df = pd.DataFrame({"P-PDG": np.arange(n, dtype="float64")}, index=idx)
    df["class"] = pd.array(class_sequence, dtype="Float64")
    return InstanceRecord(
        instance_id=instance_id, well_id="WELL-00001", source="real",
        event_code=EVENT, filepath=Path(f"{instance_id}.parquet"), df=df,
    )


def test_state_spans_basic_runs():
    seq = [N, N, N, T, T, E, E, E]
    inst = _make_instance("x", seq)
    spans = _state_spans(inst.df, EVENT)
    labels = [label for _, _, label in spans]
    assert labels == ["normal", "transient", "established"]
    # first span covers indices 0-2, second 3-4, third 5-7
    assert spans[0][0] == inst.df.index[0]
    assert spans[0][1] == inst.df.index[2]
    assert spans[1][0] == inst.df.index[3]
    assert spans[2][1] == inst.df.index[7]


def test_state_spans_handles_nan_as_unlabeled():
    seq = [N, np.nan, np.nan, N]
    inst = _make_instance("x", seq)
    spans = _state_spans(inst.df, EVENT)
    assert [label for _, _, label in spans] == ["normal", "unlabeled", "normal"]


def test_transient_durations_hand_computed():
    # transient runs 3s..5s inclusive -> 2 seconds duration (index 3 to index 5)
    seq1 = [N, N, N, T, T, T, E, E]
    inst1 = _make_instance("a", seq1)
    # no transient at all
    seq2 = [N, N, N, N]
    inst2 = _make_instance("b", seq2)
    # transient runs a single sample -> 0 seconds
    seq3 = [N, T, N]
    inst3 = _make_instance("c", seq3)

    durations = _transient_durations([inst1, inst2, inst3], event_code=EVENT)
    assert durations == pytest.approx([2.0, 0.0])


def test_transient_duration_histogram_writes_file(tmp_path: Path):
    inst = _make_instance("a", [N, N, T, T, T, E])
    out = tmp_path / "hist.png"
    transient_duration_histogram([inst], str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_transient_duration_histogram_handles_no_events(tmp_path: Path):
    inst = _make_instance("a", [N, N, N])
    out = tmp_path / "hist_empty.png"
    transient_duration_histogram([inst], str(out))
    assert out.exists()


def test_annotated_trace_figure_writes_file(tmp_path: Path):
    inst = _make_instance("a", [N, N, T, T, E, E])
    out = tmp_path / "trace.png"
    annotated_trace_figure(inst, str(out))
    assert out.exists()
    assert out.stat().st_size > 0
