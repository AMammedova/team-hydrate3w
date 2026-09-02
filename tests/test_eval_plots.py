"""
Smoke tests for src/eval/plots.py:plot_annotated_trace (Module 8, Member 4,
built on Member 1's src/data/stats.py zone-shading helper).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.plots import plot_annotated_trace

EVENT = 9
T = 100 + EVENT
E = EVENT
N = 0


def _make_df(class_sequence: list) -> pd.DataFrame:
    n = len(class_sequence)
    idx = pd.date_range("2020-01-01", periods=n, freq="1s")
    df = pd.DataFrame({"P-PDG": np.arange(n, dtype="float64")}, index=idx)
    df["class"] = pd.array(class_sequence, dtype="Float64")
    return df


def test_plot_annotated_trace_with_alarms(tmp_path: Path):
    df = _make_df([N, N, T, T, E, E])
    out = tmp_path / "trace.png"
    plot_annotated_trace(df, {"xgb": 2.0, "tcn": 3.0}, str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_annotated_trace_with_no_alarms(tmp_path: Path):
    df = _make_df([N, N, N])
    out = tmp_path / "trace_no_alarms.png"
    plot_annotated_trace(df, {}, str(out))
    assert out.exists()
