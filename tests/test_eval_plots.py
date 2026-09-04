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


def test_plot_lead_time_vs_false_alarm_rate(tmp_path: Path):
    from src.eval.plots import plot_lead_time_vs_false_alarm_rate
    curves = {
        "xgb": pd.DataFrame({"threshold": [0.1, 0.5], "false_alarm_rate": [0.05, 0.01], "lead_time": [1000, 500]}),
        "tcn": pd.DataFrame({"threshold": [0.2, 0.6], "false_alarm_rate": [0.08, 0.02], "lead_time": [1200, 600]})
    }
    out = tmp_path / "lead_time_far.png"
    plot_lead_time_vs_false_alarm_rate(curves, str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_per_well_lead_time_box(tmp_path: Path):
    from src.eval.plots import plot_per_well_lead_time_box
    metrics = pd.DataFrame({
        "well_id": ["WELL-1", "WELL-1", "WELL-2", "WELL-2"],
        "lead_time": [600, 1200, 300, np.nan],
        "model": ["xgb", "tcn", "xgb", "tcn"]
    })
    out = tmp_path / "per_well_box.png"
    plot_per_well_lead_time_box(metrics, str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_reliability_diagram(tmp_path: Path):
    from src.eval.plots import plot_reliability_diagram
    y_true = np.array([0, 1, 0, 1, 1])
    y_prob_before = np.array([0.1, 0.9, 0.2, 0.8, 0.7])
    y_prob_after = np.array([0.05, 0.95, 0.1, 0.9, 0.85])
    
    out = tmp_path / "reliability.png"
    plot_reliability_diagram(y_true, y_prob_before, y_prob_after, str(out))
    assert out.exists()
    assert out.stat().st_size > 0
