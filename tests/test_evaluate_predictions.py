"""
Tests for Module 8 — evaluate_predictions.py (M5)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.contract import RESULTS_COLUMNS
from src.eval.aggregate import generate_all_tables, load_results, summarize_folds
from src.eval.evaluate_predictions import (
    evaluate_all,
    evaluate_single_run,
    parse_stem,
)


def test_parse_stem():
    res = parse_stem("xgboost_real_only_fold0_seed42")
    assert res["model"] == "xgboost"
    assert res["condition"] == "real_only"
    assert res["fold"] == 0
    assert res["seed"] == 42

    res_deep = parse_stem("tcn_real_plus_sim_fold2_seed123")
    assert res_deep["model"] == "tcn"
    assert res_deep["condition"] == "real_plus_sim"
    assert res_deep["fold"] == 2
    assert res_deep["seed"] == 123

    with pytest.raises(ValueError):
        parse_stem("invalid_stem_format")


def _create_mock_run_npz(val_path: Path, test_path: Path):
    # Validation split: 2 normal instances (5 windows each)
    n_val = 10
    val_inst = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    val_t = np.array([0, 60, 120, 180, 240, 0, 60, 120, 180, 240], dtype=float)
    val_y = np.zeros(n_val, dtype=np.int64)
    val_probs = np.zeros((n_val, 3), dtype=np.float32)
    val_probs[:, 0] = 0.95
    val_probs[:, 1] = 0.04
    val_probs[:, 2] = 0.01

    np.savez_compressed(
        val_path,
        probs=val_probs,
        y_true=val_y,
        group=np.array([10] * n_val),
        inst_id=val_inst,
        t_end=val_t,
        is_sim=np.zeros(n_val, dtype=np.uint8),
        failure_time=np.array([np.nan] * n_val),
        blockage_time=np.array([np.nan] * n_val),
        normal_hours_group=np.array([10]),
        normal_hours_value=np.array([100.0]),
    )

    # Test split: 1 normal instance (5 windows) + 1 positive instance (5 windows)
    n_test = 10
    test_inst = np.array([3, 3, 3, 3, 3, 4, 4, 4, 4, 4])
    test_t = np.array([0, 60, 120, 180, 240, 0, 60, 120, 180, 240], dtype=float)
    test_y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 2, 2], dtype=np.int64)
    test_probs = np.zeros((n_test, 3), dtype=np.float32)
    # Normal: low positive score
    test_probs[:5, 0] = 0.95
    test_probs[:5, 1] = 0.04
    test_probs[:5, 2] = 0.01
    # Positive: alarms at t=120, failure at t=200
    test_probs[5:, 0] = 0.1
    test_probs[5:, 1] = 0.6
    test_probs[5:, 2] = 0.3

    test_failure = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 200.0, 200.0, 200.0, 200.0, 200.0])

    np.savez_compressed(
        test_path,
        probs=test_probs,
        y_true=test_y,
        group=np.array([20] * 5 + [30] * 5),
        inst_id=test_inst,
        t_end=test_t,
        is_sim=np.zeros(n_test, dtype=np.uint8),
        failure_time=test_failure,
        blockage_time=test_failure + 1000.0,
        normal_hours_group=np.array([20]),
        normal_hours_value=np.array([100.0]),
    )


def test_evaluate_single_run(tmp_path: Path):
    val_npz = tmp_path / "xgboost_real_only_fold0_seed42_val.npz"
    test_npz = tmp_path / "xgboost_real_only_fold0_seed42_test.npz"
    _create_mock_run_npz(val_npz, test_npz)

    rows, curve_df, per_well_rows = evaluate_single_run(
        val_npz,
        test_npz,
        smooth_window=1,
        min_duration=0,
        target_far=0.01,
    )

    metric_names = [r["metric_name"] for r in rows]
    assert "pr_auc" in metric_names
    assert "event_recall" in metric_names
    assert "far" in metric_names
    assert "lead_time" in metric_names
    assert "ece" in metric_names

    # Check that schema matches RESULTS_COLUMNS
    for r in rows:
        assert set(r.keys()) == set(RESULTS_COLUMNS)

    # Lead time should be failure_time (200) - first alarm onset (0 or 60 or 120)
    lead_time_row = next(r for r in rows if r["metric_name"] == "lead_time")
    assert not np.isnan(lead_time_row["value"])
    assert lead_time_row["value"] > 0

    assert not curve_df.empty
    assert "threshold" in curve_df.columns
    assert "false_alarm_rate" in curve_df.columns

    assert len(per_well_rows) == 1
    assert per_well_rows[0]["well_id"] == 30
    assert per_well_rows[0]["flagged"] is True


def test_evaluate_all_end_to_end(tmp_path: Path):
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    val_npz = outputs_dir / "tcn_real_only_fold0_seed42_val.npz"
    test_npz = outputs_dir / "tcn_real_only_fold0_seed42_test.npz"
    _create_mock_run_npz(val_npz, test_npz)

    out_results = tmp_path / "results.csv"
    figures_dir = tmp_path / "figures"
    tables_dir = tmp_path / "tables"

    metrics_df, per_well_df = evaluate_all(
        outputs_dir=outputs_dir,
        out_results=out_results,
        figures_dir=figures_dir,
        smooth_window=1,
        min_duration=0,
        target_far=0.01,
    )

    assert out_results.exists()
    df = load_results(str(out_results))
    assert len(df) == 8
    assert set(df.columns) == set(RESULTS_COLUMNS)

    # Check aggregation and table generation
    summary = summarize_folds(df)
    assert not summary.empty

    generate_all_tables(str(out_results), str(tables_dir) + "/")
    assert (tables_dir / "lead_time.tex").exists()
    assert (tables_dir / "event_recall.tex").exists()

    # Check figures
    assert (figures_dir / "lead_time_vs_far.png").exists()
    assert (figures_dir / "per_well_lead_time.png").exists()
