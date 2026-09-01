"""
Unit tests for src/data/inventory.py (Module 1, W1.3).

These build a tiny synthetic 3W-shaped dataset directory on the fly, so
they run today without needing the real ~GB-scale 3W download -- exactly
the "get something running now" step the project statement recommends
before touching real data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.inventory import ThreeWDataset, _infer_source_and_well, TRANSIENT_OFFSET


def _make_instance_df(n_rows: int, class_sequence: list[int] | None = None) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n_rows, freq="1s")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "P-PDG": rng.normal(100, 1, n_rows),
            "T-TPT": rng.normal(50, 1, n_rows),
        },
        index=idx,
    )
    if class_sequence is None:
        class_sequence = [0] * n_rows
    assert len(class_sequence) == n_rows
    df["class"] = pd.array(class_sequence, dtype="Int64")
    return df


@pytest.fixture()
def dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    (root / "0").mkdir(parents=True)
    (root / "9").mkdir(parents=True)

    # Two real Normal instances, two distinct wells.
    _make_instance_df(120).to_parquet(root / "0" / "WELL-00001_20200101000000.parquet")
    _make_instance_df(150).to_parquet(root / "0" / "WELL-00002_20200102000000.parquet")

    # One real Event-9 instance on a THIRD well: transient then established.
    n = 200
    seq = [0] * 100 + [TRANSIENT_OFFSET + 9] * 60 + [9] * 40
    _make_instance_df(n, seq).to_parquet(root / "9" / "WELL-00003_20200103000000.parquet")

    # One simulated Event-9 instance.
    n_sim = 90
    seq_sim = [0] * 30 + [TRANSIENT_OFFSET + 9] * 30 + [9] * 30
    _make_instance_df(n_sim, seq_sim).to_parquet(root / "9" / "SIMULATED_00001.parquet")

    # A second simulated instance, to check pseudo-groups don't collapse
    # into a single shared group.
    _make_instance_df(n_sim, seq_sim).to_parquet(root / "9" / "SIMULATED_00002.parquet")

    return root


def test_source_detection():
    assert _infer_source_and_well("WELL-00019_20140124093303") == ("real", "WELL-00019")
    assert _infer_source_and_well("SIMULATED_00001")[0] == "simulated"
    assert _infer_source_and_well("DRAWN_00007")[0] == "drawn"


def test_simulated_instances_never_share_a_well_group():
    src1, well1 = _infer_source_and_well("SIMULATED_00001")
    src2, well2 = _infer_source_and_well("SIMULATED_00002")
    assert src1 == src2 == "simulated"
    assert well1 != well2, "each simulated instance must get its own pseudo-group"


def test_load_instances_counts(dataset_root: Path):
    ds = ThreeWDataset(dataset_root, event_code=9)
    records = ds.load_instances()
    sources = [r.source for r in records]
    assert sources.count("real") == 3       # 2 normal + 1 event-9 real
    assert sources.count("simulated") == 2
    assert len(records) == 5


def test_no_simulated_instance_shares_well_id_with_real(dataset_root: Path):
    ds = ThreeWDataset(dataset_root, event_code=9)
    records = ds.load_instances()
    real_wells = {r.well_id for r in records if r.source == "real"}
    sim_wells = {r.well_id for r in records if r.source == "simulated"}
    assert real_wells.isdisjoint(sim_wells)


def test_has_transient_and_established_flags(dataset_root: Path):
    ds = ThreeWDataset(dataset_root, event_code=9)
    records = {r.instance_id: r for r in ds.load_instances()}
    real_event = records["WELL-00003_20200103000000"]
    assert real_event.has_transient
    assert real_event.has_established

    normal_only = records["WELL-00001_20200101000000"]
    assert not normal_only.has_transient
    assert not normal_only.has_established


def test_summary_shape_and_columns(dataset_root: Path):
    ds = ThreeWDataset(dataset_root, event_code=9)
    summary = ds.summary()
    assert len(summary) == 5
    for col in ("instance_id", "well_id", "source", "n_timesteps",
                "duration_seconds", "has_transient", "has_established"):
        assert col in summary.columns
    # missing_frac columns should exist for both raw variables
    assert "missing_frac__P-PDG" in summary.columns
    assert "missing_frac__T-TPT" in summary.columns
    # synthetic data has no missing values
    assert (summary["missing_frac__P-PDG"] == 0).all()


def test_well_ids_filtering(dataset_root: Path):
    ds = ThreeWDataset(dataset_root, event_code=9)
    real_wells = ds.well_ids(source="real")
    sim_wells = ds.well_ids(source="simulated")
    assert len(real_wells) == 3
    assert len(sim_wells) == 2


def test_missing_root_dir_raises():
    with pytest.raises(FileNotFoundError):
        ThreeWDataset("/nonexistent/path/does/not/exist", event_code=9)
