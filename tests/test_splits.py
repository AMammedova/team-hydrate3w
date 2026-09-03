"""
Unit tests for src/data/splits.py (Module 3, Member 2, W2.1-W2.3).

The synthetic fixture below mirrors the REAL well structure measured in
DATA_FINDINGS.md §2-§3, because the properties that matter here only show
up at that shape:

  * 7 positive wells carrying 14 transient instances, very unevenly
    (WELL-00042 alone has 5 of them),
  * 9 Normal wells carrying 3,377 h, even more unevenly (WELL-00002 has
    1220 h, WELL-00007 has 6 h),
  * the two populations disjoint, so a naive stratified split cannot see
    that a fold has run out of Normal hours.

A splitter that passes on a balanced toy fixture and fails on this one is
a splitter that will silently produce an unusable fold on the real data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.contract import ESTABLISHED, NORMAL, TRANSIENT
from src.data.splits import (
    load_cache,
    LEAVE_ONE_WELL_OUT,
    GroupedKFoldSplitter,
    load_cache_index,
)

# Real counts, DATA_FINDINGS.md §3: well -> number of transient instances.
POSITIVE_EVENTS = {
    "WELL-00042": 5,
    "WELL-00014": 3,
    "WELL-00037": 2,
    "WELL-00016": 1,
    "WELL-00020": 1,
    "WELL-00040": 1,
    "WELL-00041": 1,
}

# Real Normal hours, DATA_FINDINGS.md §2.
NORMAL_HOURS = {
    "WELL-00002": 1220.4,
    "WELL-00006": 674.8,
    "WELL-00001": 555.6,
    "WELL-00005": 353.1,
    "WELL-00008": 337.0,
    "WELL-00003": 154.8,
    "WELL-00019": 39.8,
    "WELL-00004": 35.8,
    "WELL-00007": 6.0,
}

N_SIM_WELLS = 4
WINDOWS_PER_INSTANCE = 10


def make_dataset(with_sim: bool = True):
    """
    Build (y, groups, instances, is_sim, well_hours) at the real shape.

    Each positive instance contributes a few Transient/Established windows
    plus Normal ones; each Normal well contributes Normal windows only.
    """
    y, groups, instances, is_sim = [], [], [], []

    def add(label, well, inst, sim=False, n=WINDOWS_PER_INSTANCE):
        y.extend([label] * n)
        groups.extend([well] * n)
        instances.extend([inst] * n)
        is_sim.extend([sim] * n)

    for well, n_events in POSITIVE_EVENTS.items():
        for e in range(n_events):
            inst = f"{well}_evt{e}"
            add(NORMAL, well, inst, n=6)
            add(TRANSIENT, well, inst, n=3)
            add(ESTABLISHED, well, inst, n=1)
        # hydrate wells also record instances with no positive phase at all
        add(NORMAL, well, f"{well}_quiet")

    for well in NORMAL_HOURS:
        add(NORMAL, well, f"{well}_normal")

    if with_sim:
        for s in range(N_SIM_WELLS):
            inst = f"SIMULATED_{s:05d}"
            add(NORMAL, f"SIM-{inst}", inst, sim=True, n=6)
            add(TRANSIENT, f"SIM-{inst}", inst, sim=True, n=3)
            add(ESTABLISHED, f"SIM-{inst}", inst, sim=True, n=1)

    return (
        np.array(y),
        np.array(groups),
        np.array(instances),
        np.array(is_sim),
        dict(NORMAL_HOURS),
    )


def folds_of(splitter, **overrides):
    y, groups, instances, is_sim, hours = make_dataset(**overrides)
    return list(
        splitter.iter_folds(
            y, groups, is_sim=is_sim, instances=instances, well_hours=hours
        )
    )


# --------------------------------------------------------------------------
# Red line 1: a well is never on two sides of a split
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_splits", [2, 3, 4, LEAVE_ONE_WELL_OUT])
def test_no_well_on_two_sides(n_splits):
    for spec in folds_of(GroupedKFoldSplitter(n_splits=n_splits)):
        test_s, val_s, train_s = set(spec.test_wells), set(spec.val_wells), set(spec.train_wells)
        assert not test_s & val_s
        assert not test_s & train_s
        assert not val_s & train_s


def test_row_indices_never_overlap():
    y, groups, instances, is_sim, hours = make_dataset()
    splitter = GroupedKFoldSplitter(n_splits=3)
    for train_idx, val_idx, test_idx in splitter.split(
        None, y, groups, is_sim=is_sim, instances=instances, well_hours=hours
    ):
        assert not set(train_idx) & set(val_idx)
        assert not set(train_idx) & set(test_idx)
        assert not set(val_idx) & set(test_idx)
        # every real row lands somewhere; no row is silently dropped
        covered = len(train_idx) + len(val_idx) + len(test_idx)
        assert covered == len(y)


# --------------------------------------------------------------------------
# Red line 2 (DL3.2): simulated instances never reach val or test
# --------------------------------------------------------------------------


def test_simulated_wells_never_in_val_or_test():
    for spec in folds_of(GroupedKFoldSplitter(n_splits=3)):
        assert not any(str(w).startswith("SIM-") for w in spec.test_wells)
        assert not any(str(w).startswith("SIM-") for w in spec.val_wells)
        assert any(str(w).startswith("SIM-") for w in spec.train_wells)


def test_real_only_condition_excludes_sim_from_train():
    splitter = GroupedKFoldSplitter(n_splits=3, include_sim_in_train=False)
    for spec in folds_of(splitter):
        assert not any(str(w).startswith("SIM-") for w in spec.train_wells)


def test_sim_detected_from_group_name_without_is_sim_array():
    """is_sim is optional: string groups carry inventory.py's SIM- prefix."""
    y, groups, instances, _is_sim, hours = make_dataset()
    splitter = GroupedKFoldSplitter(n_splits=3)
    for spec in splitter.iter_folds(y, groups, instances=instances, well_hours=hours):
        assert not any(str(w).startswith("SIM-") for w in spec.test_wells)


# --------------------------------------------------------------------------
# Fold usability: positives on both val and test, hours on test
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_splits", [2, 3])
def test_every_fold_has_positive_events_on_both_sides(n_splits):
    y, groups, instances, is_sim, hours = make_dataset()
    report = GroupedKFoldSplitter(n_splits=n_splits).fold_report(
        y, groups, is_sim=is_sim, instances=instances, well_hours=hours
    )
    assert (report["n_test_positive_events"] > 0).all()
    assert (report["n_val_positive_events"] > 0).all()


def test_normal_hours_balanced_above_the_far_floor():
    """
    The whole reason for a separate hours-balanced split: with 3 folds,
    every test fold must clear the 300 h floor even though one well holds
    36% of all Normal hours and another holds 6 h.
    """
    y, groups, instances, is_sim, hours = make_dataset()
    report = GroupedKFoldSplitter(n_splits=3).fold_report(
        y, groups, is_sim=is_sim, instances=instances, well_hours=hours
    )
    assert (report["test_normal_hours"] >= 300.0).all()
    # and the total is conserved across folds
    assert report["test_normal_hours"].sum() == pytest.approx(sum(NORMAL_HOURS.values()), abs=0.5)


def test_validation_also_clears_the_hour_floor():
    """
    Module 8 selects the threshold on VALIDATION Normal hours, so the floor
    has to hold on that side too. A proportional val slice does not give
    it: on the real cache that left one fold with 4.0 validation hours.
    """
    y, groups, instances, is_sim, hours = make_dataset()
    report = GroupedKFoldSplitter(n_splits=3, min_val_normal_hours=300.0).fold_report(
        y, groups, is_sim=is_sim, instances=instances, well_hours=hours
    )
    assert (report["val_normal_hours"] >= 300.0).all()


def test_validation_never_takes_every_normal_well_from_training():
    """An absurd floor must not empty the training side of Normal wells."""
    y, groups, instances, is_sim, hours = make_dataset()
    splitter = GroupedKFoldSplitter(n_splits=3, min_val_normal_hours=1e9)
    for spec in splitter.iter_folds(
        y, groups, is_sim=is_sim, instances=instances, well_hours=hours
    ):
        train_normal = [w for w in spec.train_wells if w in NORMAL_HOURS]
        assert train_normal


def test_thin_fold_is_reported_not_silently_accepted():
    """A fold below the floor must be loud -- strict=True turns it fatal."""
    y, groups, instances, is_sim, hours = make_dataset()
    splitter = GroupedKFoldSplitter(n_splits=3, min_test_normal_hours=2000.0, strict=True)
    with pytest.raises(ValueError, match="Normal hours"):
        list(splitter.iter_folds(y, groups, is_sim=is_sim, instances=instances, well_hours=hours))


def test_too_many_folds_is_refused():
    y, groups, instances, is_sim, hours = make_dataset()
    splitter = GroupedKFoldSplitter(n_splits=8)  # only 7 positive wells exist
    with pytest.raises(ValueError, match="carry a positive window"):
        list(splitter.iter_folds(y, groups, is_sim=is_sim, instances=instances, well_hours=hours))


# --------------------------------------------------------------------------
# Fold geometry
# --------------------------------------------------------------------------


def test_test_folds_partition_the_real_wells():
    """Across folds of one repeat, each real well is tested exactly once."""
    specs = folds_of(GroupedKFoldSplitter(n_splits=3))
    seen = [w for spec in specs for w in spec.test_wells]
    assert len(seen) == len(set(seen))
    assert set(seen) == set(POSITIVE_EVENTS) | set(NORMAL_HOURS)


def test_leave_one_well_out_gives_one_positive_well_per_fold():
    specs = folds_of(GroupedKFoldSplitter(n_splits=LEAVE_ONE_WELL_OUT))
    assert len(specs) == len(POSITIVE_EVENTS)
    for spec in specs:
        positives = [w for w in spec.test_wells if w in POSITIVE_EVENTS]
        assert len(positives) == 1


def test_positive_events_are_balanced_not_just_well_counts():
    """
    Splitting 7 wells 3/2/2 by COUNT can hand one fold 5 of 14 events.
    Balancing on events keeps every fold within a couple of events.
    """
    y, groups, instances, is_sim, hours = make_dataset()
    report = GroupedKFoldSplitter(n_splits=3).fold_report(
        y, groups, is_sim=is_sim, instances=instances, well_hours=hours
    )
    spread = report["n_test_positive_events"].max() - report["n_test_positive_events"].min()
    assert spread <= 2


def test_same_seed_gives_identical_folds():
    a = folds_of(GroupedKFoldSplitter(n_splits=3, random_state=7))
    b = folds_of(GroupedKFoldSplitter(n_splits=3, random_state=7))
    assert [s.test_wells for s in a] == [s.test_wells for s in b]


def test_repeats_reshuffle_the_assignment():
    specs = folds_of(GroupedKFoldSplitter(n_splits=3, n_repeats=2))
    assert len(specs) == 6
    r0 = [s.test_wells for s in specs if s.repeat == 0]
    r1 = [s.test_wells for s in specs if s.repeat == 1]
    assert r0 != r1


def test_nested_val_mode_keeps_val_out_of_test():
    splitter = GroupedKFoldSplitter(n_splits=3, val_mode="nested", val_frac=0.3)
    for spec in folds_of(splitter):
        assert not set(spec.val_wells) & set(spec.test_wells)
        assert spec.val_wells


# --------------------------------------------------------------------------
# fold_report / Table 1
# --------------------------------------------------------------------------


def test_fold_report_shape_and_columns():
    y, groups, instances, is_sim, hours = make_dataset()
    report = GroupedKFoldSplitter(n_splits=3, n_repeats=2).fold_report(
        y, groups, is_sim=is_sim, instances=instances, well_hours=hours
    )
    assert len(report) == 6
    for col in (
        "repeat", "fold", "n_train_wells", "n_val_wells", "n_test_wells",
        "n_val_positive_events", "n_test_positive_events", "test_normal_hours",
        "test_wells",
    ):
        assert col in report.columns


def test_fold_report_event_total_matches_the_dataset():
    """14 transient instances exist; across test folds they are counted once each."""
    y, groups, instances, is_sim, hours = make_dataset()
    report = GroupedKFoldSplitter(n_splits=3).fold_report(
        y, groups, is_sim=is_sim, instances=instances, well_hours=hours
    )
    assert report["n_test_positive_events"].sum() == sum(POSITIVE_EVENTS.values())


# --------------------------------------------------------------------------
# load_cache_index
# --------------------------------------------------------------------------


CACHE_INSTANCES = [
    # (filename stem, well, group, normal hours, transient onset)
    ("instB", "WELL-00001", 0, 10.0, np.nan),
    ("instA", "WELL-00001", 0, 5.0, np.nan),
    ("instC", "WELL-00042", 1, 1.0, 4200.0),
]


def write_cache(tmp_path, windows_per_instance=4):
    """
    A cache in build_cache.py's on-disk format. Every value in X is the
    instance's own index, so a misaligned load is detectable by value.
    """
    for i, (stem, _well, group, hours, onset) in enumerate(CACHE_INSTANCES):
        n = windows_per_instance
        np.savez_compressed(
            tmp_path / f"{stem}.npz",
            X=np.full((n, 2, 60), float(i), dtype="float32"),
            mask=np.ones((n, 2, 60), dtype="uint8"),
            y=np.zeros(n, dtype="int64"),
            group=np.full(n, group, dtype="int64"),
            inst_id=np.full(n, i, dtype="int64"),
            t_end=np.arange(n, dtype="float64"),
            is_sim=np.zeros(n, dtype="uint8"),
            failure_time=np.float64(onset),
            blockage_time=np.float64("nan"),
            normal_hours=np.float64(hours),
        )
    (tmp_path / "cache_config.json").write_text(
        json.dumps({"group_map": {"WELL-00001": 0, "WELL-00042": 1}}), encoding="utf8"
    )


def test_load_cache_index_reads_metadata_and_sums_hours(tmp_path):
    write_cache(tmp_path)
    idx = load_cache_index(tmp_path)
    assert len(idx) == 12
    assert idx.hours_by_well == {0: 15.0, 1: 1.0}
    assert idx.well_of_group[1] == "WELL-00042"


def test_load_cache_index_refuses_an_empty_cache(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_cache"):
        load_cache_index(tmp_path)


def test_per_instance_scalars_are_broadcast_to_every_window(tmp_path):
    """failure_time is one number per instance on disk, one per row here."""
    write_cache(tmp_path)
    idx = load_cache_index(tmp_path)
    onset_of = {i: onset for i, (_s, _w, _g, _h, onset) in enumerate(CACHE_INSTANCES)}
    for inst, onset in onset_of.items():
        rows = idx.failure_time[idx.inst_id == inst]
        assert len(rows) == 4
        if np.isnan(onset):
            assert np.isnan(rows).all()
        else:
            assert (rows == onset).all()
    assert np.isnan(idx.blockage_time).all()


def test_load_cache_returns_X_aligned_with_the_metadata(tmp_path):
    """
    The alignment guarantee behind every fold index: row r of X belongs to
    the instance metadata row r says it does. The fixture encodes the
    instance index into X's values, so a reordering fails here loudly --
    which is the whole point, since in real use it would fail silently.
    """
    write_cache(tmp_path)
    X, mask, idx = load_cache(tmp_path)
    assert X.shape == (12, 2, 60)
    assert mask.shape == X.shape
    assert len(idx) == len(X)
    for r in range(len(X)):
        assert X[r].min() == X[r].max() == float(idx.inst_id[r])


def test_load_cache_row_order_is_sorted_filenames_not_write_order(tmp_path):
    """instB was written first; instA must still come first when loaded."""
    write_cache(tmp_path)
    _X, _mask, idx = load_cache(tmp_path)
    assert idx.files == ["instA", "instB", "instC"]


MULTI_WELL_CACHE = [
    # (stem, group, normal hours, has a positive phase)
    ("posA", 1, 0.0, True),
    ("posB", 2, 0.0, True),
    ("normA", 10, 400.0, False),
    ("normB", 11, 350.0, False),
]


def write_multi_well_cache(tmp_path, n=6):
    for i, (stem, group, hours, positive) in enumerate(MULTI_WELL_CACHE):
        y = np.zeros(n, dtype="int64")
        if positive:
            y[-2:] = TRANSIENT
        np.savez_compressed(
            tmp_path / f"{stem}.npz",
            X=np.full((n, 2, 60), float(i), dtype="float32"),
            mask=np.ones((n, 2, 60), dtype="uint8"),
            y=y,
            group=np.full(n, group, dtype="int64"),
            inst_id=np.full(n, i, dtype="int64"),
            t_end=np.arange(n, dtype="float64"),
            is_sim=np.zeros(n, dtype="uint8"),
            failure_time=np.float64(100.0 if positive else "nan"),
            blockage_time=np.float64("nan"),
            normal_hours=np.float64(hours),
        )


def test_fold_indices_select_the_right_rows_from_a_real_cache(tmp_path):
    """
    End-to-end: cache on disk -> load_cache -> split() -> row indices.
    This is the path M3 and M4 actually take, so it is the one that has to
    hold: indices from the metadata must select the matching rows of X.
    """
    write_multi_well_cache(tmp_path)
    X, mask, idx = load_cache(tmp_path)
    splitter = GroupedKFoldSplitter(n_splits=2, min_test_normal_hours=0.0)

    folds = list(
        splitter.split(
            X, idx.y, idx.group, is_sim=idx.is_sim,
            instances=idx.inst_id, well_hours=idx.hours_by_well,
        )
    )
    assert len(folds) == 2

    for train_idx, val_idx, test_idx in folds:
        # every row is used exactly once, and X/mask index the same rows
        assert len(train_idx) + len(val_idx) + len(test_idx) == len(X)
        assert mask[test_idx].shape[0] == len(test_idx)
        # a test fold holds whole wells, and never a well seen in training
        test_groups = set(idx.group[test_idx])
        assert test_groups
        assert not test_groups & set(idx.group[train_idx])
        # each test fold carries at least one positive event
        assert (idx.y[test_idx] > 0).any()
        # rows still carry their own instance's X values
        for r in test_idx:
            assert X[r].min() == float(idx.inst_id[r])


def test_too_few_positive_wells_is_refused(tmp_path):
    """One positive well cannot support two folds -- say so, don't improvise."""
    write_cache(tmp_path)
    _X, _mask, idx = load_cache(tmp_path)
    y = idx.y.copy()
    y[idx.group == 1] = TRANSIENT
    splitter = GroupedKFoldSplitter(n_splits=2)
    with pytest.raises(ValueError, match="carry a positive window"):
        list(
            splitter.split(
                None, y, idx.group, is_sim=idx.is_sim,
                instances=idx.inst_id, well_hours=idx.hours_by_well,
            )
        )
