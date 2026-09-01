"""
Unit tests for the windowing pipeline Member 1 implements on top of
label_window(): frozen_run_mask, mask_missing, normalize_instance,
WindowBuilder.build_windows / window_masks, onset_times, normal_seconds.

The two tests that matter most, and why:

* test_mask_missing_is_causal -- forward-fill must never pull a value
  backwards in time. This is the same class of bug as the centered alarm
  smoother caught in Addendum A.9.1, one module earlier and much harder to
  see: a back-filled sample makes a window look informative before the
  information existed, which inflates every lead time downstream. The test is
  constructed so a bfill implementation fails it.

* test_build_windows_is_channels_first -- build_windows()'s original docstring
  promised (n_windows, window_size, n_channels), while Addendum A.2 and all
  four consumers (RollingFeatureExtractor, TCN, GRUClassifier,
  make_fake_data.py) expect [N, C, W]. Returning the old layout would be a
  silent transpose, not an error, so it gets an explicit test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.contract import ESTABLISHED, NORMAL, TRANSIENT, TRANSIENT_OFFSET
from src.data.inventory import InstanceRecord
from src.data.windowing import (
    VariableSelector,
    WindowBuilder,
    frozen_run_mask,
    mask_missing,
    normal_seconds,
    normalize_instance,
    onset_times,
)

EVENT = 9
RAW_T = TRANSIENT_OFFSET + EVENT   # 109
RAW_E = EVENT                       # 9


def make_instance(
    n_seconds: int = 3600,
    n_channels: int = 4,
    transient_at: int | None = None,
    established_at: int | None = None,
    nan_label_span: tuple[int, int] | None = None,
    well_id: str = "WELL-00001",
    source: str = "real",
) -> InstanceRecord:
    """A synthetic 1 Hz instance shaped exactly like a loaded 3W parquet:
    DatetimeIndex, N float variable columns, plus a raw `class` column."""
    idx = pd.date_range("2020-01-01", periods=n_seconds, freq="1s")
    data = {
        f"CH-{i}": np.linspace(0, 1, n_seconds) + i for i in range(n_channels)
    }
    raw = np.zeros(n_seconds, dtype="float64")
    if transient_at is not None:
        raw[transient_at:] = RAW_T
    if established_at is not None:
        raw[established_at:] = RAW_E
    if nan_label_span is not None:
        raw[nan_label_span[0] : nan_label_span[1]] = np.nan
    df = pd.DataFrame(data, index=idx)
    df.index.name = "timestamp"
    df["class"] = raw
    return InstanceRecord(
        instance_id="synthetic",
        well_id=well_id,
        source=source,
        event_code=EVENT,
        filepath=Path("synthetic.parquet"),
        df=df,
    )


# --------------------------------------------------------------------------
# frozen_run_mask
# --------------------------------------------------------------------------
def test_frozen_run_mask_flags_long_runs_only():
    v = np.array([1.0, 1.0, 1.0, 2.0, 3.0, 3.0])
    got = frozen_run_mask(v, min_run=3)
    assert got.tolist() == [True, True, True, False, False, False]


def test_frozen_run_mask_nan_breaks_a_run():
    """A gap is already missing; it must not be counted as a stuck sensor, and
    it must not join the runs on either side of it into one long run."""
    v = np.array([5.0, 5.0, np.nan, 5.0, 5.0])
    assert not frozen_run_mask(v, min_run=4).any()


def test_frozen_run_mask_min_run_one_is_a_noop():
    assert not frozen_run_mask(np.array([1.0, 2.0, 3.0]), min_run=1).any()


# --------------------------------------------------------------------------
# mask_missing
# --------------------------------------------------------------------------
def test_mask_missing_is_causal():
    """
    Channel is [1, NaN, NaN, 9]. A causal forward-fill gives [1, 1, 1, 9].
    A back-fill (or any centered/interpolating scheme) would put 9 -- a value
    from t=3 -- at t=1 or t=2, i.e. use the future. This test fails against
    such an implementation, which is the point of having it.
    """
    X = np.array([[[1.0, np.nan, np.nan, 9.0]]])
    filled, mask = mask_missing(X)
    assert filled[0, 0].tolist() == [1.0, 1.0, 1.0, 9.0]
    assert mask[0, 0].tolist() == [1, 0, 0, 1]


def test_mask_missing_leading_nan_uses_channel_mean_not_a_future_sample():
    X = np.array([[[np.nan, np.nan, 4.0]]])
    filled, mask = mask_missing(X, channel_means=np.array([-7.0]))
    assert filled[0, 0, 0] == pytest.approx(-7.0)
    assert filled[0, 0, 1] == pytest.approx(-7.0)   # NOT 4.0, which is later
    assert mask[0, 0].tolist() == [0, 0, 1]


def test_mask_missing_leading_nan_without_means_falls_back_to_zero():
    filled, mask = mask_missing(np.array([[[np.nan, 3.0]]]))
    assert filled[0, 0, 0] == pytest.approx(0.0)
    assert mask[0, 0, 0] == 0


def test_mask_missing_max_gap_stops_stale_carry_forward():
    X = np.array([[[1.0, np.nan, np.nan, np.nan]]])
    filled, _ = mask_missing(X, channel_means=np.array([0.0]), max_gap=1)
    # t=1 is one step old (allowed); t=2 and t=3 are too stale to trust.
    assert filled[0, 0].tolist() == [1.0, 1.0, 0.0, 0.0]


def test_mask_missing_output_dtypes_and_no_nan():
    X = np.where(np.random.default_rng(0).random((3, 5, 7)) < 0.3, np.nan, 1.0)
    filled, mask = mask_missing(X)
    assert filled.dtype == np.float32 and mask.dtype == np.uint8
    assert not np.isnan(filled).any()
    assert filled.shape == X.shape == mask.shape


def test_mask_missing_rejects_non_3d_input():
    with pytest.raises(ValueError, match=r"\[N, C, W\]"):
        mask_missing(np.zeros((4, 5)))


def test_mask_missing_zero_fill_mode():
    filled, mask = mask_missing(np.array([[[1.0, np.nan]]]), fill="zero")
    assert filled[0, 0].tolist() == [1.0, 0.0]
    assert mask[0, 0].tolist() == [1, 0]


# --------------------------------------------------------------------------
# normalize_instance
# --------------------------------------------------------------------------
def test_normalize_warmup_removes_per_instance_offset():
    """Two instances differing only by a constant offset must normalise to the
    same series -- that is exactly the well-identity confound this exists to
    kill (hydrate and Normal wells are disjoint, DATA_FINDINGS.md §2)."""
    base = np.sin(np.linspace(0, 6, 200))[None, None, :]
    mask = np.ones_like(base, dtype="uint8")
    a = normalize_instance(base, mask, method="warmup", warmup_samples=100)
    b = normalize_instance(base + 500.0, mask, method="warmup", warmup_samples=100)
    assert np.allclose(a, b, atol=1e-4)


def test_normalize_none_is_passthrough():
    X = np.arange(8, dtype="float32").reshape(1, 2, 4)
    out = normalize_instance(X, np.ones_like(X, dtype="uint8"), method="none")
    assert np.array_equal(out, X)


def test_normalize_rejects_unknown_method():
    X = np.zeros((1, 1, 4), dtype="float32")
    with pytest.raises(ValueError, match="unknown normalize method"):
        normalize_instance(X, np.ones_like(X, dtype="uint8"), method="minmax")


def test_normalize_ignores_unobserved_samples():
    X = np.array([[[0.0, 0.0, 1.0, 1.0]]])
    mask = np.array([[[0, 0, 1, 1]]], dtype="uint8")
    out = normalize_instance(X, mask, method="warmup", warmup_samples=4)
    assert out[0, 0, 0] == 0.0 and out[0, 0, 1] == 0.0


# --------------------------------------------------------------------------
# WindowBuilder
# --------------------------------------------------------------------------
def test_build_windows_is_channels_first():
    inst = make_instance(n_seconds=3600, n_channels=4)
    wb = WindowBuilder(window_size=10, stride=5, decimate=30)
    X, y, wells, t = wb.build_windows(inst, normalize="none")
    assert X.ndim == 3
    assert X.shape[1] == 4, "axis 1 must be CHANNELS (Addendum A.2), not time"
    assert X.shape[2] == 10, "axis 2 must be TIME (window_size samples)"
    assert X.dtype == np.float32
    assert len(y) == len(wells) == len(t) == X.shape[0]


def test_window_seconds_accounts_for_decimation():
    wb = WindowBuilder(window_size=60, stride=5, decimate=30)
    assert wb.window_seconds == 1800     # 30 minutes, not 60 seconds


def test_build_windows_labels_never_precede_the_onset():
    """
    The causality property that makes lead time meaningful: no window may be
    labeled TRANSIENT while its whole span is still before the annotated
    onset. Checked via window_end_time, which is what alarm.py consumes.
    """
    onset = 1800
    inst = make_instance(n_seconds=7200, transient_at=onset)
    wb = WindowBuilder(window_size=10, stride=5, decimate=30)
    _X, y, _w, t = wb.build_windows(inst, normalize="none")
    assert (y == TRANSIENT).any()
    assert t[np.argmax(y == TRANSIENT)] >= onset


def test_build_windows_most_severe_reaches_established():
    inst = make_instance(n_seconds=7200, transient_at=1800, established_at=3600)
    wb = WindowBuilder(window_size=10, stride=5, decimate=30)
    _X, y, _w, _t = wb.build_windows(inst, normalize="none")
    assert set(np.unique(y)) == {NORMAL, TRANSIENT, ESTABLISHED}


def test_nan_label_policy_drop_removes_unlabeled_windows():
    """
    Every real Event-9 instance carries unlabeled (NaN) spans -- median 11.2%
    of samples (DATA_FINDINGS.md §5) -- and the project statement never says
    what to do with them. "drop" must actually drop; "normal" must not.
    """
    inst = make_instance(n_seconds=7200, nan_label_span=(3000, 4000))
    wb_drop = WindowBuilder(window_size=10, stride=5, decimate=30, nan_label_policy="drop")
    wb_keep = WindowBuilder(window_size=10, stride=5, decimate=30, nan_label_policy="normal")
    n_drop = len(wb_drop.build_windows(inst, normalize="none")[1])
    n_keep = len(wb_keep.build_windows(inst, normalize="none")[1])
    assert n_drop < n_keep


def test_min_valid_frac_drops_mostly_missing_windows():
    inst = make_instance(n_seconds=3600, n_channels=2)
    inst.df.loc[inst.df.index[:1800], "CH-0"] = np.nan
    inst.df.loc[inst.df.index[:1800], "CH-1"] = np.nan
    strict = WindowBuilder(window_size=10, stride=5, decimate=30, min_valid_frac=0.99)
    loose = WindowBuilder(window_size=10, stride=5, decimate=30, min_valid_frac=0.0)
    assert len(strict.build_windows(inst, normalize="none")[1]) < len(
        loose.build_windows(inst, normalize="none")[1]
    )


def test_window_masks_aligns_row_for_row_with_build_windows():
    """build_cache writes X from one call and mask from the other; if their drop
    rules ever diverge, every mask channel silently belongs to a different
    window than its data."""
    inst = make_instance(n_seconds=7200, n_channels=3, nan_label_span=(2000, 2500))
    wb = WindowBuilder(window_size=10, stride=5, decimate=30)
    X, _y, _w, _t = wb.build_windows(inst, normalize="none")
    mask = wb.window_masks(inst)
    assert mask.shape == X.shape
    assert mask.dtype == np.uint8


def test_build_windows_returns_empty_when_instance_is_shorter_than_a_window():
    inst = make_instance(n_seconds=60)
    wb = WindowBuilder(window_size=60, stride=5, decimate=30)
    X, y, wells, t = wb.build_windows(inst, normalize="none")
    assert len(X) == len(y) == len(wells) == len(t) == 0
    assert X.shape[1:] == (4, 60)


def test_build_windows_rejects_bad_constructor_args():
    with pytest.raises(ValueError):
        WindowBuilder(decimate=0)
    with pytest.raises(ValueError, match="nan_label_policy"):
        WindowBuilder(nan_label_policy="impute")


# --------------------------------------------------------------------------
# onset_times / normal_seconds -- the per-instance scalars the eval needs
# --------------------------------------------------------------------------
def test_onset_times_reports_both_onsets_in_seconds():
    inst = make_instance(n_seconds=7200, transient_at=1800, established_at=5400)
    got = onset_times(inst)
    assert got["transient_onset"] == pytest.approx(1800.0)
    assert got["blockage_onset"] == pytest.approx(5400.0)


def test_onset_times_is_nan_when_a_phase_never_happens():
    """54 of the 57 real instances never reach blockage, so this is the common
    case, not an edge case -- lead_time() relies on the NaN to skip them."""
    inst = make_instance(n_seconds=3600, transient_at=1200)
    got = onset_times(inst)
    assert got["transient_onset"] == pytest.approx(1200.0)
    assert np.isnan(got["blockage_onset"])


def test_normal_seconds_counts_wall_clock_and_excludes_unlabeled_spans():
    inst = make_instance(n_seconds=1000, transient_at=800, nan_label_span=(0, 100))
    # 0..99 NaN, 100..799 normal, 800..999 transient
    assert normal_seconds(inst) == pytest.approx(700.0)


# --------------------------------------------------------------------------
# VariableSelector
# --------------------------------------------------------------------------
def test_variable_selector_drops_channels_above_the_missing_threshold():
    inst = make_instance(n_seconds=1000, n_channels=3)
    inst.df.loc[inst.df.index[:900], "CH-1"] = np.nan       # 90% missing
    sel = VariableSelector(max_missing_frac=0.5).fit([inst])
    assert "CH-1" not in sel.kept_columns
    assert "CH-0" in sel.kept_columns and "CH-2" in sel.kept_columns


def test_variable_selector_counts_frozen_runs_as_missing():
    """A sensor stuck at one value is dead, not present. If fit() ignored the
    frozen conversion transform() performs, a fully frozen channel would look
    100% present and stay in the model's input."""
    inst = make_instance(n_seconds=1000, n_channels=2)
    inst.df["CH-1"] = 42.0                                   # entirely frozen
    sel = VariableSelector(max_missing_frac=0.5, frozen_run_seconds=60).fit([inst])
    assert "CH-1" not in sel.kept_columns


def test_variable_selector_transform_keeps_the_label_column():
    inst = make_instance(n_seconds=600, n_channels=2)
    sel = VariableSelector().fit([inst])
    out = sel.transform(inst.df)
    assert "class" in out.columns, "build_windows() needs `class` to label from"


def test_variable_selector_transform_before_fit_raises():
    inst = make_instance(n_seconds=100)
    with pytest.raises(RuntimeError, match="fit"):
        VariableSelector().transform(inst.df)


def test_variable_selector_accepts_a_generator():
    """build_cache streams 801 instances through fit(); a list would hold every
    DataFrame in memory at once."""
    gen = (make_instance(n_seconds=300, n_channels=2) for _ in range(3))
    sel = VariableSelector().fit(gen)
    assert len(sel.kept_columns) == 2


def test_variable_selector_means_for_orders_by_the_requested_columns():
    inst = make_instance(n_seconds=500, n_channels=3)
    sel = VariableSelector().fit([inst])
    cols = sel.kept_columns
    means = sel.means_for(cols)
    assert means.shape == (len(cols),)
    # CH-i is linspace(0,1) + i, so its mean is ~ i + 0.5 and strictly increasing
    assert np.all(np.diff(means) > 0)
