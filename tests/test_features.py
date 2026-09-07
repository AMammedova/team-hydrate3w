"""Member 3, W3.1 — RollingFeatureExtractor.

These tests pin the things that fail SILENTLY if they regress: the column
order, the name<->column alignment importance.py depends on, causality
(features must never see past the end of the window), and the mask
contract (statistics come from present samples only).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.baselines.features import (
    PRESENCE_SUFFIX,
    SCALES_DEFAULT,
    STATS_DEFAULT,
    RollingFeatureExtractor,
)


def _ones(n=4, c=3, w=60):
    X = np.zeros((n, c, w), dtype=np.float32)
    M = np.ones((n, c, w), dtype=np.uint8)
    return X, M


# ---------------------------------------------------------------------------
# Shape / order / naming contract
# ---------------------------------------------------------------------------


def test_output_shape_matches_contract():
    X, M = _ones(n=7, c=5, w=60)
    fe = RollingFeatureExtractor()
    out = fe.transform(X, M)
    # 5 channels * 6 stats * 3 scales + 5 presence
    assert out.shape == (7, 5 * 6 * 3 + 5) == (7, 95)
    assert out.dtype == np.float32
    assert fe.n_features(5) == 95


def test_feature_names_length_and_order_match_transform():
    X, M = _ones(c=5)
    fe = RollingFeatureExtractor()
    names = fe.feature_names(n_channels=5)
    assert len(names) == fe.transform(X, M).shape[1]

    # Order is scale -> channel -> stat, then the presence block last.
    assert names[0] == "ch0|mean|scale1.0"
    assert names[1] == "ch0|std|scale1.0"
    assert names[6] == "ch1|mean|scale1.0"      # next channel, same scale
    assert names[30] == "ch0|mean|scale0.5"     # 5 ch * 6 stats later
    assert names[-5:] == [f"ch{i}|{PRESENCE_SUFFIX}" for i in range(5)]


def test_feature_names_uses_real_channel_names():
    chans = ["P-MON-CKP", "P-JUS-CKGL", "T-TPT", "T-JUS-CKP", "P-ANULAR"]
    names = RollingFeatureExtractor().feature_names(chans)
    assert names[0] == "P-MON-CKP|mean|scale1.0"
    assert "T-TPT|last_diff|scale0.25" in names
    assert names[-1] == "P-ANULAR|presence_frac"


def test_feature_names_refuses_to_invent_names():
    with pytest.raises(ValueError):
        RollingFeatureExtractor().feature_names()
    with pytest.raises(ValueError):
        RollingFeatureExtractor().feature_names(["a", "b"], n_channels=3)


def test_names_align_with_columns():
    """The alignment importance.py trusts: a column named for a channel must
    actually be computed from that channel."""
    fe = RollingFeatureExtractor()
    chans = ["A", "B", "C"]
    names = fe.feature_names(chans)
    X = np.zeros((1, 3, 60), dtype=np.float32)
    M = np.ones((1, 3, 60), dtype=np.uint8)
    X[0, 1, :] = 7.0                             # only channel "B" is non-zero
    out = fe.transform(X, M)[0]
    for i, v in enumerate(out):
        if abs(v - 7.0) < 1e-6:
            assert names[i].startswith("B|"), f"column {i} ({names[i]}) leaked channel B"


# ---------------------------------------------------------------------------
# Statistic correctness
# ---------------------------------------------------------------------------


def test_stats_on_a_known_signal():
    """One channel, no missingness, a clean ramp -- every statistic is
    analytically known."""
    w = 60
    X = np.arange(w, dtype=np.float32).reshape(1, 1, w)   # 0..59, step 1
    M = np.ones((1, 1, w), dtype=np.uint8)
    fe = RollingFeatureExtractor()
    names = fe.feature_names(["r"])
    out = fe.transform(X, M)[0]
    f = dict(zip(names, out))

    assert f["r|mean|scale1.0"] == pytest.approx(29.5)
    assert f["r|min|scale1.0"] == pytest.approx(0.0)
    assert f["r|max|scale1.0"] == pytest.approx(59.0)
    assert f["r|slope|scale1.0"] == pytest.approx(1.0)      # unit ramp
    assert f["r|last_diff|scale1.0"] == pytest.approx(59.0)
    assert f["r|std|scale1.0"] == pytest.approx(float(np.std(np.arange(w))), rel=1e-5)

    # Last quarter = samples 45..59
    assert f["r|mean|scale0.25"] == pytest.approx(52.0)
    assert f["r|min|scale0.25"] == pytest.approx(45.0)
    assert f["r|slope|scale0.25"] == pytest.approx(1.0)
    assert f["r|last_diff|scale0.25"] == pytest.approx(14.0)


def test_scales_are_taken_from_the_end_of_the_window():
    """Causality: a spike in the FIRST half must not move any 0.25-scale
    feature. If it does, the feature is reading the wrong end of the window
    and the whole lead-time story is unsound."""
    fe = RollingFeatureExtractor()
    names = fe.feature_names(["r"])
    X = np.zeros((1, 1, 60), dtype=np.float32)
    M = np.ones((1, 1, 60), dtype=np.uint8)
    base = fe.transform(X, M)[0]

    X[0, 0, 3] = 999.0                            # early spike only
    bumped = fe.transform(X, M)[0]

    quarter = [i for i, n in enumerate(names) if n.endswith("scale0.25")]
    half = [i for i, n in enumerate(names) if n.endswith("scale0.5")]
    full = [i for i, n in enumerate(names) if n.endswith("scale1.0")]
    assert np.allclose(base[quarter], bumped[quarter])
    assert np.allclose(base[half], bumped[half])
    assert not np.allclose(base[full], bumped[full])   # full window must see it


def test_statistics_ignore_masked_out_samples():
    """A masked-out sample carries a wild value; no statistic may move."""
    fe = RollingFeatureExtractor()
    X = np.ones((1, 1, 60), dtype=np.float32)
    M = np.ones((1, 1, 60), dtype=np.uint8)
    clean = fe.transform(X, M)[0]

    X[0, 0, 10] = 1e6
    M[0, 0, 10] = 0                                # ... but it is absent
    dirty = fe.transform(X, M)[0]

    # Only the presence fraction is allowed to change.
    n_stat_cols = 1 * len(STATS_DEFAULT) * len(SCALES_DEFAULT)
    assert np.allclose(clean[:n_stat_cols], dirty[:n_stat_cols])
    assert dirty[-1] == pytest.approx(59 / 60)


def test_presence_fraction_is_over_the_full_window():
    fe = RollingFeatureExtractor()
    M = np.ones((1, 2, 60), dtype=np.uint8)
    M[0, 0, :6] = 0                                 # 10% absent on channel 0
    out = fe.transform(np.zeros((1, 2, 60), dtype=np.float32), M)[0]
    assert out[-2] == pytest.approx(0.9)
    assert out[-1] == pytest.approx(1.0)


def test_slope_uses_true_time_axis_not_rank():
    """Two present samples 1 apart vs 10 apart with the same rise are NOT the
    same rate. The legacy 'rank' axis says they are."""
    X = np.zeros((1, 1, 60), dtype=np.float32)
    M = np.zeros((1, 1, 60), dtype=np.uint8)
    # Present only at t=50 (value 0) and t=59 (value 9): true rate = 1.0/sample
    M[0, 0, 50] = 1
    M[0, 0, 59] = 1
    X[0, 0, 50] = 0.0
    X[0, 0, 59] = 9.0

    idx = RollingFeatureExtractor(slope_time="index").feature_names(["r"]).index("r|slope|scale1.0")
    s_true = RollingFeatureExtractor(slope_time="index").transform(X, M)[0, idx]
    s_rank = RollingFeatureExtractor(slope_time="rank").transform(X, M)[0, idx]

    assert s_true == pytest.approx(1.0)      # 9 units over 9 samples
    assert s_rank == pytest.approx(9.0)      # gap collapsed -> 9x overstated


# ---------------------------------------------------------------------------
# Missing-data policy -- the measured design decision
# ---------------------------------------------------------------------------


def test_absent_channel_is_nan_under_default_policy():
    fe = RollingFeatureExtractor()                   # missing_policy="nan"
    names = fe.feature_names(["dead", "live"])
    X = np.ones((1, 2, 60), dtype=np.float32)
    M = np.ones((1, 2, 60), dtype=np.uint8)
    M[0, 0, :] = 0                                   # channel 0 observed nothing
    out = fe.transform(X, M)[0]

    dead = [i for i, n in enumerate(names) if n.startswith("dead|") and PRESENCE_SUFFIX not in n]
    live = [i for i, n in enumerate(names) if n.startswith("live|") and PRESENCE_SUFFIX not in n]
    assert np.isnan(out[dead]).all()
    assert not np.isnan(out[live]).any()
    # Presence fraction stays a real measurement, never NaN.
    assert out[names.index("dead|presence_frac")] == pytest.approx(0.0)


def test_absent_channel_is_zero_under_legacy_policy():
    fe = RollingFeatureExtractor(missing_policy="zero")
    X = np.ones((1, 2, 60), dtype=np.float32)
    M = np.ones((1, 2, 60), dtype=np.uint8)
    M[0, 0, :] = 0
    out = fe.transform(X, M)[0]
    assert not np.isnan(out).any()
    names = fe.feature_names(["dead", "live"])
    assert out[names.index("dead|mean|scale1.0")] == pytest.approx(0.0)


def test_single_present_sample_has_zero_spread_not_nan():
    """One sample is enough for mean/min/max, and genuinely implies zero
    spread and zero trend -- that is a measurement, not a gap."""
    fe = RollingFeatureExtractor()
    names = fe.feature_names(["r"])
    X = np.zeros((1, 1, 60), dtype=np.float32)
    M = np.zeros((1, 1, 60), dtype=np.uint8)
    M[0, 0, 59] = 1
    X[0, 0, 59] = 4.0
    out = fe.transform(X, M)[0]
    assert out[names.index("r|mean|scale1.0")] == pytest.approx(4.0)
    assert out[names.index("r|std|scale1.0")] == pytest.approx(0.0)
    assert out[names.index("r|slope|scale1.0")] == pytest.approx(0.0)
    assert out[names.index("r|last_diff|scale1.0")] == pytest.approx(0.0)


def test_no_nans_leak_when_data_is_complete():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5, 60)).astype(np.float32)
    M = np.ones((20, 5, 60), dtype=np.uint8)
    assert not np.isnan(RollingFeatureExtractor().transform(X, M)).any()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_rejects_bad_configuration():
    with pytest.raises(ValueError):
        RollingFeatureExtractor(missing_policy="mean")
    with pytest.raises(ValueError):
        RollingFeatureExtractor(slope_time="clock")
    with pytest.raises(ValueError):
        RollingFeatureExtractor(stats=["median"])
    with pytest.raises(ValueError):
        RollingFeatureExtractor(scales=[0.0, 1.0])
    with pytest.raises(ValueError):
        RollingFeatureExtractor(scales=[1.5])


def test_rejects_bad_input_shapes():
    fe = RollingFeatureExtractor()
    with pytest.raises(ValueError):
        fe.transform(np.zeros((4, 60)), np.zeros((4, 60)))          # 2-D
    with pytest.raises(ValueError):
        fe.transform(np.zeros((4, 3, 60)), np.zeros((4, 3, 30)))    # mismatched


def test_subset_of_stats_is_honoured():
    fe = RollingFeatureExtractor(stats=["mean", "slope"], scales=[1.0])
    X, M = _ones(n=2, c=4, w=60)
    out = fe.transform(X, M)
    assert out.shape == (2, 4 * 2 * 1 + 4)
    assert fe.feature_names(n_channels=4)[:2] == ["ch0|mean|scale1.0", "ch0|slope|scale1.0"]
