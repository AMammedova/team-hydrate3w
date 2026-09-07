"""Member 3 — tune.py, calibrate.py, importance.py and XGBoostBaseline.

These cover the failure modes that do not raise: hyperparameter metadata
leaking into the model, a feature-name list that silently describes the wrong
columns, a calibrator fit on a single-class fold, and importance parsing that
mangles the `last_diff` statistic.
"""

from __future__ import annotations

import numpy as np
import pytest

xgboost = pytest.importorskip("xgboost")

from src.baselines.calibrate import Calibrator, calibration_report, fit_calibrator
from src.baselines.features import RollingFeatureExtractor
from src.baselines.importance import (
    channel_importance,
    permutation_importance_table,
    scale_importance,
    stat_importance,
    summarize_importances,
)
from src.baselines.tune import SearchResult, search
from src.baselines.device import resolve_device
from src.baselines.xgb_model import XGBoostBaseline, compute_sample_weight

CHANNELS = ["P-MON-CKP", "P-JUS-CKGL", "T-TPT", "T-JUS-CKP", "P-ANULAR"]


def _dataset(n_wells=6, per_well=40, seed=0):
    """A small separable task with well groups, shaped like the real one:
    channel 0 carries a level shift on positives, the rest is noise."""
    rng = np.random.default_rng(seed)
    n = n_wells * per_well
    X = rng.normal(0, 1, (n, 5, 60)).astype(np.float32)
    mask = (rng.random((n, 5, 60)) > 0.05).astype(np.uint8)
    groups = np.repeat(np.arange(n_wells), per_well)
    y = np.zeros(n, dtype=np.int64)
    for w in range(n_wells):                      # every well carries positives
        sl = slice(w * per_well, (w + 1) * per_well)
        idx = np.arange(per_well) >= per_well - 14
        yy = np.zeros(per_well, dtype=np.int64)
        yy[idx] = 1
        yy[-4:] = 2
        y[sl] = yy
        X[sl][idx, 0] += 2.5
        X[w * per_well : (w + 1) * per_well][idx, 0, :] += 2.5
    return X, mask, y, groups


# ---------------------------------------------------------------------------
# resolve_device
# ---------------------------------------------------------------------------


def test_resolve_device_never_overstates_the_gpu():
    kw = resolve_device("auto")
    assert kw["device"] in ("cpu", "cuda")
    assert kw["tree_method"] == "hist"
    assert resolve_device("cpu")["device"] == "cpu"


def test_resolve_device_cuda_raises_when_there_is_no_gpu():
    """`prefer='cuda'` must fail loudly rather than train on CPU while the
    report claims a GPU run."""
    if resolve_device("auto")["device"] == "cuda":
        pytest.skip("this machine really has a GPU")
    with pytest.raises(RuntimeError, match="resolved to CPU"):
        resolve_device("cuda")


# ---------------------------------------------------------------------------
# tune.search
# ---------------------------------------------------------------------------


def test_search_returns_model_ready_params_only():
    """The regression that motivated SearchResult: splatting the old flat
    dict into XGBClassifier forwarded 'n_configs_tried' to the booster."""
    X, mask, y, groups = _dataset()
    Xf = RollingFeatureExtractor().transform(X, mask)
    res = search(Xf, mask, y, groups,
                 {"n_estimators": [10, 20], "max_depth": [2]},
                 n_inner_splits=3, device="cpu")

    assert isinstance(res, SearchResult)
    assert set(res.best_params) <= {"n_estimators", "max_depth"}
    assert "n_configs_tried" not in res.best_params
    assert "best_val_pr_auc" not in res.best_params

    # The whole point: this must construct and fit without junk parameters.
    clf = xgboost.XGBClassifier(
        objective="multi:softprob", num_class=3, verbosity=0, **res.best_params
    )
    clf.fit(Xf[:120], y[:120])


def test_search_records_its_budget():
    X, mask, y, groups = _dataset()
    Xf = RollingFeatureExtractor().transform(X, mask)
    grid = {"n_estimators": [10, 20], "max_depth": [2, 3]}
    res = search(Xf, mask, y, groups, grid, device="cpu")
    assert res.n_configs_tried == 4
    assert len(res.trials) == 4
    assert res.n_configs_scored <= res.n_configs_tried
    assert 0.0 <= res.best_val_pr_auc <= 1.0
    assert "mean_val_pr_auc" in res.trials.columns


def test_search_respects_n_iter_budget():
    X, mask, y, groups = _dataset()
    Xf = RollingFeatureExtractor().transform(X, mask)
    grid = {"n_estimators": [10, 20, 30], "max_depth": [2, 3, 4]}
    res = search(Xf, mask, y, groups, grid, n_iter=4, device="cpu")
    assert res.n_configs_tried == 9      # grid size is still reported honestly
    assert len(res.trials) == 4          # ... but only 4 were evaluated


def test_search_score_rows_restricts_inner_scoring():
    """The real_plus_sim guard: inner CV must be scored on the rows the outer
    fold is judged on, not on the simulated majority."""
    X, mask, y, groups = _dataset()
    Xf = RollingFeatureExtractor().transform(X, mask)
    real = np.ones(len(y), dtype=bool)
    real[groups >= 4] = False              # pretend wells 4,5 are "simulated"

    res = search(Xf, mask, y, groups, {"n_estimators": [10], "max_depth": [2]},
                 device="cpu", score_rows=real)
    assert res.n_configs_scored >= 1

    with pytest.raises(ValueError, match="score_rows"):
        search(Xf, mask, y, groups, {"n_estimators": [10]},
               device="cpu", score_rows=np.ones(len(y) + 3, dtype=bool))


def test_search_skips_inner_splits_whose_TRAINING_side_is_single_class():
    """Regression test for a silent-corruption bug found on the real cache.

    With 3 outer folds, one training fold carried every positive window in a
    single well, so grouped inner CV produced an inner training split with no
    positives at all. XGBoost does not raise there: it sets n_classes_=1,
    ignores num_class=3, and predict_proba returns a transposed (3, 2n) array
    that blows up much later inside average_precision_score.
    """
    rng = np.random.default_rng(0)
    n_wells, per_well = 4, 60
    n = n_wells * per_well
    X = rng.normal(0, 1, (n, 5, 60)).astype(np.float32)
    mask = np.ones((n, 5, 60), dtype=np.uint8)
    groups = np.repeat(np.arange(n_wells), per_well)
    y = np.zeros(n, dtype=np.int64)
    # Positives live in wells 0 and 1 only. Grouped inner CV then yields both
    # usable splits and splits whose training or validation side is
    # single-class -- the real cache's pathology, without being so degenerate
    # that no split survives at all (that case is the next test).
    y[:30] = 1
    y[per_well : per_well + 30] = 1
    X[:30, 0, :] += 3.0
    X[per_well : per_well + 30, 0, :] += 3.0
    Xf = RollingFeatureExtractor().transform(X, mask)

    # Must not raise a length-mismatch from deep inside sklearn.
    res = search(Xf, mask, y, groups, {"n_estimators": [10], "max_depth": [2]},
                 n_inner_splits=3, device="cpu")
    assert res.n_configs_scored >= 1
    assert res.trials["n_inner_skipped"].iloc[0] >= 1


def test_search_raises_clearly_when_no_inner_split_is_usable():
    rng = np.random.default_rng(1)
    n = 120
    X = rng.normal(0, 1, (n, 5, 60)).astype(np.float32)
    mask = np.ones((n, 5, 60), dtype=np.uint8)
    groups = np.repeat(np.arange(4), 30)
    y = np.zeros(n, dtype=np.int64)
    y[:5] = 1                       # all positives in well 0, too few to split
    Xf = RollingFeatureExtractor().transform(X, mask)
    with pytest.raises(RuntimeError, match="no usable inner CV split"):
        search(Xf, mask, y, groups, {"n_estimators": [10]},
               n_inner_splits=2, device="cpu")


def test_xgboost_really_does_return_garbage_on_a_single_class_fit():
    """Pins the upstream behaviour the guards exist for, so if XGBoost ever
    starts raising properly we find out and can simplify."""
    rng = np.random.default_rng(2)
    Xf = rng.normal(size=(200, 20))
    y_single = np.zeros(200, dtype=np.int64)
    clf = xgboost.XGBClassifier(
        objective="multi:softprob", num_class=3, verbosity=0, n_estimators=5
    )
    clf.fit(Xf, y_single)
    proba = clf.predict_proba(Xf[:50])
    assert proba.shape != (50, 3), (
        "XGBoost now handles a single-class fit sanely -- the tune.py guard "
        "can be revisited"
    )


def test_search_refuses_a_training_fold_with_one_well():
    X, mask, y, groups = _dataset(n_wells=1, per_well=40)
    Xf = RollingFeatureExtractor().transform(X, mask)
    with pytest.raises(ValueError, match="at least 2 well groups"):
        search(Xf, mask, y, groups, {"n_estimators": [10]}, device="cpu")


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------


def test_calibrator_improves_a_deliberately_skewed_score():
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.binomial(1, 0.1, n)
    # Push probabilities far from the base rate, as weighted training does.
    p = np.clip(np.where(y == 1, rng.beta(6, 2, n), rng.beta(2, 4, n)), 1e-4, 1 - 1e-4)

    before = calibration_report(y, p, p)["ece_before"]
    for method in ("platt", "isotonic"):
        cal = fit_calibrator(y, p, method=method)
        rep = calibration_report(y, p, cal.predict(p))
        assert rep["ece_after"] < before, f"{method} did not improve ECE"


def test_calibrator_output_stays_a_probability():
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.3, 500)
    p = rng.random(500)
    for method in ("platt", "isotonic"):
        out = fit_calibrator(y, p, method=method).predict(np.linspace(0, 1, 50))
        assert np.all((out >= 0.0) & (out <= 1.0))
        assert np.isfinite(out).all()


def test_calibrator_on_single_class_fold_is_the_identity():
    """Real folds can have no positive validation windows. That must not
    raise, and must not silently flatten the ranking threshold selection
    depends on."""
    y = np.zeros(200, dtype=int)
    p = np.linspace(0.01, 0.99, 200)
    cal = fit_calibrator(y, p)
    assert cal.degenerate and cal.method == "identity"
    assert np.allclose(cal.predict(p), p)


def test_calibrator_rejects_three_class_labels():
    """Passing y instead of (y != 0) would calibrate against class ids."""
    y = np.array([0, 1, 2, 1, 0] * 20)
    p = np.random.default_rng(0).random(100)
    with pytest.raises(ValueError, match="BINARY"):
        fit_calibrator(y, p)


def test_auto_method_picks_one_of_the_two():
    rng = np.random.default_rng(3)
    y = rng.binomial(1, 0.25, 800)
    p = np.clip(np.where(y == 1, rng.beta(5, 2, 800), rng.beta(2, 5, 800)), 1e-4, 1 - 1e-4)
    cal = fit_calibrator(y, p, method="auto")
    assert cal.method in ("platt", "isotonic")


def test_calibration_report_fields():
    y = np.array([0, 1, 0, 1])
    rep = calibration_report(y, np.array([0.2, 0.8, 0.3, 0.7]), np.array([0.1, 0.9, 0.2, 0.8]))
    assert rep["base_rate"] == pytest.approx(0.5)
    for k in ("ece_before", "ece_after", "brier_before", "brier_after",
              "ece_delta", "brier_delta"):
        assert np.isfinite(rep[k])


# ---------------------------------------------------------------------------
# importance
# ---------------------------------------------------------------------------


def test_summarize_importances_parses_last_diff_correctly():
    """The bug this replaced: splitting on '_' turned 'last_diff' into
    stat='last' and lost the scale."""
    class _M:
        feature_importances_ = np.array([0.5, 0.3, 0.2])

    df = summarize_importances(
        _M(),
        ["P-MON-CKP|last_diff|scale0.5", "T-TPT|mean|scale1.0", "P-ANULAR|presence_frac"],
    )
    row = df[df.feature == "P-MON-CKP|last_diff|scale0.5"].iloc[0]
    assert row["channel"] == "P-MON-CKP"
    assert row["stat"] == "last_diff"
    assert row["scale"] == pytest.approx(0.5)

    pres = df[df.feature == "P-ANULAR|presence_frac"].iloc[0]
    assert pres["channel"] == "P-ANULAR" and pres["stat"] == "presence_frac"


def test_summarize_importances_rejects_a_mismatched_name_list():
    class _M:
        feature_importances_ = np.array([0.5, 0.5])

    with pytest.raises(ValueError, match="feature names"):
        summarize_importances(_M(), ["only|mean|scale1.0"])


def test_importance_names_come_from_the_real_extractor():
    X, mask, y, groups = _dataset(n_wells=4, per_well=30)
    fe = RollingFeatureExtractor()
    model = XGBoostBaseline(n_estimators=20, max_depth=3)
    model.fit(X, mask, y, sample_weight=compute_sample_weight(y))

    names = fe.feature_names(CHANNELS)
    df = summarize_importances(model, names)          # accepts XGBoostBaseline
    assert len(df) == len(names)
    assert set(df["channel"]) <= set(CHANNELS)
    assert df["rank"].tolist() == list(range(1, len(df) + 1))


def test_channel_stat_scale_aggregations():
    class _M:
        feature_importances_ = np.arange(95, dtype=float)

    names = RollingFeatureExtractor().feature_names(CHANNELS)
    df = summarize_importances(_M(), names)

    ch = channel_importance(df)
    assert set(ch["channel"]) == set(CHANNELS)
    assert ch["share"].sum() == pytest.approx(1.0)
    assert ch["rank"].tolist() == list(range(1, len(ch) + 1))

    st = stat_importance(df)
    assert "presence_frac" in set(st["stat"])
    assert st["share"].sum() == pytest.approx(1.0)

    sc = scale_importance(df)
    assert set(sc["scale"]) == {1.0, 0.5, 0.25}       # presence rows dropped


def test_permutation_importance_finds_the_informative_channel():
    """The signal lives in channel 0 by construction, so permuting a
    P-MON-CKP column must cost more validation PR-AUC than permuting the
    pure-noise P-ANULAR columns."""
    X, mask, y, groups = _dataset(n_wells=6, per_well=60, seed=5)
    fe = RollingFeatureExtractor()
    Xf = fe.transform(X, mask)
    names = fe.feature_names(CHANNELS)

    tr = groups < 4
    va = ~tr
    model = XGBoostBaseline(n_estimators=80, max_depth=3)
    model.model.fit(Xf[tr], y[tr], sample_weight=compute_sample_weight(y[tr]))

    df = permutation_importance_table(
        model.model, Xf[va], y[va], names, n_repeats=3, random_state=0
    )
    ch = channel_importance(df).set_index("channel")
    assert ch.loc["P-MON-CKP", "total"] > ch.loc["P-ANULAR", "total"]


def test_permutation_importance_needs_both_classes():
    fe = RollingFeatureExtractor()
    X, mask, y, groups = _dataset(n_wells=2, per_well=20)
    Xf = fe.transform(X, mask)
    model = XGBoostBaseline(n_estimators=10)
    model.fit(X, mask, y)
    with pytest.raises(ValueError, match="single-class"):
        permutation_importance_table(
            model.model, Xf, np.zeros(len(y), dtype=int), fe.feature_names(CHANNELS)
        )


# ---------------------------------------------------------------------------
# XGBoostBaseline plumbing
# ---------------------------------------------------------------------------


def test_baseline_emits_three_columns_even_without_established_in_train():
    """A training fold can contain no Established windows -- only 118 exist in
    the whole real cache. positive_score() indexes column 2, so the model must
    still return three columns."""
    X, mask, y, _ = _dataset(n_wells=3, per_well=30)
    y = np.where(y == 2, 1, y)                    # drop class 2 entirely
    model = XGBoostBaseline(n_estimators=10)
    model.fit(X, mask, y, sample_weight=compute_sample_weight(y))
    proba = model.predict_proba(X, mask)
    assert proba.shape == (len(y), 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_compute_sample_weight_is_inverse_frequency():
    y = np.array([0] * 90 + [1] * 9 + [2] * 1)
    w = compute_sample_weight(y)
    assert w[y == 2][0] > w[y == 1][0] > w[y == 0][0]
    assert w.mean() == pytest.approx(1.0, rel=1e-6)
