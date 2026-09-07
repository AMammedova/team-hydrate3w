"""Member 3 — the XGBoost baseline runner for Result 1.

Runs the guaranteed baseline matrix:
    XGBoost x {real_only, real_plus_sim}

and writes exactly the artefacts Module 8 already knows how to read, in the
same layout tools/train_deep_models.py uses for TCN/GRU, so the head-to-head
table is a comparison of models rather than of file formats.

ORDER OF OPERATIONS (this is the part that is easy to get wrong)
----------------------------------------------------------------
    1. folds come from GroupedKFoldSplitter, real-only, built ONCE so both
       conditions are paired on identical wells;
    2. hyperparameters are searched on the TRAINING fold via grouped inner
       CV -- never on validation, never on test;
    3. the final model is fit on the training fold with inverse-frequency
       sample weights computed from that fold's own class counts;
    4. the calibrator is fit on VALIDATION;
    5. the frozen calibrator is applied to test. Nothing selected after
       seeing a test number (TEAM_5_MEMBERS.md §9.3, sync point S3).

Test-set METRICS are not computed unless --eval-test is passed. The runner
still writes test PROBABILITIES every time, because Module 8 needs them to
run the alarm chain once after the S3 freeze. Saving predictions is not the
same as looking at the score, and only the second one burns the test set.

WHY FEATURES ARE EXTRACTED ONCE FOR THE WHOLE CACHE
---------------------------------------------------
Red line 2 says the feature extractor is fit on the training fold only.
RollingFeatureExtractor has no fit: every column is a function of ONE window
and its mask, with no statistic pooled across rows, no scaler and no
vocabulary. Extracting per fold would therefore produce bit-identical
numbers at k times the cost. The absence of learned state is asserted at
startup rather than trusted, so if anyone ever adds a fit() the run stops.

Device: --device auto uses the GPU when XGBoost can actually see one and
falls back to CPU otherwise. It never claims a GPU it did not get.

Usage:
    python -m tools.train_xgb --cache data/cache
    python -m tools.train_xgb --cache data/cache --eval-test     # after S3
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines.calibrate import calibration_report, fit_calibrator
from src.baselines.features import RollingFeatureExtractor
from src.baselines.importance import (
    channel_importance,
    permutation_importance_table,
    scale_importance,
    stat_importance,
    summarize_importances,
)
from src.baselines.tune import DEFAULT_GRID, search
from src.baselines.device import resolve_device
from src.baselines.xgb_model import compute_sample_weight
from src.contract import CONDITION_REAL_ONLY, CONDITION_REAL_PLUS_SIM, RESULTS_COLUMNS
from src.data.splits import GroupedKFoldSplitter, load_cache
from src.eval.metrics import positive_score

logger = logging.getLogger("train_xgb")

MODEL_NAME = "xgboost"


# ---------------------------------------------------------------------------
# artefacts
# ---------------------------------------------------------------------------


def _append_results(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _recompose_probs(probs: np.ndarray, calibrated_pos: np.ndarray) -> np.ndarray:
    """Rebuild a 3-column probability matrix whose positive_score() equals
    `calibrated_pos`, keeping the model's Transient:Established split.

    Module 8 consumes `positive_score(probs)`. Handing it raw probabilities
    while the report quotes calibrated ones would mean the threshold was
    picked on a different axis than the figures describe, so the calibrated
    matrix is written alongside the raw one and both are kept.
    """
    probs = np.asarray(probs, dtype=np.float64)
    pos = probs[:, 1] + probs[:, 2]
    cal = np.clip(np.asarray(calibrated_pos, dtype=np.float64), 0.0, 1.0)
    # Where the model put no mass on the positive classes there is no ratio to
    # preserve; split the calibrated mass evenly instead of dividing by zero.
    with np.errstate(invalid="ignore", divide="ignore"):
        share_t = np.where(pos > 0, probs[:, 1] / np.maximum(pos, 1e-12), 0.5)
    out = np.empty_like(probs)
    out[:, 0] = 1.0 - cal
    out[:, 1] = cal * share_t
    out[:, 2] = cal * (1.0 - share_t)
    return out


def _save_outputs(
    path: Path,
    probs: np.ndarray,
    probs_cal: np.ndarray,
    idx: np.ndarray,
    index,
) -> None:
    """Everything Module 8 needs to reconstruct chronological alarms.

    Key-for-key the same contract tools/train_deep_models.py writes, plus the
    calibrated arrays, which the deep models do not have.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    groups = index.group[idx]
    keys = np.unique(groups)
    hours = np.asarray([index.hours_by_well.get(int(g), 0.0) for g in keys], dtype=np.float64)
    np.savez_compressed(
        path,
        probs=probs.astype(np.float32),
        probs_calibrated=probs_cal.astype(np.float32),
        pos_score_raw=positive_score(probs).astype(np.float32),
        pos_score_calibrated=positive_score(probs_cal).astype(np.float32),
        y_true=index.y[idx].astype(np.int64),
        group=groups.astype(np.int64),
        inst_id=index.inst_id[idx].astype(np.int64),
        t_end=index.t_end[idx].astype(np.float64),
        is_sim=index.is_sim[idx].astype(np.uint8),
        failure_time=index.failure_time[idx].astype(np.float64),
        blockage_time=index.blockage_time[idx].astype(np.float64),
        normal_hours_group=keys.astype(np.int64),
        normal_hours_value=hours,
    )


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------


def _base_real_splits(index, X, *, n_splits, n_repeats, random_state,
                      min_val_normal_hours, min_test_normal_hours) -> list[tuple]:
    """Real-only folds, built once. Simulated rows join later, train only."""
    splitter = GroupedKFoldSplitter(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
        include_sim_in_train=False,
        min_val_normal_hours=min_val_normal_hours,
        min_test_normal_hours=min_test_normal_hours,
    )
    splits = list(
        splitter.split(
            X, index.y, index.group,
            is_sim=index.is_sim, instances=index.inst_id, well_hours=index.hours_by_well,
        )
    )
    sim = index.is_sim.astype(bool)
    for fold, (tr, va, te) in enumerate(splits):
        if sim[tr].any() or sim[va].any() or sim[te].any():
            raise AssertionError(f"fold {fold}: real-only split contains simulated rows")
        gtr, gva, gte = (set(index.group[i].tolist()) for i in (tr, va, te))
        if gtr & gva or gtr & gte or gva & gte:
            raise AssertionError(f"fold {fold}: well leakage between splits")
    return splits


# ---------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------


def run_one(
    *,
    condition: str,
    fold_idx: int,
    seed: int,
    Xf: np.ndarray,
    index,
    feature_names: list[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    sim_idx: np.ndarray,
    device_kw: dict,
    param_grid: dict | None,
    n_iter: int | None,
    do_tune: bool,
    shared_params: dict | None,
    calib_method: str,
    outputs_dir: Path,
    figures_dir: Path,
    tables_dir: Path,
    eval_test: bool,
    perm_repeats: int,
) -> tuple[list[dict], dict]:
    from sklearn.metrics import average_precision_score
    from xgboost import XGBClassifier

    if condition == CONDITION_REAL_ONLY:
        tr = train_idx
    elif condition == CONDITION_REAL_PLUS_SIM:
        tr = np.concatenate([train_idx, sim_idx])
    else:
        raise ValueError(f"unknown condition {condition!r}")

    if index.is_sim[val_idx].any() or index.is_sim[test_idx].any():
        raise AssertionError("simulated windows leaked into validation/test")

    y_tr, y_va, y_te = index.y[tr], index.y[val_idx], index.y[test_idx]
    stem = f"{MODEL_NAME}_{condition}_fold{fold_idx}_seed{seed}"
    rows: list[dict] = []

    def _row(metric: str, value) -> dict:
        return dict(model=MODEL_NAME, fold=fold_idx, seed=seed, condition=condition,
                    metric_name=metric, value=float(value))

    # --- 1. hyperparameters ------------------------------------------------
    # Both conditions of a fold use the SAME hyperparameters, searched once on
    # that fold's REAL training wells. Tuning each condition separately would
    # confound the question the conditions exist to answer: real_plus_sim
    # could then win because it got a better configuration rather than because
    # simulated data helped. Sharing makes training data the only difference.
    t0 = time.perf_counter()
    DEFAULT_PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.9)
    if shared_params is not None:
        best = dict(shared_params)
    elif do_tune:
        try:
            res = search(
                Xf[tr], None, y_tr, index.group[tr], param_grid,
                n_iter=n_iter, random_state=seed, device=device_kw["device"],
                score_rows=(index.is_sim[tr] == 0),
            )
        except RuntimeError as exc:
            # A fold whose positives sit in one well cannot support grouped
            # inner CV. That is a property of the data, not a crash: fall back
            # to the default configuration, record that it happened so the
            # report can say which folds were untuned, and keep the run alive
            # so the remaining folds still produce results.
            logger.warning("[%s] tuning unavailable (%s); using default parameters", stem, exc)
            best = dict(DEFAULT_PARAMS)
            rows.append(_row("tuning_skipped", 1))
            rows.append(_row("n_configs_tried", 0))
            res = None
        if res is not None:
            best = res.best_params
            rows.append(_row("tuning_skipped", 0))
            rows.append(_row("n_configs_tried", res.n_configs_tried))
            rows.append(_row("n_configs_scored", res.n_configs_scored))
            rows.append(_row("tune_val_pr_auc", res.best_val_pr_auc))
            rows.append(_row("tune_seconds", res.seconds))
            res.trials.to_csv(tables_dir / f"{stem}_tuning_trials.csv", index=False)
            logger.info("[%s] %s", stem, res.summary())
    else:
        best = dict(DEFAULT_PARAMS)
        rows.append(_row("n_configs_tried", 1))

    # --- 2. final fit on the training fold ---------------------------------
    clf = XGBClassifier(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        verbosity=0, random_state=seed, **device_kw, **best,
    )
    if len(np.unique(y_tr)) < 2:
        raise RuntimeError(
            f"[{stem}] training fold contains a single class ({np.unique(y_tr)}). "
            f"XGBoost would silently set n_classes_=1 and return a transposed "
            f"probability array; refusing to produce a meaningless model."
        )
    clf.fit(Xf[tr], y_tr, sample_weight=compute_sample_weight(y_tr))
    fit_seconds = time.perf_counter() - t0

    # --- 3. validation, and the calibrator fit there -----------------------
    val_probs = clf.predict_proba(Xf[val_idx])
    if val_probs.shape != (len(val_idx), 3):
        raise AssertionError(
            f"[{stem}] predict_proba returned {val_probs.shape}, expected "
            f"{(len(val_idx), 3)} -- see the single-class guard above."
        )
    val_pos = positive_score(val_probs)
    y_va_bin = (y_va != 0).astype(np.int64)

    calibrator = fit_calibrator(y_va_bin, val_pos, method=calib_method)
    val_pos_cal = calibrator.predict(val_pos)
    cal_rep = calibration_report(y_va_bin, val_pos, val_pos_cal)

    # PR-AUC needs both classes present; folds with no positive validation
    # windows are real in this dataset, so this is a branch, not an assert.
    val_has_both = len(np.unique(y_va_bin)) > 1
    val_pr = float(average_precision_score(y_va_bin, val_pos)) if val_has_both else float("nan")
    rows += [
        _row("val_pr_auc", val_pr),
        _row("val_ece_before", cal_rep["ece_before"]),
        _row("val_ece_after", cal_rep["ece_after"]),
        _row("val_brier_before", cal_rep["brier_before"]),
        _row("val_brier_after", cal_rep["brier_after"]),
        _row("val_positive_events", len(np.unique(index.inst_id[val_idx][y_va != 0]))),
        _row("fit_seconds", fit_seconds),
        _row("n_features", Xf.shape[1]),
        _row("n_train_windows", len(tr)),
    ]

    # --- 4. test: predictions always, metrics only after the freeze --------
    test_probs = clf.predict_proba(Xf[test_idx])
    test_pos_cal = calibrator.predict(positive_score(test_probs))

    _save_outputs(outputs_dir / f"{stem}_val.npz", val_probs,
                  _recompose_probs(val_probs, val_pos_cal), val_idx, index)
    _save_outputs(outputs_dir / f"{stem}_test.npz", test_probs,
                  _recompose_probs(test_probs, test_pos_cal), test_idx, index)

    if eval_test:
        y_te_bin = (y_te != 0).astype(np.int64)
        if len(np.unique(y_te_bin)) > 1:
            te_rep = calibration_report(y_te_bin, positive_score(test_probs), test_pos_cal)
            rows += [
                _row("test_pr_auc", average_precision_score(y_te_bin, positive_score(test_probs))),
                _row("test_ece_before", te_rep["ece_before"]),
                _row("test_ece_after", te_rep["ece_after"]),
            ]

    # --- 5. reliability figure and importance ------------------------------
    # One figure per fold, not just fold 0. Fold 0's validation split holds a
    # single positive window, so its reliability curve is a flat line at zero
    # -- technically correct and completely uninformative. main() promotes the
    # best-populated fold's figure to the canonical filename the report cites.
    if condition == CONDITION_REAL_ONLY and val_has_both:
        from src.eval.plots import plot_reliability_diagram

        n_pos_win = int(y_va_bin.sum())
        plot_reliability_diagram(
            y_va_bin, val_pos, val_pos_cal,
            str(figures_dir / f"reliability_xgboost_fold{fold_idx}.png"),
            title=(
                f"Reliability - XGBoost {condition}, fold {fold_idx} "
                f"({n_pos_win} positive validation windows)"
            ),
        )
        rows.append(_row("val_positive_windows", n_pos_win))

    gain = summarize_importances(clf, feature_names)
    gain.to_csv(tables_dir / f"{stem}_importance_gain.csv", index=False)
    if val_has_both and perm_repeats > 0:
        perm = permutation_importance_table(
            clf, Xf[val_idx], y_va, feature_names,
            n_repeats=perm_repeats, random_state=seed,
        )
        perm.to_csv(tables_dir / f"{stem}_importance_permutation.csv", index=False)
        channel_importance(perm).to_csv(tables_dir / f"{stem}_importance_by_channel.csv", index=False)
        stat_importance(perm).to_csv(tables_dir / f"{stem}_importance_by_stat.csv", index=False)
        scale_importance(perm).to_csv(tables_dir / f"{stem}_importance_by_scale.csv", index=False)

    return rows, best


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--out-results", default="results/results.csv")
    ap.add_argument("--outputs-dir", default="results/model_outputs")
    ap.add_argument("--tables-dir", default="results/tables")
    ap.add_argument("--figures-dir", default="figures")
    ap.add_argument("--conditions", default=f"{CONDITION_REAL_ONLY},{CONDITION_REAL_PLUS_SIM}")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--n-repeats", type=int, default=1)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--min-val-normal-hours", type=float, default=300.0)
    ap.add_argument("--min-test-normal-hours", type=float, default=300.0)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--no-tune", action="store_true", help="skip the hyperparameter search")
    ap.add_argument("--n-iter", type=int, default=None,
                    help="cap on configurations; default is the full grid")
    ap.add_argument("--calibration", default="platt",
                    choices=["platt", "isotonic", "auto"])
    ap.add_argument("--perm-repeats", type=int, default=5,
                    help="permutation-importance repeats; 0 disables it")
    ap.add_argument("--missing-policy", default="nan", choices=["nan", "zero"])
    ap.add_argument("--slope-time", default="index", choices=["index", "rank"])
    ap.add_argument("--eval-test", action="store_true",
                    help="compute test metrics. Only after the S3 freeze.")
    ap.add_argument("--append", action="store_true",
                    help="append to results.csv instead of starting a fresh file")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    outputs_dir = Path(args.outputs_dir)
    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)
    for d in (outputs_dir, tables_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    X, mask, index = load_cache(args.cache)
    channels = None
    sidecar = Path(args.cache) / "cache_config.json"
    if sidecar.exists():
        meta = json.loads(sidecar.read_text(encoding="utf8"))
        channels = meta.get("kept_channels") or meta.get("channels")
    if not channels:
        channels = [f"ch{i}" for i in range(X.shape[1])]
        logger.warning("cache sidecar has no channel list; importance rows will be unnamed")

    fe = RollingFeatureExtractor(
        missing_policy=args.missing_policy, slope_time=args.slope_time
    )
    # Red line 2 guard -- see the module docstring.
    assert not hasattr(fe, "fit"), (
        "RollingFeatureExtractor grew a fit(): it now has learned state and MUST be "
        "fit per training fold instead of once over the whole cache."
    )
    t0 = time.perf_counter()
    Xf = fe.transform(X, mask)
    feature_names = fe.feature_names(list(channels))
    logger.info(
        "features: %s in %.1fs (%d NaN cells, missing_policy=%s, slope_time=%s)",
        Xf.shape, time.perf_counter() - t0, int(np.isnan(Xf).sum()),
        args.missing_policy, args.slope_time,
    )

    device_kw = resolve_device(args.device)
    logger.info("xgboost device: %s", device_kw["device"])

    splits = _base_real_splits(
        index, X,
        n_splits=args.n_splits, n_repeats=args.n_repeats, random_state=args.split_seed,
        min_val_normal_hours=args.min_val_normal_hours,
        min_test_normal_hours=args.min_test_normal_hours,
    )
    sim_idx = np.flatnonzero(index.is_sim == 1)
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    logger.info(
        "windows=%d  folds=%d  sim_windows=%d  conditions=%s  seeds=%s  eval_test=%s",
        len(index), len(splits), len(sim_idx), conditions, seeds, args.eval_test,
    )

    out_results = Path(args.out_results)
    if not args.append and out_results.exists():
        out_results.unlink()

    all_rows: list[dict] = []
    for fold_idx, (tr, va, te) in enumerate(splits):
        # One search per (fold, seed), shared by both conditions.
        shared: dict[int, dict] = {}
        for seed in seeds:
            for condition in conditions:
                rows = run_one(
                    condition=condition, fold_idx=fold_idx, seed=seed,
                    Xf=Xf, index=index, feature_names=feature_names,
                    train_idx=tr, val_idx=va, test_idx=te, sim_idx=sim_idx,
                    device_kw=device_kw,
                    param_grid=DEFAULT_GRID, n_iter=args.n_iter,
                    do_tune=not args.no_tune,
                    shared_params=shared.get(seed),
                    calib_method=args.calibration,
                    outputs_dir=outputs_dir, figures_dir=figures_dir,
                    tables_dir=tables_dir, eval_test=args.eval_test,
                    perm_repeats=args.perm_repeats,
                )
                rows, used_params = rows
                shared.setdefault(seed, used_params)
                _append_results(out_results, rows)
                all_rows += rows
                val_pr = [r["value"] for r in rows if r["metric_name"] == "val_pr_auc"]
                logger.info(
                    "fold %d | %-14s | seed %d | val PR-AUC %.4f",
                    fold_idx, condition, seed, val_pr[0] if val_pr else float("nan"),
                )

    df = pd.DataFrame(all_rows)

    # Promote the best-populated fold's reliability figure to the canonical
    # name the report references. This chooses which fold ILLUSTRATES
    # calibration, not which result is reported -- every fold's figure is
    # written alongside it, and the title states the population.
    pw = df[df.metric_name == "val_positive_windows"]
    if len(pw):
        best_fold = int(pw.loc[pw["value"].idxmax(), "fold"])
        src = figures_dir / f"reliability_xgboost_fold{best_fold}.png"
        if src.exists():
            (figures_dir / "reliability_xgboost.png").write_bytes(src.read_bytes())
            logger.info(
                "reliability figure: fold %d has the most positive validation windows "
                "(%d); promoted to reliability_xgboost.png",
                best_fold, int(pw["value"].max()),
            )

    print("\n=== validation PR-AUC by condition ===")
    piv = (
        df[df.metric_name == "val_pr_auc"]
        .groupby("condition")["value"].agg(["mean", "std", "count"])
    )
    print(piv.to_string())
    print(f"\nwrote {out_results}  ({len(df)} rows)")
    print(f"model outputs -> {outputs_dir}")
    print(f"tables        -> {tables_dir}")


if __name__ == "__main__":
    main()
