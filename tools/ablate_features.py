"""Feature-design ablation for the XGBoost baseline.

Arms: missing_policy (nan vs zero), slope_time (index vs rank), number of
timescales, presence block on/off.

Scored on VALIDATION folds only. Every arm sees the same folds and seeds, so
comparisons are paired. Several split seeds are used because 3 folds give only
3 paired cells and at least one has a single positive validation event.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines.features import RollingFeatureExtractor
from src.baselines.device import resolve_device
from src.data.splits import GroupedKFoldSplitter, load_cache
from src.eval.metrics import expected_calibration_error, positive_score

# One shared, deliberately middling XGBoost configuration. The ablation is
# about features; letting each arm tune its own model would confound the two.
FIXED_XGB = dict(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=1.0,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    verbosity=0,
)


ARMS: list[dict] = [
    # --- A + B: the two decisions baked into features.py's defaults --------
    dict(arm="A1_zero_rank",  group="encoding", missing_policy="zero", slope_time="rank",
         note="legacy: the implementation as committed in 2f59274"),
    dict(arm="A2_zero_index", group="encoding", missing_policy="zero", slope_time="index"),
    dict(arm="A3_nan_rank",   group="encoding", missing_policy="nan",  slope_time="rank"),
    dict(arm="A4_nan_index",  group="encoding", missing_policy="nan",  slope_time="index",
         note="proposed default"),
    # --- C: do three timescales earn their columns? -----------------------
    dict(arm="C1_scale_full_only", group="scales", scales=[1.0]),
    dict(arm="C2_scale_full_half", group="scales", scales=[1.0, 0.5]),
    dict(arm="C3_scale_all_three", group="scales", scales=[1.0, 0.5, 0.25]),
    # --- D: is the presence block pulling its weight? ---------------------
    dict(arm="D1_no_presence", group="presence", drop_presence=True),
    dict(arm="D2_with_presence", group="presence", drop_presence=False),
]


def _extractor(arm: dict) -> RollingFeatureExtractor:
    return RollingFeatureExtractor(
        stats=arm.get("stats"),
        scales=arm.get("scales"),
        missing_policy=arm.get("missing_policy", "nan"),
        slope_time=arm.get("slope_time", "index"),
    )


def _fit_score(
    X_tr, y_tr, X_va, y_va, *, seed: int, device_kw: dict
) -> tuple[float, float, float]:
    """Fit on the training fold, score on validation. Returns
    (PR-AUC, ECE, positive-rate-weighted Brier)."""
    from sklearn.metrics import average_precision_score
    from xgboost import XGBClassifier

    from src.baselines.xgb_model import compute_sample_weight

    clf = XGBClassifier(random_state=seed, **FIXED_XGB, **device_kw)
    clf.fit(X_tr, y_tr, sample_weight=compute_sample_weight(y_tr))
    proba = clf.predict_proba(X_va)
    score = positive_score(proba)
    y_bin = (y_va != 0).astype(np.int64)
    pr = float(average_precision_score(y_bin, score))
    ece = float(expected_calibration_error(y_bin, score, n_bins=10))
    brier = float(np.mean((score - y_bin) ** 2))
    return pr, ece, brier


def run(
    cache: str,
    out_csv: str,
    *,
    device: str = "auto",
    seeds: list[int],
    n_splits: int,
    split_seeds: list[int],
    min_val_normal_hours: float,
    min_test_normal_hours: float,
    arms: list[dict],
) -> pd.DataFrame:
    X, mask, index = load_cache(cache)
    channels = None
    sidecar = Path(cache) / "cache_config.json"
    if sidecar.exists():
        meta = json.loads(sidecar.read_text(encoding="utf8"))
        # build_cache.py writes "kept_channels"; the synthetic benchmark writes
        # "channels". Accept both rather than silently ending up with None.
        channels = meta.get("kept_channels") or meta.get("channels")
    print(f"cache={cache}  windows={len(index)}  X={X.shape}  channels={channels}")

    device_kw = resolve_device(device)
    print(f"xgboost device: {device_kw['device']}")

    # Several split seeds, not one. With 3 folds a single fold structure gives
    # 3 paired cells, and DATA_FINDINGS/fold_report show at least one of them
    # has a single positive validation event -- a cell whose PR-AUC is not
    # estimable. Re-drawing the fold structure is the only honest way to get
    # power here, and it stays on the validation side throughout.
    folds: list[tuple] = []
    fold_meta: list[dict] = []
    sim = index.is_sim.astype(bool)
    for sseed in split_seeds:
        splitter = GroupedKFoldSplitter(
            n_splits=n_splits,
            n_repeats=1,
            random_state=sseed,
            include_sim_in_train=False,      # ablation is a real-only comparison
            min_val_normal_hours=min_val_normal_hours,
            min_test_normal_hours=min_test_normal_hours,
        )
        for f, (tr, va, te) in enumerate(splitter.split(
            X, index.y, index.group,
            is_sim=index.is_sim, instances=index.inst_id, well_hours=index.hours_by_well,
        )):
            if sim[va].any():
                raise AssertionError(f"split_seed {sseed} fold {f}: simulated rows in validation")
            if set(index.group[tr].tolist()) & set(index.group[va].tolist()):
                raise AssertionError(f"split_seed {sseed} fold {f}: well leakage train/val")
            n_pos_inst = len(np.unique(index.inst_id[va][index.y[va] != 0]))
            if n_pos_inst == 0:
                raise AssertionError(f"split_seed {sseed} fold {f}: no positive validation windows")
            folds.append((tr, va, te))
            fold_meta.append(dict(
                split_seed=sseed, fold=f,
                val_positive_events=int(n_pos_inst),
                val_positive_windows=int((index.y[va] != 0).sum()),
                n_train=len(tr), n_val=len(va),
            ))

    fm = pd.DataFrame(fold_meta)
    print(f"fold cells={len(folds)} (from {len(split_seeds)} split seeds x {n_splits} folds)")
    print(fm.to_string(index=False))
    thin = fm[fm.val_positive_events < 2]
    if len(thin):
        print(
            f"\n  WARNING: {len(thin)} of {len(fm)} cells have <2 positive validation "
            f"events. Validation PR-AUC is not estimable there; those cells stay in the "
            f"paired comparison (dropping them would be selection) but they inflate sd_of_d."
        )

    rows: list[dict] = []
    for arm in arms:
        fe = _extractor(arm)
        drop_presence = bool(arm.get("drop_presence", False))
        n_ch = X.shape[1]
        keep = np.arange(fe.n_features(n_ch))
        if drop_presence:
            keep = keep[: -n_ch]

        t0 = time.perf_counter()
        Xf = fe.transform(X, mask)[:, keep]
        t_feat = time.perf_counter() - t0
        n_nan = int(np.isnan(Xf).sum())

        for fold_i, (tr, va, _te_unused) in enumerate(folds):     # test never used
            cell = fold_meta[fold_i]
            for seed in seeds:
                pr, ece, brier = _fit_score(
                    Xf[tr], index.y[tr], Xf[va], index.y[va],
                    seed=seed, device_kw=device_kw,
                )
                rows.append(dict(
                    arm=arm["arm"], group=arm["group"], fold=fold_i, seed=seed,
                    split_seed=cell["split_seed"], inner_fold=cell["fold"],
                    val_positive_events=cell["val_positive_events"],
                    val_pr_auc=pr, val_ece=ece, val_brier=brier,
                    n_features=int(Xf.shape[1]), n_nan_cells=n_nan,
                    feat_seconds=round(t_feat, 3),
                    missing_policy=arm.get("missing_policy", "nan"),
                    slope_time=arm.get("slope_time", "index"),
                ))
        done = pd.DataFrame(rows)
        m = done[done.arm == arm["arm"]]["val_pr_auc"]
        print(
            f"  {arm['arm']:22s} feats={Xf.shape[1]:4d} nan_cells={n_nan:8d} "
            f"featurise={t_feat:6.2f}s  val_PR-AUC={m.mean():.4f} +/- {m.std(ddof=1) if len(m)>1 else 0:.4f}"
        )

    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def report(df: pd.DataFrame) -> None:
    """Paired per-arm summary. Every arm saw the same (fold, seed) cells, so
    the paired difference against the group's reference arm is the honest
    comparison -- not the difference of two independent means."""
    print("\n" + "=" * 78)
    print("ABLATION SUMMARY (validation folds only)")
    print("=" * 78)

    for group, gdf in df.groupby("group", sort=False):
        ref_arm = {"encoding": "A1_zero_rank", "scales": "C3_scale_all_three",
                   "presence": "D1_no_presence"}[group]
        print(f"\n[{group}]  reference arm = {ref_arm}")
        ref = gdf[gdf.arm == ref_arm].set_index(["fold", "seed"])["val_pr_auc"]
        table = []
        for arm, adf in gdf.groupby("arm", sort=False):
            s = adf.set_index(["fold", "seed"])["val_pr_auc"]
            diff = (s - ref).dropna()
            wins = int((diff > 0).sum())
            n = len(diff)
            table.append(dict(
                arm=arm,
                n_feat=int(adf["n_features"].iloc[0]),
                pr_auc=f"{s.mean():.4f}",
                sd=f"{s.std(ddof=1):.4f}" if len(s) > 1 else "-",
                d_vs_ref=f"{diff.mean():+.4f}" if n else "-",
                sd_of_d=f"{diff.std(ddof=1):.4f}" if n > 1 else "-",
                wins=f"{wins}/{n}",
                ece=f"{adf['val_ece'].mean():.4f}",
            ))
        print(pd.DataFrame(table).to_string(index=False))

    print("\nNote: d_vs_ref is the mean PAIRED difference in validation PR-AUC")
    print("(same folds, same seeds). 'wins' counts cells where the arm beat the")
    print("reference. With few folds, treat a difference smaller than its own")
    print("sd_of_d as noise and say so in the report.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--out", default="results/ablation_features.csv")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--split-seeds", default="42,7,2024",
                    help="comma-separated CV split seeds; each gives an independent "
                         "fold structure, which is where the statistical power comes from")
    ap.add_argument("--min-val-normal-hours", type=float, default=300.0)
    ap.add_argument("--min-test-normal-hours", type=float, default=300.0)
    ap.add_argument("--only-group", default=None, help="run one group: encoding|scales|presence")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="auto uses the GPU when XGBoost can actually see one")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    arms = [a for a in ARMS if args.only_group in (None, a["group"])]

    df = run(
        args.cache, args.out,
        device=args.device,
        seeds=seeds,
        n_splits=args.n_splits,
        split_seeds=[int(s) for s in args.split_seeds.split(',') if s.strip()],
        min_val_normal_hours=args.min_val_normal_hours,
        min_test_normal_hours=args.min_test_normal_hours,
        arms=arms,
    )
    report(df)
    print(f"\nwrote {args.out}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
