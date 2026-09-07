"""Member 3 — does sensor AVAILABILITY alone predict the label?

WHY THIS PROBE EXISTS
---------------------
Two independent results pointed the same way and needed a decisive test:

  * the feature ablation found the per-channel presence-fraction block was
    the only component with consistent supporting evidence (+0.066 validation
    PR-AUC, 7/10 paired wins);
  * gain importance on the one fold with real validation signal put 40.8% of
    the total gain on those 5 columns, out of 95.

DATA_FINDINGS.md §8 records that missingness in this dataset is
well-specific: different wells instrument different sensors, and some
sensors are dead for an entire instance. If the model can read which
channels exist, it can partly identify the well -- and because the hydrate
wells and the Normal-operation wells are DISJOINT populations (§2), well
identity is very close to the label.

Per-instance normalisation does not defend against this. It removes each
recording's offset and scale; it cannot remove "this channel is absent",
which is carried by the mask itself.

So this probe trains on presence features ONLY -- no sensor values at all --
and reports validation PR-AUC. A model that cannot see a single pressure or
temperature reading should be unable to detect a hydrate. If it scores well
above the base rate, the shortcut is real and the headline numbers are
partly measuring instrumentation rather than physics.

Three arms, same folds, same seeds:
    presence_only   the 5 presence-fraction columns
    values_only     the 90 statistic columns, presence block removed
    all_features    the full 95-column matrix

Usage:
    python -m tools.probe_presence_shortcut --cache data/cache
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines.device import resolve_device
from src.baselines.features import RollingFeatureExtractor
from src.baselines.xgb_model import compute_sample_weight
from src.data.splits import GroupedKFoldSplitter, load_cache
from src.eval.metrics import positive_score

FIXED_XGB = dict(
    n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
    objective="multi:softprob", num_class=3, eval_metric="mlogloss", verbosity=0,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--out", default="results/probe_presence_shortcut.csv")
    ap.add_argument("--seeds", default="42,43")
    ap.add_argument("--split-seeds", default="42,7,2024")
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    from sklearn.metrics import average_precision_score
    from xgboost import XGBClassifier

    X, mask, index = load_cache(args.cache)
    n_ch = X.shape[1]
    channels = None
    sc = Path(args.cache) / "cache_config.json"
    if sc.exists():
        meta = json.loads(sc.read_text(encoding="utf8"))
        channels = meta.get("kept_channels") or meta.get("channels")

    fe = RollingFeatureExtractor()
    Xf = fe.transform(X, mask)
    names = fe.feature_names(list(channels) if channels else None, None if channels else n_ch)
    presence_cols = [i for i, nm in enumerate(names) if nm.endswith("presence_frac")]
    value_cols = [i for i in range(len(names)) if i not in set(presence_cols)]
    assert len(presence_cols) == n_ch

    device_kw = resolve_device(args.device)
    base_rate = float((index.y != 0).mean())
    print(f"windows={len(index)}  features={Xf.shape[1]}  "
          f"presence_cols={len(presence_cols)}  device={device_kw['device']}")
    print(f"positive base rate (all windows) = {base_rate:.4f}")

    rows = []
    for sseed in [int(s) for s in args.split_seeds.split(",") if s.strip()]:
        splitter = GroupedKFoldSplitter(
            n_splits=args.n_splits, n_repeats=1, random_state=sseed,
            include_sim_in_train=False,
        )
        for fold, (tr, va, _te) in enumerate(splitter.split(
            X, index.y, index.group, is_sim=index.is_sim,
            instances=index.inst_id, well_hours=index.hours_by_well,
        )):
            y_tr, y_va = index.y[tr], index.y[va]
            y_bin = (y_va != 0).astype(int)
            if len(np.unique(y_bin)) < 2 or len(np.unique(y_tr)) < 2:
                continue
            va_base = float(y_bin.mean())
            n_pos_ev = len(np.unique(index.inst_id[va][y_va != 0]))
            for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
                for arm, cols in (
                    ("presence_only", presence_cols),
                    ("values_only", value_cols),
                    ("all_features", list(range(Xf.shape[1]))),
                ):
                    clf = XGBClassifier(random_state=seed, **FIXED_XGB, **device_kw)
                    clf.fit(Xf[np.ix_(tr, cols)], y_tr,
                            sample_weight=compute_sample_weight(y_tr))
                    proba = clf.predict_proba(Xf[np.ix_(va, cols)])
                    pr = float(average_precision_score(y_bin, positive_score(proba)))
                    rows.append(dict(split_seed=sseed, fold=fold, seed=seed, arm=arm,
                                     n_features=len(cols), val_pr_auc=pr,
                                     val_base_rate=va_base, val_positive_events=n_pos_ev,
                                     lift_over_base=pr / va_base if va_base else np.nan))
            print(f"  split_seed={sseed} fold={fold} pos_events={n_pos_ev} "
                  f"base={va_base:.4f} " + "  ".join(
                      f"{r['arm']}={r['val_pr_auc']:.4f}"
                      for r in rows[-3:]))

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print("\n" + "=" * 74)
    print("DOES SENSOR AVAILABILITY ALONE PREDICT THE LABEL?")
    print("=" * 74)
    summ = df.groupby("arm").agg(
        n_features=("n_features", "first"),
        mean_pr_auc=("val_pr_auc", "mean"),
        median_pr_auc=("val_pr_auc", "median"),
        sd=("val_pr_auc", "std"),
        mean_lift_over_base=("lift_over_base", "mean"),
        cells=("val_pr_auc", "size"),
    )
    print(summ.round(4).to_string())
    print(f"\nwrote {args.out} ({len(df)} rows)")
    print(
        "\nReading: presence_only uses NO sensor values at all. Any PR-AUC it "
        "achieves well\nabove the validation base rate is signal carried purely by "
        "which sensors exist,\ni.e. instrumentation acting as a proxy for well "
        "identity -- not hydrate physics."
    )


if __name__ == "__main__":
    main()
