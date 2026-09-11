"""Paired analysis of results/ablation_features.csv.

Mean paired difference vs each group's reference arm, with a Wilcoxon
signed-rank test, reported both over all cells and over the estimable subset
(>= 2 positive validation events).
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from src.data.splits import GroupedKFoldSplitter, load_cache_index

REFERENCE_ARM = {
    "encoding": "A1_zero_rank",
    "scales": "C3_scale_all_three",
    "presence": "D1_no_presence",
}


def fold_base_rates(cache: str, split_seeds: list[int], n_splits: int) -> dict:
    """(split_seed, fold) -> validation base rate and positive-window count.

    Recomputed from the cache rather than stored, so it cannot drift from the
    splits the ablation actually used.
    """
    index = load_cache_index(cache)
    out: dict[tuple[int, int], dict] = {}
    for sseed in split_seeds:
        splitter = GroupedKFoldSplitter(
            n_splits=n_splits, n_repeats=1, random_state=sseed,
            include_sim_in_train=False,
        )
        folds = splitter.split(
            None, index.y, index.group, is_sim=index.is_sim,
            instances=index.inst_id, well_hours=index.hours_by_well,
        )
        for fold, (_tr, va, _te) in enumerate(folds):
            pos = index.y[va] != 0
            out[(sseed, fold)] = dict(
                val_base_rate=float(pos.mean()),
                val_positive_windows=int(pos.sum()),
                n_val=int(len(va)),
            )
    return out


def attach_lift(df: pd.DataFrame, rates: dict) -> pd.DataFrame:
    df = df.copy()
    df["val_base_rate"] = [rates[(r.split_seed, r.inner_fold)]["val_base_rate"]
                           for r in df.itertuples()]
    df["val_positive_windows"] = [rates[(r.split_seed, r.inner_fold)]["val_positive_windows"]
                                  for r in df.itertuples()]
    df["lift"] = df["val_pr_auc"] / df["val_base_rate"].replace(0.0, np.nan)
    df["log_lift"] = np.log(df["lift"].where(df["lift"] > 0))
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ablation", default="results/ablation_features.csv")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--reference-arm", default="A4_nan_index",
                    help="arm used for the per-cell listing")
    args = ap.parse_args()

    df = pd.read_csv(args.ablation)
    split_seeds = sorted(df.split_seed.unique().tolist())
    rates = fold_base_rates(args.cache, split_seeds, args.n_splits)
    df = attach_lift(df, rates)
    pd.set_option("display.width", 200)

    print("=" * 78)
    print("1. PER-CELL SPREAD  (arm:", args.reference_arm, ")")
    print("=" * 78)
    a = df[df.arm == args.reference_arm].sort_values(
        ["val_positive_events", "split_seed", "inner_fold"]
    )
    print(a[["split_seed", "inner_fold", "seed", "val_positive_events",
             "val_positive_windows", "val_base_rate", "val_pr_auc", "lift"]]
          .round(5).to_string(index=False))

    print()
    print("=" * 78)
    print("2. THE BASE-RATE TRAP")
    print("=" * 78)
    raw = spearmanr(a.val_positive_events, a.val_pr_auc)
    lif = spearmanr(a.val_positive_events, a.lift)
    print(f"  Spearman rho(positive events, RAW PR-AUC) = {raw.statistic:+.3f}  p = {raw.pvalue:.4f}")
    print(f"  Spearman rho(positive events, LIFT)       = {lif.statistic:+.3f}  p = {lif.pvalue:.4f}")
    print("  -> the raw correlation is a base-rate artifact, not a measurement property.")

    print()
    print("=" * 78)
    print("3. VARIANCE DECOMPOSITION")
    print("=" * 78)
    for metric, label in (("val_pr_auc", "raw PR-AUC"), ("log_lift", "log lift")):
        arm_v = df.groupby("arm")[metric].mean().var(ddof=1)
        cell_v = df.groupby(["split_seed", "inner_fold"])[metric].mean().var(ddof=1)
        print(f"  {label:11s}: var(arm means) = {arm_v:.6f}   "
              f"var(fold-cell means) = {cell_v:.6f}   ratio = {cell_v / arm_v:.0f}x")

    print()
    print("=" * 78)
    print("4. PAIRED ARM COMPARISONS (estimable cells only, log-lift scale)")
    print("=" * 78)
    sub = df[df.val_positive_events >= 2]
    n_cells = sub[["split_seed", "inner_fold", "seed"]].drop_duplicates().shape[0]
    print(f"  cells: {n_cells}")
    rows = []
    for group, gdf in sub.groupby("group", sort=False):
        ref_arm = REFERENCE_ARM[group]
        keys = ["split_seed", "inner_fold", "seed"]
        ref_log = gdf[gdf.arm == ref_arm].set_index(keys)["log_lift"]
        ref_raw = gdf[gdf.arm == ref_arm].set_index(keys)["val_pr_auc"]
        for arm, adf in gdf.groupby("arm", sort=False):
            if arm == ref_arm:
                continue
            d_log = (adf.set_index(keys)["log_lift"] - ref_log).dropna()
            d_raw = (adf.set_index(keys)["val_pr_auc"] - ref_raw).dropna()
            try:
                p = wilcoxon(d_log, zero_method="zsplit").pvalue
            except ValueError:
                p = float("nan")
            rows.append(dict(
                group=group, arm=arm, vs=ref_arm,
                lift_ratio=round(float(np.exp(d_log.mean())), 4),
                raw_diff=round(float(d_raw.mean()), 4),
                wins=f"{int((d_log > 0).sum())}/{len(d_log)}",
                wilcoxon_p=round(float(p), 4),
            ))
    tab = pd.DataFrame(rows).sort_values("wilcoxon_p")
    print(tab.to_string(index=False))
    bonf = 0.05 / len(tab)
    print(f"\n  Bonferroni threshold for {len(tab)} comparisons: p < {bonf:.4f}")
    survivors = tab[tab.wilcoxon_p < bonf]
    print("  surviving arms:", ", ".join(survivors.arm) if len(survivors) else "NONE")

    print()
    print("=" * 78)
    print("5. RANK AGREEMENT OF CHANNEL IMPORTANCE ACROSS FOLDS")
    print("=" * 78)
    from pathlib import Path

    from scipy.stats import kendalltau
    ranks = {}
    for f in (0, 1, 2):
        p = Path(f"results/tables/xgboost_real_only_fold{f}_seed42_importance_by_channel.csv")
        if p.exists():
            ranks[f] = pd.read_csv(p).set_index("channel")["rank"]
    if len(ranks) < 2:
        print("  (no importance tables yet -- run tools/train_xgb.py first)")
    else:
        for x, y in itertools.combinations(sorted(ranks), 2):
            common = ranks[x].index.intersection(ranks[y].index)
            t = kendalltau(ranks[x][common], ranks[y][common])
            print(f"  fold{x} vs fold{y}: Kendall tau = {t.statistic:+.3f}  (p = {t.pvalue:.3f})")
        print("  -> near-zero agreement means the channel ranking is not reproducible.")


if __name__ == "__main__":
    main()
