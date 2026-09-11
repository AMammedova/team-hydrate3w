"""Member 2 -- does the channel list survive being fit per fold?

HY IT DOES NOT RE-READ THE DATASET PER FOLD
--------------------------------------------
VariableSelector.fit thresholds each variable on a mean effective-missing
fraction weighted by instance length. That statistic decomposes: measure
every instance once, then re-aggregate over whichever subset a fold calls
training. So this streams the 3W dataset a single time, not once per fold,
and the per-fold answers are exact rather than approximate.

    python -m tools.m2_channel_constancy --cache data/cache

Writes results/channel_constancy.csv (one row per fold x variable) and
prints the verdict.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.data.availability import variable_availability_table
from src.data.build_cache import _iter_instances
from src.data.inventory import ThreeWDataset
from src.data.splits import GroupedKFoldSplitter, load_cache_index

logger = logging.getLogger(__name__)


def per_instance_availability(
    root: str, event_code: int, include_normal: bool, frozen_run_seconds: int
) -> pd.DataFrame:
    """
    One row per (instance, variable): effective-missing fraction and the
    instance length that weights it.

    Reuses M1's variable_availability_table rather than re-deriving the
    frozen-run rule, so this measures what the selector actually thresholds
    and not a second opinion about it.
    """
    ds = ThreeWDataset(root_dir=root, event_code=event_code)
    rows = []
    for i, inst in enumerate(_iter_instances(ds, event_code, include_normal, None), 1):
        table = variable_availability_table([inst], frozen_run_seconds=frozen_run_seconds)
        for _, r in table.iterrows():
            rows.append(
                {
                    "instance_id": inst.instance_id,
                    "well_id": inst.well_id,
                    "variable": str(r["variable"]),
                    "pct_effective_missing": float(r["pct_effective_missing"]),
                    "n_timesteps": int(inst.n_timesteps),
                }
            )
        if i % 50 == 0:
            logger.info("measured %d instances", i)
        del inst
    return pd.DataFrame(rows)


def kept_for_subset(avail: pd.DataFrame, wells: set, max_missing_frac: float) -> list[str]:
    """Refit the selector's decision on one fold's training wells."""
    sub = avail[avail["well_id"].isin(wells)]
    if sub.empty:
        return []
    weighted = (
        sub.assign(w=lambda d: d["pct_effective_missing"] * d["n_timesteps"])
        .groupby("variable")
        .apply(lambda g: g["w"].sum() / g["n_timesteps"].sum(), include_groups=False)
    )
    return sorted(weighted[weighted <= max_missing_frac].index)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/3W/dataset")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--val-mode", default="nested")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/channel_constancy.csv")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    sidecar = json.loads((Path(args.cache) / "cache_config.json").read_text(encoding="utf8"))
    cfg = sidecar["config"]
    frozen_kept = sorted(sidecar["kept_channels"])
    max_missing_frac = float(cfg["max_missing_frac"])
    pinned = cfg.get("channels")

    logger.info(
        "frozen cache kept %d channels%s; max_missing_frac=%.2f",
        len(frozen_kept),
        " (PINNED via --channels, so the threshold never chose them)" if pinned else "",
        max_missing_frac,
    )

    idx = load_cache_index(args.cache)
    splitter = GroupedKFoldSplitter(
        n_splits=args.n_splits, n_repeats=1, val_mode=args.val_mode, random_state=args.seed
    )
    folds = list(
        splitter.iter_folds(
            idx.y, idx.group, is_sim=idx.is_sim,
            instances=idx.inst_id, well_hours=idx.hours_by_well,
        )
    )
    name_of = idx.well_of_group

    avail = per_instance_availability(
        args.root, int(cfg["event_code"]), bool(cfg["include_normal"]),
        int(cfg.get("frozen_run_seconds", 60)),
    )

    rows, per_fold = [], {}
    for spec in folds:
        train_wells = {str(name_of.get(w, w)) for w in spec.train_wells}
        kept = kept_for_subset(avail, train_wells, max_missing_frac)
        same = sorted(kept) == frozen_kept
        per_fold[spec.fold] = sorted(kept)
        for variable in sorted(set(kept) | set(frozen_kept)):
            rows.append(
                {
                    "fold": spec.fold,
                    "variable": variable,
                    "kept_refit_on_train_wells": variable in kept,
                    "kept_in_frozen_cache": variable in frozen_kept,
                }
            )
        logger.info(
            "fold %d: refit on %d training wells keeps %d channels -- %s",
            spec.fold, len(train_wells), len(kept),
            "same list the cache used" if same else "a different list from the cache's",
        )
        if not same:
            logger.info(
                "  threshold adds %s / drops %s versus the cache's list",
                sorted(set(kept) - set(frozen_kept)) or "-",
                sorted(set(frozen_kept) - set(kept)) or "-",
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)

    print()
    print(verdict(per_fold, frozen_kept, pinned))
    print(f"\nwrote {out}")


def verdict(per_fold: dict, frozen_kept: list[str], pinned: list[str] | None) -> str:
    """
    The leakage question is whether the channel list depends on WHICH WELLS
    a fold trains on -- so the test is whether the per-fold refits agree
    with EACH OTHER. Agreeing with the cache's frozen list is a separate
    question, and when the cache pinned its channels explicitly the two
    lists were chosen by different criteria, so a difference there says
    nothing about leakage. An earlier version of this script conflated the
    two and reported leakage wherever a pinned list differed from the
    threshold's; the distinction below is the whole point of the check.
    """
    lists = [tuple(v) for v in per_fold.values()]
    folds_agree = len(set(lists)) == 1
    matches_frozen = folds_agree and sorted(lists[0]) == sorted(frozen_kept)

    if not folds_agree:
        differing = {f: sorted(set(v) ^ set(lists[0])) for f, v in per_fold.items()}
        return (
            "VERDICT: the channel list DEPENDS ON THE FOLD.\n"
            f"Per-fold symmetric differences against fold 0: {differing}\n"
            "Fitting the selector once over every instance therefore carried\n"
            "information from held-out wells into the cache. Report it as a\n"
            "limitation; do not rebuild after seeing test scores."
        )

    kept = sorted(lists[0])
    head = (
        "VERDICT: the channel list is FOLD-INVARIANT. Refitting the selector on\n"
        f"each fold's training wells alone keeps the same {len(kept)} channels every\n"
        "time, so build_cache.py's single global fit -- the one documented\n"
        "exception to 'fit on the training fold only' -- provably changed nothing.\n"
        f"  fold-invariant threshold list: {kept}"
    )
    if matches_frozen:
        return head + "\n  ...and it equals the list the cache actually used."

    only_thresh = sorted(set(kept) - set(frozen_kept))
    only_frozen = sorted(set(frozen_kept) - set(kept))
    tail = (
        f"\n  cache's list:                 {sorted(frozen_kept)}\n"
        f"  threshold would add:          {only_thresh or '-'}\n"
        f"  threshold would drop:         {only_frozen or '-'}\n"
    )
    if pinned:
        tail += (
            "\nThis difference is NOT leakage. The cache pinned its channels with\n"
            "--channels, so the missingness threshold never made the choice: the\n"
            "pinned set was selected for EVENT COVERAGE (DATA_FINDINGS.md sec 8),\n"
            "which is a different criterion and a deliberate, documented one. What\n"
            "this run establishes is the leakage part -- that the data-driven route\n"
            "does not move with the folds either."
        )
    else:
        tail += (
            "\nThe cache's list came from the same threshold but a different instance\n"
            "set, so this gap is worth explaining in the paper."
        )
    return head + tail


if __name__ == "__main__":
    main()
