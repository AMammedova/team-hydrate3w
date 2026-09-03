"""
Module 3 — Grouped Cross-Validation Splitter (Member 2, W2.1-W2.3).
See DL_Project_Statement_Hydrate3W.docx, section 6 (DL3.1-DL3.3), and
DATA_FINDINGS.md §2, which is what forced the design below.

WHY THIS IS NOT A PLAIN StratifiedGroupKFold
--------------------------------------------
Measured on the real download (DATA_FINDINGS.md §2): the hydrate wells
and the Normal-Operation wells are DISJOINT -- no well appears in both.
Two consequences drive everything here:

  1. Well identity alone predicts the label. A model can "solve" the task
     by recognising a well's sensor offsets. Well-level grouping is
     therefore not a nicety, it is the only thing making the numbers mean
     anything (per-instance normalisation in windowing.py is the other
     half of that defence).
  2. Running one StratifiedGroupKFold over the union of both well
     populations leaves fold composition to chance: one fold can get 5
     Normal wells and another 1. Normal hours are the denominator of the
     false-alarm rate, and they are wildly unequal (WELL-00002 alone has
     1220 h, WELL-00007 has 6 h), so an unlucky fold ends up with ~46 h
     of Normal -- far too little to calibrate a 1-alarm-per-100-h budget.

So we run TWO INDEPENDENT grouped splits and pair them fold by fold:

    positive wells  --(balanced on positive EVENTS)-->  P0 P1 P2
    normal wells    --(balanced on NORMAL HOURS)---->   N0 N1 N2
    fold i test wells = Pi u Ni

Balancing is longest-processing-time-first greedy (assign the heaviest
remaining well to the lightest fold), which is what keeps WELL-00042
(5 of the 14 transient instances) and WELL-00002 (36% of all Normal
hours) from dominating a single fold.

VALIDATION FOLD
---------------
split() yields a NESTED (train_idx, val_idx, test_idx). val_idx is carved
out of that fold's TRAINING wells only -- never its test wells -- because
Module 7's early stopping and Module 8's threshold/smoothing selection
both need a validation set that is not the test set.

Validation carries two jobs that need different things, so
`val_mode="nested"` (the default) picks its wells two different ways:

  * EARLY STOPPING needs positive events. Sampling `val_frac` of the
    training wells at random regularly yields zero of them across 7
    positive wells, which leaves PR-AUC undefined -- so positive wells
    are taken as one event-balanced slice of the training pool.
  * THRESHOLD SELECTION needs Normal HOURS: Module 8 tunes for 1 alarm
    per 100 h on validation. A proportional slice does not guarantee
    them -- on the real cache it left one fold with 4.0 validation hours
    -- so Normal wells are added smallest-first until
    `min_val_normal_hours` is met, and no further.

`val_mode="rotate"` (fold i validates on fold i+1's wells) is kept
because it is simpler to explain, but at k=3 it spends a full third of
the data on validation and leaves the model less training data than
validation data -- measured on the real cache: 12.4k train vs 18.3k val
windows. Prefer "nested" unless you have a reason not to.

SIMULATED INSTANCES
-------------------
Simulated pseudo-wells (inventory.py gives each its own singleton group)
are excluded from the test and validation candidate pools entirely, so
DL3.2 ("no simulated instance in a val/test fold") holds by construction
rather than by a downstream filter. They land in train_idx only when
`include_sim_in_train=True` -- that flag is how a caller switches between
the `real_only` and `real_plus_sim` conditions of Result 1.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# n_splits sentinel: one fold per positive well (7 folds on the real data).
LEAVE_ONE_WELL_OUT = -1

# inventory.py's pseudo-group prefixes for non-real instances.
_SIM_PREFIXES = ("SIM-", "DRAWN-")


# --------------------------------------------------------------------------
# Cache index -- the per-window metadata split() needs, without loading X
# --------------------------------------------------------------------------


@dataclass
class CacheIndex:
    """
    Per-window metadata for a built cache, loaded WITHOUT touching X.

    build_cache.py writes one .npz per instance holding X, mask, y, group,
    inst_id, t_end, is_sim and the scalars failure_time / blockage_time /
    normal_hours. Fold design needs everything except X and mask, and X is
    the only expensive part -- so this reads just the small arrays. That
    keeps fold_report() (Table 1 of the report) runnable on a laptop.
    """

    y: np.ndarray               # (N,) window labels 0/1/2
    group: np.ndarray           # (N,) well group id
    inst_id: np.ndarray         # (N,) instance id
    is_sim: np.ndarray          # (N,) 1 for simulated/drawn windows
    t_end: np.ndarray           # (N,) window end time, seconds from instance start
    failure_time: np.ndarray    # (N,) transient onset of this window's instance
    blockage_time: np.ndarray   # (N,) blockage onset, NaN in all but 3 instances
    hours_by_well: dict         # group id -> Normal hours summed over instances
    well_of_group: dict = field(default_factory=dict)   # group id -> "WELL-000NN"
    files: list = field(default_factory=list)           # instance order on disk
    n_per_file: list = field(default_factory=list)      # windows contributed by each

    def __len__(self) -> int:
        return int(len(self.y))


def _cache_files(cache_dir: str | Path) -> list[Path]:
    """
    THE row order of a cache, defined in exactly one place.

    Every index this module hands out (train_idx/val_idx/test_idx) is an
    offset into arrays concatenated in this order. If a caller loads X in
    a different order, the indices still "work" -- they just point at the
    wrong rows, silently, and every downstream number is wrong without a
    single error being raised. So X and the metadata must come from the
    same enumeration: load_cache() and load_cache_index() both call this.
    """
    cache = Path(cache_dir)
    files = sorted(cache.glob("*.npz"))
    if not files:
        raise FileNotFoundError(
            f"no .npz files in {cache} -- run `python -m src.data.build_cache` "
            f"first (or point at a fake-data cache if the real one isn't built yet)"
        )
    return files


def load_cache_index(cache_dir: str | Path) -> CacheIndex:
    """
    Read every `<instance>.npz` in `cache_dir` and concatenate its metadata.

    `hours_by_well` sums each instance's `normal_hours` scalar (seconds
    labeled Normal / 3600) per well; it is the denominator behind the
    `test_normal_hours` column of fold_report().

    `failure_time` and `blockage_time` are per-INSTANCE scalars on disk;
    they are broadcast to every window of their instance here so Module 8
    can compute a lead time by row without re-opening the cache (and
    without re-deriving the row order -- see _cache_files).
    """
    files = _cache_files(cache_dir)

    ys, groups, insts, sims, tends, fails, blocks, counts = [], [], [], [], [], [], [], []
    hours_by_well: dict[int, float] = {}
    for path in files:
        with np.load(path) as z:
            n = int(len(z["y"]))
            ys.append(z["y"])
            groups.append(z["group"])
            insts.append(z["inst_id"])
            sims.append(z["is_sim"])
            tends.append(z["t_end"])
            fails.append(np.full(n, float(z["failure_time"]), dtype="float64"))
            blocks.append(np.full(n, float(z["blockage_time"]), dtype="float64"))
            counts.append(n)
            g = int(z["group"][0])
            hours_by_well[g] = hours_by_well.get(g, 0.0) + float(z["normal_hours"])

    well_of_group: dict[int, str] = {}
    sidecar = Path(cache_dir) / "cache_config.json"
    if sidecar.exists():
        group_map = json.loads(sidecar.read_text(encoding="utf8")).get("group_map", {})
        well_of_group = {int(v): k for k, v in group_map.items()}

    return CacheIndex(
        y=np.concatenate(ys),
        group=np.concatenate(groups),
        inst_id=np.concatenate(insts),
        is_sim=np.concatenate(sims),
        t_end=np.concatenate(tends),
        failure_time=np.concatenate(fails),
        blockage_time=np.concatenate(blocks),
        hours_by_well=hours_by_well,
        well_of_group=well_of_group,
        files=[p.stem for p in files],
        n_per_file=counts,
    )


def load_cache(cache_dir: str | Path) -> tuple[np.ndarray, np.ndarray, CacheIndex]:
    """
    Load `X`, `mask` and the matching CacheIndex in ONE guaranteed row order.

        X, mask, idx = load_cache("data/cache")
        for train_idx, val_idx, test_idx in splitter.split(
            X, idx.y, idx.group, is_sim=idx.is_sim,
            instances=idx.inst_id, well_hours=idx.hours_by_well,
        ):
            model.fit(X[train_idx], mask[train_idx], idx.y[train_idx])

    Use this rather than globbing the cache yourself: fold indices are
    positions in this concatenation, and a different enumeration order
    misaligns every row without raising anything.

    X is `[N, C, W]` float32, channels-first (contract §0.1) -- roughly
    100 MB for the real 5-channel cache, so it fits in memory; nothing
    here streams.
    """
    files = _cache_files(cache_dir)
    index = load_cache_index(cache_dir)

    Xs, masks = [], []
    for path in files:
        with np.load(path) as z:
            Xs.append(z["X"])
            masks.append(z["mask"])

    X = np.concatenate(Xs)
    mask = np.concatenate(masks)
    if len(X) != len(index) or len(mask) != len(index):
        raise AssertionError(
            f"cache is inconsistent: X has {len(X)} rows, mask {len(mask)}, metadata "
            f"{len(index)} -- rebuild the cache rather than indexing into this"
        )
    if X.ndim != 3:
        raise AssertionError(f"expected channels-first [N, C, W], got shape {X.shape}")
    return X, mask, index


# --------------------------------------------------------------------------
# Fold specification
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldSpec:
    """Which wells are on which side of one (repeat, fold)."""

    repeat: int
    fold: int
    test_wells: tuple
    val_wells: tuple
    train_wells: tuple


def _is_sim_group(g) -> bool:
    return isinstance(g, str) and g.startswith(_SIM_PREFIXES)


def _balanced_fold_assignment(
    wells: Sequence,
    weights: Mapping,
    k: int,
    rng: np.random.Generator,
    tolerance: float = 0.10,
) -> list[list]:
    """
    Longest-processing-time-first greedy: heaviest well to the lightest
    fold.

    Balancing on a weight (events, hours) rather than well COUNT is the
    point: 7 wells split 3/2/2 by count still puts 5 of the 14 transient
    instances in whichever fold holds WELL-00042.

    Plain LPT is deterministic, so `n_repeats > 1` would re-derive the
    same partition every time and the resulting "mean ± std" would
    understate the variance from fold composition -- the one thing extra
    repeats exist to measure. The fold is therefore drawn at random from
    those within `tolerance` of the lightest load (a fraction of the mean
    per-fold load). Tolerance 0 reduces to textbook LPT.
    """
    wells = list(wells)
    if not wells:
        return [[] for _ in range(k)]
    shuffled = list(rng.permutation(np.array(wells, dtype=object)))
    order = sorted(shuffled, key=lambda w: -float(weights.get(w, 0.0)))

    total = sum(float(weights.get(w, 0.0)) for w in wells)
    slack = tolerance * total / k

    folds: list[list] = [[] for _ in range(k)]
    loads = np.zeros(k, dtype=float)
    for well in order:
        candidates = np.flatnonzero(loads <= loads.min() + slack)
        j = int(rng.choice(candidates))
        folds[j].append(well)
        loads[j] += float(weights.get(well, 0.0))
    return folds


def _val_normal_wells(
    pool: Sequence, hours: Mapping, floor: float, rng: np.random.Generator
) -> list:
    """
    Pick validation Normal wells by HOURS, smallest first, until `floor` is
    reached -- and never take the last Normal well away from training.

    Smallest-first is deliberate. Taking one large well would clear the
    floor in a single pick but hand validation most of the fold's Normal
    windows; the small wells clear it while leaving the bulk for training,
    and they make the validation set more diverse at the same time.
    """
    pool = list(pool)
    if len(pool) <= 1:
        return []
    ordered = sorted(rng.permutation(np.array(pool, dtype=object)),
                     key=lambda w: float(hours.get(w, 0.0)))
    picked: list = []
    total = 0.0
    for well in ordered[:-1]:          # keep at least one Normal well in train
        if total >= floor:
            break
        picked.append(well)
        total += float(hours.get(well, 0.0))
    return picked


class GroupedKFoldSplitter:
    """
    Well-level nested CV splitter for the 3W hydrate task.

    Parameters
    ----------
    n_splits
        Number of folds, or LEAVE_ONE_WELL_OUT (-1) for one fold per
        positive well. DATA_FINDINGS.md §6 recommends 3: with 7 positive
        wells, 5 folds leaves 1-2 positive wells per test fold and the
        per-fold metrics stop being estimable. Decide from fold_report()
        and justify the choice in the report.
    n_repeats
        Repeats of the whole scheme with a reshuffled assignment. Reduced
        to 1 for the 7 Sep deadline (TEAM_5_MEMBERS.md §0).
    val_mode
        "nested" (default): partition this fold's training wells into
        round(1/val_frac) balanced parts and validate on one of them.
        "rotate": fold i validates on fold (i+1)'s wells -- simpler, but
        at k=3 it leaves the model less training data than validation data.
    min_test_normal_hours
        Fold-level sanity floor for the false-alarm denominator. A fold
        below this cannot support a 1-per-100-h budget; the splitter warns
        (or raises, with `strict=True`) instead of letting M5 compute a
        threshold on 46 hours of Normal.
    min_val_normal_hours
        The same floor on the VALIDATION side, where Module 8 actually
        selects the threshold. Only used by val_mode="nested"; it is what
        decides how many Normal wells validation borrows from training.
    include_sim_in_train
        False reproduces the `real_only` condition of Result 1; True the
        `real_plus_sim` condition. Simulated wells are never eligible for
        val/test either way.
    """

    def __init__(
        self,
        n_splits: int = 3,
        n_repeats: int = 1,
        val_frac: float = 0.2,
        group_col: str = "well_id",
        random_state: int = 42,
        val_mode: str = "nested",
        min_test_normal_hours: float = 300.0,
        min_val_normal_hours: float = 300.0,
        include_sim_in_train: bool = True,
        strict: bool = False,
    ) -> None:
        if val_mode not in ("nested", "rotate"):
            raise ValueError(f"val_mode must be 'nested' or 'rotate', got {val_mode!r}")
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.val_frac = val_frac
        self.group_col = group_col
        self.random_state = random_state
        self.val_mode = val_mode
        self.min_test_normal_hours = min_test_normal_hours
        self.min_val_normal_hours = min_val_normal_hours
        self.include_sim_in_train = include_sim_in_train
        self.strict = strict

    # -- internals ---------------------------------------------------------

    def _well_table(
        self,
        y: np.ndarray,
        groups: np.ndarray,
        is_sim: np.ndarray | None,
        instances: np.ndarray | None,
        well_hours: Mapping | None,
    ) -> pd.DataFrame:
        """
        One row per well: window/event counts, Normal hours, sim flag, and
        whether the well carries any positive window at all.

        "Events" are distinct INSTANCES containing at least one positive
        window when `instances` is given (14 transient instances over 7
        wells on the real data); without it, positive windows are counted
        instead and the fold_report column is named accordingly.
        """
        y = np.asarray(y)
        groups = np.asarray(groups)
        if len(y) != len(groups):
            raise ValueError(f"y has {len(y)} rows but groups has {len(groups)}")

        if is_sim is None:
            is_sim = np.array([_is_sim_group(g) for g in groups], dtype=bool)
            if groups.dtype.kind in "iu":
                logger.warning(
                    "groups are integer ids and is_sim was not passed -- assuming "
                    "no simulated instances. Pass is_sim (CacheIndex.is_sim) or "
                    "simulated wells may leak into val/test folds."
                )
        is_sim = np.asarray(is_sim).astype(bool)

        df = pd.DataFrame({"group": groups, "y": y, "is_sim": is_sim})
        if instances is not None:
            df["inst"] = np.asarray(instances)

        rows = []
        for well, part in df.groupby("group", sort=True):
            positive = part["y"] > 0
            if instances is not None:
                n_events = int(part.loc[positive, "inst"].nunique())
            else:
                n_events = int(positive.sum())
            rows.append(
                {
                    "well": well,
                    "n_windows": int(len(part)),
                    "n_positive_windows": int(positive.sum()),
                    "n_positive_events": n_events,
                    "is_sim": bool(part["is_sim"].any()),
                    "normal_hours": float((well_hours or {}).get(well, 0.0)),
                }
            )
        table = pd.DataFrame(rows).set_index("well")
        table["is_positive_well"] = table["n_positive_windows"] > 0
        return table

    def _n_folds(self, n_positive_wells: int) -> int:
        if self.n_splits == LEAVE_ONE_WELL_OUT:
            return n_positive_wells
        if self.n_splits < 2:
            raise ValueError(
                f"n_splits must be >= 2 or LEAVE_ONE_WELL_OUT (-1), got {self.n_splits}"
            )
        if self.n_splits > n_positive_wells:
            raise ValueError(
                f"n_splits={self.n_splits} but only {n_positive_wells} well(s) carry a "
                f"positive window -- at least one test fold would contain zero "
                f"positives and its event recall would be undefined. Lower n_splits "
                f"(DATA_FINDINGS.md §6 recommends 3) or use LEAVE_ONE_WELL_OUT."
            )
        return self.n_splits

    def iter_folds(
        self,
        y: np.ndarray,
        groups: np.ndarray,
        *,
        is_sim: np.ndarray | None = None,
        instances: np.ndarray | None = None,
        well_hours: Mapping | None = None,
    ) -> Iterator[FoldSpec]:
        """
        Yield the WELL-level fold design, before it is turned into row
        indices. fold_report() and split() are both thin wrappers around
        this, so the table in the report describes exactly the folds the
        models were trained on.
        """
        table = self._well_table(y, groups, is_sim, instances, well_hours)
        real = table[~table["is_sim"]]
        positive_wells = list(real.index[real["is_positive_well"]])
        normal_wells = list(real.index[~real["is_positive_well"]])
        if not positive_wells:
            raise ValueError("no well carries a positive window -- cannot build folds")

        k = self._n_folds(len(positive_wells))
        pos_weight = real["n_positive_events"].to_dict()
        hour_weight = real["normal_hours"].to_dict()
        sim_wells = set(table.index[table["is_sim"]])
        positive_set = set(positive_wells)

        for repeat in range(self.n_repeats):
            rng = np.random.default_rng(self.random_state + repeat)
            pos_folds = _balanced_fold_assignment(positive_wells, pos_weight, k, rng)
            norm_folds = _balanced_fold_assignment(normal_wells, hour_weight, k, rng)

            fold_wells = [
                tuple(sorted(set(pos_folds[i]) | set(norm_folds[i]), key=str))
                for i in range(k)
            ]

            for i in range(k):
                test_wells = fold_wells[i]

                if self.val_mode == "rotate":
                    val_wells = fold_wells[(i + 1) % k]
                else:
                    # nested: spend a slice of THIS fold's training pool on
                    # validation. The two populations are chosen by different
                    # rules because they answer different questions:
                    #   positives  -> early stopping needs SOME events, so an
                    #                 event-balanced 1/m slice is enough;
                    #   normals    -> threshold selection needs enough HOURS to
                    #                 resolve 1 alarm per 100 h, so wells are
                    #                 added until the floor is met rather than
                    #                 by proportion. Taking a proportional
                    #                 slice instead left one real fold with
                    #                 4.0 validation hours.
                    pool = [w for j, g in enumerate(fold_wells) if j != i for w in g]
                    pool_pos = [w for w in pool if w in positive_set]
                    pool_norm = [w for w in pool if w not in positive_set]
                    m = max(2, int(round(1.0 / self.val_frac)))
                    m = min(m, max(2, len(pool_pos)))
                    val_pos = _balanced_fold_assignment(pool_pos, pos_weight, m, rng)[0]
                    val_norm = _val_normal_wells(
                        pool_norm, hour_weight, self.min_val_normal_hours, rng
                    )
                    val_wells = tuple(sorted(set(val_pos) | set(val_norm), key=str))

                held_out = set(test_wells) | set(val_wells)
                train_wells = set(real.index) - held_out
                if self.include_sim_in_train:
                    train_wells |= sim_wells
                train_wells = tuple(sorted(train_wells, key=str))

                self._check_fold(repeat, i, table, test_wells, val_wells, train_wells)
                yield FoldSpec(
                    repeat=repeat,
                    fold=i,
                    test_wells=tuple(test_wells),
                    val_wells=tuple(val_wells),
                    train_wells=train_wells,
                )

    def _check_fold(self, repeat, fold, table, test_wells, val_wells, train_wells) -> None:
        """
        The guards that make the red lines in TEAM_5_MEMBERS.md §9 provable
        rather than aspirational. Raises on leakage (always) and on an
        unusable fold (warns, or raises when strict=True).
        """
        test_s, val_s, train_s = set(test_wells), set(val_wells), set(train_wells)
        overlap = (test_s & val_s) | (test_s & train_s) | (val_s & train_s)
        if overlap:
            raise AssertionError(
                f"repeat {repeat} fold {fold}: wells {sorted(map(str, overlap))} appear "
                f"on more than one side of the split -- this is exactly the leakage "
                f"grouped CV exists to prevent"
            )

        sim_wells = set(table.index[table["is_sim"]])
        bad_sim = sim_wells & (test_s | val_s)
        if bad_sim:
            raise AssertionError(
                f"repeat {repeat} fold {fold}: simulated wells {sorted(map(str, bad_sim))} "
                f"reached a val/test fold (DL3.2 forbids it)"
            )

        val_events = (
            int(table.loc[list(val_wells), "n_positive_events"].sum()) if val_wells else 0
        )
        if val_events == 0:
            msg = (
                f"repeat {repeat} fold {fold}: validation fold has zero positive events. "
                f"Early stopping on PR-AUC and threshold selection are both undefined "
                f"here -- lower n_splits or switch val_mode."
            )
            if self.strict:
                raise ValueError(msg)
            logger.warning(msg)

        test_hours = 0.0
        if test_wells:
            sub = table.loc[list(test_wells)]
            test_hours = float(sub.loc[~sub["is_positive_well"], "normal_hours"].sum())
        if test_hours < self.min_test_normal_hours:
            msg = (
                f"repeat {repeat} fold {fold}: only {test_hours:.1f} Normal hours in the "
                f"test fold (floor {self.min_test_normal_hours:.0f} h). A 1-alarm-per-100-h "
                f"budget cannot be measured on this fold; report it as a limitation or "
                f"reduce the number of folds."
            )
            if self.strict:
                raise ValueError(msg)
            logger.warning(msg)

    # -- public API --------------------------------------------------------

    def split(
        self,
        X: np.ndarray | None,
        y: np.ndarray,
        groups: np.ndarray,
        *,
        is_sim: np.ndarray | None = None,
        instances: np.ndarray | None = None,
        well_hours: Mapping | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Yield (train_idx, val_idx, test_idx) row indices per (repeat, fold).

        `X` is accepted for sklearn-shaped call sites and is only length
        checked -- the split is decided from `groups` and `y` alone.
        """
        groups = np.asarray(groups)
        if X is not None and len(X) != len(groups):
            raise ValueError(f"X has {len(X)} rows but groups has {len(groups)}")

        for spec in self.iter_folds(
            y, groups, is_sim=is_sim, instances=instances, well_hours=well_hours
        ):
            train_idx = np.flatnonzero(np.isin(groups, spec.train_wells))
            val_idx = np.flatnonzero(np.isin(groups, spec.val_wells))
            test_idx = np.flatnonzero(np.isin(groups, spec.test_wells))
            if len(np.intersect1d(train_idx, test_idx)) or len(
                np.intersect1d(val_idx, test_idx)
            ):
                raise AssertionError(
                    f"repeat {spec.repeat} fold {spec.fold}: row-level overlap between "
                    f"splits despite disjoint wells -- groups array is inconsistent"
                )
            yield train_idx, val_idx, test_idx

    def fold_report(
        self,
        y: np.ndarray,
        groups: np.ndarray,
        *,
        is_sim: np.ndarray | None = None,
        instances: np.ndarray | None = None,
        well_hours: Mapping | None = None,
        well_names: Mapping | None = None,
    ) -> pd.DataFrame:
        """
        Table 1 of the report: one row per (repeat, fold).

        Run this BEFORE any model training (DL3.3). It is what tells you
        whether n_splits is sane given 14 positive instances over 7 wells.
        Two columns decide that:

          n_val_positive_events   0 => early stopping / threshold selection
                                  are undefined for that fold.
          test_normal_hours       the false-alarm denominator (added on top
                                  of the statement's column list, per
                                  DATA_FINDINGS.md §2.3). Below ~300 h a
                                  1-per-100-h budget is not measurable.

        `test_normal_hours` sums `normal_hours` over the test wells that
        carry NO positive window. That is deliberately not the same set as
        "the class-0 folder": 43 of the 57 real Event-9 instances never
        reach a positive phase (DATA_FINDINGS.md §3), and build_cache
        records no class-folder flag (every file is loaded with
        event_code=9), so the cache cannot tell a class-0 recording from a
        quiet class-9 one. Both are all-Normal recordings, so both are
        legitimate false-alarm denominators -- the quiet hydrate-well ones
        are simply the harder negatives.

        Normal stretches INSIDE a positive well are excluded, which is the
        conservative direction: it can only understate the denominator,
        never inflate it. If src/eval/thresholds.py wants those hours too,
        that is a deliberate widening to agree on, not a silent one.
        """
        table = self._well_table(y, groups, is_sim, instances, well_hours)
        groups_arr = np.asarray(groups)
        event_col = "n_positive_events" if instances is not None else "n_positive_windows"

        def _name(w):
            return str((well_names or {}).get(w, w))

        def _side(wells):
            wells = list(wells)
            if not wells:
                return {"wells": 0, "windows": 0, "events": 0, "hours": 0.0, "names": ""}
            sub = table.loc[wells]
            normal_only = sub[~sub["is_positive_well"]]
            return {
                "wells": len(wells),
                "windows": int(sub["n_windows"].sum()),
                "events": int(sub[event_col].sum()),
                "hours": float(normal_only["normal_hours"].sum()),
                "names": ",".join(sorted(_name(w) for w in wells)),
            }

        rows = []
        for spec in self.iter_folds(
            y, groups_arr, is_sim=is_sim, instances=instances, well_hours=well_hours
        ):
            tr = _side(spec.train_wells)
            va = _side(spec.val_wells)
            te = _side(spec.test_wells)
            rows.append(
                {
                    "repeat": spec.repeat,
                    "fold": spec.fold,
                    "n_train_wells": tr["wells"],
                    "n_val_wells": va["wells"],
                    "n_test_wells": te["wells"],
                    "n_train_windows": tr["windows"],
                    "n_val_windows": va["windows"],
                    "n_test_windows": te["windows"],
                    "n_val_positive_events": va["events"],
                    "n_test_positive_events": te["events"],
                    "val_normal_hours": round(va["hours"], 1),
                    "test_normal_hours": round(te["hours"], 1),
                    "test_wells": te["names"],
                }
            )
        return pd.DataFrame(rows)


def _cli() -> None:
    """
    `python -m src.data.splits --cache data/cache --n-splits 3`

    Prints Table 1 straight from a built cache and writes it to
    results/fold_report.csv, so the report never hand-copies a fold count.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Print the grouped-CV fold report (Table 1 of the report)."
    )
    parser.add_argument("--cache", default="data/cache", help="dir written by build_cache")
    parser.add_argument(
        "--n-splits", type=int, default=3, help="folds, or -1 for leave-one-well-out"
    )
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--val-mode", default="nested", choices=["nested", "rotate"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/fold_report.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    idx = load_cache_index(args.cache)
    splitter = GroupedKFoldSplitter(
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        val_mode=args.val_mode,
        random_state=args.seed,
    )
    report = splitter.fold_report(
        idx.y,
        idx.group,
        is_sim=idx.is_sim,
        instances=idx.inst_id,
        well_hours=idx.hours_by_well,
        well_names=idx.well_of_group,
    )
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(report.to_string(index=False))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    _cli()
