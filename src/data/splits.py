from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from src.data.fold_report_latex import fold_report_to_latex

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

    cache = Path(cache_dir)
    files = sorted(cache.glob("*.npz"))
    if not files:
        raise FileNotFoundError(
            f"no .npz files in {cache} -- run `python -m src.data.build_cache` "
            f"first (or point at a fake-data cache if the real one isn't built yet)"
        )
    return files


def load_cache_index(cache_dir: str | Path) -> CacheIndex:

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
    pool: Sequence,
    hours: Mapping,
    floor: float,
    rng: np.random.Generator,
) -> list:

    pool = list(pool)

    if len(pool) <= 1:
        return []

    ordered = sorted(
        rng.permutation(np.array(pool, dtype=object)),
        key=lambda w: float(hours.get(w, 0.0)),
    )

    picked: list = []
    total = 0.0

    for well in ordered[:-1]:  # keep at least one Normal well in train
        if total >= floor:
            break

        picked.append(well)
        total += float(hours.get(well, 0.0))

    return picked


class GroupedKFoldSplitter:

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
            raise ValueError(
                f"val_mode must be 'nested' or 'rotate', got {val_mode!r}"
            )

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
            raise ValueError(
                f"y has {len(y)} rows but groups has {len(groups)}"
            )

        if is_sim is None:
            is_sim = np.array(
                [_is_sim_group(g) for g in groups],
                dtype=bool,
            )

            if groups.dtype.kind in "iu":
                logger.warning(
                    "groups are integer ids and is_sim was not passed -- assuming "
                    "no simulated instances. Pass is_sim (CacheIndex.is_sim) or "
                    "simulated wells may leak into val/test folds."
                )

        is_sim = np.asarray(is_sim).astype(bool)

        df = pd.DataFrame(
            {
                "group": groups,
                "y": y,
                "is_sim": is_sim,
            }
        )

        if instances is not None:
            df["inst"] = np.asarray(instances)

        rows = []

        for well, part in df.groupby("group", sort=True):
            positive = part["y"] > 0

            if instances is not None:
                n_events = int(
                    part.loc[positive, "inst"].nunique()
                )
            else:
                n_events = int(positive.sum())

            rows.append(
                {
                    "well": well,
                    "n_windows": int(len(part)),
                    "n_positive_windows": int(positive.sum()),
                    "n_positive_events": n_events,
                    "is_sim": bool(part["is_sim"].any()),
                    "normal_hours": float(
                        (well_hours or {}).get(well, 0.0)
                    ),
                }
            )

        table = pd.DataFrame(rows).set_index("well")
        table["is_positive_well"] = (
            table["n_positive_windows"] > 0
        )

        return table

    def _n_folds(self, n_positive_wells: int) -> int:
        if self.n_splits == LEAVE_ONE_WELL_OUT:
            return n_positive_wells

        if self.n_splits < 2:
            raise ValueError(
                f"n_splits must be >= 2 or LEAVE_ONE_WELL_OUT (-1), "
                f"got {self.n_splits}"
            )

        if self.n_splits > n_positive_wells:
            raise ValueError(
                f"n_splits={self.n_splits} but only {n_positive_wells} "
                f"well(s) carry a positive window -- at least one test "
                f"fold would contain zero positives and its event recall "
                f"would be undefined. Lower n_splits "
                f"(DATA_FINDINGS.md §6 recommends 3) or use "
                f"LEAVE_ONE_WELL_OUT."
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
        table = self._well_table(
            y,
            groups,
            is_sim,
            instances,
            well_hours,
        )

        real = table[~table["is_sim"]]

        positive_wells = list(
            real.index[real["is_positive_well"]]
        )
        normal_wells = list(
            real.index[~real["is_positive_well"]]
        )

        if not positive_wells:
            raise ValueError(
                "no well carries a positive window -- cannot build folds"
            )

        k = self._n_folds(len(positive_wells))

        pos_weight = real["n_positive_events"].to_dict()
        hour_weight = real["normal_hours"].to_dict()

        sim_wells = set(
            table.index[table["is_sim"]]
        )
        positive_set = set(positive_wells)

        for repeat in range(self.n_repeats):
            rng = np.random.default_rng(
                self.random_state + repeat
            )

            pos_folds = _balanced_fold_assignment(
                positive_wells,
                pos_weight,
                k,
                rng,
            )

            norm_folds = _balanced_fold_assignment(
                normal_wells,
                hour_weight,
                k,
                rng,
            )

            fold_wells = [
                tuple(
                    sorted(
                        set(pos_folds[i]) | set(norm_folds[i]),
                        key=str,
                    )
                )
                for i in range(k)
            ]

            for i in range(k):
                test_wells = fold_wells[i]

                if self.val_mode == "rotate":
                    val_wells = fold_wells[(i + 1) % k]

                else:
                    # Nested validation from this fold's training pool.
                    #
                    # Positive wells:
                    # early stopping needs positive events, so take one
                    # event-balanced validation slice.
                    #
                    # Normal wells:
                    # threshold selection needs enough Normal HOURS, so add
                    # the smallest Normal wells until the validation floor
                    # is met, while retaining at least one Normal well in train.
                    pool = [
                        w
                        for j, g in enumerate(fold_wells)
                        if j != i
                        for w in g
                    ]

                    pool_pos = [
                        w for w in pool
                        if w in positive_set
                    ]
                    pool_norm = [
                        w for w in pool
                        if w not in positive_set
                    ]

                    m = max(
                        2,
                        int(round(1.0 / self.val_frac)),
                    )
                    m = min(
                        m,
                        max(2, len(pool_pos)),
                    )

                    val_pos = _balanced_fold_assignment(
                        pool_pos,
                        pos_weight,
                        m,
                        rng,
                    )[0]

                    val_norm = _val_normal_wells(
                        pool_norm,
                        hour_weight,
                        self.min_val_normal_hours,
                        rng,
                    )

                    val_wells = tuple(
                        sorted(
                            set(val_pos) | set(val_norm),
                            key=str,
                        )
                    )

                held_out = (
                    set(test_wells)
                    | set(val_wells)
                )

                train_wells = (
                    set(real.index)
                    - held_out
                )

                if self.include_sim_in_train:
                    train_wells |= sim_wells

                train_wells = tuple(
                    sorted(
                        train_wells,
                        key=str,
                    )
                )

                self._check_fold(
                    repeat,
                    i,
                    table,
                    test_wells,
                    val_wells,
                    train_wells,
                )

                yield FoldSpec(
                    repeat=repeat,
                    fold=i,
                    test_wells=tuple(test_wells),
                    val_wells=tuple(val_wells),
                    train_wells=train_wells,
                )

    def _check_fold(
        self,
        repeat,
        fold,
        table,
        test_wells,
        val_wells,
        train_wells,
    ) -> None:
        test_s = set(test_wells)
        val_s = set(val_wells)
        train_s = set(train_wells)

        # 1. Well-level leakage check.
        overlap = (
            (test_s & val_s)
            | (test_s & train_s)
            | (val_s & train_s)
        )

        if overlap:
            raise AssertionError(
                f"repeat {repeat} fold {fold}: wells "
                f"{sorted(map(str, overlap))} appear on more than one side "
                f"of the split -- this is exactly the leakage grouped CV "
                f"exists to prevent"
            )

        # 2. Simulated-data leakage check.
        sim_wells = set(
            table.index[table["is_sim"]]
        )

        bad_sim = (
            sim_wells
            & (test_s | val_s)
        )

        if bad_sim:
            raise AssertionError(
                f"repeat {repeat} fold {fold}: simulated wells "
                f"{sorted(map(str, bad_sim))} reached a val/test fold "
                f"(DL3.2 forbids it)"
            )

        # 3. Training positive-well diversity.
        real_train_wells = [
            w
            for w in train_wells
            if w not in sim_wells
        ]

        train_positive_wells = [
            w
            for w in real_train_wells
            if bool(
                table.loc[
                    w,
                    "is_positive_well",
                ]
            )
        ]

        if len(train_positive_wells) < 2:
            msg = (
                f"repeat {repeat} fold {fold}: training split contains only "
                f"{len(train_positive_wells)} real positive well(s): "
                f"{sorted(map(str, train_positive_wells))}. "
                f"Unseen-well generalisation is poorly identified when the "
                f"model learns positive behaviour from fewer than 2 real wells. "
                f"Consider redesigning validation allocation or report this "
                f"explicitly as a limitation."
            )

            if self.strict:
                raise ValueError(msg)

            logger.warning(msg)

        # 4. Validation must contain positive events.
        val_events = (
            int(
                table.loc[
                    list(val_wells),
                    "n_positive_events",
                ].sum()
            )
            if val_wells
            else 0
        )

        if val_events == 0:
            msg = (
                f"repeat {repeat} fold {fold}: validation fold has zero "
                f"positive events. Early stopping on PR-AUC and threshold "
                f"selection are undefined here -- lower n_splits or switch "
                f"val_mode."
            )

            if self.strict:
                raise ValueError(msg)

            logger.warning(msg)

        # 5. Test Normal-hours floor.
        test_hours = 0.0

        if test_wells:
            sub = table.loc[
                list(test_wells)
            ]

            normal_only = sub[
                ~sub["is_positive_well"]
            ]

            test_hours = float(
                normal_only[
                    "normal_hours"
                ].sum()
            )

        if test_hours < self.min_test_normal_hours:
            msg = (
                f"repeat {repeat} fold {fold}: only "
                f"{test_hours:.1f} Normal hours in the test fold "
                f"(floor {self.min_test_normal_hours:.0f} h). "
                f"A 1-alarm-per-100-h budget cannot be measured reliably "
                f"on this fold; report it as a limitation or reduce the "
                f"number of folds."
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
    ) -> Iterator[
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
    ]:
        """
        Yield (train_idx, val_idx, test_idx) row indices per (repeat, fold).

        `X` is accepted for sklearn-shaped call sites and is only length
        checked -- the split is decided from `groups` and `y` alone.
        """
        groups = np.asarray(groups)

        if X is not None and len(X) != len(groups):
            raise ValueError(
                f"X has {len(X)} rows but groups has {len(groups)}"
            )

        for spec in self.iter_folds(
            y,
            groups,
            is_sim=is_sim,
            instances=instances,
            well_hours=well_hours,
        ):
            train_idx = np.flatnonzero(
                np.isin(
                    groups,
                    spec.train_wells,
                )
            )

            val_idx = np.flatnonzero(
                np.isin(
                    groups,
                    spec.val_wells,
                )
            )

            test_idx = np.flatnonzero(
                np.isin(
                    groups,
                    spec.test_wells,
                )
            )

            if (
                len(np.intersect1d(train_idx, test_idx))
                or len(np.intersect1d(val_idx, test_idx))
            ):
                raise AssertionError(
                    f"repeat {spec.repeat} fold {spec.fold}: row-level overlap "
                    f"between splits despite disjoint wells -- groups array is "
                    f"inconsistent"
                )

            yield (
                train_idx,
                val_idx,
                test_idx,
            )

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

        table = self._well_table(
            y,
            groups,
            is_sim,
            instances,
            well_hours,
        )

        groups_arr = np.asarray(groups)

        event_col = (
            "n_positive_events"
            if instances is not None
            else "n_positive_windows"
        )

        def _name(w):
            return str(
                (well_names or {}).get(
                    w,
                    w,
                )
            )

        def _side(wells):
            wells = list(wells)

            if not wells:
                return {
                    "wells": 0,
                    "windows": 0,
                    "events": 0,
                    "hours": 0.0,
                    "positive_wells": 0,
                    "names": "",
                }

            sub = table.loc[wells]

            normal_only = sub[
                ~sub["is_positive_well"]
            ]

            return {
                "wells": len(wells),
                "windows": int(
                    sub["n_windows"].sum()
                ),
                "events": int(
                    sub[event_col].sum()
                ),
                "hours": float(
                    normal_only[
                        "normal_hours"
                    ].sum()
                ),
                "positive_wells": int(
                    (
                        sub["is_positive_well"]
                        & ~sub["is_sim"]
                    ).sum()
                ),
                                "names": ",".join(
                    sorted(
                        _name(w)
                        for w in wells
                    )
                ),
            }

        rows = []

        for spec in self.iter_folds(
            y,
            groups_arr,
            is_sim=is_sim,
            instances=instances,
            well_hours=well_hours,
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

                    "n_train_positive_wells": tr["positive_wells"],
                    "n_val_positive_wells": va["positive_wells"],
                    "n_test_positive_wells": te["positive_wells"],

                    "n_train_windows": tr["windows"],
                    "n_val_windows": va["windows"],
                    "n_test_windows": te["windows"],

                    "n_val_positive_events": va["events"],
                    "n_test_positive_events": te["events"],

                    "val_normal_hours": round(
                        va["hours"],
                        1,
                    ),
                    "test_normal_hours": round(
                        te["hours"],
                        1,
                    ),

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

    parser.add_argument(
        "--cache",
        default="data/cache",
        help="dir written by build_cache",
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=3,
        help="folds, or -1 for leave-one-well-out",
    )

    parser.add_argument(
        "--n-repeats",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--val-mode",
        default="nested",
        choices=[
            "nested",
            "rotate",
        ],
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--out",
        default="results/fold_report.csv",
    )

    parser.add_argument(
        "--latex",
        default=None,
        help="also write the booktabs version here, e.g. "
             "report/tables/fold_report.tex (the paper \\input{}s it)",
    )

    parser.add_argument(
        "--label",
        default="tab:folds",
        help="LaTeX label for --latex output",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    idx = load_cache_index(
        args.cache
    )

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

    with pd.option_context(
        "display.width",
        200,
        "display.max_columns",
        50,
    ):
        print(
            report.to_string(
                index=False
            )
        )

    out = Path(
        args.out
    )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        out,
        index=False,
    )

    print(
        f"\nwrote {out}"
    )

    if args.latex:
        tex = Path(
            args.latex
        )
        tex.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        tex.write_text(
            fold_report_to_latex(
                report,
                label=args.label,
            ),
            encoding="utf8",
        )
        print(
            f"wrote {tex}"
        )


if __name__ == "__main__":
    _cli()
