"""
Well x channel availability, straight from parquet column statistics.

Why this exists (W1.4, and the 289-zero-window result in DATA_FINDINGS.md §8):
a single global channel list cannot serve this dataset, because different wells
instrument different channels. WELL-00002 records only P-ANULAR / P-PDG /
P-TPT / T-TPT; WELL-00008 records only P-JUS-CKGL. Any list fit on the union
leaves most wells with mostly-absent inputs, and min_valid_frac then discards
every one of their windows.

Reads null_count from each parquet's row-group statistics rather than the data
itself, so all 801 files scan in seconds.

    python tools/channel_availability.py [--root data/3W/dataset] [--event 9]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

NON_VARIABLE = {"class", "state", "timestamp", "__index_level_0__"}


def well_of(p: Path) -> str:
    m = re.match(r"(WELL-\d+)_", p.name)
    if m:
        return m.group(1)
    return "SIMULATED" if p.name.startswith("SIMULATED") else "DRAWN"


def scan(root: Path, event: int) -> pd.DataFrame:
    rows = []
    for cls in (str(event), "0"):
        for p in sorted((root / cls).glob("*.parquet")):
            md = pq.ParquetFile(p).metadata
            names = [f.name for f in pq.ParquetFile(p).schema_arrow]
            n_rows = md.num_rows
            nulls = {n: 0 for n in names}
            for rg in range(md.num_row_groups):
                for col in range(md.row_group(rg).num_columns):
                    st = md.row_group(rg).column(col)
                    name = st.path_in_schema
                    if st.statistics is not None and name in nulls:
                        nulls[name] += st.statistics.null_count
            row = {
                "instance": p.stem,
                "well": well_of(p),
                "cls": cls,
                "n_rows": n_rows,
            }
            for n in names:
                if n not in NON_VARIABLE:
                    row[n] = 1.0 - (nulls[n] / n_rows if n_rows else 1.0)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/3W/dataset")
    ap.add_argument("--event", type=int, default=9)
    ap.add_argument("--present-thresh", type=float, default=0.5,
                    help="a channel counts as available in a well when its mean "
                         "present-fraction across that well's instances is above this")
    ap.add_argument("--out", default="results/channel_availability.csv")
    args = ap.parse_args()

    df = scan(Path(args.root), args.event)
    chans = [c for c in df.columns if c not in ("instance", "well", "cls", "n_rows")]

    # Weight each instance by its length, so a 1-hour instance does not count
    # as much as a 40-hour one.
    def wmean(g: pd.DataFrame) -> pd.Series:
        w = g["n_rows"].to_numpy(dtype="float64")
        return pd.Series(
            {c: float(np.average(g[c].to_numpy(dtype="float64"), weights=w)) for c in chans}
        )

    per_well = df.groupby("well").apply(wmean, include_groups=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    per_well.to_csv(args.out)

    avail = per_well >= args.present_thresh
    hydrate_wells = sorted(
        w for w in df.loc[df.cls == str(args.event), "well"].unique() if w.startswith("WELL")
    )
    normal_wells = sorted(df.loc[df.cls == "0", "well"].unique())

    pd.set_option("display.width", 200)
    print("=== channels available per well (present fraction >= %.2f) ===" % args.present_thresh)
    print(avail.loc[[w for w in avail.index if w.startswith("WELL")]].astype(int).to_string())

    print("\n=== channel counts ===")
    print(f"{'channel':<16} {'hydrate wells':>14} {'normal wells':>13} {'both':>6}")
    both = []
    for c in chans:
        h = sum(bool(avail.loc[w, c]) for w in hydrate_wells if w in avail.index)
        n = sum(bool(avail.loc[w, c]) for w in normal_wells if w in avail.index)
        print(f"{c:<16} {h:>10}/{len(hydrate_wells):<3} {n:>9}/{len(normal_wells):<3} "
              f"{'YES' if h and n else '':>6}")
        if h and n:
            both.append((c, h, n))

    print("\n=== channels present in at least one hydrate AND one normal well ===")
    for c, h, n in sorted(both, key=lambda t: -(t[1] + t[2])):
        print(f"  {c:<16} hydrate {h}/{len(hydrate_wells)}  normal {n}/{len(normal_wells)}")

    print("\n=== usable (wells x channels) trade-off ===")
    print("For each candidate channel set = channels available in >= K of the 9 normal")
    print("wells, how many wells keep every channel in that set:")
    for K in range(len(normal_wells), 0, -1):
        chan_set = [
            c for c in chans
            if sum(bool(avail.loc[w, c]) for w in normal_wells if w in avail.index) >= K
        ]
        if not chan_set:
            continue
        keep_n = [w for w in normal_wells if all(bool(avail.loc[w, c]) for c in chan_set)]
        keep_h = [w for w in hydrate_wells if all(bool(avail.loc[w, c]) for c in chan_set)]
        print(f"  K>={K}: {len(chan_set):>2} channels -> {len(keep_n)}/{len(normal_wells)} normal wells, "
              f"{len(keep_h)}/{len(hydrate_wells)} hydrate wells   {chan_set}")

    print(f"\nwrote per-well table to {args.out}")


if __name__ == "__main__":
    main()
