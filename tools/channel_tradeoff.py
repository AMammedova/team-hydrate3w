"""
Pick the channel set by what it costs in POSITIVE EVENTS, not in wells.

Why this supersedes tools/channel_availability.py's ranking (DATA_FINDINGS.md
§9): that script scored a channel as "available in a well" using a per-well
mean over that well's instances. It therefore missed instance-level sensor
death, which is what actually decides whether an event survives. Three of the
14 real transient events -- WELL-00040_20181013160242,
WELL-00041_20181013160201 and WELL-00014_20170214190000 (one of only three
events that reach blockage) -- have P-TPT and T-TPT frozen flat for the entire
recording, while P-ANULAR / P-JUS-CKGL / P-MON-CKP / QGL / T-JUS-CKP are fully
alive in all three. A P-TPT+T-TPT cache silently drops those events.

So the figure of merit is per instance, not per well:

    usable_frac(instance, set) = mean over channels of (1 - missing_or_frozen)

which is what WindowBuilder's min_valid_frac actually tests, aggregated over
the instance instead of per window. An instance contributes its event when
usable_frac >= min_valid_frac.

    python tools/channel_tradeoff.py [--min-valid-frac 0.5] [--max-set-size 8]

Prints, for greedy sets of increasing size and for a few named candidates:
transient events kept (of 14), blockage events kept (of 3), positive wells kept
(of 7), and Normal operating hours kept (of ~3,646).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.inventory import ThreeWDataset
from src.data.windowing import frozen_run_mask

TRANSIENT_RAW = 109
BLOCKED_RAW = 9


def usable_matrix(root: str, event: int, frozen_run_seconds: int) -> pd.DataFrame:
    """
    One row per real instance, one column per channel: the fraction of samples
    that are both present and not frozen. Plus the columns the trade-off needs
    (well, hours, whether the instance carries a transient / a blockage).
    """
    ds = ThreeWDataset(root, event_code=event)
    rows = []
    for cls in (str(event), "0"):
        for p in sorted((Path(root) / cls).glob("WELL-*.parquet")):
            inst = ds._load_one(p, event)
            raw = inst.df["class"].to_numpy(dtype="float64")
            row = {
                "instance": p.stem,
                "well": re.match(r"(WELL-\d+)_", p.name).group(1),
                "cls": cls,
                "hours": float((raw == 0).sum()) / 3600.0,
                "has_transient": bool((raw == TRANSIENT_RAW).any()),
                "has_blockage": bool((raw == BLOCKED_RAW).any()),
            }
            for c in inst.variable_columns():
                v = inst.df[c].to_numpy(dtype="float64")
                bad = np.isnan(v) | frozen_run_mask(v, frozen_run_seconds)
                row[c] = 1.0 - float(bad.mean())
            rows.append(row)
    return pd.DataFrame(rows)


META = ("instance", "well", "cls", "hours", "has_transient", "has_blockage")


def score(df: pd.DataFrame, chans: list[str], min_valid_frac: float) -> dict:
    usable = df[chans].mean(axis=1) >= min_valid_frac
    keep = df[usable]
    tr = keep[keep.has_transient]
    return {
        "n_channels": len(chans),
        "transient_events": int(len(tr)),
        "blockage_events": int(keep.has_blockage.sum()),
        "positive_wells": int(tr.well.nunique()),
        "normal_hours": round(float(keep.loc[keep.cls == "0", "hours"].sum()), 1),
        "channels": chans,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/3W/dataset")
    ap.add_argument("--event", type=int, default=9)
    ap.add_argument("--frozen-run-seconds", type=int, default=60)
    ap.add_argument("--min-valid-frac", type=float, default=0.5)
    ap.add_argument("--max-set-size", type=int, default=8)
    ap.add_argument("--out", default="results/channel_tradeoff.csv")
    args = ap.parse_args()

    df = usable_matrix(args.root, args.event, args.frozen_run_seconds)
    chans = [c for c in df.columns if c not in META]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(args.out).with_name("channel_usable_per_instance.csv"), index=False)

    total_tr = int(df.has_transient.sum())
    total_bl = int(df.has_blockage.sum())
    total_h = round(float(df.loc[df.cls == "0", "hours"].sum()), 1)
    print(f"totals: {total_tr} transient events, {total_bl} blockage events, "
          f"{df[df.has_transient].well.nunique()} positive wells, {total_h} normal hours\n")

    print("=== single channels, ranked by transient events then normal hours ===")
    singles = sorted(
        (score(df, [c], args.min_valid_frac) for c in chans),
        key=lambda s: (-s["transient_events"], -s["normal_hours"]),
    )
    for s in singles[:10]:
        print(f"  {s['channels'][0]:<14} events {s['transient_events']:>2}/{total_tr}  "
              f"blockage {s['blockage_events']}/{total_bl}  wells {s['positive_wells']}/7  "
              f"hours {s['normal_hours']:>7}")

    print("\n=== greedy sets (add the channel that keeps the most events, then hours) ===")
    chosen: list[str] = []
    results = []
    for _ in range(min(args.max_set_size, len(chans))):
        best = None
        for c in chans:
            if c in chosen:
                continue
            s = score(df, chosen + [c], args.min_valid_frac)
            key = (s["transient_events"], s["blockage_events"], s["normal_hours"])
            if best is None or key > best[0]:
                best = (key, c, s)
        if best is None:
            break
        chosen.append(best[1])
        results.append(best[2])
        s = best[2]
        print(f"  n={s['n_channels']}: events {s['transient_events']:>2}/{total_tr}  "
              f"blockage {s['blockage_events']}/{total_bl}  wells {s['positive_wells']}/7  "
              f"hours {s['normal_hours']:>7}  {s['channels']}")

    named = {
        "P-TPT+T-TPT (current cache)": ["P-TPT", "T-TPT"],
        "5-ch alive in the 3 dead-sensor events": [
            "P-ANULAR", "P-JUS-CKGL", "P-MON-CKP", "QGL", "T-JUS-CKP"],
        "K>=7 set from channel_availability": [
            "ESTADO-PXO", "ESTADO-W2", "ESTADO-XO", "P-MON-CKP", "P-PDG", "P-TPT", "T-TPT"],
    }
    print("\n=== named candidates ===")
    for label, cs in named.items():
        cs = [c for c in cs if c in chans]
        s = score(df, cs, args.min_valid_frac)
        print(f"  {label}")
        print(f"      n={s['n_channels']}  events {s['transient_events']:>2}/{total_tr}  "
              f"blockage {s['blockage_events']}/{total_bl}  wells {s['positive_wells']}/7  "
              f"hours {s['normal_hours']:>7}")

    pd.DataFrame(results).to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
