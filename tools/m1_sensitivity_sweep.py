"""
Member 1, W1's sensitivity experiments (TEAM_5_MEMBERS.md sec 2): three
one-factor-at-a-time sweeps around the primary 5-channel cache config,
each measured by how many windows -- and, more importantly, how many REAL
events -- survive.

    python -m tools.m1_sensitivity_sweep --root data/3W/dataset

Reuses src.data.build_cache.build_cache() directly (not the CLI) so results
land straight in one comparison table instead of five separate cache_config
files someone has to cross-reference by hand. An arm whose cache directory
already exists is summarised from it rather than rebuilt, so re-running this
after an interruption costs seconds instead of ~5 minutes per arm.

WHY THE REAL-ONLY COLUMNS EXIST. Window counts are dominated by the 150
simulated instances, which contain all three phases in every instance and so
mask what a config change costs on real data. The columns that decide
anything are real_transient_events / real_blockage_events / real_positive_wells
-- the same event-coverage criterion DATA_FINDINGS.md sec 9 used to reject
choosing channels by well coverage.

Arms:
  1. label_rule sweep       {most_severe, final_timestep, majority}, primary
                             5-channel set. is_monotonic_severity=57/57
                             (DATA_FINDINGS.md sec 1) predicts most_severe and
                             final_timestep land on identical counts -- this
                             sweep is what actually checks that.
  2. channel set            primary 5-channel arm vs the P-TPT/T-TPT arm
                             (DATA_FINDINGS.md sec 8-9): more normal hours,
                             fewer surviving events.
  3. nan_label_policy       drop (primary) vs normal, primary 5-channel set --
                             how many windows a NaN-labeled span costs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.build_cache import DEFAULT_CONFIG, build_cache

PRIMARY_CHANNELS = ["P-MON-CKP", "P-JUS-CKGL", "T-TPT", "T-JUS-CKP", "P-ANULAR"]
TPT_CHANNELS = ["P-TPT", "T-TPT"]


def _summarise(sweep: str, out_dir: str, summary: pd.DataFrame) -> dict:
    """One comparison row, from an arm's cache_summary.csv + cache_config.json."""
    sidecar = json.loads((Path(out_dir) / "cache_config.json").read_text(encoding="utf8"))
    cfg = sidecar["config"]
    real = summary[summary["source"] == "real"]
    kept = real[real["n_windows"] > 0]
    return {
        "sweep": sweep,
        "out_dir": out_dir,
        "label_rule": cfg["label_rule"],
        "nan_label_policy": cfg["nan_label_policy"],
        "channels": ",".join(sidecar["kept_channels"]),
        "n_instances": sidecar["n_instances"],
        "n_windows_total": sidecar["n_windows_total"],
        "n_windows_dropped_total": sidecar["n_windows_dropped_total"],
        # Real-data event coverage -- the numbers that actually decide the config.
        "real_transient_events": int((real["n_transient"] > 0).sum()),
        "real_blockage_events": int((real["n_established"] > 0).sum()),
        "real_positive_wells": int(real.loc[real["n_transient"] > 0, "well_id"].nunique()),
        "real_normal_hours": round(float(kept["normal_hours"].sum()), 1),
        # Window-level counts, all sources -- kept for completeness.
        "n_transient_windows": int(summary["n_transient"].sum()),
        "n_established_windows": int(summary["n_established"].sum()),
    }


def _run_arm(root: str, out_dir: str, sweep: str, config_overrides: dict, rebuild: bool) -> dict:
    cache_summary = Path(out_dir) / "cache_summary.csv"
    if cache_summary.exists() and not rebuild:
        print(f"[{sweep}] reusing existing cache at {out_dir}")
        summary = pd.read_csv(cache_summary)
    else:
        cfg = {**DEFAULT_CONFIG, "channels": PRIMARY_CHANNELS, **config_overrides}
        summary = build_cache(root, out_dir, cfg)
    return _summarise(sweep, out_dir, summary)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/3W/dataset")
    ap.add_argument("--out", default="results/m1_sensitivity_sweep.csv")
    ap.add_argument("--primary-cache", default="data/cache",
                    help="the already-built primary cache (most_severe / drop / 5-channel)")
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild every arm even if its cache directory already exists")
    args = ap.parse_args()

    arms = [
        # (sweep name, out_dir, config overrides)
        ("primary (baseline)", args.primary_cache, {}),
        ("label_rule", "data/cache_labelrule_final_timestep", {"label_rule": "final_timestep"}),
        ("label_rule", "data/cache_labelrule_majority", {"label_rule": "majority"}),
        ("channel_set", "data/cache_tpt", {"channels": TPT_CHANNELS}),
        ("nan_label_policy", "data/cache_nanpolicy_normal", {"nan_label_policy": "normal"}),
    ]

    rows = []
    for sweep, out_dir, overrides in arms:
        # The primary cache is never rebuilt here -- it is the cache every other
        # module trains against, and silently regenerating it from this script
        # would be a surprising side effect.
        rebuild = args.rebuild and out_dir != args.primary_cache
        rows.append(_run_arm(args.root, out_dir, sweep, overrides, rebuild))

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print()
    print(df.to_string(index=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
