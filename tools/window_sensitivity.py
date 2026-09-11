"""
Validation-only sensitivity study for window duration.

This script compares:
- 15-minute history: decimate=30, window_size=30
- 30-minute history: decimate=30, window_size=60

It only builds/summarizes caches.
It does NOT train models and does NOT touch test evaluation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.build_cache import DEFAULT_CONFIG, build_cache


ROOT = "data/3W/dataset"

PRIMARY_CACHE = "data/cache"

ARMS = [
    {
        "name": "15min",
        "out_dir": "data/cache_sens_15min",
        "decimate": 30,
        "window_size": 30,
        "reuse_primary": False,
    },
    {
        "name": "30min_current",
        "out_dir": PRIMARY_CACHE,
        "decimate": 30,
        "window_size": 60,
        "reuse_primary": True,
    },
]


def load_or_build_arm(arm: dict) -> pd.DataFrame:
    out_dir = Path(arm["out_dir"])
    summary_path = out_dir / "cache_summary.csv"

    if summary_path.exists():
        print(f"[{arm['name']}] reusing existing cache: {out_dir}")
        return pd.read_csv(summary_path)

    if arm["reuse_primary"]:
        raise RuntimeError(
            f"Primary cache expected at {out_dir}, but cache_summary.csv is missing."
        )

    cfg = {
        **DEFAULT_CONFIG,
        "decimate": arm["decimate"],
        "window_size": arm["window_size"],
    }

    print(
        f"[{arm['name']}] building cache "
        f"(decimate={arm['decimate']}, window_size={arm['window_size']})"
    )

    return build_cache(
        ROOT,
        str(out_dir),
        cfg,
    )


def summarize_arm(arm: dict, summary: pd.DataFrame) -> dict:
    real = summary[summary["source"] == "real"]
    kept_real = real[real["n_windows"] > 0]

    return {
        "arm": arm["name"],
        "decimate": arm["decimate"],
        "window_size": arm["window_size"],
        "history_minutes": (
            arm["decimate"] * arm["window_size"] / 60
        ),
        "real_instances_kept": int(
            (real["n_windows"] > 0).sum()
        ),
        "real_transient_events": int(
            (real["n_transient"] > 0).sum()
        ),
        "real_established_events": int(
            (real["n_established"] > 0).sum()
        ),
        "real_positive_wells": int(
            real.loc[
                real["n_transient"] > 0,
                "well_id",
            ].nunique()
        ),
        "real_normal_hours": round(
            float(kept_real["normal_hours"].sum()),
            1,
        ),
        "total_windows": int(
            summary["n_windows"].sum()
        ),
        "transient_windows_total": int(
            summary["n_transient"].sum()
        ),
        "established_windows_total": int(
            summary["n_established"].sum()
        ),
    }


def main() -> None:
    rows = []

    for arm in ARMS:
        summary = load_or_build_arm(arm)
        rows.append(
            summarize_arm(
                arm,
                summary,
            )
        )

    df = pd.DataFrame(rows)

    out_path = Path(
        "results/window_sensitivity_cache_summary.csv"
    )
    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        out_path,
        index=False,
    )

    print()
    print(df.to_string(index=False))
    print()
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()