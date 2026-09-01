"""
Member 1, W1.1 — the fake data generator. Highest priority, hour one.

Produces synthetic arrays matching the data contract exactly (team
contract §0.1), so Members 2, 3, and 4 can develop and test their entire
pipeline before the real 3W cache is ready. When the real cache lands,
everyone changes one path and reruns -- nothing else about their code
should need to change, which is exactly why matching the contract
shapes/dtypes exactly (not approximately) matters here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def make_fake_dataset(
    out_dir: str | Path,
    n_wells: int = 8,
    n_instances: int = 40,
    n_channels: int = 20,
    window_size: int = 60,
    windows_per_instance: int = 30,
    seed: int = 42,
) -> None:
    """
    Writes one .npz per fake "instance" into out_dir, matching the
    contract:
        X:        float32 [N, C, W]
        mask:     uint8   [N, C, W]
        y:        int64   [N]            0=normal, 1=transient, 2=blocked
        group:    int64   [N]            well id
        inst_id:  int64   [N]
        t_end:    float64 [N]
        is_sim:   uint8   [N]
    plus per-instance scalars failure_time, normal_hours.

    A detectable ramp is planted in two channels before the label
    flips to 2, and some values are randomly dropped from the mask, so
    downstream code is forced to handle both from day one.
    """
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_sim = max(1, n_instances // 5)  # ~20% simulated, roughly matching the real ratio's spirit
    is_sim_flags = np.array([0] * (n_instances - n_sim) + [1] * n_sim)
    rng.shuffle(is_sim_flags)

    for inst_idx in range(n_instances):
        is_sim = int(is_sim_flags[inst_idx])
        well_id = -1 if is_sim else int(rng.integers(0, n_wells))  # sim instances: no shared real well
        n_windows = windows_per_instance

        # Baseline signal: smooth-ish noise per channel.
        X = rng.normal(0, 1, size=(n_windows, n_channels, window_size)).astype("float32")

        # Randomly choose whether this instance ever reaches blockage.
        has_event = rng.random() < 0.6
        y = np.zeros(n_windows, dtype="int64")
        if has_event:
            onset = int(rng.integers(n_windows // 3, n_windows // 2))
            blocked_start = int(rng.integers(onset + n_windows // 6, n_windows - 2))
            y[onset:blocked_start] = 1
            y[blocked_start:] = 2
            # Plant a detectable ramp in two channels starting at onset.
            ramp_channels = rng.choice(n_channels, size=2, replace=False)
            ramp_len = n_windows - onset
            ramp = np.linspace(0, 3, ramp_len)[:, None]
            for ch in ramp_channels:
                X[onset:, ch, :] += ramp[:, :1]
            failure_time = float(blocked_start * window_size)
        else:
            failure_time = np.nan

        mask = (rng.random(size=X.shape) > 0.05).astype("uint8")  # ~5% missing, forces mask handling

        group = np.full(n_windows, well_id, dtype="int64")
        inst_id = np.full(n_windows, inst_idx, dtype="int64")
        t_end = (np.arange(n_windows) * window_size).astype("float64")
        is_sim_arr = np.full(n_windows, is_sim, dtype="uint8")
        normal_hours = float((y == 0).sum() * window_size / 3600.0)

        np.savez(
            out_dir / f"fake_{inst_idx:04d}.npz",
            X=X, mask=mask, y=y, group=group, inst_id=inst_id,
            t_end=t_end, is_sim=is_sim_arr,
            failure_time=failure_time, normal_hours=normal_hours,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/fake/")
    parser.add_argument("--n_wells", type=int, default=8)
    parser.add_argument("--n_instances", type=int, default=40)
    parser.add_argument("--n_channels", type=int, default=20)
    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--windows_per_instance", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    make_fake_dataset(
        args.out, args.n_wells, args.n_instances, args.n_channels,
        args.window_size, args.windows_per_instance, args.seed,
    )
    print(f"Wrote {args.n_instances} fake instances to {args.out}")
