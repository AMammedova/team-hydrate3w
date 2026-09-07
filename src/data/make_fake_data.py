"""
Member 1, W1.1 — fake data generator.

Produces synthetic arrays matching the current cached-window data contract,
so downstream splitting, baseline, deep-model, and evaluation code can be
developed and smoke-tested before the real 3W cache is available.

This is development/test data only. Numerical performance on this dataset
must never be reported as a project result.
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
    Write one .npz per fake source instance.

    Current cache contract
    ----------------------
    Per-window arrays:
        X:        float32 [N, C, W]
        mask:     uint8   [N, C, W]
        y:        int64   [N]
                  0 = Normal
                  1 = Transient
                  2 = Established
        group:    int64   [N]
        inst_id:  int64   [N]
        t_end:    float64 [N]
        is_sim:   uint8   [N]

    Per-instance scalars:
        failure_time: float64
            Transient onset time. NaN when no event exists.

        blockage_time: float64
            Established/blockage onset time. NaN when no blockage exists.

        normal_hours: float64
            Amount of Normal operating time represented by the instance.

    A detectable synthetic ramp is planted in two channels when an event
    begins. Roughly 5% of values are marked unavailable so downstream code
    must exercise the explicit mask path.
    """
    rng = np.random.default_rng(seed)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_sim = max(1, n_instances // 5)
    is_sim_flags = np.array(
        [0] * (n_instances - n_sim) + [1] * n_sim,
        dtype="uint8",
    )
    rng.shuffle(is_sim_flags)

    for inst_idx in range(n_instances):
        is_sim = int(is_sim_flags[inst_idx])

        # Real fake instances reuse a small set of well IDs so grouped
        # splitting can be exercised.
        #
        # Simulated instances receive their own negative pseudo-group IDs;
        # they must never masquerade as real wells.
        if is_sim:
            well_id = -(inst_idx + 1)
        else:
            well_id = int(rng.integers(0, n_wells))

        n_windows = windows_per_instance

        # --------------------------------------------------------------
        # Synthetic sensor values
        # --------------------------------------------------------------
        X = rng.normal(
            0.0,
            1.0,
            size=(n_windows, n_channels, window_size),
        ).astype("float32")

        y = np.zeros(n_windows, dtype="int64")

        failure_time = np.nan
        blockage_time = np.nan

        # --------------------------------------------------------------
        # Synthetic event
        # --------------------------------------------------------------
        has_event = rng.random() < 0.6

        if has_event:
            onset = int(
                rng.integers(
                    n_windows // 3,
                    n_windows // 2,
                )
            )

            blocked_start = int(
                rng.integers(
                    onset + max(1, n_windows // 6),
                    n_windows - 1,
                )
            )

            y[onset:blocked_start] = 1
            y[blocked_start:] = 2

            # Plant a detectable temporal pattern.
            n_ramp_channels = min(2, n_channels)
            ramp_channels = rng.choice(
                n_channels,
                size=n_ramp_channels,
                replace=False,
            )

            ramp_len = n_windows - onset
            ramp = np.linspace(
                0.0,
                3.0,
                ramp_len,
                dtype="float32",
            )

            for ch in ramp_channels:
                X[onset:, ch, :] += ramp[:, None]

            # Current project semantics:
            # failure_time  = Transient onset
            # blockage_time = Established onset
            failure_time = float(onset * window_size)
            blockage_time = float(blocked_start * window_size)

        # --------------------------------------------------------------
        # Availability mask
        # --------------------------------------------------------------
        mask = (
            rng.random(size=X.shape) > 0.05
        ).astype("uint8")

        group = np.full(
            n_windows,
            well_id,
            dtype="int64",
        )

        inst_id = np.full(
            n_windows,
            inst_idx,
            dtype="int64",
        )

        # Synthetic endpoint timestamps.
        t_end = (
            np.arange(n_windows, dtype="float64")
            * float(window_size)
        )

        is_sim_arr = np.full(
            n_windows,
            is_sim,
            dtype="uint8",
        )

        normal_hours = float(
            (y == 0).sum()
            * window_size
            / 3600.0
        )

        np.savez(
            out_dir / f"fake_{inst_idx:04d}.npz",
            X=X,
            mask=mask,
            y=y,
            group=group,
            inst_id=inst_id,
            t_end=t_end,
            is_sim=is_sim_arr,
            failure_time=np.float64(failure_time),
            blockage_time=np.float64(blockage_time),
            normal_hours=np.float64(normal_hours),
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
        out_dir=args.out,
        n_wells=args.n_wells,
        n_instances=args.n_instances,
        n_channels=args.n_channels,
        window_size=args.window_size,
        windows_per_instance=args.windows_per_instance,
        seed=args.seed,
    )

    print(
        f"Wrote {args.n_instances} fake instances to {args.out}"
    )