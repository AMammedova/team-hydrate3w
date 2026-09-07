"""Member 3 — a physics-shaped synthetic cache for BASELINE DESIGN ABLATIONS.

WHAT THIS IS NOT
----------------
Not a substitute for the real 3W cache and never a source of a reported
headline number. Every number in the paper's Results comes from
`data/cache` built by src/data/build_cache.py from the real download.

WHAT IT IS FOR
--------------
Choosing between feature-extractor designs needs an A/B with a known ground
truth and enough repetitions to separate a real effect from fold noise. Doing
that on the 14 real positive instances would burn the test folds, and
src/data/make_fake_data.py is too structureless to discriminate between
feature designs (it plants a ramp in two random channels of pure white
noise, so almost any feature set scores near-perfectly).

So this generator reproduces the *documented pathologies* of the real data
(DATA_FINDINGS.md) instead:

  * well-level instrumentation offsets and gains, so absolute sensor level
    is a confound and per-instance normalisation actually matters (§2);
  * positive wells and Normal-Operation wells are DISJOINT (§2);
  * whole-recording dead/frozen sensors in a subset of instances (§9) --
    the pathology the missing_policy ablation exists to answer;
  * irregular missingness plus contiguous dropout gaps (§5) -- what the
    slope_time ablation is about;
  * unlabeled (NaN class) spans, so nan_label_policy is exercised (§5);
  * AR(1) sensor noise rather than white noise, so trend features are not
    trivially separable.

The hydrate signature follows the physical mechanism in the 3W descriptor:
a restriction builds in the service line, so pressure UPSTREAM of the choke
climbs, and the temperature downstream of it falls as flow drops and
Joule-Thomson cooling sets in. P-ANULAR is deliberately left as a distractor
that carries no signal, so a feature-importance ranking that puts it on top
is visibly wrong.

Instances are written through the REAL pipeline -- WindowBuilder,
mask_missing, normalize_instance -- so the arrays a model sees here differ
from the real cache only in where the numbers came from.

Usage:
    python -m tools.make_baseline_benchmark --out data/benchmark --seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.contract import EVENT_CODE, TRANSIENT_RAW_LABEL
from src.data.inventory import InstanceRecord
from src.data.windowing import WindowBuilder, normal_seconds, onset_times

# The main-arm channel set (DATA_FINDINGS.md §9). Order is the cache order.
CHANNELS = ["P-MON-CKP", "P-JUS-CKGL", "T-TPT", "T-JUS-CKP", "P-ANULAR"]

# Response of each channel to a fully-developed hydrate restriction, in
# per-channel natural units. Sign is what matters: pressures upstream of the
# restriction rise, temperatures downstream fall, the annulus is unaffected.
SIGNATURE = {
    "P-MON-CKP": +1.00,     # upstream of production choke -- primary signature
    "P-JUS-CKGL": +0.55,    # gas-lift choke discharge follows the restriction
    "T-TPT": -0.35,         # mild cooling at the TPT
    "T-JUS-CKP": -0.80,     # downstream of the choke: JT cooling, clear drop
    "P-ANULAR": 0.00,       # distractor: annulus does not see the service line
}

# Well-level instrumentation: each well reads the same physics on a
# different offset and gain, which is exactly the confound that lets a model
# "solve" the task by recognising the well (DATA_FINDINGS.md §2).
_WELL_OFFSET_SPREAD = 8.0
_WELL_GAIN_SPREAD = 0.35


def _ar1(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    """AR(1) noise. White noise would make any trend feature trivially
    separable; real sensor noise is correlated, which is what makes the
    slope estimate a genuinely contested statistic."""
    e = rng.normal(0.0, sigma, size=n)
    out = np.empty(n, dtype=np.float64)
    out[0] = e[0]
    for i in range(1, n):
        out[i] = phi * out[i - 1] + e[i]
    return out


def _ramp(n: int, start: int, rise_len: int) -> np.ndarray:
    """0 before `start`, saturating to 1 over `rise_len` samples after it."""
    t = np.arange(n, dtype=np.float64)
    r = np.clip((t - start) / max(rise_len, 1), 0.0, 1.0)
    r[t < start] = 0.0
    return r


def _build_instance_frame(
    rng: np.random.Generator,
    *,
    n_raw: int,
    well_offsets: np.ndarray,
    well_gains: np.ndarray,
    is_positive: bool,
    reaches_blockage: bool,
    signal_strength: float,
    noise_sigma: float,
    dead_channels: list[int],
    base_missing: float,
    n_gaps: int,
    unlabeled_frac: float,
) -> pd.DataFrame:
    """One instance as a 1 Hz DataFrame with a `class` column, in the exact
    shape ThreeWDataset hands to WindowBuilder."""
    n_ch = len(CHANNELS)
    values = np.zeros((n_raw, n_ch), dtype=np.float64)

    raw_class = np.zeros(n_raw, dtype=np.float64)
    if is_positive:
        # Transient starts somewhere in the middle third; the ramp develops
        # over a stretch comparable to the real median transient (~3.4 h).
        onset = int(rng.integers(n_raw // 4, n_raw // 2))
        rise_len = int(rng.integers(n_raw // 6, n_raw // 3))
        raw_class[onset:] = TRANSIENT_RAW_LABEL
        if reaches_blockage:
            blockage = min(n_raw - 1, onset + rise_len + int(rng.integers(0, n_raw // 8)))
            raw_class[blockage:] = EVENT_CODE
        severity = _ramp(n_raw, onset, rise_len)
    else:
        severity = np.zeros(n_raw, dtype=np.float64)

    for c, name in enumerate(CHANNELS):
        drift = _ar1(rng, n_raw, phi=0.999, sigma=noise_sigma * 0.02)
        noise = _ar1(rng, n_raw, phi=0.95, sigma=noise_sigma)
        signal = SIGNATURE[name] * signal_strength * severity
        values[:, c] = well_offsets[c] + well_gains[c] * (signal + noise + drift)

    # --- missingness -------------------------------------------------------
    drop = rng.random((n_raw, n_ch)) < base_missing
    for _ in range(n_gaps):                       # contiguous dropout gaps
        c = int(rng.integers(0, n_ch))
        start = int(rng.integers(0, max(1, n_raw - 600)))
        drop[start : start + int(rng.integers(120, 600)), c] = True
    values[drop] = np.nan

    # Whole-recording dead sensors (DATA_FINDINGS.md §9). Modelled as NaN
    # rather than a frozen constant because VariableSelector.frozen_run_mask
    # converts a frozen run into NaN before windowing anyway.
    for c in dead_channels:
        values[:, c] = np.nan

    if unlabeled_frac > 0:                        # unlabeled spans (§5)
        n_unl = int(unlabeled_frac * n_raw)
        start = int(rng.integers(0, max(1, n_raw - n_unl)))
        raw_class[start : start + n_unl] = np.nan

    df = pd.DataFrame(values, columns=CHANNELS)
    df["class"] = raw_class
    df.index = pd.date_range("2018-01-01", periods=n_raw, freq="1s")
    return df


def make_benchmark(
    out_dir: str | Path,
    *,
    seed: int = 0,
    n_positive_wells: int = 7,
    n_normal_wells: int = 9,
    instances_per_positive_well: int = 2,
    instances_per_normal_well: int = 2,
    n_sim_instances: int = 6,
    n_raw: int = 9000,
    signal_strength: float = 1.0,
    noise_sigma: float = 0.55,
    base_missing: float = 0.06,
    dead_channel_prob: float = 0.25,
    dead_channel_mode: str = "independent",
    window_size: int = 60,
    stride: int = 5,
    decimate: int = 30,
) -> dict:
    """Write a contract-complete cache to `out_dir` and return a summary.

    `dead_channel_mode`:
      "independent"     -- a dead sensor is assigned without looking at the
                           label. This is the CONSERVATIVE setting and the
                           one the headline ablation uses: any win for the
                           NaN encoding here cannot come from the model
                           exploiting a correlation between deadness and the
                           answer, only from not being lied to about a dead
                           sensor's level.
      "event_correlated"-- dead sensors concentrated in positive instances,
                           which is what the real data actually looks like
                           (§9). Reported as a second arm so the difference
                           between the two is visible rather than assumed.
    """
    if dead_channel_mode not in ("independent", "event_correlated"):
        raise ValueError(f"unknown dead_channel_mode {dead_channel_mode!r}")

    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.npz"):
        stale.unlink()

    builder = WindowBuilder(
        window_size=window_size,
        stride=stride,
        decimate=decimate,
        label_rule="most_severe",
        nan_label_policy="drop",
    )

    # Positive and Normal wells are disjoint populations (DATA_FINDINGS §2).
    pos_wells = [f"WELL-{i:05d}" for i in range(1, n_positive_wells + 1)]
    nrm_wells = [f"WELL-{i:05d}" for i in range(50, 50 + n_normal_wells)]
    all_wells = pos_wells + nrm_wells
    group_map = {w: i for i, w in enumerate(all_wells)}

    well_offsets = {
        w: rng.normal(0.0, _WELL_OFFSET_SPREAD, size=len(CHANNELS)) for w in all_wells
    }
    well_gains = {
        w: 1.0 + rng.normal(0.0, _WELL_GAIN_SPREAD, size=len(CHANNELS)) for w in all_wells
    }

    plan: list[tuple[str, str, bool, bool]] = []   # (inst_id, well, positive, blockage)
    for w in pos_wells:
        for k in range(instances_per_positive_well):
            # ~3 of 14 real positive instances reach a blockage (§9).
            plan.append((f"{w}_pos{k}", w, True, rng.random() < 0.22))
    for w in nrm_wells:
        for k in range(instances_per_normal_well):
            plan.append((f"{w}_nrm{k}", w, False, False))
    for k in range(n_sim_instances):
        plan.append((f"SIM-{k:03d}_sim", f"SIM-{k:03d}", True, rng.random() < 0.3))

    n_ch = len(CHANNELS)
    summary = {
        "instances_written": 0,
        "instances_empty": 0,
        "windows": 0,
        "dead_channel_instances": 0,
        "positive_instances": 0,
    }
    inst_id_of = {inst: i for i, (inst, *_rest) in enumerate(plan)}

    for inst_id, well, is_positive, blockage in plan:
        is_sim = well.startswith("SIM-")
        if dead_channel_mode == "independent":
            p_dead = dead_channel_prob
        else:
            p_dead = dead_channel_prob * (2.0 if is_positive else 0.25)
        dead: list[int] = []
        if rng.random() < p_dead:
            # Never kill every channel -- an instance with no sensors at all
            # is dropped by min_valid_frac and teaches nothing.
            dead = [int(rng.integers(0, n_ch))]
            if rng.random() < 0.3:
                extra = int(rng.integers(0, n_ch))
                if extra not in dead:
                    dead.append(extra)
        if dead:
            summary["dead_channel_instances"] += 1

        df = _build_instance_frame(
            rng,
            n_raw=n_raw,
            well_offsets=well_offsets[well] if not is_sim else rng.normal(0, 2, n_ch),
            well_gains=well_gains[well] if not is_sim else np.ones(n_ch),
            is_positive=is_positive,
            reaches_blockage=blockage,
            # Simulated instances are cleaner than reality -- that is the
            # whole reason real_plus_sim is a question and not a given.
            signal_strength=signal_strength * (1.6 if is_sim else 1.0),
            noise_sigma=noise_sigma * (0.6 if is_sim else 1.0),
            dead_channels=[] if is_sim else dead,
            base_missing=base_missing * (0.2 if is_sim else 1.0),
            n_gaps=0 if is_sim else int(rng.integers(0, 4)),
            unlabeled_frac=0.0 if is_sim else float(rng.uniform(0.0, 0.18)),
        )

        inst = InstanceRecord(
            instance_id=inst_id,
            well_id=well,
            source="simulated" if is_sim else "real",
            event_code=EVENT_CODE,
            filepath=Path(inst_id),
            df=df,
        )

        X, y, _wells, t_end = builder.build_windows(inst, normalize="warmup")
        if len(X) == 0:
            summary["instances_empty"] += 1
            continue
        mask = builder.window_masks(inst)
        onsets = onset_times(inst)

        np.savez_compressed(
            out / f"{inst_id}.npz",
            X=X.astype("float32"),
            mask=mask.astype("uint8"),
            y=y.astype("int64"),
            group=np.full(len(y), group_map.get(well, 900 + inst_id_of[inst_id]), dtype="int64"),
            inst_id=np.full(len(y), inst_id_of[inst_id], dtype="int64"),
            t_end=t_end.astype("float64"),
            is_sim=np.full(len(y), int(is_sim), dtype="uint8"),
            failure_time=np.float64(onsets["transient_onset"]),
            blockage_time=np.float64(onsets["blockage_onset"]),
            normal_hours=np.float64(normal_seconds(inst) / 3600.0),
        )
        summary["instances_written"] += 1
        summary["windows"] += int(len(y))
        summary["positive_instances"] += int(bool((y != 0).any()))

    sidecar = {
        "channels": CHANNELS,
        "group_map": group_map,
        "window_size": window_size,
        "stride": stride,
        "decimate": decimate,
        "label_rule": "most_severe",
        "nan_label_policy": "drop",
        "SYNTHETIC": True,
        "generator": "tools/make_baseline_benchmark.py",
        "dead_channel_mode": dead_channel_mode,
        "seed": seed,
        "signature": SIGNATURE,
        "warning": "SYNTHETIC BENCHMARK -- not the 3W dataset. Never report as a result.",
    }
    (out / "cache_config.json").write_text(json.dumps(sidecar, indent=2), encoding="utf8")
    summary["out_dir"] = str(out)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/benchmark")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-raw", type=int, default=9000)
    ap.add_argument("--signal-strength", type=float, default=1.0)
    ap.add_argument("--noise-sigma", type=float, default=0.55)
    ap.add_argument("--dead-channel-prob", type=float, default=0.25)
    ap.add_argument(
        "--dead-channel-mode", default="independent",
        choices=["independent", "event_correlated"],
    )
    args = ap.parse_args()

    summary = make_benchmark(
        args.out,
        seed=args.seed,
        n_raw=args.n_raw,
        signal_strength=args.signal_strength,
        noise_sigma=args.noise_sigma,
        dead_channel_prob=args.dead_channel_prob,
        dead_channel_mode=args.dead_channel_mode,
    )
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
