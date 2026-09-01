"""
Member 1, W1.9 — raw parquet -> cached windowed .npz.

One .npz per source instance, matching the team data contract (§0.1) and
make_fake_data.py's output key-for-key, so Members 2-5 change one path and
nothing else when the real cache lands.

    python -m src.data.build_cache --root data/3W/dataset --out data/cache

Every run also writes `<out>/cache_config.json`: the exact config used, the
frozen channel list with its measured missing fractions, the well -> group-id
map, and per-instance window/drop counts. Deterministic given the same config,
and nothing downstream has to guess how the cache was made.

WHAT THE `failure_time` FIELD MEANS HERE -- read this before using it.
The project statement (DL8.3) defines lead time against ESTABLISHED-blockage
onset. Only 3 of the 57 real Event-9 instances ever reach blockage
(DATA_FINDINGS.md §3), so on real data that definition yields an n=3 metric
that is undefined in most CV folds. Team decision:

    failure_time  := transient onset   (14 real instances, 7 wells)
    blockage_time := established onset (3 real instances, 3 wells) -- carried
                     alongside so the statement's original definition can still
                     be reported as a secondary row with its n stated openly.

THE ONE LEAKAGE CAVEAT IN THIS MODULE, STATED PLAINLY.
DL2.1 says VariableSelector must be fit on training folds only. A cache, by
construction, is built once for every instance -- before the folds exist. What
is fit here is the *channel list* and the per-channel fallback means, which
W1.5 explicitly freezes at dataset level. Two things keep that honest:

  * `--fit-wells` restricts the fit to a named subset of wells, which is the
    strict per-fold path (build one cache per fold) if the team wants it.
  * the sidecar records the per-channel missing fractions that produced the
    list, so M2 can verify the same channels survive fold by fold. If they do
    -- the expected outcome, since missingness is a sensor property, not a
    fold property -- the global fit provably changed nothing and the report
    says so with numbers instead of a promise.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.contract import EVENT_CODE, NORMAL
from src.data.inventory import ThreeWDataset, _infer_source_and_well
from src.data.windowing import (
    VariableSelector,
    WindowBuilder,
    normal_seconds,
    onset_times,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: dict = {
    "event_code": EVENT_CODE,
    # Windowing -- see WindowBuilder's docstring for units. 60 samples at
    # decimate=30 is a 30-minute window, chosen against the measured transient
    # durations (median 12,332 s), not the old unvalidated 60-second default.
    "window_size": 60,
    "stride": 5,
    "decimate": 30,
    "label_rule": "most_severe",
    "min_valid_frac": 0.5,
    "nan_label_policy": "drop",
    # Missingness / frozen sensors
    "max_missing_frac": 0.5,
    "frozen_run_seconds": 60,
    # Per-instance normalisation. Mandatory here, not cosmetic: the hydrate and
    # Normal wells are disjoint in this dataset, so raw sensor level alone
    # separates the classes (DATA_FINDINGS.md §2).
    "normalize": "warmup",
    "include_simulated": True,
    "include_normal": True,
}


def _well_ids_by_filename(ds: ThreeWDataset, event_code: int) -> dict[str, str]:
    """instance_id -> well_id, from filenames only (no parquet read). Used to
    build a stable well -> group-id map before any data is loaded."""
    out: dict[str, str] = {}
    for label in (NORMAL, event_code):
        for fp in ds._iter_parquet_files(label):
            _source, well = _infer_source_and_well(fp.stem)
            out[fp.stem] = well
    return out


def _group_map(well_by_instance: dict[str, str]) -> dict[str, int]:
    """
    well_id -> int64 group id, for the `group` array the contract requires.

    Real wells get 0, 1, 2, ... in sorted order. Simulated/drawn instances get
    a distinct NEGATIVE id each (-1, -2, ...), never a shared one: they do not
    belong to a real physical well, and letting several of them collapse into
    one group would tell GroupedKFoldSplitter they are the same well and can be
    split apart freely (DL1.1 / DL3.2).
    """
    real = sorted({w for w in well_by_instance.values() if not w.startswith(("SIM-", "DRAWN-"))})
    mapping = {w: i for i, w in enumerate(real)}
    pseudo = sorted({w for w in well_by_instance.values() if w.startswith(("SIM-", "DRAWN-"))})
    for i, w in enumerate(pseudo, start=1):
        mapping[w] = -i
    return mapping


def _iter_instances(ds: ThreeWDataset, event_code: int, include_normal: bool, limit: int | None):
    """Stream InstanceRecords one at a time. load_instances() would hold every
    DataFrame at once -- ~2.7 GB for the 594 Normal instances alone."""
    n = 0
    labels = [event_code] + ([NORMAL] if include_normal else [])
    for label in labels:
        for fp in ds._iter_parquet_files(label):
            yield ds._load_one(fp, event_code)
            n += 1
            if limit is not None and n >= limit:
                return


def build_cache(root: str, out_dir: str, config: dict | None = None) -> pd.DataFrame:
    """
    Builds the windowed cache. Returns the per-instance summary DataFrame that
    is also written to `<out>/cache_summary.csv`.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    event_code = int(cfg["event_code"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = ThreeWDataset(
        root,
        event_code=event_code,
        include_simulated=bool(cfg["include_simulated"]),
        include_normal=bool(cfg["include_normal"]),
    )

    well_by_instance = _well_ids_by_filename(ds, event_code)
    group_of = _group_map(well_by_instance)
    inst_id_of = {stem: i for i, stem in enumerate(sorted(well_by_instance))}

    fit_wells = cfg.get("fit_wells")
    selector = VariableSelector(
        max_missing_frac=float(cfg["max_missing_frac"]),
        frozen_run_seconds=int(cfg["frozen_run_seconds"]),
    )

    logger.info("pass 1/2: fitting the channel list (streamed)")
    def _fit_stream():
        for inst in _iter_instances(ds, event_code, bool(cfg["include_normal"]), cfg.get("limit")):
            if fit_wells and inst.well_id not in fit_wells:
                continue
            yield inst
    selector.fit(_fit_stream())
    kept = selector.kept_columns
    logger.info("kept %d channels: %s", len(kept), kept)

    builder = WindowBuilder(
        window_size=int(cfg["window_size"]),
        stride=int(cfg["stride"]),
        label_rule=str(cfg["label_rule"]),
        min_valid_frac=float(cfg["min_valid_frac"]),
        decimate=int(cfg["decimate"]),
        nan_label_policy=str(cfg["nan_label_policy"]),
    )
    channel_means = selector.means_for(kept)

    logger.info("pass 2/2: windowing and writing the cache")
    rows = []
    for inst in _iter_instances(ds, event_code, bool(cfg["include_normal"]), cfg.get("limit")):
        inst.df = selector.transform(inst.df)
        X, y, _wells, t_end = builder.build_windows(
            inst, channel_means=channel_means, normalize=str(cfg["normalize"])
        )
        mask = builder.window_masks(inst, channel_means=channel_means)
        if len(X) != len(mask):
            raise AssertionError(
                f"{inst.instance_id}: build_windows kept {len(X)} windows but "
                f"window_masks kept {len(mask)} -- the drop rules disagree"
            )

        n_win = len(X)
        onsets = onset_times(inst)
        n_possible = max(0, (len(inst.df) // builder.decimate) - builder.window_size + 1)
        n_possible = len(range(0, n_possible, builder.stride)) if n_possible > 0 else 0

        if n_win:
            np.savez_compressed(
                out / f"{inst.instance_id}.npz",
                X=X,
                mask=mask,
                y=y,
                group=np.full(n_win, group_of[inst.well_id], dtype="int64"),
                inst_id=np.full(n_win, inst_id_of[inst.instance_id], dtype="int64"),
                t_end=t_end.astype("float64"),
                is_sim=np.full(n_win, int(inst.source != "real"), dtype="uint8"),
                # Team decision (see module docstring): failure_time is the
                # TRANSIENT onset; blockage_time keeps the statement's original
                # definition available as a secondary metric.
                failure_time=np.float64(onsets["transient_onset"]),
                blockage_time=np.float64(onsets["blockage_onset"]),
                normal_hours=np.float64(normal_seconds(inst) / 3600.0),
            )
        else:
            logger.warning(
                "%s produced 0 usable windows (all dropped by nan_label_policy=%s "
                "or min_valid_frac=%.2f)", inst.instance_id, cfg["nan_label_policy"],
                cfg["min_valid_frac"],
            )

        rows.append(
            {
                "instance_id": inst.instance_id,
                "well_id": inst.well_id,
                "group": group_of[inst.well_id],
                "source": inst.source,
                "n_timesteps": inst.n_timesteps,
                "n_windows": n_win,
                "n_windows_dropped": max(0, n_possible - n_win),
                "n_normal": int((y == 0).sum()) if n_win else 0,
                "n_transient": int((y == 1).sum()) if n_win else 0,
                "n_established": int((y == 2).sum()) if n_win else 0,
                "transient_onset_s": onsets["transient_onset"],
                "blockage_onset_s": onsets["blockage_onset"],
                "normal_hours": normal_seconds(inst) / 3600.0,
            }
        )
        del inst

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "cache_summary.csv", index=False)

    sidecar = {
        "config": {k: v for k, v in cfg.items() if k != "fit_wells"},
        "fit_wells": sorted(fit_wells) if fit_wells else None,
        "window_seconds": builder.window_seconds,
        "stride_seconds": builder.stride * builder.decimate,
        "kept_channels": kept,
        "channel_missing_frac": {c: round(selector.missing_frac_[c], 6) for c in kept},
        "dropped_channels": {
            c: round(f, 6)
            for c, f in sorted(selector.missing_frac_.items())
            if c not in set(kept)
        },
        "group_map": group_of,
        "n_instances": int(len(summary)),
        "n_windows_total": int(summary["n_windows"].sum()),
        "n_windows_dropped_total": int(summary["n_windows_dropped"].sum()),
    }
    (out / "cache_config.json").write_text(json.dumps(sidecar, indent=2), encoding="utf8")

    logger.info(
        "wrote %d instances, %d windows (%d dropped) to %s",
        len(summary), sidecar["n_windows_total"], sidecar["n_windows_dropped_total"], out,
    )
    return summary


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Build the windowed .npz cache (W1.9).")
    parser.add_argument("--root", default="data/3W/dataset")
    parser.add_argument("--out", default="data/cache")
    parser.add_argument("--event", type=int, default=EVENT_CODE)
    parser.add_argument("--window-size", type=int, default=DEFAULT_CONFIG["window_size"],
                        help="decimated samples per window (not seconds)")
    parser.add_argument("--stride", type=int, default=DEFAULT_CONFIG["stride"])
    parser.add_argument("--decimate", type=int, default=DEFAULT_CONFIG["decimate"],
                        help="raw 1 Hz samples averaged into one decimated sample")
    parser.add_argument("--label-rule", default=DEFAULT_CONFIG["label_rule"],
                        choices=["most_severe", "final_timestep", "majority"])
    parser.add_argument("--nan-label-policy", default=DEFAULT_CONFIG["nan_label_policy"],
                        choices=["drop", "normal"])
    parser.add_argument("--normalize", default=DEFAULT_CONFIG["normalize"],
                        choices=["warmup", "instance_robust", "none"])
    parser.add_argument("--min-valid-frac", type=float, default=DEFAULT_CONFIG["min_valid_frac"])
    parser.add_argument("--max-missing-frac", type=float, default=DEFAULT_CONFIG["max_missing_frac"])
    parser.add_argument("--fit-wells", default=None,
                        help="comma-separated well ids the channel list may be fit on "
                             "(the strict per-fold path; default: every instance)")
    parser.add_argument("--no-normal", action="store_true", help="skip class-0 instances")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after N instances -- for a fast smoke test")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = {
        "event_code": args.event,
        "window_size": args.window_size,
        "stride": args.stride,
        "decimate": args.decimate,
        "label_rule": args.label_rule,
        "nan_label_policy": args.nan_label_policy,
        "normalize": args.normalize,
        "min_valid_frac": args.min_valid_frac,
        "max_missing_frac": args.max_missing_frac,
        "include_normal": not args.no_normal,
        "fit_wells": set(args.fit_wells.split(",")) if args.fit_wells else None,
        "limit": args.limit,
    }
    summary = build_cache(args.root, args.out, cfg)
    print(summary.groupby("source")[["n_windows", "n_transient", "n_established"]].sum().to_string())
    print(f"\ntotal windows: {int(summary['n_windows'].sum()):,}")
    print(f"instances with 0 usable windows: {int((summary['n_windows'] == 0).sum())}")


if __name__ == "__main__":
    _cli()
