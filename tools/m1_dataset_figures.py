"""
Member 1, W1.4 + W1.10 -- regenerates every dataset table and figure this
module owns, from the real 3W download. One entry point so run_all.sh can
call it and no number in the Dataset section is ever typed by hand.

    python -m tools.m1_dataset_figures --root data/3W/dataset

Writes:
    results/variable_availability.csv     per-variable present/frozen/keep table
    figures/transient_duration_hist.png   transient phase durations (14 events)
    figures/annotated_trace.png           one real instance, state zones shaded

Only REAL event-9 instances are loaded (57 of them, ~400 MB): the availability
question is about real instrumentation, and the simulated instances have no
missing or frozen sensors to measure.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.contract import EVENT_CODE
from src.data.availability import variable_availability_table
from src.data.inventory import ThreeWDataset
from src.data.stats import annotated_trace_figure, transient_duration_histogram

logger = logging.getLogger(__name__)


def _load_real_event_instances(root: str, event_code: int) -> list:
    ds = ThreeWDataset(root, event_code=event_code, include_normal=False, include_simulated=False)
    instances = [ds._load_one(fp, event_code) for fp in ds._iter_parquet_files(event_code)]
    return [inst for inst in instances if inst.source == "real"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/3W/dataset")
    ap.add_argument("--event", type=int, default=EVENT_CODE)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--figures-dir", default="figures")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.figures_dir).mkdir(parents=True, exist_ok=True)

    logger.info("loading real event-%d instances...", args.event)
    instances = _load_real_event_instances(args.root, args.event)
    logger.info("loaded %d real instances", len(instances))

    table = variable_availability_table(instances)
    out_csv = Path(args.results_dir) / "variable_availability.csv"
    table.to_csv(out_csv, index=False)
    print(table.to_string(index=False))
    print(f"\nwrote {out_csv}")

    hist_path = Path(args.figures_dir) / "transient_duration_hist.png"
    transient_duration_histogram(instances, str(hist_path))
    print(f"wrote {hist_path}")

    # The annotated trace uses the longest instance that reaches blockage, so
    # the figure shows all three phases (Normal -> Transient -> Established)
    # rather than a truncated one -- only 3 real instances qualify.
    with_blockage = [i for i in instances if i.has_established]
    pick = max(with_blockage or instances, key=lambda i: i.n_timesteps)
    trace_path = Path(args.figures_dir) / "annotated_trace.png"
    annotated_trace_figure(pick, str(trace_path))
    print(f"wrote {trace_path}  (instance: {pick.instance_id})")


if __name__ == "__main__":
    main()
