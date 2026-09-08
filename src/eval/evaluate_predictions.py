"""
Module 8 — Offline threshold selection and test-fold evaluation. Member 5.

This module is the critical bridge connecting raw model predictions
(saved by tools/train_xgb.py and tools/train_deep_models.py as
results/model_outputs/<stem>_val.npz and <stem>_test.npz) to the final
metrics in results/results.csv and the paper figures.

S3 Freeze Protocol:
1. Load validation predictions (_val.npz).
2. Extract Normal-operation validation instances.
3. Select threshold via thresholds.select_threshold() matching target_far
   (default: 1 false alarm / 100 operating-hours) using the data-adaptive grid.
4. FREEZE the threshold, smooth_window, and min_duration.
5. Apply the frozen policy UNCHANGED to test predictions (_test.npz).
6. Compute test metrics:
   - event_recall: fraction of real hydrate events flagged before failure_time.
   - lead_time: median time (seconds) from first alarm to failure_time.
   - far: false alarms per operating hour on test normal instances.
   - pr_auc: binary PR-AUC on test windows.
   - ece: Expected Calibration Error on test windows.
7. Append rows to results/results.csv adhering strictly to RESULTS_COLUMNS.
8. Generate lead_time_vs_far.png and per_well_lead_time.png in figures/.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from src.contract import RESULTS_COLUMNS
from src.eval.alarm import alarm_times, first_alarm_before, lead_time
from src.eval.metrics import (
    event_recall,
    expected_calibration_error,
    false_alarms_per_operating_hour,
    positive_score,
    pr_auc,
)
from src.eval.plots import (
    plot_lead_time_vs_false_alarm_rate,
    plot_per_well_lead_time_box,
)
from src.eval.thresholds import select_threshold, select_threshold_curve

logger = logging.getLogger(__name__)


def parse_stem(stem: str) -> dict[str, Any]:
    """Parse stem like 'xgboost_real_only_fold0_seed42' into components."""
    match = re.match(
        r"^(?P<model>.+)_(?P<condition>real_only|real_plus_sim)_fold(?P<fold>\d+)_seed(?P<seed>\d+)$",
        stem,
    )
    if not match:
        raise ValueError(
            f"Cannot parse output stem '{stem}'. "
            f"Expected format: <model>_<condition>_fold<K>_seed<S>"
        )
    d = match.groupdict()
    return {
        "model": d["model"],
        "condition": d["condition"],
        "fold": int(d["fold"]),
        "seed": int(d["seed"]),
    }


def _extract_positive_score(data: np.lib.npyio.NpzFile) -> np.ndarray:
    """Extract positive score array, preferring calibrated scores if present."""
    if "pos_score_calibrated" in data:
        return np.asarray(data["pos_score_calibrated"], dtype=np.float64)
    if "probs_calibrated" in data:
        return np.asarray(positive_score(data["probs_calibrated"]), dtype=np.float64)
    if "pos_score_raw" in data:
        return np.asarray(data["pos_score_raw"], dtype=np.float64)
    if "probs" in data:
        return np.asarray(positive_score(data["probs"]), dtype=np.float64)
    raise KeyError("No suitable probability array found in output .npz")


def evaluate_single_run(
    val_path: Path,
    test_path: Path,
    *,
    smooth_window: int = 5,
    min_duration: float = 0.0,
    target_far: float = 1 / 100,
) -> tuple[list[dict[str, Any]], pd.DataFrame, list[dict[str, Any]]]:
    """Select threshold on validation and evaluate frozen policy on test.

    Returns
    -------
    rows : list of dict
        Metric rows matching RESULTS_COLUMNS.
    curve_df : pd.DataFrame
        Threshold sweep curve on validation data.
    per_well_rows : list of dict
        Per-event lead times with well_id, model, condition, fold, seed.
    """
    stem = val_path.stem[:-4] if val_path.stem.endswith("_val") else val_path.stem
    meta = parse_stem(stem)

    with np.load(val_path) as val_data, np.load(test_path) as test_data:
        val_score = _extract_positive_score(val_data)
        val_inst_id = np.asarray(val_data["inst_id"])
        val_y = np.asarray(val_data["y_true"], dtype=np.int64)
        val_t_end = np.asarray(val_data["t_end"], dtype=np.float64)
        val_normal_hours_arr = np.asarray(val_data["normal_hours_value"], dtype=np.float64)
        val_normal_hours = float(np.sum(val_normal_hours_arr))

        # 1. Build validation Normal instances
        # An instance is strictly Normal if all its windows have y_true == 0
        y_val_proba: dict[int, np.ndarray] = {}
        y_val_time: dict[int, np.ndarray] = {}

        unique_val_insts = np.unique(val_inst_id)
        for u in unique_val_insts:
            inst_mask = val_inst_id == u
            if np.all(val_y[inst_mask] == 0):
                order = np.argsort(val_t_end[inst_mask])
                y_val_proba[int(u)] = val_score[inst_mask][order]
                y_val_time[int(u)] = val_t_end[inst_mask][order]

        # 2. Threshold selection on validation Normal instances ONLY
        chosen_thresh = select_threshold(
            y_val_proba=y_val_proba,
            y_val_time=y_val_time,
            val_normal_hours=val_normal_hours,
            smooth_window=smooth_window,
            min_duration=min_duration,
            target_far=target_far,
        )

        curve_df = select_threshold_curve(
            y_val_proba=y_val_proba,
            y_val_time=y_val_time,
            val_normal_hours=val_normal_hours,
            smooth_window=smooth_window,
            min_duration=min_duration,
        )

        # 3. Evaluate on test set under frozen threshold
        test_score = _extract_positive_score(test_data)
        test_inst_id = np.asarray(test_data["inst_id"])
        test_y = np.asarray(test_data["y_true"], dtype=np.int64)
        test_t_end = np.asarray(test_data["t_end"], dtype=np.float64)
        test_group = np.asarray(test_data["group"], dtype=np.int64)
        test_is_sim = np.asarray(test_data["is_sim"], dtype=np.uint8)
        test_failure_time = np.asarray(test_data["failure_time"], dtype=np.float64)
        test_normal_hours_arr = np.asarray(test_data["normal_hours_value"], dtype=np.float64)
        test_normal_hours = float(np.sum(test_normal_hours_arr))

        # False alarms on Normal test instances
        total_test_false_alarms = 0
        unique_test_insts = np.unique(test_inst_id)
        for u in unique_test_insts:
            inst_mask = test_inst_id == u
            if np.all(test_y[inst_mask] == 0):
                order = np.argsort(test_t_end[inst_mask])
                p_inst = test_score[inst_mask][order]
                t_inst = test_t_end[inst_mask][order]
                alarms = alarm_times(
                    proba=p_inst,
                    t=t_inst,
                    smooth_window=smooth_window,
                    threshold=chosen_thresh,
                    min_duration=min_duration,
                )
                total_test_false_alarms += len(alarms)

        test_far = false_alarms_per_operating_hour(
            total_test_false_alarms, test_normal_hours
        )

        # Event recall and lead times on positive real test instances
        events_flagged: list[bool] = []
        lead_times: list[float] = []
        per_well_rows: list[dict[str, Any]] = []

        for u in unique_test_insts:
            inst_mask = test_inst_id == u
            # Real positive instance: not simulated and has transient/established labels
            if np.any(test_is_sim[inst_mask] == 0) and np.any(test_y[inst_mask] != 0):
                order = np.argsort(test_t_end[inst_mask])
                p_inst = test_score[inst_mask][order]
                t_inst = test_t_end[inst_mask][order]
                inst_fail = float(test_failure_time[inst_mask][0])
                well_id = int(test_group[inst_mask][0])

                alarms = alarm_times(
                    proba=p_inst,
                    t=t_inst,
                    smooth_window=smooth_window,
                    threshold=chosen_thresh,
                    min_duration=min_duration,
                )
                first_alarm = first_alarm_before(alarms, inst_fail)
                lt = lead_time(inst_fail, first_alarm)

                flagged = first_alarm is not None
                events_flagged.append(flagged)
                if lt is not None:
                    lead_times.append(float(lt))

                per_well_rows.append(
                    {
                        "well_id": well_id,
                        "instance_id": int(u),
                        "lead_time": float(lt) if lt is not None else np.nan,
                        "flagged": flagged,
                        "model": meta["model"],
                        "condition": meta["condition"],
                        "fold": meta["fold"],
                        "seed": meta["seed"],
                    }
                )

        ev_recall = (
            event_recall(events_flagged, len(events_flagged))
            if events_flagged
            else float("nan")
        )
        med_lead_time = float(np.median(lead_times)) if lead_times else float("nan")

        # Binary window metrics on test set
        test_y_bin = (test_y != 0).astype(np.int64)
        if len(np.unique(test_y_bin)) > 1:
            test_pr_val = pr_auc(test_y_bin, test_score)
            test_ece_val = expected_calibration_error(test_y_bin, test_score)
        else:
            test_pr_val = float("nan")
            test_ece_val = float("nan")

        # Assemble rows matching RESULTS_COLUMNS
        def _make_row(name: str, val: float) -> dict[str, Any]:
            return {
                "model": meta["model"],
                "fold": meta["fold"],
                "seed": meta["seed"],
                "condition": meta["condition"],
                "metric_name": name,
                "value": float(val),
            }

        rows = [
            _make_row("pr_auc", test_pr_val),
            _make_row("event_recall", ev_recall),
            _make_row("far", test_far),
            _make_row("lead_time", med_lead_time),
            _make_row("ece", test_ece_val),
            _make_row("threshold", chosen_thresh),
            _make_row("n_test_events", float(len(events_flagged))),
            _make_row("n_test_flagged", float(sum(events_flagged))),
        ]

        return rows, curve_df, per_well_rows


def evaluate_all(
    outputs_dir: Path | str = "results/model_outputs",
    out_results: Path | str | None = "results/results.csv",
    figures_dir: Path | str | None = "figures",
    *,
    smooth_window: int = 5,
    min_duration: float = 0.0,
    target_far: float = 1 / 100,
    append: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find all matching _val.npz / _test.npz pairs, evaluate, and write outputs."""
    outputs_path = Path(outputs_dir)
    if not outputs_path.exists():
        logger.warning("Outputs directory does not exist: %s", outputs_path)
        return pd.DataFrame(columns=RESULTS_COLUMNS), pd.DataFrame()

    val_files = sorted(outputs_path.glob("*_val.npz"))
    if not val_files:
        logger.warning("No *_val.npz files found in %s", outputs_path)
        return pd.DataFrame(columns=RESULTS_COLUMNS), pd.DataFrame()

    all_metric_rows: list[dict[str, Any]] = []
    all_per_well_rows: list[dict[str, Any]] = []
    curves_by_model: dict[str, pd.DataFrame] = {}

    for val_file in val_files:
        stem = val_file.name[:-8]  # strip '_val.npz'
        test_file = outputs_path / f"{stem}_test.npz"
        if not test_file.exists():
            logger.warning(
                "Matching test file %s not found for %s, skipping",
                test_file,
                val_file,
            )
            continue

        try:
            rows, curve_df, per_well_rows = evaluate_single_run(
                val_file,
                test_file,
                smooth_window=smooth_window,
                min_duration=min_duration,
                target_far=target_far,
            )
            all_metric_rows.extend(rows)
            all_per_well_rows.extend(per_well_rows)

            meta = parse_stem(stem)
            key = f"{meta['model']}_{meta['condition']}_f{meta['fold']}"
            curves_by_model[key] = curve_df
            logger.info("Successfully evaluated: %s", stem)
        except Exception as e:
            logger.error("Error evaluating %s: %s", stem, e, exc_info=True)

    metrics_df = pd.DataFrame(all_metric_rows, columns=RESULTS_COLUMNS)
    per_well_df = pd.DataFrame(all_per_well_rows)

    # Write / update results.csv
    if out_results is not None and not metrics_df.empty:
        out_csv_path = Path(out_results)
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        if out_csv_path.exists() and append:
            existing = pd.read_csv(out_csv_path)
            # Remove any overlapping entries to avoid duplication
            combined = pd.concat([existing, metrics_df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["model", "fold", "seed", "condition", "metric_name"],
                keep="last",
            )
            combined.to_csv(out_csv_path, index=False)
            logger.info("Updated %s with %d rows", out_csv_path, len(combined))
        else:
            metrics_df.to_csv(out_csv_path, index=False)
            logger.info("Wrote %d rows to %s", len(metrics_df), out_csv_path)

    # Generate figures if directory is provided
    if figures_dir is not None:
        fig_path = Path(figures_dir)
        fig_path.mkdir(parents=True, exist_ok=True)

        if curves_by_model:
            far_curve_out = fig_path / "lead_time_vs_far.png"
            plot_lead_time_vs_false_alarm_rate(
                curves=curves_by_model,
                out_path=str(far_curve_out),
                target_far=target_far,
            )
            logger.info("Saved %s", far_curve_out)

        if not per_well_df.empty:
            box_out = fig_path / "per_well_lead_time.png"
            plot_per_well_lead_time_box(
                fold_metrics=per_well_df,
                out_path=str(box_out),
            )
            logger.info("Saved %s", box_out)

    return metrics_df, per_well_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline threshold selection and test-fold evaluation (M5)"
    )
    parser.add_argument(
        "--outputs-dir",
        default="results/model_outputs",
        help="Directory containing *_val.npz and *_test.npz files",
    )
    parser.add_argument(
        "--out-results",
        default="results/results.csv",
        help="Output CSV path for metrics",
    )
    parser.add_argument(
        "--figures-dir",
        default="figures",
        help="Output directory for generated paper figures",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Trailing causal smoothing window (default: 5)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.0,
        help="Minimum alarm duration in seconds (default: 0.0)",
    )
    parser.add_argument(
        "--target-far",
        type=float,
        default=1 / 100,
        help="Target false-alarm rate per operating hour (default: 0.01)",
    )
    parser.add_argument(
        "--no-append",
        action="store_true",
        help="Overwrite out-results instead of updating/appending",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    evaluate_all(
        outputs_dir=args.outputs_dir,
        out_results=args.out_results,
        figures_dir=args.figures_dir,
        smooth_window=args.smooth_window,
        min_duration=args.min_duration,
        target_far=args.target_far,
        append=not args.no_append,
    )


if __name__ == "__main__":
    main()
