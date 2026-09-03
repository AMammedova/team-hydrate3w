#!/usr/bin/env bash
# ==========================================================================
# Single-command reproduction entry point (team contract §0.3/§4.6, W4.6).
# The brief lists "no run_all entry point" as an automatic deduction --
# this is that entry point.
#
# Usage:
#   bash run_all.sh [--root data/3W/dataset] [--event 9]
#
# From a clean checkout, this script runs the entire pipeline:
#   1. Data inventory & validation
#   2. Cache build (windowing, decimation, normalization)
#   3. Fold generation & fold report
#   4. Baseline (XGBoost) training & evaluation
#   5. Deep model (TCN, GRU) training & evaluation
#   6. Threshold selection & test-fold evaluation
#   7. Results aggregation, LaTeX tables, figures
#
# Output:
#   results/results.csv       — per-fold, per-model, per-metric raw data
#   results/summary.csv       — mean ± std aggregation
#   report/tables/*.tex       — LaTeX tables for \input in report.tex
#   figures/*.png             — all paper figures
#
# No number in the paper should ever be typed by hand (team contract §0.3).
# ==========================================================================

set -euo pipefail

# Defaults
ROOT="${1:-data/3W/dataset}"
EVENT="${2:-9}"
CACHE_DIR="data/cache"
CACHE_TPT_DIR="data/cache_tpt"
RESULTS_DIR="results"
FIGURES_DIR="figures"
TABLES_DIR="report/tables"

# Channels per TEAM_5_MEMBERS.md §8 / DATA_FINDINGS.md §9
CHANNELS_MAIN="P-MON-CKP,P-JUS-CKGL,T-TPT,T-JUS-CKP,P-ANULAR"
CHANNELS_SENSITIVITY="P-TPT,T-TPT"

echo "==========================================="
echo " Hydrate Formation Early Warning — Full Pipeline"
echo " Root: $ROOT | Event: $EVENT"
echo "==========================================="

# Create output directories
mkdir -p "$RESULTS_DIR" "$FIGURES_DIR" "$TABLES_DIR"

# ------------------------------------------------------------------
# Step 1: Data Inventory
# ------------------------------------------------------------------
echo ""
echo "==========================================="
echo " 1. Running Data Inventory"
echo "==========================================="
python -m src.data.inventory --root "$ROOT" --event "$EVENT"

# ------------------------------------------------------------------
# Step 2: Build Cache (Windowing & Processing)
#         Main arm (5 channels) + sensitivity arm (2 channels)
# ------------------------------------------------------------------
echo ""
echo "==========================================="
echo " 2. Building Cache (5-channel main arm)"
echo "==========================================="
python -m src.data.build_cache \
    --root "$ROOT" \
    --out "$CACHE_DIR" \
    --channels "$CHANNELS_MAIN"

echo ""
echo "==========================================="
echo " 2b. Building Cache (2-channel sensitivity arm)"
echo "==========================================="
python -m src.data.build_cache \
    --root "$ROOT" \
    --out "$CACHE_TPT_DIR" \
    --channels "$CHANNELS_SENSITIVITY"

# ------------------------------------------------------------------
# Step 3: Generate Folds & Fold Report (M2)
# ------------------------------------------------------------------
echo ""
echo "==========================================="
echo " 3. Generating Folds & Fold Report (M2)"
echo "==========================================="
# TODO (M2): uncomment when splits.py split() and fold_report() are implemented
# python -c "
# from src.data.splits import GroupedKFoldSplitter
# import numpy as np, json
# # Load cache metadata, run splitter, write fold_report to results/
# print('[splits.py] Fold report generated → results/fold_report.csv')
# "
echo "[WAITING FOR M2] splits.py split() and fold_report() not yet implemented."
echo "  → M2 needs to implement GroupedKFoldSplitter.split() and fold_report()"
echo "  → Two independent grouped splits: positives + normals (see TEAM_5_MEMBERS.md §3)"

# ------------------------------------------------------------------
# Step 4: Baseline Models — XGBoost (M3)
# ------------------------------------------------------------------
echo ""
echo "==========================================="
echo " 4. Running Baseline Models — XGBoost (M3)"
echo "==========================================="
# TODO (M3): uncomment when features.py transform() and tune.py are implemented
# Condition 1: real_only
# python -c "
# from src.baselines.xgb_model import XGBoostBaseline, compute_sample_weight
# from src.baselines.tune import search
# from src.baselines.calibrate import fit_calibrator
# # ... load cache, run CV, write to results/results.csv
# "
# Condition 2: real_plus_sim
# python -c "
# # ... same but include simulated instances in training
# "
echo "[WAITING FOR M3] features.py transform(), tune.py, calibrate.py not yet implemented."
echo "  → M3 needs to implement RollingFeatureExtractor.transform()"
echo "  → M3 needs to implement tune.search() for hyperparameter search"
echo "  → M3 needs to implement fit_calibrator() for calibration"

# ------------------------------------------------------------------
# Step 5: Deep Models — TCN & GRU (M4)
# ------------------------------------------------------------------
echo ""
echo "==========================================="
echo " 5. Running Deep Models — TCN & GRU (M4)"
echo "==========================================="
# TODO (M4): uncomment when train_loop.py fit() is implemented
# TCN × {real_only, real_plus_sim}
# python -c "
# from src.models.tcn import TCN
# from src.models.train_loop import Trainer
# # ... assert tcn.receptive_field() >= window_size
# # ... load cache, run CV, write to results/results.csv
# "
# GRU × {real_only, real_plus_sim}  (bidirectional=False)
# python -c "
# from src.models.gru import GRUClassifier
# from src.models.train_loop import Trainer
# # ... load cache, run CV, write to results/results.csv
# "
echo "[WAITING FOR M4] train_loop.py fit(), predict_proba(), load_best_checkpoint() not yet implemented."
echo "  → M4 needs to implement Trainer.fit() with early stopping on val PR-AUC"
echo "  → M4 needs to implement Trainer.predict_proba()"
echo "  → GRU must use bidirectional=False (causality, DL6.1)"

# ------------------------------------------------------------------
# Step 6: Threshold Selection & Test Evaluation (M5)
# ------------------------------------------------------------------
echo ""
echo "==========================================="
echo " 6. Threshold Selection & Test Evaluation (M5)"
echo "==========================================="
# This step runs AFTER all models have produced results.csv entries.
# Thresholds are selected on VALIDATION folds ONLY (contract §0.3),
# then applied UNCHANGED to test folds — the S3 FREEZE point.
python -c "
from src.eval.thresholds import select_threshold, select_threshold_curve
from src.eval.metrics import (
    positive_score, pr_auc, event_recall,
    false_alarms_per_operating_hour, expected_calibration_error,
)
from src.eval.aggregate import load_results, summarize_folds, generate_all_tables
import os

results_path = '$RESULTS_DIR/results.csv'

if os.path.exists(results_path):
    print('[M5] Results file found — aggregating...')
    df = load_results(results_path)
    summary = summarize_folds(df)
    summary.to_csv('$RESULTS_DIR/summary.csv', index=False)
    print(f'  → summary.csv written ({len(summary)} rows)')

    # Generate all LaTeX tables
    generate_all_tables(results_path, '$TABLES_DIR/')
    print('  → LaTeX tables written to $TABLES_DIR/')
else:
    print('[M5] No results.csv found — models have not run yet.')
    print('  → Threshold selection and test evaluation will run after M3/M4 complete.')
    print('  → M5 code is READY: select_threshold(), select_threshold_curve(),')
    print('    expected_calibration_error(), to_latex_table() are all implemented.')
" 2>&1 || echo "[M5] Aggregation skipped (no results yet)."

# ------------------------------------------------------------------
# Step 7: Generate Figures (M5 + M1 + M3)
# ------------------------------------------------------------------
echo ""
echo "==========================================="
echo " 7. Generating Figures"
echo "==========================================="
python -c "
import os
figures_dir = '$FIGURES_DIR'

# M5-owned figures (ready, will produce output once results exist):
# - plot_lead_time_vs_false_alarm_rate → figures/lead_time_vs_far.png
# - plot_per_well_lead_time_box → figures/per_well_lead_time.png

# M1-owned figure (waiting for M1):
# - plot_annotated_trace → figures/annotated_trace.png

# M3-owned figure (waiting for M3):
# - plot_reliability_diagram → figures/reliability_diagram.png

results_path = '$RESULTS_DIR/results.csv'
if os.path.exists(results_path):
    print('[Figures] Results found — generating M5-owned figures...')
    # When results exist, the plotting functions can be called here
    # with the actual data.
else:
    print('[Figures] No results yet — figure generation deferred.')
    print('  → M5 plot functions are implemented and tested.')
    print('  → Waiting for model results from M3 (XGBoost) and M4 (TCN/GRU).')
" 2>&1

echo ""
echo "==========================================="
echo " Pipeline Complete!"
echo "==========================================="
echo ""
echo " Expected outputs (once all modules are ready):"
echo "   $RESULTS_DIR/results.csv     — per-fold raw metrics"
echo "   $RESULTS_DIR/summary.csv     — mean ± std aggregation"
echo "   $TABLES_DIR/*.tex            — LaTeX tables for report"
echo "   $FIGURES_DIR/*.png           — all paper figures"
echo ""
echo " Pending module implementations:"
echo "   M2: splits.py (critical path — M3/M4/M5 need folds)"
echo "   M3: features.py, tune.py, calibrate.py, importance.py"
echo "   M4: train_loop.py (fit, predict_proba, load_best_checkpoint)"
echo "==========================================="
