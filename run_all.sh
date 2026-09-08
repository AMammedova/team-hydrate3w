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
if [ -d "$CACHE_DIR" ] && [ -n "$(ls -A "$CACHE_DIR"/*.npz 2>/dev/null)" ]; then
  python -m src.data.splits \
      --cache "$CACHE_DIR" \
      --n-splits 3 \
      --n-repeats 1 \
      --val-mode nested \
      --seed 42 \
      --out "$RESULTS_DIR/fold_report.csv"
  echo "[M2] Fold report generated → $RESULTS_DIR/fold_report.csv"
else
  echo "[M2] SKIPPED: no cache at $CACHE_DIR -- run step 2 first."
fi

# ------------------------------------------------------------------
# Step 4: Baseline Models — XGBoost (M3)
# ------------------------------------------------------------------
echo ""
echo "==========================================="
echo " 4. Running Baseline Models — XGBoost (M3)"
echo "==========================================="
# M3 owns this step. tools/train_xgb.py runs the full Result-1 baseline matrix
# (XGBoost x {real_only, real_plus_sim}) and writes, for every fold:
#   results/results.csv                  validation metrics, contract schema
#   results/model_outputs/*_val.npz      probabilities Module 8 selects on
#   results/model_outputs/*_test.npz     probabilities Module 8 scores ONCE
#   results/tables/*_importance_*.csv    gain + permutation importance
#   figures/reliability_xgboost.png      before/after calibration
#
# --device auto uses the GPU when XGBoost can genuinely see one and falls back
# to CPU otherwise; it never claims a GPU it did not get.
#
# Test METRICS stay off until the S3 freeze. Test probabilities are written
# every run, because Module 8 needs them; writing predictions is not the same
# as reading the score. After the freeze, re-run this step with --eval-test.
XGB_EVAL_TEST="${XGB_EVAL_TEST:-}"

if [ -d "$CACHE_DIR" ] && [ -n "$(ls -A "$CACHE_DIR"/*.npz 2>/dev/null)" ]; then
  python -m tools.train_xgb \
      --cache "$CACHE_DIR" \
      --out-results "$RESULTS_DIR/results.csv" \
      --outputs-dir "$RESULTS_DIR/model_outputs" \
      --tables-dir "$RESULTS_DIR/tables" \
      --figures-dir "$FIGURES_DIR" \
      --conditions real_only,real_plus_sim \
      --seeds 42 \
      --n-splits 3 \
      --device auto \
      --calibration platt \
      --append \
      ${XGB_EVAL_TEST}
else
  echo "[M3] SKIPPED: no cache at $CACHE_DIR -- run step 2 first."
fi

# ------------------------------------------------------------------
# Step 5: Deep Models — TCN & GRU (M4)
# ------------------------------------------------------------------
echo ""
echo "==========================================="
echo " 5. Running Deep Models — TCN & GRU (M4)"
echo "==========================================="
DEEP_DEVICE="${DEEP_DEVICE:-auto}"
if [ "$DEEP_DEVICE" = "auto" ]; then
  if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEEP_DEVICE="cuda"
  else
    DEEP_DEVICE="cpu"
  fi
fi

DEEP_EXTRA_ARGS=""
if [ "$DEEP_DEVICE" = "cpu" ]; then
  DEEP_EXTRA_ARGS="--no-amp"
fi

if [ -d "$CACHE_DIR" ] && [ -n "$(ls -A "$CACHE_DIR"/*.npz 2>/dev/null)" ]; then
  python -m tools.train_deep_models \
      --cache "$CACHE_DIR" \
      --models tcn,gru \
      --conditions real_only,real_plus_sim \
      --seeds 42 \
      --n-splits 3 \
      --n-repeats 1 \
      --device "$DEEP_DEVICE" \
      --out-results "$RESULTS_DIR/results.csv" \
      --outputs-dir "$RESULTS_DIR/model_outputs" \
      --checkpoint-root checkpoints \
      ${DEEP_EXTRA_ARGS}
  echo "[M4] Deep model training complete; probabilities written to $RESULTS_DIR/model_outputs"
else
  echo "[M4] SKIPPED: no cache at $CACHE_DIR -- run step 2 first."
fi

# ------------------------------------------------------------------
# Step 6: Threshold Selection & Test Evaluation (M5)
# ------------------------------------------------------------------
echo ""
echo "==========================================="
echo " 6. Threshold Selection & Test Evaluation (M5)"
echo "==========================================="
# This step runs AFTER models have produced results/model_outputs/*.npz.
# Thresholds are selected on VALIDATION folds ONLY (contract §0.3),
# then applied UNCHANGED to test folds — the S3 FREEZE point.
if [ -d "$RESULTS_DIR/model_outputs" ] && [ -n "$(ls -A "$RESULTS_DIR/model_outputs"/*_val.npz 2>/dev/null)" ]; then
  python -m src.eval.evaluate_predictions \
      --outputs-dir "$RESULTS_DIR/model_outputs" \
      --out-results "$RESULTS_DIR/results.csv" \
      --figures-dir "$FIGURES_DIR" \
      --smooth-window 5 \
      --min-duration 0.0 \
      --target-far 0.01

  echo "[M5] Generating LaTeX summary tables..."
  python -c "
from src.eval.aggregate import load_results, summarize_folds, generate_all_tables
import os
results_path = '$RESULTS_DIR/results.csv'
if os.path.exists(results_path):
    df = load_results(results_path)
    summary = summarize_folds(df)
    summary.to_csv('$RESULTS_DIR/summary.csv', index=False)
    generate_all_tables(results_path, '$TABLES_DIR/')
    print('[M5] LaTeX tables written to $TABLES_DIR/')
"
else
  echo "[M5] No model output files in $RESULTS_DIR/model_outputs yet. Skipping evaluation."
fi

# ------------------------------------------------------------------
# Step 7: Summary & Status
# ------------------------------------------------------------------
echo ""
echo "==========================================="
echo " 7. Pipeline Summary"
echo "==========================================="
echo " Expected outputs (once full GPU runs complete):"
echo "   $RESULTS_DIR/results.csv     — per-fold raw metrics"
echo "   $RESULTS_DIR/summary.csv     — mean ± std aggregation"
echo "   $TABLES_DIR/*.tex            — LaTeX tables for report"
echo "   $FIGURES_DIR/*.png           — all paper figures"
echo ""
echo " Pipeline Complete!"
echo "==========================================="
