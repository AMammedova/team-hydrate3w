#!/usr/bin/env bash
# ==========================================================================
# Single-command reproduction entry point.
#
# Usage:
#   bash run_all.sh
#
# Optional positional arguments:
#   bash run_all.sh data/3W/dataset 9
#
# Pipeline:
#   1. Data inventory
#   2. Main 5-channel cache build
#   3. Leak-free grouped fold generation
#   4. XGBoost training + validation outputs
#   5. TCN and GRU training + validation/test predictions
#   6. Validation-only threshold selection followed by ONE frozen test eval
#   7. Results aggregation, LaTeX tables and figures
#
# Frozen experiment settings:
#   channels       = P-MON-CKP,P-JUS-CKGL,T-TPT,T-JUS-CKP,P-ANULAR
#   folds          = 3
#   repeats        = 1
#   split seed     = 42
#   model seed     = 42
#   deep batch     = 64
#   deep LR        = 1e-3
#   max epochs     = 100
#   patience       = 10
#   deep precision = FP32 (--no-amp)
#
# NOTE:
# FP16 and BF16 mixed-precision smoke tests produced non-finite GRU
# validation probabilities on the project GPU. Final deep-model runs
# therefore use FP32.
# ==========================================================================

set -euo pipefail


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

ROOT="${1:-data/3W/dataset}"
EVENT="${2:-9}"

CACHE_DIR="data/cache"
RESULTS_DIR="results"
OUTPUTS_DIR="$RESULTS_DIR/model_outputs"
FIGURES_DIR="figures"
TABLES_DIR="$RESULTS_DIR/tables"
CHECKPOINT_DIR="checkpoints/final"

CHANNELS_MAIN="P-MON-CKP,P-JUS-CKGL,T-TPT,T-JUS-CKP,P-ANULAR"

N_SPLITS=3
N_REPEATS=1
SPLIT_SEED=42
MODEL_SEED=42

BATCH_SIZE=64
MAX_EPOCHS=100
PATIENCE=10
NUM_WORKERS=0
LEARNING_RATE="1e-3"

SMOOTH_WINDOW=5
MIN_DURATION="0.0"
TARGET_FAR="0.01"


echo "============================================================"
echo " Hydrate Formation Early Warning — Reproduction Pipeline"
echo "============================================================"
echo " Dataset root : $ROOT"
echo " Event        : $EVENT"
echo " Cache        : $CACHE_DIR"
echo " Results      : $RESULTS_DIR"
echo "============================================================"


mkdir -p \
    "$RESULTS_DIR" \
    "$OUTPUTS_DIR" \
    "$FIGURES_DIR" \
    "$TABLES_DIR" \
    "$CHECKPOINT_DIR"


# ==========================================================================
# Step 1 — Data inventory
# ==========================================================================

echo ""
echo "============================================================"
echo " 1. Data Inventory"
echo "============================================================"

python -m src.data.inventory \
    --root "$ROOT" \
    --event "$EVENT"


# ==========================================================================
# Step 2 — Main cache
# ==========================================================================

echo ""
echo "============================================================"
echo " 2. Building Main 5-Channel Cache"
echo "============================================================"

python -m src.data.build_cache \
    --root "$ROOT" \
    --out "$CACHE_DIR" \
    --channels "$CHANNELS_MAIN"


# ==========================================================================
# Step 3 — Frozen grouped folds
# ==========================================================================

echo ""
echo "============================================================"
echo " 3. Generating Frozen Grouped Folds"
echo "============================================================"

if compgen -G "$CACHE_DIR/*.npz" > /dev/null; then

    python -m src.data.splits \
        --cache "$CACHE_DIR" \
        --n-splits "$N_SPLITS" \
        --n-repeats "$N_REPEATS" \
        --val-mode nested \
        --seed "$SPLIT_SEED" \
        --out "$RESULTS_DIR/fold_report.csv"

    echo "[M2] Fold report written to:"
    echo "     $RESULTS_DIR/fold_report.csv"

else

    echo "[ERROR] No cache .npz files found in $CACHE_DIR"
    exit 1

fi


# ==========================================================================
# Step 4 — XGBoost baseline
# ==========================================================================

echo ""
echo "============================================================"
echo " 4. XGBoost Baseline"
echo "============================================================"

# XGBoost writes validation and test PREDICTIONS.
#
# It does NOT use test metrics for model or threshold selection.
# Test predictions remain untouched until Step 6, where the validation
# operating point is frozen and then applied once to the test folds.

python -m tools.train_xgb \
    --cache "$CACHE_DIR" \
    --out-results "$RESULTS_DIR/results.csv" \
    --outputs-dir "$OUTPUTS_DIR" \
    --tables-dir "$TABLES_DIR" \
    --figures-dir "$FIGURES_DIR" \
    --conditions real_only,real_plus_sim \
    --seeds "$MODEL_SEED" \
    --n-splits "$N_SPLITS" \
    --n-repeats "$N_REPEATS" \
    --split-seed "$SPLIT_SEED" \
    --device auto \
    --calibration platt \
    --append


# ==========================================================================
# Step 5 — TCN and GRU
# ==========================================================================

echo ""
echo "============================================================"
echo " 5. Deep Models — TCN + GRU"
echo "============================================================"

DEEP_DEVICE="${DEEP_DEVICE:-auto}"

if [ "$DEEP_DEVICE" = "auto" ]; then

    if python -c \
        "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" \
        2>/dev/null
    then
        DEEP_DEVICE="cuda"
    else
        DEEP_DEVICE="cpu"
    fi

fi

echo "[M4] Deep-model device: $DEEP_DEVICE"

# Frozen final configuration:
#
# FP16 AMP -> non-finite GRU probabilities
# BF16 AMP -> non-finite GRU probabilities
#
# Therefore final reproducible training uses FP32.
DEEP_EXTRA_ARGS="--no-amp"


python -m tools.train_deep_models \
    --cache "$CACHE_DIR" \
    --models tcn,gru \
    --conditions real_only,real_plus_sim \
    --seeds "$MODEL_SEED" \
    --n-splits "$N_SPLITS" \
    --n-repeats "$N_REPEATS" \
    --split-seed "$SPLIT_SEED" \
    --batch-size "$BATCH_SIZE" \
    --max-epochs "$MAX_EPOCHS" \
    --patience "$PATIENCE" \
    --num-workers "$NUM_WORKERS" \
    --lr "$LEARNING_RATE" \
    --device "$DEEP_DEVICE" \
    --out-results "$RESULTS_DIR/results.csv" \
    --outputs-dir "$OUTPUTS_DIR" \
    --checkpoint-root "$CHECKPOINT_DIR" \
    $DEEP_EXTRA_ARGS

echo "[M4] Deep-model training complete."


# ==========================================================================
# Step 6 — Frozen threshold + ONE test evaluation
# ==========================================================================

echo ""
echo "============================================================"
echo " 6. Frozen Operating Point + Test Evaluation"
echo "============================================================"

# IMPORTANT:
#
# Threshold selection is performed using validation Normal-operation
# instances only.
#
# The selected threshold, smoothing window and minimum-duration policy are
# then frozen and applied unchanged to the corresponding unseen test fold.
#
# FAR numerator and denominator both use the same strictly-Normal
# population.

if compgen -G "$OUTPUTS_DIR/*_val.npz" > /dev/null; then

    python -m src.eval.evaluate_predictions \
        --outputs-dir "$OUTPUTS_DIR" \
        --out-results "$RESULTS_DIR/results.csv" \
        --figures-dir "$FIGURES_DIR" \
        --smooth-window "$SMOOTH_WINDOW" \
        --min-duration "$MIN_DURATION" \
        --target-far "$TARGET_FAR"

else

    echo "[ERROR] No validation prediction files found in $OUTPUTS_DIR"
    exit 1

fi


# ==========================================================================
# Step 7 — Aggregate final results
# ==========================================================================

echo ""
echo "============================================================"
echo " 7. Aggregating Final Results"
echo "============================================================"

python - <<'PY'
from src.eval.aggregate import (
    generate_all_tables,
    load_results,
    summarize_folds,
)

results_path = "results/results.csv"
summary_path = "results/summary.csv"
tables_dir = "results/tables"

df = load_results(results_path)

summary = summarize_folds(df)
summary.to_csv(summary_path, index=False)

generate_all_tables(
    results_path,
    tables_dir,
)

print()
print("Final summary:")
print(summary.to_string(index=False))

print()
print(f"Summary written to: {summary_path}")
print(f"Tables written to : {tables_dir}")
PY


# ==========================================================================
# Complete
# ==========================================================================

echo ""
echo "============================================================"
echo " Pipeline Complete"
echo "============================================================"
echo ""
echo "Final outputs:"
echo "  $RESULTS_DIR/results.csv"
echo "  $RESULTS_DIR/summary.csv"
echo "  $RESULTS_DIR/fold_report.csv"
echo "  $RESULTS_DIR/tables/"
echo "  $OUTPUTS_DIR/"
echo "  $FIGURES_DIR/"
echo ""
echo "============================================================"