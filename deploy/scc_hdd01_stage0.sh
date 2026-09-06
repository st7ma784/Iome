#!/bin/bash
# Stage 0: per-modality contrastive pretraining on scc-hdd-01.
# Runs all four modalities in sequence (memory constraint — 30 GB RAM).
# Each run uses OMP_NUM_THREADS=60 and batch=64 (pair dataset, no spatial
# activations needed during forward, much lighter than reconstruction).

set -euo pipefail

CACHE=/data/iome_cache
SPLITS=$CACHE/splits
IOME_DIR="$HOME/iome"
LOG_DIR=$CACHE/train_logs
CKPT_DIR=$CACHE/ckpts/stage0

mkdir -p "$LOG_DIR" "$CKPT_DIR"

export WANDB_API_KEY=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
export WANDB_ENTITY=st7ma784
export WANDB_PROJECT=iome
export WANDB_MODE=online
export PYTHONPATH=$IOME_DIR/src:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=60
export MKL_NUM_THREADS=60
PYTHON=$HOME/miniconda3/bin/python

echo "[hdd01-stage0] Starting per-modality pretraining"

for MOD in sd smag tec dmsp; do
    CACHE_MOD=$CACHE/$( [[ "$MOD" == "smag" ]] && echo "supermag" \
                      || [[ "$MOD" == "sd"   ]] && echo "superdarn" \
                      || echo "$MOD" )
    echo "[hdd01-stage0] === $MOD ==="
    $PYTHON -u "$IOME_DIR/scripts/train_stage0.py" \
        --modality    "$MOD"             \
        --cache_dir   "$CACHE_MOD"       \
        --splits_dir  "$SPLITS"          \
        --stats_dir   "$SPLITS"          \
        --ckpt_dir    "$CKPT_DIR"        \
        --wandb_name  "stage0-$MOD"      \
        --accelerator cpu                \
        --devices     1                  \
        --precision   32                 \
        --batch_size  32                 \
        --max_steps   20000              \
        --num_workers 4                  \
        --window_steps 15                \
        --tau         0.07               \
        --lr          1e-3               \
        2>&1 | tee "$LOG_DIR/stage0-$MOD.log"
    echo "[hdd01-stage0] $MOD done → $CKPT_DIR/stage0_${MOD}_encoder.pt"
done

echo "[hdd01-stage0] All modalities complete. Encoder weights in $CKPT_DIR"

# ------------------------------------------------------------------
# Causal lag analysis (auto-runs after all encoders are ready)
# ------------------------------------------------------------------
echo "[hdd01-stage0] Running causal lag analysis..."
$PYTHON -u "$IOME_DIR/scripts/analyse_lag.py" \
    --ckpt_stage0_dir "$CKPT_DIR"             \
    --splits_dir      "$SPLITS"               \
    --cache_sd        "$CACHE/superdarn"      \
    --cache_smag      "$CACHE/supermag"       \
    --cache_tec       "$CACHE/tec"            \
    --cache_dmsp      "$CACHE/dmsp"           \
    --stats_dir       "$SPLITS"              \
    --out             "$SPLITS/lag_matrix.json" \
    --max_lag_steps   30                      \
    --n_samples       5000                    \
    2>&1 | tee "$LOG_DIR/lag_analysis.log"

echo "[hdd01-stage0] Lag matrix → $SPLITS/lag_matrix.json"
echo "[hdd01-stage0] Now run: bash deploy/scc_hdd01_stage1.sh"
