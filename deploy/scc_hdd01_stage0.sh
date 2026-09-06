#!/bin/bash
# Stage 0: per-modality contrastive pretraining on scc-hdd-01.
#
# Runs all four modalities in PARALLEL to cut wall time from ~220h to ~55h.
# Each gets 20 OMP threads (80 cores / 4) and batch=4 — the Swin transformer
# encoder activations at 180×360 use ~5 GB per run at batch=4, so 4×5=20 GB
# fits within the 30 GB RAM limit with headroom for the OS.
#
# batch=8+ OOMs silently: the attention backward over 45×90 patches blows up.

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
PYTHON=$HOME/miniconda3/bin/python

echo "[hdd01-stage0] Starting parallel per-modality pretraining (4 × 20 OMP threads)"

PIDS=()
for MOD in sd smag tec dmsp; do
    if   [[ "$MOD" == "smag" ]]; then CACHE_MOD=$CACHE/supermag
    elif [[ "$MOD" == "sd"   ]]; then CACHE_MOD=$CACHE/superdarn
    else                               CACHE_MOD=$CACHE/$MOD
    fi
    echo "[hdd01-stage0] Launching $MOD → $LOG_DIR/stage0-$MOD.log"
    OMP_NUM_THREADS=20 MKL_NUM_THREADS=20 \
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
        --batch_size  4                  \
        --max_steps   20000              \
        --num_workers 0                  \
        --window_steps 15                \
        --tau         0.07               \
        --lr          1e-3               \
        > "$LOG_DIR/stage0-$MOD.log" 2>&1 &
    PIDS+=($!)
done

echo "[hdd01-stage0] PIDs: ${PIDS[*]}"
echo "[hdd01-stage0] Waiting for all four modalities..."

# Wait for each and check exit codes
ALL_OK=true
for i in "${!PIDS[@]}"; do
    MOD_LIST=(sd smag tec dmsp)
    MOD=${MOD_LIST[$i]}
    if wait "${PIDS[$i]}"; then
        echo "[hdd01-stage0] $MOD finished OK → $CKPT_DIR/stage0_${MOD}_encoder.pt"
    else
        echo "[hdd01-stage0] $MOD FAILED (exit ${?}) — check $LOG_DIR/stage0-$MOD.log"
        ALL_OK=false
    fi
done

if [ "$ALL_OK" = false ]; then
    echo "[hdd01-stage0] One or more modalities failed — aborting lag analysis"
    exit 1
fi

echo "[hdd01-stage0] All modalities complete. Encoder weights in $CKPT_DIR"

# ------------------------------------------------------------------
# Causal lag analysis (auto-runs after all encoders are ready)
# ------------------------------------------------------------------
echo "[hdd01-stage0] Running causal lag analysis..."
OMP_NUM_THREADS=60 MKL_NUM_THREADS=60 \
$PYTHON -u "$IOME_DIR/scripts/analyse_lag.py" \
    --ckpt_stage0_dir "$CKPT_DIR"             \
    --splits_dir      "$SPLITS"               \
    --cache_sd        "$CACHE/superdarn"      \
    --cache_smag      "$CACHE/supermag"       \
    --cache_tec       "$CACHE/tec"            \
    --cache_dmsp      "$CACHE/dmsp"           \
    --stats_dir       "$SPLITS"               \
    --out             "$SPLITS/lag_matrix.json" \
    --max_lag_steps   30                      \
    --n_samples       5000                    \
    2>&1 | tee "$LOG_DIR/lag_analysis.log"

echo "[hdd01-stage0] Lag matrix → $SPLITS/lag_matrix.json"
echo "[hdd01-stage0] Now run: bash deploy/scc_hdd01_stage1.sh"
