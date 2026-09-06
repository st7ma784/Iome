#!/bin/bash
# Stage 1 on scc-hdd-01 — 80-core CPU, /data is local disk.
# 146M param model on 180×360 grids uses ~10-12GB activation memory during
# backprop, so only ONE run fits in 30GB RAM. Use all 60 OMP threads for compute.

set -euo pipefail

CACHE=/data/iome_cache
SPLITS=$CACHE/splits
IOME_DIR="$HOME/iome"
LOG_DIR=$CACHE/train_logs
CKPT_DIR=$CACHE/ckpts/stage1-hdd01

mkdir -p "$LOG_DIR" "$CKPT_DIR"

export WANDB_API_KEY=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
export WANDB_ENTITY=st7ma784
export WANDB_PROJECT=iome
export WANDB_MODE=online
export PYTHONPATH=$IOME_DIR/src:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
PYTHON=$HOME/miniconda3/bin/python
# One run: 60 OMP threads for dense matrix ops, 4 data workers, leftover for OS
export OMP_NUM_THREADS=60
export MKL_NUM_THREADS=60

echo "[hdd01] $(hostname) — single CPU stage-1 (60 OMP threads)"

# ------------------------------------------------------------------
# Grid conversion check
# ------------------------------------------------------------------
SD_SHAPE=$($PYTHON -c "
import numpy as np; from pathlib import Path
f = next(Path('$CACHE/superdarn').glob('*_sd.npy'), None)
print(np.load(f, mmap_mode='r').shape[1] if f else 0)
" 2>/dev/null || echo 0)
if [[ "$SD_SHAPE" == "120" ]]; then
    echo "[hdd01] Converting 120×120 → 180×360 ..."
    $PYTHON "$IOME_DIR/scripts/convert_grid_120_to_180360.py" \
        --cache_root "$CACHE" --workers 40 --no_backup \
        2>&1 | tee "$LOG_DIR/grid_convert_hdd01.log"
    echo "[hdd01] Conversion done."
fi

# ------------------------------------------------------------------
# Single training run
# ------------------------------------------------------------------
# Use lag matrix if available (produced by analyse_lag.py after Stage 0)
LAG_ARGS=""
if [[ -f "$SPLITS/lag_matrix.json" ]]; then
    LAG_ARGS="--lag_matrix $SPLITS/lag_matrix.json"
    echo "[hdd01] Using lag-corrected cross-modal CLIP: $SPLITS/lag_matrix.json"
fi

# Use stage0 encoder weights if available
STAGE0_ARGS=""
STAGE0_CKPT_DIR="$CACHE/ckpts/stage0"
if [[ -d "$STAGE0_CKPT_DIR" ]]; then
    STAGE0_ARGS="--ckpt_stage0_dir $STAGE0_CKPT_DIR"
    echo "[hdd01] Loading stage0 encoder weights from $STAGE0_CKPT_DIR"
fi

nohup $PYTHON -u "$IOME_DIR/scripts/train_stage1.py" \
    --splits_dir  "$SPLITS"             \
    --cache_sd    "$CACHE/superdarn"    \
    --cache_smag  "$CACHE/supermag"     \
    --cache_tec   "$CACHE/tec"          \
    --cache_dmsp  "$CACHE/dmsp"         \
    --ckpt_dir    "$CKPT_DIR"           \
    --omni_dir    "$CACHE/omni"         \
    --stats_dir   "$SPLITS"             \
    --batch_size  1                     \
    --max_steps   50000                 \
    --num_workers 2                     \
    --accelerator cpu                   \
    --devices     1                     \
    --precision   32                    \
    --p_mod_drop  0.3                   \
    --wandb_project iome                \
    --wandb_name  hdd01-drop0.3-tau0.1  \
    $STAGE0_ARGS                        \
    $LAG_ARGS                           \
    > "$LOG_DIR/stage1-hdd01.log" 2>&1 &

PID=$!
echo "[hdd01] PID $PID  (batch=1, workers=2, OMP=60)"
echo "$PID" > "$LOG_DIR/stage1-hdd01.pid"
echo "[hdd01] log: $LOG_DIR/stage1-hdd01.log"
