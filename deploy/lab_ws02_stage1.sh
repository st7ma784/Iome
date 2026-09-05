#!/bin/bash
# Stage 1 on scc-ws-02 — 3× Tesla P100 16GB.
# Run from: scc-ws-02, in the /home/user/Iome directory.
# Usage: bash deploy/lab_ws02_stage1.sh

set -euo pipefail

CACHE=/data5/iome_cache
SPLITS=$CACHE/splits
IOME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR=$CACHE/train_logs
CKPT_DIR=$CACHE/ckpts/stage1-ws02

mkdir -p "$LOG_DIR" "$CKPT_DIR"

export WANDB_API_KEY=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
export WANDB_ENTITY=st7ma784
export WANDB_PROJECT=iome
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH=$IOME_DIR/src:${PYTHONPATH:-}

echo "[ws02] $(hostname) — 3× P100"
echo "[ws02] ckpt: $CKPT_DIR"
echo "[ws02] log : $LOG_DIR/stage1-ws02.log"

nohup conda run -n open-ce python "$IOME_DIR/scripts/train_stage1.py" \
    --splits_dir  "$SPLITS"             \
    --cache_sd    "$CACHE/superdarn"    \
    --cache_smag  "$CACHE/supermag"     \
    --cache_tec   "$CACHE/tec"          \
    --cache_dmsp  "$CACHE/dmsp"         \
    --ckpt_dir    "$CKPT_DIR"           \
    --omni_dir    "$CACHE/omni"         \
    --stats_dir   "$SPLITS"             \
    --batch_size  4                     \
    --max_steps   50000                 \
    --num_workers 4                     \
    --accelerator cpu                   \
    --devices     1                     \
    --precision   32                    \
    --p_mod_drop  0.3                   \
    --wandb_project iome                \
    > "$LOG_DIR/stage1-ws02.log" 2>&1 &

PID=$!
echo "[ws02] PID $PID"
echo "$PID" > "$LOG_DIR/stage1-ws02.pid"
echo "[ws02] tail -f $LOG_DIR/stage1-ws02.log"
