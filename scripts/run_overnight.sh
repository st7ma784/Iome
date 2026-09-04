#!/bin/bash
# Overnight local test run: Stage 1 → Stage 2 (sequential).
# Logs to /data5/iome_cache/train_logs/
# Usage: nohup bash scripts/run_overnight.sh >> /data5/iome_cache/train_logs/overnight.log 2>&1 &

set -euo pipefail

CACHE=/data5/iome_cache
CKPT_DIR=$CACHE/ckpts
SPLITS=$CACHE/splits
LOG_DIR=$CACHE/train_logs

mkdir -p "$LOG_DIR" "$CKPT_DIR/stage1" "$CKPT_DIR/stage2" "$CKPT_DIR/stage3"

export WANDB_API_KEY=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
export WANDB_MODE=online
export PYTHONPATH=/home/user/Iome/src:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Restrict to GPU 0 (leave GPU 1/2 free for other users if present)
export CUDA_VISIBLE_DEVICES=0

CONDA_PYTHON=/home/user/miniconda3/envs/open-ce/bin/python
IOME_DIR=/home/user/Iome

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── Stage 1: Contrastive pre-training ───────────────────────────────────────
log "=== Stage 1 start ==="

$CONDA_PYTHON "$IOME_DIR/scripts/train_stage1.py" \
    --cache_sd   "$CACHE/superdarn"  \
    --cache_smag "$CACHE/supermag"  \
    --cache_tec  "$CACHE/tec"       \
    --omni_dir   "$CACHE/omni"      \
    --stats_dir  "$SPLITS"          \
    --ts_train   "$SPLITS/ts_train.json" \
    --ts_val     "$SPLITS/ts_val.json"   \
    --ckpt_dir   "$CKPT_DIR/stage1" \
    --batch_size 16                 \
    --max_steps  25000              \
    --num_workers 6                 \
    --wandb_project iome-local-test \
    2>&1 | tee "$LOG_DIR/stage1.log"

log "=== Stage 1 done ==="

# ── Stage 2: Dynamics pre-training ──────────────────────────────────────────
STAGE1_CKPT=$(ls -t "$CKPT_DIR/stage1/"*.ckpt 2>/dev/null | grep -v last | head -1)
if [[ -z "$STAGE1_CKPT" ]]; then
    STAGE1_CKPT=$(ls -t "$CKPT_DIR/stage1/last.ckpt" 2>/dev/null || true)
fi
if [[ -z "$STAGE1_CKPT" ]]; then
    log "ERROR: no Stage 1 checkpoint found — skipping Stage 2"
    exit 1
fi
log "Stage 1 checkpoint: $STAGE1_CKPT"
log "=== Stage 2 start ==="

$CONDA_PYTHON "$IOME_DIR/scripts/train_stage2.py" \
    --stage1_ckpt "$STAGE1_CKPT"    \
    --cache_sd   "$CACHE/superdarn"  \
    --cache_smag "$CACHE/supermag"  \
    --cache_tec  "$CACHE/tec"       \
    --omni_dir   "$CACHE/omni"      \
    --stats_dir  "$SPLITS"          \
    --ts_train   "$SPLITS/ts_train.json" \
    --ts_val     "$SPLITS/ts_val.json"   \
    --ckpt_dir   "$CKPT_DIR/stage2" \
    --batch_size 16                 \
    --max_steps  15000              \
    --num_workers 6                 \
    --wandb_project iome-local-test \
    2>&1 | tee "$LOG_DIR/stage2.log"

log "=== Stage 2 done ==="
log "All stages complete. Checkpoints in $CKPT_DIR/"
