#!/bin/bash
# Stage 1: local lab run (scc-ws / scc-hdd-01 data at /data5).
# Usage:  bash deploy/lab_stage1.sh
# Runs in background via nohup; tail logs at /data5/iome_cache/train_logs/stage1.log

set -euo pipefail

CACHE=/data5/iome_cache
SPLITS=$CACHE/splits
IOME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR=$CACHE/train_logs
CKPT_DIR=$CACHE/ckpts/stage1

mkdir -p "$LOG_DIR" "$CKPT_DIR"

export WANDB_API_KEY=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
export WANDB_ENTITY=PGNTeam
export WANDB_PROJECT=iome
export WANDB_MODE=online
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$IOME_DIR/src:${PYTHONPATH:-}

echo "[lab] iome dir : $IOME_DIR"
echo "[lab] splits   : $SPLITS"
echo "[lab] ckpt dir : $CKPT_DIR"
echo "[lab] log      : $LOG_DIR/stage1.log"
echo "[lab] GPU      : $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'not detected yet')"

# Regenerate splits with union logic if ts_avail.json is missing
if [[ ! -f "$SPLITS/ts_avail.json" ]]; then
    echo "[lab] ts_avail.json not found — running make_timestamps.py ..."
    conda run -n open-ce python "$IOME_DIR/scripts/make_timestamps.py" \
        --cache_root "$CACHE" \
        --out_dir    "$SPLITS" \
        --min_modalities 2
    echo "[lab] splits regenerated."
fi

nohup conda run -n open-ce python "$IOME_DIR/scripts/train_stage1.py" \
    --splits_dir  "$SPLITS"             \
    --cache_sd    "$CACHE/superdarn"    \
    --cache_smag  "$CACHE/supermag"     \
    --cache_tec   "$CACHE/tec"          \
    --cache_dmsp  "$CACHE/dmsp"         \
    --ckpt_dir    "$CKPT_DIR"           \
    --omni_dir    "$CACHE/omni"         \
    --stats_dir   "$SPLITS"             \
    --batch_size  16                    \
    --max_steps   50000                 \
    --num_workers 8                     \
    --p_mod_drop  0.3                   \
    --wandb_project iome                \
    > "$LOG_DIR/stage1.log" 2>&1 &

PID=$!
echo "[lab] launched stage 1 as PID $PID"
echo "$PID" > "$LOG_DIR/stage1.pid"
echo "[lab] tail -f $LOG_DIR/stage1.log"
