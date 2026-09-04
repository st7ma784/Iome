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
export WANDB_ENTITY=st7ma784
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

# Check if grid conversion is still needed (any 120x120 files remaining)
SD_SHAPE=$(conda run -n open-ce python -c "import numpy as np; from pathlib import Path; f=next(Path('$CACHE/superdarn').glob('*_sd.npy'),None); print(np.load(f,mmap_mode='r').shape[1] if f else 0)" 2>/dev/null || echo 0)
if [[ "$SD_SHAPE" == "120" ]]; then
    echo "[lab] Legacy 120x120 data detected — running grid conversion first..."
    conda run -n open-ce python "$IOME_DIR/scripts/convert_grid_120_to_180360.py" \
        --cache_root "$CACHE" --workers 12 --no_backup \
        2>&1 | tee "$LOG_DIR/grid_convert.log"
    echo "[lab] Grid conversion done."
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
