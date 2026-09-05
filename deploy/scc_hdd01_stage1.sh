#!/bin/bash
# Stage 1 on scc-hdd-01 — 80-core CPU, /data is local disk.
# Runs two parallel jobs to saturate all 80 cores:
#   Run A: batch=32, 40 workers, p_drop=0.3, tau=0.10  (main)
#   Run B: batch=16, 20 workers, p_drop=0.5, tau=0.07  (probe)

set -euo pipefail

CACHE=/data/iome_cache
SPLITS=$CACHE/splits
IOME_DIR="$HOME/iome"
LOG_DIR=$CACHE/train_logs
CKPT_A=$CACHE/ckpts/stage1-hdd01a
CKPT_B=$CACHE/ckpts/stage1-hdd01b

mkdir -p "$LOG_DIR" "$CKPT_A" "$CKPT_B"

export WANDB_API_KEY=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
export WANDB_ENTITY=st7ma784
export WANDB_PROJECT=iome
export WANDB_MODE=online
export PYTHONPATH=$IOME_DIR/src:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
PYTHON=$HOME/miniconda3/bin/python
# Each run gets 4 OMP/MKL threads; 2 runs × 4 = 8 + 80 data workers leaves room
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "[hdd01] $(hostname) — dual CPU stage-1"

# ------------------------------------------------------------------
# Run grid conversion first if still needed (local disk → fast)
# ------------------------------------------------------------------
SD_SHAPE=$($PYTHON -c "
import numpy as np; from pathlib import Path
f = next(Path('$CACHE/superdarn').glob('*_sd.npy'), None)
print(np.load(f, mmap_mode='r').shape[1] if f else 0)
" 2>/dev/null || echo 0)
if [[ "$SD_SHAPE" == "120" ]]; then
    echo "[hdd01] Converting 120×120 → 180×360 (local disk) ..."
    $PYTHON "$IOME_DIR/scripts/convert_grid_120_to_180360.py" \
        --cache_root "$CACHE" --workers 40 --no_backup \
        2>&1 | tee "$LOG_DIR/grid_convert_hdd01.log"
    echo "[hdd01] Conversion done."
fi

# ------------------------------------------------------------------
# Run A — main hyperparameters (40 data workers)
# ------------------------------------------------------------------
nohup $PYTHON -u "$IOME_DIR/scripts/train_stage1.py" \
    --splits_dir  "$SPLITS"             \
    --cache_sd    "$CACHE/superdarn"    \
    --cache_smag  "$CACHE/supermag"     \
    --cache_tec   "$CACHE/tec"          \
    --cache_dmsp  "$CACHE/dmsp"         \
    --ckpt_dir    "$CKPT_A"             \
    --omni_dir    "$CACHE/omni"         \
    --stats_dir   "$SPLITS"             \
    --batch_size  32                    \
    --max_steps   50000                 \
    --num_workers 40                    \
    --accelerator cpu                   \
    --devices     1                     \
    --precision   32                    \
    --p_mod_drop  0.3                   \
    --wandb_project iome                \
    --wandb_name  hdd01a-drop0.3-tau0.1 \
    > "$LOG_DIR/stage1-hdd01a.log" 2>&1 &
PID_A=$!
echo "[hdd01] Run A PID $PID_A  (batch=32, workers=40, drop=0.3, tau=0.10)"
echo "$PID_A" > "$LOG_DIR/stage1-hdd01a.pid"

# ------------------------------------------------------------------
# Run B — higher dropout, lower temperature (20 data workers)
# ------------------------------------------------------------------
nohup $PYTHON -u "$IOME_DIR/scripts/train_stage1.py" \
    --splits_dir  "$SPLITS"             \
    --cache_sd    "$CACHE/superdarn"    \
    --cache_smag  "$CACHE/supermag"     \
    --cache_tec   "$CACHE/tec"          \
    --cache_dmsp  "$CACHE/dmsp"         \
    --ckpt_dir    "$CKPT_B"             \
    --omni_dir    "$CACHE/omni"         \
    --stats_dir   "$SPLITS"             \
    --batch_size  16                    \
    --max_steps   50000                 \
    --num_workers 20                    \
    --accelerator cpu                   \
    --devices     1                     \
    --precision   32                    \
    --p_mod_drop  0.5                   \
    --tau         0.07                  \
    --wandb_project iome                \
    --wandb_name  hdd01b-drop0.5-tau0.07 \
    > "$LOG_DIR/stage1-hdd01b.log" 2>&1 &
PID_B=$!
echo "[hdd01] Run B PID $PID_B  (batch=16, workers=20, drop=0.5, tau=0.07)"
echo "$PID_B" > "$LOG_DIR/stage1-hdd01b.pid"

echo "[hdd01] logs: $LOG_DIR/stage1-hdd01{a,b}.log"
