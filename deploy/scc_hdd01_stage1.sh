#!/bin/bash
# Stage 1 on scc-hdd-01 — 80-core CPU, /data5 is local disk.
# SSH to scc-hdd-01 and run:  bash ~/iome/deploy/scc_hdd01_stage1.sh
# Or launch remotely from ws02:
#   ssh scc-hdd-01 "bash ~/iome/deploy/scc_hdd01_stage1.sh"

set -euo pipefail

CACHE=/data5/iome_cache
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
PYTHON=$HOME/miniconda3/bin/python
# Limit to 40 threads so the machine stays responsive
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "[hdd01] $(hostname) — CPU training"
echo "[hdd01] ckpt: $CKPT_DIR"
echo "[hdd01] log : $LOG_DIR/stage1-hdd01.log"

# Run grid conversion first if still needed (local disk → fast)
SD_SHAPE=$($PYTHON -c "import numpy as np; from pathlib import Path; f=next(Path('$CACHE/superdarn').glob('*_sd.npy'),None); print(np.load(f,mmap_mode='r').shape[1] if f else 0)" 2>/dev/null || echo 0)
if [[ "$SD_SHAPE" == "120" ]]; then
    echo "[hdd01] Converting 120x120 → 180x360 (local disk) ..."
    $PYTHON "$IOME_DIR/scripts/convert_grid_120_to_180360.py" \
        --cache_root "$CACHE" --workers 40 --no_backup \
        2>&1 | tee "$LOG_DIR/grid_convert_hdd01.log"
    echo "[hdd01] Conversion done."
fi

nohup $PYTHON "$IOME_DIR/scripts/train_stage1.py" \
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
    --num_workers 16                    \
    --accelerator cpu                   \
    --devices     1                     \
    --precision   32                    \
    --p_mod_drop  0.3                   \
    --wandb_project iome                \
    > "$LOG_DIR/stage1-hdd01.log" 2>&1 &

PID=$!
echo "[hdd01] PID $PID"
echo "$PID" > "$LOG_DIR/stage1-hdd01.pid"
echo "[hdd01] tail -f $LOG_DIR/stage1-hdd01.log"
