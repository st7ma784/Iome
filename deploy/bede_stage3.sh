#!/bin/bash
# Stage 3: End-to-end fine-tuning.
# Submit from login:   ghbatch deploy/bede_stage3.sh
# Submit from ghlogin: sbatch  deploy/bede_stage3.sh

#SBATCH --account=bdlan12
#SBATCH --job-name=iome-stage3
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=/nobackup/projects/bdlan12/smander3/iome_logs/stage3_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mt.alpha@gmail.com

PROJECT=bdlan12
USERNAME=smander3
CONDA_ROOT=/nobackup/projects/bdlan12/$USERNAME/aarch64/miniconda
IOME_ENV=/nobackup/projects/bdlan12/conda/iome

NOBACKUP=/nobackup/projects/$PROJECT/$USERNAME
IOME_DIR=$NOBACKUP/iome
CACHE_SD=$NOBACKUP/iome_data/superdarn
CACHE_SMAG=$NOBACKUP/iome_data/supermag
CACHE_TEC=$NOBACKUP/iome_data/tec
CACHE_DMSP=$NOBACKUP/iome_data/dmsp
CACHE_OMNI=$NOBACKUP/iome_data/omni
SPLITS=$NOBACKUP/iome_data/splits

STAGE2_CKPT=$(ls -t "$NOBACKUP/iome_ckpts/stage2/"*.ckpt 2>/dev/null | head -1)
if [[ -z "$STAGE2_CKPT" ]]; then
    echo "ERROR: no Stage 2 checkpoint found under $NOBACKUP/iome_ckpts/stage2/"
    exit 1
fi
echo "[bede] loading Stage 2 ckpt: $STAGE2_CKPT"

mkdir -p "$NOBACKUP/iome_logs" "$NOBACKUP/iome_ckpts/stage3"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$IOME_ENV"

if [[ "$CONDA_PREFIX" != "$IOME_ENV" ]]; then
    echo "ERROR: conda activate failed (CONDA_PREFIX='$CONDA_PREFIX')"
    exit 1
fi

python -c "import triton" 2>/dev/null \
    || { echo "[bede] installing triton..."; pip install triton --quiet; }

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY=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
export WANDB_ENTITY=st7ma784
export WANDB_MODE=online
export PYTHONPATH=$IOME_DIR/src:$PYTHONPATH

python "$IOME_DIR/scripts/train_stage3.py" \
    --stage2_ckpt "$STAGE2_CKPT"               \
    --splits_dir  "$SPLITS"                    \
    --cache_sd    "$CACHE_SD"                  \
    --cache_smag  "$CACHE_SMAG"                \
    --cache_tec   "$CACHE_TEC"                 \
    --cache_dmsp  "$CACHE_DMSP"                \
    --ckpt_dir    "$NOBACKUP/iome_ckpts/stage3" \
    --omni_dir    "$CACHE_OMNI"                \
    --stats_dir   "$SPLITS"                    \
    --batch_size  32                           \
    --max_steps   30000                        \
    --num_workers 16                           \
    --p_mod_drop  0.15                         \
    --wandb_project iome
