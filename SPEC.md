# Iome — Multi-Modal Ionospheric Fusion: Software Specification

## 1. Overview

This project is a proof-of-concept multi-modal ionospheric data fusion system. Three independent observing networks — SuperDARN (radar convection), SuperMAG (ground magnetometers), and GPS-TEC (ionospheric electron content) — each measure a different projection of the same underlying ionospheric state. The hypothesis is that a shared latent representation can be learned from all three simultaneously, each modality acting as a cross-check on the others, resulting in better forecasting than any single source alone.

The architecture follows the three-stage self-supervised pipeline developed in the design conversations: contrastive alignment of encoders → reconstruction training → latent dynamics for forecasting. Everything is written in PyTorch Lightning and deployed on Bede (Durham HPC, Grace-Hopper H100 nodes).

---

## 2. Repository Layout

```
Iome/
├── src/
│   └── iome/
│       ├── data/
│       │   ├── luna.py            # LUNA/SMB → MinIO transfer helpers (one per source)
│       │   ├── superdarn.py       # SuperDARN dataset & normalisation
│       │   ├── supermag.py        # SuperMAG dataset & normalisation
│       │   ├── tec.py             # GPS-TEC / IONEX dataset & normalisation
│       │   └── datamodule.py      # TriModalDataModule (combined LightningDataModule)
│       ├── models/
│       │   ├── encoders.py        # SuperDARNEncoder, SuperMAGEncoder, TECEncoder
│       │   ├── decoders.py        # matching decoders
│       │   ├── dynamics.py        # FiLM-conditioned latent dynamics
│       │   ├── losses.py          # InfoNCE, reconstruction, dynamics losses
│       │   └── fusion.py          # UnifiedIonosphereModel (assembles all parts)
│       ├── train/
│       │   ├── stage1.py          # Stage1ContrastiveModule (LightningModule)
│       │   ├── stage2.py          # Stage2DynamicsModule
│       │   └── stage3.py          # Stage3FinetuneModule
│       └── viz/
│           └── logging.py         # wandb artifact helpers (panels, latent plots, etc.)
├── scripts/
│   ├── luna_to_minio_supermag.py  # SMB → MinIO for SuperMAG
│   └── luna_to_minio_tec.py       # SMB → MinIO for TEC
├── deploy/
│   ├── bede_stage1.sh             # SLURM: contrastive pretraining
│   ├── bede_stage2.sh             # SLURM: dynamics + reconstruction
│   └── bede_stage3.sh             # SLURM: end-to-end fine-tuning
├── .claude/
│   └── commands/
│       └── bede.md                # rsync + submit reference (same as SuperDARN)
└── pyproject.toml
```

---

## 3. Data Sources

### 3.1 Common Grid

All three modalities are co-registered to the same polar grid used by SuperDARN:

- **Projection**: azimuthal equidistant, centred on the magnetic pole
- **Coordinate system**: magnetic latitude (mlat) × magnetic local time (MLT)
- **Extent**: 50–90° mlat (poleward of the convection boundary)
- **Grid resolution**: 120 × 120 pixels (matching the existing `g120` SuperDARN dataset)
- **Time cadence**: 2-minute epochs (SuperDARN's natural cadence; other sources interpolated)

### 3.2 SuperDARN

- **LUNA path**: `/PY/SPP/data/SuperDARN/cnvmap/`
- **Format**: RST cnvmap binary → processed via existing `pydarnio` pipeline
- **Channels** (already implemented, 6-channel `(6, 120, 120)` float32):

| Ch | Name | Description |
|----|------|-------------|
| 0 | `obs_vel_north` | SH-fitted northward E×B drift in radar-covered cells (m/s) |
| 1 | `obs_vel_east` | SH-fitted eastward E×B drift (m/s) |
| 2 | `model_vel_north` | Background Weimer/TS96 northward drift (m/s) |
| 3 | `model_vel_east` | Background eastward drift (m/s) |
| 4 | `soft_occ` | Radar coverage: tanh(splat_weight / median), ∈ [0,1] |
| 5 | `boundary_dist` | Signed mlat distance from Heppner-Maynard boundary (°) |

- **DataModule**: adapt `DatasetFromMinioBucket` from SuperDARN directly.

### 3.3 SuperMAG

- **LUNA path**: `/PY/SPP/data/SuperMAG/`
- **Format**: per-station CSV/HDF5, 1-minute resolution
- **Processing**:
  1. Read station list with (mlat, mlon) positions
  2. Aggregate 2-minute windows (mean of 2 samples)
  3. Gaussian-splat dBn, dBe perturbations onto the 120×120 polar grid (same `_splat` kernel as SuperDARN, σ=2.0 pixels since magnetometers are sparser)
  4. Compute `soft_occ` = tanh(coverage_weight / median_weight)
- **Channels** (`(3, 120, 120)` float32):

| Ch | Name | Description |
|----|------|-------------|
| 0 | `dBn` | Northward magnetic field perturbation (nT) |
| 1 | `dBe` | Eastward magnetic field perturbation (nT) |
| 2 | `soft_occ` | Station coverage confidence ∈ [0,1] |

### 3.4 GPS-TEC

- **LUNA path**: `/PY/SPP/data/TEC/IONEX/`
- **Format**: IONEX files (standard ITU format; one file per day, maps every 15 min or 5 min depending on product)
- **Processing**:
  1. Parse IONEX → geographic (lat, lon) TEC maps
  2. Convert geographic → magnetic (mlat, mlon) using AACGM or similar
  3. Interpolate to 2-minute epochs (linear between available maps)
  4. Regrid onto the 120×120 azimuthal equidistant projection
- **Channels** (`(2, 120, 120)` float32):

| Ch | Name | Description |
|----|------|-------------|
| 0 | `VTEC` | Vertical total electron content (TECU) |
| 1 | `dVTEC_dt` | Time derivative of VTEC (TECU/2min), finite diff of adjacent maps |

### 3.5 OMNI Solar Wind

- **Source**: same OMNI pipeline already in SuperDARN (NASA OMNIWeb)
- **Features** (`u_dim = 8`): Bx, By, Bz, |B|, Vx, density, Kp, coupling function ε
- **Role**: conditioning vector for the dynamics model; not an encoder input

---

## 4. Data Pipeline

### 4.1 LUNA → MinIO Transfer (`scripts/luna_to_minio_*.py`)

Follow the existing `lunaToMinio.py` pattern (SMBConnection → ThreadPoolExecutor → MinIO.put_object). One script per source, parameterised by date range and LUNA share path. Run once on a login node before training.

```
scripts/luna_to_minio_supermag.py --share /PY/SPP/data/SuperMAG/ --start 2012-01 --end 2023-12
scripts/luna_to_minio_tec.py      --share /PY/SPP/data/TEC/IONEX/ --start 2012-01 --end 2023-12
```

### 4.2 TriModalDataModule (`src/iome/data/datamodule.py`)

A single `LightningDataModule` that loads and synchronises all three modalities.

```python
class TriModalDataModule(LightningDataModule):
    """
    Returns dicts keyed by modality for each 2-minute epoch.

    Batch shape:
      batch["sd"]    : (B, 6, 120, 120)  float32
      batch["smag"]  : (B, 3, 120, 120)  float32
      batch["tec"]   : (B, 2, 120, 120)  float32
      batch["omni"]  : (B, 8)            float32
      batch["mask"]  : (B, 3)            bool   (which modalities are available)
      batch["t"]     : (B,)              int64  (unix epoch of window start)
    ```
    """
```

**Implementation notes**:
- Index by (year, doy, hhmm) tuples; inner join across modalities to find epochs where all three are available; store missing-modality masks for graceful degradation.
- Same DataLoader settings as Bede-tested SuperDARN:
  ```python
  DataLoader(dataset, num_workers=16,
             multiprocessing_context='forkserver',
             persistent_workers=False,
             prefetch_factor=2,
             pin_memory=False)   # GH100 unified memory: pin_memory=False
  ```
- `prepare_data()`: download from MinIO → local `/tmp` NVMe on Bede if available; otherwise stream from NFS.
- `setup()`: 80/20 train/val split by date (split at a year boundary, not randomly, to avoid leakage through temporal autocorrelation).
- Normalisation: compute per-channel mean/std on training split, cache as `.npz` alongside data hash (same `dataset_hash` pattern as SuperDARN).

---

## 5. Model Architecture

### 5.1 Shared Latent Dimension

`D = 256` (tunable). All encoders and decoders project to/from this dimension.

### 5.2 Per-Modality Encoders (`src/iome/models/encoders.py`)

Each encoder is a `nn.Module` with the signature `forward(x) -> z` where `z: (B, D)`.

**SuperDARNEncoder** — reuse the existing Pangu patch-embed + 2 EarthSpecificBlocks, then global-average-pool:
```python
class SuperDARNEncoder(nn.Module):
    # PatchEmbed2D(6, embed_dim=128, patch_size=4)
    # 2x EarthSpecificBlock(embed_dim, n_heads=8)
    # AdaptiveAvgPool → Linear(embed_dim, D)
```

**SuperMAGEncoder** — lighter; magnetometer coverage is sparser so a shallower network suffices:
```python
class SuperMAGEncoder(nn.Module):
    # Conv2d(3→64, k=7, pad=3) → GELU → Conv2d(64→128, k=5) → GELU
    # Conv2d(128→256, k=3, stride=2) × 2
    # AdaptiveAvgPool2d(1) → Flatten → Linear(256, D)
```

**TECEncoder** — TEC maps are spatially smooth; a small ResNet-style stack:
```python
class TECEncoder(nn.Module):
    # Conv2d(2→64, k=7) → LayerNorm → GELU
    # 3x ResBlock(64)
    # Conv2d(64→128, stride=2) → 2x ResBlock(128)
    # AdaptiveAvgPool2d(1) → Flatten → Linear(128, D)
```

All encoders are registered as `nn.ModuleDict` inside `UnifiedIonosphereModel`:
```python
self.encoders = nn.ModuleDict({
    "sd":   SuperDARNEncoder(...),
    "smag": SuperMAGEncoder(...),
    "tec":  TECEncoder(...),
})
```

### 5.3 Shared Latent Fusion

Mean fusion (Stage 1 & 2); learned cross-attention fusion (Stage 3 optional):
```python
def fuse(z_dict, mask):
    # z_dict: {"sd": (B,D), "smag": (B,D), "tec": (B,D)}
    # mask:   (B, 3) bool — which views are available
    stacked = torch.stack(list(z_dict.values()), dim=1)  # (B, V, D)
    stacked = stacked * mask.unsqueeze(-1).float()
    count   = mask.float().sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
    return stacked.sum(dim=1) / count  # (B, D)
```

### 5.4 Latent Dynamics (`src/iome/models/dynamics.py`)

FiLM-conditioned MLP (same pattern as `FiLMLayer` in SuperDARN):
```python
class LatentDynamics(nn.Module):
    # u_embed: Linear(u_dim, hidden) → GELU → Linear(hidden, D)
    # core:    Linear(D, 4*D) → GELU → Linear(4*D, D)
    # FiLM:    (γ, β) = MLP(u_embed) → z * (1+γ) + β
    # output:  z_{t+1} = z_t + FiLM(core(z_t), u_t)  [residual]
```

### 5.5 Per-Modality Decoders (`src/iome/models/decoders.py`)

Each decoder maps `z: (B, D)` back to the modality's grid:
```python
class SuperDARNDecoder(nn.Module):
    # Linear(D, embed_dim * n_patches^2) → reshape → PatchRecovery2D → (B, 6, 120, 120)

class SuperMAGDecoder(nn.Module):
    # Linear(D, 256) → reshape (B, 256, 1, 1) → ConvTranspose2d upsampling → (B, 3, 120, 120)

class TECDecoder(nn.Module):
    # Linear(D, 128) → reshape → ConvTranspose2d stack → (B, 2, 120, 120)
```

Zero-init the final conv/linear in each decoder (same reasoning as Pangu: start from persistence, not random noise).

---

## 6. Loss Functions (`src/iome/models/losses.py`)

### 6.1 InfoNCE Contrastive Loss

Positives: different encoder views of the **same** timestep.
Negatives: encodings from **different** timesteps in the batch.

```python
def infonce_loss(z_shared, z_views, tau=0.1):
    """
    z_shared: (B, D) — mean-fused latent per timestep
    z_views:  list of V tensors, each (B, D) — per-encoder latents
    """
    z_norm = F.normalize(z_shared, dim=1)          # (B, D)
    sim_matrix = z_norm @ z_norm.T / tau            # (B, B)  — cross-time similarity

    z_views_norm = [F.normalize(z, dim=1) for z in z_views]
    stacked = torch.stack(z_views_norm, dim=0)      # (V, B, D)
    sim_pos = torch.einsum('bd,vbd->vb', z_norm, stacked) / tau  # (V, B)

    numerator   = torch.exp(sim_pos).sum(dim=0)     # (B,)
    denominator = torch.exp(sim_matrix).sum(dim=1)  # (B,)
    return (-torch.log(numerator / denominator.clamp(min=1e-8))).mean()
```

### 6.2 Reconstruction Loss

Per-modality weighted MSE, masking unobserved cells using the `soft_occ` channel:
```python
L_recon = sum_i  w_i * weighted_mse(decoder_i(z), y_i, occ_mask_i)
```
Weights: `w_sd=1.0, w_smag=0.8, w_tec=0.6` (SuperDARN is primary; TEC is smoothest and easiest).

### 6.3 Dynamics Loss

```python
L_dyn = MSE(z_{t+1}_pred, z_{t+1}_actual.detach())
```
Note: `.detach()` on the target so the dynamics loss does not back-propagate into the encoders during Stage 2 frozen-encoder runs.

### 6.4 Full Objectives per Stage

| Stage | Loss |
|-------|------|
| 1 | L_cont + 0.5 × L_recon |
| 2 | L_recon + 1.0 × L_dyn + 0.1 × L_cont |
| 3 | L_recon + 1.0 × L_dyn + 0.05 × L_cont |

---

## 7. Training Stages (LightningModules)

Each stage is a separate `LightningModule` that loads a checkpoint from the previous stage.

### 7.1 Stage 1 — Contrastive Pretraining (`src/iome/train/stage1.py`)

**Trains**: all three encoders + all three decoders  
**Frozen**: dynamics model (not instantiated)  
**Goal**: shared latent where same-time views cluster, different times separate

```python
class Stage1ContrastiveModule(LightningModule):
    def training_step(self, batch, batch_idx):
        z_dict = {k: self.encoders[k](batch[k]) for k in ["sd","smag","tec"]}
        z_shared = fuse(z_dict, batch["mask"])
        z_views  = list(z_dict.values())

        l_cont  = infonce_loss(z_shared, z_views, tau=self.hparams.tau)
        l_recon = sum(w * recon_loss(self.decoders[k](z_shared), batch[k])
                      for k, w in zip(["sd","smag","tec"], [1.0, 0.8, 0.6]))
        loss = l_cont + self.hparams.lambda_recon * l_recon
        self.log_dict({"train/l_cont": l_cont, "train/l_recon": l_recon, "train/loss": loss})
        return loss

    def configure_optimizers(self):
        # AdamW, lr=3e-4, weight decay 0.05, cosine with 500-step warmup
```

**Convergence signal**: `train/l_cont` drops; alignment score (avg cosine sim of same-time pairs) rises above 0.7.  
**Checkpoint**: save `encoders` + `decoders` state dicts.

### 7.2 Stage 2 — Dynamics + Reconstruction (`src/iome/train/stage2.py`)

**Loads**: Stage 1 encoder + decoder weights  
**Trains**: dynamics model + decoders; encoders at 10× lower LR (light fine-tune)  
**Goal**: forecastable latent; accurate per-modality reconstruction

```python
class Stage2DynamicsModule(LightningModule):
    def training_step(self, batch, batch_idx):
        # batch contains consecutive pairs (t, t+1)
        z_t   = fuse({k: self.encoders[k](batch["x"][k]) for k in MODALITIES}, batch["mask_x"])
        z_tp1 = fuse({k: self.encoders[k](batch["y"][k]) for k in MODALITIES}, batch["mask_y"])
        z_tp1_pred = self.dynamics(z_t, batch["omni"])

        l_dyn   = F.mse_loss(z_tp1_pred, z_tp1.detach())
        l_recon = sum(w * recon_loss(self.decoders[k](z_t), batch["x"][k])
                      for k, w in WEIGHTS.items())
        l_cont  = infonce_loss(z_t, [enc(batch["x"][k]) for k, enc in ...], tau=0.2)
        loss = l_recon + self.hparams.lambda_dyn * l_dyn + 0.1 * l_cont
```

**Convergence signal**: `train/l_dyn` drops; per-modality RMSE on val set converges.

### 7.3 Stage 3 — End-to-End Fine-tuning (`src/iome/train/stage3.py`)

**Loads**: Stage 2 checkpoint  
**Trains**: everything (encoders, dynamics, decoders) at lower LR (1e-4)  
**Goal**: remove encoder/dynamics mismatch; optimise for multi-step forecast horizon  
**Loss**: same as Stage 2, `lambda_cont=0.05`

Optionally add a multi-step rollout loss over k=4 timesteps (same TBPTT pattern as `rl_forecast`).

---

## 8. Bede Deployment

### 8.1 Conda Environment (build once on `ghlogin`)

```bash
ssh smander3@bede.dur.ac.uk
ghlogin   # aarch64 session — MUST build env here, not on login node (ppc64le)
source /nobackup/projects/bdlan12/smander3/aarch64/miniconda/etc/profile.d/conda.sh
conda create -n iome python=3.11 -y
conda activate /nobackup/projects/bdlan12/conda/iome
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pytorch-lightning wandb numpy scipy pysmb minio
pip install aacgmv2   # magnetic coordinate conversion for TEC regridding
python -c "import torch; print(torch.cuda.is_available())"  # must print True
```

### 8.2 Paths

```
PROJECT=bdlan12
USERNAME=smander3
NOBACKUP=/nobackup/projects/bdlan12/smander3
IOME_REPO=$NOBACKUP/iome
IOME_DATA=$NOBACKUP/iome_data      # synced from LUNA via MinIO
IOME_LOGS=$NOBACKUP/iome_logs
IOME_ENV=/nobackup/projects/bdlan12/conda/iome
CONDA_ROOT=$NOBACKUP/aarch64/miniconda
```

### 8.3 SLURM Script Pattern (`deploy/bede_stage1.sh`)

```bash
#!/bin/bash
#SBATCH --account=bdlan12
#SBATCH --job-name=iome-stage1
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=/nobackup/projects/bdlan12/smander3/iome_logs/stage1_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=st7ma784@gmail.com

PROJECT=bdlan12
USERNAME=smander3
CONDA_ROOT=/nobackup/projects/bdlan12/$USERNAME/aarch64/miniconda
IOME_ENV=/nobackup/projects/bdlan12/conda/iome

NOBACKUP=/nobackup/projects/$PROJECT/$USERNAME
mkdir -p "$NOBACKUP/iome_logs"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$IOME_ENV"
if [[ "$CONDA_PREFIX" != "$IOME_ENV" ]]; then
    echo "ERROR: conda activate failed"; exit 1
fi

python -c "import triton" 2>/dev/null \
    || pip install triton --quiet

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SUPERDARN_PIN_MEMORY=0       # GH100: unified memory, no pinning needed
export WANDB_MODE=online

cd "$NOBACKUP/iome" || exit 1

python -m iome.train.stage1 \
    --data_dir $NOBACKUP/iome_data \
    --log_dir  $NOBACKUP/iome_logs \
    --latent_dim 256 \
    --batch_size 512 \
    --num_workers 16 \
    --precision bf16-mixed \
    --max_epochs 50 \
    --tau 0.1 \
    --lambda_recon 0.5 \
    --wandb
```

`bede_stage2.sh` and `bede_stage3.sh` differ only in `--job-name`, `--finetune_from`, and the Python module invoked.

### 8.4 Sync and Submit (`.claude/commands/bede.md`)

```bash
# Sync code
rsync -av --exclude='__pycache__' --exclude='*.pyc' \
    /home/user/Iome/src/iome/ \
    smander3@bede.dur.ac.uk:/nobackup/projects/bdlan12/smander3/iome/src/iome/
rsync -av /home/user/Iome/deploy/ \
    smander3@bede.dur.ac.uk:/nobackup/projects/bdlan12/smander3/iome/deploy/

# Submit
ssh smander3@bede.dur.ac.uk "cd /nobackup/projects/bdlan12/smander3/iome && ghbatch deploy/bede_stage1.sh"

# Monitor
ssh smander3@bede.dur.ac.uk "squeue -u smander3"
ssh smander3@bede.dur.ac.uk "tail -f /nobackup/projects/bdlan12/smander3/iome_logs/stage1_JOB_ID.log"
```

---

## 9. Visualisation & W&B Logging

All logging follows the pattern in `Pangu._log_validation_diagnostics_*`: accumulate tensors on GPU each step, convert to numpy once per epoch in `on_validation_epoch_end`.

### 9.1 Per-Modality Reconstruction Panels

For each modality at each validation epoch, log a 4-row comparison panel:

```
Row 0: Input (x_t)
Row 1: Target (x_{t+1})
Row 2: Prediction (decoder(z_t))
Row 3: |Error| = |pred - target|
```

One panel per modality per epoch → W&B image artifacts under `stage1/val_panel_sd`, `stage1/val_panel_smag`, `stage1/val_panel_tec`.

### 9.2 "What the Encoder Ignores" — Residual Maps

The residual `x - decoder(encoder(x))` shows what each encoder discards (its "noise definition"). Log as a separate panel per modality:

```python
def log_residual_panels(self, batch, batch_idx):
    for k in ["sd", "smag", "tec"]:
        z  = self.encoders[k](batch[k])
        x_hat = self.decoders[k](z)
        residual = (batch[k] - x_hat).abs()
        # log residual mean per channel as a bar chart
        # log residual spatial map as wandb.Image
```

This is the primary interpretability artifact: a modality's encoder will learn to ignore measurement noise, sparsity patterns, and artefacts that the other modalities do not corroborate. The residual map makes this explicit.

### 9.3 Contrastive Diagnostics

Log per epoch:
- **Alignment** (↑ good): mean cosine similarity of z_shared pairs from the same timestep across different encoders
- **Uniformity** (↑ good): negative log of mean pairwise Gaussian kernel over all z_shared (Wang & Isola 2020 metric), indicates how well the latent space is filled
- **Confusion matrix**: for a held-out batch, which timesteps does the contrastive classifier confuse? (Log as a wandb.Table of (true_t, predicted_t) for hard negatives)

```python
"stage1/alignment":   cosine_sim_same_time.mean()
"stage1/uniformity":  -torch.log(torch.exp(-2 * pairwise_sq_dist).mean())
```

### 9.4 Latent Space Visualisation

Once per N epochs, compute UMAP of a few thousand z_shared vectors on the validation set. Colour by:
- Timestamp (should show temporal clustering)
- Kp index (should show geomagnetic activity gradient)
- Season (should show seasonal modulation)

Log as a W&B scatter artifact. This reveals whether the latent organises physically meaningful structure.

### 9.5 Cross-Modal Reconstruction

Decode one modality's encoding through another modality's decoder:

```python
z_from_tec = self.encoders["tec"](batch["tec"])
sd_from_tec = self.decoders["sd"](z_from_tec)   # SD predicted from TEC only
```

Compare `sd_from_tec` to the actual SuperDARN observation. Measures how much of the SD signal is recoverable from TEC alone — i.e., how well the latent captures the shared ionospheric state. Log RMSE per channel as `diag/cross_sd_from_tec`, etc.

### 9.6 Lead-Time Skill Curve (Stage 2+)

Roll the dynamics model forward k steps and measure forecast RMSE against each modality's observations, as a function of k:

```python
for k in range(1, 13):   # 1 to 12 steps = 2 to 24 minutes ahead
    z_pred_k = z_0
    for _ in range(k):
        z_pred_k = self.dynamics(z_pred_k, omni[step])
    for mod in ["sd", "smag", "tec"]:
        rmse_k_mod = rmse(self.decoders[mod](z_pred_k), y_k[mod])
```

Log as `stage2/lead_time_sd`, etc. (one W&B Table per modality, columns: horizon_minutes, rmse).

### 9.7 W&B Run Naming Convention

```
iome-stage1-YYYYMMDD-HHMMSS
iome-stage2-from-<stage1_run_id>
iome-stage3-from-<stage2_run_id>
```

Tag all runs with `{stage, latent_dim, tau, dataset}` so they can be grouped in W&B.

---

## 10. Key Design Decisions & Rationale

| Decision | Rationale |
|----------|-----------|
| Mean fusion (not concat) | Allows graceful missing-modality handling; concat requires imputation |
| Separate encoder per modality | Modalities have incompatible spatial structure (sparse/dense/smooth) |
| Zero-init decoder output | Persistence forecast from epoch 1; avoids large early gradients from random outputs |
| 80/20 date-split (not random) | Temporal autocorrelation means random splits leak; adjacent days are highly correlated |
| `pin_memory=False` on Bede GH | GH100 has unified CPU-GPU address space; pinning counts against the same VRAM budget |
| `forkserver` multiprocessing | Avoids inheriting stale NFS file descriptors from forked workers |
| Residual dynamics (`z + f(z,u)`) | Initialises to identity (zero output head); network learns corrections not full state |
| `tau=0.1` for InfoNCE | Standard value; lower = harder negatives, faster convergence but training instability risk |

---

## 11. Development Order

1. **`scripts/luna_to_minio_supermag.py`** — verify SuperMAG data is accessible on LUNA and can be staged to MinIO.
2. **`scripts/luna_to_minio_tec.py`** — same for IONEX TEC files.
3. **`src/iome/data/supermag.py`** + **`tec.py`** — implement grid projection and channel extraction; unit-test against SuperDARN's `_records_to_grid` as a reference.
4. **`src/iome/data/datamodule.py`** — `TriModalDataModule`; verify time alignment and missing-modality masks on a local subset.
5. **`src/iome/models/encoders.py`** + **`decoders.py`** — implement all six modules; verify shapes with a random tensor smoke test.
6. **`src/iome/models/losses.py`** — implement and unit-test InfoNCE (check that same-time positives do get lower loss than different-time negatives).
7. **`src/iome/train/stage1.py`** — Stage1 module; run a 5-minute overfit test locally to verify gradients flow through all encoders.
8. **`deploy/bede_stage1.sh`** — deploy Stage 1 on Bede; verify alignment metric rises and uniformity stabilises.
9. **`src/iome/train/stage2.py`** → **`deploy/bede_stage2.sh`**
10. **`src/iome/train/stage3.py`** → **`deploy/bede_stage3.sh`**
11. **`src/iome/viz/logging.py`** — residual panels, latent UMAP, cross-modal reconstruction artifacts.
