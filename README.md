# Iome — Ionospheric Multimodal Encoder

Self-supervised learning of a shared ionospheric state representation from
SuperDARN, SuperMAG, GPS-TEC, and DMSP particle precipitation — with a
solar-wind-conditioned dynamics model for short-range forecasting.

---

## The regression-to-mean problem

Every statistical model of the high-latitude ionosphere faces the same trap.

The ionosphere spends most of its time in quiet conditions. SuperDARN coverage
is sparse (typically 8% of the polar cap per 2-minute scan), so when you fit a
spherical-harmonic expansion to the radar echoes, the underdetermined system
reverts to its prior — Weimer 2005 climatology, i.e. the statistical mean over
thousands of quiet-time maps. TEC IONEX maps do the same: a smooth spherical
basis cannot represent sharp storm-time gradients, so it returns to the
zonal-mean background. SuperMAG station coverage has gaps; Gaussian-splatted
station data reverts to zero where no stations exist.

The result is that every conventional product performs well by one metric —
low mean-squared error on the full dataset — and does so by predicting
approximately the same quiet-time pattern for >90% of epochs. During the 10%
of time that actually matters scientifically (substorms, sudden commencements,
geomagnetic storms), the mean-reversion is at its worst precisely because those
events deviate most from the mean.

**An autoencoder trained with MSE reconstruction has exactly the same problem.**
Given sparse, noisy inputs, the optimal MSE prediction in unseen regions is the
conditional mean — which is the climatological background. The network learns
to output "average ionosphere" not because it lacks capacity, but because that
is the correct answer to the wrong question.

---

## Why contrastive learning changes the question

The key insight is that **discriminability is the objective, not fidelity**.

Instead of asking "what does this map look like?", a contrastive model asks
"is this map the same ionospheric state as that one?" Two SuperDARN maps from
a quiet interval 10 minutes apart should produce similar latent vectors. A
quiet-time map and a substorm-onset map should produce very different ones.
A model that outputs the same latent for every quiet-time epoch cannot satisfy
this — even though quiet epochs are identical in MSE terms.

Formally, we use the CLIP / NT-Xent loss:

```
L = CE( z @ z.T / τ,  I )
```

where `z` is a batch of normalised latent vectors, `τ` is a temperature
(0.07), and the target is the identity matrix — each sample should be most
similar to itself and dissimilar from all other samples in the batch. The
cross-entropy formulation creates a softmax competition: if the model
produces the same vector for two different maps, it cannot distinguish them,
and the off-diagonal terms penalise that failure sharply. This is equivalent
to maximising a lower bound on the mutual information between the input and
its latent representation (van den Oord et al. 2018).

The cross-entropy form has an additional property relevant to geophysics: the
softmax encourages **discrete, compositional representations** rather than
smooth interpolations. The ionosphere is driven by a relatively small number
of physical regimes — substorm growth/expansion/recovery, convection
enhancement, particle precipitation boundaries, storm sudden commencement.
These are not points on a continuum; they are qualitatively different system
states. CE loss biases the model toward learning regime boundaries rather than
means within regimes. This is the grokking argument: the model is pushed to
learn the underlying classification (what regime is this epoch in?) beyond
the training distribution, rather than interpolating between memorised
examples.

Reconstruction is retained as a secondary loss — but it only prevents the
latent from becoming geometrically arbitrary, not to drive the primary signal.

---

## The temporal-scale problem and Phase 0

There is a second obstacle: the cause-effect chain in magnetospheric physics
operates on timescales of 20–60 minutes, much longer than a single 2-minute
snapshot.

```
Solar wind pressure pulse
    → dayside reconnection   (~5 min lag)
        → magnetospheric energy loading
            → substorm onset: SuperMAG dBn spike   (~15–30 min)
                → field-aligned current → convection: SuperDARN vlos   (~5 min)
                    → ionospheric conductivity: TEC enhancement   (~5 min)
```

A model trained on single-snapshot cross-modal alignment will try to
simultaneously align SuperMAG and SuperDARN within the same 2-minute window.
But the SuperDARN convection response to the substorm onset visible in SuperMAG
may be 15–30 minutes later. Within-snapshot alignment forces the model to treat
these as unrelated, because they are not co-occurring — they are sequentially
related.

**Phase 0** addresses this by training each encoder on its own modality first,
using temporal proximity as the supervisory signal. Two snapshots of SuperDARN
from the same 30-minute window are treated as a "positive pair" — they represent
the same geomagnetic state. Snapshots from different windows in the batch are
negatives. This is temporal contrastive learning (van den Oord et al. CPC,
2018): the encoder learns to cluster similar states and separate different ones
**within a single modality**, without needing cross-modal alignment at all.

After Phase 0, each encoder has learned a meaningful geometry over its own
observable: quiet conditions cluster away from substorm conditions, storm-time
enhanced convection is distinguishable from quiet-time drift patterns, and so
on. **Phase 1** then aligns these per-modality geometries using a wider
temporal window (16 minutes = 8 steps), so that the cause (substorm onset in
SuperMAG) and its effect (convection response in SuperDARN) are both visible
within a single training example.

---

## Training curriculum

### Phase 0 — Per-modality self-supervised pretraining

Each of the four encoders is trained independently. No cross-modal signal.
No labels. No physical assumptions.

**Positive pair**: snapshot at time *t* and snapshot at time *t* + *k*,
where *k* is drawn uniformly from 1–15 steps (2–30 min). Near-in-time
snapshots represent the same geomagnetic state.

**Negatives**: all other samples in the batch (random timestamps, random
states).

**Loss**: symmetric CLIP, `CE(z_a @ z_b.T / 0.07, arange(B))`.

A projection MLP (256 → 512 → 128) sits above the encoder during Phase 0
and is discarded afterwards — a standard SimCLR practice that prevents the
contrastive geometry from collapsing into the encoder's representational
bottleneck. Only the encoder weights are carried forward.

**Variance regularisation**: `mean_d max(0, 1 − std_b(z_d))` penalises
latent dimensions that collapse to a constant across the batch. This is
especially important when the batch is small (batch = 1 on CPU-only
machines), where the CLIP loss itself has no gradient (log(1) = 0).

### Phase 1 — Cross-modal alignment with wider context

Encoder weights initialised from Phase 0. Now train the full model including
the fusion layer and decoders.

**Cross-modal CLIP**: for all pairs of available modalities at the same
timestamp, `CE(z_sd @ z_smag.T / τ, arange)`. Same-timestamp embeddings
of different sensors must agree on the ionospheric state.

**Temporal CLIP**: `CE(z_t @ z_{t+Δ}.T / τ, arange)` with Δ = 8 steps
(16 minutes). The shared latent should be more similar to nearby states
than distant ones — temporal coherence of the representation.

**Reconstruction**: all modalities decoded from `z_shared`, including
modalities excluded from encoding by dropout. This provides cross-modal
reconstruction signal (encode SuperMAG → decode SuperDARN) without
requiring simultaneous within-snapshot cause-effect alignment.

### Phase 2 — Latent dynamics (in development)

Freeze encoders. Train the FiLM-MLP dynamics model to predict `z_{t+Δ}`
from `z_t` conditioned on OMNI solar-wind features. The latent space at
this point should already encode meaningful state variation (Phase 0) and
cross-modal coherence (Phase 1), so the dynamics model learns transitions
between well-defined physical states rather than between unstructured
vectors.

### Phase 3 — End-to-end fine-tune

Unfreeze all parameters and fine-tune jointly.

---

## Architecture

All modalities are co-registered to a global **180 × 360** equirectangular
grid in magnetic apex coordinates (MLAT −90°→+90°, MLT 0→24 h). Both
hemispheres are represented natively.

```
SuperDARN (6, 180, 360) ──► SD Encoder ──────►┐
SuperMAG  (3, 180, 360) ──► SM Encoder ──────►├──► Reliability-weighted
GPS-TEC   (2, 180, 360) ──► TC Encoder ──────►┤   mean fusion → z_t (256-d)
DMSP      (5, 180, 360) ──► DP Encoder ──────►┘          │
                                                           ├──► Decoders (reconstruct each modality)
                                        OMNI (8-d) ──►    │
                                               FiLM-MLP ──►└──► z_{t+Δ} ──► Decoders (forecast)
```

**Grid channels:**

| Modality | Channels | Notes |
|----------|----------|-------|
| SuperDARN | vlos_n, vlos_e, model_vlos_n, model_vlos_e, obs_occ, soft_occ | obs_occ masks sparse coverage |
| SuperMAG | dBn, dBe, soft_occ | ≥40° mlat, both hemispheres |
| GPS-TEC | VTEC, dVTEC/dt | IONEX 2.5°×5° regridded |
| DMSP SSJ4/5 | log(e⁻ flux), log(e⁻ energy), log(i⁺ flux), log(i⁺ energy), soft_occ | Track coverage mask |

**Model size**: 146 M parameters (encoder-decoder), 3 M dynamics MLP.

**Occupancy channels** use binary cross-entropy loss (not MSE) — they are
soft masks ∈ [0,1], not field values.

---

## Repository layout

```
Iome/
├── src/iome/
│   ├── data/
│   │   ├── superdarn.py       # SuperDARNDataset — YYYYMMDD_sd.npy (6,180,360)
│   │   ├── supermag.py        # SuperMAGDataset — YYYYMMDDTHHMM_smag.npy (3,180,360)
│   │   ├── tec.py             # TECDataset — YYYYMMDDTHHMM_tec.npy (2,180,360)
│   │   ├── dmsp.py            # DMSPDataset — YYYYMMDDTHHMM_dmsp.npy (5,180,360)
│   │   ├── pairs.py           # TemporalPairDataset (Phase 0 positive pairs)
│   │   └── datamodule.py      # TriModalDataModule (Lightning, all four modalities)
│   ├── models/
│   │   ├── encoders.py        # Per-modality CNN encoders → (B, 256)
│   │   ├── decoders.py        # Matching transposed-conv decoders
│   │   ├── dynamics.py        # FiLM-conditioned latent dynamics MLP
│   │   ├── losses.py          # clip_loss, infonce_loss, reconstruction_loss, variance_loss
│   │   └── fusion.py          # UnifiedIonosphereModel (assembles all parts)
│   └── train/
│       ├── stage0.py          # Stage0ModalityModule — per-modality CLIP pretraining
│       ├── stage1.py          # Stage1ContrastiveModule — cross-modal alignment
│       ├── stage2.py          # Stage2DynamicsModule
│       └── stage3.py          # Stage3FinetuneModule
├── scripts/
│   ├── train_stage0.py        # Phase 0 entry point (one modality at a time)
│   ├── train_stage1.py        # Phase 1 entry point
│   ├── train_stage2.py        # Phase 2 entry point
│   ├── train_stage3.py        # Phase 3 entry point
│   ├── ingest_superdarn.py    # cnvmap → _sd.npy grids
│   ├── ingest_supermag.py     # CSV/HDF5 → _smag.npy grids
│   ├── ingest_tec.py          # IONEX → _tec.npy grids
│   ├── ingest_omni.py         # OMNIWeb → omni_YYYY.npy dicts
│   ├── make_timestamps.py     # Build ts_avail.json (union of available epochs)
│   ├── compute_stats.py       # Per-channel mean/std → stats_{mod}.npy
│   └── convert_grid_120_to_180360.py  # Reproject legacy 120×120 to 180×360
├── deploy/
│   ├── scc_hdd01_stage0.sh    # hdd01 CPU: sequential 4-modality Phase 0
│   ├── scc_hdd01_stage1.sh    # hdd01 CPU: Phase 1
│   ├── lab_ws02_stage1.sh     # ws02 CPU: Phase 1 (GPU NVML broken)
│   └── bede_stage{1,2,3}.sh   # SLURM H100 scripts
└── LOGBOOK.md                 # Running record of training symptoms and fixes
```

---

## Data

### Ingest (run once)

```bash
# SuperDARN cnvmap binaries → daily .npy grids
python scripts/ingest_superdarn.py \
    --input /path/to/cnvmap/ --out /data/iome_cache/superdarn

# SuperMAG per-station files → per-epoch grids
python scripts/ingest_supermag.py --out /data/iome_cache/supermag

# GPS-TEC IONEX (auto-downloads from CDDIS) → per-epoch grids
python scripts/ingest_tec.py \
    --years 1999 2000 2001 --out /data/iome_cache/tec

# OMNI solar wind (auto-downloads from OMNIWeb)
python scripts/ingest_omni.py --out /data/iome_cache/omni

# Build epoch list (union: any epoch with ≥2 modalities)
python scripts/make_timestamps.py \
    --cache_sd /data/iome_cache/superdarn \
    --cache_smag /data/iome_cache/supermag \
    --cache_tec  /data/iome_cache/tec \
    --out /data/iome_cache/splits

# Normalisation statistics (training split only)
python scripts/compute_stats.py \
    --ts_train /data/iome_cache/splits/ts_train.json \
    --cache_sd ... --cache_smag ... --cache_tec ... \
    --out /data/iome_cache/splits
```

### Current dataset (1999–2001, ~456 k epochs)

| Modality | Grid | Epochs |
|----------|------|--------|
| SuperDARN | (6, 180, 360) daily | 634 daily maps |
| SuperMAG | (3, 180, 360) per-epoch | 456,480 |
| GPS-TEC | (2, 180, 360) per-epoch | 456,480 |
| DMSP SSJ4/5 | (5, 180, 360) per-epoch | (partial) |

Splits: 80/10/10 by calendar day (not epoch) to prevent temporal leakage.

---

## Training

### Phase 0 — per-modality pretraining (run first, once)

```bash
# On scc-hdd-01 (80-core CPU, 30 GB RAM): all four modalities sequentially
ssh scc-hdd-01 "nohup bash ~/iome/deploy/scc_hdd01_stage0.sh \
    > /data/iome_cache/train_logs/stage0-all.log 2>&1 &"

# Or single modality (can parallelise across machines):
python scripts/train_stage0.py \
    --modality   sd \
    --cache_dir  /data/iome_cache/superdarn \
    --splits_dir /data/iome_cache/splits \
    --stats_dir  /data/iome_cache/splits \
    --ckpt_dir   /data/iome_cache/ckpts/stage0 \
    --batch_size 64 --max_steps 20000 --tau 0.07
```

Saves `stage0_{mod}_encoder.pt` (encoder weights only; projection head discarded).

### Phase 1 — cross-modal alignment

```bash
python scripts/train_stage1.py \
    --splits_dir     /data/iome_cache/splits \
    --cache_sd       /data/iome_cache/superdarn \
    --cache_smag     /data/iome_cache/supermag \
    --cache_tec      /data/iome_cache/tec \
    --cache_dmsp     /data/iome_cache/dmsp \
    --stats_dir      /data/iome_cache/splits \
    --omni_dir       /data/iome_cache/omni \
    --ckpt_dir       /data/iome_cache/ckpts/stage1 \
    --ckpt_stage0_dir /data/iome_cache/ckpts/stage0 \
    --delta_t_steps  8 \
    --tau            0.07 \
    --batch_size     32 --max_steps 50000
```

---

## Key metrics to watch in W&B (`st7ma784/iome`)

| Metric | Healthy range | Failure mode |
|--------|--------------|--------------|
| `train/l_clip` (Phase 0) | Starts at log(B) ≈ 4.2, drops to <1.0 | Stays at log(B) → encoder collapsed |
| `train/l_var` | Starts near 1.0, drops toward 0 | Stays at 1.0 → all dims collapsed |
| `train/grad_norm` | <1.0 (clipped) | Pre-clip norm >10 → consider lower LR |
| `val/recon_{mod}` | Should drop below predict-zero baseline (~0.75) | Plateau at 0.75 → latent not encoding that modality |
| `train/n_mods` | Mean ~2 for p_drop=0.3 | Always 1 → dropout too aggressive |

---

## Environment

```bash
# Local (scc-ws-02, open-ce conda env)
conda activate open-ce
export PYTHONPATH=/home/user/Iome/src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=""   # GPU NVML broken on ws02; CPU only

# scc-hdd-01
export OMP_NUM_THREADS=60        # 60 cores for compute; 4 workers for I/O
export PYTHONPATH=~/iome/src:$PYTHONPATH
```

Dependencies: `torch`, `pytorch-lightning`, `wandb`, `numpy`, `scipy`,
`aacgmv2` (coordinate transforms), `requests` (IONEX/OMNI downloads).

---

## References

- Ruohoniemi & Baker 1998 — SuperDARN spherical-harmonic convection mapping (RST)
- Weimer 2005 — statistical background convection (the mean we are trying to escape)
- Gjerloev 2012 — SuperMAG database
- Hernández-Pajares et al. 2009 — IGS global VTEC maps (IONEX)
- van den Oord et al. 2018 — CPC / InfoNCE: contrastive predictive coding
- Chen et al. 2020 — SimCLR: projection head + NT-Xent loss
- Radford et al. 2021 — CLIP: symmetric CE cross-modal alignment
- Bardes et al. 2022 — VICReg: variance regularisation
- Hafner et al. 2023 — Dreamer v3: discrete latent world model with CE loss
- Pérez et al. 2018 — FiLM: feature-wise linear modulation for conditioning
