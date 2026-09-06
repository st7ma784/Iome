# Iome Training Logbook

Chronological record of significant changes, symptoms, and findings during development and training of the Unified Ionospheric Model (SuperDARN + SuperMAG + GPS-TEC + DMSP, 180×360 global magnetic-coordinate grid).

---

## 2026-09-04 — Initial multi-machine overnight run

### Setup
- **Architecture**: 146 M param `UnifiedIonosphereModel` — 4 CNN encoders → reliability-weighted fusion → shared latent (D=256) → LatentDynamics MLP → 4 decoders. Grid: (180, 360) global equirectangular in magnetic coordinates.
- **Stage 1 objective**: InfoNCE contrastive loss + λ·smooth_l1 reconstruction. Modality dropout (p=0.3) during encoding; decode ALL modalities to get cross-modal supervision.
- **Machines**: scc-ws-02 (3× P100, GPU broken — see below), scc-hdd-01 (80-core CPU, 30 GB RAM).
- **wandb project**: `st7ma784/iome`

### Symptoms / findings

**GPU NVML failure (ws02)**  
`CUDACachingAllocator` asserts on `nvmlInit_v2()` — driver/library version mismatch (NVML lib 580.173 vs kernel driver). Validation pass completes (runs before first CUDA alloc), then crashes on first training batch. Fix: `CUDA_VISIBLE_DEVICES=""`, CPU only.

**Grid shape mismatch — smag/tec 120×120 vs sd 180×360**  
`RuntimeError: size of tensor a (360) must match tensor b (120)` in `losses.py:93`. Old ingest produced 120×120 northern-polar grids; new pipeline uses 180×360 global. SuperDARN had already been converted; smag/tec had not. ~912k files converted on hdd01 local disk (scipy zoom, factors 1/3 and 3 for lat/lon axes). Northern hemisphere MLAT 50–90° → rows 0:40 of new canvas.

**Corrupt partial-write files (NFS kill)**  
Attempted conversion via NFS (/data5 on ws02 = /data on hdd01) was killed mid-write, leaving ~2600 truncated files. Added shape guards to all four `_load` methods — wrong-shape or unreadable files silently return zeros rather than crashing the dataloader.

**FiLMLayer broadcast bug (z_next shape (B,B,D))**  
`mask.unsqueeze(-1)` was unconditional: for 2-D input `(B,D)` it produced `(B,1)→(B,1,1)`, then `(B,D)×(B,1,1) = (B,B,D)`. Fixed to only unsqueeze when `x.ndim == 3`.

**`/dev/shm` exhaustion (hdd01)**  
40+20 dataloader workers × ~83 MB/sample × prefetch_factor=2 → ~10 GB of shared memory. `/dev/shm` capped at 16 GB, already partially used. Fix: 4 workers per run, OpenMP threads for compute instead (`OMP_NUM_THREADS=60`).

**OOM at batch=4 on hdd01 (newer PyTorch)**  
Python 3.13 + latest PyTorch uses ~15 GB for batch=4 on 180×360 grids during backprop (large activation tensors from CNN encoders). ws02's `open-ce` env (Python 3.11, older PyTorch) uses only ~6.5 GB for the same batch. hdd01 dropped to batch=1. Root cause not fully diagnosed — likely difference in activation checkpointing defaults or allocator behaviour between PyTorch versions.

---

## 2026-09-05 — NaN cascade in training

### Symptom
Run `gwodobta` (ws02): training loss goes NaN at step 28 and stays NaN for all subsequent steps. Visible in wandb:

| step | train/loss | train/l_cont | train/grad_norm | train/n_enc_mods |
|------|-----------|-------------|-----------------|-----------------|
| 5    | 2.76      | 2.01        | 319             | 2               |
| 9    | 0.78      | **0**       | 527             | 1               |
| 26   | 2.24      | 1.50        | 1056 (rising)   | 2               |
| 28   | **NaN**   | NaN         | NaN             | 3               |

### Root cause 1 — InfoNCE numerator bug (primary)
The implementation summed V view-similarities in the numerator:
```python
numerator = torch.exp(sim_pos).sum(dim=0)   # (B,) — sums over V views  ← WRONG
denominator = torch.exp(sim_matrix).sum(dim=1)
loss = -log(numerator / denominator)
```
With V=2 well-aligned views, `numerator ≈ 2·exp(s_pos)` which can exceed `denominator` (which includes only one copy of the positive). This makes `log(ratio) > 0` → **negative loss** → optimizer maximises similarity rather than discriminating → gradients explode.

**Fix**: Compute each view as a separate InfoNCE term via `logsumexp`:
```python
for z_v in z_views:
    sim_pos  = (z_v_norm * z_norm).sum(dim=1) / tau   # one positive per anchor
    total   += (log_denom - sim_pos).mean()            # always ≥ 0
```
`log_denom = logsumexp(z_norm @ z_norm.T / tau, dim=1)` includes the positive diagonal, so the loss is bounded below by 0.

Also: when `n_enc_mods=1` (one modality survived dropout), the previous code returned `l_cont=0` without a gradient, making `λ·l_recon` the only signal — not harmful per se but the zero was masking the real issue.

### Root cause 2 — No gradient clipping
`on_before_optimizer_step` logged `clip_grad_norm_(..., max_norm=inf)` (measuring only). Grad norm rose monotonically from 319 to 1056 before NaN. **Fix**: clip at `max_norm=1.0`.

### Additional fix — BCE for occupancy channels
`soft_occ` channels are soft binary masks ∈ [0,1]. Smooth-L1 is not a calibrated loss for binary targets near zero. **Fix**: `binary_cross_entropy_with_logits` for occupancy channels; smooth-L1 retained for physics channels (velocities, VTEC, dB fields) which are heavy-tailed continuous.

### Additional fix — NaN guard
`training_step` now returns `None` on non-finite loss. Lightning skips the `optimizer.step()` for that batch — model weights stay clean rather than being poisoned by a NaN gradient.

### On the user's cross-entropy question
Cross-entropy on continuous spatial fields requires discretisation (binning values → softmax over bins, as in VQ-VAE decoders). Appropriate for multimodal distributions (e.g. vlos has ±200 m/s modes) but needs architectural decoder changes (output shape B×C×num_bins×H×W). Deferred — fix the NaN sources first; revisit if smooth-L1 under-captures the multimodal structure of the fields.

---

## Active runs (as of 2026-09-05 ~22:35)

| Machine | wandb run | batch | OMP | p_drop | τ | notes |
|---------|-----------|-------|-----|--------|---|-------|
| scc-ws-02 | (starting) | 4 | — | 0.3 | 0.10 | CPU, conda open-ce |
| scc-hdd-01 | `ic5lcs3v` | 1 | 60 | 0.3 | 0.10 | CPU, Python 3.13 |

Both runs include post-fix code: corrected InfoNCE, grad clip=1.0, BCE occ, NaN guard.

---

## 2026-09-05 — LR logging as 0

### Symptom
`train/lr` shows 0 throughout warmup.

### Root cause
`configure_optimizers` created a `LambdaLR` warmup scheduler but never returned it — only `CosineAnnealingLR` was passed to Lightning. The warmup was silently discarded. CosineAnnealingLR was the active scheduler from step 0, meaning LR started at the peak (3e-4) and immediately decayed — no warmup at all. Additionally `lr_lambda(0) = 0/500 = 0`, so whatever logged LR=0 was reading the discarded scheduler's state.

### Fix
`SequentialLR([LinearLR(1e-3→1.0, 500 steps), CosineAnnealingLR(→1e-6)])`. LR ramps from 3e-7 at step 0 to 3e-4 at step 500, then cosine-decays to 1e-6 at step 50k.

---

## 2026-09-06 — Training curriculum redesign: Phase 0 + revised Phase 1

### Symptom
Runs plateau quickly across all modalities. `recon_smag` and `recon_tec`
never meaningfully drop below the predict-zero baseline (~0.75).

### Root cause: temporal scale mismatch
The magnetospheric cause-effect chain (solar wind → substorm onset → convection
response → TEC enhancement) unfolds over 20–60 minutes. At 2-minute cadence,
a single snapshot sits in the middle of this chain with no causal context.
The model cannot learn that SuperMAG dBn and SuperDARN vlos are linked because
the substorm onset (visible in smag) precedes the convection response (visible
in sd) by 10–30 minutes — never in the same frame. Encoders plateau at
mean-field patterns because that is all a single snapshot supports.

### On CE vs InfoNCE for this problem
`CE(z @ z.T / τ, arange(B))` (NT-Xent / CLIP formulation) is preferred here
because:
1. The softmax competition is over all B−1 negatives simultaneously — sharper
   gradients than sequential InfoNCE terms.
2. CE loss landscape encourages discrete, compositional representations
   (the grokking intuition — model learns to classify states rather than
   interpolate between them). This matters for learning a posterior over
   ionospheric states, which are driven by discrete physical regimes
   (quiet, substorm, storm).
3. The identity-matrix target is exact: each sample should be most similar
   to itself and dissimilar from all others in the batch.

### Revised curriculum

**Phase 0 (new): per-modality self-supervised pretraining**
- Train each encoder independently on its own data stream.
- Positive pairs: snapshot at t and snapshot at t+k, k ∈ [1, 15] steps (2–30 min).
  Near timesteps represent similar ionospheric states.
- Loss: symmetric CLIP, `CE(z_a @ z_b.T / 0.07, arange(B))`.
- Projection head (2-layer MLP, 256→512→128) sits above encoder during
  pretraining; discarded afterwards. Only encoder weights saved.
- Teaches each encoder to produce distinctive state embeddings from its own
  data alone. No cross-modal alignment needed yet.
- 20k steps per modality (~2 hours on hdd01 CPU).

**Phase 1 (updated): cross-modal alignment with wider temporal context**
- Load Phase 0 encoder weights.
- `delta_t_steps=8` (16 min) instead of 1 (2 min) — positive pair spans enough
  time for cause-effect to manifest across modalities.
- Cross-modal CLIP: `CE(z_sd @ z_smag.T / τ, arange)` for all modality pairs
  at the same timestamp. Same-timestamp embeddings of different sensors should
  agree on ionospheric state.
- Temporal CLIP: `CE(z_shared_t @ z_shared_{t+delta}.T / τ, arange)` — nearby
  states should be more similar than distant ones.
- Reconstruction retained as secondary signal (λ=0.5).
- Variance regularisation retained (λ=0.04).

### Relevant prior work
- **CPC** (van den Oord 2018): predicts future latents from past sequence via
  InfoNCE. Most directly related to the "learn a posterior over future states"
  framing.
- **CLIP** (Radford 2021): CE cross-modal alignment, the exact loss used here.
- **Dreamer v3** (Hafner 2023): discrete latent world model trained with CE
  — the grokking-toward-underlying-physics intuition.
- **MoCo v3**: momentum encoder for larger effective negative batch without
  memory cost. Relevant if batch stays constrained at 1–4.

### New files
- `src/iome/data/pairs.py` — `TemporalPairDataset` wrapper
- `src/iome/models/losses.py` — `clip_loss(z_a, z_b, tau)` added
- `src/iome/train/stage0.py` — `Stage0ModalityModule` + `ProjectionHead`
- `scripts/train_stage0.py` — per-modality pretraining entry point
- `deploy/scc_hdd01_stage0.sh` — sequential 4-modality run on hdd01

### Modified files
- `src/iome/train/stage1.py` — cross-modal CLIP, temporal CLIP, loads stage0 weights
- `scripts/train_stage1.py` — `--ckpt_stage0_dir`, `--delta_t_steps` args

---

## 2026-09-06 — Dynamic lag: soft temporal alignment

### Problem with fixed lag
The causal propagation delay from SuperMAG substorm onset to SuperDARN convection
response is **not constant**.  It varies with:
- Solar wind Bz (stronger southward → faster energy injection → shorter lag, ~8 min vs ~25 min)
- Substorm intensity (stronger onset → faster convection response)
- Substorm phase (growth vs expansion vs recovery have different cross-modal coupling)
- MLT sector (dawn/dusk asymmetry in convection patterns)

A single `lag_offsets["sd"] = 8` is the mean over all these conditions.

### Three approaches considered

**1. Soft window attention (implemented)**
Encode each modality at K snapshots t, t+1, ..., t+K-1.  A small MLP takes
`z_smag(t)` → attention weights over K positions (softmax).  The attended `z_X`
is the state-conditioned lag mixture.  Window encodings are **detached** from
gradient, so memory cost is O(1) in K — the gradient flows only through the
attention weights (the lag predictor) and `z_smag`.  Target encoder gets
gradients via reconstruction and temporal CLIP.

Advantage: naturally handles multimodal lag distributions (e.g., weak substorms
peak at lag=4, strong at lag=12 — the attention can learn a bimodal weight).

**2. LagPredictor + differentiable linear interpolation**
`MLP(z_smag)` → continuous scalar `λ ∈ [0, K]`, then interpolate:
`z_aligned = (ceil(λ) - λ) * z_floor + (λ - floor(λ)) * z_ceil`.
Only 2 encoder calls per modality.  Interpretable (log `λ` conditioned on
substorm index or Bz).  Cannot represent bimodal lag distributions.
To use: replace `TemporalAlignmentHead` with `LagPredictor(latent_dim, n_targets=3)`.

**3. Cross-correlation soft-argmax**
Compute `corr[k] = cos_sim(z_smag(t), z_X(t+k))` for k in window during each step,
then take a differentiable soft-argmax: `λ = Σ k * softmax(β * corr)[k]`.
No learnable parameters — the lag falls entirely out of the loss landscape.
Most memory-intensive (K encoder forward passes in the loop).  A clean baseline
to check that methods 1 and 2 find the same peak.

### Chosen approach
**Soft window attention** (`TemporalAlignmentHead`).  Enabled via
`--align_window_steps K` (K=8 recommended = 16 min window at 2-min cadence).
Fixed-lag fallback remains active when K=0 (default).

### Metrics to decide when to change method

Logged to wandb per step when window is active:
- `train/lag_mean_{mod}` — expected lag = Σ k * w_k in steps (×2 = minutes)
  Watch for: constant value → head collapsed to always pick same lag → switch to method 2
- `train/lag_entropy_{mod}` — entropy of attention distribution
  Watch for: entropy → 0 → attention is deterministic (one lag dominates);
             entropy → log(K) → attention is uniform (head learned nothing)
  Healthy: intermediate, varying with substorm level

Additional checks (not auto-logged, run manually):
- Correlate `lag_mean_sd` with Dst index or AL index — should be anticorrelated
  during storm main phase (shorter lag under strong driving)
- Compare distribution of `lag_mean_sd` during quiet vs substorm hours —
  should shift to shorter values during substorms
- If `lag_entropy > 0.9 * log(K)` for >1000 steps, head is not learning:
  try larger τ in the attention temperature (anneal softmax sharpness), or
  switch to method 2 (LagPredictor interpolation)
- If `lag_mean_sd` is outside [2, 15] steps (4–30 min) at convergence, the
  physics is wrong — check smag encoder quality from Stage 0

### Implementation
- `src/iome/models/temporal_align.py` — `TemporalAlignmentHead`, `encode_window`
- `src/iome/data/datamodule.py` — `QuadModalDataset` returns `xs_window (B,K,C,H,W)`
  when `align_window_steps > 0`; fixed-lag `xs_aligned` still returned always
- `src/iome/train/stage1.py` — soft attention cross-modal CLIP when window present;
  logs `lag_mean_{mod}` and `lag_entropy_{mod}`
- `scripts/train_stage1.py` — `--align_window_steps` arg

---

## 2026-09-06 — Fixed lag-corrected cross-modal CLIP

### Motivation
Phase 1 originally paired `z_smag(t)` with `z_sd(t)` — same timestamp, different sensors.
But the substorm onset visible in SuperMAG precedes the convection response in SuperDARN by
10–30 minutes.  Pairing same-timestamp latents forces the model to align states that are NOT
causally contemporaneous.  The lag analysis (`analyse_lag.py`) recovers the empirical delays
from latent-space cross-correlation.

### Implementation
**`src/iome/data/datamodule.py`**:
- `QuadModalDataset` accepts `lag_offsets: {mod: steps}` (smag=0 as reference)
- Returns `xs_aligned` in each batch — each modality loaded at `t + lag_offsets[mod]`
  rather than `t`.  E.g. `xs_aligned["sd"]` = sd at `t + 8` aligns causally with smag at `t`.
- `TriModalDataModule` accepts `lag_matrix` dict and converts via `_lag_offsets_from_matrix()`
- `_valid` range safely trimmed to avoid index overflow for any combination of offsets

**`src/iome/train/stage1.py`**:
- Separate forward pass through `xs_aligned` (no dropout) → `out_aln`
- Cross-modal CLIP now uses `z_views_aln` from lag-aligned inputs
- Temporal CLIP unchanged (still uses `z_shared_t` vs `z_shared_{t+delta}`)

**`scripts/train_stage1.py`**: `--lag_matrix /path/lag_matrix.json` arg

**`deploy/scc_hdd01_stage0.sh`**: auto-runs `analyse_lag.py` after all four encoders finish

**`deploy/scc_hdd01_stage1.sh`**: auto-picks up `$SPLITS/lag_matrix.json` if present;
also auto-loads stage0 encoder weights if `$CACHE/ckpts/stage0` exists

### Expected effect
Cross-modal CLIP loss should decrease faster because the model is now aligning
latents that represent the same physical event at its natural causal phase offset —
smag substorm onset paired with sd convection response 16 min later.

---

## Outstanding / watch list

- [ ] Confirm NaN-free training for ≥ 500 steps on both machines
- [ ] Gradient norm should plateau below 1.0 after clipping; if pre-clip norm stays >10 long-term, consider reducing LR (currently 3e-4)
- [ ] hdd01 batch=1 — very slow (1 sample/step). Consider gradient accumulation (`accumulate_grad_batches=8`) to get effective batch=8 without OOM
- [ ] DMSP stats missing from splits dir — DMSP returns zeros if stats not found; add `compute_stats.py` run for DMSP channel stats
- [ ] `l_cont=0` when only 1 modality encoded: fixed InfoNCE now returns a valid gradient even with 1 view (single-view pulls that view's encoding toward z_shared)
- [ ] Consider gradient checkpointing in encoders to bring hdd01 back to batch=4+
- [ ] Bede HPC deployment once local runs confirm stability
