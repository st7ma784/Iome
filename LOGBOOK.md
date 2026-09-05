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

## Outstanding / watch list

- [ ] Confirm NaN-free training for ≥ 500 steps on both machines
- [ ] Gradient norm should plateau below 1.0 after clipping; if pre-clip norm stays >10 long-term, consider reducing LR (currently 3e-4)
- [ ] hdd01 batch=1 — very slow (1 sample/step). Consider gradient accumulation (`accumulate_grad_batches=8`) to get effective batch=8 without OOM
- [ ] DMSP stats missing from splits dir — DMSP returns zeros if stats not found; add `compute_stats.py` run for DMSP channel stats
- [ ] `l_cont=0` when only 1 modality encoded: fixed InfoNCE now returns a valid gradient even with 1 view (single-view pulls that view's encoding toward z_shared)
- [ ] Consider gradient checkpointing in encoders to bring hdd01 back to batch=4+
- [ ] Bede HPC deployment once local runs confirm stability
