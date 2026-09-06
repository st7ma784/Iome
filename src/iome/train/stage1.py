"""
Stage 1: Cross-modal alignment with wider temporal context.

Builds on per-modality encoders pretrained in Stage 0.  The objective is
to align representations across modalities using a symmetric CLIP loss —
z_sd_t and z_smag_t (same timestamp, different sensors) should be more
similar than z_sd_t and z_smag_{t'} for t ≠ t'.

The dataset is sampled with a wider delta_t (8–30 steps = 16–60 min) so
that the reconstruction target sits far enough ahead for cause-effect
relations to have manifested across modalities.

Losses:
  l_clip_cross  — CLIP across all pairs of available modalities at time t
  l_clip_temp   — CLIP between t and t+delta within each modality (temporal coherence)
  l_recon       — reconstruction of all modalities from z_shared
  l_var         — variance regularisation on z_shared

Loading Stage 0 weights:
  Pass ckpt_stage0_dir to load encoder state_dicts saved by train_stage0.py.
  Encoders start from good single-modal representations rather than random init.
"""

import itertools
from pathlib import Path
from typing import Optional

import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from ..models.fusion import UnifiedIonosphereModel
from ..models.losses import (
    clip_loss, variance_loss, multi_modal_recon_loss, per_modality_recon_losses,
)

MODALITIES = ("sd", "smag", "tec", "dmsp")


class Stage1ContrastiveModule(pl.LightningModule):
    """
    Args:
        model:              UnifiedIonosphereModel instance
        ckpt_stage0_dir:    directory of stage0_{mod}_encoder.pt files (optional)
        lr:                 peak learning rate
        weight_decay:       AdamW weight decay
        tau:                CLIP temperature (0.07 recommended)
        lambda_recon:       reconstruction loss weight
        lambda_temp:        temporal CLIP weight (within-modality across time)
        warmup_steps:       linear LR warmup steps
        max_steps:          total training steps
        p_mod_drop:         per-modality dropout probability
    """

    def __init__(
        self,
        model:              UnifiedIonosphereModel,
        ckpt_stage0_dir:    Optional[Path] = None,
        lr:                 float = 1e-4,
        weight_decay:       float = 1e-2,
        tau:                float = 0.07,
        lambda_recon:       float = 0.5,
        lambda_temp:        float = 0.25,
        warmup_steps:       int   = 500,
        max_steps:          int   = 50_000,
        p_mod_drop:         float = 0.2,
    ):
        super().__init__()
        self.model        = model
        self.tau          = tau
        self.lambda_recon = lambda_recon
        self.lambda_temp  = lambda_temp
        self.warmup_steps = warmup_steps
        self.max_steps    = max_steps
        self.p_mod_drop   = p_mod_drop
        self.save_hyperparameters(ignore=["model"])

        if ckpt_stage0_dir is not None:
            self._load_stage0_encoders(Path(ckpt_stage0_dir))

    def _load_stage0_encoders(self, ckpt_dir: Path):
        loaded = []
        for mod in MODALITIES:
            p = ckpt_dir / f"stage0_{mod}_encoder.pt"
            if p.exists():
                sd = torch.load(p, map_location="cpu")
                self.model.encoders[mod].load_state_dict(sd)
                loaded.append(mod)
        if loaded:
            print(f"[stage1] Loaded stage0 encoder weights: {loaded}")

    # ------------------------------------------------------------------

    def _unpack(self, batch):
        xs        = batch["xs"]
        ys        = batch["ys"]
        xs_next   = batch.get("xs_next", {})
        u         = batch["u"]
        omni_mask = batch.get("omni_mask", None)
        return xs, ys, xs_next, u, omni_mask

    def _step(self, batch, p_drop: float):
        xs, ys, xs_next, u, omni_mask = self._unpack(batch)
        decode_mods = tuple(ys.keys())

        out      = self.model(xs,      u, omni_mask, decode_mods=decode_mods, p_mod_drop=p_drop)
        out_next = self.model(xs_next, u, omni_mask, decode_mods=(), p_mod_drop=0.0)

        z_views      = out["z_views"]        # {mod: (B, D)}
        z_shared     = out["z_shared"]       # (B, D)
        z_shared_next= out_next["z_shared"]  # (B, D)

        # --- Cross-modal CLIP: align modality pairs at same timestamp ---
        l_clip_cross = z_shared.new_tensor(0.0)
        mod_pairs = list(itertools.combinations(
            [m for m in MODALITIES if m in z_views], 2
        ))
        if mod_pairs:
            for m_a, m_b in mod_pairs:
                l_clip_cross = l_clip_cross + clip_loss(z_views[m_a], z_views[m_b], tau=self.tau)
            l_clip_cross = l_clip_cross / len(mod_pairs)

        # --- Temporal CLIP: z_shared_t vs z_shared_{t+delta} should be closer
        #     than different-timestep pairs (temporal coherence across the window) ---
        l_clip_temp = clip_loss(z_shared, z_shared_next, tau=self.tau)

        # --- Reconstruction ---
        l_recon = multi_modal_recon_loss(out["y_hats"], ys)

        # --- Variance regularisation on shared latent ---
        l_var = variance_loss(z_shared)

        loss = (l_clip_cross
                + self.lambda_temp  * l_clip_temp
                + self.lambda_recon * l_recon
                + 0.04              * l_var)

        per_mod = per_modality_recon_losses(out["y_hats"], ys)

        return {
            "loss":          loss,
            "l_clip_cross":  l_clip_cross,
            "l_clip_temp":   l_clip_temp,
            "l_recon":       l_recon,
            "l_var":         l_var,
            "n_mods":        torch.tensor(float(len(z_views))),
            **{f"recon_{k}": v for k, v in per_mod.items()},
        }

    def training_step(self, batch, batch_idx):
        losses = self._step(batch, p_drop=self.p_mod_drop)
        loss = losses["loss"]
        if not torch.isfinite(loss):
            self.log("train/nan_skipped", 1.0)
            return None
        self.log("train/loss", loss, prog_bar=True)
        self.log_dict({"train/" + k: v for k, v in losses.items() if k != "loss"}, prog_bar=False)
        return loss

    def on_before_optimizer_step(self, optimizer):
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.log("train/grad_norm", norm, prog_bar=False)

    def validation_step(self, batch, batch_idx):
        losses = self._step(batch, p_drop=0.0)
        self.log_dict({"val/" + k: v for k, v in losses.items()}, prog_bar=True)

    def configure_optimizers(self):
        opt    = AdamW(self.model.parameters(), lr=self.hparams.lr,
                       weight_decay=self.hparams.weight_decay)
        warmup = LinearLR(opt, start_factor=1e-3, end_factor=1.0,
                          total_iters=self.warmup_steps)
        cosine = CosineAnnealingLR(opt, T_max=self.max_steps - self.warmup_steps,
                                   eta_min=1e-6)
        sched  = SequentialLR(opt, schedulers=[warmup, cosine],
                              milestones=[self.warmup_steps])
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
