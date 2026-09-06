"""
Stage 0: Per-modality self-supervised contrastive pretraining.

Each encoder is trained independently on its own data stream using temporal
positive pairs — nearby snapshots (within ±window_steps × 2 min) should
produce similar embeddings; different-timestep snapshots in the batch are
negatives.

Loss: symmetric CLIP / NT-Xent — CE(z_a @ z_b.T / τ, arange(B))

A small projection MLP sits on top of the encoder during pretraining (standard
SimCLR practice). Only the encoder weights are saved; the projector is
discarded. This keeps the encoder representation general rather than overfit
to the contrastive geometry.

Why this before stage 1:
  A single 2-min snapshot cannot contain the full cause-effect chain
  (solar wind → substorm → convection onset, ~20-60 min). Per-modality
  pretraining first teaches each encoder to produce distinctive state
  embeddings from its own data alone, without needing cross-modal alignment
  on timescales too short for the physics to manifest.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from ..models.losses import clip_loss, variance_loss


class ProjectionHead(nn.Module):
    """2-layer MLP projection head (SimCLR-style). Discarded after pretraining."""

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class Stage0ModalityModule(pl.LightningModule):
    """
    Args:
        encoder:       one of SuperDARNEncoder / SuperMAGEncoder / TECEncoder / DMSPEncoder
        modality:      string name for logging ("sd", "smag", "tec", "dmsp")
        latent_dim:    encoder output dimension D (default 256)
        proj_dim:      projector output dimension (default 128)
        tau:           CLIP temperature (0.07 is standard)
        lambda_var:    weight on variance regularisation (prevents collapse at small B)
        lr:            peak learning rate
        warmup_steps:  linear LR warmup steps
        max_steps:     total training steps
    """

    def __init__(
        self,
        encoder:       nn.Module,
        modality:      str,
        latent_dim:    int   = 256,
        proj_dim:      int   = 128,
        tau:           float = 0.07,
        lambda_var:    float = 0.04,
        lr:            float = 1e-3,
        warmup_steps:  int   = 500,
        max_steps:     int   = 20_000,
    ):
        super().__init__()
        self.encoder    = encoder
        self.projector  = ProjectionHead(latent_dim, hidden_dim=512, out_dim=proj_dim)
        self.modality   = modality
        self.tau        = tau
        self.lambda_var = lambda_var
        self.save_hyperparameters(ignore=["encoder"])

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """Encode → project (used for contrastive loss only)."""
        return self.projector(self.encoder(x))

    def _step(self, batch):
        z_a = self._embed(batch["anchor"])
        z_b = self._embed(batch["positive"])
        l_clip = clip_loss(z_a, z_b, tau=self.tau)
        # Variance reg on raw encoder output (not projector) — keeps encoder spread
        z_enc = self.encoder(batch["anchor"])
        l_var  = variance_loss(z_enc)
        loss   = l_clip + self.lambda_var * l_var
        return {"loss": loss, "l_clip": l_clip, "l_var": l_var}

    def training_step(self, batch, batch_idx):
        losses = self._step(batch)
        loss = losses["loss"]
        if not torch.isfinite(loss):
            self.log(f"train/{self.modality}/nan_skipped", 1.0)
            return None
        self.log(f"train/{self.modality}/loss",   loss,             prog_bar=True)
        self.log(f"train/{self.modality}/l_clip", losses["l_clip"], prog_bar=False)
        self.log(f"train/{self.modality}/l_var",  losses["l_var"],  prog_bar=False)
        return loss

    def on_before_optimizer_step(self, optimizer):
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.log(f"train/{self.modality}/grad_norm", norm, prog_bar=False)

    def validation_step(self, batch, batch_idx):
        losses = self._step(batch)
        self.log(f"val/{self.modality}/loss",   losses["loss"],   prog_bar=True)
        self.log(f"val/{self.modality}/l_clip", losses["l_clip"], prog_bar=False)

    def configure_optimizers(self):
        # Train encoder + projector together; projector discarded after pretraining
        opt    = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-2)
        warmup = LinearLR(opt, start_factor=1e-3, end_factor=1.0,
                          total_iters=self.hparams.warmup_steps)
        cosine = CosineAnnealingLR(opt,
                                   T_max=self.hparams.max_steps - self.hparams.warmup_steps,
                                   eta_min=1e-6)
        sched  = SequentialLR(opt, schedulers=[warmup, cosine],
                              milestones=[self.hparams.warmup_steps])
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
