"""
Stage 1: Contrastive pre-training.

Encoders are trained to produce a shared latent space via InfoNCE loss.
A lightweight reconstruction penalty keeps the latents geometrically
meaningful (prevents dimensional collapse).  Dynamics model is NOT updated.

Modality dropout: during training, a random subset of available modalities
is passed to the encoders.  ALL available modalities are decoded, giving a
cross-modal reconstruction signal even for sensors not used in encoding.
This forces z_shared to carry information about every modality it will later
need to reconstruct from partial sensor sets.
"""

import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..models.fusion import UnifiedIonosphereModel
from ..models.losses import stage1_loss


class Stage1ContrastiveModule(pl.LightningModule):
    """
    Args:
        model:         UnifiedIonosphereModel instance
        lr:            peak learning rate
        weight_decay:  AdamW weight decay
        tau:           InfoNCE temperature
        lambda_recon:  reconstruction loss weight
        warmup_steps:  linear warm-up before cosine decay
        max_steps:     total training steps (for cosine schedule)
        p_mod_drop:    per-modality drop probability during training (0 = off)
    """

    def __init__(
        self,
        model: UnifiedIonosphereModel,
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        tau: float = 0.1,
        lambda_recon: float = 0.5,
        warmup_steps: int = 500,
        max_steps: int = 50_000,
        p_mod_drop: float = 0.3,
    ):
        super().__init__()
        self.model = model
        self.tau = tau
        self.lambda_recon = lambda_recon
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.p_mod_drop = p_mod_drop
        self.save_hyperparameters(ignore=["model"])

    # ------------------------------------------------------------------

    def _unpack(self, batch):
        xs        = batch["xs"]                        # {mod: (B, C, H, W)}
        ys        = batch["ys"]                        # same mods, recon targets
        u         = batch["u"]                         # (B, u_dim)
        omni_mask = batch.get("omni_mask", None)
        return xs, ys, u, omni_mask

    def _step(self, batch, p_drop: float):
        xs, ys, u, omni_mask = self._unpack(batch)
        # decode_mods: ALL mods in ys — includes any that are dropped from encoding
        decode_mods = tuple(ys.keys())
        out = self.model(xs, u, omni_mask, decode_mods=decode_mods, p_mod_drop=p_drop)
        return stage1_loss(
            z_shared=out["z_shared"],
            z_views=list(out["z_views"].values()),
            y_hats=out["y_hats"],
            ys=ys,               # targets include dropped mods → cross-modal signal
            tau=self.tau,
            lambda_recon=self.lambda_recon,
        )

    def training_step(self, batch, batch_idx):
        losses = self._step(batch, p_drop=self.p_mod_drop)
        self.log_dict({"train/" + k: v for k, v in losses.items()}, prog_bar=True)
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        losses = self._step(batch, p_drop=0.0)
        self.log_dict({"val/" + k: v for k, v in losses.items()}, prog_bar=True)

    # ------------------------------------------------------------------

    def configure_optimizers(self):
        opt = AdamW(
            self.model.encoder_parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = CosineAnnealingLR(opt, T_max=self.max_steps - self.warmup_steps)

        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            return 1.0

        warmup = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
