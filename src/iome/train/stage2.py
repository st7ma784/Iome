"""
Stage 2: Dynamics + reconstruction training with frozen encoders.

Encoders are loaded from the Stage 1 checkpoint and frozen.
Only the LatentDynamics model and decoders are updated.

The dynamics model predicts z_{t+1} from z_t and u_t; the decoders
reconstruct observations from both the current and next latent.

Modality dropout trains decoders to reconstruct any modality from a
latent derived from a partial sensor set — without touching encoder weights.
"""

import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..models.fusion import UnifiedIonosphereModel
from ..models.losses import stage2_loss


class Stage2DynamicsModule(pl.LightningModule):
    """
    Args:
        model:           UnifiedIonosphereModel
        lr:              peak learning rate for dynamics + decoders
        weight_decay:    AdamW weight decay
        tau:             InfoNCE temperature (weak contrastive anchor)
        lambda_dyn:      dynamics loss weight
        lambda_cont:     contrastive loss weight (small — encoders frozen)
        max_steps:       total steps for cosine schedule
        p_mod_drop:      per-modality drop probability during training
    """

    def __init__(
        self,
        model: UnifiedIonosphereModel,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        tau: float = 0.2,
        lambda_dyn: float = 1.0,
        lambda_cont: float = 0.1,
        max_steps: int = 50_000,
        p_mod_drop: float = 0.2,
    ):
        super().__init__()
        self.model = model
        self.tau = tau
        self.lambda_dyn = lambda_dyn
        self.lambda_cont = lambda_cont
        self.max_steps = max_steps
        self.p_mod_drop = p_mod_drop
        self.save_hyperparameters(ignore=["model"])

        # Freeze encoders
        for p in self.model.encoders.parameters():
            p.requires_grad_(False)
        for p in self.model.log_mod_weights.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------

    def _unpack(self, batch):
        xs        = batch["xs"]
        xs_next   = batch["xs_next"]
        ys        = batch["ys"]
        ys_next   = batch["ys_next"]
        u         = batch["u"]
        omni_mask = batch.get("omni_mask", None)
        return xs, xs_next, ys, ys_next, u, omni_mask

    def _step(self, batch, p_drop: float):
        xs, xs_next, ys, ys_next, u, omni_mask = self._unpack(batch)
        decode_mods = tuple(ys.keys())

        # Encoders are frozen — run them under no_grad for efficiency.
        # apply_mod_drop is called manually so xs_enc is available for z_target too.
        with torch.no_grad():
            xs_enc  = self.model.apply_mod_drop(xs, p_drop)
            z_views, z_shared = self.model.encode(xs_enc)
            _, z_target       = self.model.encode(xs_next)   # no dropout on t+1

        # Only dynamics and decoders are in the compute graph
        z_next  = self.model.dynamics(z_shared, u, omni_mask)
        y_hats  = self.model.decode(z_shared, decode_mods)

        return stage2_loss(
            z_shared=z_shared,
            z_views=list(z_views.values()),
            z_pred=z_next,
            z_target=z_target,
            y_hats=y_hats,
            ys=ys,
            tau=self.tau,
            lambda_dyn=self.lambda_dyn,
            lambda_cont=self.lambda_cont,
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
        params = self.model.non_encoder_parameters()
        opt = AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = CosineAnnealingLR(opt, T_max=self.max_steps)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
