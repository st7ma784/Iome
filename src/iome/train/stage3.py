"""
Stage 3: End-to-end fine-tuning.

All parameters are unfrozen.  The full objective (reconstruction + dynamics +
weak contrastive) is optimised jointly at a lower learning rate so that the
encoder representations can adapt to the forecast task without drifting far
from the Stage 1 alignment.

Modality dropout continues at a lower rate — by stage 3 the decoders are
already capable of cross-modal reconstruction; dropout here prevents
overfitting to always-available modality combinations.
"""

import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..models.fusion import UnifiedIonosphereModel
from ..models.losses import stage3_loss


class Stage3FinetuneModule(pl.LightningModule):
    """
    Args:
        model:           UnifiedIonosphereModel (loaded from Stage 2 checkpoint)
        lr_encoders:     learning rate for encoder parameters (lower)
        lr_rest:         learning rate for dynamics + decoders
        weight_decay:    AdamW weight decay
        tau:             InfoNCE temperature
        lambda_dyn:      dynamics loss weight
        lambda_cont:     contrastive anchor weight (very small)
        max_steps:       total steps for cosine schedule
        p_mod_drop:      per-modality drop probability during training
    """

    def __init__(
        self,
        model: UnifiedIonosphereModel,
        lr_encoders: float = 3e-5,
        lr_rest: float = 1e-4,
        weight_decay: float = 1e-2,
        tau: float = 0.2,
        lambda_dyn: float = 1.0,
        lambda_cont: float = 0.05,
        max_steps: int = 30_000,
        p_mod_drop: float = 0.15,
    ):
        super().__init__()
        self.model = model
        self.tau = tau
        self.lambda_dyn = lambda_dyn
        self.lambda_cont = lambda_cont
        self.max_steps = max_steps
        self.p_mod_drop = p_mod_drop
        self.save_hyperparameters(ignore=["model"])

        # Unfreeze everything
        for p in self.model.parameters():
            p.requires_grad_(True)

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

        # Full forward with dropout on xs; xs_next encoded separately without dropout
        out    = self.model(xs, u, omni_mask, decode_mods=decode_mods, p_mod_drop=p_drop)
        _, z_target = self.model.encode(xs_next)

        return stage3_loss(
            z_shared=out["z_shared"],
            z_views=list(out["z_views"].values()),
            z_pred=out["z_next"],
            z_target=z_target,
            y_hats=out["y_hats"],
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
        param_groups = [
            {"params": list(self.model.encoder_parameters()),  "lr": self.hparams.lr_encoders},
            {"params": self.model.non_encoder_parameters(),    "lr": self.hparams.lr_rest},
        ]
        opt = AdamW(param_groups, weight_decay=self.hparams.weight_decay)
        scheduler = CosineAnnealingLR(opt, T_max=self.max_steps)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
