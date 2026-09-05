"""
Loss functions for all three training stages.

  - infonce_loss:        cross-time contrastive loss on the shared latent
  - reconstruction_loss: per-modality weighted MSE / BCE
  - dynamics_loss:       MSE between predicted and actual z_{t+1}
  - stage_loss:          combines the above with the per-stage lambda schedule
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

MODALITIES = ("sd", "smag", "tec", "dmsp")

# Per-modality loss weights
RECON_WEIGHTS = {"sd": 1.0, "smag": 0.8, "tec": 0.6, "dmsp": 0.7}


# ---------------------------------------------------------------------------
# InfoNCE contrastive loss
# ---------------------------------------------------------------------------

def infonce_loss(
    z_shared: torch.Tensor,
    z_views: List[torch.Tensor],
    tau: float = 0.1,
) -> torch.Tensor:
    """
    Multi-view temporal InfoNCE loss.

    Positives: different encoder views of the SAME timestep (z_views[i] vs z_shared).
    Negatives: shared latents from DIFFERENT timesteps in the batch.

    This enforces two properties simultaneously:
      1. Cross-view agreement at the same time (different modalities → same z).
      2. Temporal discriminability (z_t ≠ z_{t'} for t ≠ t').

    Args:
        z_shared: (B, D) mean-fused latent per timestep in the batch
        z_views:  list of V tensors (B, D), one per available modality encoder
        tau:      InfoNCE temperature (0.07–0.2; lower = harder negatives)

    Returns:
        scalar loss
    """
    B = z_shared.shape[0]
    z_norm = F.normalize(z_shared, dim=1)                        # (B, D)

    # Cross-time similarity matrix — denominator candidates
    sim_matrix = z_norm @ z_norm.T / tau                         # (B, B)

    # Positive similarities: each per-view encoding vs the shared latent at same t
    if not z_views:
        return z_shared.new_tensor(0.0)

    z_views_norm = [F.normalize(z, dim=1) for z in z_views]
    stacked = torch.stack(z_views_norm, dim=0)                   # (V, B, D)
    sim_pos = torch.einsum("bd,vbd->vb", z_norm, stacked) / tau  # (V, B)

    # InfoNCE: sum positive scores / sum all scores
    numerator   = torch.exp(sim_pos).sum(dim=0)                  # (B,)
    denominator = torch.exp(sim_matrix).sum(dim=1)               # (B,)
    return (-torch.log(numerator / denominator.clamp(min=1e-8))).mean()


# ---------------------------------------------------------------------------
# Reconstruction loss
# ---------------------------------------------------------------------------

def reconstruction_loss(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    occ_channel: Optional[int] = None,
) -> torch.Tensor:
    """
    Weighted Smooth-L1 reconstruction loss with optional occupancy masking.

    For SuperDARN, channels 0-1 (obs velocity) are only supervised in
    radar-covered cells (soft_occ > 0.05).  Channels 2-5 are always supervised.

    For SuperMAG and TEC, occ_channel=None → full-grid supervision.

    Args:
        y_hat:        (B, C, H, W) predicted grid
        y:            (B, C, H, W) target grid
        occ_channel:  index of the soft-occupancy channel (None = no masking)

    Returns:
        scalar loss
    """
    err = F.smooth_l1_loss(y_hat, y, reduction="none", beta=0.1)  # (B, C, H, W)

    if occ_channel is not None:
        occ = (y[:, occ_channel : occ_channel + 1] > 0.05).float()  # (B, 1, H, W)
        # Observed channels get the occupancy mask; physics/background channels don't
        mask = torch.ones_like(err)
        mask[:, :2] = occ.expand_as(mask[:, :2])
        return (err * mask).sum() / mask.sum().clamp(min=1.0)

    return err.mean()


def multi_modal_recon_loss(
    y_hats: Dict[str, torch.Tensor],
    ys: Dict[str, torch.Tensor],
    weights: Dict[str, float] = RECON_WEIGHTS,
) -> torch.Tensor:
    """
    Sum of per-modality reconstruction losses, weighted by modality importance.
    """
    total = None
    for k in MODALITIES:
        if k not in y_hats or k not in ys:
            continue
        occ_ch = 4 if k == "sd" else (2 if k == "smag" else (4 if k == "dmsp" else None))
        l = weights.get(k, 1.0) * reconstruction_loss(y_hats[k], ys[k], occ_ch)
        total = l if total is None else total + l
    return total if total is not None else torch.tensor(0.0)


def per_modality_recon_losses(
    y_hats: Dict[str, torch.Tensor],
    ys: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Per-modality (unweighted) reconstruction losses for logging."""
    OCC = {"sd": 4, "smag": 2, "dmsp": 4}
    return {
        k: reconstruction_loss(y_hats[k], ys[k], OCC.get(k))
        for k in MODALITIES if k in y_hats and k in ys
    }


# ---------------------------------------------------------------------------
# Dynamics loss
# ---------------------------------------------------------------------------

def dynamics_loss(z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    """
    MSE between predicted z_{t+1} and actual z_{t+1}.

    z_target is detached so this loss does not back-propagate into the encoders
    when they are frozen (Stage 2).
    """
    return F.mse_loss(z_pred, z_target.detach())


# ---------------------------------------------------------------------------
# Stage-specific combined objectives
# ---------------------------------------------------------------------------

def stage1_loss(
    z_shared: torch.Tensor,
    z_views: List[torch.Tensor],
    y_hats: Dict[str, torch.Tensor],
    ys: Dict[str, torch.Tensor],
    tau: float = 0.1,
    lambda_recon: float = 0.5,
) -> Dict[str, torch.Tensor]:
    l_cont  = infonce_loss(z_shared, z_views, tau=tau)
    l_recon = multi_modal_recon_loss(y_hats, ys)
    loss    = l_cont + lambda_recon * l_recon
    return {"loss": loss, "l_cont": l_cont, "l_recon": l_recon}


def stage2_loss(
    z_shared: torch.Tensor,
    z_views: List[torch.Tensor],
    z_pred: torch.Tensor,
    z_target: torch.Tensor,
    y_hats: Dict[str, torch.Tensor],
    ys: Dict[str, torch.Tensor],
    tau: float = 0.2,
    lambda_dyn: float = 1.0,
    lambda_cont: float = 0.1,
) -> Dict[str, torch.Tensor]:
    l_recon = multi_modal_recon_loss(y_hats, ys)
    l_dyn   = dynamics_loss(z_pred, z_target)
    l_cont  = infonce_loss(z_shared, z_views, tau=tau)
    loss    = l_recon + lambda_dyn * l_dyn + lambda_cont * l_cont
    return {"loss": loss, "l_recon": l_recon, "l_dyn": l_dyn, "l_cont": l_cont}


def stage3_loss(
    z_shared: torch.Tensor,
    z_views: List[torch.Tensor],
    z_pred: torch.Tensor,
    z_target: torch.Tensor,
    y_hats: Dict[str, torch.Tensor],
    ys: Dict[str, torch.Tensor],
    tau: float = 0.2,
    lambda_dyn: float = 1.0,
    lambda_cont: float = 0.05,
) -> Dict[str, torch.Tensor]:
    # Same structure as stage2 but with smaller lambda_cont
    return stage2_loss(
        z_shared, z_views, z_pred, z_target, y_hats, ys,
        tau=tau, lambda_dyn=lambda_dyn, lambda_cont=lambda_cont,
    )
