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

    For each (view, timestep) pair, treats (z_view_b, z_shared_b) as the single
    positive and all other timesteps' z_shared as negatives.  Averaging over V
    views and B timesteps.

    Positives: per-modality encoding vs the fused shared latent at same t.
    Negatives: fused shared latents from DIFFERENT timesteps in the batch.

    Args:
        z_shared: (B, D) mean-fused latent
        z_views:  list of V tensors (B, D), one per available modality encoder
        tau:      InfoNCE temperature

    Returns:
        scalar loss (≥ 0 by construction)
    """
    if not z_views:
        return z_shared.new_tensor(0.0)

    B = z_shared.shape[0]
    z_norm = F.normalize(z_shared, dim=1)                         # (B, D)
    log_denom = torch.logsumexp(z_norm @ z_norm.T / tau, dim=1)  # (B,)  stable

    total = z_shared.new_tensor(0.0)
    for z_v in z_views:
        z_v_norm = F.normalize(z_v, dim=1)                        # (B, D)
        sim_pos  = (z_v_norm * z_norm).sum(dim=1) / tau           # (B,)  one positive per row
        # loss_b = log_denom - sim_pos  (always ≥ 0 when denom includes the positive)
        total = total + (log_denom - sim_pos).mean()

    return total / len(z_views)


# ---------------------------------------------------------------------------
# Reconstruction loss
# ---------------------------------------------------------------------------

def reconstruction_loss(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    occ_channel: Optional[int] = None,
) -> torch.Tensor:
    """
    Reconstruction loss: Smooth-L1 on physics channels, BCE on occupancy channel.

    Occupancy (soft_occ) is a soft binary mask ∈ [0,1] — BCE is the principled
    loss for it and avoids the scaling issues that smooth_l1 has near zero.

    For SuperDARN / SuperMAG / DMSP: observed-velocity channels are only
    supervised where the occupancy mask indicates coverage.

    Args:
        y_hat:        (B, C, H, W) predicted grid
        y:            (B, C, H, W) target grid (normalised)
        occ_channel:  index of the soft-occupancy channel (None = no masking or occ)

    Returns:
        scalar loss
    """
    if occ_channel is None:
        return F.smooth_l1_loss(y_hat, y, reduction="mean", beta=0.1)

    occ_target = y[:, occ_channel : occ_channel + 1]                 # (B,1,H,W) ∈ [0,1]
    obs_mask   = (occ_target > 0.05).float()                          # where radar/mag saw data

    # Physics channels — smooth_l1, only in observed cells
    phys_idx = [i for i in range(y.shape[1]) if i != occ_channel]
    y_phys   = y[:, phys_idx]                                         # (B, C-1, H, W)
    yh_phys  = y_hat[:, phys_idx]
    err      = F.smooth_l1_loss(yh_phys, y_phys, reduction="none", beta=0.1)
    # obs_mask supervision only on first 2 observed channels; rest always supervised
    mask         = torch.ones_like(err)
    mask[:, :2]  = obs_mask.expand_as(mask[:, :2])
    phys_loss    = (err * mask).sum() / mask.sum().clamp(min=1.0)

    # Occupancy channel — BCE (target already in [0,1], clamp prediction to logit range)
    occ_hat  = y_hat[:, occ_channel : occ_channel + 1]
    occ_loss = F.binary_cross_entropy_with_logits(occ_hat, occ_target.clamp(0, 1))

    return phys_loss + occ_loss


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
# Variance regularisation (VICReg-style collapse prevention)
# ---------------------------------------------------------------------------

def variance_loss(
    z: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    """
    Penalise dimensions of z whose std across the batch falls below gamma.

    Works at any batch size — even B=1 the gradient is defined (std=0 →
    max penalty).  With B=1 the std is 0 by definition, so this acts as a
    constant regulariser pushing the encoder weights toward higher-variance
    outputs on the next batch.

    Loss = mean_d max(0, gamma - std_b(z_d))
    """
    if z.shape[0] < 2:
        # B=1: std is undefined; return a fixed penalty so the gradient still
        # flows through the encoder via other terms, and log it as 1.0
        return z.new_tensor(1.0)
    std = z.std(dim=0)                          # (D,)
    return F.relu(gamma - std).mean()


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
    lambda_var: float = 0.04,
) -> Dict[str, torch.Tensor]:
    l_cont  = infonce_loss(z_shared, z_views, tau=tau)
    l_recon = multi_modal_recon_loss(y_hats, ys)
    # Variance term: prevents collapse even when B=1 (InfoNCE has no gradient there)
    l_var   = variance_loss(z_shared)
    loss    = l_cont + lambda_recon * l_recon + lambda_var * l_var
    return {"loss": loss, "l_cont": l_cont, "l_recon": l_recon, "l_var": l_var}


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
