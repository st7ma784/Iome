"""
UnifiedIonosphereModel: assembles encoders, latent dynamics, and decoders.

Fusion strategy: availability-weighted mean of per-modality latents.
Missing or dropped modalities contribute nothing to the fused z.

Modality dropout (training only):
    Each call to forward() can randomly zero out modalities before encoding.
    The model then decodes ALL requested modalities from the reduced z — this
    is the cross-modal reconstruction signal that makes the latent transferable
    across any subset of sensors.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .encoders import SuperDARNEncoder, SuperMAGEncoder, TECEncoder, DMSPEncoder
from .dynamics import LatentDynamics
from .decoders import SuperDARNDecoder, SuperMAGDecoder, TECDecoder, DMSPDecoder

MODALITIES = ("sd", "smag", "tec", "dmsp")


class UnifiedIonosphereModel(nn.Module):
    """
    Multi-modal ionospheric nowcast + forecast model.

    Forward pass (training, with modality dropout):
        xs_enc    = dropout(xs, p_mod_drop)       ← encode only surviving modalities
        z_views   = {mod: encoder(xs_enc[mod])}
        z_shared  = weighted_mean(z_views)
        z_next    = dynamics(z_shared, u, omni_mask)
        y_hats    = {mod: decoder(z_shared) for mod in decode_mods}  ← includes dropped ones

    The loss module computes reconstruction vs xs (all available, including dropped)
    and InfoNCE only on the encoded views (xs_enc).

    Args:
        latent_dim:  shared latent dimension D
        u_dim:       OMNI feature dimension (default 8)
        dyn_hidden:  hidden width of the LatentDynamics MLP
        dyn_layers:  depth of the LatentDynamics MLP
    """

    def __init__(
        self,
        latent_dim: int = 256,
        u_dim: int = 8,
        dyn_hidden: int = 512,
        dyn_layers: int = 4,
        wind_window_size: int = 1,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # --- Per-modality encoders -------------------------------------------
        self.encoders = nn.ModuleDict({
            "sd":   SuperDARNEncoder(latent_dim=latent_dim),
            "smag": SuperMAGEncoder(latent_dim=latent_dim),
            "tec":  TECEncoder(latent_dim=latent_dim),
            "dmsp": DMSPEncoder(latent_dim=latent_dim),
        })

        # Learnable per-modality reliability weights (log-scale for positivity)
        self.log_mod_weights = nn.ParameterDict({
            mod: nn.Parameter(torch.zeros(1)) for mod in MODALITIES
        })

        # --- Latent dynamics (FiLM lives here) --------------------------------
        self.dynamics = LatentDynamics(
            latent_dim=latent_dim,
            u_dim=u_dim,
            hidden_dim=dyn_hidden,
            n_layers=dyn_layers,
            wind_window_size=wind_window_size,
        )

        # --- Per-modality decoders --------------------------------------------
        self.decoders = nn.ModuleDict({
            "sd":   SuperDARNDecoder(latent_dim=latent_dim),
            "smag": SuperMAGDecoder(latent_dim=latent_dim),
            "tec":  TECDecoder(latent_dim=latent_dim),
            "dmsp": DMSPDecoder(latent_dim=latent_dim),
        })

    # ------------------------------------------------------------------
    # Modality dropout (call before encode during training)
    # ------------------------------------------------------------------

    def apply_mod_drop(
        self,
        xs: Dict[str, torch.Tensor],
        p_drop: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Randomly drop modalities from xs with probability p_drop each.
        Always retains at least one modality.

        Args:
            xs:     dict of {mod: (B, C, H, W)}
            p_drop: per-modality drop probability (0 = no dropout)

        Returns:
            xs_enc: subset of xs after dropout
        """
        if p_drop <= 0.0 or not self.training:
            return xs
        keys = list(xs.keys())
        kept = [k for k in keys if torch.rand(1).item() > p_drop]
        if not kept:
            kept = [keys[int(torch.randint(len(keys), (1,)).item())]]
        return {k: xs[k] for k in kept}

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(
        self,
        xs: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Encode all modalities in xs and fuse into z_shared.

        Args:
            xs: {mod: (B, C, H, W)} for whichever modalities are available

        Returns:
            z_views:  {mod: (B, D)} per-modality latents
            z_shared: (B, D) reliability-weighted mean latent
        """
        z_views: Dict[str, torch.Tensor] = {}
        for mod in MODALITIES:
            if mod in xs:
                z_views[mod] = self.encoders[mod](xs[mod])

        z_shared = self._weighted_fuse(z_views)
        return z_views, z_shared

    def _weighted_fuse(self, z_views: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Reliability-weighted mean of available per-modality latents.
        Weights are exp(log_mod_weight) — always positive, learned end-to-end.
        """
        if not z_views:
            raise ValueError("At least one modality must be present.")
        weights = {mod: self.log_mod_weights[mod].exp() for mod in z_views}
        total_w = sum(weights.values())
        fused = sum(w * z_views[mod] for mod, w in weights.items()) / total_w
        return fused

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(
        self,
        z: torch.Tensor,
        modalities: Optional[tuple] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Decode z through specified modality decoders.

        Args:
            z:          (B, D)
            modalities: which decoders to run; defaults to all four.
        """
        if modalities is None:
            modalities = MODALITIES
        return {mod: self.decoders[mod](z) for mod in modalities if mod in self.decoders}

    # ------------------------------------------------------------------
    # Full forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        xs: Dict[str, torch.Tensor],
        u: torch.Tensor,
        omni_mask: Optional[torch.Tensor] = None,
        omni_mask_window: Optional[torch.Tensor] = None,
        decode_mods: Optional[tuple] = None,
        p_mod_drop: float = 0.0,
    ) -> Dict[str, object]:
        """
        Full nowcast + one-step forecast with optional modality dropout.

        Args:
            xs:               {mod: (B, C, H, W)} all available observation tensors at t
            u:                (B, u_dim) single OMNI snapshot, OR
                              (B, K, u_dim) past-K window when wind_window_size > 1
            omni_mask:        (B, 1) float — 1.0 when u is valid (single-snapshot mode)
            omni_mask_window: (B, K) float — per-step mask (window mode)
            decode_mods:      modalities to reconstruct (default: all keys in xs)
            p_mod_drop:       per-modality drop probability (applied during training only)

        Returns:
            dict with:
                xs_enc       — {mod: tensor} modalities actually encoded (after dropout)
                z_views      — {mod: (B, D)} per-modality latents (from xs_enc)
                z_shared     — (B, D) fused latent
                z_next       — (B, D) predicted latent at t+1
                y_hats       — {mod: (B, C, H, W)} reconstructions at t
                y_next_hats  — {mod: (B, C, H, W)} forecast at t+1
                wind_weights — (B, K) OMNI alignment weights, or None
        """
        if decode_mods is None:
            decode_mods = tuple(xs.keys())

        # Apply modality dropout — xs_enc ⊆ xs
        xs_enc = self.apply_mod_drop(xs, p_mod_drop)

        z_views, z_shared = self.encode(xs_enc)
        z_next, wind_weights = self.dynamics(z_shared, u, omni_mask, omni_mask_window)

        # Decode ALL requested modalities (includes cross-modal reconstruction
        # for any that were dropped from xs_enc but are still in decode_mods)
        y_hats      = self.decode(z_shared, decode_mods)
        y_next_hats = self.decode(z_next,   decode_mods)

        return {
            "xs_enc":      xs_enc,
            "z_views":     z_views,
            "z_shared":    z_shared,
            "z_next":      z_next,
            "y_hats":      y_hats,
            "y_next_hats": y_next_hats,
            "wind_weights": wind_weights,
        }

    # ------------------------------------------------------------------
    # Stage-specific parameter groups (for frozen-encoder training)
    # ------------------------------------------------------------------

    def encoder_parameters(self):
        return list(self.encoders.parameters()) + list(self.log_mod_weights.parameters())

    def dynamics_parameters(self):
        return self.dynamics.parameters()

    def decoder_parameters(self):
        return self.decoders.parameters()

    def non_encoder_parameters(self):
        """Dynamics + decoders — used when encoders are frozen in Stage 2."""
        return (list(self.dynamics.parameters()) +
                list(self.decoders.parameters()))
