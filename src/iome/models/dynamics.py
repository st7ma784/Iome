"""
Latent dynamics model with FiLM solar-wind conditioning.

This is the "intermediate model" that sits between the per-modality encoders
and the per-modality decoders.  All FiLM layers live here — not in the encoders.

Design rationale
----------------
FiLM (Feature-wise Linear Modulation) is a mechanism for conditioning a neural
network on an external scalar signal without entangling that signal with the
observation-specific processing.  Placing FiLM in the dynamics model means:

  - Encoders are pure, reusable observation processors.
  - The solar-wind driver (OMNI) only influences how the latent STATE EVOLVES,
    not how a raw observation is mapped into the latent space.
  - Missing OMNI data (mask=0) passes through as an identity — the dynamics
    model degrades gracefully to an unconditional transition.
  - Adding / removing a modality does not require touching solar-wind logic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# FiLM primitive
# ---------------------------------------------------------------------------

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation conditioned on a solar-wind scalar vector.

        out = (1 + γ · mask) · x + β · mask

    mask=0 is an exact identity — the model is unchanged when OMNI data is absent.
    Initialised to near-zero so the network starts as unconditioned and learns
    how to use the driver signal over training.
    """

    def __init__(self, u_dim: int, feature_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(u_dim, feature_dim * 2),
            nn.SiLU(),
            nn.Linear(feature_dim * 2, feature_dim * 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, u: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, feature_dim) or (B, N, feature_dim)
            u:    (B, u_dim) OMNI feature vector
            mask: (B, 1) float — 1.0 when OMNI is available, 0.0 otherwise
        """
        gamma, beta = self.mlp(u).chunk(2, dim=-1)          # each (B, feature_dim)
        # Broadcast over any leading sequence dimension
        if x.ndim == 3:
            gamma = gamma.unsqueeze(1)
            beta  = beta.unsqueeze(1)
            mask  = mask.unsqueeze(-1)    # (B,1) → (B,1,1) for 3D x
        return x * (1.0 + gamma * mask) + beta * mask


# ---------------------------------------------------------------------------
# Latent dynamics model
# ---------------------------------------------------------------------------

class LatentDynamics(nn.Module):
    """
    Solar-wind-conditioned latent state transition:

        z_{t+1} = z_t + FiLM_MLP(z_t, u_t)

    FiLM is applied after every hidden layer so that the solar-wind driver can
    modulate the entire depth of the transition network, not just its output.
    This is more expressive than a single FiLM at the end and was validated in
    the SuperDARN Pangu model (film_enc, film_bot, film_bot2, film_dec) — here
    we reproduce that multi-layer conditioning philosophy in the dynamics model
    rather than scattering it across encoder stages.

    Args:
        latent_dim:  dimension of z (must match encoder output)
        u_dim:       OMNI feature dimension (Bx, By, Bz, |B|, Vx, n, Kp, ε = 8)
        hidden_dim:  width of the internal MLP layers
        n_layers:    number of FiLM-conditioned hidden layers
    """

    def __init__(
        self,
        latent_dim: int = 256,
        u_dim: int = 8,
        hidden_dim: int = 512,
        n_layers: int = 4,
    ):
        super().__init__()
        assert n_layers >= 1

        # Embed the OMNI driver once; each FiLMLayer then reads this embedding.
        self.u_embed = nn.Sequential(
            nn.Linear(u_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Input projection into hidden space
        self.in_proj = nn.Linear(latent_dim, hidden_dim)

        # FiLM-conditioned hidden layers
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
            for _ in range(n_layers)
        ])
        self.film = nn.ModuleList([
            FiLMLayer(hidden_dim, hidden_dim)      # conditioned on embedded u, not raw u
            for _ in range(n_layers)
        ])

        # Output projection back to latent space; zero-init for identity start
        self.out_proj = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Layer norm on the residual path
        self.norm = nn.LayerNorm(latent_dim)

    def forward(
        self,
        z: torch.Tensor,
        u: torch.Tensor,
        omni_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            z:          (B, latent_dim) current shared latent state
            u:          (B, u_dim) OMNI solar-wind features
            omni_mask:  (B, 1) float — 1.0 when u is valid, 0.0 if OMNI is missing.
                        If None, treated as all-ones (fully conditioned).
        Returns:
            z_next:     (B, latent_dim)
        """
        if omni_mask is None:
            omni_mask = torch.ones(z.shape[0], 1, device=z.device, dtype=z.dtype)

        u_emb = self.u_embed(u)          # (B, hidden_dim)

        h = self.in_proj(self.norm(z))   # (B, hidden_dim)
        for layer, film in zip(self.layers, self.film):
            h = film(layer(h), u_emb, omni_mask)

        delta = self.out_proj(h)
        return z + delta                 # residual: starts as identity (zero-init)
