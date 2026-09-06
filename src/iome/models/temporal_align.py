"""
Soft temporal alignment for state-conditioned dynamic lag correction.

The causal lag between ionospheric modalities is not constant: it varies
with substorm intensity, solar wind Bz, and substorm phase.  A substorm
under strong driving (Bz << 0) may propagate to SuperDARN convection
in 8 minutes; a weak event may take 25 minutes.

TemporalAlignmentHead makes the lag a soft, state-dependent distribution
rather than a fixed offset.  Given z_anchor (the SuperMAG embedding,
which encodes current magnetospheric state) and a window of K target
embeddings z_window[t, t+1, ..., t+K-1], it outputs a weighted mixture
whose weights depend on z_anchor.

Differentiating through the lag:
  The attention weights ARE the lag distribution.  Gradients of the CLIP
  loss w.r.t. the weights indicate whether the selected lag was too early
  or too late, and flow back through the attention MLP to update the lag
  predictor.  Window encodings are detached so gradient cost is O(1)
  in K regardless of window size — the target encoder still receives
  gradients via the primary (non-window) CLIP and reconstruction paths.

Logging:
  Call .mean_lag() on the returned weights to get the expected lag in steps
  for each sample — useful for sanity-checking against substorm timelines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAlignmentHead(nn.Module):
    """
    Soft temporal attention: attend over a window of K target embeddings,
    with attention weights conditioned on the anchor (smag) state.

    Args:
        latent_dim:  dimension of z vectors (same for anchor and window)
        window_size: K — number of lag positions (0, 1, ..., K-1 steps)
        hidden_dim:  MLP hidden size (default latent_dim // 2)
    """

    def __init__(self, latent_dim: int, window_size: int, hidden_dim: int = None):
        super().__init__()
        h = hidden_dim or latent_dim // 2
        self.window_size = window_size
        self.attn = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, h),
            nn.GELU(),
            nn.Linear(h, window_size),
        )

    def forward(
        self,
        z_anchor: torch.Tensor,   # (B, D)  — smag embedding, drives lag prediction
        z_window: torch.Tensor,   # (B, K, D) — window of target embeddings, detached
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            z_aligned  (B, D)  — soft-attended target embedding
            attn_weights (B, K) — lag distribution (softmax, sums to 1)
        """
        logits  = self.attn(z_anchor)                              # (B, K)
        weights = torch.softmax(logits, dim=-1)                    # (B, K)
        z_out   = (weights.unsqueeze(-1) * z_window).sum(dim=1)   # (B, D)
        return z_out, weights

    @staticmethod
    def mean_lag(attn_weights: torch.Tensor) -> torch.Tensor:
        """Expected lag in steps: E[k] = Σ k * w_k.  Shape: (B,)."""
        K = attn_weights.shape[-1]
        lags = torch.arange(K, device=attn_weights.device, dtype=attn_weights.dtype)
        return (attn_weights * lags).sum(dim=-1)


def encode_window(
    encoder:    nn.Module,
    snapshots:  torch.Tensor,   # (B, K, C, H, W) — K snapshots per sample
    latent_dim: int,
) -> torch.Tensor:
    """
    Encode a window of K snapshots efficiently by folding K into the batch dim.

    Uses torch.no_grad() so window encodings are detached — gradients flow
    through the attention weights but not through the K encoder calls.

    Returns: (B, K, latent_dim)
    """
    B, K, C, H, W = snapshots.shape
    with torch.no_grad():
        flat = snapshots.view(B * K, C, H, W)
        z_flat = F.normalize(encoder(flat), dim=-1)   # (B*K, D)
    return z_flat.view(B, K, latent_dim).detach()
