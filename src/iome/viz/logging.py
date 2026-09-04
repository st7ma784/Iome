"""
Wandb logging helpers for reconstruction panels, RMSE maps, and latent diagnostics.

Each helper returns a wandb.Image (or dict of them) ready for log().
"""

import io
from typing import Dict, List, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ---------------------------------------------------------------------------
# Reconstruction panel (4 rows: input | target | prediction | error)
# ---------------------------------------------------------------------------

def recon_panel(
    x: torch.Tensor,
    y: torch.Tensor,
    y_hat: torch.Tensor,
    channel_names: List[str],
    title: str = "",
    vmax_abs: Optional[float] = None,
) -> "wandb.Image":
    """
    Build a [4 × C] grid: input / target / prediction / absolute error.

    Args:
        x, y, y_hat: (C, H, W) tensors (already denormalised or raw)
        channel_names: length-C list of channel labels
    """
    import wandb
    C = x.shape[0]
    fig, axes = plt.subplots(4, C, figsize=(3 * C, 12), squeeze=False)
    rows = ["Input", "Target", "Prediction", "|Error|"]
    data = [x, y, y_hat, (y_hat - y).abs()]

    for row_idx, (row_label, tensor) in enumerate(zip(rows, data)):
        for c in range(C):
            ax  = axes[row_idx][c]
            arr = tensor[c].cpu().float().numpy()
            vm  = vmax_abs or max(abs(arr.min()), abs(arr.max()), 1e-6)
            cmap = "RdBu_r" if row_idx < 3 else "hot"
            im = ax.imshow(arr, cmap=cmap, vmin=-vm if row_idx < 3 else 0,
                           vmax=vm, origin="upper")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if row_idx == 0:
                ax.set_title(channel_names[c], fontsize=8)
            if c == 0:
                ax.set_ylabel(row_label, fontsize=8)
            ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    img = _fig_to_wandb(fig)
    plt.close(fig)
    return img


# ---------------------------------------------------------------------------
# RMSE spatial map
# ---------------------------------------------------------------------------

def rmse_map(
    errors: torch.Tensor,
    title: str = "RMSE",
) -> "wandb.Image":
    """
    Args:
        errors: (N, C, H, W) absolute errors over validation set
    Returns wandb.Image of (C, H, W) RMS map.
    """
    import wandb
    rms = errors.pow(2).mean(dim=0).sqrt().cpu().numpy()  # (C, H, W)
    C   = rms.shape[0]
    fig, axes = plt.subplots(1, C, figsize=(3 * C, 3), squeeze=False)
    for c in range(C):
        ax = axes[0][c]
        im = ax.imshow(rms[c], cmap="viridis", origin="upper")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"ch {c}", fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    plt.tight_layout()
    img = _fig_to_wandb(fig)
    plt.close(fig)
    return img


# ---------------------------------------------------------------------------
# Lead-time RMSE curve
# ---------------------------------------------------------------------------

def lead_time_curve(
    lead_rmse: Dict[int, float],
    title: str = "Lead-time RMSE",
) -> "wandb.Image":
    """
    Args:
        lead_rmse: {lead_step: scalar_rmse}
    """
    import wandb
    leads  = sorted(lead_rmse)
    rmses  = [lead_rmse[l] for l in leads]
    labels = [f"{l*2} min" for l in leads]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(leads, rmses, marker="o")
    ax.set_xticks(leads)
    ax.set_xticklabels(labels, rotation=45, fontsize=7)
    ax.set_ylabel("RMSE")
    ax.set_title(title)
    plt.tight_layout()
    img = _fig_to_wandb(fig)
    plt.close(fig)
    return img


# ---------------------------------------------------------------------------
# Cross-modal reconstruction panel
# ---------------------------------------------------------------------------

def cross_modal_panel(
    z_source_mod: str,
    z_target_mod: str,
    z: torch.Tensor,
    y_hat: torch.Tensor,
    y: torch.Tensor,
    channel_names: List[str],
) -> "wandb.Image":
    """
    Show what modality B's decoder produces when given modality A's encoder output.
    Useful for auditing alignment / what each head ignores as noise.
    """
    title = f"Encode {z_source_mod} → Decode {z_target_mod}"
    return recon_panel(
        x=y,     # show target as "input" reference
        y=y,
        y_hat=y_hat,
        channel_names=channel_names,
        title=title,
    )


# ---------------------------------------------------------------------------
# Latent UMAP
# ---------------------------------------------------------------------------

def latent_umap(
    z_dict: Dict[str, torch.Tensor],
    title: str = "Latent UMAP",
) -> "wandb.Image":
    """
    Project per-modality latent clouds to 2D with UMAP.

    Args:
        z_dict: {mod: (N, D)} tensors
    """
    import wandb
    try:
        import umap
    except ImportError:
        return None

    colours = {"sd": "steelblue", "smag": "tomato", "tec": "seagreen"}
    all_z, labels = [], []
    for mod, z in z_dict.items():
        z_np = z.cpu().float().numpy()
        all_z.append(z_np)
        labels.extend([mod] * len(z_np))

    all_z  = np.concatenate(all_z, axis=0)
    emb    = umap.UMAP(n_components=2, random_state=42).fit_transform(all_z)

    fig, ax = plt.subplots(figsize=(6, 5))
    offset  = 0
    for mod, z in z_dict.items():
        n = len(z)
        ax.scatter(emb[offset:offset + n, 0], emb[offset:offset + n, 1],
                   c=colours.get(mod, "grey"), label=mod, alpha=0.5, s=8)
        offset += n
    ax.legend()
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    img = _fig_to_wandb(fig)
    plt.close(fig)
    return img


# ---------------------------------------------------------------------------
# Alignment / uniformity scalars (Wang & Isola 2020)
# ---------------------------------------------------------------------------

def alignment_uniformity(
    z1: torch.Tensor,
    z2: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute alignment and uniformity from two (N, D) normalised latent tensors.
    """
    z1 = torch.nn.functional.normalize(z1.float(), dim=1)
    z2 = torch.nn.functional.normalize(z2.float(), dim=1)

    align    = (z1 - z2).pow(2).sum(dim=1).mean().item()
    uniform1 = torch.pdist(z1).pow(2).mul(-2).exp().mean().log().item()
    uniform2 = torch.pdist(z2).pow(2).mul(-2).exp().mean().log().item()
    return {"align": align, "uniform": (uniform1 + uniform2) / 2}


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _fig_to_wandb(fig) -> "wandb.Image":
    import wandb
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    return wandb.Image(buf)
