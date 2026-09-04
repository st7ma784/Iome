"""
Per-modality encoders: pure observation → latent vector.

No solar-wind conditioning here.  FiLM layers live exclusively in
LatentDynamics (dynamics.py) so that each encoder is a clean, reusable
observation model that does not assume any particular driver dataset is present.
"""

import math
import numpy as np
import torch
import torch.nn as nn

from .layers import (
    PatchEmbed2D, DownSample, EarthSpecificBlock,
)
from iome.data.grid import NLAT, NMLT

PATCH = 4   # patch size for SuperDARN patch-embed


# ---------------------------------------------------------------------------
# SuperDARN encoder
# ---------------------------------------------------------------------------

class SuperDARNEncoder(nn.Module):
    """
    Encodes a (B, 6, NLAT, NMLT) SuperDARN convection grid to a latent vector.

    Mirrors the Pangu encoder path (patch-embed → EarthSpecificBlocks →
    downsample → deeper blocks → global avg pool → projection) but with all
    FiLM layers removed.  Solar-wind conditioning is the dynamics model's job.

    Args:
        latent_dim: output latent dimension D
        embed_dim:  patch-embedding width (default 128, matching SuperDARN Pangu)
        num_heads:  attention heads at each resolution
        window_size: 3-tuple for the spatial window (pl, lat, lon)
        drop_path_rate: stochastic depth rate across the 8 blocks
    """

    def __init__(
        self,
        latent_dim: int = 256,
        embed_dim: int = 128,
        num_heads: tuple = (8, 16),
        window_size: tuple = (2, 8, 16),
        drop_path_rate: float = 0.1,
        in_chans: int = 6,
        grid_h: int = NLAT,
        grid_w: int = NMLT,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        # Non-square patch grid: NLAT=180, NMLT=360, PATCH=4 → 45×90 patches
        n_ph = math.ceil(grid_h / PATCH)       # 45 in lat
        n_pw = math.ceil(grid_w / PATCH)       # 90 in lon
        n_ph_ds = math.ceil(n_ph / 2)          # 23 after downsample
        n_pw_ds = math.ceil(n_pw / 2)          # 45 after downsample

        # Single "pressure level" → (1, n_ph, n_pw)
        res_hi = (1, n_ph, n_pw)
        res_lo = (1, n_ph_ds, n_pw_ds)

        dp = np.linspace(0, drop_path_rate, 4).tolist()

        self.patch_embed = PatchEmbed2D(
            img_size=(grid_h, grid_w),
            patch_size=(PATCH, PATCH),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # Shallow encoder at full patch resolution
        self.enc_hi = nn.Sequential(*[
            EarthSpecificBlock(
                dim=embed_dim, input_resolution=res_hi,
                num_heads=num_heads[0], window_size=window_size,
                shift_size=(0, 0, 0) if i % 2 == 0 else (1, window_size[1] // 2, window_size[2] // 2),
                mlp_ratio=mlp_ratio, drop_path=dp[i],
            )
            for i in range(2)
        ])

        self.downsample = DownSample(embed_dim, res_hi, res_lo)

        # Deeper encoder at half patch resolution
        self.enc_lo = nn.Sequential(*[
            EarthSpecificBlock(
                dim=embed_dim * 2, input_resolution=res_lo,
                num_heads=num_heads[1], window_size=window_size,
                shift_size=(0, 0, 0) if i % 2 == 0 else (1, window_size[1] // 2, window_size[2] // 2),
                mlp_ratio=mlp_ratio, drop_path=dp[i + 2],
            )
            for i in range(2)
        ])

        bottleneck_dim = embed_dim * 2
        self.pool = nn.Sequential(
            nn.LayerNorm(bottleneck_dim),
            nn.Linear(bottleneck_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 6, NLAT, NMLT) normalised SuperDARN grid
        Returns:
            z: (B, latent_dim)
        """
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)   # (B, N, C)
        tokens = self.enc_hi(tokens)
        tokens = self.downsample(tokens)
        tokens = self.enc_lo(tokens)
        z = self.pool(tokens.mean(dim=1))                          # (B, D)
        return z


# ---------------------------------------------------------------------------
# SuperMAG encoder
# ---------------------------------------------------------------------------

class SuperMAGEncoder(nn.Module):
    """
    Encodes a (B, 3, NLAT, NMLT) magnetometer perturbation grid to a latent vector.

    Magnetometer stations are much sparser than SuperDARN radars, so a shallower
    convolutional architecture is appropriate.  No FiLM — solar conditioning is
    handled downstream in the dynamics model.

    Channels: dBn (nT), dBe (nT), soft_occ
    """

    def __init__(self, latent_dim: int = 256, in_chans: int = 3, grid_h: int = NLAT, grid_w: int = NMLT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, 64, 7, padding=3, bias=False),
            nn.GroupNorm(8, 64), nn.GELU(),

            nn.Conv2d(64, 128, 5, padding=2, bias=False),
            nn.GroupNorm(16, 128), nn.GELU(),

            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, 128), nn.GELU(),

            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, 256), nn.GELU(),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, NLAT, NMLT)
        Returns:
            z: (B, latent_dim)
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# TEC encoder
# ---------------------------------------------------------------------------

class TECEncoder(nn.Module):
    """
    Encodes a (B, 2, NLAT, NMLT) GPS-TEC grid to a latent vector.

    TEC maps are spatially smooth (GPS coverage is denser and more uniform than
    ground magnetometers), so a small ResNet-style stack captures the signal well.
    No FiLM — solar conditioning is handled in the dynamics model.

    Channels: VTEC (TECU), dVTEC/dt (TECU / 2min)
    """

    def __init__(self, latent_dim: int = 256, in_chans: int = 2, grid_h: int = NLAT, grid_w: int = NMLT):
        super().__init__()

        def resblock(c):
            return nn.Sequential(
                nn.Conv2d(c, c, 3, padding=1, bias=False),
                nn.GroupNorm(min(32, c), c), nn.GELU(),
                nn.Conv2d(c, c, 3, padding=1, bias=False),
                nn.GroupNorm(min(32, c), c),
            )

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, 64, 7, padding=3, bias=False),
            nn.GroupNorm(8, 64), nn.GELU(),
        )
        self.res1 = resblock(64)
        self.act1 = nn.GELU()

        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, 128), nn.GELU(),
        )
        self.res2 = resblock(128)
        self.act2 = nn.GELU()

        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, 256), nn.GELU(),
        )
        self.res3 = resblock(256)
        self.act3 = nn.GELU()

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2, NLAT, NMLT)
        Returns:
            z: (B, latent_dim)
        """
        x = self.stem(x)
        x = self.act1(x + self.res1(x))
        x = self.down1(x)
        x = self.act2(x + self.res2(x))
        x = self.down2(x)
        x = self.act3(x + self.res3(x))
        return self.head(x)


# ---------------------------------------------------------------------------
# DMSP encoder
# ---------------------------------------------------------------------------

class DMSPEncoder(nn.Module):
    """
    Encodes a (B, 5, NLAT, NMLT) DMSP particle precipitation grid to a latent vector.

    DMSP tracks are even sparser than SuperMAG stations (~15 passes/day per sat),
    so a lightweight CNN with AdaptiveAvgPool is appropriate.

    Channels: log1p(e_flux), log1p(e_energy), log1p(i_flux), log1p(i_energy), soft_occ
    """

    def __init__(self, latent_dim: int = 256, in_chans: int = 5, grid_h: int = NLAT, grid_w: int = NMLT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, 64, 7, padding=3, bias=False),
            nn.GroupNorm(8, 64), nn.GELU(),

            nn.Conv2d(64, 128, 5, padding=2, bias=False),
            nn.GroupNorm(16, 128), nn.GELU(),

            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, 128), nn.GELU(),

            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, 256), nn.GELU(),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 5, NLAT, NMLT)
        Returns:
            z: (B, latent_dim)
        """
        return self.net(x)
