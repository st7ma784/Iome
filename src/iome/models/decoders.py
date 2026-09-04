"""
Per-modality decoders: latent vector → observation grid.

Pure decoders — no solar-wind conditioning.  They reconstruct each modality's
observation space from the shared latent z.  Zero-initialised output layers so
training starts from persistence (predict no change) rather than random noise.
"""

import math
import torch
import torch.nn as nn

from .layers import PatchRecovery2D, UpSample, EarthSpecificBlock
from iome.data.grid import NLAT, NMLT

PATCH = 4


# ---------------------------------------------------------------------------
# SuperDARN decoder
# ---------------------------------------------------------------------------

class SuperDARNDecoder(nn.Module):
    """
    Decodes (B, latent_dim) → (B, 6, NLAT, NMLT) SuperDARN convection grid.

    Mirrors the Pangu decoder path (linear projection → upsample → EarthSpecificBlocks
    → patch recovery) without FiLM layers.

    The final ConvTranspose2d is zero-initialised so the network starts from the
    persistence forecast (decoder predicts zero delta).
    """

    def __init__(
        self,
        latent_dim: int = 256,
        embed_dim: int = 128,
        num_heads: tuple = (16, 8),
        window_size: tuple = (2, 8, 16),
        out_chans: int = 6,
        grid_h: int = NLAT,
        grid_w: int = NMLT,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        # Non-square: NLAT=180, NMLT=360, PATCH=4 → 45×90 patches at hi-res
        n_ph    = math.ceil(grid_h / PATCH)       # 45
        n_pw    = math.ceil(grid_w / PATCH)       # 90
        n_ph_ds = math.ceil(n_ph / 2)             # 23
        n_pw_ds = math.ceil(n_pw / 2)             # 45

        res_hi = (1, n_ph, n_pw)
        res_lo = (1, n_ph_ds, n_pw_ds)

        bottleneck_dim = embed_dim * 2
        self.in_proj = nn.Linear(latent_dim, bottleneck_dim * n_ph_ds * n_pw_ds)
        self._res_lo   = res_lo
        self._n_lo_h   = n_ph_ds
        self._n_lo_w   = n_pw_ds
        self._n_hi_h   = n_ph
        self._n_hi_w   = n_pw

        self.dec_lo = nn.Sequential(*[
            EarthSpecificBlock(
                dim=bottleneck_dim, input_resolution=res_lo,
                num_heads=num_heads[0], window_size=window_size,
                shift_size=(0, 0, 0) if i % 2 == 0 else (1, window_size[1] // 2, window_size[2] // 2),
                mlp_ratio=mlp_ratio,
            )
            for i in range(2)
        ])

        self.upsample = UpSample(bottleneck_dim, embed_dim, res_lo, res_hi)

        self.dec_hi = nn.Sequential(*[
            EarthSpecificBlock(
                dim=embed_dim, input_resolution=res_hi,
                num_heads=num_heads[1], window_size=window_size,
                shift_size=(0, 0, 0) if i % 2 == 0 else (1, window_size[1] // 2, window_size[2] // 2),
                mlp_ratio=mlp_ratio,
            )
            for i in range(2)
        ])

        self.patch_recovery = PatchRecovery2D(
            img_size=(grid_h, grid_w),
            patch_size=(PATCH, PATCH),
            in_chans=embed_dim,
            out_chans=out_chans,
        )
        nn.init.zeros_(self.patch_recovery.conv.weight)
        nn.init.zeros_(self.patch_recovery.conv.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim)
        Returns:
            y: (B, 6, NLAT, NMLT)
        """
        B = z.shape[0]
        tokens = self.in_proj(z).reshape(B, self._n_lo_h * self._n_lo_w, -1)
        tokens = self.dec_lo(tokens)
        tokens = self.upsample(tokens)
        tokens = self.dec_hi(tokens)
        tokens = tokens.transpose(1, 2).reshape(B, -1, self._n_hi_h, self._n_hi_w)
        return self.patch_recovery(tokens)


# ---------------------------------------------------------------------------
# SuperMAG decoder
# ---------------------------------------------------------------------------

class SuperMAGDecoder(nn.Module):
    """
    Decodes (B, latent_dim) → (B, 3, NLAT, NMLT) magnetometer perturbation grid.
    """

    def __init__(self, latent_dim: int = 256, out_chans: int = 3,
                 grid_h: int = NLAT, grid_w: int = NMLT):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        # Project to 12×23, then 4× 2x upsample → 192×368, crop to NLAT×NMLT
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, 256 * 12 * 23),
            nn.GELU(),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),   # 12×23→24×46
            nn.GroupNorm(16, 128), nn.GELU(),
            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1),   # 24×46→48×92
            nn.GroupNorm(8,  64),  nn.GELU(),
            nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1),   # 48×92→96×184
            nn.GroupNorm(4,  32),  nn.GELU(),
            nn.ConvTranspose2d(32,  32,  4, stride=2, padding=1),   # 96×184→192×368
            nn.GroupNorm(4,  32),  nn.GELU(),
            nn.Conv2d(32, out_chans, 3, padding=1),                  # crop to NLAT×NMLT
        )
        nn.init.zeros_(self.up[-1].weight)
        nn.init.zeros_(self.up[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim)
        Returns:
            y: (B, 3, NLAT, NMLT)
        """
        B = z.shape[0]
        x = self.proj(z).reshape(B, 256, 12, 23)
        x = self.up(x)
        h, w = x.shape[2], x.shape[3]
        pt = (h - self.grid_h) // 2
        pl = (w - self.grid_w) // 2
        return x[:, :, pt:pt + self.grid_h, pl:pl + self.grid_w]


# ---------------------------------------------------------------------------
# TEC decoder
# ---------------------------------------------------------------------------

class TECDecoder(nn.Module):
    """
    Decodes (B, latent_dim) → (B, 2, NLAT, NMLT) GPS-TEC grid.

    TEC maps are smooth so the decoder doesn't need as many parameters as
    the SuperDARN decoder.
    """

    def __init__(self, latent_dim: int = 256, out_chans: int = 2,
                 grid_h: int = NLAT, grid_w: int = NMLT):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        # Project to 12×23, then 4× 2x upsample → 192×368, crop to NLAT×NMLT
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, 128 * 12 * 23),
            nn.GELU(),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),    # 12×23→24×46
            nn.GroupNorm(8, 64), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),     # 24×46→48×92
            nn.GroupNorm(4, 32), nn.GELU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),     # 48×92→96×184
            nn.GroupNorm(2, 16), nn.GELU(),
            nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1),     # 96×184→192×368
            nn.GroupNorm(2, 16), nn.GELU(),
            nn.Conv2d(16, out_chans, 3, padding=1),
        )
        nn.init.zeros_(self.up[-1].weight)
        nn.init.zeros_(self.up[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim)
        Returns:
            y: (B, 2, NLAT, NMLT)
        """
        B = z.shape[0]
        x = self.proj(z).reshape(B, 128, 12, 23)
        x = self.up(x)
        h, w = x.shape[2], x.shape[3]
        pt = (h - self.grid_h) // 2
        pl = (w - self.grid_w) // 2
        return x[:, :, pt:pt + self.grid_h, pl:pl + self.grid_w]


# ---------------------------------------------------------------------------
# DMSP decoder
# ---------------------------------------------------------------------------

class DMSPDecoder(nn.Module):
    """
    Decodes (B, latent_dim) → (B, 5, NLAT, NMLT) DMSP particle precipitation grid.
    """

    def __init__(self, latent_dim: int = 256, out_chans: int = 5,
                 grid_h: int = NLAT, grid_w: int = NMLT):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, 128 * 12 * 23),
            nn.GELU(),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),    # 12×23→24×46
            nn.GroupNorm(8, 64), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),     # 24×46→48×92
            nn.GroupNorm(4, 32), nn.GELU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),     # 48×92→96×184
            nn.GroupNorm(2, 16), nn.GELU(),
            nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1),     # 96×184→192×368
            nn.GroupNorm(2, 16), nn.GELU(),
            nn.Conv2d(16, out_chans, 3, padding=1),
        )
        nn.init.zeros_(self.up[-1].weight)
        nn.init.zeros_(self.up[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim)
        Returns:
            y: (B, 5, NLAT, NMLT)
        """
        B = z.shape[0]
        x = self.proj(z).reshape(B, 128, 12, 23)
        x = self.up(x)
        h, w = x.shape[2], x.shape[3]
        pt = (h - self.grid_h) // 2
        pl = (w - self.grid_w) // 2
        return x[:, :, pt:pt + self.grid_h, pl:pl + self.grid_w]
