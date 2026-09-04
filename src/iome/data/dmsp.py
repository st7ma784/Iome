"""
DMSP SSJ4/5 particle precipitation dataset → (5, 180, 360) global grid.

Files are produced by scripts/ingest_dmsp.py.
Each file covers one 2-min epoch: YYYYMMDDTHHMM_dmsp.npy

Grid channels (log1p-scaled, then normalised):
  0: log1p(e_flux)    — electron total energy flux  (eV/cm²/ster/s)
  1: log1p(e_energy)  — electron characteristic energy (eV)
  2: log1p(i_flux)    — ion total energy flux
  3: log1p(i_energy)  — ion characteristic energy (eV)
  4: soft_occ         — Gaussian-weighted track coverage ∈ [0, 1]
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .grid import NLAT, NMLT

N_CHANS = 5


class DMSPDataset(Dataset):
    """
    Args:
        timestamps:    sorted list of "YYYYMMDDTHHMM" strings
        cache_dir:     directory of *_dmsp.npy files
        stats:         {"mean": (5,), "std": (5,)} normalisation
        delta_t_steps: steps between x_t and x_{t+1}
    """

    def __init__(
        self,
        timestamps: list[str],
        cache_dir: Optional[Path] = None,
        stats: Optional[dict] = None,
        delta_t_steps: int = 1,
    ):
        self.timestamps    = timestamps
        self.cache_dir     = Path(cache_dir) if cache_dir else None
        self.stats         = stats
        self.delta_t_steps = delta_t_steps
        self._valid        = list(range(len(timestamps) - delta_t_steps))

    def __len__(self):
        return len(self._valid)

    def __getitem__(self, idx):
        i = self._valid[idx]
        x      = self._load(self.timestamps[i])
        x_next = self._load(self.timestamps[i + self.delta_t_steps])
        return {"dmsp": x, "dmsp_next": x_next, "ts": self.timestamps[i]}

    def _load(self, ts: str) -> torch.Tensor:
        fname = ts.replace("-", "").replace(":", "") + "_dmsp.npy"
        if self.cache_dir:
            p = self.cache_dir / fname
            if p.exists():
                grid = np.load(p, mmap_mode="r").astype(np.float32)
                return self._normalise(grid)
        return self._normalise(np.zeros((N_CHANS, NLAT, NMLT), dtype=np.float32))

    def _normalise(self, grid: np.ndarray) -> torch.Tensor:
        if self.stats is not None:
            mean = self.stats["mean"][:, None, None]
            std  = self.stats["std"][:, None, None]
            grid = (grid - mean) / std.clip(min=1e-6)
        return torch.from_numpy(grid)
