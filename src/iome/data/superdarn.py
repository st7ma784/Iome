"""
SuperDARN dataset adapter.

Wraps the pre-saved daily .npy grids produced by ingest_superdarn.py.
Files are named YYYYMMDD_sd.npy (one per day, daily convection map).
Any 2-min timestamp in a given day maps to the same daily grid.

Grid shape: (6, 180, 360)  float16 on disk, float32 in tensors
Channels  : vlos_n, vlos_e, model_vlos_n, model_vlos_e, obs_occ, soft_occ
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .grid import NLAT, NMLT

N_CHANS = 6


class SuperDARNDataset(Dataset):
    """
    Args:
        timestamps:    sorted list of ISO-8601 strings "YYYY-MM-DDTHH:MM"
        cache_dir:     local directory of YYYYMMDD_sd.npy files
        stats:         dict with "mean" and "std" arrays of shape (6,)
        delta_t_steps: how many 2-min steps ahead xs_next is relative to xs
    """

    def __init__(
        self,
        timestamps: list[str],
        cache_dir: Path,
        stats: Optional[dict] = None,
        delta_t_steps: int = 1,
    ):
        self.timestamps    = timestamps
        self.cache_dir     = Path(cache_dir)
        self.stats         = stats
        self.delta_t_steps = delta_t_steps

        self._valid = list(range(len(timestamps) - delta_t_steps))

    def __len__(self):
        return len(self._valid)

    def __getitem__(self, idx):
        i = self._valid[idx]
        x      = self._load(self.timestamps[i])
        x_next = self._load(self.timestamps[i + self.delta_t_steps])
        return {"sd": x, "sd_next": x_next, "ts": self.timestamps[i]}

    # ------------------------------------------------------------------

    def _ts_to_date_key(self, ts: str) -> str:
        """'YYYY-MM-DDTHH:MM' or 'YYYYMMDDTHHMM' → 'YYYYMMDD'."""
        compact = ts.replace("-", "").replace(":", "")
        return compact[:8]

    def _load(self, ts: str) -> torch.Tensor:
        date_key = self._ts_to_date_key(ts)
        fname = f"{date_key}_sd.npy"

        path = self.cache_dir / fname
        if path.exists():
            try:
                grid = np.load(path, mmap_mode="r")
                if grid.shape != (N_CHANS, NLAT, NMLT):
                    raise ValueError(f"unexpected shape {grid.shape}")
            except Exception:
                grid = np.zeros((N_CHANS, NLAT, NMLT), dtype=np.float32)
        else:
            grid = np.zeros((N_CHANS, NLAT, NMLT), dtype=np.float32)

        grid = grid.astype(np.float32)
        if self.stats is not None:
            mean = self.stats["mean"][:, None, None]
            std  = self.stats["std"][:, None, None]
            grid = (grid - mean) / std.clip(min=1e-6)

        return torch.from_numpy(grid)
