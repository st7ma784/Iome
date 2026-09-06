"""
SuperMAG dataset: ground magnetometer perturbations → (3, 180, 360) global grid.

SuperMAG ASCII files contain per-station dBn, dBe in nT at 1-min resolution.
We bin to 2-min epochs to match SuperDARN cadence, then Gaussian-splat the
sparse station network onto the global equirectangular magnetic-coordinate grid.
Both hemispheres are included (|mlat| ≥ 40°).

Grid channels:
  0: dBn (nT)   — northward perturbation
  1: dBe (nT)   — eastward perturbation
  2: soft_occ   — Gaussian-weighted station coverage mask
"""

import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .grid import mlat_mlt_to_pixel, splat_to_grid, NLAT, NMLT

N_CHANS = 3


# ---------------------------------------------------------------------------
# SuperMAG file parser
# ---------------------------------------------------------------------------

def parse_supermag_file(raw_bytes: bytes) -> dict[str, np.ndarray]:
    """
    Parse a SuperMAG SuperMAG2ASCII (or flat CSV) file.

    Returns dict keyed by "YYYYMMDDTHHMMSS" → ndarray of shape (N_stations, 5):
        mlat, mlt, dBn, dBe, valid_flag
    """
    text   = raw_bytes.decode("utf-8", errors="replace")
    lines  = text.splitlines()
    epochs: dict[str, list] = {}

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            # Expected columns: YYYY MM DD HH MM SS SITE MLAT MLT dBn dBe
            year, mo, day, hr, mn, sc = map(int, parts[:6])
            mlat = float(parts[7])
            mlt  = float(parts[8])
            dbn  = float(parts[9])
            dbe  = float(parts[10])
        except (ValueError, IndexError):
            continue

        if abs(mlat) < 40.0:
            continue  # below |40°| mlat threshold

        ts_key = f"{year:04d}{mo:02d}{day:02d}T{hr:02d}{mn:02d}{sc:02d}"
        epochs.setdefault(ts_key, []).append([mlat, mlt, dbn, dbe, 1.0])

    return {k: np.array(v, dtype=np.float32) for k, v in epochs.items()}


def epoch_to_grid(records: np.ndarray) -> np.ndarray:
    """
    (N, 5) station records → (3, 180, 360) grid.
    """
    mlat, mlt, dbn, dbe = records[:, 0], records[:, 1], records[:, 2], records[:, 3]
    rows, cols = mlat_mlt_to_pixel(mlat, mlt)
    mask       = (0 <= rows) & (rows < NLAT)   # cols wrap on MLT axis
    rows, cols = rows[mask], cols[mask]
    dbn, dbe   = dbn[mask], dbe[mask]

    field_n, occ = splat_to_grid(rows, cols, dbn)
    field_e, _   = splat_to_grid(rows, cols, dbe)

    return np.stack([field_n, field_e, occ], axis=0)  # (3, 180, 360)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SuperMAGDataset(Dataset):
    """
    Args:
        timestamps:    sorted list of "YYYY-MM-DDTHH:MM" strings
        cache_dir:     local .npy cache directory
        stats:         {"mean": (3,), "std": (3,)} normalisation
        delta_t_steps: steps between xs and xs_next
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
        return {"smag": x, "smag_next": x_next, "ts": self.timestamps[i]}

    # ------------------------------------------------------------------

    def _load(self, ts: str) -> torch.Tensor:
        fname = ts.replace(":", "").replace("-", "") + "_smag.npy"
        p = self.cache_dir / fname
        if p.exists():
            try:
                grid = np.load(p, mmap_mode="r")
                if grid.shape != (3, NLAT, NMLT):
                    raise ValueError(f"unexpected shape {grid.shape}")
                return self._normalise(grid)
            except Exception:
                return torch.zeros(3, NLAT, NMLT)
        return torch.zeros(3, NLAT, NMLT)

    def _normalise(self, grid: np.ndarray) -> torch.Tensor:
        grid = grid.astype(np.float32)
        if self.stats is not None:
            mean = self.stats["mean"][:, None, None]
            std  = self.stats["std"][:, None, None]
            grid = (grid - mean) / std.clip(min=1e-6)
        return torch.from_numpy(grid)
