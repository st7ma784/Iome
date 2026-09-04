"""
GPS-TEC dataset: IONEX → (2, 180, 360) global grid.

IONEX files contain global 2.5°×5° lat/lon VTEC maps at 1- or 2-hour cadence.
We:
  1. Parse VTEC maps from IONEX (using ionex package or manual parser).
  2. Compute dVTEC/dt (TECU per 2-min epoch) by differencing adjacent maps
     (interpolated to 2-min cadence to match SuperDARN).
  3. Convert geographic lat/lon to magnetic coordinates.
  4. Splat both fields onto the 180×360 global equirectangular magnetic grid.

Grid channels:
  0: VTEC (TECU)
  1: dVTEC/dt (TECU / 2 min)
"""

import io
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .grid import geo_latlon_to_pixel, splat_to_grid, NLAT, NMLT

try:
    from .luna import get_minio_bytes
except ImportError:
    def get_minio_bytes(**_):
        raise RuntimeError("minio not installed; set cache_dir instead")

BUCKET   = "tec-data"
N_CHANS  = 2
TECU_NAN = 9999.0   # IONEX fill value


# ---------------------------------------------------------------------------
# IONEX parser
# ---------------------------------------------------------------------------

def parse_ionex(raw_bytes: bytes) -> dict:
    """
    Minimal IONEX 1.0 parser.

    Returns:
        {
          "epoch":    list of datetime strings "YYYYMMDDTHHmmss",
          "lat":      (N_lat,) array
          "lon":      (N_lon,) array
          "vtec":     (N_epochs, N_lat, N_lon) float32 in TECU
        }
    """
    text   = raw_bytes.decode("ascii", errors="replace")
    lines  = text.splitlines()

    # Parse header
    lat_grid = lon_grid = None
    exp      = -1

    for line in lines:
        if "LAT1 / LAT2 / DLAT" in line:
            vals     = line[:60].split()
            lat1, lat2, dlat = float(vals[0]), float(vals[1]), float(vals[2])
            lat_grid = np.arange(lat1, lat2 + dlat / 2, dlat)
        if "LON1 / LON2 / DLON" in line:
            vals     = line[:60].split()
            lon1, lon2, dlon = float(vals[0]), float(vals[1]), float(vals[2])
            lon_grid = np.arange(lon1, lon2 + dlon / 2, dlon)
        if "EXPONENT" in line:
            exp = int(line[:60].split()[0])
        if "END OF HEADER" in line:
            break

    if lat_grid is None or lon_grid is None:
        raise ValueError("IONEX header missing LAT/LON grids")

    exponent = 10 ** exp
    epochs, maps = [], []
    i = 0

    while i < len(lines):
        line = lines[i]
        if "START OF TEC MAP" in line:
            epoch_str = None
            tec_rows  = []
            i += 1
            while i < len(lines) and "END OF TEC MAP" not in lines[i]:
                l = lines[i]
                if "EPOCH OF CURRENT MAP" in l:
                    parts     = l[:60].split()
                    yr, mo, d = int(parts[0]), int(parts[1]), int(parts[2])
                    hr, mn, s = int(parts[3]), int(parts[4]), int(parts[5])
                    epoch_str = f"{yr:04d}{mo:02d}{d:02d}T{hr:02d}{mn:02d}{s:02d}"
                elif re.match(r"\s*[-+]?\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+", l):
                    # Lat header row — skip
                    pass
                else:
                    # TEC values row
                    vals = [float(v) * exponent for v in l.split()]
                    if vals:
                        tec_rows.append(vals)
                i += 1
            if epoch_str and tec_rows:
                arr = np.array(tec_rows, dtype=np.float32)
                arr[arr >= TECU_NAN * exponent * 0.9] = np.nan
                epochs.append(epoch_str)
                maps.append(arr)
        i += 1

    return {
        "epoch": epochs,
        "lat":   lat_grid.astype(np.float32),
        "lon":   lon_grid.astype(np.float32),
        "vtec":  np.stack(maps, axis=0) if maps else np.empty((0, len(lat_grid), len(lon_grid))),
    }


# ---------------------------------------------------------------------------
# Grid projection
# ---------------------------------------------------------------------------

def ionex_to_polar_grid(
    vtec_map: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project a (N_lat, N_lon) VTEC map onto the 180×360 global grid.

    Returns (vtec_grid, occ) each of shape (NLAT, NMLT).
    """
    LON, LAT = np.meshgrid(lon_grid, lat_grid)  # both (N_lat, N_lon)
    mask     = np.isfinite(vtec_map)             # full globe
    rows, cols = geo_latlon_to_pixel(LAT[mask], LON[mask])
    values     = vtec_map[mask]
    ok = (rows >= 0) & (rows < NLAT)
    return splat_to_grid(rows[ok], cols[ok], values[ok], sigma=2.0)


def build_tec_grid(
    vtec_now: np.ndarray,
    vtec_prev: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
) -> np.ndarray:
    """
    Build a (2, 180, 360) TEC grid from two VTEC maps (TECU).

    vtec_prev is used only for the derivative channel; both arrays
    must be aligned to the same lat/lon grid.
    """
    vtec_field, occ   = ionex_to_polar_grid(vtec_now,  lat_grid, lon_grid)
    dvtec             = vtec_now - vtec_prev   # (N_lat, N_lon) delta
    dvtec_field, _    = ionex_to_polar_grid(dvtec, lat_grid, lon_grid)
    return np.stack([vtec_field, dvtec_field], axis=0)  # (2, NLAT, NMLT)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TECDataset(Dataset):
    """
    Args:
        timestamps:    sorted list of "YYYY-MM-DDTHH:MM" strings
        cache_dir:     local .npy cache (optional)
        stats:         {"mean": (2,), "std": (2,)} normalisation
        delta_t_steps: steps between xs and xs_next
    """

    def __init__(
        self,
        timestamps: list[str],
        cache_dir: Optional[Path] = None,
        stats: Optional[dict] = None,
        minio_endpoint: str = "localhost:9000",
        minio_access_key: str = "minioadmin",
        minio_secret_key: str = "minioadmin",
        delta_t_steps: int = 1,
    ):
        self.timestamps    = timestamps
        self.cache_dir     = Path(cache_dir) if cache_dir else None
        self.stats         = stats
        self.minio_cfg     = dict(
            bucket=BUCKET,
            endpoint=minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
        )
        self.delta_t_steps = delta_t_steps
        self._valid = list(range(len(timestamps) - delta_t_steps))

    def __len__(self):
        return len(self._valid)

    def __getitem__(self, idx):
        i = self._valid[idx]
        x      = self._load(self.timestamps[i])
        x_next = self._load(self.timestamps[i + self.delta_t_steps])
        return {"tec": x, "tec_next": x_next, "ts": self.timestamps[i]}

    # ------------------------------------------------------------------

    def _load(self, ts: str) -> torch.Tensor:
        fname = ts.replace(":", "").replace("-", "") + "_tec.npy"
        if self.cache_dir:
            p = self.cache_dir / fname
            if p.exists():
                grid = np.load(p, mmap_mode="r")
                return self._normalise(grid)

        raw  = get_minio_bytes(**self.minio_cfg, obj_name=fname)
        grid = np.load(io.BytesIO(raw))
        return self._normalise(grid)

    def _normalise(self, grid: np.ndarray) -> torch.Tensor:
        grid = grid.astype(np.float32)
        if self.stats is not None:
            mean = self.stats["mean"][:, None, None]
            std  = self.stats["std"][:, None, None]
            grid = (grid - mean) / std.clip(min=1e-6)
        return torch.from_numpy(grid)
