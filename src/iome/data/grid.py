"""
Shared grid utilities: Gaussian splat from sparse points onto a global
equirectangular magnetic-coordinate grid.

Coordinate system
-----------------
  Row axis  : magnetic latitude, +90° (NH pole, row 0) → −90° (SH pole, row NLAT−1)
  Column axis: magnetic local time, 0 h (col 0) → 24 h (col NMLT−1), wraps
                                                  (col NMLT−1 and col 0 are adjacent)

Grid size
---------
  NLAT × NMLT = 180 × 360  → 1° mlat per row, 4 min MLT per column.

Both hemispheres are represented natively: NH high-latitude region in rows
0–40 (90°→50° mlat), SH in rows 140–180 (−50°→−90° mlat). The equatorial
band (rows 40–140) will be sparse for all auroral sensors and is left as zero.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Grid parameters
# ---------------------------------------------------------------------------

NLAT = 180   # rows  — 1° per row,    90° → −90°
NMLT = 360   # cols  — 4 min per col, 0 h → 24 h  (wraps)

# Convenience alias kept for backward-compat imports that used `GRID`
GRID = NMLT  # not square any more; callers should prefer NLAT/NMLT


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def mlat_mlt_to_pixel(
    mlat: np.ndarray,
    mlt:  np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Magnetic latitude (°) and MLT (h) → fractional (row, col).

    Row  0      = 90° N mlat (NH pole)
    Row  NLAT-1 = 90° S mlat (SH pole)
    Col  0      = MLT 0 h (midnight)
    Col  NMLT-1 = MLT ≈ 24 h  (wraps: col NMLT-1 and col 0 are adjacent)

    Points outside [−90, 90] mlat are returned but will be clipped / masked
    by the caller.  MLT values outside [0, 24) are taken modulo 24.
    """
    mlat = np.asarray(mlat, dtype=np.float32)
    mlt  = np.asarray(mlt,  dtype=np.float32)
    row = (90.0 - mlat) / 180.0 * NLAT             # 90° → 0,  −90° → NLAT
    col = (mlt % 24.0)  / 24.0  * NMLT             # 0 h → 0,  24 h → NMLT
    return row, col


def geo_latlon_to_pixel(
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Geographic lat/lon → pixel via a simple dipole-tilt-free approximation.
    Accurate to ±10° at high latitudes; replace with aacgmv2 in production.
    """
    mlat_approx = np.asarray(lat,  dtype=np.float32)
    mlt_approx  = (np.asarray(lon, dtype=np.float32) / 15.0) % 24.0
    return mlat_mlt_to_pixel(mlat_approx, mlt_approx)


# ---------------------------------------------------------------------------
# Gaussian splat  (handles MLT column wrap-around)
# ---------------------------------------------------------------------------

def gaussian_splat(
    rows:       np.ndarray,
    cols:       np.ndarray,
    values:     np.ndarray,
    grid:       np.ndarray,        # (NLAT, NMLT) accumulator, modified in-place
    weight_acc: np.ndarray,        # (NLAT, NMLT) weight accumulator, modified in-place
    sigma:      float = 1.5,
) -> None:
    """
    Accumulate point observations onto a grid in-place with Gaussian weighting.

    The column (MLT) axis wraps: a blob near col 0 spills into the high-col
    edge and vice-versa.
    """
    nlat, nmlt = grid.shape
    radius = int(np.ceil(3 * sigma))
    dy, dx = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(np.float32)
    kernel = np.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2))

    for r, c, v in zip(rows, cols, values):
        ri, ci = int(round(float(r))), int(round(float(c)))

        # Row bounds (no wrapping in latitude)
        r0 = ri - radius; r1 = ri + radius + 1
        kr0 = max(0, -r0); r0 = max(0, r0); r1 = min(nlat, r1)
        kr1 = kr0 + (r1 - r0)
        if r0 >= r1:
            continue

        # Column indices with MLT wraparound
        c_idxs = np.arange(ci - radius, ci + radius + 1) % nmlt   # (2*radius+1,)
        kc_all = np.arange(2 * radius + 1)

        k_row = kernel[kr0:kr1, :]   # (row_slice, 2*radius+1)
        v_float = float(v)

        for kc, c_idx in zip(kc_all, c_idxs):
            grid[r0:r1, c_idx]      += k_row[:, kc] * v_float
            weight_acc[r0:r1, c_idx] += k_row[:, kc]


def splat_to_grid(
    rows:       np.ndarray,
    cols:       np.ndarray,
    values:     np.ndarray,
    sigma:      float = 1.5,
    min_weight: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project sparse point measurements onto the (NLAT, NMLT) global grid.

    Returns:
        field: (NLAT, NMLT) weighted-average field, 0 where no coverage
        occ:   (NLAT, NMLT) soft occupancy ∈ [0, 1]
    """
    grid_acc   = np.zeros((NLAT, NMLT), dtype=np.float32)
    weight_acc = np.zeros((NLAT, NMLT), dtype=np.float32)
    gaussian_splat(rows, cols, values, grid_acc, weight_acc, sigma=sigma)
    occ   = np.tanh(weight_acc)
    field = np.where(weight_acc > min_weight,
                     grid_acc / weight_acc.clip(min=min_weight), 0.0)
    return field.astype(np.float32), occ.astype(np.float32)
