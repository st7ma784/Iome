"""
Convert SuperDARN cnvmap files to per-day .npy grids on the global 180×360 grid.

cnvmap files contain fitted convection maps (spherical harmonic potential fit to
LOS velocity measurements from multiple radars).

Output: {cache_root}/superdarn/YYYYMMDD_sd.npy   float16, shape (6, 180, 360)
    channels: [vlos_n, vlos_e, model_vlos_n, model_vlos_e, obs_occ, soft_occ]

    vlos_n/e:       northward/eastward pseudo-components of observed LOS velocity
                    (projected via kvect: v_N ≈ v·cos(kvect), v_E ≈ v·sin(kvect))
    model_vlos_n/e: same decomposition for the model (fitted) velocity grid
    obs_occ:        number of raw radar vectors per grid cell (hard occupancy)
    soft_occ:       Gaussian-splat occupancy from model grid

cnvmap files are one file per day; each file may contain multiple records
(one per ≈2-min scan-cycle).

Usage:
    python scripts/ingest_superdarn.py \
        --cnvmap_dir /data5/TrainingConvMaps/cnvmaps \
        --cache_root /data5/iome_cache
"""

import argparse
import importlib.util
import math
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Load grid helpers (avoids importing the torch-dependent package __init__)
# ---------------------------------------------------------------------------

_grid_spec = importlib.util.spec_from_file_location(
    "iome_grid",
    Path(__file__).resolve().parents[1] / "src" / "iome" / "data" / "grid.py",
)
_grid = importlib.util.module_from_spec(_grid_spec)
_grid_spec.loader.exec_module(_grid)
mlat_mlt_to_pixel = _grid.mlat_mlt_to_pixel
splat_to_grid     = _grid.splat_to_grid
NLAT              = _grid.NLAT
NMLT              = _grid.NMLT

# ---------------------------------------------------------------------------
# DMap binary parser (pure Python, no pydarn dependency)
# ---------------------------------------------------------------------------

# Type code → (struct format char, byte size)
_DMAP_FMT = {1: ('b', 1), 2: ('h', 2), 3: ('i', 4), 4: ('f', 4), 8: ('d', 8)}


def _read_cstring(buf: bytes, off: int):
    end = buf.index(b'\x00', off)
    return buf[off:end].decode('ascii', errors='replace'), end + 1


def _parse_dmap(data: bytes) -> list[dict]:
    """Parse a DMap binary file into a list of record dicts."""
    records = []
    off = 0
    while off + 8 <= len(data):
        code, size = struct.unpack_from('<ii', data, off)
        if size < 16 or off + size > len(data):
            break
        rec_buf = data[off: off + size]
        off += size

        nscalar, nvector = struct.unpack_from('<ii', rec_buf, 8)
        roff = 16
        scalars = {}

        for _ in range(nscalar):
            name, roff = _read_cstring(rec_buf, roff)
            dtype = rec_buf[roff]; roff += 1   # type is 1 byte in this RST version
            if dtype == 9:
                val, roff = _read_cstring(rec_buf, roff)
            elif dtype in _DMAP_FMT:
                fmt, sz = _DMAP_FMT[dtype]
                val = struct.unpack_from('<' + fmt, rec_buf, roff)[0]
                roff += sz
            else:
                val = None
            scalars[name] = val

        vectors = {}
        for _ in range(nvector):
            name, roff = _read_cstring(rec_buf, roff)
            dtype = rec_buf[roff]; roff += 1
            ndim  = struct.unpack_from('<i', rec_buf, roff)[0]; roff += 4
            dims  = list(struct.unpack_from(f'<{ndim}i', rec_buf, roff)); roff += 4 * ndim
            n = 1
            for d in dims:
                n *= d
            if dtype == 9:
                vals = []
                for _ in range(n):
                    s, roff = _read_cstring(rec_buf, roff)
                    vals.append(s)
            elif dtype in _DMAP_FMT:
                fmt, sz = _DMAP_FMT[dtype]
                vals = list(struct.unpack_from(f'<{n}{fmt}', rec_buf, roff))
                roff += sz * n
            else:
                vals = []
            vectors[name] = (dims, vals)

        records.append({'scalars': scalars, 'vectors': vectors})
    return records


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def _mlon_to_mlt(mlon: np.ndarray, ut_hour: float) -> np.ndarray:
    """
    Approximate conversion from magnetic longitude to MLT.

    MLT = (MLON / 15 + UT_hours) % 24

    This is exact when the geomagnetic pole aligns with geographic pole (it
    does not, but the error is small compared to 120-pixel grid resolution).
    """
    return (mlon / 15.0 + ut_hour) % 24.0


def _decompose_los(vel: np.ndarray, kvect_deg: np.ndarray):
    """
    Project LOS velocity into magnetic N and E pseudo-components.

    SuperDARN kvect is measured clockwise from magnetic North.
    v_N ≈ v_los * cos(kvect)
    v_E ≈ v_los * sin(kvect)

    This is only correct when the true flow is exactly along kvect;
    for the ML model, the projected components carry the spatial pattern.
    """
    theta = np.deg2rad(kvect_deg)
    return vel * np.cos(theta), vel * np.sin(theta)


# ---------------------------------------------------------------------------
# Record → grid
# ---------------------------------------------------------------------------

def _record_to_grid(rec: dict) -> tuple[str, np.ndarray] | None:
    """
    Convert one cnvmap DMap record to date key + (6, NLAT, NMLT) float16 array.
    Returns None if the record has no usable vectors.

    Key format: "YYYYMMDD" (daily maps — one per file).

    Both NH (mlat > 0) and SH (mlat < 0) vectors are placed natively on the
    global equirectangular grid.  No hemisphere mirroring is applied.
    """
    s = rec['scalars']
    v = rec['vectors']

    try:
        yr  = int(s['start.year'])
        mo  = int(s['start.month'])
        dy  = int(s['start.day'])
        hr  = int(s['start.hour'])
        mn  = int(s['start.minute'])
    except (KeyError, TypeError, ValueError):
        return None

    ut_hour = hr + mn / 60.0
    key = f"{yr:04d}{mo:02d}{dy:02d}"

    # ---- observed (sparse) vectors ----
    obs_grid_n = np.zeros((NLAT, NMLT), dtype=np.float32)
    obs_grid_e = np.zeros((NLAT, NMLT), dtype=np.float32)
    obs_occ    = np.zeros((NLAT, NMLT), dtype=np.float32)

    if 'vector.mlat' in v and 'vector.mlon' in v:
        mlat = np.array(v['vector.mlat'][1], dtype=np.float32)   # signed: NH +, SH −
        mlon = np.array(v['vector.mlon'][1], dtype=np.float32)
        kv   = np.array(v['vector.kvect'][1], dtype=np.float32)
        vel  = np.array(v['vector.vel.median'][1], dtype=np.float32)

        mask = (np.abs(mlat) >= 40.0) & np.isfinite(vel)
        if mask.any():
            mlt = _mlon_to_mlt(mlon[mask], ut_hour)
            rows, cols = mlat_mlt_to_pixel(mlat[mask], mlt)
            ok = (rows >= 0) & (rows < NLAT)   # cols wrap, no col clip needed
            if ok.any():
                vn, ve = _decompose_los(vel[mask][ok], kv[mask][ok])
                obs_grid_n, obs_occ = splat_to_grid(rows[ok], cols[ok], vn, sigma=1.5)
                obs_grid_e, _       = splat_to_grid(rows[ok], cols[ok], ve, sigma=1.5)

    # ---- model (fitted, dense) vectors ----
    mod_grid_n = np.zeros((NLAT, NMLT), dtype=np.float32)
    mod_grid_e = np.zeros((NLAT, NMLT), dtype=np.float32)
    mod_soft   = None

    if 'model.mlat' in v and 'model.mlon' in v:
        mmlat = np.array(v['model.mlat'][1], dtype=np.float32)   # signed
        mmlon = np.array(v['model.mlon'][1], dtype=np.float32)
        mkv   = np.array(v['model.kvect'][1], dtype=np.float32)
        mvel  = np.array(v['model.vel.median'][1], dtype=np.float32)

        mask = (np.abs(mmlat) >= 40.0) & np.isfinite(mvel)
        if mask.any():
            mmlt = _mlon_to_mlt(mmlon[mask], ut_hour)
            rows, cols = mlat_mlt_to_pixel(mmlat[mask], mmlt)
            ok = (rows >= 0) & (rows < NLAT)
            if ok.any():
                vn, ve = _decompose_los(mvel[mask][ok], mkv[mask][ok])
                mod_grid_n, _        = splat_to_grid(rows[ok], cols[ok], vn, sigma=1.5)
                mod_grid_e, mod_soft = splat_to_grid(rows[ok], cols[ok], ve, sigma=1.5)

    if mod_soft is None:
        mod_soft = np.zeros((NLAT, NMLT), dtype=np.float32)

    grid = np.stack([
        obs_grid_n,   # ch 0: observed northward E×B (m/s)
        obs_grid_e,   # ch 1: observed eastward  E×B (m/s)
        mod_grid_n,   # ch 2: Weimer model northward
        mod_grid_e,   # ch 3: Weimer model eastward
        obs_occ,      # ch 4: radar coverage occupancy
        mod_soft,     # ch 5: model soft occupancy
    ], axis=0).astype(np.float16)

    return key, grid


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def _process_cnvmap(path: Path) -> dict[str, np.ndarray]:
    """Parse one cnvmap file → {epoch_key: (6,120,120) float16}."""
    with open(path, 'rb') as f:
        data = f.read()

    records = _parse_dmap(data)
    grids: dict[str, np.ndarray] = {}
    for rec in records:
        result = _record_to_grid(rec)
        if result is None:
            continue
        key, grid = result
        if key not in grids:
            grids[key] = grid
        else:
            # Multiple records mapping to the same 2-min bin: average
            grids[key] = ((grids[key].astype(np.float32) + grid.astype(np.float32)) / 2).astype(np.float16)
    return grids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnvmap_dir", type=Path,
                    default=Path("/data5/TrainingConvMaps/cnvmaps"))
    ap.add_argument("--cache_root", type=Path,
                    default=Path("/data5/iome_cache"))
    args = ap.parse_args()

    out_dir = args.cache_root / "superdarn"
    out_dir.mkdir(parents=True, exist_ok=True)

    cnvmap_files = sorted(args.cnvmap_dir.glob("*.cnvmap"))
    print(f"Found {len(cnvmap_files)} cnvmap files")

    total_written = 0
    for fpath in cnvmap_files:
        # Resume check: see if daily map already saved
        date_str = fpath.stem   # e.g. "19990101"
        if (out_dir / f"{date_str}_sd.npy").exists():
            print(f"  {fpath.name}  skip")
            continue

        try:
            grids = _process_cnvmap(fpath)
        except Exception as exc:
            print(f"  {fpath.name}  ERROR: {exc}")
            continue

        saved = 0
        for key, grid in grids.items():
            p = out_dir / f"{key}_sd.npy"
            if not p.exists():
                np.save(p, grid)
                saved += 1
        total_written += saved
        print(f"  {fpath.name}  {len(grids)} maps  → {saved} new files")

    print(f"\nDone. Total files written: {total_written}")


if __name__ == "__main__":
    main()
