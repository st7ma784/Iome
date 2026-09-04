"""
Download DMSP SSJ4/5 particle precipitation data and project onto the 180×360
global equirectangular magnetic-coordinate grid.

DMSP satellites F12–F15 carry the SSJ4/5 electrostatic analysers measuring
electron and ion differential energy flux along the satellite track.  Each
polar pass takes ~15 min; at four satellites we get ~60 passes/day.

Data source: NASA SPDF CDAWeb
  Base: https://cdaweb.gsfc.nasa.gov/pub/data/dmsp/dmsp{sat}/ssj/precipitating-electrons-ions/{year}/
  File: dmsp-{sat}_ssj_precipitating-electrons-ions_{YYYYMMDD}_v*.cdf
  Available: F12–F15 from 2000 onwards

Variables extracted (AACGM magnetic coordinates already in file):
  SC_AACGM_LAT    – magnetic latitude  (°)
  SC_AACGM_LTIME  – magnetic local time (h)
  ELE_TOTAL_ENERGY_FLUX   – electron total energy flux (mW m⁻²)
  ELE_AVG_ENERGY          – electron characteristic energy (eV)
  ION_TOTAL_ENERGY_FLUX   – ion total energy flux (mW m⁻²)
  ION_AVG_ENERGY          – ion characteristic energy (eV)

Output: {cache_root}/dmsp/YYYYMMDDTHHMM_dmsp.npy  float16 (5, 180, 360)
  Channels: [e_flux, e_energy, i_flux, i_energy, soft_occ]
  Values are Gaussian-splatted onto the global grid; zero where no coverage.
  Satellite tracks with |mlat| < 40° are discarded (auroral zone only).

Each CDF file covers one satellite-day.  We bin observations into 2-min epochs
aligned to the same grid as SuperDARN/SuperMAG/TEC.  Multiple satellite passes
covering the same epoch are averaged (all four satellites are treated equally).

Resume-safe: skips epochs whose file already exists.

Usage:
    python scripts/ingest_dmsp.py \
        --start 2000-01-01 --end 2001-12-31 \
        --cache_root /data5/iome_cache \
        [--satellites f12 f13 f14 f15] \
        [--workers 4]
"""

import argparse
import importlib.util
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Grid helpers
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

EPOCH_MIN    = 2       # 2-minute cadence
MLAT_ABS_MIN = 40.0   # auroral zone threshold
N_CHANS      = 5

# SPDF CDAWeb — actual directory structure discovered from live server
CDAWEB_BASE = (
    "https://cdaweb.gsfc.nasa.gov/pub/data/dmsp/"
    "dmsp{sat}/ssj/precipitating-electrons-ions/{year:04d}/"
)
CDAWEB_FNAME = "dmsp-{sat}_ssj_precipitating-electrons-ions_{date}_v{ver}.cdf"


# ---------------------------------------------------------------------------
# CDF download
# ---------------------------------------------------------------------------

def _list_remote_cdfs(session: requests.Session, url: str) -> list[str]:
    """Parse Apache directory listing, return all .cdf filenames found."""
    try:
        resp = session.get(url, timeout=30)
        if resp.status_code != 200:
            return []
        names = []
        for line in resp.text.splitlines():
            if '.cdf"' in line:
                # href="dmsp-f13_ssj_..._v1.1.5.cdf"
                import re as _re
                m = _re.search(r'href="([^"]+\.cdf)"', line)
                if m:
                    names.append(m.group(1))
        return names
    except Exception:
        return []


def _download_cdf(session: requests.Session, sat: str, day: date) -> bytes | None:
    """Download the SSJ CDF for one satellite-day.  Returns raw bytes or None."""
    year     = day.year
    date_str = day.strftime("%Y%m%d")
    base     = CDAWEB_BASE.format(sat=sat, year=year)

    # List directory to find the versioned filename (version varies by file)
    fnames    = _list_remote_cdfs(session, base)
    prefix    = f"dmsp-{sat}_ssj_precipitating-electrons-ions_{date_str}_"
    candidates = [f for f in fnames if f.startswith(prefix)]

    if not candidates:
        # Fallback: try common versions without directory listing
        for ver in ("v1.1.2", "v1.1.3", "v1.1.4", "v1.1.5", "v1.2.0"):
            candidates.append(f"dmsp-{sat}_ssj_precipitating-electrons-ions_{date_str}_{ver}.cdf")

    for fname in candidates:
        url = base + fname
        for attempt in range(3):
            try:
                resp = session.get(url, timeout=120, stream=True)
                if resp.status_code == 200:
                    data = b"".join(resp.iter_content(chunk_size=1 << 16))
                    if len(data) > 1000:
                        return data
                elif resp.status_code == 404:
                    break
            except Exception:
                if attempt == 2:
                    break
                time.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------------------
# CDF parsing (pure Python via spacepy.pycdf or fallback to cdflib)
# ---------------------------------------------------------------------------

def _read_cdf_vars(raw_bytes: bytes, varnames: list[str]) -> dict[str, np.ndarray] | None:
    """
    Parse CDF bytes using cdflib (preferred) or spacepy.pycdf.
    Returns dict of {varname: ndarray} or None on failure.
    """
    import tempfile, os as _os
    with tempfile.NamedTemporaryFile(suffix=".cdf", delete=False) as f:
        f.write(raw_bytes)
        tmp = f.name
    try:
        return _read_with_cdflib(tmp, varnames)
    except Exception:
        pass
    try:
        return _read_with_spacepy(tmp, varnames)
    except Exception:
        return None
    finally:
        _os.unlink(tmp)


def _read_with_cdflib(path: str, varnames: list[str]) -> dict[str, np.ndarray]:
    import cdflib
    cdf = cdflib.CDF(path)
    out = {}
    for v in varnames:
        try:
            out[v] = np.asarray(cdf.varget(v), dtype=np.float32).squeeze()
        except Exception:
            pass
    return out


def _read_with_spacepy(path: str, varnames: list[str]) -> dict[str, np.ndarray]:
    from spacepy import pycdf
    with pycdf.CDF(path) as cdf:
        out = {}
        for v in varnames:
            try:
                out[v] = np.asarray(cdf[v][...], dtype=np.float32).squeeze()
            except Exception:
                pass
    return out


# ---------------------------------------------------------------------------
# Record → grid
# ---------------------------------------------------------------------------

_VARNAMES = [
    "SC_AACGM_LAT",
    "SC_AACGM_LTIME",        # magnetic local time in hours
    "ELE_TOTAL_ENERGY_FLUX",
    "ELE_AVG_ENERGY",
    "ION_TOTAL_ENERGY_FLUX",
    "ION_AVG_ENERGY",
    "Epoch",                  # CDF epoch (ms since 0 AD) for time binning
]

# Physical thresholds (units: eV/cm²/ster/s for flux, eV for energy)
# Fill values are NaN; isfinite() handles them. Only clip clearly unphysical outliers.
_FLUX_MAX    = 1e15   # eV/cm²/ster/s — hard upper clip (noise guard)
_ENERGY_MAX  = 1e5    # eV (~100 keV; SSJ channel ceiling ~30 keV)
_ENERGY_MIN  = 1.0    # eV


def _cdf_to_epoch_grids(raw_bytes: bytes) -> dict[str, np.ndarray]:
    """
    Parse one satellite-day CDF → {epoch_key: (5, NLAT, NMLT) float16}.
    """
    data = _read_cdf_vars(raw_bytes, _VARNAMES)
    if data is None or "SC_AACGM_LAT" not in data:
        return {}

    mlat    = data.get("SC_AACGM_LAT")
    mlt     = data.get("SC_AACGM_LTIME")
    e_flux  = data.get("ELE_TOTAL_ENERGY_FLUX")
    e_eng   = data.get("ELE_AVG_ENERGY")
    i_flux  = data.get("ION_TOTAL_ENERGY_FLUX")
    i_eng   = data.get("ION_AVG_ENERGY")
    epoch   = data.get("Epoch")

    if mlat is None or mlt is None:
        return {}

    n = mlat.shape[0]

    # Fallback zero arrays for missing channels
    def _safe(arr, n):
        if arr is not None and arr.shape[0] == n:
            return arr
        return np.zeros(n, dtype=np.float32)

    e_flux = _safe(e_flux, n)
    e_eng  = _safe(e_eng,  n)
    i_flux = _safe(i_flux, n)
    i_eng  = _safe(i_eng,  n)

    # Decode Epoch (CDF_EPOCH: ms since 01-Jan-0000 at 00:00)
    if epoch is not None and epoch.shape[0] == n:
        # CDF epoch: ms since 0 AD; convert to Unix seconds
        # CDF epoch for 1970-01-01T00:00:00 = 62167219200000.0 ms
        CDF_EPOCH_1970 = 62167219200000.0
        unix_s = (epoch.astype(np.float64) - CDF_EPOCH_1970) / 1000.0
    else:
        unix_s = np.zeros(n, dtype=np.float64)

    # Filter by |mlat| and valid values
    mask = (
        (np.abs(mlat) >= MLAT_ABS_MIN) &
        np.isfinite(mlat) & np.isfinite(mlt) &
        np.isfinite(e_flux) & np.isfinite(e_eng) &
        np.isfinite(i_flux) & np.isfinite(i_eng) &
        (e_flux >= 0) & (e_flux <= _FLUX_MAX) &
        (e_eng  >= _ENERGY_MIN) & (e_eng <= _ENERGY_MAX) &
        (i_flux >= 0) & (i_flux <= _FLUX_MAX) &
        (i_eng  >= _ENERGY_MIN) & (i_eng <= _ENERGY_MAX)
    )
    if not mask.any():
        return {}

    mlat_m   = mlat[mask]
    mlt_m    = mlt[mask]
    e_flux_m = e_flux[mask]
    e_eng_m  = e_eng[mask]
    i_flux_m = i_flux[mask]
    i_eng_m  = i_eng[mask]
    unix_m   = unix_s[mask]

    rows, cols = mlat_mlt_to_pixel(mlat_m, mlt_m)
    ok = (rows >= 0) & (rows < NLAT)

    rows_ok  = rows[ok]
    cols_ok  = cols[ok]
    unix_ok  = unix_m[ok]
    e_flux_ok = e_flux_m[ok]
    e_eng_ok  = e_eng_m[ok]
    i_flux_ok = i_flux_m[ok]
    i_eng_ok  = i_eng_m[ok]

    # Bin by 2-min epoch
    epoch_keys: dict[str, list] = {}
    for j in range(len(unix_ok)):
        dt = datetime.fromtimestamp(float(unix_ok[j]), tz=timezone.utc)
        epoch_min = (dt.minute // EPOCH_MIN) * EPOCH_MIN
        key = dt.strftime("%Y%m%d") + f"T{dt.hour:02d}{epoch_min:02d}"
        epoch_keys.setdefault(key, []).append(j)

    grids: dict[str, np.ndarray] = {}
    for key, idxs in epoch_keys.items():
        idx = np.array(idxs)
        r, c = rows_ok[idx], cols_ok[idx]
        ef = e_flux_ok[idx]
        ee = e_eng_ok[idx]
        if_arr = i_flux_ok[idx]
        ie = i_eng_ok[idx]

        g_ef, occ  = splat_to_grid(r, c, np.log1p(ef), sigma=1.5)
        g_ee, _    = splat_to_grid(r, c, np.log1p(ee), sigma=1.5)
        g_if, _    = splat_to_grid(r, c, np.log1p(if_arr), sigma=1.5)
        g_ie, _    = splat_to_grid(r, c, np.log1p(ie), sigma=1.5)

        grids[key] = np.stack([g_ef, g_ee, g_if, g_ie, occ], axis=0).astype(np.float16)

    return grids


# ---------------------------------------------------------------------------
# Per-satellite-day fetch and ingest
# ---------------------------------------------------------------------------

def _process_sat_day(
    session: requests.Session,
    sat: str,
    day: date,
) -> dict[str, np.ndarray]:
    raw = _download_cdf(session, sat, day)
    if raw is None:
        return {}
    try:
        return _cdf_to_epoch_grids(raw)
    except Exception as exc:
        print(f"    {sat} {day.isoformat()} parse error: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Merge grids from multiple satellites at the same epoch
# ---------------------------------------------------------------------------

def _merge_grids(
    acc: dict[str, list[np.ndarray]],
    new: dict[str, np.ndarray],
) -> None:
    """Accumulate per-satellite epoch grids into a list keyed by epoch."""
    for k, g in new.items():
        acc.setdefault(k, []).append(g.astype(np.float32))


def _average_grids(acc: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    out = {}
    for k, gs in acc.items():
        if len(gs) == 1:
            out[k] = gs[0].astype(np.float16)
        else:
            mean = np.mean(gs, axis=0)
            out[k] = mean.astype(np.float16)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start",      default="2000-01-01")
    ap.add_argument("--end",        default="2001-12-31")
    ap.add_argument("--cache_root", type=Path, default=Path("/data5/iome_cache"))
    ap.add_argument("--satellites", nargs="+", default=["f12", "f13", "f14", "f15"],
                    help="DMSP satellites to ingest (default: f12 f13 f14 f15)")
    ap.add_argument("--workers",    type=int, default=4,
                    help="Parallel satellite downloads per day")
    ap.add_argument("--rate_limit_s", type=float, default=0.5)
    args = ap.parse_args()

    out_dir = args.cache_root / "dmsp"
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "iome-ingest/0.1 (research)"

    start   = date.fromisoformat(args.start)
    end     = date.fromisoformat(args.end)
    current = start

    while current <= end:
        date_str = current.strftime("%Y%m%d")
        print(f"DMSP {current.isoformat()} [{' '.join(args.satellites)}]", end="  ", flush=True)

        # Resume: skip if all 2-min epochs for this day already exist.
        # Simple heuristic: check first epoch of day.
        first_key = date_str + "T0000"
        if (out_dir / f"{first_key}_dmsp.npy").exists():
            print("skip")
            current += timedelta(days=1)
            continue

        # Download all satellites in parallel
        acc: dict[str, list[np.ndarray]] = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {
                pool.submit(_process_sat_day, session, sat, current): sat
                for sat in args.satellites
            }
            for fut in as_completed(futs):
                sat = futs[fut]
                result = fut.result()
                if result:
                    _merge_grids(acc, result)
                    print(f"{sat}({len(result)})", end=" ", flush=True)
                else:
                    print(f"{sat}(-)", end=" ", flush=True)

        grids = _average_grids(acc)
        saved = 0
        for key, grid in grids.items():
            p = out_dir / f"{key}_dmsp.npy"
            if not p.exists():
                np.save(p, grid)
                saved += 1

        print(f"→ {len(grids)} epochs, {saved} new files")
        current += timedelta(days=1)
        time.sleep(args.rate_limit_s)


if __name__ == "__main__":
    main()
