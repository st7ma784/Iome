"""
Download IGS IONEX files from NASA CDDIS and project onto the 180×360 global grid.

IONEX files contain global 2.5°×5° VTEC maps every 1–2 hours.
We interpolate to 2-min cadence and project the full globe (all latitudes)
onto the standard equirectangular magnetic-coordinate grid.

Auth: NASA Earthdata (.netrc)
    machine urs.earthdata.nasa.gov
      login    YOUR_EARTHDATA_USERNAME
      password YOUR_EARTHDATA_PASSWORD
    Register free at: https://urs.earthdata.nasa.gov

Alternatively, set EARTHDATA_USER and EARTHDATA_PASSWORD env vars.

Output: {cache_root}/tec/YYYYMMDDTHHMM_tec.npy   float16, shape (2, 180, 360)
    channels: [VTEC (TECU), dVTEC/dt (TECU/2min)]

Falls back to CODE AIUB (ftp.aiub.unibe.ch) if CDDIS is unavailable.

Usage:
    python scripts/ingest_tec.py \
        --start 2015-01-01 --end 2015-12-31 \
        --cache_root /scc-hdd-01/iome_cache
"""

import argparse
import importlib.util
import gzip  # still used in _decompress for .gz detection
import io
import netrc
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests
from requests.auth import HTTPBasicAuth

_grid_spec = importlib.util.spec_from_file_location(
    "iome_grid",
    Path(__file__).resolve().parents[1] / "src" / "iome" / "data" / "grid.py",
)
_grid = importlib.util.module_from_spec(_grid_spec)
_grid_spec.loader.exec_module(_grid)
geo_latlon_to_pixel = _grid.geo_latlon_to_pixel
splat_to_grid       = _grid.splat_to_grid
NLAT                = _grid.NLAT
NMLT                = _grid.NMLT

EPOCH_MIN  = 2        # 2-minute target cadence
TECU_NAN   = 9999.0  # IONEX fill value

# IONEX product sources
# Pre-2023: old short-filename convention (.Z for <2000, .gz for newer)
# 2023+: new IGS RINEX3 long-filename convention (.gz)
CDDIS_BASE  = "https://cddis.nasa.gov/archive/gnss/products/ionex/{year}/{ddd:03d}/igsg{ddd:03d}0.{yy:02d}i"
CDDIS_NEW   = "https://cddis.nasa.gov/archive/gnss/products/ionex/{year}/{ddd:03d}/IGS0OPSFIN_{year}{ddd:03d}0000_01D_02H_GIM.INX.gz"
CODE_BASE   = "https://ftp.aiub.unibe.ch/CODE/{year}/CODG{ddd:03d}0.{yy:02d}I"


# ---------------------------------------------------------------------------
# IONEX parsing (inlined to avoid importing the torch-dependent Dataset module)
# ---------------------------------------------------------------------------

def parse_ionex(raw: bytes) -> dict:
    import re as _re
    text   = raw.decode("ascii", errors="replace")
    lines  = text.splitlines()
    lat_grid = lon_grid = None
    exp = -1
    for line in lines:
        if "LAT1 / LAT2 / DLAT" in line:
            v = line[:60].split()
            lat_grid = np.arange(float(v[0]), float(v[1]) + float(v[2]) / 2, float(v[2]))
        if "LON1 / LON2 / DLON" in line:
            v = line[:60].split()
            lon_grid = np.arange(float(v[0]), float(v[1]) + float(v[2]) / 2, float(v[2]))
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
            epoch_str, tec_rows = None, []
            i += 1
            while i < len(lines) and "END OF TEC MAP" not in lines[i]:
                l = lines[i]
                if "EPOCH OF CURRENT MAP" in l:
                    p = l[:60].split()
                    epoch_str = f"{int(p[0]):04d}{int(p[1]):02d}{int(p[2]):02d}T{int(p[3]):02d}{int(p[4]):02d}{int(p[5]):02d}"
                elif "LAT/LON" in l or "START OF RMS" in l:
                    pass  # per-latitude header row — skip
                elif l.strip():
                    try:
                        vals = [float(v) * exponent for v in l.split()]
                        tec_rows.extend(vals)
                    except ValueError:
                        pass  # ignore any other non-numeric line
                i += 1
            if epoch_str and tec_rows:
                n_lat, n_lon = len(lat_grid), len(lon_grid)
                expected = n_lat * n_lon
                if len(tec_rows) < expected:
                    # Pad with NaN if map is truncated (rare in malformed files)
                    tec_rows.extend([float('nan')] * (expected - len(tec_rows)))
                flat = np.array(tec_rows[:expected], dtype=np.float32).reshape(n_lat, n_lon)
                flat[flat >= TECU_NAN * exponent * 0.9] = np.nan
                epochs.append(epoch_str)
                maps.append(flat)
        i += 1
    return {
        "epoch": epochs,
        "lat": lat_grid.astype(np.float32),
        "lon": lon_grid.astype(np.float32),
        "vtec": np.stack(maps, axis=0) if maps else np.empty((0, len(lat_grid), len(lon_grid))),
    }


def ionex_to_polar_grid(vtec_map, lat_grid, lon_grid):
    LON, LAT = np.meshgrid(lon_grid, lat_grid)
    mask = np.isfinite(vtec_map)   # full globe — both hemispheres
    rows, cols = geo_latlon_to_pixel(LAT[mask], LON[mask])
    ok = (rows >= 0) & (rows < NLAT)
    return splat_to_grid(rows[ok], cols[ok], vtec_map[mask][ok], sigma=2.0)


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _earthdata_auth() -> HTTPBasicAuth | None:
    """Read Earthdata credentials from env vars or ~/.netrc."""
    user = os.environ.get("EARTHDATA_USER")
    pw   = os.environ.get("EARTHDATA_PASSWORD")
    if user and pw:
        return HTTPBasicAuth(user, pw)
    try:
        nrc  = netrc.netrc()
        host = nrc.authenticators("urs.earthdata.nasa.gov")
        if host:
            return HTTPBasicAuth(host[0], host[2])
    except (FileNotFoundError, netrc.NetrcParseError):
        pass
    return None


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _decompress(data: bytes) -> bytes:
    """Decompress .gz or .Z (Unix compress) bytes."""
    import subprocess
    if data[:2] == b'\x1f\x8b':
        return gzip.decompress(data)
    # Unix compress (.Z) — LZW, handled by system uncompress
    result = subprocess.run(["uncompress", "-c"], input=data, capture_output=True)
    if result.returncode == 0:
        return result.stdout
    raise ValueError(f"uncompress failed: {result.stderr.decode()[:200]}")


def _download_ionex(session: requests.Session, day: date,
                    auth: HTTPBasicAuth | None) -> bytes | None:
    """
    Try CDDIS (igsg, .Z then .gz) then CODE AIUB (codg, .gz then .Z).
    Returns raw decompressed IONEX bytes or None.
    """
    ddd  = day.timetuple().tm_yday
    yy   = day.year % 100
    year = day.year

    candidates = [
        CDDIS_BASE.format(year=year, ddd=ddd, yy=yy) + ".Z",
        CDDIS_BASE.format(year=year, ddd=ddd, yy=yy) + ".gz",
        CDDIS_NEW.format(year=year, ddd=ddd),
        CODE_BASE.format(year=year, ddd=ddd, yy=yy)  + ".gz",
        CODE_BASE.format(year=year, ddd=ddd, yy=yy)  + ".Z",
    ]

    for url in candidates:
        for attempt in range(3):
            try:
                resp = session.get(url, auth=auth, allow_redirects=True, timeout=120)
                if resp.status_code == 200 and len(resp.content) > 1000:
                    raw = _decompress(resp.content)
                    print(f"    {url.split('/')[-1]}  {len(resp.content)//1024}KB → {len(raw)//1024}KB")
                    return raw
                elif resp.status_code in (401, 403):
                    print(f"    auth failure ({url.split('/')[-1]})")
                    break
            except Exception as exc:
                if attempt == 2:
                    print(f"    failed {url.split('/')[-1]}: {exc}")
                time.sleep(5)

    return None


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def _interpolate_vtec_maps(
    vtec: np.ndarray,      # (N_maps, N_lat, N_lon)
    map_times_h: list[float],   # hours since midnight for each map
    target_min: int,            # target 2-min epoch (minutes since midnight)
) -> np.ndarray:
    """Linearly interpolate VTEC between adjacent maps to a target time."""
    t = target_min / 60.0   # hours since midnight
    if t <= map_times_h[0]:
        return vtec[0]
    if t >= map_times_h[-1]:
        return vtec[-1]
    for i in range(len(map_times_h) - 1):
        if map_times_h[i] <= t <= map_times_h[i + 1]:
            alpha = (t - map_times_h[i]) / (map_times_h[i + 1] - map_times_h[i])
            return (1 - alpha) * vtec[i] + alpha * vtec[i + 1]
    return vtec[-1]


# ---------------------------------------------------------------------------
# Per-day processing
# ---------------------------------------------------------------------------

def _process_day(raw: bytes, day: date) -> dict[str, np.ndarray]:
    """
    Parse already-decompressed IONEX bytes, interpolate to 2-min, grid → epoch dict.

    Returns {epoch_key: (2, 180, 360) float16}
    """
    parsed = parse_ionex(raw)
    if not parsed["epoch"] or parsed["vtec"].shape[0] < 2:
        return {}

    vtec  = parsed["vtec"]            # (N_maps, N_lat, N_lon)
    lats  = parsed["lat"]
    lons  = parsed["lon"]
    epochs = parsed["epoch"]           # list of "YYYYMMDDTHHmmss" strings

    # Extract hours since midnight for each IONEX map
    map_hours = []
    for ep in epochs:
        dt = datetime.strptime(ep, "%Y%m%dT%H%M%S")
        map_hours.append(dt.hour + dt.minute / 60.0 + dt.second / 3600.0)

    grids: dict[str, np.ndarray] = {}

    # Iterate every 2-min epoch in the day
    for minute in range(0, 1440, EPOCH_MIN):
        key = day.strftime("%Y%m%d") + "T" + f"{minute // 60:02d}{minute % 60:02d}"

        vtec_now = _interpolate_vtec_maps(vtec, map_hours, minute)
        vtec_prv = _interpolate_vtec_maps(vtec, map_hours, max(0, minute - EPOCH_MIN))

        field_now,  occ = ionex_to_polar_grid(vtec_now, lats, lons)
        field_prv,  _   = ionex_to_polar_grid(vtec_prv, lats, lons)
        delta_field     = field_now - field_prv

        grid = np.stack([field_now, delta_field], axis=0).astype(np.float16)
        grids[key] = grid

    return grids


# ---------------------------------------------------------------------------
# Disk
# ---------------------------------------------------------------------------

def _find_cache_dir(roots: list[Path], min_free_gb: float = 10.0) -> Path:
    for root in roots:
        try:
            stat = os.statvfs(root)
            free_gb = stat.f_frsize * stat.f_bavail / 1e9
            if free_gb > min_free_gb:
                return root
        except Exception:
            continue
    return roots[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start",       default="2015-01-01")
    ap.add_argument("--end",         default="2015-12-31")
    ap.add_argument("--cache_root",  type=Path, default=Path("/scc-hdd-01/iome_cache"))
    ap.add_argument("--cache_root2", type=Path, default=None)
    ap.add_argument("--rate_limit_s", type=float, default=1.0)
    args = ap.parse_args()

    roots = [args.cache_root]
    if args.cache_root2:
        roots.append(args.cache_root2)
    for r in roots:
        (r / "tec").mkdir(parents=True, exist_ok=True)

    auth    = _earthdata_auth()
    session = requests.Session()
    session.headers["User-Agent"] = "iome-ingest/0.1 (research)"

    if auth is None:
        print("WARNING: no Earthdata credentials found — CDDIS downloads will likely fail.")
        print("  Either add ~/.netrc entry for urs.earthdata.nasa.gov")
        print("  or set EARTHDATA_USER / EARTHDATA_PASSWORD env vars.")
        print("  Register free at https://urs.earthdata.nasa.gov")

    start   = date.fromisoformat(args.start)
    end     = date.fromisoformat(args.end)
    current = start

    while current <= end:
        print(f"TEC {current.isoformat()}", end=" ... ", flush=True)

        # Resume check: see if first epoch of the day already exists
        first_key = current.strftime("%Y%m%dT0000")
        cache_dirs = [r / "tec" for r in roots]
        if any((d / f"{first_key}_tec.npy").exists() for d in cache_dirs):
            print("skip")
            current += timedelta(days=1)
            continue

        raw = _download_ionex(session, current, auth)
        if raw is None:
            print("FAILED (no IONEX source available)")
            current += timedelta(days=1)
            time.sleep(args.rate_limit_s)
            continue

        grids = _process_day(raw, current)
        print(f"{len(grids)} epochs", end=" ")

        cache_dir = _find_cache_dir(roots) / "tec"
        saved = 0
        for key, grid in grids.items():
            p = cache_dir / f"{key}_tec.npy"
            if not p.exists():
                np.save(p, grid)
                saved += 1
        print(f"→ {saved} new files")

        current += timedelta(days=1)
        time.sleep(args.rate_limit_s)


if __name__ == "__main__":
    main()
