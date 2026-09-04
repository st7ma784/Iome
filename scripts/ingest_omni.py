"""
Download OMNI-2 hourly solar-wind data from NASA SPDF and convert to a
per-timestamp lookup table for use as u (B, 8) in the dynamics model.

Output: {cache_root}/omni/omni_{YYYY}.npy  — numpy .npy of a Python dict
    keys:   "YYYYDDDTHH" (year + day-of-year + hour, zero-padded)
    values: float32 array [Bx_GSM, By_GSM, Bz_GSM, |B|, Vx, n_p, Kp, eps]

eps (Newell coupling ε) is computed from IMF and solar-wind speed:
    ε = V_sw · |Bt|^2 · sin^4(θ/2)
    where Bt = sqrt(By^2 + Bz^2), θ = atan2(|By|, Bz) (clock angle)

OMNI fill values (999.9, 9999., 99999. etc.) are replaced with NaN.
Timestamps with any NaN feature → omni_mask=0 in training.

Usage:
    python scripts/ingest_omni.py --start 2015 --end 2020 --cache_root /scc-hdd-01/iome_cache
"""

import argparse
import struct
from pathlib import Path

import numpy as np
import requests

SPDF_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat"

# OMNI-2 column indices (0-based, space-separated fixed-width)
# Full format: https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2.text
COL_YEAR  = 0
COL_DOY   = 1
COL_HOUR  = 2
COL_B_MAG = 8    # |B| scalar (nT)
COL_BX    = 12   # Bx GSM (nT)
COL_BY    = 15   # By GSM (nT)
COL_BZ    = 16   # Bz GSM (nT)
COL_VX    = 23   # Plasma speed (km/s)
COL_NP    = 22   # Proton density (n/cc)
COL_KP    = 37   # Kp index (tenths of Kp)

# OMNI fill / missing-data sentinels (values ≥ threshold → NaN)
FILL_THRESH = {
    COL_B_MAG: 999.9,
    COL_BX:    999.9,
    COL_BY:    999.9,
    COL_BZ:    999.9,
    COL_VX:    9999.,
    COL_NP:    999.9,
    COL_KP:    99.,
}


def _newell_eps(by, bz, vx, b_mag):
    """Simplified Newell coupling: V · |Bt|^2 · sin^4(theta/2)."""
    bt    = np.sqrt(by ** 2 + bz ** 2)
    theta = np.arctan2(np.abs(by), bz)        # clock angle
    eps   = np.abs(vx) * bt ** 2 * np.sin(theta / 2) ** 4
    return eps.astype(np.float32)


def _parse_year(text: str) -> dict:
    records = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 55:
            continue
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            continue

        year = int(vals[COL_YEAR])
        doy  = int(vals[COL_DOY])
        hour = int(vals[COL_HOUR])
        key  = f"{year:04d}{doy:03d}T{hour:02d}"

        feats = np.array([
            vals[COL_BX],
            vals[COL_BY],
            vals[COL_BZ],
            vals[COL_B_MAG],
            vals[COL_VX],
            vals[COL_NP],
            vals[COL_KP] / 10.0,        # Kp in proper units
            0.0,                         # placeholder for ε
        ], dtype=np.float32)

        # Fill detection
        for col_idx, (src_col, thresh) in enumerate(zip(
            [COL_BX, COL_BY, COL_BZ, COL_B_MAG, COL_VX, COL_NP, COL_KP],
            list(FILL_THRESH.values()),
        )):
            if abs(vals[src_col]) >= thresh:
                feats[col_idx] = np.nan

        # Newell coupling
        bx, by, bz, bmag, vx, n = feats[0], feats[1], feats[2], feats[3], feats[4], feats[5]
        if not any(np.isnan([by, bz, vx])):
            feats[7] = _newell_eps(by, bz, vx, bmag)
        else:
            feats[7] = np.nan

        records[key] = feats

    return records


def ingest_year(year: int, out_dir: Path, session: requests.Session):
    out_path = out_dir / f"omni_{year}.npy"
    if out_path.exists():
        print(f"  skip {year} (already cached)")
        return

    url  = SPDF_URL.format(year=year)
    print(f"  fetch {url}")
    resp = session.get(url, timeout=120)
    resp.raise_for_status()

    records = _parse_year(resp.text)
    print(f"  parsed {len(records)} hourly records for {year}")
    np.save(out_path, records)
    print(f"  saved  {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start",      type=int, default=2014)
    ap.add_argument("--end",        type=int, default=2023, help="inclusive")
    ap.add_argument("--cache_root", type=Path, default=Path("/scc-hdd-01/iome_cache"))
    args = ap.parse_args()

    out_dir = args.cache_root / "omni"
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "iome-ingest/0.1 (research)"

    for year in range(args.start, args.end + 1):
        print(f"OMNI {year}")
        try:
            ingest_year(year, out_dir, session)
        except Exception as exc:
            print(f"  ERROR: {exc}")


if __name__ == "__main__":
    main()
