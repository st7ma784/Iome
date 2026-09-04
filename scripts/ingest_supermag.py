"""
Download SuperMAG ground magnetometer data and project onto the 180×360 global grid.

Uses the official SuperMAG Python API (supermag-api.py in repo root).
Username: st7ma784  (or override with SUPERMAG_USER env var)

Strategy per day:
  1. SuperMAGGetInventory  → list of active stations
  2. Fetch all stations; filter by |mlat| ≥ 40° in the gridder
  3. SuperMAGGetData (threaded) → 1 HTTP call per station, full-day extent
  4. Bin 1-min records into 2-min epochs, Gaussian-splat → (3, 180, 360)
  5. Save float16 .npy per epoch to scc-hdd-01 (overflow to scc-hdd-02)

Output: {cache_root}/supermag/YYYYMMDDTHHMM_smag.npy  float16 (3, 180, 360)
Channels: [dBn_nT, dBe_nT, soft_occ]

Resume-safe: skips days whose first-epoch file already exists.

Usage:
    python scripts/ingest_supermag.py \
        --start 2015-01-01 --end 2015-12-31 \
        --cache_root /scc-hdd-01/iome_cache \
        [--cache_root2 /scc-hdd-02/iome_cache] \
        [--workers 8]
"""

import importlib.util
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap the official SuperMAG API from the repo root
# ---------------------------------------------------------------------------

_API_PATH = Path(__file__).resolve().parents[1] / "supermag-api.py"
if not _API_PATH.exists():
    sys.exit(f"ERROR: supermag-api.py not found at {_API_PATH}")

_spec = importlib.util.spec_from_file_location("supermag_api", _API_PATH)
_sm   = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sm)

SuperMAGGetInventory = _sm.SuperMAGGetInventory
SuperMAGGetData      = _sm.SuperMAGGetData

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

EPOCH_MIN    = 2       # target cadence (minutes)
MLAT_ABS_MIN = 40.0   # both hemispheres: |mlat| ≥ 40°
FILL_VALUE   = 999999.0


# ---------------------------------------------------------------------------
# Per-station fetch
# ---------------------------------------------------------------------------

def _fetch_station(logon: str, start: date, station: str) -> list | None:
    """
    Fetch one full day of data for a single station.
    Returns list-of-dicts FORMAT='list' or None on failure.
    """
    for attempt in range(3):
        try:
            status, data = SuperMAGGetData(
                logon,
                start.strftime("%Y-%m-%dT00:00"),
                86400,                           # full day in seconds
                "all,delta=start,baseline=yearly",
                station,
                FORMAT="list",
            )
            if status and isinstance(data, list) and len(data) > 0:
                return data
            return None
        except Exception as exc:
            if attempt == 2:
                return None
            time.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------------------
# Build epoch grids from a day's worth of station records
# ---------------------------------------------------------------------------

def _day_records_to_grids(
    all_station_data: list[list],   # list of per-station record lists
) -> dict[str, np.ndarray]:
    """
    Merge all station records into {epoch_key: (3, 180, 360) float16}.
    """
    # Accumulate per-epoch point clouds:  epoch_key → [(mlat, mlt, dbn, dbe), ...]
    epoch_points: dict[str, list] = {}

    for records in all_station_data:
        if not records:
            continue
        for rec in records:
            try:
                tval   = float(rec["tval"])
                mcolat = float(rec.get("mcolat", 0.0))
                mlat   = 90.0 - mcolat
                if abs(mlat) < MLAT_ABS_MIN:
                    continue

                mlt = float(rec.get("mlt", 0.0))
                # N['nez'] = northward perturbation in magnetic (NEZ) frame = dBn
                n_dict = rec.get("N", {})
                e_dict = rec.get("E", {})
                dbn = float(n_dict.get("nez", FILL_VALUE))
                dbe = float(e_dict.get("nez", FILL_VALUE))
                if abs(dbn) >= FILL_VALUE or abs(dbe) >= FILL_VALUE:
                    continue

                dt    = datetime.fromtimestamp(tval, tz=timezone.utc)
                epoch = dt.replace(
                    minute=(dt.minute // EPOCH_MIN) * EPOCH_MIN,
                    second=0, microsecond=0,
                )
                key = epoch.strftime("%Y%m%dT%H%M")
                epoch_points.setdefault(key, []).append((mlat, mlt, dbn, dbe))

            except (KeyError, TypeError, ValueError):
                continue

    # Grid each epoch
    grids: dict[str, np.ndarray] = {}
    for key, pts in epoch_points.items():
        arr  = np.array(pts, dtype=np.float32)      # (N, 4)
        mlat_a, mlt_a, dbn_a, dbe_a = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        rows, cols = mlat_mlt_to_pixel(mlat_a, mlt_a)

        ok = (rows >= 0) & (rows < NLAT)   # cols wrap on MLT axis
        if ok.sum() < 2:
            continue

        field_n, occ = splat_to_grid(rows[ok], cols[ok], dbn_a[ok])
        field_e, _   = splat_to_grid(rows[ok], cols[ok], dbe_a[ok])
        grids[key]   = np.stack([field_n, field_e, occ], axis=0).astype(np.float16)

    return grids


# ---------------------------------------------------------------------------
# Disk helpers
# ---------------------------------------------------------------------------

def _pick_root(roots: list[Path], min_free_gb: float = 10.0) -> Path:
    for root in roots:
        try:
            st = os.statvfs(root)
            if st.f_frsize * st.f_bavail / 1e9 > min_free_gb:
                return root
        except Exception:
            pass
    return roots[0]


def _already_cached(roots: list[Path], day: date) -> bool:
    first_key = day.strftime("%Y%m%dT0000")
    return any(
        (r / "supermag" / f"{first_key}_smag.npy").exists()
        for r in roots
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start",        default="2015-01-01")
    ap.add_argument("--end",          default="2015-12-31")
    ap.add_argument("--cache_root",   type=Path, default=Path("/scc-hdd-01/iome_cache"))
    ap.add_argument("--cache_root2",  type=Path, default=None)
    ap.add_argument("--user",         default=os.environ.get("SUPERMAG_USER", "st7ma784"))
    ap.add_argument("--workers",      type=int, default=8,
                    help="Parallel station fetches per day")
    ap.add_argument("--rate_limit_s", type=float, default=1.0,
                    help="Pause between days (seconds)")
    args = ap.parse_args()

    roots = [args.cache_root]
    if args.cache_root2:
        roots.append(args.cache_root2)
    for r in roots:
        (r / "supermag").mkdir(parents=True, exist_ok=True)

    start   = date.fromisoformat(args.start)
    end     = date.fromisoformat(args.end)
    current = start

    while current <= end:
        print(f"SuperMAG {current.isoformat()}", end="  ", flush=True)

        if _already_cached(roots, current):
            print("skip")
            current += timedelta(days=1)
            continue

        # Step 1: inventory
        try:
            status, stations = SuperMAGGetInventory(
                args.user,
                current.strftime("%Y-%m-%dT00:00"),
                86400,
            )
        except Exception as exc:
            print(f"inventory error ({exc}) — skipping")
            current += timedelta(days=1)
            time.sleep(args.rate_limit_s * 5)
            continue
        if not status or not stations:
            print("inventory failed")
            current += timedelta(days=1)
            time.sleep(args.rate_limit_s)
            continue

        # Step 2: filter to polar stations — we don't have mlat from inventory,
        # so fetch all and let the gridder filter by mcolat in the record.
        # Typical northern inventory is 200-400 stations; filter to a
        # reasonable polar subset to cut API calls.
        polar = [s for s in stations if isinstance(s, str) and s.strip()]
        print(f"{len(polar)} stations", end="  ", flush=True)

        # Step 3: threaded fetch
        all_data: list[list] = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {
                pool.submit(_fetch_station, args.user, current, st): st
                for st in polar
            }
            for fut in as_completed(futs):
                result = fut.result()
                if result:
                    all_data.append(result)

        print(f"→ {len(all_data)} ok", end="  ", flush=True)

        # Step 4: grid
        grids = _day_records_to_grids(all_data)
        print(f"{len(grids)} epochs", end="  ", flush=True)

        # Step 5: save
        out_dir = _pick_root(roots) / "supermag"
        saved = 0
        for key, grid in grids.items():
            p = out_dir / f"{key}_smag.npy"
            if not p.exists():
                np.save(p, grid)
                saved += 1
        print(f"→ {saved} written")

        current += timedelta(days=1)
        time.sleep(args.rate_limit_s)


if __name__ == "__main__":
    main()
