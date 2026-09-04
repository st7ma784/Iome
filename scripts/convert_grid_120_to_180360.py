"""
Convert legacy 120×120 northern-polar .npy files to the 180×360 global
equirectangular grid used by the current model.

Old grid: (C, 120, 120)
  - Row 0   = 90° N MLAT (pole),  row 119 ≈ 50° N MLAT
  - Col 0   = MLT 0 h,            col 119 ≈ MLT 23.8 h
  - Coverage: northern polar cap only (MLAT 50–90°, northern hemisphere)

New grid: (C, 180, 360)
  - Row 0   = 90° N MLAT,  row 179 = 90° S MLAT  (1°/row)
  - Col 0   = MLT 0 h,    col 359 = MLT ≈ 24 h   (4 min/col, wraps)
  - Coverage: global (old data lands in rows 0–40, southern half stays zero)

Zoom factors per channel: spatial (120→40, 120→360) = (1/3, 3).
Output is placed in rows 0:40 of the full 180×360 canvas.

Usage:
    python scripts/convert_grid_120_to_180360.py \
        --cache_root /data5/iome_cache \
        [--workers 8] [--dry_run]

Converts superdarn/, supermag/, tec/ subdirectories in-place,
saving originals as *.npy.bak (use --no_backup to skip).
"""

import argparse
import shutil
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from scipy.ndimage import zoom


OLD_SIZE = 120
NEW_NLAT = 180
NEW_NMLT = 360
# Rows in new grid that correspond to 90°→50° N MLAT
NH_ROWS = 40   # ceil((90-50)/1°)


def _convert_file(args):
    path, backup, dry_run = args
    try:
        arr = np.load(path)
    except Exception as e:
        return path, f"LOAD ERROR: {e}"

    if arr.ndim != 3 or arr.shape[1:] != (OLD_SIZE, OLD_SIZE):
        # Already converted or unexpected shape
        return path, f"SKIP shape={arr.shape}"

    C = arr.shape[0]
    out = np.zeros((C, NEW_NLAT, NEW_NMLT), dtype=np.float32)

    for c in range(C):
        # bilinear zoom: rows 120→40 (÷3), cols 120→360 (×3)
        zoomed = zoom(arr[c].astype(np.float32), (NH_ROWS / OLD_SIZE, NEW_NMLT / OLD_SIZE), order=1)
        # Clip occupancy-like channels that might exceed [0,1] after interpolation
        out[c, 0:NH_ROWS, :] = np.clip(zoomed, arr[c].min(), arr[c].max())

    if dry_run:
        return path, f"DRY_RUN {arr.shape} -> {out.shape}"

    if backup:
        shutil.copy2(path, str(path) + ".bak")

    np.save(path, out)
    return path, f"OK {arr.shape} -> {out.shape}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", type=Path, default=Path("/data5/iome_cache"))
    ap.add_argument("--modalities", nargs="+", default=["superdarn", "supermag", "tec"])
    ap.add_argument("--workers",    type=int,  default=8)
    ap.add_argument("--no_backup",  action="store_true")
    ap.add_argument("--dry_run",    action="store_true")
    args = ap.parse_args()

    tasks = []
    for mod in args.modalities:
        d = args.cache_root / mod
        if not d.exists():
            print(f"  {mod}: directory not found, skipping")
            continue
        files = sorted(d.glob("*.npy"))
        print(f"  {mod}: {len(files)} files")
        for f in files:
            tasks.append((f, not args.no_backup, args.dry_run))

    print(f"\nConverting {len(tasks)} files with {args.workers} workers ...")
    print("(Backups saved as *.npy.bak — delete with: find /data5/iome_cache -name '*.bak' -delete)\n")

    ok = skipped = errors = 0
    with Pool(processes=args.workers) as pool:
        for path, msg in pool.imap_unordered(_convert_file, tasks, chunksize=64):
            if msg.startswith("OK"):
                ok += 1
            elif msg.startswith("SKIP"):
                skipped += 1
            else:
                errors += 1
                print(f"  ERROR {path.name}: {msg}")

        # Progress line every 10k
            total_done = ok + skipped + errors
            if total_done % 10_000 == 0:
                print(f"  {total_done}/{len(tasks)}  ok={ok} skip={skipped} err={errors}")

    print(f"\nDone: {ok} converted, {skipped} skipped (already new shape), {errors} errors")


if __name__ == "__main__":
    main()
