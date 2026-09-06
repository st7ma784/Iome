"""
Compute per-modality normalisation statistics (mean and std per channel)
from a random sample of cached .npy files.

Output: {cache_root}/splits/stats_{sd,smag,tec}.npy  (dict with 'mean','std')

Usage:
    python scripts/compute_stats.py \
        --cache_root /data5/iome_cache \
        --out_dir    /data5/iome_cache/splits \
        --n_samples  2000
"""

import argparse
import random
from pathlib import Path

import numpy as np


def _sample_files(directory: Path, pattern: str, n: int, seed: int) -> list[Path]:
    all_files = sorted(directory.glob(pattern))
    if len(all_files) <= n:
        return all_files
    rng = random.Random(seed)
    return rng.sample(all_files, n)


def compute_stats(files: list[Path], n_channels: int) -> dict:
    """Welford online algorithm for mean and M2 over all files."""
    count = 0
    mean  = np.zeros(n_channels, dtype=np.float64)
    M2    = np.zeros(n_channels, dtype=np.float64)

    for p in files:
        try:
            arr = np.load(p).astype(np.float32)   # (C, H, W)
        except Exception:
            continue

        if arr.ndim != 3 or arr.shape[0] != n_channels:
            continue

        # Compute per-channel mean over spatial dims for this sample
        for c in range(n_channels):
            vals = arr[c].ravel()
            # Exclude NaN and zeros (unobserved)
            vals = vals[np.isfinite(vals) & (vals != 0.0)]
            if len(vals) == 0:
                continue
            for v in vals:
                count += 1
                delta = v - mean[c]
                mean[c]  += delta / count
                delta2 = v - mean[c]
                M2[c]   += delta * delta2

    # Reset count per channel approach (simpler):
    # Actually let's just do a vectorised pass for efficiency
    return None   # placeholder — see below


def compute_stats_v2(files: list[Path], n_channels: int) -> dict:
    """Collect per-channel values in arrays, compute mean/std."""
    accum = [[] for _ in range(n_channels)]

    for p in files:
        try:
            arr = np.load(p).astype(np.float32)
        except Exception:
            continue
        if arr.ndim != 3 or arr.shape[0] != n_channels:
            continue
        for c in range(n_channels):
            vals = arr[c].ravel()
            vals = vals[np.isfinite(vals) & (vals != 0.0)]
            if len(vals):
                # subsample to keep memory reasonable
                if len(vals) > 500:
                    idx = np.random.choice(len(vals), 500, replace=False)
                    vals = vals[idx]
                accum[c].append(vals)

    mean = np.zeros(n_channels, dtype=np.float32)
    std  = np.ones(n_channels,  dtype=np.float32)
    for c in range(n_channels):
        if accum[c]:
            all_vals = np.concatenate(accum[c])
            mean[c] = float(np.nanmean(all_vals))
            std[c]  = float(np.nanstd(all_vals))
            if std[c] < 1e-6:
                std[c] = 1.0

    return {"mean": mean, "std": std}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", type=Path, default=Path("/data5/iome_cache"))
    ap.add_argument("--out_dir",    type=Path, default=None)
    ap.add_argument("--n_samples",  type=int,  default=2000)
    ap.add_argument("--seed",       type=int,  default=42)
    args = ap.parse_args()

    out_dir = args.out_dir or args.cache_root / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    modalities = [
        ("sd",   args.cache_root / "superdarn", "*_sd.npy",   6),
        ("smag", args.cache_root / "supermag",  "*_smag.npy", 3),
        ("tec",  args.cache_root / "tec",       "*_tec.npy",  2),
        ("dmsp", args.cache_root / "dmsp",      "*_dmsp.npy", 5),
    ]

    for name, directory, pattern, n_chans in modalities:
        if not directory.exists():
            print(f"  {name}: directory not found, skipping")
            continue

        files = _sample_files(directory, pattern, args.n_samples, args.seed)
        print(f"  {name}: {len(files)} files sampled", flush=True)

        stats = compute_stats_v2(files, n_chans)
        out_path = out_dir / f"stats_{name}.npy"
        np.save(out_path, stats)

        print(f"    mean: {stats['mean']}")
        print(f"    std:  {stats['std']}")
        print(f"    → {out_path}")


if __name__ == "__main__":
    main()
