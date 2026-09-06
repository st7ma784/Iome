"""
Post-Phase-0 causal lag analysis.

After per-modality encoders are pretrained, encode all training timestamps
and compute pairwise cross-correlations in latent space as a function of time
lag. The lag at which similarity peaks is the empirical propagation delay
between modalities.

Also runs superposed epoch analysis around SuperMAG onset events (large
|dBn/dt| = substorm proxy) to get a physics-grounded lag distribution.

Outputs:
    lag_matrix.json   — {("a","b"): lag_steps} for all pairs, positive = b leads a
    lag_curves.npz    — full correlation vs lag arrays for plotting

Usage:
    python scripts/analyse_lag.py \
        --ckpt_stage0_dir /data/iome_cache/ckpts/stage0 \
        --splits_dir      /data/iome_cache/splits \
        --cache_sd        /data/iome_cache/superdarn \
        --cache_smag      /data/iome_cache/supermag \
        --cache_tec       /data/iome_cache/tec \
        --cache_dmsp      /data/iome_cache/dmsp \
        --stats_dir       /data/iome_cache/splits \
        --out             /data/iome_cache/splits/lag_matrix.json \
        --max_lag_steps   30 \
        --n_samples       5000
"""

import argparse
import json
import itertools
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from iome.models.encoders import (
    SuperDARNEncoder, SuperMAGEncoder, TECEncoder, DMSPEncoder,
)
from iome.data.superdarn import SuperDARNDataset
from iome.data.supermag  import SuperMAGDataset
from iome.data.tec       import TECDataset
from iome.data.dmsp      import DMSPDataset

ENCODER_CLS = {"sd": SuperDARNEncoder, "smag": SuperMAGEncoder,
               "tec": TECEncoder,      "dmsp": DMSPEncoder}
DATASET_CLS = {"sd": SuperDARNDataset, "smag": SuperMAGDataset,
               "tec": TECDataset,      "dmsp": DMSPDataset}
CACHE_NAMES = {"sd": "superdarn", "smag": "supermag", "tec": "tec", "dmsp": "dmsp"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_stage0_dir", type=Path, required=True)
    ap.add_argument("--splits_dir",      type=Path, required=True)
    ap.add_argument("--cache_sd",        type=Path, default=None)
    ap.add_argument("--cache_smag",      type=Path, default=None)
    ap.add_argument("--cache_tec",       type=Path, default=None)
    ap.add_argument("--cache_dmsp",      type=Path, default=None)
    ap.add_argument("--stats_dir",       type=Path, default=None)
    ap.add_argument("--out",             type=Path,
                    default=Path("lag_matrix.json"))
    ap.add_argument("--max_lag_steps",   type=int, default=30,
                    help="Maximum lag to test in 2-min steps (default 30 = 60 min)")
    ap.add_argument("--n_samples",       type=int, default=5000,
                    help="Number of timestamps to encode (random subsample)")
    ap.add_argument("--latent_dim",      type=int, default=256)
    ap.add_argument("--batch_size",      type=int, default=32)
    return ap.parse_args()


def load_encoder(mod: str, ckpt_dir: Path, latent_dim: int) -> torch.nn.Module:
    enc = ENCODER_CLS[mod](latent_dim=latent_dim)
    p   = ckpt_dir / f"stage0_{mod}_encoder.pt"
    if p.exists():
        enc.load_state_dict(torch.load(p, map_location="cpu"))
        print(f"  Loaded {p.name}")
    else:
        print(f"  WARNING: {p} not found — using random weights")
    enc.eval()
    return enc


def encode_all(
    encoder: torch.nn.Module,
    dataset,
    timestamps: list,
    n_samples: int,
    batch_size: int,
) -> tuple[np.ndarray, list]:
    """Encode a random subset of timestamps. Returns (z_array, ts_list)."""
    rng = np.random.default_rng(42)
    idx = rng.choice(len(timestamps), size=min(n_samples, len(timestamps)), replace=False)
    idx = np.sort(idx)
    selected_ts = [timestamps[i] for i in idx]

    zs = []
    with torch.no_grad():
        for start in range(0, len(selected_ts), batch_size):
            batch_ts = selected_ts[start : start + batch_size]
            tensors  = torch.stack([dataset._load(ts) for ts in batch_ts])
            z        = F.normalize(encoder(tensors), dim=1)
            zs.append(z.numpy())

    return np.concatenate(zs, axis=0), selected_ts


def cross_corr_at_lag(
    z_a: np.ndarray, ts_a: list,
    z_b: np.ndarray, ts_b: list,
    lag_steps: int,
) -> float:
    """
    Mean cosine similarity between z_a(t) and z_b(t + lag_steps).

    Builds a timestamp → index map for b, then looks up t+lag for each t in a.
    Only timestamps present in both are counted.
    """
    ts_b_map = {ts: i for i, ts in enumerate(ts_b)}

    # Represent timestamps as indices into the global sorted ts list
    # and compute offset naively via string lookup of shifted timestamps.
    # Simpler: convert to datetime offsets
    from datetime import datetime, timedelta

    def parse_ts(ts):
        for fmt in ("%Y%m%dT%H%M", "%Y-%m-%dT%H:%M"):
            try:
                return datetime.strptime(ts, fmt)
            except ValueError:
                pass
        raise ValueError(f"Unknown timestamp format: {ts}")

    delta = timedelta(minutes=2 * lag_steps)

    sims = []
    for i, ts in enumerate(ts_a):
        dt_shifted = parse_ts(ts) + delta
        # Re-format in both candidate formats
        ts_shifted_long  = dt_shifted.strftime("%Y-%m-%dT%H:%M")
        ts_shifted_short = dt_shifted.strftime("%Y%m%dT%H%M")
        j = ts_b_map.get(ts_shifted_long, ts_b_map.get(ts_shifted_short))
        if j is not None:
            sims.append(float(z_a[i] @ z_b[j]))

    return float(np.mean(sims)) if sims else float("nan")


def superposed_epoch_analysis(
    z_smag: np.ndarray, ts_smag: list,
    z_other: np.ndarray, ts_other: list,
    mod_other: str,
    max_lag_steps: int,
    onset_percentile: float = 95.0,
) -> np.ndarray:
    """
    Use large |z_smag| norm as a substorm onset proxy.
    For each onset event, measure z_other norm at t+lag for lag in [-max, +max].
    Returns mean z_other norm as function of lag (shape: 2*max_lag+1).
    """
    norms_smag = np.linalg.norm(z_smag, axis=1)
    threshold  = np.percentile(norms_smag, onset_percentile)
    onset_mask = norms_smag > threshold

    from datetime import datetime, timedelta
    def parse_ts(ts):
        for fmt in ("%Y%m%dT%H%M", "%Y-%m-%dT%H:%M"):
            try: return datetime.strptime(ts, fmt)
            except ValueError: pass

    ts_other_map = {ts: i for i, ts in enumerate(ts_other)}
    lags = list(range(-max_lag_steps, max_lag_steps + 1))
    epoch_norms = [[] for _ in lags]

    onset_indices = np.where(onset_mask)[0]
    for oi in onset_indices:
        t0 = parse_ts(ts_smag[oi])
        for li, lag in enumerate(lags):
            dt = t0 + timedelta(minutes=2 * lag)
            ts_shifted = dt.strftime("%Y%m%dT%H%M")
            j = ts_other_map.get(ts_shifted)
            if j is not None:
                epoch_norms[li].append(np.linalg.norm(z_other[j]))

    return np.array([np.mean(v) if v else np.nan for v in epoch_norms])


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    ts_train = json.loads((args.splits_dir / "ts_train.json").read_text())
    print(f"Training timestamps: {len(ts_train)}")

    cache_dirs = {
        "sd":   args.cache_sd,
        "smag": args.cache_smag,
        "tec":  args.cache_tec,
        "dmsp": args.cache_dmsp,
    }

    def load_stats(mod):
        if args.stats_dir is None: return None
        p = args.stats_dir / f"stats_{mod}.npy"
        return np.load(p, allow_pickle=True).item() if p.exists() else None

    # Load available modalities
    encoders, datasets = {}, {}
    for mod in ("sd", "smag", "tec", "dmsp"):
        cdir = cache_dirs[mod]
        if cdir is None or not Path(cdir).exists():
            print(f"  Skipping {mod} (no cache dir)")
            continue
        print(f"\nLoading {mod}...")
        encoders[mod] = load_encoder(mod, args.ckpt_stage0_dir, args.latent_dim)
        datasets[mod] = DATASET_CLS[mod](
            timestamps=ts_train,
            cache_dir=cdir,
            stats=load_stats(mod),
            delta_t_steps=1,
        )

    # Encode all available modalities
    print(f"\nEncoding {args.n_samples} timestamps per modality...")
    latents = {}
    for mod, enc in encoders.items():
        print(f"  {mod}...", flush=True)
        z, ts = encode_all(enc, datasets[mod], ts_train,
                           args.n_samples, args.batch_size)
        latents[mod] = (z, ts)
        print(f"  {mod}: {z.shape[0]} samples encoded, mean norm={np.linalg.norm(z, axis=1).mean():.3f}")

    # Pairwise cross-correlation vs lag
    mods     = list(latents.keys())
    lags     = list(range(-args.max_lag_steps, args.max_lag_steps + 1))
    lag_matrix    = {}   # (a, b) → peak lag in steps
    all_curves    = {}   # (a, b) → array of corr values

    print(f"\nComputing pairwise cross-correlations (lag = {lags[0]}…{lags[-1]} steps)...")
    for mod_a, mod_b in itertools.permutations(mods, 2):
        z_a, ts_a = latents[mod_a]
        z_b, ts_b = latents[mod_b]
        curve = []
        for lag in tqdm(lags, desc=f"{mod_a}→{mod_b}", leave=False):
            curve.append(cross_corr_at_lag(z_a, ts_a, z_b, ts_b, lag))
        curve = np.array(curve)
        all_curves[(mod_a, mod_b)] = curve

        # Peak lag: positive lag means z_b(t+lag) aligns best with z_a(t)
        # → z_b lags z_a by `peak_lag` steps
        valid = ~np.isnan(curve)
        if valid.any():
            peak_idx  = int(np.nanargmax(curve))
            peak_lag  = lags[peak_idx]
            peak_corr = float(curve[peak_idx])
        else:
            peak_lag, peak_corr = 0, float("nan")

        lag_matrix[f"{mod_a}->{mod_b}"] = {
            "lag_steps":   peak_lag,
            "lag_minutes": peak_lag * 2,
            "peak_corr":   peak_corr,
        }
        print(f"  {mod_a} → {mod_b}: peak lag = {peak_lag} steps ({peak_lag*2} min), "
              f"corr = {peak_corr:.4f}")

    # Superposed epoch analysis around SuperMAG onsets
    if "smag" in latents:
        print("\nSuperposed epoch analysis (SuperMAG onset proxy)...")
        sea_curves = {}
        z_smag, ts_smag = latents["smag"]
        for mod_b in [m for m in mods if m != "smag"]:
            z_b, ts_b = latents[mod_b]
            sea = superposed_epoch_analysis(
                z_smag, ts_smag, z_b, ts_b, mod_b, args.max_lag_steps
            )
            sea_curves[f"smag_onset->{mod_b}"] = sea
            valid = ~np.isnan(sea)
            if valid.any():
                peak_idx = int(np.nanargmax(sea))
                peak_lag = lags[peak_idx]
                print(f"  SuperMAG onset → {mod_b}: response peaks at lag={peak_lag} steps "
                      f"({peak_lag*2} min)")
    else:
        sea_curves = {}

    # Save outputs
    result = {
        "lag_steps_range": [lags[0], lags[-1]],
        "step_duration_min": 2,
        "pairs": lag_matrix,
        "superposed_epoch": {k: v.tolist() for k, v in sea_curves.items()},
        # Compact lag-only dict for use as --lag_matrix in train_stage1.py
        "lag_matrix": {
            k: v["lag_steps"] for k, v in lag_matrix.items()
        },
    }
    args.out.write_text(json.dumps(result, indent=2))
    print(f"\nLag matrix saved → {args.out}")

    # Save full curves for plotting
    curves_path = args.out.with_suffix(".npz")
    np.savez(curves_path,
             lags=np.array(lags),
             **{f"{'_'.join(k)}": v for k, v in all_curves.items()},
             **{k.replace("->", "_"): v for k, v in sea_curves.items()})
    print(f"Full curves saved → {curves_path}")

    # Print recommended Phase 1 delta_t_steps
    if "smag->sd" in lag_matrix:
        smag_to_sd = lag_matrix["smag->sd"]["lag_steps"]
        print(f"\nRecommended --delta_t_steps for Phase 1: {max(smag_to_sd, 8)} "
              f"(smag→sd lag = {smag_to_sd} steps = {smag_to_sd*2} min)")


if __name__ == "__main__":
    main()
