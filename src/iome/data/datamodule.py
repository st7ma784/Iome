"""
TriModalDataModule: PyTorch Lightning DataModule for all four modalities.

Supports training with any available subset of sensors per epoch (union
timestamps).  Missing modalities are simply absent from the batch dict —
the model handles any combination via modality dropout training.

Each batch item:
    {
        "xs":        {mod: (B, C, H, W)},   # inputs at t   (available mods only)
        "ys":        {mod: (B, C, H, W)},   # reconstruction targets (= xs)
        "xs_next":   {mod: (B, C, H, W)},   # inputs at t+1
        "ys_next":   {mod: (B, C, H, W)},
        "u":         (B, u_dim),             # OMNI solar-wind
        "omni_mask": (B, 1),                 # 1.0 if OMNI valid
        "avail":     {mod: bool},            # which modalities are present
        "ts":        List[str],
    }
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

from .superdarn import SuperDARNDataset
from .supermag   import SuperMAGDataset
from .tec        import TECDataset
from .dmsp       import DMSPDataset

MODALITIES = ("sd", "smag", "tec", "dmsp")


# ---------------------------------------------------------------------------
# OMNI loader
# ---------------------------------------------------------------------------

def load_omni(
    timestamps: List[str],
    omni_dir: Optional[Path] = None,
) -> tuple[np.ndarray, np.ndarray]:
    from datetime import datetime
    N   = len(timestamps)
    u   = np.zeros((N, 8), dtype=np.float32)
    msk = np.zeros((N, 1), dtype=np.float32)
    if omni_dir is None or not Path(omni_dir).exists():
        return u, msk
    year_cache: dict[int, dict] = {}
    for i, ts in enumerate(timestamps):
        try:
            fmt = "%Y%m%dT%H%M" if len(ts) == 13 else "%Y-%m-%dT%H:%M"
            dt  = datetime.strptime(ts, fmt)
        except ValueError:
            continue
        year = dt.year
        doy  = dt.timetuple().tm_yday
        key  = f"{year:04d}{doy:03d}T{dt.hour:02d}"
        if year not in year_cache:
            p = Path(omni_dir) / f"omni_{year}.npy"
            year_cache[year] = np.load(p, allow_pickle=True).item() if p.exists() else {}
        row = year_cache[year].get(key)
        if row is not None:
            r = np.where(np.isfinite(np.array(row[:8], dtype=np.float32)),
                         np.array(row[:8], dtype=np.float32), 0.0)
            u[i]   = r
            msk[i] = 1.0
    return u, msk


# ---------------------------------------------------------------------------
# Joined quad-modal dataset
# ---------------------------------------------------------------------------

class QuadModalDataset(Dataset):
    """
    Joins up to four per-modality datasets on shared timestamps.
    Each sample returns only the modalities that have data at that timestamp.

    Two lag-correction modes (mutually exclusive, lag_offsets takes priority):

    lag_offsets — fixed lag per modality, from analyse_lag.py.  Each modality
        in xs_aligned is loaded at t + lag_offsets[mod].  Fast: no extra IO.

    align_window_steps — dynamic soft attention lag.  Returns xs_window[mod]
        as a (K, C, H, W) tensor of K snapshots at t, t+1, ..., t+K-1.
        Stage1 encodes this window and attends over it with TemporalAlignmentHead.
        K = align_window_steps.  This path is more expensive but allows the lag
        to vary per sample based on magnetospheric state.
    """

    def __init__(
        self,
        timestamps: List[str],
        avail_map:  Dict[str, List[str]],   # ts → [mod, ...]
        sd_dataset:   Optional[SuperDARNDataset] = None,
        smag_dataset: Optional[SuperMAGDataset]  = None,
        tec_dataset:  Optional[TECDataset]       = None,
        dmsp_dataset: Optional[DMSPDataset]      = None,
        u:            Optional[np.ndarray] = None,
        omni_mask:    Optional[np.ndarray] = None,
        delta_t_steps:      int = 1,
        lag_offsets:        Optional[Dict[str, int]] = None,
        align_window_steps: int = 0,
    ):
        self._ts    = timestamps
        self._avail = avail_map
        self._dsets = {
            "sd":   sd_dataset,
            "smag": smag_dataset,
            "tec":  tec_dataset,
            "dmsp": dmsp_dataset,
        }
        self._u              = u
        self._omni_mask      = omni_mask
        self._delta          = delta_t_steps
        self._lag            = lag_offsets or {}
        self._window_steps   = align_window_steps  # K; 0 = disabled

        # Trim valid range to avoid index overflow for any loaded offset
        fixed_offsets = list(self._lag.values()) if self._lag else []
        all_offsets   = fixed_offsets + [delta_t_steps]
        if align_window_steps > 0:
            all_offsets.append(align_window_steps - 1)
        min_off = min(0, min(all_offsets))
        max_off = max(all_offsets)
        self._valid = list(range(-min_off, len(timestamps) - max_off))

    def __len__(self):
        return len(self._valid)

    def __getitem__(self, idx):
        i  = self._valid[idx]
        ts = self._ts[i]
        avail = set(self._avail.get(ts, list(self._dsets.keys())))

        sample = {"xs": {}, "ys": {}, "xs_next": {}, "ys_next": {},
                  "xs_aligned": {}, "xs_window": {},
                  "ts": ts, "avail": {mod: (mod in avail) for mod in MODALITIES}}

        for mod, ds in self._dsets.items():
            if ds is None or mod not in avail:
                continue
            x      = ds._load(self._ts[i])
            x_next = ds._load(self._ts[i + self._delta])
            sample["xs"][mod]      = x
            sample["ys"][mod]      = x
            sample["xs_next"][mod] = x_next
            sample["ys_next"][mod] = x_next

            # Fixed-lag alignment (from analyse_lag.py output)
            if self._lag:
                off = self._lag.get(mod, 0)
                sample["xs_aligned"][mod] = ds._load(self._ts[i + off]) if off != 0 else x
            else:
                sample["xs_aligned"][mod] = x

            # Dynamic-lag window: K snapshots at t, t+1, ..., t+K-1
            if self._window_steps > 0:
                frames = torch.stack([
                    ds._load(self._ts[i + k]) for k in range(self._window_steps)
                ])  # (K, C, H, W)
                sample["xs_window"][mod] = frames

        if self._u is not None:
            sample["u"]         = torch.from_numpy(self._u[i])
            sample["omni_mask"] = torch.from_numpy(self._omni_mask[i])
        else:
            sample["u"]         = torch.zeros(8)
            sample["omni_mask"] = torch.zeros(1)

        return sample


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class TriModalDataModule(pl.LightningDataModule):
    """
    Args:
        timestamps_train / val / test: sorted lists of "YYYYMMDDTHHMM" strings
        avail_map:  dict {ts: [mod, ...]} from make_timestamps.py (optional;
                    if None all modalities are assumed present at every ts)
        cache_dir_{sd,smag,tec,dmsp}: local .npy cache directories
        stats_{sd,smag,tec,dmsp}: normalisation dicts
        omni_dir:   directory of omni_{YYYY}.npy files
        batch_size, num_workers, pin_memory: DataLoader kwargs
        delta_t_steps: steps between t and t+1
        lag_matrix: compact lag dict from analyse_lag.py, e.g.
                    {"smag->sd": 8, "smag->tec": 5, ...} — used to build
                    per-modality offsets for lag-corrected cross-modal CLIP
    """

    def __init__(
        self,
        timestamps_train: List[str],
        timestamps_val:   List[str],
        timestamps_test:  List[str],
        avail_map:      Optional[Dict[str, List[str]]] = None,
        cache_dir_sd:   Optional[Path] = None,
        cache_dir_smag: Optional[Path] = None,
        cache_dir_tec:  Optional[Path] = None,
        cache_dir_dmsp: Optional[Path] = None,
        stats_sd:   Optional[dict] = None,
        stats_smag: Optional[dict] = None,
        stats_tec:  Optional[dict] = None,
        stats_dmsp: Optional[dict] = None,
        omni_dir:   Optional[Path] = None,
        batch_size:   int = 16,
        num_workers:  int = 16,
        pin_memory:   bool = False,
        delta_t_steps: int = 1,
        lag_matrix:         Optional[Dict[str, int]] = None,
        align_window_steps: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[
            "timestamps_train", "timestamps_val", "timestamps_test", "avail_map",
        ])
        self._timestamps = {
            "train": timestamps_train,
            "val":   timestamps_val,
            "test":  timestamps_test,
        }
        self._avail_map = avail_map or {}
        self._lag_offsets = _lag_offsets_from_matrix(lag_matrix) if lag_matrix else {}

    def _make_dset(self, split: str) -> QuadModalDataset:
        ts = self._timestamps[split]
        hp = self.hparams

        sd_ds = SuperDARNDataset(
            timestamps=ts, cache_dir=hp.cache_dir_sd,
            stats=hp.stats_sd, delta_t_steps=hp.delta_t_steps,
        ) if hp.cache_dir_sd else None

        smag_ds = SuperMAGDataset(
            timestamps=ts, cache_dir=hp.cache_dir_smag,
            stats=hp.stats_smag, delta_t_steps=hp.delta_t_steps,
        ) if hp.cache_dir_smag else None

        tec_ds = TECDataset(
            timestamps=ts, cache_dir=hp.cache_dir_tec,
            stats=hp.stats_tec, delta_t_steps=hp.delta_t_steps,
        ) if hp.cache_dir_tec else None

        dmsp_ds = DMSPDataset(
            timestamps=ts, cache_dir=hp.cache_dir_dmsp,
            stats=hp.stats_dmsp, delta_t_steps=hp.delta_t_steps,
        ) if hp.cache_dir_dmsp else None

        u, mask = load_omni(ts, hp.omni_dir)

        return QuadModalDataset(
            timestamps=ts,
            avail_map=self._avail_map,
            sd_dataset=sd_ds,
            smag_dataset=smag_ds,
            tec_dataset=tec_ds,
            dmsp_dataset=dmsp_ds,
            u=u,
            omni_mask=mask,
            delta_t_steps=hp.delta_t_steps,
            lag_offsets=self._lag_offsets or None,
            align_window_steps=hp.align_window_steps,
        )

    def setup(self, stage=None):
        self._train = self._make_dset("train")
        self._val   = self._make_dset("val")
        self._test  = self._make_dset("test")

    def _loader(self, dset, shuffle):
        nw = self.hparams.num_workers
        kwargs = {}
        if nw > 0:
            kwargs["multiprocessing_context"] = "forkserver"
            kwargs["persistent_workers"] = True
        return DataLoader(
            dset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=nw,
            pin_memory=self.hparams.pin_memory,
            collate_fn=_collate,
            **kwargs,
        )

    def train_dataloader(self): return self._loader(self._train, shuffle=True)
    def val_dataloader(self):   return self._loader(self._val,   shuffle=False)
    def test_dataloader(self):  return self._loader(self._test,  shuffle=False)


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def _lag_offsets_from_matrix(lag_matrix: Dict[str, int]) -> Dict[str, int]:
    """
    Convert the compact lag_matrix (from analyse_lag.py) into per-modality
    load offsets relative to smag as the causal reference.

    lag_matrix["smag->X"] = k means smag leads X by k steps — the sd/tec/dmsp
    response to a smag substorm onset manifests k steps later.  Loading X at
    t + k aligns it with smag at t on the same causal moment.
    """
    offsets: Dict[str, int] = {"smag": 0}
    for key, k in lag_matrix.items():
        a, b = key.split("->")
        if a == "smag" and b not in offsets:
            offsets[b] = max(0, k)   # only forward offsets make causal sense
    for mod in ("sd", "tec", "dmsp"):
        offsets.setdefault(mod, 0)
    return offsets


def _collate(batch: List[dict]) -> dict:
    keys = batch[0].keys()
    out  = {}
    for k in keys:
        if k in ("xs", "ys", "xs_next", "ys_next", "xs_aligned"):
            # Only stack modalities present in ALL items of the batch
            all_mods = set(batch[0][k].keys())
            for b in batch[1:]:
                all_mods &= set(b[k].keys())
            out[k] = {mod: torch.stack([b[k][mod] for b in batch]) for mod in all_mods}
        elif k == "xs_window":
            # xs_window[mod] per sample is (K, C, H, W); batch to (B, K, C, H, W)
            all_mods = set(batch[0][k].keys())
            for b in batch[1:]:
                all_mods &= set(b[k].keys())
            out[k] = {mod: torch.stack([b[k][mod] for b in batch]) for mod in all_mods}
        elif k == "avail":
            # Union across batch: mark mod available if ANY item has it
            out[k] = {mod: any(b[k].get(mod, False) for b in batch) for mod in MODALITIES}
        elif k == "ts":
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out
