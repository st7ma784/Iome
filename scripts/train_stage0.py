"""
Stage 0: per-modality contrastive pretraining.

Trains each encoder independently using temporal positive pairs.
Run once per modality (or all four in parallel on separate processes).

Usage:
    python scripts/train_stage0.py \
        --modality   sd \
        --cache_dir  /data5/iome_cache/superdarn \
        --stats_dir  /data5/iome_cache/splits \
        --splits_dir /data5/iome_cache/splits \
        --ckpt_dir   /data5/iome_cache/ckpts/stage0 \
        --wandb_project iome

Saved artefacts (in ckpt_dir):
    stage0_{mod}_encoder.pt   — encoder state_dict only (projector discarded)
    stage0_{mod}_last.ckpt    — full Lightning checkpoint (resume training)
"""

import argparse
import json
import numpy as np
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

from iome.models.encoders import (
    SuperDARNEncoder, SuperMAGEncoder, TECEncoder, DMSPEncoder,
)
from iome.data.superdarn import SuperDARNDataset
from iome.data.supermag  import SuperMAGDataset
from iome.data.tec       import TECDataset
from iome.data.dmsp      import DMSPDataset
from iome.data.pairs     import TemporalPairDataset
from iome.train.stage0   import Stage0ModalityModule


ENCODER_CLS = {
    "sd":   SuperDARNEncoder,
    "smag": SuperMAGEncoder,
    "tec":  TECEncoder,
    "dmsp": DMSPEncoder,
}

DATASET_CLS = {
    "sd":   SuperDARNDataset,
    "smag": SuperMAGDataset,
    "tec":  TECDataset,
    "dmsp": DMSPDataset,
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality",    required=True, choices=list(ENCODER_CLS))
    ap.add_argument("--cache_dir",   type=Path, required=True)
    ap.add_argument("--splits_dir",  type=Path, default=None)
    ap.add_argument("--ts_train",    type=Path, default=None)
    ap.add_argument("--ts_val",      type=Path, default=None)
    ap.add_argument("--stats_dir",   type=Path, default=None)
    ap.add_argument("--ckpt_dir",    type=Path, required=True)
    ap.add_argument("--window_steps",type=int,   default=15,
                    help="Max positive offset in 2-min steps (default 15 = ±30 min)")
    ap.add_argument("--wandb_project", default="iome")
    ap.add_argument("--wandb_entity",  default="st7ma784")
    ap.add_argument("--wandb_name",    default=None)
    ap.add_argument("--accelerator",   default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--devices",       type=int, default=1)
    ap.add_argument("--precision",     default="bf16-mixed")
    ap.add_argument("--batch_size",    type=int,   default=64)
    ap.add_argument("--max_steps",     type=int,   default=20_000)
    ap.add_argument("--num_workers",   type=int,   default=8)
    ap.add_argument("--latent_dim",    type=int,   default=256)
    ap.add_argument("--proj_dim",      type=int,   default=128)
    ap.add_argument("--tau",           type=float, default=0.07)
    ap.add_argument("--lr",            type=float, default=1e-3)
    return ap.parse_args()


def _load_json(path):
    if path and Path(path).exists():
        return json.loads(Path(path).read_text())
    return None


def _load_stats(stats_dir, mod):
    if stats_dir is None:
        return None
    p = Path(stats_dir) / f"stats_{mod}.npy"
    return np.load(p, allow_pickle=True).item() if p.exists() else None


def main():
    args = parse_args()
    mod  = args.modality
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    sd = args.splits_dir
    ts_train = _load_json(args.ts_train or (sd / "ts_train.json" if sd else None))
    ts_val   = _load_json(args.ts_val   or (sd / "ts_val.json"   if sd else None))
    if ts_train is None:
        raise ValueError("Provide --splits_dir or --ts_train")
    if ts_val is None:
        ts_val = ts_train[:2_000]

    stats = _load_stats(args.stats_dir, mod)

    base_train = DATASET_CLS[mod](timestamps=ts_train, cache_dir=args.cache_dir,
                                  stats=stats, delta_t_steps=args.window_steps)
    base_val   = DATASET_CLS[mod](timestamps=ts_val,   cache_dir=args.cache_dir,
                                  stats=stats, delta_t_steps=args.window_steps)

    ds_train = TemporalPairDataset(base_train, mod_key=mod, window_steps=args.window_steps)
    ds_val   = TemporalPairDataset(base_val,   mod_key=mod, window_steps=args.window_steps)

    nw = args.num_workers
    loader_kwargs = dict(num_workers=nw, pin_memory=False)
    if nw > 0:
        loader_kwargs["multiprocessing_context"] = "fork"   # forkserver deadlocks on hdd01
        loader_kwargs["persistent_workers"] = True
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,  **loader_kwargs)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    encoder = ENCODER_CLS[mod](latent_dim=args.latent_dim)
    module  = Stage0ModalityModule(
        encoder=encoder,
        modality=mod,
        latent_dim=args.latent_dim,
        proj_dim=args.proj_dim,
        tau=args.tau,
        lr=args.lr,
        max_steps=args.max_steps,
    )

    run_name = args.wandb_name or f"stage0-{mod}"
    logger   = WandbLogger(project=args.wandb_project, entity=args.wandb_entity,
                           name=run_name)
    ckpt_cb  = ModelCheckpoint(
        dirpath=str(args.ckpt_dir),
        filename=f"stage0-{mod}-{{step:06d}}-{{val/{mod}/loss:.4f}}",
        monitor=f"val/{mod}/loss",
        mode="min",
        save_top_k=2,
        save_last=True,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_steps=args.max_steps,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        logger=logger,
        callbacks=[ckpt_cb, lr_cb],
        val_check_interval=500,
        log_every_n_steps=50,
    )
    trainer.fit(module, dl_train, dl_val)

    # Save encoder weights only (projector discarded)
    encoder_path = args.ckpt_dir / f"stage0_{mod}_encoder.pt"
    torch.save(module.encoder.state_dict(), encoder_path)
    print(f"[stage0] Encoder saved → {encoder_path}")


if __name__ == "__main__":
    main()
