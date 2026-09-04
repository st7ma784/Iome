"""Stage 3 training entry point: end-to-end fine-tuning."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from iome.models.fusion   import UnifiedIonosphereModel
from iome.train.stage3    import Stage3FinetuneModule
from iome.data.datamodule import TriModalDataModule


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2_ckpt", type=Path, required=True)
    # Splits
    ap.add_argument("--splits_dir",  type=Path, default=None)
    ap.add_argument("--ts_train",    type=Path, default=None)
    ap.add_argument("--ts_val",      type=Path, default=None)
    ap.add_argument("--ts_test",     type=Path, default=None)
    ap.add_argument("--ts_avail",    type=Path, default=None)
    # Caches
    ap.add_argument("--cache_sd",    type=Path, default=None)
    ap.add_argument("--cache_smag",  type=Path, default=None)
    ap.add_argument("--cache_tec",   type=Path, default=None)
    ap.add_argument("--cache_dmsp",  type=Path, default=None)
    ap.add_argument("--stats_dir",   type=Path, default=None)
    ap.add_argument("--omni_dir",    type=Path, default=None)
    ap.add_argument("--ckpt_dir",    type=Path, required=True)
    # Training
    ap.add_argument("--wandb_project", default="iome")
    ap.add_argument("--wandb_entity",  default="st7ma784")
    ap.add_argument("--accelerator",   default="gpu", choices=["gpu","cpu"])
    ap.add_argument("--devices",       type=int, default=1)
    ap.add_argument("--precision",     default="bf16-mixed")
    ap.add_argument("--batch_size",  type=int,   default=32)
    ap.add_argument("--max_steps",   type=int,   default=30_000)
    ap.add_argument("--num_workers", type=int,   default=16)
    ap.add_argument("--latent_dim",  type=int,   default=256)
    ap.add_argument("--p_mod_drop",  type=float, default=0.15)
    return ap.parse_args()


def _load_json(path):
    if path and path.exists():
        return json.loads(path.read_text())
    return None


def _load_stats(stats_dir, mod):
    if stats_dir is None:
        return None
    p = stats_dir / f"stats_{mod}.npy"
    return np.load(p, allow_pickle=True).item() if p.exists() else None


def main():
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    sd = args.splits_dir
    ts_train  = _load_json(args.ts_train or (sd / "ts_train.json" if sd else None))
    ts_val    = _load_json(args.ts_val   or (sd / "ts_val.json"   if sd else None))
    ts_test   = _load_json(args.ts_test  or (sd / "ts_test.json"  if sd else None))
    avail_map = _load_json(args.ts_avail or (sd / "ts_avail.json" if sd else None))

    if ts_train is None:
        raise ValueError("Provide --splits_dir or --ts_train")

    dm = TriModalDataModule(
        timestamps_train=ts_train,
        timestamps_val=ts_val or ts_train[:2_000],
        timestamps_test=ts_test or ts_train[:2_000],
        avail_map=avail_map,
        cache_dir_sd=args.cache_sd,
        cache_dir_smag=args.cache_smag,
        cache_dir_tec=args.cache_tec,
        cache_dir_dmsp=args.cache_dmsp,
        stats_sd=_load_stats(args.stats_dir, "sd"),
        stats_smag=_load_stats(args.stats_dir, "smag"),
        stats_tec=_load_stats(args.stats_dir, "tec"),
        stats_dmsp=_load_stats(args.stats_dir, "dmsp"),
        omni_dir=args.omni_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    model = UnifiedIonosphereModel(latent_dim=args.latent_dim)
    s2_ckpt = torch.load(args.stage2_ckpt, map_location="cpu")
    state   = {k.removeprefix("model."): v for k, v in s2_ckpt["state_dict"].items()}
    model.load_state_dict(state, strict=False)

    module = Stage3FinetuneModule(
        model=model,
        max_steps=args.max_steps,
        p_mod_drop=args.p_mod_drop,
    )

    logger  = WandbLogger(project=args.wandb_project, entity=args.wandb_entity, name="stage3")
    ckpt_cb = ModelCheckpoint(
        dirpath=str(args.ckpt_dir),
        filename="stage3-{step:06d}-{val/loss:.4f}",
        monitor="val/loss", mode="min", save_top_k=3, save_last=True,
        save_weights_only=True,
    )
    trainer = pl.Trainer(
        max_steps=args.max_steps,
        accelerator=args.accelerator, devices=args.devices, precision=args.precision,
        logger=logger, callbacks=[ckpt_cb, LearningRateMonitor("step")],
        val_check_interval=500, log_every_n_steps=50,
    )
    trainer.fit(module, dm)


if __name__ == "__main__":
    main()
