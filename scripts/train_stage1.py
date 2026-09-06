"""
Stage 1 training entry point.

Usage:
    python scripts/train_stage1.py \
        --splits_dir /data5/iome_cache/splits \
        --cache_sd   /data5/iome_cache/superdarn \
        --cache_smag /data5/iome_cache/supermag \
        --cache_tec  /data5/iome_cache/tec \
        --cache_dmsp /data5/iome_cache/dmsp \
        --stats_dir  /data5/iome_cache/stats \
        --omni_dir   /data5/iome_cache/omni \
        --ckpt_dir   /data5/iome_cache/ckpts/stage1 \
        --wandb_project iome
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from iome.models.fusion import UnifiedIonosphereModel
from iome.train.stage1  import Stage1ContrastiveModule
from iome.data.datamodule import TriModalDataModule


def parse_args():
    ap = argparse.ArgumentParser()
    # Timestamp splits (produced by make_timestamps.py)
    ap.add_argument("--splits_dir",  type=Path, default=None,
                    help="Directory with ts_train.json / ts_val.json / ts_avail.json")
    ap.add_argument("--ts_train",    type=Path, default=None)
    ap.add_argument("--ts_val",      type=Path, default=None)
    ap.add_argument("--ts_test",     type=Path, default=None)
    ap.add_argument("--ts_avail",    type=Path, default=None,
                    help="ts_avail.json mapping {ts: [mods]} from make_timestamps.py")
    # Cache dirs
    ap.add_argument("--cache_sd",    type=Path, default=None)
    ap.add_argument("--cache_smag",  type=Path, default=None)
    ap.add_argument("--cache_tec",   type=Path, default=None)
    ap.add_argument("--cache_dmsp",  type=Path, default=None)
    ap.add_argument("--stats_dir",   type=Path, default=None,
                    help="Directory with stats_{mod}.npy files")
    ap.add_argument("--omni_dir",    type=Path, default=None)
    ap.add_argument("--ckpt_dir",        type=Path, required=True)
    ap.add_argument("--ckpt_stage0_dir", type=Path, default=None,
                    help="Directory of stage0_{mod}_encoder.pt files from train_stage0.py")
    # Training
    ap.add_argument("--delta_t_steps", type=int, default=8,
                    help="Steps between t and t+delta for temporal CLIP (default 8 = 16 min)")
    ap.add_argument("--lag_matrix", type=Path, default=None,
                    help="lag_matrix.json from analyse_lag.py — enables fixed-lag cross-modal CLIP")
    ap.add_argument("--align_window_steps", type=int, default=0,
                    help="Window size K for soft-attention dynamic lag (0=disabled, e.g. 8=16 min). "
                         "Enables TemporalAlignmentHead; requires more memory than fixed lag.")
    ap.add_argument("--wandb_project", default="iome")
    ap.add_argument("--wandb_entity",  default="st7ma784")
    ap.add_argument("--wandb_name",    default="stage1")
    ap.add_argument("--accelerator",   default="gpu", choices=["gpu","cpu"])
    ap.add_argument("--devices",       type=int, default=1)
    ap.add_argument("--precision",     default="bf16-mixed")
    ap.add_argument("--batch_size",  type=int,   default=32)
    ap.add_argument("--max_steps",   type=int,   default=50_000)
    ap.add_argument("--num_workers", type=int,   default=16)
    ap.add_argument("--latent_dim",  type=int,   default=256)
    ap.add_argument("--tau",         type=float, default=0.1)
    ap.add_argument("--lambda_recon",type=float, default=0.5)
    ap.add_argument("--lr",          type=float, default=3e-4)
    ap.add_argument("--p_mod_drop",  type=float, default=0.3,
                    help="Per-modality dropout probability during training")
    return ap.parse_args()


def _load_json(path: Path):
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

    # Resolve timestamp paths (splits_dir shortcut)
    sd = args.splits_dir
    ts_train = _load_json(args.ts_train or (sd / "ts_train.json" if sd else None))
    ts_val   = _load_json(args.ts_val   or (sd / "ts_val.json"   if sd else None))
    ts_test  = _load_json(args.ts_test  or (sd / "ts_test.json"  if sd else None))
    avail_map= _load_json(args.ts_avail or (sd / "ts_avail.json" if sd else None))

    if ts_train is None:
        raise ValueError("Provide --splits_dir or --ts_train")

    lag_matrix = None
    if args.lag_matrix and Path(args.lag_matrix).exists():
        lag_json   = json.loads(Path(args.lag_matrix).read_text())
        lag_matrix = lag_json.get("lag_matrix")   # compact {A->B: steps} dict
        print(f"Loaded lag matrix: {lag_matrix}")

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
        delta_t_steps=args.delta_t_steps,
        lag_matrix=lag_matrix,
        align_window_steps=args.align_window_steps,
    )

    model  = UnifiedIonosphereModel(latent_dim=args.latent_dim)
    module = Stage1ContrastiveModule(
        model=model,
        ckpt_stage0_dir=args.ckpt_stage0_dir,
        lr=args.lr,
        tau=args.tau,
        lambda_recon=args.lambda_recon,
        max_steps=args.max_steps,
        p_mod_drop=args.p_mod_drop,
        align_window_steps=args.align_window_steps,
    )

    logger  = WandbLogger(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name)
    ckpt_cb = ModelCheckpoint(
        dirpath=str(args.ckpt_dir),
        filename="stage1-{step:06d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        save_weights_only=True,
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
    trainer.fit(module, dm)


if __name__ == "__main__":
    main()
