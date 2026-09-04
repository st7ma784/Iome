"""
Build train/val/test timestamp lists from the UNION of available modalities.

Timestamp inclusion rule: at least `--min_modalities` (default 2) of the four
sensors must have data for a given 2-min epoch.  This replaces the old strict
intersection so training works even when some sensors have gaps.

Modality presence:
  SuperDARN: one daily file YYYYMMDD_sd.npy — covers all 720 epochs of that day
  SuperMAG:  per-epoch YYYYMMDDTHHMM_smag.npy
  GPS-TEC:   per-epoch YYYYMMDDTHHMM_tec.npy
  DMSP:      per-epoch YYYYMMDDTHHMM_dmsp.npy (2000+ only)

Output:
    ts_train.json  ]
    ts_val.json    }  sorted lists of "YYYYMMDDTHHMM" strings
    ts_test.json   ]
    ts_avail.json      {ts: [list of available modality names]}

Default split: 80/10/10 by shuffled day (seed=42).

Usage:
    python scripts/make_timestamps.py \
        --cache_root /data5/iome_cache \
        --out_dir    /data5/iome_cache/splits \
        [--min_modalities 2]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

EPOCH_MIN = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root",      type=Path, default=Path("/data5/iome_cache"))
    ap.add_argument("--out_dir",         type=Path, default=None)
    ap.add_argument("--train_frac",      type=float, default=0.80)
    ap.add_argument("--val_frac",        type=float, default=0.10)
    ap.add_argument("--seed",            type=int,   default=42)
    ap.add_argument("--min_modalities",  type=int,   default=2,
                    help="Minimum number of modalities present to include epoch")
    args = ap.parse_args()

    out_dir = args.out_dir or args.cache_root / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    sd_dir   = args.cache_root / "superdarn"
    smag_dir = args.cache_root / "supermag"
    tec_dir  = args.cache_root / "tec"
    dmsp_dir = args.cache_root / "dmsp"

    # ------------------------------------------------------------------
    # 1. Collect days with any data
    # ------------------------------------------------------------------
    sd_days = set()
    for p in sd_dir.glob("*_sd.npy"):
        sd_days.add(p.stem[:8])

    # All candidate days: union of SD days + days represented in smag/tec
    smag_days = set(p.stem[:8] for p in smag_dir.glob("*_smag.npy"))
    tec_days  = set(p.stem[:8] for p in tec_dir.glob("*_tec.npy"))
    dmsp_days = set(p.stem[:8] for p in dmsp_dir.glob("*_dmsp.npy")) if dmsp_dir.exists() else set()

    all_days = sorted(sd_days | smag_days | tec_days | dmsp_days)
    print(f"Days with any data: {len(all_days)}")
    print(f"  SD: {len(sd_days)}  SMAG: {len(smag_days)}  TEC: {len(tec_days)}  DMSP: {len(dmsp_days)}")

    # ------------------------------------------------------------------
    # 2. For each day, scan 2-min epochs for union presence
    # ------------------------------------------------------------------
    all_ts: list[str] = []
    avail_map: dict[str, list[str]] = {}

    for day in all_days:
        for minute in range(0, 1440, EPOCH_MIN):
            key = f"{day}T{minute // 60:02d}{minute % 60:02d}"
            present = []
            if day in sd_days:
                present.append("sd")
            if (smag_dir / f"{key}_smag.npy").exists():
                present.append("smag")
            if (tec_dir / f"{key}_tec.npy").exists():
                present.append("tec")
            if dmsp_dir.exists() and (dmsp_dir / f"{key}_dmsp.npy").exists():
                present.append("dmsp")

            if len(present) >= args.min_modalities:
                all_ts.append(key)
                avail_map[key] = present

    print(f"Epochs with ≥{args.min_modalities} modalities: {len(all_ts)}")

    if not all_ts:
        print("No epochs found — check cache directories.")
        return

    # ------------------------------------------------------------------
    # 3. Split by day (shuffled, contiguous blocks per day preserved)
    # ------------------------------------------------------------------
    day_ts: dict[str, list[str]] = defaultdict(list)
    for ts in all_ts:
        day_ts[ts[:8]].append(ts)

    days_with_data = sorted(day_ts.keys())
    n_days = len(days_with_data)

    import random
    rng = random.Random(args.seed)
    day_list = days_with_data[:]
    rng.shuffle(day_list)

    n_train = int(n_days * args.train_frac)
    n_val   = int(n_days * args.val_frac)

    train_days = set(day_list[:n_train])
    val_days   = set(day_list[n_train:n_train + n_val])
    test_days  = set(day_list[n_train + n_val:])

    ts_train = sorted([ts for ts in all_ts if ts[:8] in train_days])
    ts_val   = sorted([ts for ts in all_ts if ts[:8] in val_days])
    ts_test  = sorted([ts for ts in all_ts if ts[:8] in test_days])

    print(f"  Train: {len(ts_train)} epochs from {n_train} days")
    print(f"  Val:   {len(ts_val)} epochs from {n_val} days")
    print(f"  Test:  {len(ts_test)} epochs from {len(test_days)} days")

    # Modality coverage stats
    from collections import Counter
    combo_counts = Counter(tuple(sorted(v)) for v in avail_map.values())
    print("\nModality combination frequencies:")
    for combo, cnt in combo_counts.most_common(10):
        print(f"  {'+'.join(combo)}: {cnt}")

    # ------------------------------------------------------------------
    # 4. Write JSON outputs
    # ------------------------------------------------------------------
    for fname, data in [("ts_train.json", ts_train),
                         ("ts_val.json",   ts_val),
                         ("ts_test.json",  ts_test)]:
        p = out_dir / fname
        with open(p, "w") as f:
            json.dump(data, f)
        print(f"  Wrote {p}  ({len(data)} entries)")

    # Availability map — {ts: [mod, ...]} — used by datamodule to skip loading
    avail_path = out_dir / "ts_avail.json"
    with open(avail_path, "w") as f:
        json.dump(avail_map, f)
    print(f"  Wrote {avail_path}  ({len(avail_map)} entries)")


if __name__ == "__main__":
    main()
