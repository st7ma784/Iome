"""
TemporalPairDataset — wraps a single-modality dataset to yield (anchor, positive)
pairs for per-modality contrastive pretraining (Phase 0).

Positive: a snapshot sampled uniformly within [1, window_steps] of the anchor.
Near timesteps represent similar ionospheric states; the batch provides negatives.
"""

from torch.utils.data import Dataset
import torch


class TemporalPairDataset(Dataset):
    """
    Args:
        base_dataset:  any single-modality dataset with .timestamps and ._load()
        mod_key:       modality name ("sd", "smag", "tec", "dmsp")
        window_steps:  max positive offset in 2-min steps (default 15 = ±30 min)
    """

    def __init__(self, base_dataset, mod_key: str, window_steps: int = 15):
        self.base        = base_dataset
        self.mod_key     = mod_key
        self.window      = window_steps
        # valid anchor indices: need window steps of headroom at the end
        self._valid = list(range(len(base_dataset.timestamps) - window_steps))

    def __len__(self):
        return len(self._valid)

    def __getitem__(self, idx):
        i      = self._valid[idx]
        offset = torch.randint(1, self.window + 1, (1,)).item()
        anchor   = self.base._load(self.base.timestamps[i])
        positive = self.base._load(self.base.timestamps[i + offset])
        return {"anchor": anchor, "positive": positive, "ts": self.base.timestamps[i]}
