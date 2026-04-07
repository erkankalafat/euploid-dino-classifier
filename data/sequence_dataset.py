"""Dataset over cached per-embryo feature tensors."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class EmbryoSequenceDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, features_dir: Path,
                 frame_dropout: float = 0.0, training: bool = False):
        self.df = manifest.reset_index(drop=True)
        self.features_dir = Path(features_dir)
        self.frame_dropout = frame_dropout
        self.training = training

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        blob = torch.load(self.features_dir / f"{row['embryo_id']}.pt", map_location="cpu")
        feats: torch.Tensor = blob["features"]            # (N, 384)
        mask: torch.Tensor = blob["mask"].clone()          # (N,) True=real
        label = int(row["label"])

        if self.training and self.frame_dropout > 0:
            drop = (torch.rand(mask.shape) < self.frame_dropout) & mask
            # ensure at least one real frame remains
            if (mask & ~drop).any():
                mask = mask & ~drop

        return feats, mask, label, row["embryo_id"]


def collate(batch):
    feats = torch.stack([b[0] for b in batch], dim=0)
    masks = torch.stack([b[1] for b in batch], dim=0)
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    ids = [b[3] for b in batch]
    return feats, masks, labels, ids
