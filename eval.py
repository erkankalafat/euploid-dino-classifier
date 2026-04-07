"""Evaluate a fold checkpoint and plot attention pooling weights."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from data.sequence_dataset import EmbryoSequenceDataset, collate
from models.temporal_attention import TemporalAttentionClassifier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--plot-dir", type=Path, default=Path("logs/attention"))
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.manifest or cfg["data"]["manifest"])
    ds = EmbryoSequenceDataset(df, cfg["data"]["features_dir"])
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], collate_fn=collate)

    model = TemporalAttentionClassifier(
        feature_dim=cfg["data"]["feature_dim"],
        d_model=cfg["model"]["d_model"], n_heads=cfg["model"]["n_heads"],
        n_layers=cfg["model"]["n_layers"], mlp_dim=cfg["model"]["mlp_dim"],
        dropout=cfg["model"]["dropout"], num_classes=cfg["model"]["num_classes"],
        max_len=cfg["data"]["num_frames"],
    ).to(device)
    blob = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(blob["model"])
    model.eval()

    args.plot_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    all_y, all_p = [], []
    with torch.no_grad():
        for feats, mask, labels, ids in loader:
            feats, mask = feats.to(device), mask.to(device)
            logits, attn = model(feats, mask)
            prob = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_y.extend(labels.numpy().tolist())
            all_p.extend(prob.tolist())
            for i, eid in enumerate(ids):
                w = attn[i].cpu().numpy()
                plt.figure(figsize=(6, 2))
                plt.bar(range(len(w)), w)
                plt.title(f"{eid} label={labels[i].item()} p_eup={prob[i]:.2f}")
                plt.xlabel("frame index")
                plt.tight_layout()
                plt.savefig(args.plot_dir / f"{eid}.png", dpi=80)
                plt.close()

    from sklearn.metrics import balanced_accuracy_score, average_precision_score, roc_auc_score
    y, p = np.array(all_y), np.array(all_p)
    print({
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "bal_acc": balanced_accuracy_score(y, (p >= 0.5).astype(int)),
    })


if __name__ == "__main__":
    main()
