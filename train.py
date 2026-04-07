"""5-fold stratified CV training with resumable Drive checkpoints."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import balanced_accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from data.sequence_dataset import EmbryoSequenceDataset, collate
from models.temporal_attention import TemporalAttentionClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weights(labels: np.ndarray) -> torch.Tensor | None:
    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    ratio = n_pos / max(len(labels), 1)
    if 0.4 <= ratio <= 0.6:
        return None
    total = n_pos + n_neg
    w = torch.tensor([total / (2 * n_neg), total / (2 * n_pos)], dtype=torch.float)
    return w


def metrics(y_true, y_prob) -> dict:
    y_pred = (np.asarray(y_prob) >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }


def run_fold(fold: int, train_idx, val_idx, df, cfg, device, ckpt_path: Path):
    train_ds = EmbryoSequenceDataset(df.iloc[train_idx], cfg["data"]["features_dir"],
                                     frame_dropout=cfg["train"]["frame_dropout"], training=True)
    val_ds = EmbryoSequenceDataset(df.iloc[val_idx], cfg["data"]["features_dir"], training=False)
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                              collate_fn=collate, num_workers=2, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                            collate_fn=collate, num_workers=2)

    model = TemporalAttentionClassifier(
        feature_dim=cfg["data"]["feature_dim"],
        d_model=cfg["model"]["d_model"], n_heads=cfg["model"]["n_heads"],
        n_layers=cfg["model"]["n_layers"], mlp_dim=cfg["model"]["mlp_dim"],
        dropout=cfg["model"]["dropout"], num_classes=cfg["model"]["num_classes"],
        max_len=cfg["data"]["num_frames"],
    ).to(device)

    weights = class_weights(df.iloc[train_idx]["label"].to_numpy())
    if weights is not None:
        weights = weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg["train"]["label_smoothing"])

    optim = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                              weight_decay=cfg["train"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg["train"]["epochs"])

    start_epoch = 0
    best = {"roc_auc": -1.0}
    if ckpt_path.exists():
        blob = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(blob["model"])
        optim.load_state_dict(blob["optim"])
        sched.load_state_dict(blob["sched"])
        start_epoch = blob["epoch"] + 1
        best = blob.get("best", best)
        print(f"[fold {fold}] resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        model.train()
        for feats, mask, labels, _ in train_loader:
            feats, mask, labels = feats.to(device), mask.to(device), labels.to(device)
            logits, _ = model(feats, mask)
            loss = criterion(logits, labels)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
        sched.step()

        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for feats, mask, labels, _ in val_loader:
                feats, mask = feats.to(device), mask.to(device)
                logits, _ = model(feats, mask)
                prob = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                ys.extend(labels.numpy().tolist())
                ps.extend(prob.tolist())
        m = metrics(np.array(ys), np.array(ps))
        print(f"[fold {fold}] epoch {epoch}: {m}")
        if m["roc_auc"] > best["roc_auc"]:
            best = {**m, "epoch": epoch}

        torch.save({
            "fold": fold, "epoch": epoch, "model": model.state_dict(),
            "optim": optim.state_dict(), "sched": sched.state_dict(), "best": best,
        }, ckpt_path)

    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--checkpoint-dir", type=Path, default=None,
                    help="Where to write resumable checkpoints (point at Drive in Colab)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    set_seed(cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(cfg["data"]["manifest"])
    ckpt_dir = args.checkpoint_dir or Path(cfg["train"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(n_splits=cfg["train"]["folds"], shuffle=True, random_state=cfg["train"]["seed"])
    fold_results = []
    for fold, (tr, va) in enumerate(skf.split(df, df["label"])):
        ckpt = ckpt_dir / f"fold{fold}.pt"
        best = run_fold(fold, tr, va, df, cfg, device, ckpt)
        fold_results.append(best)
        (ckpt_dir / f"fold{fold}_best.json").write_text(json.dumps(best, indent=2))

    summary = {
        k: {"mean": float(np.mean([r[k] for r in fold_results])),
            "std": float(np.std([r[k] for r in fold_results]))}
        for k in ("roc_auc", "pr_auc", "bal_acc")
    }
    (ckpt_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2))
    print("CV summary:", summary)


if __name__ == "__main__":
    main()
