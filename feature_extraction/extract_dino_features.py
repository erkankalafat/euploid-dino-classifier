"""Extract frozen DINO features for each embryo video.

For every row in manifest.csv:
  1. Decode the video with torchvision.io.read_video.
  2. Sample frames at fixed stride (default 10), N samples (default 40).
     Pad with black frames if the clip is shorter than stride*N.
  3. Apply the same eval transform as dino_training/EmbryoFeatureDataset
     (Resize -> CenterCrop(crop_size) -> ImageNet normalize).
  4. Forward through the frozen DINO **teacher** backbone.
  5. Save {features, mask, label} to features/{embryo_id}.pt.

Already-cached embryos are skipped, so this script is resumable.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.io import read_video
from tqdm import tqdm
import pandas as pd


# ---------------------------------------------------------------------------
# DINO loader — builds a bare ViT-S/16 backbone and loads teacher weights from
# the dino_training checkpoint (which stores a MultiCropWrapper, so backbone
# weights are prefixed "backbone." and head weights live under "head.").
# ---------------------------------------------------------------------------

def load_dino(dino_ckpt: Path, device: torch.device, img_size: int = 384):
    from dino_training.model import VisionTransformer  # type: ignore

    model = VisionTransformer(
        img_size=img_size,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
    )

    ckpt = torch.load(dino_ckpt, map_location="cpu", weights_only=False)
    # Prefer teacher (smoother features in DINO); fall back to student.
    raw = ckpt.get("teacher") or ckpt.get("student") or ckpt.get("state_dict") or ckpt

    # Strip MultiCropWrapper / DDP prefixes; keep only backbone.* keys.
    state = {}
    for k, v in raw.items():
        nk = k.replace("module.", "")
        if nk.startswith("backbone."):
            state[nk[len("backbone."):]] = v
        # head.* keys are dropped — we only need the encoder

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[dino] loaded {dino_ckpt.name} | "
          f"matched={len(state) - len(unexpected)} "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"[dino] sample missing: {missing[:3]}")

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def build_eval_transform(crop_size: int = 384) -> transforms.Compose:
    """Mirrors dino_training/dataset.py::EmbryoFeatureDataset transform."""
    return transforms.Compose([
        transforms.Resize(int(crop_size * 256 / 224),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def sample_frame_indices(total: int, num_frames: int, stride: int) -> tuple[list[int], int]:
    """Fixed-stride sampling. Indices past EOS are clamped; caller treats those
    positions as padding via the returned `real` count."""
    idxs = [i * stride for i in range(num_frames)]
    real = sum(1 for i in idxs if i < total)
    idxs = [min(i, max(total - 1, 0)) for i in idxs]
    return idxs, real


def extract_one(video_path: Path, num_frames: int, stride: int,
                transform, model, device, crop_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    video, _, _ = read_video(str(video_path), pts_unit="sec", output_format="THWC")
    total = video.shape[0]
    if total == 0:
        raise RuntimeError(f"empty video: {video_path}")

    idxs, real = sample_frame_indices(total, num_frames, stride)
    H, W = int(video.shape[1]), int(video.shape[2])

    tensors = []
    for i, idx in enumerate(idxs):
        if i < real:
            img = Image.fromarray(video[idx].numpy())
        else:
            img = Image.new("RGB", (W, H), (0, 0, 0))
        tensors.append(transform(img))
    batch = torch.stack(tensors, dim=0).to(device, non_blocking=True)  # (N, 3, S, S)

    with torch.no_grad():
        feats = model(batch)  # (N, 384) — CLS token (return_all_tokens=False)
    mask = torch.zeros(num_frames, dtype=torch.bool)
    mask[:real] = True
    return feats.cpu(), mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--features-dir", type=Path, default=Path("features"))
    ap.add_argument("--dino-ckpt", type=Path, required=True)
    ap.add_argument("--dino-training-root", type=Path, default=None,
                    help="Path whose child is the dino_training/ package; prepended to sys.path")
    ap.add_argument("--num-frames", type=int, default=40)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--crop-size", type=int, default=384,
                    help="Must match DINOConfig.global_crop_size used during pretraining")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.dino_training_root:
        sys.path.insert(0, str(args.dino_training_root.resolve()))

    args.features_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = load_dino(args.dino_ckpt, device, img_size=args.crop_size)
    transform = build_eval_transform(crop_size=args.crop_size)

    df = pd.read_csv(args.manifest)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="extract"):
        out = args.features_dir / f"{row['embryo_id']}.pt"
        if out.exists():
            continue
        try:
            feats, mask = extract_one(
                Path(row["video_path"]), args.num_frames, args.stride,
                transform, model, device, args.crop_size,
            )
        except Exception as e:
            print(f"[err] {row['embryo_id']}: {e}")
            continue
        torch.save({"features": feats, "mask": mask, "label": int(row["label"])}, out)


if __name__ == "__main__":
    main()
