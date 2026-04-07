"""Extract frozen DINO features for each embryo video.

For every row in manifest.csv:
  1. Decode the video with torchvision.io.read_video.
  2. Sample frames at fixed stride (default 10), taking N=40 samples.
     If the video ends early, pad with black frames (and record a mask).
  3. Apply the DINO eval transform (reused from dino_training/).
  4. Forward through the frozen DINO student backbone.
  5. Save {features, mask} to features/{embryo_id}.pt.

Already-cached embryos are skipped, so this script is resumable.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.io import read_video
from tqdm import tqdm
import pandas as pd

# dino_training/ is expected to live next to this repo on Colab; the user will
# upload it there. Add a --dino-training-root flag so we can prepend it to sys.path.


def load_dino(dino_ckpt: Path, device: torch.device):
    from dino_training.models import build_dino  # type: ignore
    model = build_dino()
    ckpt = torch.load(dino_ckpt, map_location="cpu")
    state = ckpt.get("student", ckpt.get("state_dict", ckpt))
    # strip common prefixes
    state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[dino] loaded {dino_ckpt} | missing={len(missing)} unexpected={len(unexpected)}")
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_transform():
    from dino_training.dataset import build_eval_transform  # type: ignore
    return build_eval_transform()


def sample_frame_indices(total: int, num_frames: int, stride: int) -> tuple[list[int], int]:
    """Return list of frame indices of length `num_frames` using fixed stride.

    Indices beyond the clip length are clamped; caller treats them as padding
    and the second return value is the count of *real* (non-pad) frames.
    """
    idxs = [i * stride for i in range(num_frames)]
    real = sum(1 for i in idxs if i < total)
    idxs = [min(i, max(total - 1, 0)) for i in idxs]
    return idxs, real


def extract_one(video_path: Path, num_frames: int, stride: int,
                transform, model, device) -> tuple[torch.Tensor, torch.Tensor]:
    # read_video returns (T, H, W, C) uint8
    video, _, _ = read_video(str(video_path), pts_unit="sec", output_format="THWC")
    total = video.shape[0]
    if total == 0:
        raise RuntimeError(f"empty video: {video_path}")

    idxs, real = sample_frame_indices(total, num_frames, stride)
    frames = video[idxs]  # (N, H, W, C)
    # Convert to CHW float for transform; transform expects PIL or tensor per dino_training
    from PIL import Image
    tensors = []
    for i, frame in enumerate(frames):
        if i < real:
            img = Image.fromarray(frame.numpy())
        else:
            img = Image.new("RGB", (frame.shape[1], frame.shape[0]), (0, 0, 0))
        tensors.append(transform(img))
    batch = torch.stack(tensors, dim=0).to(device, non_blocking=True)  # (N, 3, 224, 224)

    with torch.no_grad():
        feats = model(batch)  # (N, 384)
    mask = torch.zeros(num_frames, dtype=torch.bool)
    mask[:real] = True  # True = real, False = padded
    return feats.cpu(), mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--features-dir", type=Path, default=Path("features"))
    ap.add_argument("--dino-ckpt", type=Path, required=True)
    ap.add_argument("--dino-training-root", type=Path, default=None,
                    help="Path to dino_training/ parent; prepended to sys.path")
    ap.add_argument("--num-frames", type=int, default=40)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.dino_training_root:
        sys.path.insert(0, str(args.dino_training_root.resolve()))

    args.features_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = load_dino(args.dino_ckpt, device)
    transform = load_transform()

    df = pd.read_csv(args.manifest)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="extract"):
        out = args.features_dir / f"{row['embryo_id']}.pt"
        if out.exists():
            continue
        try:
            feats, mask = extract_one(
                Path(row["video_path"]), args.num_frames, args.stride,
                transform, model, device,
            )
        except Exception as e:
            print(f"[err] {row['embryo_id']}: {e}")
            continue
        torch.save({"features": feats, "mask": mask, "label": int(row["label"])}, out)


if __name__ == "__main__":
    main()
