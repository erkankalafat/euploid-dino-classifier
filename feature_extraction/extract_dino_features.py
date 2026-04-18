"""Extract frozen DINO features for each embryo video.

For every row in manifest.csv:
  1. Decode the video with torchvision.io.read_video.
  2. Sample frames at fixed stride (default 10), N samples (default 40).
     Pad with black frames if the clip is shorter than stride*N.
  3. Apply the same eval transform as dino_training/EmbryoFeatureDataset
     (Resize -> CenterCrop(crop_size) -> ImageNet normalize).
  4. Forward through the frozen DINO **teacher** backbone and collect features
     according to --feature-source:
       - cls          : layer-11 CLS token                              (384-d)
       - patch_mean   : layer-11 patch tokens, mean-pooled              (384-d)
       - multi_layer  : patch tokens at --layers, mean-pooled, concat   (384*L-d)
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
    raw = ckpt.get("teacher") or ckpt.get("student") or ckpt.get("state_dict") or ckpt

    state = {}
    for k, v in raw.items():
        nk = k.replace("module.", "")
        if nk.startswith("backbone."):
            state[nk[len("backbone."):]] = v

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[dino] loaded {dino_ckpt.name} | "
          f"matched={len(state) - len(unexpected)} "
          f"missing={len(missing)} unexpected={len(unexpected)}")

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def build_eval_transform(crop_size: int = 384) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(crop_size * 256 / 224),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


class FeatureExtractor:
    """Produce a per-frame feature tensor from a frozen ViT according to `source`.

    Layer indices are 0-based over `model.blocks` (depth=12, so 0..11). Patch
    pooling averages the token sequence after dropping the CLS token at idx 0.
    """

    def __init__(self, model, source: str = "multi_layer", layers=(5, 7, 9, 11)):
        self.model = model
        self.source = source
        self.layers = list(layers)
        self._captured: dict[int, torch.Tensor] = {}
        self._handles = []

        if source == "multi_layer":
            for li in self.layers:
                def _make_hook(idx):
                    def _hook(_module, _inp, out):
                        self._captured[idx] = out
                    return _hook
                self._handles.append(
                    model.blocks[li].register_forward_hook(_make_hook(li))
                )

    @property
    def feature_dim(self) -> int:
        if self.source == "multi_layer":
            return 384 * len(self.layers)
        return 384

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.source == "cls":
            return self.model(x)  # (B, 384)
        if self.source == "patch_mean":
            tokens = self.model(x, return_all_tokens=True)  # (B, 1+P, 384)
            return tokens[:, 1:].mean(dim=1)
        if self.source == "multi_layer":
            self._captured.clear()
            _ = self.model(x)  # triggers hooks
            pooled = [self._captured[li][:, 1:].mean(dim=1) for li in self.layers]
            return torch.cat(pooled, dim=-1)  # (B, 384*L)
        raise ValueError(f"unknown feature source: {self.source}")

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def sample_frame_indices(total: int, num_frames: int, stride: int) -> tuple[list[int], int]:
    idxs = [i * stride for i in range(num_frames)]
    real = sum(1 for i in idxs if i < total)
    idxs = [min(i, max(total - 1, 0)) for i in idxs]
    return idxs, real


def extract_one(video_path: Path, num_frames: int, stride: int,
                transform, extractor: FeatureExtractor, device,
                crop_size: int) -> tuple[torch.Tensor, torch.Tensor]:
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
        feats = extractor(batch)  # (N, feature_dim)
    mask = torch.zeros(num_frames, dtype=torch.bool)
    mask[:real] = True
    return feats.cpu(), mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--features-dir", type=Path, default=Path("features"))
    ap.add_argument("--dino-ckpt", type=Path, required=True)
    ap.add_argument("--dino-training-root", type=Path, default=None)
    ap.add_argument("--num-frames", type=int, default=40)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--crop-size", type=int, default=384)
    ap.add_argument("--feature-source", choices=["cls", "patch_mean", "multi_layer"],
                    default="multi_layer")
    ap.add_argument("--layers", type=str, default="5,7,9,11",
                    help="Comma-separated block indices for --feature-source multi_layer")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.dino_training_root:
        sys.path.insert(0, str(args.dino_training_root.resolve()))

    args.features_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = load_dino(args.dino_ckpt, device, img_size=args.crop_size)

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    extractor = FeatureExtractor(model, source=args.feature_source, layers=layers)
    print(f"[extract] source={args.feature_source} layers={layers} feature_dim={extractor.feature_dim}")
    transform = build_eval_transform(crop_size=args.crop_size)

    df = pd.read_csv(args.manifest)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="extract"):
        out = args.features_dir / f"{row['embryo_id']}.pt"
        if out.exists():
            continue
        try:
            feats, mask = extract_one(
                Path(row["video_path"]), args.num_frames, args.stride,
                transform, extractor, device, args.crop_size,
            )
        except Exception as e:
            print(f"[err] {row['embryo_id']}: {e}")
            continue
        torch.save({
            "features": feats,
            "mask": mask,
            "label": int(row["label"]),
            "source": args.feature_source,
            "layers": layers if args.feature_source == "multi_layer" else None,
        }, out)

    extractor.close()


if __name__ == "__main__":
    main()
