"""Dump the 40 sampled frames for a few embryos as PNGs for sanity-checking."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image
from torchvision.io import read_video

from feature_extraction.extract_dino_features import sample_frame_indices


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("logs/preview"))
    ap.add_argument("--num-frames", type=int, default=40)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--n-embryos", type=int, default=3)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest).head(args.n_embryos)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        video, _, _ = read_video(row["video_path"], pts_unit="sec", output_format="THWC")
        total = video.shape[0]
        idxs, real = sample_frame_indices(total, args.num_frames, args.stride)
        print(f"{row['embryo_id']}: total={total} real={real} pad={args.num_frames - real}")
        sub = args.out_dir / row["embryo_id"]
        sub.mkdir(parents=True, exist_ok=True)
        for i, idx in enumerate(idxs):
            if i < real:
                Image.fromarray(video[idx].numpy()).save(sub / f"{i:02d}.png")
            else:
                Image.new("RGB", (video.shape[2], video.shape[1]), (0, 0, 0)).save(sub / f"{i:02d}_pad.png")


if __name__ == "__main__":
    main()
