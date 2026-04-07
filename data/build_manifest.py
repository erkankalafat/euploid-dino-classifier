"""Walk a local videos root and emit manifest.csv.

Each embryo lives in its own folder named like ``30998132_1781_1_E`` where the
trailing token (``E`` or ``A``) is the ploidy label. Each folder is expected to
contain exactly one video file.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
LABEL_MAP = {"E": 1, "A": 0}  # 1 = euploid, 0 = aneuploid


def parse_label(folder_name: str) -> int | None:
    suffix = folder_name.rsplit("_", 1)[-1].strip().upper()
    return LABEL_MAP.get(suffix)


def find_video(folder: Path) -> Path | None:
    vids = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    if not vids:
        return None
    if len(vids) > 1:
        print(f"[warn] {folder.name}: {len(vids)} videos, picking {vids[0].name}")
    return vids[0]


def build(videos_root: Path, out_csv: Path) -> None:
    rows = []
    skipped = 0
    for folder in sorted(p for p in videos_root.iterdir() if p.is_dir()):
        label = parse_label(folder.name)
        if label is None:
            print(f"[skip] {folder.name}: cannot parse E/A label")
            skipped += 1
            continue
        video = find_video(folder)
        if video is None:
            print(f"[skip] {folder.name}: no video file found")
            skipped += 1
            continue
        rows.append({
            "embryo_id": folder.name,
            "label": label,
            "video_path": str(video.resolve()),
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["embryo_id", "label", "video_path"])
        writer.writeheader()
        writer.writerows(rows)

    n_e = sum(1 for r in rows if r["label"] == 1)
    n_a = len(rows) - n_e
    print(f"Wrote {out_csv} | total={len(rows)} euploid={n_e} aneuploid={n_a} skipped={skipped}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-root", type=Path, required=True,
                    help="Local directory containing one folder per embryo")
    ap.add_argument("--out", type=Path, default=Path("data/manifest.csv"))
    args = ap.parse_args()
    build(args.videos_root, args.out)


if __name__ == "__main__":
    main()
