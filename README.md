# euploid-dino-classifier

Downstream ploidy classifier built on frozen DINO ViT-S/16 features from
[`timelapse-dino`](https://github.com/erkankalafat/timelapse-dino).

## Pipeline

1. **Manifest** — `python -m data.build_manifest --videos-root /path/to/videos`
   Walks one folder per embryo (`<id>_..._E` or `_A`), one video per folder.
2. **Preview** (sanity) — `python -m scripts.preview_frames --n-embryos 3`
   Dumps the 40 sampled frames per embryo as PNGs to `logs/preview/`.
3. **Extract features** — resumable, skips cached embryos:
   ```
   python -m feature_extraction.extract_dino_features \
     --dino-ckpt /path/to/dino_student.pth \
     --dino-training-root /path/to/parent_of_dino_training
   ```
   Caches `features/{embryo_id}.pt` = `{features:(40,384), mask:(40,), label}`.
4. **Train** — 5-fold stratified CV with resumable checkpoints (point at Drive):
   ```
   python train.py --checkpoint-dir /content/drive/MyDrive/euploid_ckpts
   ```
5. **Eval / interpretability** —
   `python eval.py --checkpoint /content/drive/MyDrive/euploid_ckpts/fold0.pt`
   writes per-embryo attention plots to `logs/attention/`.

## Sampling

Fixed stride (default 10), N=40 samples per clip. Clips shorter than
`stride * N` are padded with black frames; padded positions are excluded from
Transformer attention and from attention pooling via key-padding masks.
Override with `--num-frames` / `--stride` on extraction.

## Architecture

Frozen DINO ViT-S/16 → (40, 384) per clip → Linear 384→192 + learned positional
embedding → 2-layer Transformer (4 heads, dim 192, MLP 384, dropout 0.3) →
single-query attention pool → MLP 192→64→2. ~150k trainable params.
