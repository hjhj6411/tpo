#!/usr/bin/env python3
"""Extract Qwen VL image embeddings from existing img2dataset tar shards.

Reads the old CLIP corpus tar files as immutable image bytes and writes a separate
QwenEmb corpus under QwenEmb/corpus by default.
"""

from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from qwenemb_encoder import QwenEmbConfig, QwenEmbEncoder


def iter_images_from_tar(tar_path: Path):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    with tarfile.open(tar_path, "r") as tf:
        for member in tf:
            if not member.isfile():
                continue
            suffix = Path(member.name).suffix.lower()
            if suffix not in exts:
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            try:
                img = Image.open(io.BytesIO(f.read())).convert("RGB")
            except Exception:
                continue
            key = Path(member.name).stem
            yield key, img


def limit_iterator(iterator, limit: int | None):
    if limit is None:
        yield from iterator
        return
    for i, item in enumerate(iterator):
        if i >= limit:
            break
        yield item


def batched(iterator, batch_size: int):
    keys, images = [], []
    for key, img in iterator:
        keys.append(key)
        images.append(img)
        if len(images) >= batch_size:
            yield keys, images
            keys, images = [], []
    if images:
        yield keys, images


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clip-corpus", default="/home1/hjhj6411/pod_bench/data/clip_corpus")
    p.add_argument("--out", default="/home1/hjhj6411/pod_bench/QwenEmb/corpus")
    p.add_argument("--model-id", default="Qwen/Qwen3-VL-Embedding-8B")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--device-map", default=None, help="Optional HF device_map, kept for CLI compatibility.")
    p.add_argument("--dim", type=int, default=None, help="Optional Matryoshka truncation dimension.")
    p.add_argument("--max-image-size", type=int, default=448,
                   help="Resize image long side before Qwen encoding; use 384 if OOM persists.")
    p.add_argument("--limit-shards", type=int, default=None)
    p.add_argument("--limit-images", type=int, default=None, help="Debug cap per process, after shard split.")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    clip_corpus = Path(args.clip_corpus)
    out = Path(args.out)
    shard_out = out / "shards"
    shard_out.mkdir(parents=True, exist_ok=True)

    tar_paths = sorted((clip_corpus / "images").glob("*.tar"))
    if args.limit_shards is not None:
        tar_paths = tar_paths[: args.limit_shards]
    if not tar_paths:
        raise SystemExit(f"No tar shards found under {clip_corpus / 'images'}")

    assigned = [p for i, p in enumerate(tar_paths) if i % args.num_gpus == args.gpu]
    print(f"[qwenemb] assigned shards={len(assigned)} / total={len(tar_paths)} gpu={args.gpu}/{args.num_gpus}")

    cfg = QwenEmbConfig(
        model_id=args.model_id,
        device=args.device,
        device_map=args.device_map,
        dtype=args.dtype,
        dim=args.dim,
        max_image_size=args.max_image_size,
    )
    encoder = QwenEmbEncoder(cfg)

    n_total = 0
    for tar_path in assigned:
        stem = tar_path.stem
        npy_path = shard_out / f"{stem}.npy"
        pq_path = shard_out / f"{stem}.parquet"
        if npy_path.exists() and pq_path.exists() and not args.force:
            print(f"[skip] {stem}")
            continue

        all_keys = []
        all_embs = []
        n_shard = 0
        iterator = limit_iterator(iter_images_from_tar(tar_path), args.limit_images)

        for keys, images in tqdm(batched(iterator, args.batch), desc=stem):
            embs = encoder.encode_images(images)
            if embs.shape[0] != len(keys):
                raise RuntimeError(f"Embedding batch mismatch: got {embs.shape[0]} for {len(keys)} images")
            all_keys.extend(keys)
            all_embs.append(embs)
            n_shard += len(keys)

        if not all_embs:
            print(f"[warn] empty shard: {tar_path}")
            continue
        mat = np.concatenate(all_embs, axis=0).astype("float32", copy=False)
        np.save(npy_path, mat)
        pd.DataFrame({"key": all_keys, "shard": stem}).to_parquet(pq_path, index=False)
        n_total += n_shard
        print(f"[done] {stem}: {n_shard} vectors dim={mat.shape[1]}")

    print(f"[qwenemb] process complete vectors_written={n_total}")


if __name__ == "__main__":
    main()
