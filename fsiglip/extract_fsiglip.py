#!/usr/bin/env python3
"""
extract_fsiglip.py
------------------
Encode the EXISTING img2dataset webdataset tars with Marqo-FashionSigLIP and
write L2-normalized image embeddings to a NEW corpus folder. Reads clip_corpus
read-only; never modifies it.

Per-tar resumable: each shard -> <out>/shards/<tarstem>.npy + .parquet
Re-running skips shards already done. Multi-GPU by shard split.

Launch one process per GPU:
  for G in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$G python extract_fsiglip.py --gpu $G --num-gpus 4 &
  done; wait
"""
import argparse, io, glob, os, tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import open_clip
from PIL import Image

MODEL = "hf-hub:Marqo/marqo-fashionSigLIP"
DIM   = 768

CLIP_CORPUS = "/home1/hjhj6411/pod_bench/data/clip_corpus"
OUT_CORPUS  = "/home1/hjhj6411/pod_bench/data/fashionsiglip_corpus"


class TarDataset(Dataset):
    """Decode + preprocess one webdataset tar; runs in DataLoader workers so the
    GPU isn't starved by PIL resize. Tar opened lazily per worker process."""
    def __init__(self, tar_path, preprocess):
        self.tar_path = tar_path
        self.preprocess = preprocess
        self._tf = None
        with tarfile.open(tar_path, "r") as tf:
            names = tf.getnames()
        self.keys = sorted({n.rsplit(".", 1)[0]
                            for n in names if n.endswith(".jpg")})

    def _tar(self):
        if self._tf is None:
            self._tf = tarfile.open(self.tar_path, "r")
        return self._tf

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        k = self.keys[i]
        tf = self._tar()
        try:
            img = Image.open(io.BytesIO(tf.extractfile(k + ".jpg").read())).convert("RGB")
            x = self.preprocess(img)
        except Exception:
            return None
        cap = ""
        try:
            m = tf.extractfile(k + ".txt")
            if m is not None:
                cap = m.read().decode("utf-8", "ignore").strip()
        except Exception:
            cap = ""
        return x, k, cap


def _collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    xs   = torch.stack([b[0] for b in batch])
    keys = [b[1] for b in batch]
    caps = [b[2] for b in batch]
    return xs, keys, caps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--num-gpus", type=int, default=1)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8,
                    help="DataLoader workers for tar decode+preprocess")
    ap.add_argument("--clip-corpus", default=CLIP_CORPUS)
    ap.add_argument("--out", default=OUT_CORPUS)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL)
    model = model.to(device).eval()
    torch.backends.cuda.matmul.allow_tf32 = True

    shard_dir = Path(args.out) / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    tars = sorted(glob.glob(os.path.join(args.clip_corpus, "images", "*.tar")))
    mine = [t for i, t in enumerate(tars) if i % args.num_gpus == args.gpu]
    print(f"[gpu{args.gpu}] {len(mine)}/{len(tars)} tars, batch={args.batch}, "
          f"workers={args.workers}, device={device}")

    for tar_path in mine:
        stem = Path(tar_path).stem
        npy_out = shard_dir / f"{stem}.npy"
        pq_out  = shard_dir / f"{stem}.parquet"
        if npy_out.exists() and pq_out.exists():
            print(f"[gpu{args.gpu}] skip {stem} (done)")
            continue

        ds = TarDataset(tar_path, preprocess)
        dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers,
                        collate_fn=_collate, pin_memory=True,
                        persistent_workers=False, prefetch_factor=4)

        keys, caps, embs = [], [], []
        n_seen = 0
        for batch in dl:
            if batch is None:
                continue
            xs, bkeys, bcaps = batch
            xs = xs.to(device, non_blocking=True)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                f = model.encode_image(xs)
            f = torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy().astype("float32")
            embs.append(f); keys.extend(bkeys); caps.extend(bcaps)
            n_seen += len(bkeys)
        print(f"[gpu{args.gpu}] {stem}: encoded {n_seen}")

        if not embs:
            print(f"[gpu{args.gpu}] {stem}: 0 images, writing empty")
            np.save(npy_out, np.zeros((0, DIM), dtype="float32"))
            pd.DataFrame({"key": [], "caption": []}).to_parquet(pq_out)
            continue

        arr = np.concatenate(embs, axis=0)
        np.save(npy_out, arr)
        pd.DataFrame({"key": keys, "caption": caps}).to_parquet(pq_out)
        print(f"[gpu{args.gpu}] {stem}: {arr.shape[0]} vecs -> {npy_out.name}")

    print(f"[gpu{args.gpu}] done.")


if __name__ == "__main__":
    main()