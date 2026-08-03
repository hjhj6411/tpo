#!/usr/bin/env python3
"""Build a FAISS cosine/IP index for QwenEmb shard embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype("float32", copy=False)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return x / denom


def read_metadata(clip_corpus: Path, metadata_glob: str):
    frames = []
    for path in sorted(clip_corpus.glob(metadata_glob)):
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue
        cols = set(df.columns)
        key_col = None
        for c in ("key", "__key__", "id", "sample_id"):
            if c in cols:
                key_col = c
                break
        if key_col is None:
            # img2dataset parquet often has a file path; derive stem when possible.
            for c in ("image_path", "path", "filename"):
                if c in cols:
                    df = df.assign(key=df[c].astype(str).map(lambda s: Path(s).stem))
                    key_col = "key"
                    break
        if key_col is None or "url" not in cols:
            continue
        keep = [key_col, "url"]
        caption_col = None
        for c in ("caption", "text", "description"):
            if c in cols:
                caption_col = c
                keep.append(c)
                break
        sub = df[keep].rename(columns={key_col: "key"})
        if caption_col and caption_col != "caption":
            sub = sub.rename(columns={caption_col: "caption"})
        if "caption" not in sub.columns:
            sub["caption"] = ""
        frames.append(sub[["key", "url", "caption"]])
    if not frames:
        return pd.DataFrame(columns=["key", "url", "caption"])
    meta = pd.concat(frames, ignore_index=True)
    meta = meta.drop_duplicates("key", keep="first")
    return meta


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clip-corpus", default="/home1/hjhj6411/pod_bench/data/clip_corpus")
    p.add_argument("--corpus", default="/home1/hjhj6411/pod_bench/QwenEmb/corpus")
    p.add_argument("--metadata-glob", default="**/*.parquet")
    p.add_argument("--use-gpu", action="store_true")
    p.add_argument("--index-name", default="pod_qwenemb")
    return p.parse_args()


def main():
    args = parse_args()
    import faiss

    corpus = Path(args.corpus)
    clip_corpus = Path(args.clip_corpus)
    shard_dir = corpus / "shards"
    npys = sorted(shard_dir.glob("*.npy"))
    if not npys:
        raise SystemExit(f"No .npy shards found under {shard_dir}")

    mats = []
    rows = []
    for npy in tqdm(npys, desc="load shards"):
        mat = np.load(npy)
        if mat.ndim != 2:
            raise RuntimeError(f"Bad embedding shape {mat.shape}: {npy}")
        mats.append(mat.astype("float32", copy=False))
        pq = npy.with_suffix(".parquet")
        if not pq.exists():
            raise RuntimeError(f"Missing sidecar parquet: {pq}")
        rows.append(pd.read_parquet(pq))

    x = l2_normalize(np.concatenate(mats, axis=0))
    ids = pd.concat(rows, ignore_index=True)
    if len(ids) != x.shape[0]:
        raise RuntimeError(f"ids/vector mismatch: ids={len(ids)} vectors={x.shape[0]}")

    print(f"[qwenemb] vectors={x.shape[0]} dim={x.shape[1]}")
    index = faiss.IndexFlatIP(x.shape[1])
    if args.use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(x)
    if args.use_gpu:
        index = faiss.index_gpu_to_cpu(index)

    meta = read_metadata(clip_corpus, args.metadata_glob)
    print(f"[qwenemb] metadata rows={len(meta)}")
    if not meta.empty:
        ids = ids.merge(meta, on="key", how="left")
    else:
        ids["url"] = ""
        ids["caption"] = ""
    ids["url"] = ids.get("url", "").fillna("")
    ids["caption"] = ids.get("caption", "").fillna("")

    corpus.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(corpus / "index.faiss"))
    ids.to_parquet(corpus / "ids.parquet", index=False)
    meta_json = {
        "backend": "qwenemb",
        "index_name": args.index_name,
        "vectors": int(x.shape[0]),
        "dim": int(x.shape[1]),
        "with_url": int((ids["url"].astype(str) != "").sum()),
    }
    (corpus / "meta.json").write_text(json.dumps(meta_json, indent=2), encoding="utf-8")
    print(f"DONE vectors={meta_json['vectors']} with_url={meta_json['with_url']} dim={meta_json['dim']}")


if __name__ == "__main__":
    main()
