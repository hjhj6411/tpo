#!/usr/bin/env python3
"""
build_faiss_fsiglip2.py
-----------------------
Concatenate per-shard FashionSigLIP2 embeddings into one IndexFlatIP (cosine =
inner product on L2-normalized vectors), and build the row->{url,caption} table
the KNN server returns. Joins shard `key` to metadata parquet for `url`.
Mirror of fsiglip/build_faiss_fsiglip.py; only the corpus paths/meta differ.

  python fsiglip2/build_faiss_fsiglip2.py
Outputs in fashionsiglip2_corpus/:
  index.faiss, ids.parquet (columns: key, url, caption), meta.json
"""
import glob, json, os
from pathlib import Path

import numpy as np
import pandas as pd
import faiss

CLIP_CORPUS = "/home1/hjhj6411/pod_bench/data/clip_corpus"
OUT_CORPUS  = "/home1/hjhj6411/pod_bench/data/fashionsiglip2_corpus"
MODEL = "srpone/zooclaw-fashionsiglip2"
DIM = 768


def load_url_map(clip_corpus):
    """key -> url from img2dataset metadata parquets."""
    mp = {}
    for pq in glob.glob(os.path.join(clip_corpus, "embeddings", "metadata", "*.parquet")):
        df = pd.read_parquet(pq, columns=["key", "url"])
        mp.update(dict(zip(df["key"].astype(str), df["url"].astype(str))))
    return mp


def main():
    out = Path(OUT_CORPUS)
    shard_dir = out / "shards"
    npys = sorted(glob.glob(str(shard_dir / "*.npy")))
    assert npys, f"no shard .npy in {shard_dir} — run fsiglip2/extract_fsiglip2.py first"

    print(f"loading url map from metadata ...")
    url_map = load_url_map(CLIP_CORPUS)
    print(f"  url map: {len(url_map)} keys")

    index = faiss.IndexFlatIP(DIM)
    keys_all, caps_all, urls_all = [], [], []
    total = 0
    for npy in npys:
        stem = Path(npy).stem
        arr = np.load(npy)
        if arr.shape[0] == 0:
            continue
        assert arr.shape[1] == DIM, f"{stem}: dim {arr.shape[1]} != {DIM}"
        meta = pd.read_parquet(shard_dir / f"{stem}.parquet")
        index.add(np.ascontiguousarray(arr.astype("float32")))
        ks = meta["key"].astype(str).tolist()
        keys_all.extend(ks)
        caps_all.extend(meta["caption"].astype(str).tolist())
        urls_all.extend(url_map.get(k, "") for k in ks)
        total += arr.shape[0]
        print(f"  + {stem}: {arr.shape[0]}  (cum {total})")

    assert index.ntotal == len(keys_all) == total
    faiss.write_index(index, str(out / "index.faiss"))
    pd.DataFrame({"key": keys_all, "url": urls_all, "caption": caps_all}) \
        .to_parquet(out / "ids.parquet")
    json.dump({"model": MODEL, "dim": DIM,
               "count": total, "metric": "ip", "normalized": True},
              open(out / "meta.json", "w"), indent=2)
    n_url = sum(1 for u in urls_all if u)
    print(f"\nDONE  vectors={total}  with_url={n_url}  -> {out/'index.faiss'}")


if __name__ == "__main__":
    main()
