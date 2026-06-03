#!/usr/bin/env python3
"""
serve_fsiglip_knn.py
--------------------
Minimal KNN service compatible with the collector's KnnRetriever HTTP fallback.

Request  (POST /knn-service):
  {"text": "...", "modality": "image", "num_images": 24, "indice_name": "..."}
Response:
  [{"url": "...", "image_url": "...", "caption": "..."}, ...]

Query text is embedded with the SAME FashionSigLIP model used to build the index,
L2-normalized, searched against IndexFlatIP (cosine).

  python serve_fsiglip_knn.py --port 1235 --gpu 0
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import open_clip
import faiss
from flask import Flask, request, jsonify

MODEL = "hf-hub:Marqo/marqo-fashionSigLIP"
OUT_CORPUS = "/home1/hjhj6411/pod_bench/data/fashionsiglip_corpus"

app = Flask(__name__)
_state = {}


def _load(corpus, device):
    out = Path(corpus)
    index = faiss.read_index(str(out / "index.faiss"))
    ids = pd.read_parquet(out / "ids.parquet")
    model = open_clip.create_model_and_transforms(MODEL)[0].to(device).eval()
    tok = open_clip.get_tokenizer(MODEL)
    _state.update(index=index, ids=ids, model=model, tok=tok, device=device,
                  urls=ids["url"].tolist(), caps=ids["caption"].tolist())
    print(f"loaded index ntotal={index.ntotal}, ids={len(ids)}, device={device}")


def _embed_text(q):
    toks = _state["tok"]([q]).to(_state["device"])
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        f = _state["model"].encode_text(toks)
    f = torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy().astype("float32")
    return np.ascontiguousarray(f)


@app.route("/knn-service", methods=["POST"])
def knn():
    body = request.get_json(force=True)
    q = body.get("text", "")
    k = int(body.get("num_images", body.get("num_result_ids", 20)))
    if not q:
        return jsonify([])
    qv = _embed_text(q)
    sims, idxs = _state["index"].search(qv, k)
    urls, caps = _state["urls"], _state["caps"]
    out = []
    for rank, i in enumerate(idxs[0]):
        if i < 0:
            continue
        u = urls[i]
        if not u:
            continue
        out.append({"url": u, "image_url": u, "caption": caps[i],
                    "similarity": float(sims[0][rank])})
    return jsonify(out)


@app.route("/health")
def health():
    return jsonify({"ntotal": _state["index"].ntotal})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1235)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--corpus", default=OUT_CORPUS)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    _load(args.corpus, device)
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
