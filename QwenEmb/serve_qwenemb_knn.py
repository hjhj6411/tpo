#!/usr/bin/env python3
"""Serve QwenEmb FAISS index behind the clip-retrieval-compatible HTTP API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

from qwenemb_encoder import QwenEmbConfig, QwenEmbEncoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="/home1/hjhj6411/pod_bench/QwenEmb/corpus")
    p.add_argument("--model-id", default="Qwen/Qwen3-VL-Embedding-8B")
    p.add_argument("--port", type=int, default=1236)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--device-map", default=None)
    p.add_argument("--dim", type=int, default=None)
    p.add_argument("--faiss-gpu", action="store_true")
    p.add_argument("--index-name", default="pod_qwenemb")
    return p.parse_args()


def main():
    args = parse_args()
    import faiss

    corpus = Path(args.corpus)
    index_path = corpus / "index.faiss"
    ids_path = corpus / "ids.parquet"
    meta_path = corpus / "meta.json"
    if not index_path.exists() or not ids_path.exists():
        raise SystemExit(f"Missing index/ids under {corpus}; run build_faiss_qwenemb.py first")

    index = faiss.read_index(str(index_path))
    if args.faiss_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    ids = pd.read_parquet(ids_path).reset_index(drop=True)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    cfg = QwenEmbConfig(
        model_id=args.model_id,
        device=args.device,
        device_map=args.device_map,
        dtype=args.dtype,
        dim=args.dim,
    )
    encoder = QwenEmbEncoder(cfg)

    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify({
            "backend": "qwenemb",
            "ntotal": int(index.ntotal),
            "dim": int(index.d),
            "index_name": args.index_name,
            "meta": meta,
        })

    @app.post("/knn-service")
    def knn_service():
        payload = request.get_json(force=True, silent=True) or {}
        text = payload.get("text") or payload.get("query") or ""
        if not text:
            return jsonify([])
        n = int(payload.get("num_images") or payload.get("num_result_ids") or 20)
        n = max(1, min(n, 200))
        q = encoder.encode_text([text])
        if q.shape[1] != index.d:
            return jsonify({
                "error": f"query dim {q.shape[1]} != index dim {index.d}; use matching --dim/model",
            }), 400
        sims, idxs = index.search(q.astype("float32"), n)
        out = []
        for sim, idx in zip(sims[0], idxs[0]):
            if idx < 0 or idx >= len(ids):
                continue
            row = ids.iloc[int(idx)]
            url = str(row.get("url") or "")
            caption = str(row.get("caption") or "")
            if not url:
                continue
            out.append({
                "url": url,
                "image_url": url,
                "caption": caption,
                "title": caption[:120],
                "key": str(row.get("key") or ""),
                "similarity": float(sim),
                "source": "qwenemb_faiss",
            })
        return jsonify(out)

    print(f"[qwenemb] serving ntotal={index.ntotal} dim={index.d} port={args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
