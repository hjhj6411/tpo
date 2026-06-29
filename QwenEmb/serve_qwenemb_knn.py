#!/usr/bin/env python3
"""Serve QwenEmb FAISS index behind the clip-retrieval-compatible HTTP API.

Routes:
  GET  /health
  POST /knn-service
      Full-text retrieval over the FAISS image corpus.
  POST /score-candidates
      Score existing FAISS candidates against axis text queries.
  POST /score-image-files
      Score local image/crop files against text queries with QwenEmb.

All score routes return raw QwenEmb cosine/IP similarities. They are not VLM
judgement probabilities and are not expected to be bounded like binary coverage
values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from PIL import Image

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


def _normalise_texts(payload: dict[str, Any]) -> tuple[list[str], list[str]]:
    texts_obj = payload.get("texts") or payload.get("queries") or payload.get("text") or payload.get("query")
    if isinstance(texts_obj, dict):
        names = [str(k) for k in texts_obj.keys()]
        texts = [str(v) for v in texts_obj.values()]
        return names, texts
    if isinstance(texts_obj, list):
        texts = [str(x) for x in texts_obj]
        names = [f"score_{i}" for i in range(len(texts))]
        return names, texts
    if isinstance(texts_obj, str):
        return ["score"], [texts_obj]
    return [], []


def _open_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def main():
    args = parse_args()
    import faiss

    corpus = Path(args.corpus)
    index_path = corpus / "index.faiss"
    ids_path = corpus / "ids.parquet"
    meta_path = corpus / "meta.json"
    if not index_path.exists() or not ids_path.exists():
        raise SystemExit(f"Missing index/ids under {corpus}; run build_faiss_qwenemb.py first")

    # Keep a CPU index for vector reconstruction / axis scoring. Optionally create a
    # GPU copy only for fast nearest-neighbour search.
    cpu_index = faiss.read_index(str(index_path))
    search_index = cpu_index
    if args.faiss_gpu:
        res = faiss.StandardGpuResources()
        search_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

    ids = pd.read_parquet(ids_path).reset_index(drop=True)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    key_to_indices: dict[str, list[int]] = {}
    if "key" in ids.columns:
        for i, key in enumerate(ids["key"].astype(str).tolist()):
            key_to_indices.setdefault(key, []).append(i)

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
            "ntotal": int(search_index.ntotal),
            "dim": int(search_index.d),
            "index_name": args.index_name,
            "candidate_scoring": True,
            "candidate_score_route": "/score-candidates",
            "image_file_score_route": "/score-image-files",
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
        if q.shape[1] != search_index.d:
            return jsonify({
                "error": f"query dim {q.shape[1]} != index dim {search_index.d}; use matching --dim/model",
            }), 400
        sims, idxs = search_index.search(q.astype("float32"), n)
        out = []
        for sim, idx in zip(sims[0], idxs[0]):
            if idx < 0 or idx >= len(ids):
                continue
            idx_int = int(idx)
            row = ids.iloc[idx_int]
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
                "faiss_index": idx_int,
                "similarity": float(sim),
                "source": "qwenemb_faiss",
            })
        return jsonify(out)

    @app.post("/score-candidates")
    def score_candidates():
        payload = request.get_json(force=True, silent=True) or {}
        names, texts = _normalise_texts(payload)
        if not texts:
            return jsonify({"error": "Missing texts/queries/text/query"}), 400

        indices = payload.get("indices") or payload.get("faiss_indices")
        keys = payload.get("keys")
        resolved: list[int] = []
        if indices is not None:
            try:
                resolved = [int(i) for i in indices]
            except Exception:
                return jsonify({"error": "indices must be a list of integers"}), 400
        elif keys is not None:
            for key in keys:
                hits = key_to_indices.get(str(key), [])
                resolved.append(int(hits[0]) if hits else -1)
        else:
            return jsonify({"error": "Missing indices/faiss_indices or keys"}), 400

        q = encoder.encode_text(texts).astype("float32")
        if q.shape[1] != cpu_index.d:
            return jsonify({
                "error": f"query dim {q.shape[1]} != index dim {cpu_index.d}; use matching --dim/model",
            }), 400

        rows = []
        for idx in resolved:
            if idx < 0 or idx >= cpu_index.ntotal:
                rows.append({"faiss_index": int(idx), "error": "index_out_of_range"})
                continue
            vec = np.asarray(cpu_index.reconstruct(int(idx)), dtype="float32")
            scores = q @ vec
            row = ids.iloc[int(idx)] if int(idx) < len(ids) else {}
            out = {
                "faiss_index": int(idx),
                "key": str(row.get("key") or "") if hasattr(row, "get") else "",
                "source": "qwenemb_axis_score",
            }
            for name, score in zip(names, scores.tolist()):
                out[str(name)] = float(score)
            rows.append(out)
        return jsonify({"scores": rows, "texts": dict(zip(names, texts))})

    @app.post("/score-image-files")
    def score_image_files():
        payload = request.get_json(force=True, silent=True) or {}
        names, texts = _normalise_texts(payload)
        if not texts:
            return jsonify({"error": "Missing texts/queries/text/query"}), 400
        paths_obj = payload.get("image_paths") or payload.get("paths") or payload.get("images")
        if not isinstance(paths_obj, list) or not paths_obj:
            return jsonify({"error": "Missing non-empty image_paths/paths/images list"}), 400
        image_paths = [str(p) for p in paths_obj]

        q = encoder.encode_text(texts).astype("float32")
        if q.shape[1] != cpu_index.d:
            return jsonify({
                "error": f"query dim {q.shape[1]} != index dim {cpu_index.d}; use matching --dim/model",
            }), 400

        valid_images = []
        valid_positions = []
        rows: list[dict[str, Any]] = [
            {"image_path": p, "source": "qwenemb_image_file_score"} for p in image_paths
        ]
        for pos, path in enumerate(image_paths):
            try:
                valid_images.append(_open_rgb(path))
                valid_positions.append(pos)
            except Exception as e:
                rows[pos]["error"] = f"image_open_failed: {e}"

        if valid_images:
            emb = encoder.encode_images(valid_images).astype("float32")
            scores = emb @ q.T
            for local_i, pos in enumerate(valid_positions):
                for name, score in zip(names, scores[local_i].tolist()):
                    rows[pos][str(name)] = float(score)

        return jsonify({"scores": rows, "texts": dict(zip(names, texts))})

    print(f"[qwenemb] serving ntotal={search_index.ntotal} dim={search_index.d} port={args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
