#!/usr/bin/env python3
"""
serve_fsiglip2_knn.py
---------------------
KNN + scoring service for the zooclaw-FashionSigLIP2 corpus. Route-for-route
mirror of fsiglip/serve_fsiglip_knn.py, so every existing collector works by
changing ONLY --client-url to this server's port. The open_clip loader is
swapped for transformers (SigLIP2-base-patch16-384 fine-tune); text encoding
uses the canonical SigLIP recipe (padding="max_length").

Routes (same request/response contracts as the v1 server):
  POST /knn-service              {"text", "num_images"} -> [{url, image_url, caption, similarity}]
  POST /knn-service-ensemble     {"texts": [...]} prompt-ensemble retrieval
  POST /patch-coverage           pattern patch coverage (grid argmax tiles)
  POST /patch-color-coverage     color patch coverage (13-way color argmax)
  POST /score-image-files        {"texts": {k: prompt}, "image_paths": [...]}
                                 -> {"scores": [{k: sim}, ...]}
  GET  /health

Run side-by-side with the v1 server (default port 1236 vs 1235):
  python fsiglip2/serve_fsiglip2_knn.py --port 1236 --gpu 0
"""
import argparse, io, base64, os, threading
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import faiss
from PIL import Image
from flask import Flask, request, jsonify
from transformers import AutoModel, AutoProcessor

_IMG_SESSION = None


def _img_session():
    global _IMG_SESSION
    if _IMG_SESSION is None:
        import requests
        _IMG_SESSION = requests.Session()
        _IMG_SESSION.headers.update({"User-Agent": "Mozilla/5.0 (pod-bench-server)"})
    return _IMG_SESSION


MODEL = "srpone/zooclaw-fashionsiglip2"
OUT_CORPUS = "/home1/hjhj6411/pod_bench/data/fashionsiglip2_corpus"

_DEFAULT_COLORS = ["black", "white", "gray", "navy", "blue", "red", "pink",
                   "orange", "yellow", "green", "brown", "beige", "purple"]

app = Flask(__name__)
_state = {}


def _load(corpus, device, model_id):
    out = Path(corpus)
    index = faiss.read_index(str(out / "index.faiss"))
    ids = pd.read_parquet(out / "ids.parquet")
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    _state.update(index=index, ids=ids, model=model, processor=processor,
                  device=device, urls=ids["url"].tolist(), caps=ids["caption"].tolist())
    print(f"loaded index ntotal={index.ntotal}, ids={len(ids)}, device={device}, "
          f"model={model_id}")


class _NullCtx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def _amp():
    return (torch.autocast(device_type="cuda", dtype=torch.float16)
            if str(_state["device"]).startswith("cuda") else _NullCtx())


def _pooler_output(features):
    if torch.is_tensor(features):
        return features
    if getattr(features, "pooler_output", None) is not None:
        return features.pooler_output
    raise TypeError(f"expected tensor or pooled model output, got {type(features).__name__}")


def _encode_norm(texts):
    """Encode a list of texts -> L2-normalized (N, D) float32 matrix.
    SigLIP text towers are trained with padding='max_length' — required here."""
    proc = _state["processor"]
    toks = proc(text=list(texts), padding="max_length", truncation=True,
                return_tensors="pt").to(_state["device"])
    with torch.no_grad(), _amp():
        f = _pooler_output(_state["model"].get_text_features(**toks))
    f = torch.nn.functional.normalize(f.float(), dim=-1)
    return f.cpu().numpy().astype("float32")


def _embed_text(q):
    """Single text -> (1, D) normalized query vector."""
    f = _encode_norm([q])
    return np.ascontiguousarray(f)


def _embed_ensemble(texts):
    """Prompt-ensemble: encode each text, L2-normalize, AVERAGE, re-normalize."""
    mat = _encode_norm(texts)                  # (N, D), each row unit-norm
    mean = mat.mean(axis=0, keepdims=True)     # (1, D)
    norm = np.linalg.norm(mean, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-12, 1.0, norm)
    mean = mean / norm
    return np.ascontiguousarray(mean.astype("float32"))


def _search(qv, k):
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
    return out


# ---------- image encoding + patch coverage (same model, no extra load) ----------
def _load_pil(src):
    """src is a URL (http...), a local path, or a data:/base64 string -> RGB PIL."""
    if not isinstance(src, str):
        return Image.open(src).convert("RGB")
    if src.startswith("http"):
        r = _img_session().get(src, timeout=20); r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    if src.startswith("data:"):
        payload = src.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
    if os.path.exists(src):
        return Image.open(src).convert("RGB")
    try:
        return Image.open(io.BytesIO(base64.b64decode(src))).convert("RGB")
    except Exception:
        return Image.open(src).convert("RGB")  # let it raise a clear FileNotFound


_PIL_CACHE = OrderedDict()
_PIL_CACHE_MAX = 64
_PIL_CACHE_LOCK = threading.Lock()


def _load_pil_cached(src):
    if not (isinstance(src, str) and src.startswith("http")):
        return _load_pil(src)
    with _PIL_CACHE_LOCK:
        if src in _PIL_CACHE:
            _PIL_CACHE.move_to_end(src)
            return _PIL_CACHE[src]
    img = _load_pil(src)
    with _PIL_CACHE_LOCK:
        _PIL_CACHE[src] = img
        while len(_PIL_CACHE) > _PIL_CACHE_MAX:
            _PIL_CACHE.popitem(last=False)
    return img


@torch.no_grad()
def _encode_images_norm(pil_list):
    """Encode PIL images with the server's SigLIP2 + its own processor,
    L2-normalize. Same model/transform as retrieval, so scores share a scale."""
    proc = _state["processor"]
    batch = proc(images=pil_list, return_tensors="pt")["pixel_values"].to(_state["device"])
    with torch.no_grad(), _amp():
        f = _pooler_output(_state["model"].get_image_features(pixel_values=batch))
    return torch.nn.functional.normalize(f.float(), dim=-1)


def _img_feats_np(pil_list):
    """Image features as L2-normalized numpy (N, D)."""
    return _encode_images_norm(pil_list).float().cpu().numpy().astype("float32")


def _make_tiles(img, grid, drop_white=True, white_thresh=0.95, white_frac=0.85):
    """Split into grid*grid tiles. Optionally drop near-white (background) tiles."""
    W, H = img.size
    tw, th = W // grid, H // grid
    tiles = []
    arr_full = np.asarray(img).astype("float32") / 255.0
    for r in range(grid):
        for c in range(grid):
            x1, y1 = c * tw, r * th
            x2 = (c + 1) * tw if c < grid - 1 else W
            y2 = (r + 1) * th if r < grid - 1 else H
            tile = img.crop((x1, y1, x2, y2))
            if drop_white:
                sub = arr_full[y1:y2, x1:x2]
                bright = (sub.min(axis=-1) > white_thresh).mean()
                if bright > white_frac:
                    continue
            tiles.append(tile)
    return tiles


@app.route("/patch-coverage", methods=["POST"])
def patch_coverage():
    """Same contract as the v1 route — see fsiglip/serve_fsiglip_knn.py."""
    body = request.get_json(force=True)
    imgs = body.get("images")
    if not imgs:
        one = body.get("image")
        imgs = [one] if one else []
    pattern = (body.get("pattern") or "striped").replace("_", " ").lower()
    color = (body.get("color") or "").replace("_", " ").lower()
    garment = (body.get("garment") or "garment").replace("_", " ").lower()
    grid = int(body.get("grid", 3))
    drop_white = bool(body.get("drop_white", True))

    pat_txt = _encode_norm([f"a close-up of {pattern} patterned fabric",
                            "a close-up of plain solid-colored fabric"])
    target_is_solid = pattern in ("", "solid", "plain", "none")
    c = f"{color} " if color and color not in ("none", "unknown") else ""
    spec = (f"a {c}{garment} with an all-over {pattern} pattern, studio product shot, not cropped"
            if not target_is_solid
            else f"a {c}{garment}, studio product shot, not cropped")
    spec_txt = _encode_norm([spec])

    pat_t = torch.from_numpy(pat_txt).to(_state["device"])
    spec_t = torch.from_numpy(spec_txt).to(_state["device"])

    out = []
    for src in imgs:
        try:
            img = _load_pil_cached(src)
        except Exception as e:
            out.append({"src": str(src)[:80], "error": str(e)[:120]})
            continue
        gfeat = _encode_images_norm([img])
        gsim = float((gfeat @ spec_t.T)[0, 0].cpu())
        tiles = _make_tiles(img, grid, drop_white=drop_white)
        if not tiles:
            out.append({"src": str(src)[:80], "global_sim": gsim,
                        "coverage": 0.0, "n_pattern": 0, "n_valid": 0,
                        "target_is_solid": target_is_solid})
            continue
        tfeat = _encode_images_norm(tiles)
        sims = (tfeat @ pat_t.T)                  # (P,2): [patterned, plain]
        is_pat = (sims[:, 0] > sims[:, 1]).cpu().numpy()
        match = is_pat if not target_is_solid else ~is_pat
        out.append({"src": str(src)[:80], "global_sim": gsim,
                    "coverage": float(match.mean()),
                    "n_pattern": int(is_pat.sum()), "n_valid": int(len(tiles)),
                    "target_is_solid": target_is_solid})
    return jsonify(out)


@app.route("/patch-color-coverage", methods=["POST"])
def patch_color_coverage():
    """Same contract as the v1 route — see fsiglip/serve_fsiglip_knn.py."""
    body = request.get_json(force=True)
    imgs = body.get("images")
    if not imgs:
        one = body.get("image")
        imgs = [one] if one else []
    color = (body.get("color") or "").replace("_", " ").strip().lower()
    pattern = (body.get("pattern") or "").replace("_", " ").strip().lower()
    grid = int(body.get("grid", 3))
    drop_white = bool(body.get("drop_white", True))
    vocab = [str(c).replace("_", " ").strip().lower()
             for c in (body.get("colors") or _DEFAULT_COLORS)]
    if not color:
        return jsonify([{"error": "missing 'color'"} for _ in imgs] or
                       [{"error": "missing 'color'"}])
    if color not in vocab:
        vocab = [color] + vocab
    tgt_idx = vocab.index(color)

    if pattern and pattern not in ("solid", "plain", "none", "unknown"):
        prompts = [f"a close-up of {c} {pattern} fabric" for c in vocab]
    else:
        prompts = [f"a close-up of plain {c} colored fabric" for c in vocab]
    anchors = _encode_norm(prompts)              # (C, D) float32 numpy

    out = []
    for src in imgs:
        try:
            img = _load_pil_cached(src)
        except Exception as e:
            out.append({"src": str(src)[:80], "error": str(e)[:120]})
            continue
        tiles = _make_tiles(img, grid, drop_white=drop_white)
        if not tiles:
            out.append({"src": str(src)[:80], "coverage": 0.0,
                        "n_target": 0, "n_valid": 0,
                        "top_other": None, "top_other_frac": 0.0})
            continue
        tfeat = _img_feats_np(tiles)             # (P, D)
        sims = tfeat @ anchors.T                 # (P, C)
        am = sims.argmax(axis=1)
        n_valid = len(tiles)
        n_target = int((am == tgt_idx).sum())
        others = [vocab[int(i)] for i in am if int(i) != tgt_idx]
        if others:
            top_other, n_other = Counter(others).most_common(1)[0]
            top_other_frac = round(n_other / n_valid, 3)
        else:
            top_other, top_other_frac = None, 0.0
        out.append({"src": str(src)[:80],
                    "coverage": float(n_target / n_valid),
                    "n_target": n_target, "n_valid": n_valid,
                    "top_other": top_other, "top_other_frac": top_other_frac})
    return jsonify(out)


@app.route("/knn-service", methods=["POST"])
def knn():
    body = request.get_json(force=True)
    q = body.get("text", "")
    k = int(body.get("num_images", body.get("num_result_ids", 20)))
    if not q:
        return jsonify([])
    qv = _embed_text(q)
    return jsonify(_search(qv, k))


@app.route("/knn-service-ensemble", methods=["POST"])
def knn_ensemble():
    body = request.get_json(force=True)
    texts = body.get("texts")
    if not texts:
        t = body.get("text", "")
        texts = [t] if t else []
    texts = [str(t).strip() for t in texts if str(t).strip()]
    k = int(body.get("num_images", body.get("num_result_ids", 20)))
    if not texts:
        return jsonify([])
    qv = _embed_ensemble(texts) if len(texts) > 1 else _embed_text(texts[0])
    return jsonify(_search(qv, k))


@app.route("/score-image-files", methods=["POST"])
def score_image_files():
    """Same contract as the v1 route: {"texts": {k: prompt}, "image_paths": [...]}
    -> {"scores": [{k: sim}, ...]}."""
    body = request.get_json(force=True)
    texts = body.get("texts") or {}
    paths = body.get("image_paths") or body.get("images") or []

    if not isinstance(texts, dict) or not texts:
        return jsonify({"error": "missing non-empty dict field: texts"}), 400
    if not isinstance(paths, list) or not paths:
        return jsonify({"error": "missing non-empty list field: image_paths"}), 400

    keys = list(texts.keys())
    prompts = [str(texts[k]) for k in keys]

    try:
        anchors_np = _encode_norm(prompts)
        anchors_t = torch.from_numpy(anchors_np).to(_state["device"])
    except Exception as e:
        return jsonify({"error": f"text_encode_failed: {e}"}), 500

    out = []
    for src in paths:
        try:
            img = _load_pil_cached(str(src))
            feat = _encode_images_norm([img])
            sims = (feat @ anchors_t.T)[0].detach().float().cpu().numpy()
            out.append({k: float(v) for k, v in zip(keys, sims)})
        except Exception as e:
            out.append({"error": str(e)[:200]})

    return jsonify({"scores": out})


@app.route("/health")
def health():
    return jsonify({"ntotal": _state["index"].ntotal, "model": MODEL})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1236,
                    help="default 1236 so the v1 server (1235) can run side-by-side")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--corpus", default=OUT_CORPUS)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    _load(args.corpus, device, args.model)

    # optional expert routes (shared with the v1 server implementation)
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "fsiglip"))
        from segmentation_expert import GarmentSegmenter
        from serve_expert_routes import register_expert_routes
        _state["segmenter"] = GarmentSegmenter(
            backend=os.environ.get("SEG_BACKEND", "heuristic"))
        register_expert_routes(
            app, _state,
            load_pil=_load_pil_cached,
            encode_norm=_encode_norm,
            img_feats=_img_feats_np,
            make_tiles=_make_tiles)
        print("expert routes registered")
    except Exception as e:
        print(f"expert routes skipped: {e}")

    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
