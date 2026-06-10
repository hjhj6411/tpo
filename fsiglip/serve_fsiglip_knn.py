#!/usr/bin/env python3
"""
serve_fsiglip_knn.py
--------------------
Minimal KNN service compatible with the collector's KnnRetriever HTTP fallback.

Request  (POST /knn-service):
  {"text": "...", "modality": "image", "num_images": 24, "indice_name": "..."}
Response:
  [{"url": "...", "image_url": "...", "caption": "...", "similarity": ...}, ...]

NEW (POST /knn-service-ensemble):  prompt-ensemble retrieval.
  {"texts": ["a photo of a navy shirt", "a navy shirt, full garment shown", ...],
   "num_images": 24, "indice_name": "..."}
  Each text is encoded with the SAME FashionSigLIP model, L2-normalized, AVERAGED
  in embedding space, then re-normalized and searched (CLIP prompt-ensemble; the
  averaged text embedding is more robust than any single prompt). Response format
  is identical to /knn-service. Back-compat: a single "text" is also accepted here.

NEW (POST /patch-coverage):  PATTERN patch coverage (grid tiles, each tile
  classified "{pattern} fabric" vs "plain fabric" by argmax; coverage =
  patterned_tiles / non-background_tiles).

NEW (POST /patch-color-coverage):  COLOR patch coverage.
  {"images": [url|path|b64, ...] OR "image": one,
   "color": "navy", "pattern": "striped"|"solid"|null, "garment": "...",
   "grid": 5, "drop_white": true, "colors": optional vocab override}
  Each tile's BASE COLOR is classified by argmax over per-color text anchors
  that hold the pattern constant ("a close-up of {c} {pattern} fabric"), so the
  decision varies on COLOR ONLY. A 13-way (full vocabulary) argmax is essential:
  a binary "navy vs plain" anchor cannot separate navy from blue, because both
  sit closer to "navy" than to "plain". With the full vocab, a blue tile loses
  the argmax to the "blue" anchor and is counted against the target.
  coverage = tiles whose argmax == target color / valid tiles.
  Also returns top_other = the most frequent non-target argmax color (direct
  view of confusions like navy<->blue, beige<->brown).

Query text is embedded with the SAME FashionSigLIP model used to build the index,
L2-normalized, searched against IndexFlatIP (cosine).

  python serve_fsiglip_knn.py --port 1235 --gpu 0
"""
import argparse, json, io, base64, os, threading
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import open_clip
import faiss
from PIL import Image
from flask import Flask, request, jsonify

# requests is only needed to fetch image URLs in /patch-coverage; imported lazily
# so the server runs in envs without it when images are passed as local paths.
_IMG_SESSION = None


def _img_session():
    global _IMG_SESSION
    if _IMG_SESSION is None:
        import requests
        _IMG_SESSION = requests.Session()
        _IMG_SESSION.headers.update({"User-Agent": "Mozilla/5.0 (pod-bench-server)"})
    return _IMG_SESSION

MODEL = "hf-hub:Marqo/marqo-fashionSigLIP"
OUT_CORPUS = "/home1/hjhj6411/pod_bench/data/fashionsiglip_corpus"

# default color vocabulary = POD-Bench FASHION_ATTRIBUTE_AXES["color"] (13 colors).
# A request may override it via the "colors" field.
_DEFAULT_COLORS = ["black", "white", "gray", "navy", "blue", "red", "pink",
                   "orange", "yellow", "green", "brown", "beige", "purple"]

app = Flask(__name__)
_state = {}


def _load(corpus, device):
    out = Path(corpus)
    index = faiss.read_index(str(out / "index.faiss"))
    ids = pd.read_parquet(out / "ids.parquet")
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL)
    model = model.to(device).eval()
    tok = open_clip.get_tokenizer(MODEL)
    _state.update(index=index, ids=ids, model=model, tok=tok, preprocess=preprocess,
                  device=device, urls=ids["url"].tolist(), caps=ids["caption"].tolist())
    print(f"loaded index ntotal={index.ntotal}, ids={len(ids)}, device={device}")


def _encode_norm(texts):
    """Encode a list of texts -> L2-normalized (N, D) float32 matrix."""
    toks = _state["tok"](list(texts)).to(_state["device"])
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        f = _state["model"].encode_text(toks)
    f = torch.nn.functional.normalize(f.float(), dim=-1)
    return f.cpu().numpy().astype("float32")


def _embed_text(q):
    """Single text -> (1, D) normalized query vector."""
    f = _encode_norm([q])
    return np.ascontiguousarray(f)


def _embed_ensemble(texts):
    """Prompt-ensemble: encode each text, L2-normalize, AVERAGE, re-normalize.
    Returns (1, D) float32. This matches the standard CLIP prompt-ensembling
    recipe (average over the embedding space, then normalize)."""
    mat = _encode_norm(texts)                 # (N, D), each row unit-norm
    mean = mat.mean(axis=0, keepdims=True)     # (1, D)
    # re-normalize the averaged vector for cosine/IP search
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
        # data:<mime>;base64,<payload>
        payload = src.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
    # local path if it exists; otherwise try to decode as raw base64
    if os.path.exists(src):
        return Image.open(src).convert("RGB")
    try:
        return Image.open(io.BytesIO(base64.b64decode(src))).convert("RGB")
    except Exception:
        return Image.open(src).convert("RGB")  # let it raise a clear FileNotFound


# Small URL->PIL LRU cache: when a collector scores the SAME candidate list on
# BOTH /patch-coverage and /patch-color-coverage (--color-cov all), every image
# would otherwise be downloaded twice. Local paths / base64 are not cached.
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
    """Encode a list of PIL images with the server's SigLIP + its own preprocess,
    L2-normalize. Same model/transform as retrieval, so scores are on the same scale."""
    pre = _state["preprocess"]
    batch = torch.stack([pre(im) for im in pil_list]).to(_state["device"])
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16) \
            if _state["device"].startswith("cuda") else _NullCtx():
        f = _state["model"].encode_image(batch)
    return torch.nn.functional.normalize(f.float(), dim=-1)


def _img_feats_np(pil_list):
    """Image features as L2-normalized numpy (N, D). Thin wrapper so coverage
    routes can do their scoring math in numpy (and tests can stub the encoder)."""
    return _encode_images_norm(pil_list).float().cpu().numpy().astype("float32")


class _NullCtx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def _make_tiles(img, grid, drop_white=True, white_thresh=0.95, white_frac=0.85):
    """Split into grid*grid tiles. Optionally drop near-white (background) tiles:
    a tile whose fraction of bright pixels exceeds white_frac is treated as
    background and excluded from the coverage denominator."""
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
                bright = (sub.min(axis=-1) > white_thresh).mean()  # near-white pixels
                if bright > white_frac:
                    continue
            tiles.append(tile)
    return tiles


@app.route("/patch-coverage", methods=["POST"])
def patch_coverage():
    """Body: {"images": [url|path|b64, ...] OR "image": one,
             "pattern": "striped", "grid": 3, "drop_white": true}
    For each image: split into grid*grid tiles (background tiles optionally
    dropped), classify each tile as '{pattern} fabric' vs 'plain solid fabric'
    with the SAME SigLIP, return coverage = patterned_tiles / valid_tiles.
    Also returns the whole-image global similarity to the all-over spec, for
    side-by-side comparison. All scores share the retrieval model's scale."""
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

    # text anchors (encoded once, shared scale with retrieval)
    pat_txt = _encode_norm([f"a close-up of {pattern} patterned fabric",
                            "a close-up of plain solid-colored fabric"])
    c = f"{color} " if color and color not in ("none", "unknown") else ""
    spec = (f"a {c}{garment} with an all-over {pattern} pattern, studio product shot, not cropped"
            if pattern not in ("", "solid", "plain", "none")
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
        # global similarity (whole image vs all-over spec)
        gfeat = _encode_images_norm([img])
        gsim = float((gfeat @ spec_t.T)[0, 0].cpu())
        # patch coverage
        tiles = _make_tiles(img, grid, drop_white=drop_white)
        if not tiles:
            out.append({"src": str(src)[:80], "global_sim": gsim,
                        "coverage": 0.0, "n_pattern": 0, "n_valid": 0})
            continue
        tfeat = _encode_images_norm(tiles)
        sims = (tfeat @ pat_t.T)                  # (P,2): [pattern, plain]
        is_pat = (sims[:, 0] > sims[:, 1]).cpu().numpy()
        out.append({"src": str(src)[:80], "global_sim": gsim,
                    "coverage": float(is_pat.mean()),
                    "n_pattern": int(is_pat.sum()), "n_valid": int(len(tiles))})
    return jsonify(out)


@app.route("/patch-color-coverage", methods=["POST"])
def patch_color_coverage():
    """COLOR twin of /patch-coverage — same tiling, different anchors.

    Body: {"images": [url|path|b64, ...] OR "image": one,
           "color": "navy",                 # REQUIRED target color
           "pattern": "striped"|"solid"|null,  # option's pattern context
           "garment": "...",                # accepted for API symmetry; unused
                                            # (tiles are fabric crops, the garment
                                            # word adds noise at tile scale)
           "grid": 5, "drop_white": true,
           "colors": optional vocab override (default: the 13 POD-Bench colors)}

    Per tile, argmax over per-color anchors. The PATTERN IS HELD CONSTANT in the
    anchor text so the argmax varies on color only:
        non-solid pattern: "a close-up of {c} {pattern} fabric"
        solid/unknown:     "a close-up of plain {c} colored fabric"
    coverage = tiles whose argmax == target color / valid tiles.

    Returns per image:
      {"src", "coverage", "n_target", "n_valid",
       "top_other": most frequent non-target argmax color (None if all target),
       "top_other_frac": its fraction of valid tiles}
    """
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

    # per-color anchors, pattern held constant -> argmax decides on color only
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
    # back-compat: allow a single "text" too
    if not texts:
        t = body.get("text", "")
        texts = [t] if t else []
    texts = [str(t).strip() for t in texts if str(t).strip()]
    k = int(body.get("num_images", body.get("num_result_ids", 20)))
    if not texts:
        return jsonify([])
    qv = _embed_ensemble(texts) if len(texts) > 1 else _embed_text(texts[0])
    return jsonify(_search(qv, k))


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
