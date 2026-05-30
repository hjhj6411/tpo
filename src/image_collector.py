"""
image_collector.py — POD-Bench v2 image collection (LOCAL Amazon FAISS backend).

This is a self-contained rewrite. It does NOT depend on the external
`clip-retrieval` package or any HTTP knn-service. Instead it queries the
per-garment FAISS indices produced by `scripts/build_faiss_index.py`
(FashionSigLIP text embeddings over Amazon product titles).

Pipeline per option (A/B/C/D):
  1. Build a text query from the option attributes (color + pattern + garment).
  2. Encode it with FashionSigLIP and KNN-search the garment's FAISS sub-index.
  3. Download candidate image URLs in rank order.
  4. Verify with a VLM (closed-set garment / color / pattern check).
  5. Keep the first candidate that passes; else fall back to Google (optional).

After all four are collected, enforce homogeneity WITHIN the matched-garment
pairs only — {A,B} share a garment and differ on the active axis; {C,D} likewise.
We never compare across the A<->C garment gap (different garments SHOULD differ).

Exposed names (imported by collect_images_clip_retrieval.py if you still use it):
  AttributeVerifier, HomogeneityChecker, download_image, google_image_search

Typical use (local backend):
  python -m src.image_collector \
      --plan_path data/options/option_plans.jsonl \
      --index_dir $AMAZON_META_DIR/faiss \
      --output data/images/collection_log.jsonl \
      --image_root data/images \
      --verify lenient --limit 30
"""

import argparse
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import save_jsonl, load_jsonl, save_json, log_step, call_vlm, parse_json_response
from configs.config import IMAGES_DIR, OPTIONS_DIR, IMAGE_COLLECTION

# Lazy/global model handles (loaded once)
_CLIP = {"model": None, "tokenizer": None, "preprocess": None, "device": None}

UA = {"User-Agent": "Mozilla/5.0 (compatible; POD-Bench/2.0)"}


# ════════════════════════════════════════════════════════════
#  FashionSigLIP loader (shared with build_faiss_index.py)
# ════════════════════════════════════════════════════════════

def _load_clip():
    if _CLIP["model"] is not None:
        return _CLIP
    import torch
    import open_clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, _ = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionSigLIP")
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionSigLIP")
    _CLIP.update(model=model, tokenizer=tokenizer, preprocess=preprocess, device=device)
    return _CLIP


def _encode_text(text):
    import torch
    c = _load_clip()
    tokens = c["tokenizer"]([text[:77]]).to(c["device"])
    with torch.no_grad():
        feat = c["model"].encode_text(tokens, normalize=True)
    return feat.cpu().float().numpy().astype("float32")  # (1, dim)


def _encode_image(pil_img):
    import torch
    c = _load_clip()
    t = c["preprocess"](pil_img).unsqueeze(0).to(c["device"])
    with torch.no_grad():
        feat = c["model"].encode_image(t, normalize=True)
    return feat.cpu().float().numpy().astype("float32")  # (1, dim)


# ════════════════════════════════════════════════════════════
#  Local FAISS index over Amazon products (per garment)
# ════════════════════════════════════════════════════════════

class LocalFaissRetriever:
    """Loads per-garment FAISS sub-indices built by scripts/build_faiss_index.py.

    Each garment dir holds index.faiss (FashionSigLIP text vectors over titles)
    and ids.json ([{title, image_url, asin, garment_key, ...}, ...]).
    Query: encode the option text with FashionSigLIP, KNN-search the garment's
    index, return ranked candidate product records (with image_url).
    """

    def __init__(self, index_dir):
        import faiss  # noqa
        self.faiss = faiss
        self.index_dir = Path(index_dir)
        self._cache = {}  # garment_key -> (index, ids_list)
        if not self.index_dir.exists():
            raise FileNotFoundError(
                f"FAISS index dir not found: {self.index_dir}. "
                f"Build it first with scripts/build_faiss_index.py")
        available = sorted(p.name for p in self.index_dir.iterdir() if p.is_dir())
        print(f"  [faiss] index dir: {self.index_dir}")
        print(f"  [faiss] garments available: {available}")

    def _load_garment(self, garment_key):
        if garment_key in self._cache:
            return self._cache[garment_key]
        gdir = self.index_dir / garment_key
        idx_path = gdir / "index.faiss"
        ids_path = gdir / "ids.json"
        if not idx_path.exists() or not ids_path.exists():
            self._cache[garment_key] = (None, None)
            return None, None
        import json
        index = self.faiss.read_index(str(idx_path))
        with open(ids_path, encoding="utf-8") as f:
            ids = json.load(f)
        self._cache[garment_key] = (index, ids)
        return index, ids

    def search(self, query, top_k=20, garment_key=None):
        if not garment_key:
            return []
        index, ids = self._load_garment(garment_key)
        if index is None:
            print(f"    [faiss] no index for garment '{garment_key}'")
            return []
        qvec = _encode_text(query)
        self.faiss.normalize_L2(qvec)
        k = min(top_k, index.ntotal)
        if k <= 0:
            return []
        sims, idxs = index.search(qvec, k)
        out = []
        for rank, j in enumerate(idxs[0]):
            if j < 0 or j >= len(ids):
                continue
            rec = ids[j]
            url = rec.get("image_url") or rec.get("url")
            if url:
                out.append({"image_url": url,
                            "title": (rec.get("title") or "")[:120],
                            "asin": rec.get("asin"),
                            "score": float(sims[0][rank]),
                            "source": "amazon_faiss"})
        return out


# ════════════════════════════════════════════════════════════
#  Image download
# ════════════════════════════════════════════════════════════

def download_image(url, dest_path, min_side=224, timeout=15):
    """Download + validate an image. Returns True on success (saved as JPEG)."""
    try:
        resp = requests.get(url, timeout=timeout, headers=UA)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return False
    w, h = img.size
    if min(w, h) < min_side:
        return False
    try:
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(dest_path, "JPEG", quality=92)
        return True
    except Exception:
        return False


# ════════════════════════════════════════════════════════════
#  Optional Google Custom Search fallback
# ════════════════════════════════════════════════════════════

def google_image_search(query, top_k=5):
    """Google Custom Search image fallback. Needs GOOGLE_API_KEY + GOOGLE_CSE_ID.
    Returns [] silently if unconfigured."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    cse_id = os.environ.get("GOOGLE_CSE_ID")
    if not api_key or not cse_id:
        return []
    try:
        resp = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={"key": api_key, "cx": cse_id, "q": query,
                    "searchType": "image", "num": min(top_k, 10),
                    "imgType": "photo", "safe": "active"},
            timeout=15)
        resp.raise_for_status()
        items = resp.json().get("items", [])
        return [{"image_url": it["link"], "title": it.get("title", "")[:120],
                 "source": "google"} for it in items if it.get("link")]
    except Exception as e:
        print(f"    [google] search failed: {e}")
        return []


# ════════════════════════════════════════════════════════════
#  VLM attribute verifier (closed-set)
# ════════════════════════════════════════════════════════════

VERIFY_SYSTEM = """You are a precise fashion product-image inspector.
Given ONE product image and a target description, answer whether the image matches.
Judge three attributes independently and report what you actually see.
Output ONLY JSON:
{"garment_match": true/false, "color_match": true/false, "pattern_match": true/false,
 "seen_garment": "...", "seen_color": "...", "seen_pattern": "...",
 "model_gender": "man"/"woman"/"unclear", "is_clean_product_shot": true/false}"""


class AttributeVerifier:
    """VLM closed-set verification of a downloaded image against target attrs."""

    def __init__(self, provider_override=None):
        self.provider = provider_override

    def verify(self, image_path, attrs):
        garment = (attrs.get("garment_category") or "").replace("_", " ")
        color = attrs.get("color") or ""
        pattern = (attrs.get("pattern") or "").replace("_", " ")
        target = f"garment={garment}; color={color}; pattern={pattern or 'solid/plain'}"
        prompt = (f"Target description: {target}\n"
                  f"Does the product image match each attribute? "
                  f"Report what you see and the three match booleans.")
        try:
            resp = call_vlm(prompt, image_paths=[str(image_path)],
                            stage="image_verifier", system=VERIFY_SYSTEM,
                            max_tokens=256, temperature=0.0,
                            provider_override=self.provider)
            parsed = parse_json_response(resp) or {}
        except Exception as e:
            parsed = {"error": str(e)[:120]}
        parsed["passes"] = bool(parsed.get("garment_match")
                                and parsed.get("color_match")
                                and parsed.get("pattern_match"))
        return parsed

    def check_set_consistency(self, verifs):
        """All four images should depict the same model gender (avoid a man's
        coat vs a woman's dress confound). Returns dict with a 'consistent' flag."""
        genders = [v.get("model_gender", "unclear") for v in verifs.values()]
        known = [g for g in genders if g in ("man", "woman")]
        consistent = len(set(known)) <= 1
        return {"genders": genders, "consistent": consistent}


# ════════════════════════════════════════════════════════════
#  Homogeneity checker (CLIP distance + SSIM) within pairs
# ════════════════════════════════════════════════════════════

class HomogeneityChecker:
    """Pairwise visual-homogeneity gates for matched-garment pairs."""

    def __init__(self):
        self._skimage_ssim = None

    def _img(self, path):
        return Image.open(path).convert("RGB")

    def clip_distances(self, paths):
        """Mean pairwise cosine DISTANCE (1 - sim) among the given images.
        Lower = more visually homogeneous. Returns {'mean_distance', 'max_distance'}."""
        feats = [_encode_image(self._img(p)) for p in paths]
        dists = []
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                sim = float((feats[i] * feats[j]).sum())
                dists.append(1.0 - sim)
        if not dists:
            return {"mean_distance": 0.0, "max_distance": 0.0, "passed": True}
        mean_d, max_d = float(np.mean(dists)), float(max(dists))
        th = IMAGE_COLLECTION["max_clip_distance_within_options"]
        return {"mean_distance": mean_d, "max_distance": max_d,
                "threshold": th, "passed": max_d <= th}

    def ssim_pair(self, path_a, path_b, size=(256, 256)):
        """Structural similarity between two images (grayscale, resized).
        Returns a float in [-1, 1]; -1.0 on failure."""
        try:
            from skimage.metrics import structural_similarity as ssim
        except Exception:
            return -1.0
        try:
            a = np.array(self._img(path_a).resize(size).convert("L"))
            b = np.array(self._img(path_b).resize(size).convert("L"))
            return float(ssim(a, b))
        except Exception:
            return -1.0


# ════════════════════════════════════════════════════════════
#  Per-plan collection
# ════════════════════════════════════════════════════════════

def augment(search_query):
    """Bias retrieval/text-match toward clean product shots."""
    return f"{search_query}, fashion product photo, plain background"


def _verify_pass(vr, mode):
    if mode == "off":
        return True
    if mode == "lenient":
        return bool(vr.get("garment_match") and vr.get("color_match"))
    return bool(vr.get("passes"))  # strict: garment AND color AND pattern


def _download_one(cands, img_path, attrs, verifier, mode, top_n=10):
    tmp = Path(str(img_path) + ".tmp.jpg")
    for cand in cands[:top_n]:
        if not download_image(cand["image_url"], tmp):
            continue
        if mode == "off":
            tmp.rename(img_path)
            return True, cand.get("title", ""), {"passes": True, "model_gender": "unclear"}
        vr = verifier.verify(tmp, attrs)
        if _verify_pass(vr, mode):
            tmp.rename(img_path)
            return True, cand.get("title", ""), vr
        tmp.unlink(missing_ok=True)
    return False, "", {}


def collect_for_plan(plan, retriever, checker, verifier, out_root, mode, use_google,
                     top_k=20):
    qid = plan["query_id"]
    out_dir = Path(out_root) / qid
    out_dir.mkdir(parents=True, exist_ok=True)
    active_axis = plan["active_axis"]

    result = {"query_id": qid, "user_id": plan["user_id"], "domain": "fashion",
              "active_axis": active_axis, "main_category": plan.get("main_category"),
              "scenario_archetype": plan.get("scenario_archetype"),
              "options": {}, "all_collected": False,
              "homogeneity_AB": None, "homogeneity_CD": None,
              "structure_preserved": None, "set_consistency": None}

    paths, verifs = {}, {}
    for k in "ABCD":
        opt = plan["options"][k]
        attrs = opt["attributes"]
        img_path = out_dir / f"{k}.jpg"
        if img_path.exists():
            paths[k] = img_path
            verifs[k] = {"passes": True, "model_gender": "unclear"}
            result["options"][k] = {"image_path": str(img_path), "source": "cached",
                                    "search_query": opt["search_query"]}
            continue

        q = augment(opt["search_query"])
        cands = retriever.search(q, top_k=top_k, garment_key=attrs.get("garment_category"))
        ok, title, vr = _download_one(cands, img_path, attrs, verifier, mode)
        src = "amazon_faiss"
        if not ok and use_google:
            ok, title, vr = _download_one(
                google_image_search(opt["search_query"], top_k=5),
                img_path, attrs, verifier, mode, top_n=5)
            src = "google" if ok else "FAILED"
        elif not ok:
            src = "FAILED"

        if ok:
            paths[k] = img_path
            verifs[k] = vr
            result["options"][k] = {"image_path": str(img_path), "source": src,
                                    "search_query": opt["search_query"],
                                    "source_title": title, "verification": vr}
        else:
            result["options"][k] = {"image_path": None, "source": "FAILED",
                                    "search_query": opt["search_query"]}

    result["all_collected"] = len(paths) == 4
    if result["all_collected"]:
        result["set_consistency"] = verifier.check_set_consistency(verifs)
        # homogeneity WITHIN matched-garment pairs only
        result["homogeneity_AB"] = checker.clip_distances([paths["A"], paths["B"]])
        result["homogeneity_CD"] = checker.clip_distances([paths["C"], paths["D"]])
        if active_axis in ("color", "pattern"):
            ssim_val = checker.ssim_pair(paths["A"], paths["B"])
            th = IMAGE_COLLECTION["min_ssim_for_color_variants"]
            result["structure_preserved"] = {
                "ssim_A_vs_B": ssim_val, "threshold": th,
                "passed": (ssim_val < 0) or (ssim_val >= th)}
    return result


def run(plan_path, output_path, image_root, index_dir, limit, mode, use_google, top_k):
    log_step(f"Image Collection (LOCAL Amazon FAISS, verify={mode})")
    plans = load_jsonl(plan_path)
    print(f"  Loaded {len(plans)} plans")

    retriever = LocalFaissRetriever(index_dir)
    checker = HomogeneityChecker()
    verifier = AttributeVerifier()

    if Path(output_path).exists():
        existing = load_jsonl(output_path)
        done = {r["query_id"] for r in existing}
        results, todo = existing, [p for p in plans if p["query_id"] not in done]
        print(f"  Resuming: {len(done)} already collected")
    else:
        results, todo = [], plans
    if limit > 0:
        todo = todo[:limit]

    for i, plan in enumerate(todo):
        print(f"  [{i+1}/{len(todo)}] {plan['query_id']} ({plan['active_axis']})")
        try:
            results.append(collect_for_plan(plan, retriever, checker, verifier,
                                            image_root, mode, use_google, top_k))
            if (i + 1) % 10 == 0:
                save_jsonl(results, output_path)
        except Exception as e:
            print(f"    ERROR: {e}")
        time.sleep(0.05)

    save_jsonl(results, output_path)
    n_done = sum(1 for r in results if r["all_collected"])
    print(f"\n  ✓ Saved {len(results)} records: complete={n_done} "
          f"({n_done/max(len(results),1):.0%})")

    # summary stats
    src_counter = {}
    for r in results:
        for k in "ABCD":
            s = r["options"].get(k, {}).get("source", "?")
            src_counter[s] = src_counter.get(s, 0) + 1
    print("  source breakdown:", src_counter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan_path", type=Path, default=OPTIONS_DIR / "option_plans.jsonl")
    ap.add_argument("--index_dir", type=Path,
                    default=Path(os.environ.get("AMAZON_META_DIR",
                                 "/home/hjhj6411/fashion/data/amazon/fashion_only")) / "faiss")
    ap.add_argument("--output", type=Path, default=IMAGES_DIR / "collection_log.jsonl")
    ap.add_argument("--image_root", type=Path, default=IMAGES_DIR)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--verify", choices=["strict", "lenient", "off"], default="lenient")
    ap.add_argument("--google", action="store_true", help="enable Google image fallback")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.force and Path(args.output).exists():
        Path(args.output).unlink()
    run(args.plan_path, args.output, args.image_root, args.index_dir,
        args.limit, args.verify, args.google, args.top_k)


if __name__ == "__main__":
    main()
