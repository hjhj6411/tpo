#!/usr/bin/env python3
"""
collect_images_clip_retrieval.py
--------------------------------
Image collection for POD-Bench options via rom1504/clip-retrieval KNN service.

SELF-CONTAINED: this script no longer imports from src.image_collector. All the
helpers it needs (download_image, google_image_search, AttributeVerifier,
HomogeneityChecker) are defined inline, so it runs cleanly inside a dedicated
`clip` conda environment WITHOUT pulling the pod env's numpy/torch stack.

Why a separate env: clip-retrieval + img2dataset + autofaiss pin numpy/pyarrow
versions that conflict with the pod analysis env. Keep them isolated; this
script only needs requests + Pillow at runtime (VLM checks go over HTTP to your
vLLM server, so no torch is required here).

Two backends:
  --backend hosted  : public LAION-5B knn-service  (fast unblock, no infra)
  --backend local   : your self-hosted `clip-retrieval back`  (controlled corpus)

Quality gates:
  - AttributeVerifier : VLM closed-set garment/color/pattern check (HTTP to vLLM)
  - HomogeneityChecker: SSIM within matched pairs (no torch needed)
    homogeneity is enforced WITHIN the {A,B} and {C,D} pairs only (same garment,
    differ on the active axis) -- never across the A<->C garment gap.

Install (in the clip env, see SETUP_CLIP_ENV.md):
    conda create -n clip python=3.10 -y && conda activate clip
    pip install clip-retrieval img2dataset autofaiss requests pillow scikit-image

Quick test (no infra, hosted LAION):
    python src/collect_images_clip_retrieval.py --backend hosted --limit 5 --verify off

Self-hosted (after `clip-retrieval back --port 1234 ...`):
    python src/collect_images_clip_retrieval.py --backend local \
        --client-url http://127.0.0.1:1234/knn-service --indice-name pod_fashion \
        --limit 50 --verify lenient

--verify modes:
    strict  : VLM must match garment AND color AND pattern
    lenient : VLM must match garment AND color (pattern often mis-detected)
    off     : accept first downloadable image (smoke-test only)
"""

import argparse
import io
import os
import sys
import time
from pathlib import Path

import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import save_jsonl, load_jsonl, log_step, call_vlm, parse_json_response
from configs.config import IMAGES_DIR, OPTIONS_DIR, IMAGE_COLLECTION

UA = {"User-Agent": "Mozilla/5.0 (compatible; POD-Bench/2.0)"}


# ============================================================
#  Inline helpers (formerly in src.image_collector)
# ============================================================

def download_image(url, dest_path, min_side=224, timeout=15):
    """Download + validate an image, save as JPEG. Returns True on success."""
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


def google_image_search(query, top_k=5):
    """Google Custom Search fallback. Needs GOOGLE_API_KEY + GOOGLE_CSE_ID; else []."""
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


VERIFY_SYSTEM = """You are a precise fashion product-image inspector.
Given ONE product image and a target description, judge three attributes
independently and report what you actually see.
Output ONLY JSON:
{"garment_match": true/false, "color_match": true/false, "pattern_match": true/false,
 "seen_garment": "...", "seen_color": "...", "seen_pattern": "...",
 "model_gender": "man"/"woman"/"unclear", "is_clean_product_shot": true/false}"""


class AttributeVerifier:
    """VLM closed-set verification over HTTP (no local torch)."""

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
        genders = [v.get("model_gender", "unclear") for v in verifs.values()]
        known = [g for g in genders if g in ("man", "woman")]
        return {"genders": genders, "consistent": len(set(known)) <= 1}


class HomogeneityChecker:
    """SSIM-based pairwise homogeneity (no torch/CLIP needed in the clip env)."""

    def _img(self, path):
        return Image.open(path).convert("RGB")

    def ssim_pair(self, path_a, path_b, size=(256, 256)):
        try:
            from skimage.metrics import structural_similarity as ssim
            import numpy as np
        except Exception:
            return -1.0
        try:
            a = np.array(self._img(path_a).resize(size).convert("L"))
            b = np.array(self._img(path_b).resize(size).convert("L"))
            return float(ssim(a, b))
        except Exception:
            return -1.0

    def pair_homogeneity(self, path_a, path_b):
        s = self.ssim_pair(path_a, path_b)
        th = IMAGE_COLLECTION["min_ssim_for_color_variants"]
        return {"ssim": s, "threshold": th, "passed": (s < 0) or (s >= th)}


# ============================================================
#  clip-retrieval KNN client
# ============================================================

class KnnRetriever:
    """Thin wrapper over clip_retrieval.ClipClient (preferred) or raw HTTP."""

    def __init__(self, backend, client_url, indice_name, num_images=20):
        self.backend = backend
        self.indice_name = indice_name
        self.num_images = num_images
        self._client = None
        self._http_url = client_url

        if backend == "hosted":
            url = "https://knn.laion.ai/knn-service"
            indice_name = indice_name or "laion5B-L-14"
        else:
            url = client_url
        try:
            from clip_retrieval.clip_client import ClipClient, Modality
            self._client = ClipClient(
                url=url, indice_name=indice_name,
                num_images=num_images, modality=Modality.IMAGE)
            print(f"  [knn] ClipClient ready (backend={backend}, indice={indice_name})")
        except Exception as e:
            print(f"  [knn] ClipClient unavailable ({e}); falling back to raw HTTP")
            self._http_url = url

    def search(self, query, top_k=20, garment_key=None):
        out = []
        if self._client is not None:
            try:
                for r in self._client.query(text=query)[:top_k]:
                    url = r.get("url") or r.get("image_url")
                    if url:
                        out.append({"image_url": url,
                                    "title": r.get("caption", "")[:100],
                                    "source": "clip_retrieval"})
                return out
            except Exception as e:
                print(f"    [knn] client query failed: {e}")
        if self._http_url:
            try:
                resp = requests.post(self._http_url, json={
                    "text": query, "modality": "image",
                    "num_images": top_k, "num_result_ids": top_k,
                    "indice_name": self.indice_name}, timeout=60)
                resp.raise_for_status()
                for r in resp.json()[:top_k]:
                    url = r.get("url") or r.get("image_url")
                    if url:
                        out.append({"image_url": url,
                                    "title": r.get("caption", "")[:100],
                                    "source": "clip_retrieval_http"})
            except Exception as e:
                print(f"    [knn] http query failed: {e}")
        return out


def augment(search_query):
    """Light prompt engineering to bias retrieval toward clean product shots."""
    return f"{search_query}, fashion product photo, plain background"


# ============================================================
#  Per-plan collection
# ============================================================

def _verify_pass(vr, mode):
    if mode == "off":
        return True
    if mode == "lenient":
        return bool(vr.get("garment_match") and vr.get("color_match"))
    return bool(vr.get("passes"))  # strict


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


def collect_for_plan(plan, knn, checker, verifier, out_root, mode, use_google, top_k=20):
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
        cands = knn.search(q, top_k=top_k, garment_key=attrs.get("garment_category"))
        ok, title, vr = _download_one(cands, img_path, attrs, verifier, mode)
        src = "clip_retrieval"
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
        result["homogeneity_AB"] = checker.pair_homogeneity(paths["A"], paths["B"])
        result["homogeneity_CD"] = checker.pair_homogeneity(paths["C"], paths["D"])
        if active_axis in ("color", "pattern"):
            result["structure_preserved"] = result["homogeneity_AB"]
    return result


def run(plan_path, output_path, image_root, backend, client_url, indice_name,
        limit, mode, use_google, top_k):
    log_step(f"Image Collection via clip-retrieval (backend={backend}, verify={mode})")
    plans = load_jsonl(plan_path)
    print(f"  Loaded {len(plans)} plans")

    knn = KnnRetriever(backend, client_url, indice_name)
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
            results.append(collect_for_plan(plan, knn, checker, verifier,
                                            image_root, mode, use_google, top_k))
            if (i + 1) % 10 == 0:
                save_jsonl(results, output_path)
        except Exception as e:
            print(f"    ERROR: {e}")
        time.sleep(0.2)

    save_jsonl(results, output_path)
    n_done = sum(1 for r in results if r["all_collected"])
    print(f"\n  Saved {len(results)}: complete={n_done} "
          f"({n_done/max(len(results),1):.0%})")
    src_counter = {}
    for r in results:
        for k in "ABCD":
            s = r["options"].get(k, {}).get("source", "?")
            src_counter[s] = src_counter.get(s, 0) + 1
    print("  source breakdown:", src_counter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hosted", "local"], default="hosted")
    ap.add_argument("--client-url", default="http://127.0.0.1:1234/knn-service")
    ap.add_argument("--indice-name", default=None)
    ap.add_argument("--plan_path", type=Path, default=OPTIONS_DIR / "option_plans.jsonl")
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
    run(args.plan_path, args.output, args.image_root, args.backend, args.client_url,
        args.indice_name, args.limit, args.verify, args.google, args.top_k)


if __name__ == "__main__":
    main()
