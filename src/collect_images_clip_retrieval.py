#!/usr/bin/env python3
"""
collect_images_clip_retrieval.py
--------------------------------
Image collection for POD-Bench options via rom1504/clip-retrieval KNN service.
Replaces the broken AmazonCatalogIndex FAISS path; keeps the repo's quality gates.

Two backends:
  --backend hosted  : public LAION-5B knn-service  (fast unblock, no infra)
  --backend local   : your self-hosted `clip-retrieval back`  (controlled corpus)

Quality gates (reused from src.image_collector):
  - AttributeVerifier : VLM closed-set garment/color/pattern check
  - HomogeneityChecker: CLIP distance + SSIM
  FIX vs original: homogeneity is enforced WITHIN the {A,B} and {C,D} pairs only
  (same garment, differ on the active axis) — never across the A<->C garment gap.

Install:
    pip install clip-retrieval

Quick test (no infra):
    python src/collect_images_clip_retrieval.py --backend hosted --limit 5 --verify lenient

Self-hosted (after `clip-retrieval back --port 1234 ...`):
    python scripts/collect_images_clip_retrieval.py --backend local \
        --client-url http://127.0.0.1:1234/knn-service --indice-name pod_fashion \
        --limit 50 --verify lenient

--verify modes:
    strict  : VLM must match garment AND color AND pattern (original behavior)
    lenient : VLM must match garment AND color (pattern often mis-detected)
    off     : accept first downloadable image (smoke-test only — diagnoses "0 collected")
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import save_jsonl, load_jsonl, log_step
from src.image_collector import (
    AttributeVerifier, HomogeneityChecker, download_image, google_image_search,
)
from configs.config import IMAGES_DIR, OPTIONS_DIR, IMAGE_COLLECTION


# ── clip-retrieval KNN client ──────────────────────────────────────────

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
                num_images=num_images,
                modality=Modality.IMAGE,
            )
            print(f"  [knn] ClipClient ready (backend={backend}, indice={indice_name})")
        except Exception as e:
            print(f"  [knn] ClipClient unavailable ({e}); falling back to raw HTTP")
            self._http_url = url

    def search(self, query, top_k=20, garment_key=None):
        # garment_key kept for signature-compat with the old AmazonCatalogIndex
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
        # raw HTTP fallback to a `clip-retrieval back` knn-service
        if self._http_url:
            import requests
            try:
                resp = requests.post(self._http_url, json={
                    "text": query, "modality": "image",
                    "num_images": top_k, "num_result_ids": top_k,
                    "indice_name": self.indice_name,
                }, timeout=60)
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


# ── per-plan collection ────────────────────────────────────────────────

def _verify_pass(vr, mode):
    if mode == "off":
        return True
    if mode == "lenient":
        return bool(vr.get("garment_match") and vr.get("color_match"))
    return bool(vr.get("passes"))  # strict


def _download_one(cands, img_path, attrs, verifier, mode, top_n=10):
    tmp = img_path.with_suffix(".tmp.jpg")
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


def collect_for_plan(plan, knn, checker, verifier, out_root, mode, use_google):
    qid = plan["query_id"]
    out_dir = out_root / qid
    out_dir.mkdir(parents=True, exist_ok=True)
    active_axis = plan["active_axis"]

    result = {"query_id": qid, "user_id": plan["user_id"], "domain": "fashion",
              "active_axis": active_axis, "main_category": plan.get("main_category"),
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
        cands = knn.search(q, top_k=20, garment_key=attrs.get("garment_category"))
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
        # homogeneity WITHIN matched-garment pairs only (the corrected gate)
        result["homogeneity_AB"] = checker.clip_distances([paths["A"], paths["B"]])
        result["homogeneity_CD"] = checker.clip_distances([paths["C"], paths["D"]])
        if active_axis in ("color", "pattern"):
            ssim_val = checker.ssim_pair(paths["A"], paths["B"])
            th = IMAGE_COLLECTION["min_ssim_for_color_variants"]
            result["structure_preserved"] = {
                "ssim_A_vs_B": ssim_val, "threshold": th,
                "passed": (ssim_val < 0) or (ssim_val >= th)}
    return result


def run(plan_path, output_path, image_root, backend, client_url, indice_name,
        limit, mode, use_google):
    log_step(f"Image Collection via clip-retrieval (backend={backend}, verify={mode})")
    plans = load_jsonl(plan_path)
    print(f"  Loaded {len(plans)} plans")

    knn = KnnRetriever(backend, client_url, indice_name)
    checker = HomogeneityChecker()
    verifier = AttributeVerifier()

    if output_path.exists():
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
                                             image_root, mode, use_google))
            if (i + 1) % 10 == 0:
                save_jsonl(results, output_path)
        except Exception as e:
            print(f"    ERROR: {e}")
        time.sleep(0.2)

    save_jsonl(results, output_path)
    n_done = sum(1 for r in results if r["all_collected"])
    print(f"\n  ✓ Saved {len(results)}: complete={n_done}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hosted", "local"], default="hosted")
    ap.add_argument("--client-url", default="http://127.0.0.1:1234/knn-service")
    ap.add_argument("--indice-name", default=None)
    ap.add_argument("--plan_path", type=Path, default=OPTIONS_DIR / "option_plans.jsonl")
    ap.add_argument("--output", type=Path, default=IMAGES_DIR / "collection_log.jsonl")
    ap.add_argument("--image_root", type=Path, default=IMAGES_DIR)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verify", choices=["strict", "lenient", "off"], default="lenient")
    ap.add_argument("--google", action="store_true", help="enable Google image fallback")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.force and args.output.exists():
        args.output.unlink()
    run(args.plan_path, args.output, args.image_root, args.backend, args.client_url,
        args.indice_name, args.limit, args.verify, args.google)


if __name__ == "__main__":
    main()
