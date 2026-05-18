"""
Image Collector
================

For each option plan, collect 4 images (one per option) and verify quality:

Sources (priority order):
  1. Amazon Reviews 2023 metadata (free, already on disk)
  2. Google Custom Search API (free 100/day, fallback)

Quality checks:
  - Min resolution 224x224
  - 4 options must be visually homogeneous (CLIP distance check)
"""

import argparse
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from .utils import save_jsonl, load_jsonl, log_step

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import IMAGES_DIR, OPTIONS_DIR, IMAGE_COLLECTION


# ─────────────────────────────────────────────
# Amazon Reviews 2023 metadata search
# ─────────────────────────────────────────────

class AmazonCatalogIndex:
    """In-memory keyword index over Amazon Reviews 2023 metadata.

    Expected files at $AMAZON_META_DIR/*.jsonl with fields:
      title, images (list of dict or str), main_category, ...
    """

    def __init__(self, index_path: Path = None):
        self.index_path = index_path or Path(
            os.environ.get("AMAZON_META_DIR", "/home/hjhj6411/fashion/data/amazon")
        )
        self._loaded = False
        self._items = []

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._loaded = True
        if not self.index_path.exists():
            print(f"  Amazon catalog not found at {self.index_path}, skipping.")
            return

        meta_files = list(self.index_path.glob("meta_*.jsonl"))
        if not meta_files:
            meta_files = list(self.index_path.glob("*.jsonl"))

        print(f"  Indexing {len(meta_files)} Amazon meta files...")
        n_loaded = 0
        for mf in meta_files[:5]:
            try:
                with open(mf, encoding="utf-8") as f:
                    for line in f:
                        try:
                            item = json.loads(line)
                            title = (item.get("title") or item.get("name") or "").lower()
                            images = item.get("images") or item.get("image") or []
                            img_url = None
                            if isinstance(images, list) and images:
                                first = images[0]
                                if isinstance(first, dict):
                                    img_url = (first.get("large") or first.get("hi_res")
                                                or first.get("thumb"))
                                elif isinstance(first, str):
                                    img_url = first
                            elif isinstance(images, str):
                                img_url = images

                            if title and img_url:
                                self._items.append({
                                    "title": title, "image_url": img_url,
                                    "category": item.get("main_category", ""),
                                })
                                n_loaded += 1
                        except Exception:
                            continue
                        if n_loaded >= 50000:
                            break
            except Exception as e:
                print(f"    Skipped {mf.name}: {e}")
            if n_loaded >= 50000:
                break
        print(f"  Indexed {n_loaded} Amazon items")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        self._ensure_loaded()
        if not self._items:
            return []
        tokens = [t.lower() for t in query.split() if len(t) > 2]
        if not tokens:
            return []
        scored = []
        for it in self._items:
            score = sum(1 for t in tokens if t in it["title"])
            if score >= max(1, len(tokens) // 2):
                scored.append((score, it))
        scored.sort(key=lambda x: -x[0])
        return [s[1] for s in scored[:top_k]]


# ─────────────────────────────────────────────
# Google Custom Search (free 100/day)
# ─────────────────────────────────────────────

def google_image_search(query: str, top_k: int = 5) -> list[dict]:
    api_key = os.environ.get("GOOGLE_API_KEY")
    cse_id = os.environ.get("GOOGLE_CSE_ID")
    if not api_key or not cse_id:
        return []
    try:
        resp = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={"key": api_key, "cx": cse_id, "q": query,
                    "searchType": "image", "num": min(top_k, 10),
                    "safe": "active", "imgSize": "medium"},
            timeout=20,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
        return [{"image_url": it.get("link"), "title": it.get("title", ""),
                 "source": "google"} for it in items]
    except Exception as e:
        print(f"    Google search error: {e}")
        return []


# ─────────────────────────────────────────────
# Image download
# ─────────────────────────────────────────────

def download_image(url: str, target_path: Path,
                    min_resolution=(224, 224)) -> bool:
    try:
        resp = requests.get(url, timeout=15,
                            headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        if img.size[0] < min_resolution[0] or img.size[1] < min_resolution[1]:
            return False
        img.thumbnail((512, 512), Image.LANCZOS)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(target_path, "JPEG", quality=90)
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────
# CLIP-based homogeneity check
# ─────────────────────────────────────────────

class CLIPHomogeneity:
    def __init__(self):
        self._model = None
        self._processor = None
        self._device = None
        self._torch = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            name = "openai/clip-vit-base-patch32"
            print(f"  Loading CLIP ({name}) on {self._device}")
            self._model = CLIPModel.from_pretrained(name).to(self._device).eval()
            self._processor = CLIPProcessor.from_pretrained(name)
            self._torch = torch
        except Exception as e:
            print(f"  CLIP unavailable: {e}")
            self._model = "DISABLED"

    def check(self, image_paths: list[Path]) -> dict:
        self._ensure_loaded()
        if self._model == "DISABLED" or len(image_paths) < 2:
            return {"mean_distance": 0, "max_distance": 0, "passed": True,
                    "note": "skipped"}
        try:
            images = [Image.open(p).convert("RGB") for p in image_paths]
            inputs = self._processor(images=images, return_tensors="pt").to(self._device)
            with self._torch.no_grad():
                feats = self._model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            sims = (feats @ feats.T).cpu().numpy()
            dists = 1 - sims
            n = len(image_paths)
            pairs = [dists[i, j] for i in range(n) for j in range(i+1, n)]
            mean_d = float(np.mean(pairs))
            max_d = float(np.max(pairs))
            threshold = IMAGE_COLLECTION["max_clip_distance_within_options"]
            return {"mean_distance": mean_d, "max_distance": max_d,
                    "passed": max_d <= threshold, "threshold": threshold}
        except Exception as e:
            return {"mean_distance": 0, "max_distance": 0, "passed": True,
                    "note": f"failed: {e}"}


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def collect_for_plan(plan: dict, amazon: AmazonCatalogIndex,
                      checker: CLIPHomogeneity, out_root: Path) -> dict:
    query_id = plan["query_id"]
    out_dir = out_root / query_id
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "query_id": query_id,
        "user_id": plan["user_id"],
        "domain": "fashion",
        "main_category": plan["main_category"],
        "options": {},
        "all_collected": False,
        "homogeneity": None,
    }

    image_paths = []
    for opt_key in "ABCD":
        opt = plan["options"][opt_key]
        query = opt["search_query"]
        img_path = out_dir / f"{opt_key}.jpg"

        if img_path.exists():
            image_paths.append(img_path)
            result["options"][opt_key] = {
                "image_path": str(img_path),
                "source": "cached",
                "search_query": query,
            }
            continue

        downloaded = False
        # 1) Amazon
        for cand in amazon.search(query, top_k=3):
            if download_image(cand["image_url"], img_path):
                downloaded = True
                result["options"][opt_key] = {
                    "image_path": str(img_path),
                    "source": "amazon",
                    "search_query": query,
                    "source_title": cand.get("title", "")[:100],
                }
                break

        # 2) Google fallback
        if not downloaded:
            for cand in google_image_search(query, top_k=3):
                if download_image(cand["image_url"], img_path):
                    downloaded = True
                    result["options"][opt_key] = {
                        "image_path": str(img_path),
                        "source": "google",
                        "search_query": query,
                        "source_title": cand.get("title", "")[:100],
                    }
                    break

        if downloaded:
            image_paths.append(img_path)
        else:
            result["options"][opt_key] = {
                "image_path": None, "source": "FAILED",
                "search_query": query,
            }

    result["all_collected"] = len(image_paths) == 4
    if result["all_collected"]:
        result["homogeneity"] = checker.check(image_paths)
    return result


def run_pipeline(plan_path: Path, output_path: Path,
                 image_root: Path, limit: int = 0):
    log_step("Image Collector")
    plans = load_jsonl(plan_path)
    print(f"  Loaded {len(plans)} plans")

    amazon = AmazonCatalogIndex()
    checker = CLIPHomogeneity()

    if output_path.exists():
        existing = load_jsonl(output_path)
        done = {r["query_id"] for r in existing}
        results = existing
        todo = [p for p in plans if p["query_id"] not in done]
        print(f"  Resuming: {len(done)} already collected")
    else:
        results = []
        todo = plans

    if limit > 0:
        todo = todo[:limit]

    n_complete = 0
    n_homo = 0

    for i, plan in enumerate(todo):
        print(f"\n  [{i+1}/{len(todo)}] {plan['query_id']}")
        try:
            r = collect_for_plan(plan, amazon, checker, image_root)
            results.append(r)
            if r["all_collected"]:
                n_complete += 1
                homo = r.get("homogeneity") or {}
                if homo.get("passed", True):
                    n_homo += 1
                    status = "✓"
                else:
                    status = f"⚠ heterogeneous (d_max={homo.get('max_distance', 0):.3f})"
            else:
                missing = [k for k, v in r["options"].items()
                            if v.get("source") == "FAILED"]
                status = f"✗ missing {missing}"
            print(f"    {status}")
            if (i + 1) % 10 == 0:
                save_jsonl(results, output_path)
        except Exception as e:
            print(f"    ERROR: {e}")
        time.sleep(0.5)

    save_jsonl(results, output_path)
    print(f"\n  ✓ Saved {len(results)}, complete {n_complete}, homo {n_homo}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan_path", type=Path,
                        default=OPTIONS_DIR / "option_plans.jsonl")
    parser.add_argument("--output", type=Path,
                        default=IMAGES_DIR / "collection_log.jsonl")
    parser.add_argument("--image_root", type=Path, default=IMAGES_DIR)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    run_pipeline(args.plan_path, args.output, args.image_root, args.limit)


if __name__ == "__main__":
    main()
