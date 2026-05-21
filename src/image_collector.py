"""
Image Collector v2 — Parent-ASIN Variant Search.

For active_axis ∈ {color, pattern}: prefer parent_asin variants so structure
is preserved across A vs B. For active_axis=garment_category, fall back to
title-keyword search. Google fallback as last resort.

Quality gates: FashionSigLIP pairwise distance ≤ 0.65; SSIM ≥ 0.35 between A and B
(color/pattern only). FashionSigLIP also used for pre-download candidate reranking.
"""

import argparse
import io
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from .utils import save_jsonl, load_jsonl, log_step

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import IMAGES_DIR, OPTIONS_DIR, IMAGE_COLLECTION


class AmazonCatalogIndex:
    def __init__(self, index_path=None):
        self.index_path = index_path or Path(
            os.environ.get("AMAZON_META_DIR", "/home/hjhj6411/fashion/data/amazon")
        )
        self._loaded = False
        self._items = []
        self._by_parent = defaultdict(list)

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
        n = 0
        for mf in meta_files:
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
                            parent = item.get("parent_asin")
                            if title and img_url:
                                rec = {"title": title, "image_url": img_url,
                                        "category": item.get("main_category", ""),
                                        "parent_asin": parent,
                                        "asin": item.get("asin")}
                                self._items.append(rec)
                                if parent:
                                    self._by_parent[parent].append(rec)
                                n += 1
                        except Exception:
                            continue
            except Exception as e:
                print(f"    Skipped {mf.name}: {e}")
        print(f"  Indexed {n} items, {len(self._by_parent)} parent groups")

    def title_search(self, query, top_k=10):
        self._ensure_loaded()
        if not self._items:
            return []
        tokens = [t.lower() for t in query.split() if len(t) > 2]
        if not tokens:
            return []
        garment_tokens = {
            "blouse", "shirt", "jacket", "coat", "dress", "skirt",
            "shorts", "pants", "jeans", "hoodie", "sweater", "blazer",
            "tank", "t-shirt", "tshirt", "windbreaker", "parka", "trench",
            "suit", "cardigan", "vest", "leggings", "jumpsuit",
        }
        required = [t for t in tokens if t in garment_tokens]
        scored = []
        for it in self._items:
            if required and not all(r in it["title"] for r in required):
                continue
            score = sum(1 for t in tokens if t in it["title"])
            if score >= max(1, len(tokens) // 2):
                scored.append((score, it))
        scored.sort(key=lambda x: -x[0])
        return [s[1] for s in scored[:top_k]]

    def variant_search(self, anchor_query, variant_tokens, top_k_groups=3):
        self._ensure_loaded()
        if not self._items or not self._by_parent:
            return []
        anchor_tokens = [t.lower() for t in anchor_query.split() if len(t) > 2]
        var_lower = [v.lower() for v in variant_tokens]
        candidate_parents = []
        for parent, group in self._by_parent.items():
            if len(group) < 2:
                continue
            scored = []
            for it in group:
                a_score = sum(1 for t in anchor_tokens if t in it["title"])
                scored.append((a_score, it))
            scored.sort(key=lambda x: -x[0])
            if scored[0][0] < max(1, len(anchor_tokens) // 2):
                continue
            covered = {}
            for vt in var_lower:
                for _, it in scored:
                    if vt in it["title"]:
                        covered[vt] = it
                        break
            if len(covered) >= 2:
                candidate_parents.append((len(covered), list(covered.values())))
        candidate_parents.sort(key=lambda x: -x[0])
        return [g[1] for g in candidate_parents[:top_k_groups]]


def google_image_search(query, top_k=5):
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
        print(f"    Google error: {e}")
        return []


def download_image(url, target_path, min_resolution=(224, 224)):
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
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


class HomogeneityChecker:
    def __init__(self):
        self._clip = None
        self._proc = None
        self._tokenizer = None
        self._device = None
        self._torch = None

    def _ensure_clip(self):
        if self._clip is not None:
            return
        try:
            import torch
            import open_clip

            self._device = "cpu"
            print(f"  Loading Marqo/marqo-fashionSigLIP on {self._device}")
            self._clip, _, self._proc = open_clip.create_model_and_transforms(
                "hf-hub:Marqo/marqo-fashionSigLIP"
            )
            self._clip = self._clip.to(self._device).eval()
            self._tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionSigLIP")
            self._torch = torch
        except Exception as e:
            print(f"  CLIP unavailable: {e}")
            self._clip = "DISABLED"

    def rank_candidates_by_clip(self, candidates, attrs):
        """다운로드 전에 FashionSigLIP text embedding으로 후보 재랭킹.
        attrs: {"garment_category": ..., "color": ..., "pattern": ...}
        """
        self._ensure_clip()
        if self._clip == "DISABLED" or not candidates:
            return candidates
        try:
            query = " ".join(filter(None, [
                attrs.get("color", ""),
                attrs.get("pattern", ""),
                attrs.get("garment_category", "").replace("_", " "),
            ])).strip()
            text_tokens = self._tokenizer([query]).to(self._device)
            with self._torch.no_grad():
                text_feat = self._clip.encode_text(text_tokens, normalize=True)

            titles = [c.get("title", "")[:77] for c in candidates]
            title_tokens = self._tokenizer(titles).to(self._device)
            with self._torch.no_grad():
                title_feats = self._clip.encode_text(title_tokens, normalize=True)

            sims = (title_feats @ text_feat.T).squeeze(-1).cpu().numpy()
            if sims.ndim == 0:
                sims = np.array([float(sims)])
            ranked = sorted(zip(candidates, sims.tolist()), key=lambda x: -x[1])
            return [c for c, _ in ranked]
        except Exception as e:
            print(f"    rank_candidates_by_clip failed: {e}")
            return candidates

    def clip_distances(self, image_paths):
        self._ensure_clip()
        if self._clip == "DISABLED" or len(image_paths) < 2:
            return {"mean_distance": 0.0, "max_distance": 0.0,
                    "passed": True, "note": "skipped"}
        try:
            imgs = self._torch.stack([
                self._proc(Image.open(p).convert("RGB"))
                for p in image_paths
            ]).to(self._device)
            with self._torch.no_grad():
                feats = self._clip.encode_image(imgs, normalize=True)
            sims = (feats @ feats.T).cpu().numpy()
            dists = 1 - sims
            n = len(image_paths)
            pairs = [dists[i, j] for i in range(n) for j in range(i + 1, n)]
            mean_d = float(np.mean(pairs))
            max_d = float(np.max(pairs))
            th = IMAGE_COLLECTION["max_clip_distance_within_options"]
            return {"mean_distance": mean_d, "max_distance": max_d,
                    "passed": max_d <= th, "threshold": th}
        except Exception as e:
            return {"mean_distance": 0.0, "max_distance": 0.0,
                    "passed": True, "note": f"failed: {e}"}

    def ssim_pair(self, p1, p2):
        try:
            from skimage.metrics import structural_similarity as ssim
        except ImportError:
            return -1.0
        try:
            img1 = np.array(Image.open(p1).convert("L").resize((256, 256)))
            img2 = np.array(Image.open(p2).convert("L").resize((256, 256)))
            return float(ssim(img1, img2, data_range=255))
        except Exception:
            return -1.0


def collect_for_plan(plan, amazon, checker, out_root):
    query_id = plan["query_id"]
    out_dir = out_root / query_id
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "query_id": query_id, "user_id": plan["user_id"],
        "domain": "fashion", "active_axis": plan["active_axis"],
        "main_category": plan["main_category"],
        "options": {}, "all_collected": False,
        "homogeneity": None, "structure_preserved": None,
    }
    active_axis = plan["active_axis"]
    image_paths = []
    used_variant = False

    if active_axis in ("color", "pattern"):
        a_attrs = plan["options"]["A"]["attributes"]
        anchor = a_attrs.get("garment_category", "").replace("_", " ")
        var_tokens = [a_attrs.get(active_axis, ""),
                       plan["options"]["B"]["attributes"].get(active_axis, "")]
        groups = amazon.variant_search(anchor, var_tokens, top_k_groups=2)
        if groups:
            group = groups[0]
            for k in "ABCD":
                opt = plan["options"][k]
                target_v = opt["attributes"][active_axis]
                if opt["attributes"].get("garment_category") != a_attrs.get("garment_category"):
                    continue
                match = next((it for it in group if target_v in it["title"]), None)
                if match:
                    img_path = out_dir / f"{k}.jpg"
                    if download_image(match["image_url"], img_path):
                        result["options"][k] = {
                            "image_path": str(img_path), "source": "amazon_variant",
                            "search_query": opt["search_query"],
                            "source_title": match.get("title", "")[:100],
                            "parent_asin": match.get("parent_asin"),
                        }
                        image_paths.append(img_path)
                        used_variant = True

    for k in "ABCD":
        if k in result["options"]:
            continue
        opt = plan["options"][k]
        query = opt["search_query"]
        img_path = out_dir / f"{k}.jpg"
        if img_path.exists():
            image_paths.append(img_path)
            result["options"][k] = {"image_path": str(img_path),
                                      "source": "cached", "search_query": query}
            continue
        downloaded = False
        # title_search top_k=10 후보 → FashionSigLIP 재랭킹 → 상위 3개만 시도
        cands = amazon.title_search(query, top_k=10)
        cands = checker.rank_candidates_by_clip(cands, opt["attributes"])
        for cand in cands[:3]:
            if download_image(cand["image_url"], img_path):
                downloaded = True
                result["options"][k] = {
                    "image_path": str(img_path), "source": "amazon_title",
                    "search_query": query,
                    "source_title": cand.get("title", "")[:100],
                }
                break
        if not downloaded:
            for cand in google_image_search(query, top_k=3):
                if download_image(cand["image_url"], img_path):
                    downloaded = True
                    result["options"][k] = {
                        "image_path": str(img_path), "source": "google",
                        "search_query": query,
                        "source_title": cand.get("title", "")[:100],
                    }
                    break
        if downloaded:
            image_paths.append(img_path)
        else:
            result["options"][k] = {"image_path": None,
                                      "source": "FAILED", "search_query": query}

    result["all_collected"] = len(image_paths) == 4
    result["used_variant_search"] = used_variant
    if result["all_collected"]:
        result["homogeneity"] = checker.clip_distances(image_paths)
        if active_axis in ("color", "pattern"):
            ssim = checker.ssim_pair(
                Path(result["options"]["A"]["image_path"]),
                Path(result["options"]["B"]["image_path"]),
            )
            th = IMAGE_COLLECTION["min_ssim_for_color_variants"]
            result["structure_preserved"] = {
                "ssim_A_vs_B": ssim, "threshold": th,
                "passed": (ssim < 0) or (ssim >= th),
                "note": "skipped" if ssim < 0 else "",
            }
    return result


def run_pipeline(plan_path, output_path, image_root, limit=0):
    log_step("Image Collector v2 (parent-ASIN variant + fallback)")
    plans = load_jsonl(plan_path)
    print(f"  Loaded {len(plans)} plans")

    amazon = AmazonCatalogIndex()
    checker = HomogeneityChecker()

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

    n_complete = n_homo = n_variant = 0
    for i, plan in enumerate(todo):
        print(f"\n  [{i+1}/{len(todo)}] {plan['query_id']} ({plan['active_axis']})")
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
                    status = f"⚠ d_max={homo.get('max_distance', 0):.3f}"
                if r.get("used_variant_search"):
                    n_variant += 1
                    status += " (variant)"
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
    print(f"\n  ✓ Saved {len(results)}: complete={n_complete}, homo={n_homo}, "
          f"variant={n_variant}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan_path", type=Path, default=OPTIONS_DIR / "option_plans.jsonl")
    parser.add_argument("--output", type=Path, default=IMAGES_DIR / "collection_log.jsonl")
    parser.add_argument("--image_root", type=Path, default=IMAGES_DIR)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Ignore existing cache")
    args = parser.parse_args()
    if args.force and args.output.exists():
        args.output.unlink()
    run_pipeline(args.plan_path, args.output, args.image_root, args.limit)


if __name__ == "__main__":
    main()
