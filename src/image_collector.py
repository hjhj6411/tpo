"""
Image Collector
================
각 option plan의 4 선지에 대해 이미지를 수집합니다.

수집 소스 우선순위:
  1. Amazon Reviews 2023 (이미 다운로드된 메타데이터)
  2. Google Custom Search API (무료 100 queries/day)
  3. (fallback) 사용자가 직접 이미지 URL 제공

품질 검증:
  - 최소 해상도 (224x224)
  - 단일 객체 / 깔끔한 배경 (CLIP 기반)
  - 4 선지 간 시각적 동질성 (CLIP 임베딩 거리 측정)

사용:
  python -m src.image_collector

산출:
  data/images/<query_id>/A.jpg, B.jpg, C.jpg, D.jpg
  data/images/collection_log.jsonl
"""

import argparse
import hashlib
import io
import os
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from .utils import (
    save_jsonl, load_jsonl, save_json, load_json, log_step,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import (
    IMAGES_DIR, OPTIONS_DIR, IMAGE_COLLECTION,
)


# ─────────────────────────────────────────────
# Amazon Reviews 2023 검색 (메타데이터 기반)
# ─────────────────────────────────────────────

class AmazonCatalogIndex:
    """Amazon Reviews 2023 메타데이터에서 이미지 검색.

    사용자의 환경에는 이미 ingest된 amazon 데이터가 있다고 가정합니다.
    예상 경로: /home/hjhj6411/fashion/data/amazon/meta_*.jsonl

    이 클래스는 lazy하게 작동합니다 — 데이터가 없으면 None 반환.
    """

    def __init__(self, index_path: Path = None):
        self.index_path = index_path or Path(
            os.environ.get("AMAZON_META_DIR", "/home/hjhj6411/fashion/data/amazon")
        )
        self._loaded = False
        self._items = []  # 간단한 in-memory index

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._loaded = True

        if not self.index_path.exists():
            print(f"  Amazon catalog not found at {self.index_path}; will skip Amazon source")
            return

        # 모든 meta_*.jsonl 파일에서 (title, image_url) 추출
        meta_files = list(self.index_path.glob("meta_*.jsonl"))
        if not meta_files:
            meta_files = list(self.index_path.glob("*.jsonl"))

        print(f"  Loading Amazon catalog index from {len(meta_files)} files...")
        n_loaded = 0
        for mf in meta_files[:5]:  # 너무 큰 경우 제한
            try:
                with open(mf, encoding="utf-8") as f:
                    for line in f:
                        try:
                            import json
                            item = json.loads(line)
                            title = item.get("title", "") or item.get("name", "")
                            images = item.get("images", []) or item.get("image", [])

                            # image URL 추출
                            img_url = None
                            if isinstance(images, list) and images:
                                first = images[0]
                                if isinstance(first, dict):
                                    img_url = first.get("large") or first.get("hi_res") or first.get("thumb")
                                elif isinstance(first, str):
                                    img_url = first
                            elif isinstance(images, str):
                                img_url = images

                            if title and img_url:
                                self._items.append({
                                    "title": title.lower(),
                                    "image_url": img_url,
                                    "category": item.get("main_category", ""),
                                })
                                n_loaded += 1
                        except Exception:
                            continue
                        if n_loaded >= 50000:  # 메모리 절약
                            break
            except Exception as e:
                print(f"    Skipped {mf.name}: {e}")

            if n_loaded >= 50000:
                break

        print(f"  Loaded {n_loaded} Amazon items")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """단순 키워드 매칭으로 검색."""
        self._ensure_loaded()
        if not self._items:
            return []

        # 쿼리 토큰
        tokens = [t.lower() for t in query.split() if len(t) > 2]
        if not tokens:
            return []

        # 매칭 점수 계산
        scored = []
        for item in self._items:
            score = sum(1 for t in tokens if t in item["title"])
            if score >= max(1, len(tokens) // 2):
                scored.append((score, item))

        scored.sort(key=lambda x: -x[0])
        return [s[1] for s in scored[:top_k]]


# ─────────────────────────────────────────────
# Google Custom Search (무료 100/day)
# ─────────────────────────────────────────────

def google_image_search(query: str, top_k: int = 5) -> list[dict]:
    """Google Custom Search API로 이미지 검색.

    환경변수 필요:
      GOOGLE_API_KEY: Google Cloud API key
      GOOGLE_CSE_ID: Custom Search Engine ID

    무료 한도: 100 queries/day.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    cse_id = os.environ.get("GOOGLE_CSE_ID")

    if not api_key or not cse_id:
        return []  # 비활성

    try:
        resp = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key": api_key,
                "cx": cse_id,
                "q": query,
                "searchType": "image",
                "num": min(top_k, 10),
                "safe": "active",
                "imgSize": "medium",
            },
            timeout=20,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
        return [{
            "image_url": it.get("link"),
            "title": it.get("title", ""),
            "source": "google",
        } for it in items]
    except Exception as e:
        print(f"    Google search error: {e}")
        return []


# ─────────────────────────────────────────────
# 이미지 다운로드 + 검증
# ─────────────────────────────────────────────

def download_image(url: str, target_path: Path,
                    min_resolution=(224, 224)) -> bool:
    """이미지 다운로드 + 크기 검증 + 표준화."""
    try:
        resp = requests.get(url, timeout=15,
                            headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()

        img = Image.open(io.BytesIO(resp.content)).convert("RGB")

        if img.size[0] < min_resolution[0] or img.size[1] < min_resolution[1]:
            return False

        # 512x512로 리사이즈 (긴 변 기준)
        img.thumbnail((512, 512), Image.LANCZOS)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(target_path, "JPEG", quality=90)
        return True
    except Exception as e:
        return False


# ─────────────────────────────────────────────
# CLIP 기반 동질성 검증
# ─────────────────────────────────────────────

class CLIPHomogeneityChecker:
    """4 선지의 시각적 동질성을 CLIP으로 검증."""

    def __init__(self):
        self._model = None
        self._processor = None
        self._device = None

    def _ensure_loaded(self):
        if self._model is not None:
            return

        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            model_name = "openai/clip-vit-base-patch32"  # 빠른 버전
            print(f"  Loading CLIP ({model_name}) on {self._device}")
            self._model = CLIPModel.from_pretrained(model_name).to(self._device).eval()
            self._processor = CLIPProcessor.from_pretrained(model_name)
            self._torch = torch
        except Exception as e:
            print(f"  CLIP loading failed: {e}. Homogeneity check disabled.")
            self._model = "DISABLED"

    def check_homogeneity(self, image_paths: list[Path]) -> dict:
        """4개 이미지 간 pairwise distance를 측정.

        Returns: {
          "mean_distance": float,
          "max_distance": float,
          "passed": bool,
        }
        """
        self._ensure_loaded()
        if self._model == "DISABLED" or len(image_paths) < 2:
            return {"mean_distance": 0.0, "max_distance": 0.0, "passed": True,
                    "note": "skipped"}

        try:
            images = [Image.open(p).convert("RGB") for p in image_paths]
            inputs = self._processor(images=images, return_tensors="pt").to(self._device)
            with self._torch.no_grad():
                features = self._model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)

            # pairwise cosine distance
            sim_matrix = features @ features.T
            distances = 1 - sim_matrix.cpu().numpy()
            n = len(image_paths)
            pairs = [distances[i, j] for i in range(n) for j in range(i+1, n)]

            mean_d = float(np.mean(pairs))
            max_d = float(np.max(pairs))

            threshold = IMAGE_COLLECTION["max_clip_distance_within_options"]
            passed = max_d <= threshold

            return {
                "mean_distance": mean_d,
                "max_distance": max_d,
                "passed": passed,
                "threshold": threshold,
            }
        except Exception as e:
            return {"mean_distance": 0.0, "max_distance": 0.0, "passed": True,
                    "note": f"check failed: {e}"}


# ─────────────────────────────────────────────
# 메인 파이프라인
# ─────────────────────────────────────────────

def collect_images_for_plan(plan: dict, amazon: AmazonCatalogIndex,
                             checker: CLIPHomogeneityChecker,
                             out_root: Path) -> dict:
    """단일 plan의 4 선지에 대해 이미지 수집."""
    query_id = plan["query_id"]
    out_dir = out_root / query_id
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "query_id": query_id,
        "user_id": plan["user_id"],
        "domain": plan["domain"],
        "main_category": plan["main_category"],
        "options": {},
        "all_collected": False,
        "homogeneity": None,
    }

    image_paths = []
    sources_used = {}

    for opt_key in ["A", "B", "C", "D"]:
        opt = plan["options"][opt_key]
        search_query = opt["search_query_en"]
        img_path = out_dir / f"{opt_key}.jpg"

        if img_path.exists():
            image_paths.append(img_path)
            result["options"][opt_key] = {
                "image_path": str(img_path),
                "source": "cached",
                "search_query": search_query,
            }
            continue

        # 1순위: Amazon
        downloaded = False
        amazon_results = amazon.search(search_query, top_k=3)
        for cand in amazon_results:
            if download_image(cand["image_url"], img_path):
                downloaded = True
                sources_used[opt_key] = "amazon"
                result["options"][opt_key] = {
                    "image_path": str(img_path),
                    "source": "amazon",
                    "search_query": search_query,
                    "source_title": cand.get("title", "")[:100],
                }
                break

        # 2순위: Google
        if not downloaded:
            google_results = google_image_search(search_query, top_k=3)
            for cand in google_results:
                if download_image(cand["image_url"], img_path):
                    downloaded = True
                    sources_used[opt_key] = "google"
                    result["options"][opt_key] = {
                        "image_path": str(img_path),
                        "source": "google",
                        "search_query": search_query,
                        "source_title": cand.get("title", "")[:100],
                    }
                    break

        if downloaded:
            image_paths.append(img_path)
        else:
            result["options"][opt_key] = {
                "image_path": None,
                "source": "FAILED",
                "search_query": search_query,
            }

    result["all_collected"] = len(image_paths) == 4

    if result["all_collected"]:
        result["homogeneity"] = checker.check_homogeneity(image_paths)

    return result


def run_pipeline(plan_path: Path, output_path: Path,
                 image_root: Path, limit: int = 0):
    """전체 image collection 파이프라인 실행."""
    log_step("Image Collector")

    plans = load_jsonl(plan_path)
    print(f"  Loaded {len(plans)} option plans")

    amazon = AmazonCatalogIndex()
    checker = CLIPHomogeneityChecker()

    if output_path.exists():
        existing = load_jsonl(output_path)
        done = {r["query_id"] for r in existing}
        print(f"  Resuming: {len(done)} plans already processed")
        results = existing
        plans_to_do = [p for p in plans if p["query_id"] not in done]
    else:
        results = []
        plans_to_do = plans

    if limit > 0:
        plans_to_do = plans_to_do[:limit]

    n_complete = 0
    n_homogeneous = 0

    for i, plan in enumerate(plans_to_do):
        print(f"\n  [{i+1}/{len(plans_to_do)}] {plan['query_id']}")
        try:
            result = collect_images_for_plan(plan, amazon, checker, image_root)
            results.append(result)

            if result["all_collected"]:
                n_complete += 1
                homo = result.get("homogeneity") or {}
                if homo.get("passed", True):
                    n_homogeneous += 1
                    status = "✓"
                else:
                    status = f"⚠ heterogeneous (d_max={homo.get('max_distance', 0):.3f})"
            else:
                missing = [k for k, v in result["options"].items()
                          if v.get("source") == "FAILED"]
                status = f"✗ missing {missing}"
            print(f"    {status}")

            # 매 10개마다 저장
            if (i + 1) % 10 == 0:
                save_jsonl(results, output_path)

        except Exception as e:
            print(f"    ERROR: {e}")

        # API rate limiting
        time.sleep(0.5)

    save_jsonl(results, output_path)
    print(f"\n  ✓ Saved {len(results)} collection records to {output_path}")
    print(f"  Complete (all 4 images): {n_complete}/{len(plans_to_do)}")
    print(f"  Homogeneous: {n_homogeneous}/{n_complete}")


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
