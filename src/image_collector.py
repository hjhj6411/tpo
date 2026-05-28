"""
Image Collector v3 — FAISS Semantic Retrieval (FashionSigLIP primary).

Changes from v2:
  - AmazonCatalogIndex: keyword title_search / variant_search 제거
    → FAISS inner-product ANN 검색으로 교체
  - HomogeneityChecker.rank_candidates_by_clip 제거 (FAISS가 이미 의미 정렬)
  - collect_for_plan: variant/title 두 브랜치 → semantic_search 단일 흐름
  - Google fallback은 그대로 유지

Quality gates (unchanged):
  1. AttributeVerifier: VLM closed-set check
  2. SetConsistencyVerifier: 4-option gender + framing consistency
  3. HomogeneityChecker: FashionSigLIP pairwise distance
  4. SSIM for color/pattern variants

오프라인 인덱스 빌드:
  python scripts/build_faiss_index.py \
    --meta-dir $AMAZON_META_DIR \
    --output-dir $AMAZON_META_DIR/faiss
"""

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from .utils import save_jsonl, load_jsonl, log_step, call_vlm, parse_json_response

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import IMAGES_DIR, OPTIONS_DIR, IMAGE_COLLECTION


KNOWN_COLORS = [
    "red", "blue", "green", "yellow", "orange", "purple", "pink",
    "black", "white", "gray", "grey", "brown", "beige", "navy",
    "teal", "cyan", "magenta", "olive", "maroon", "khaki", "cream",
    "sky blue", "light blue", "dark blue", "light gray", "dark gray",
]
KNOWN_PATTERNS = [
    "solid", "striped", "plaid", "floral", "graphic", "animal_print",
    "checkered", "polka_dot", "geometric", "abstract", "camouflage",
    "tie_dye", "paisley",
]
FEMININE_GARMENTS = {"blouse", "dress", "skirt", "sundress", "camisole"}
MALE_TITLE_TOKENS = {"men", "mens", "men's", "male", "boy", "boys"}


# ──────────────────────────────────────────────────────────────
# Quality Gates (unchanged from v2)
# ──────────────────────────────────────────────────────────────

class AttributeVerifier:
    """VLM closed-set verification."""
    STAGE = "image_verifier"
    SYSTEM = "You are a fashion image analyst. Answer strictly in JSON."

    def verify(self, image_path, expected_attrs):
        garment = expected_attrs.get("garment_category", "").replace("_", " ")
        color = expected_attrs.get("color", "")
        pattern = expected_attrs.get("pattern", "").replace("_", " ")
        prompt = (
            f"Look at this fashion product image.\n"
            f"1. Is the primary garment a {garment}? (yes/no)\n"
            f"2. Dominant color? Choose from: {', '.join(KNOWN_COLORS)}\n"
            f"3. Pattern? Choose from: {', '.join(KNOWN_PATTERNS)}\n"
            f"4. Model gender? male/female/no_model/unclear\n\n"
            f"Respond ONLY with JSON:\n"
            f'{{"garment_ok": true, "detected_color": "...", '
            f'"detected_pattern": "...", "model_gender": "..."}}'
        )
        try:
            raw = call_vlm(prompt=prompt, image_paths=[str(image_path)],
                           stage=self.STAGE, system=self.SYSTEM,
                           max_tokens=128, temperature=0.0)
            parsed = parse_json_response(raw)
            if not parsed:
                return self._fail("VLM parse error")
            garment_match = bool(parsed.get("garment_ok", False))
            detected_color = str(parsed.get("detected_color", "")).lower()
            detected_pattern = str(parsed.get("detected_pattern", "")).lower()
            model_gender = str(parsed.get("model_gender", "unclear")).lower()
            color_match = self._fuzzy(color, detected_color)
            pattern_match = self._fuzzy(pattern, detected_pattern)
            passes = garment_match and color_match and pattern_match
            reason = None
            if not passes:
                parts = []
                if not garment_match: parts.append(f"garment(expected={garment})")
                if not color_match: parts.append(f"color(exp={color},got={detected_color})")
                if not pattern_match: parts.append(f"pattern(exp={pattern},got={detected_pattern})")
                reason = "; ".join(parts)
            return {"passes": passes, "garment_match": garment_match,
                    "color_match": color_match, "pattern_match": pattern_match,
                    "detected_color": detected_color, "detected_pattern": detected_pattern,
                    "model_gender": model_gender, "rejection_reason": reason}
        except Exception as e:
            return self._fail(str(e))

    @staticmethod
    def _fuzzy(expected, detected):
        exp = expected.lower().replace("_", " ")
        det = detected.lower().replace("_", " ")
        return exp in det or det in exp

    @staticmethod
    def _fail(reason):
        return {"passes": False, "garment_match": False, "color_match": False,
                "pattern_match": False, "detected_color": "", "detected_pattern": "",
                "model_gender": "unclear", "rejection_reason": reason}

    def check_set_consistency(self, verifications):
        genders = [v["model_gender"] for v in verifications.values()
                   if v.get("model_gender") not in ("no_model", "unclear", None)]
        if len(genders) < 2:
            return {"passed": True, "gender_consistent": True, "note": "insufficient data"}
        unique = set(genders)
        consistent = len(unique) == 1
        return {"passed": consistent, "gender_consistent": consistent,
                "detected_genders": genders,
                "note": "" if consistent else f"mixed genders: {sorted(unique)}"}


# ──────────────────────────────────────────────────────────────
# AmazonCatalogIndex v3 — FAISS ANN
# ──────────────────────────────────────────────────────────────

class AmazonCatalogIndex:
    """
    FAISS inner-product index over FashionSigLIP title embeddings.

    오프라인 빌드: scripts/build_faiss_index.py
    온라인 검색:   self.search(query_text, top_k)

    인덱스 파일 구조 ($AMAZON_META_DIR/faiss/):
      amazon_fashion.index  — FAISS IndexFlatIP (L2-normalized)
      amazon_fashion_ids.json — [{"title":..., "image_url":..., "asin":..., ...}, ...]
    """

    _FAISS_SUBDIR = "faiss"
    _INDEX_FILE = "amazon_fashion.index"
    _IDS_FILE = "amazon_fashion_ids.json"

    def __init__(self, meta_dir=None):
        base = Path(
            meta_dir
            or os.environ.get("AMAZON_META_DIR", "/home/hjhj6411/fashion/data/amazon")
        )
        self._faiss_dir = base / self._FAISS_SUBDIR
        self._index = None
        self._items = []       # list of dict: {title, image_url, asin, ...}
        self._model = None
        self._tokenizer = None
        self._device = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Internal: lazy load index + model
    # ------------------------------------------------------------------

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._loaded = True

        index_path = self._faiss_dir / self._INDEX_FILE
        ids_path = self._faiss_dir / self._IDS_FILE

        if not index_path.exists() or not ids_path.exists():
            print(
                f"  [AmazonCatalogIndex] FAISS index not found at {self._faiss_dir}.\n"
                f"  Run: python scripts/build_faiss_index.py --meta-dir $AMAZON_META_DIR\n"
                f"  Falling back to empty index."
            )
            return

        try:
            import faiss
            self._index = faiss.read_index(str(index_path))
            with open(ids_path, encoding="utf-8") as f:
                self._items = json.load(f)
            print(f"  [AmazonCatalogIndex] Loaded {len(self._items):,} items from FAISS index.")
        except Exception as e:
            print(f"  [AmazonCatalogIndex] Failed to load index: {e}")
            return

        self._load_clip()

    def _load_clip(self):
        try:
            import torch
            import open_clip
            self._device = "cpu" #"cuda" if __import__("torch").cuda.is_available() else "cpu"
            model, _, _ = open_clip.create_model_and_transforms(
                "hf-hub:Marqo/marqo-fashionSigLIP")
            self._model = model.to(self._device).eval()
            self._tokenizer = open_clip.get_tokenizer(
                "hf-hub:Marqo/marqo-fashionSigLIP")
        except Exception as e:
            print(f"  [AmazonCatalogIndex] FashionSigLIP unavailable: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        텍스트 쿼리 → FashionSigLIP 인코딩 → FAISS ANN → top_k items.
        인덱스/모델 미로드 시 빈 리스트 반환 (Google fallback이 처리).
        """
        self._ensure_loaded()
        if self._index is None or self._model is None or not self._items:
            return []

        import torch
        import faiss

        tokens = self._tokenizer([query]).to(self._device)
        with torch.no_grad():
            q_emb = self._model.encode_text(tokens, normalize=True).cpu().numpy().astype("float32")
        faiss.normalize_L2(q_emb)

        k = min(top_k, self._index.ntotal)
        _, indices = self._index.search(q_emb, k)
        return [self._items[i] for i in indices[0] if 0 <= i < len(self._items)]


# ──────────────────────────────────────────────────────────────
# HomogeneityChecker (rank_candidates_by_clip 제거)
# ──────────────────────────────────────────────────────────────

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
            self._clip, _, self._proc = open_clip.create_model_and_transforms(
                "hf-hub:Marqo/marqo-fashionSigLIP")
            self._clip = self._clip.to(self._device).eval()
            self._tokenizer = open_clip.get_tokenizer(
                "hf-hub:Marqo/marqo-fashionSigLIP")
            self._torch = torch
        except Exception as e:
            print(f"  CLIP unavailable: {e}")
            self._clip = "DISABLED"

    def clip_distances(self, image_paths):
        self._ensure_clip()
        if self._clip == "DISABLED" or len(image_paths) < 2:
            return {"mean_distance": 0.0, "max_distance": 0.0, "passed": True, "note": "skipped"}
        try:
            imgs = self._torch.stack(
                [self._proc(Image.open(p).convert("RGB")) for p in image_paths]
            ).to(self._device)
            with self._torch.no_grad():
                feats = self._clip.encode_image(imgs, normalize=True)
            sims = (feats @ feats.T).cpu().numpy()
            dists = 1 - sims
            n = len(image_paths)
            pairs = [dists[i, j] for i in range(n) for j in range(i + 1, n)]
            mean_d, max_d = float(np.mean(pairs)), float(np.max(pairs))
            th = IMAGE_COLLECTION["max_clip_distance_within_options"]
            return {"mean_distance": mean_d, "max_distance": max_d,
                    "passed": max_d <= th, "threshold": th}
        except Exception as e:
            return {"mean_distance": 0.0, "max_distance": 0.0,
                    "passed": True, "note": f"failed: {e}"}

    def ssim_pair(self, p1, p2):
        try:
            from skimage.metrics import structural_similarity as ssim
            img1 = np.array(Image.open(p1).convert("L").resize((256, 256)))
            img2 = np.array(Image.open(p2).convert("L").resize((256, 256)))
            return float(ssim(img1, img2, data_range=255))
        except Exception:
            return -1.0


# ──────────────────────────────────────────────────────────────
# Utility helpers (unchanged)
# ──────────────────────────────────────────────────────────────

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
            timeout=20)
        resp.raise_for_status()
        return [{"image_url": it.get("link"), "title": it.get("title", ""),
                 "source": "google"} for it in resp.json().get("items", [])]
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


def _try_download_verified(candidates, img_path, attrs, verifier, top_n=5):
    tmp_path = img_path.with_suffix(".tmp.jpg")
    for cand in candidates[:top_n]:
        if not download_image(cand["image_url"], tmp_path):
            continue
        vr = verifier.verify(tmp_path, attrs)
        if vr["passes"]:
            tmp_path.rename(img_path)
            return True, cand.get("title", "")[:100], vr
        else:
            tmp_path.unlink(missing_ok=True)
    return False, "", {}


# ──────────────────────────────────────────────────────────────
# Per-plan collection (v2 두 브랜치 → 단일 semantic_search 흐름)
# ──────────────────────────────────────────────────────────────

def collect_for_plan(plan, amazon: AmazonCatalogIndex, checker: HomogeneityChecker,
                     verifier: AttributeVerifier, out_root: Path):
    query_id = plan["query_id"]
    out_dir = out_root / query_id
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "query_id": query_id,
        "user_id": plan["user_id"],
        "domain": "fashion",
        "active_axis": plan["active_axis"],
        "main_category": plan.get("main_category"),
        "options": {},
        "all_collected": False,
        "homogeneity": None,
        "structure_preserved": None,
        "set_consistency": None,
    }

    active_axis = plan["active_axis"]
    image_paths = []
    verifications = {}

    for k in "ABCD":
        opt = plan["options"][k]
        attrs = opt["attributes"]
        query = opt["search_query"]
        img_path = out_dir / f"{k}.jpg"

        # 캐시 히트
        if img_path.exists():
            image_paths.append(img_path)
            result["options"][k] = {
                "image_path": str(img_path),
                "source": "cached",
                "search_query": query,
            }
            verifications[k] = {"passes": True, "model_gender": "unclear"}
            continue

        # ① FAISS semantic search (keyword filter 없음, 전체 pool)
        cands = amazon.search(query, top_k=20)
        ok, src_title, vr = _try_download_verified(cands, img_path, attrs, verifier, top_n=10)
        src = "amazon_semantic"

        # ② Google fallback
        if not ok:
            gcands = google_image_search(query, top_k=5)
            ok, src_title, vr = _try_download_verified(gcands, img_path, attrs, verifier, 5)
            src = "google" if ok else "FAILED"

        if ok:
            result["options"][k] = {
                "image_path": str(img_path),
                "source": src,
                "search_query": query,
                "source_title": src_title,
                "verification": vr,
            }
            image_paths.append(img_path)
            verifications[k] = vr
        else:
            result["options"][k] = {
                "image_path": None,
                "source": "FAILED",
                "search_query": query,
            }

    result["all_collected"] = len(image_paths) == 4

    if result["all_collected"]:
        result["set_consistency"] = verifier.check_set_consistency(verifications)
        result["homogeneity"] = checker.clip_distances(image_paths)
        if active_axis in ("color", "pattern"):
            ssim_val = checker.ssim_pair(
                Path(result["options"]["A"]["image_path"]),
                Path(result["options"]["B"]["image_path"]),
            )
            th = IMAGE_COLLECTION["min_ssim_for_color_variants"]
            result["structure_preserved"] = {
                "ssim_A_vs_B": ssim_val,
                "threshold": th,
                "passed": (ssim_val < 0) or (ssim_val >= th),
            }

    return result


# ──────────────────────────────────────────────────────────────
# Pipeline runner (unchanged)
# ──────────────────────────────────────────────────────────────

def run_pipeline(plan_path, output_path, image_root, limit=0):
    log_step("Image Collector v3 (FAISS semantic retrieval + Google fallback)")
    plans = load_jsonl(plan_path)
    print(f"  Loaded {len(plans)} plans")

    amazon = AmazonCatalogIndex()
    checker = HomogeneityChecker()
    verifier = AttributeVerifier()

    if output_path.exists():
        existing = load_jsonl(output_path)
        done = {r["query_id"] for r in existing}
        results = existing
        todo = [p for p in plans if p["query_id"] not in done]
        print(f"  Resuming: {len(done)} collected")
    else:
        results = []
        todo = plans

    if limit > 0:
        todo = todo[:limit]

    for i, plan in enumerate(todo):
        print(f"  [{i+1}/{len(todo)}] {plan['query_id']} ({plan['active_axis']})")
        try:
            r = collect_for_plan(plan, amazon, checker, verifier, out_root=image_root)
            results.append(r)
            if (i + 1) % 10 == 0:
                save_jsonl(results, output_path)
        except Exception as e:
            print(f"    ERROR: {e}")
        time.sleep(0.5)

    save_jsonl(results, output_path)
    n_complete = sum(1 for r in results if r["all_collected"])
    print(f"\n  ✓ Saved {len(results)}: complete={n_complete}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan_path", type=Path, default=OPTIONS_DIR / "option_plans.jsonl")
    parser.add_argument("--output", type=Path, default=IMAGES_DIR / "collection_log.jsonl")
    parser.add_argument("--image_root", type=Path, default=IMAGES_DIR)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.force and args.output.exists():
        args.output.unlink()
    run_pipeline(args.plan_path, args.output, args.image_root, args.limit)


if __name__ == "__main__":
    main()