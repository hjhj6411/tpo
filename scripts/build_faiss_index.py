"""
build_faiss_index.py — Multi-Stage Image Filtering Pipeline

Stage 0: 메타데이터 기반 coarse filter  (GARMENT_AMAZON_CATS + TITLE_BLOCKLIST)
Stage 1: title–image CLIP score filter  (LAION style)
Stage 2: pure-image 구조 필터           (DataComp/OmniCorpus style)
Stage 3: 카테고리별 균형 샘플링 + dedup  (DataComp style)

출력 구조 ($OUTPUT_DIR/):
  blazer/  index.faiss + ids.json
  coat/    index.faiss + ids.json
  dress/   index.faiss + ids.json
  ...      (garment_key별 sub-index)

Usage:
  python scripts/build_faiss_index.py \\
    --meta-dir $AMAZON_META_DIR \\
    --output-dir $AMAZON_META_DIR/faiss \\
    --batch-size 256 \\
    --clip-score-threshold 0.20 \\
    --max-per-garment 30000 \\
    --garment blazer coat dress        # 특정 garment만 (생략시 전체)
"""

import argparse
import io
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests
import torch
import open_clip
import faiss
from PIL import Image


# ──────────────────────────────────────────────────────────────
# Stage 0 — garment taxonomy (메타데이터 기반 coarse filter)
# ──────────────────────────────────────────────────────────────

GARMENT_AMAZON_CATS: dict[str, set[str]] = {
    "t_shirt":     {"T-Shirts", "Graphic Tees", "Tops & Tees"},
    "polo_shirt":  {"Polos", "Polo Shirts"},
    "shirt":       {"Dress Shirts", "Casual Button-Down Shirts", "Shirts"},
    "blouse":      {"Blouses & Button-Down Shirts", "Blouses", "Tops, Tees & Blouses"},
    "sweater":     {"Sweaters", "Pullovers", "Cardigans", "Turtlenecks"},
    "hoodie":      {"Hoodies", "Sweatshirts", "Fleece"},
    "suit_vest":   {"Suit Vests", "Waistcoats"},
    "jacket":      {"Jackets", "Denim Jackets", "Leather Jackets", "Vests"},
    "pea_coat":    {"Pea Coats", "Peacoats"},
    "long_coat":   {"Overcoats", "Wool & Blends"},
    "coat":        {"Coats", "Wool & Blends", "Overcoats"},
    "blazer":      {"Blazers", "Suit Jackets", "Suits & Blazers"},
    "fleece":       {"Fleece", "Fleeces"},
    "windbreaker": {"Windbreakers", "Rain Jackets", "Anoraks"},
    "dress":       {"Dresses", "Maxi Dresses", "Mini Dresses", "Midi Dresses",
                    "Evening Gowns", "Cocktail Dresses"},
    "skirt":       {"Skirts", "A-Line Skirts", "Pencil Skirts", "Mini Skirts"},
    "pants":       {"Pants", "Dress Pants", "Casual Pants", "Trousers", "Chinos"},
    "jeans":       {"Jeans", "Skinny Jeans", "Straight-Leg Jeans", "Bootcut Jeans"},
    "shorts":      {"Shorts", "Bermuda Shorts", "Denim Shorts"},
    "suit_jacket": {"Suits", "Two-Piece Suits", "Three-Piece Suits", "Sport Coats"},
    "tank_top":    {"Tank Tops", "Camisoles", "Halter Tops", "Sleeveless Tops"},
}

GARMENT_TITLE_KEYWORDS: dict[str, list[str]] = {
    "t_shirt":     ["t-shirt", "tshirt", "tee shirt", "graphic tee"],
    "polo_shirt":  ["polo shirt", "polo"],
    "shirt":       ["shirt", "button-down", "button down", "oxford"],
    "blouse":      ["blouse"],
    "sweater":     ["sweater", "pullover", "cardigan", "knitwear", "knit top"],
    "hoodie":      ["hoodie", "hooded sweatshirt", "sweatshirt"],
    "suit_vest":   ["suit vest", "waistcoat"],
    "jacket":      ["jacket", "denim jacket", "leather jacket"],
    "pea_coat":    ["pea coat", "peacoat"],
    "long_coat":   ["long coat"],
    # generic "coat" bucket: keep only if a coarse class is genuinely needed.
    # It must never be used to populate long_coat cells.
    "coat":        ["coat"],
    "blazer":      ["blazer", "suit jacket"],
    "fleece":       ["fleece"],
    "windbreaker": ["windbreaker", "wind jacket", "rain jacket", "anorak"],
    "dress":       ["dress", "gown", "sundress"],
    "skirt":       ["skirt"],
    "pants":       ["pants", "trousers", "chinos", "slacks"],
    "jeans":       ["jeans", "denim pants"],
    "shorts":      ["shorts"],
    "suit_jacket": ["suit", "two-piece", "three-piece", "sport coat"],
    "tank_top":    ["tank top", "camisole", "halter top", "sleeveless top"],
}

TITLE_BLOCKLIST = {
    "mask", "face covering", "glove", "gloves", "sock", "socks",
    "underwear", "bra", "brief", "panty", "panties", "diaper",
    "pm2.5", "filter", "hat", "cap", "scarf", "belt",
    "shoe", "shoes", "boot", "boots", "sneaker", "sandal",
    "legging", "tights", "stockings", "shapewear", "swimsuit", "bikini",
    "gym wear", "athletic wear", "yoga pants", "activewear",
}


def _classify_garment(item: dict) -> str | None:
    """
    Stage 0: 아이템을 garment key로 분류.
    우선순위: Amazon category 매칭 > title keyword 매칭
    반환: garment_key 또는 None (분류 불가)
    """
    title_lower = (item.get("title") or "").lower()

    # 블랙리스트 먼저 확인
    for block in TITLE_BLOCKLIST:
        if block in title_lower:
            return None

    # categories 필드에서 Amazon 카테고리 매칭
    cats = item.get("categories") or []
    if isinstance(cats, str):
        cats = [cats]
    cat_set = {c.strip() for c in cats}

    for garment_key, amazon_cats in GARMENT_AMAZON_CATS.items():
        if cat_set & amazon_cats:  # 교집합 있으면 매칭
            return garment_key

    # fallback: title keyword
    for garment_key, keywords in GARMENT_TITLE_KEYWORDS.items():
        for kw in keywords:
            if kw in title_lower:
                return garment_key

    return None


# ──────────────────────────────────────────────────────────────
# Stage 1 — title–image CLIP score filter (LAION style)
# ──────────────────────────────────────────────────────────────

def _compute_clip_score(model, tokenizer, preprocess, device: str,
                        title: str, img_url: str) -> float:
    """
    FashionSigLIP으로 title–image cosine similarity 계산.
    LAION 방식: score < threshold → 메타와 이미지 불일치로 제거.
    실패 시 -1.0 반환 (필터에서 제거 대상).
    """
    try:
        resp = requests.get(img_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        tokens = tokenizer([title[:77]]).to(device)
        with torch.no_grad():
            img_feat = model.encode_image(img_tensor, normalize=True)
            txt_feat = model.encode_text(tokens, normalize=True)
        score = float((img_feat * txt_feat).sum())
        return score
    except Exception:
        return -1.0


# ──────────────────────────────────────────────────────────────
# Stage 2 — pure-image 구조 필터 (DataComp/OmniCorpus style)
# ──────────────────────────────────────────────────────────────

def _passes_image_structure(img_url: str,
                             min_size: int = 256,
                             min_ratio: float = 0.5,
                             max_ratio: float = 2.0) -> tuple[bool, Image.Image | None]:
    """
    DataComp/OmniCorpus 스타일 pure-image 구조 검사.
    1. 해상도: min(h,w) >= min_size
    2. 비율:   min_ratio <= w/h <= max_ratio  (배너/긴 광고 제거)
    3. 색 다양성: unique colors(k-means 10) >= 3  (단색 더미 이미지 제거)
                  AND <= 8  (콜라주/광고 이미지 제거 — 너무 많은 색 영역)

    반환: (passes, PIL.Image or None)
    """
    try:
        resp = requests.get(img_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return False, None

    w, h = img.size

    # 해상도 체크
    if min(w, h) < min_size:
        return False, None

    # 비율 체크
    ratio = w / h
    if not (min_ratio <= ratio <= max_ratio):
        return False, None

    # 색 다양성 체크 (k-means 대신 빠른 근사: 64x64 리사이즈 후 unique pixel count)
    small = img.resize((64, 64))
    arr = np.array(small).reshape(-1, 3)
    # 4비트 양자화로 색 수 근사
    quantized = (arr >> 4).astype(np.uint8)
    unique_colors = len(set(map(tuple, quantized)))
    if unique_colors < 3:   # 단색 더미 이미지
        return False, None
    if unique_colors > 800: # 콜라주/광고 (64x64=4096px 기준 ~20% 이상 고유색)
        return False, None

    return True, img


# ──────────────────────────────────────────────────────────────
# Meta streaming
# ──────────────────────────────────────────────────────────────

def iter_meta(meta_dir: Path):
    """JSONL meta 파일 스트리밍. categories 필드 포함해서 yield."""
    meta_files = sorted(meta_dir.glob("meta_*.jsonl"))
    if not meta_files:
        meta_files = sorted(meta_dir.glob("*.jsonl"))
    for mf in meta_files:
        with open(mf, encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    title = (item.get("title") or "").strip()
                    images = item.get("images") or item.get("image") or []
                    img_url = None
                    if isinstance(images, list) and images:
                        first = images[0]
                        img_url = (
                            (first.get("large") or first.get("hi_res") or first.get("thumb"))
                            if isinstance(first, dict) else
                            (first if isinstance(first, str) else None)
                        )
                    elif isinstance(images, str):
                        img_url = images

                    # categories: list 또는 string
                    cats = item.get("categories") or item.get("category") or []
                    if isinstance(cats, str):
                        cats = [cats]

                    if title and img_url:
                        yield {
                            "title": title,
                            "image_url": img_url,
                            "asin": item.get("asin"),
                            "parent_asin": item.get("parent_asin"),
                            "main_category": item.get("main_category", ""),
                            "categories": cats,
                        }
                except Exception:
                    continue


# ──────────────────────────────────────────────────────────────
# Main: per-garment sub-index builder
# ──────────────────────────────────────────────────────────────

def build_garment_index(
    garment_key: str,
    items: list[dict],
    model,
    tokenizer,
    preprocess,
    device: str,
    output_dir: Path,
    batch_size: int,
    clip_score_threshold: float,
    max_per_garment: int,
    run_stage1: bool,
    run_stage2: bool,
):
    """
    단일 garment에 대한 인덱스 빌드.
    Stage 1 (CLIP score) + Stage 2 (image structure) 통과 아이템만 인덱싱.
    """
    out_dir = output_dir / garment_key
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.faiss"
    ids_path = out_dir / "ids.json"

    print(f"  [{garment_key}] {len(items):,} candidates after Stage 0")

    # ── Stage 1: title–image CLIP score filter ──────────────────
    passed_s1 = []
    if run_stage1:
        scores = []
        for i, item in enumerate(items):
            score = _compute_clip_score(
                model, tokenizer, preprocess, device,
                item["title"], item["image_url"]
            )
            scores.append(score)
            if (i + 1) % 500 == 0:
                print(f"    Stage1 {i+1}/{len(items)} scored...")

        # threshold: max(fixed_threshold, 20th percentile) — DataComp 스타일 상대 컷
        valid_scores = [s for s in scores if s >= 0]
        if valid_scores:
            percentile_20 = float(np.percentile(valid_scores, 20))
            effective_threshold = max(clip_score_threshold, percentile_20)
        else:
            effective_threshold = clip_score_threshold

        passed_s1 = [
            (item, scores[i])
            for i, item in enumerate(items)
            if scores[i] >= effective_threshold
        ]
        # CLIP score 내림차순 정렬 (품질 상위 우선)
        passed_s1.sort(key=lambda x: x[1], reverse=True)
        print(f"  [{garment_key}] Stage 1 passed: {len(passed_s1):,} "
              f"(threshold={effective_threshold:.3f})")
    else:
        passed_s1 = [(item, 1.0) for item in items]

    # max_per_garment 제한 (Stage 2 전에 상위 N개만 처리해서 속도 향상)
    pool = passed_s1[:max_per_garment * 3]  # Stage 2 손실 감안해 3× 여유

    # ── Stage 2: pure-image 구조 필터 ──────────────────────────
    passed_s2 = []
    if run_stage2:
        for i, (item, score) in enumerate(pool):
            ok, _ = _passes_image_structure(item["image_url"])
            if ok:
                passed_s2.append((item, score))
            if len(passed_s2) >= max_per_garment:
                break
            if (i + 1) % 500 == 0:
                print(f"    Stage2 {i+1}/{len(pool)}, passed={len(passed_s2)}...")
        print(f"  [{garment_key}] Stage 2 passed: {len(passed_s2):,}")
    else:
        passed_s2 = pool[:max_per_garment]

    final_items = [item for item, _ in passed_s2[:max_per_garment]]

    if not final_items:
        print(f"  [{garment_key}] ⚠ No items passed. Skipping.")
        return

    # ── Stage 3: near-duplicate 제거 (LAION style, title-level) ─
    # title embedding으로 cosine > 0.98인 중복 제거
    print(f"  [{garment_key}] Stage 3: dedup over {len(final_items):,} items...")
    titles_trunc = [it["title"][:77] for it in final_items]
    all_embs = []
    for i in range(0, len(titles_trunc), batch_size):
        batch = titles_trunc[i: i + batch_size]
        tokens = tokenizer(batch).to(device)
        with torch.no_grad():
            emb = model.encode_text(tokens, normalize=True).cpu().float().numpy()
        all_embs.append(emb)

    emb_matrix = np.concatenate(all_embs, axis=0).astype("float32")
    faiss.normalize_L2(emb_matrix)

    # 중복 제거: 각 벡터에서 가장 가까운 이웃이 0.98 이상이면 중복
    tmp_index = faiss.IndexFlatIP(emb_matrix.shape[1])
    tmp_index.add(emb_matrix)
    # top-2 검색 (자기 자신 + 최근접)
    k_nn = min(2, emb_matrix.shape[0])
    sims, idxs = tmp_index.search(emb_matrix, k_nn)

    keep_mask = np.ones(len(final_items), dtype=bool)
    for i in range(len(final_items)):
        if not keep_mask[i]:
            continue
        for j_rank in range(1, k_nn):  # 자기 자신(rank=0) 제외
            j = idxs[i, j_rank]
            if j != i and sims[i, j_rank] >= 0.98 and keep_mask[j]:
                # 인덱스가 큰 쪽을 제거 (score 낮은 쪽)
                keep_mask[max(i, j)] = False

    deduped_items = [item for item, keep in zip(final_items, keep_mask) if keep]
    print(f"  [{garment_key}] After dedup: {len(deduped_items):,} items")

    # ── FAISS 인덱스 빌드 ───────────────────────────────────────
    titles_final = [it["title"][:77] for it in deduped_items]
    final_embs = []
    for i in range(0, len(titles_final), batch_size):
        batch = titles_final[i: i + batch_size]
        tokens = tokenizer(batch).to(device)
        with torch.no_grad():
            emb = model.encode_text(tokens, normalize=True).cpu().float().numpy()
        final_embs.append(emb)

    emb_final = np.concatenate(final_embs, axis=0).astype("float32")
    faiss.normalize_L2(emb_final)

    dim = emb_final.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb_final)
    faiss.write_index(index, str(index_path))

    # garment_key 메타 추가 후 저장
    for it in deduped_items:
        it["garment_key"] = garment_key
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(deduped_items, f, ensure_ascii=False)

    print(f"  [{garment_key}] ✓ index saved ({index.ntotal:,} vectors, dim={dim})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-dir", type=Path,
                        default=Path(os.environ.get(
                            "AMAZON_META_DIR",
                            "/home/hjhj6411/fashion/data/amazon/fashion_only")))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--clip-score-threshold", type=float, default=0.20,
                        help="LAION style: title-image cosine similarity 하한")
    parser.add_argument("--max-per-garment", type=int, default=30000,
                        help="DataComp style: garment당 최대 아이템 수 (dedup 후)")
    parser.add_argument("--garment", nargs="+", default=None,
                        help="특정 garment_key만 빌드 (생략시 전체)")
    parser.add_argument("--no-stage1", action="store_true",
                        help="CLIP score filter 건너뜀 (테스트용)")
    parser.add_argument("--no-stage2", action="store_true",
                        help="image structure filter 건너뜀 (테스트용)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or (args.meta_dir / "faiss")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 모델 로드 (1회)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading FashionSigLIP on {device}...")
    model, preprocess, _ = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionSigLIP")
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionSigLIP")

    target_garments = set(args.garment) if args.garment else set(GARMENT_AMAZON_CATS.keys())

    # Stage 0: 전체 메타 스캔 → garment별 분류
    print(f"Stage 0: streaming meta from {args.meta_dir}...")
    garment_buckets: dict[str, list[dict]] = defaultdict(list)
    total = 0
    for item in iter_meta(args.meta_dir):
        total += 1
        gk = _classify_garment(item)
        if gk and gk in target_garments:
            garment_buckets[gk].append(item)
        if total % 100_000 == 0:
            print(f"  Scanned {total:,} items, classified: "
                  + ", ".join(f"{k}={len(v)}" for k, v in sorted(garment_buckets.items())))

    print(f"Stage 0 done. Total scanned: {total:,}")
    for gk, items in sorted(garment_buckets.items()):
        print(f"  {gk}: {len(items):,}")

    # garment별 인덱스 빌드 (Stage 1–3)
    for garment_key in sorted(target_garments):
        items = garment_buckets.get(garment_key, [])
        if not items:
            print(f"  [{garment_key}] No items found. Skipping.")
            continue

        out_dir = output_dir / garment_key
        if not args.overwrite and (out_dir / "index.faiss").exists():
            print(f"  [{garment_key}] Already built. Use --overwrite to rebuild.")
            continue

        build_garment_index(
            garment_key=garment_key,
            items=items,
            model=model,
            tokenizer=tokenizer,
            preprocess=preprocess,
            device=device,
            output_dir=output_dir,
            batch_size=args.batch_size,
            clip_score_threshold=args.clip_score_threshold,
            max_per_garment=args.max_per_garment,
            run_stage1=not args.no_stage1,
            run_stage2=not args.no_stage2,
        )

    print("\n✓ All garment indexes built.")


if __name__ == "__main__":
    main()
