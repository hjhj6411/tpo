#!/usr/bin/env python3
"""
amazon_to_clip_corpus.py
------------------------
Turn local Amazon Reviews 2023 *metadata* JSONL into a URL list (CSV) that
img2dataset -> clip-retrieval can consume to build an IMAGE index.

Amazon Reviews 2023 item-metadata schema (https://amazon-reviews-2023.github.io/):
  title (str), main_category (str), categories (list[str]),
  parent_asin (str), store (str), price, average_rating, rating_number,
  images: list[ {"thumb":url, "large":url, "hi_res":url, "variant":"MAIN"|"PT01"|...} ]
  videos, features, description, details, bought_together

This reads meta_*.jsonl (or *.jsonl) under --meta-dir, picks one product image
per item (MAIN/hi_res preferred), optionally filters to clothing, dedups by URL,
and writes a CSV with columns:  url, caption, asin, parent_asin, main_category, categories

Usage:
  python scripts/amazon_to_clip_corpus.py \
    --meta-dir /home1/hjhj6411/pod_bench/data/amazon/fashion_only \
    --out ./corpus/amazon_fashion_urls.csv \
    --image-pref hi_res --max-items 300000 --clothing-only

Then:
  img2dataset --url_list ./corpus/amazon_fashion_urls.csv --input_format csv \
    --url_col url --caption_col caption \
    --output_format webdataset --output_folder ./corpus/amazon_wds \
    --processes_count 16 --thread_count 64 --image_size 384 \
    --resize_mode keep_ratio --number_sample_per_shard 10000
"""

import argparse
import csv
import json
import os
from pathlib import Path

# Broad clothing keywords (used only if --clothing-only). Footwear/accessories
# are excluded so the index stays focused on garments the benchmark uses.
CLOTHING_KW = [
    "shirt", "t-shirt", "tshirt", "tee", "blouse", "sweater", "pullover",
    "cardigan", "hoodie", "sweatshirt", "jacket", "coat", "overcoat",
    "trench", "blazer", "parka", "puffer", "windbreaker", "anorak", "dress",
    "gown", "skirt", "pants", "trousers", "chinos", "slacks", "jeans",
    "shorts", "suit", "tank top", "camisole", "vest",
]
EXCLUDE_KW = [
    "shoe", "boot", "sneaker", "sandal", "sock", "glove", "hat", "cap",
    "scarf", "belt", "bag", "wallet", "watch", "jewelry", "necklace",
    "earring", "ring ", "sunglass", "underwear", "bra", "panty", "lingerie",
    "swimsuit", "bikini", "legging", "tights", "stocking",
]


def pick_image_url(images, pref="hi_res"):
    """Choose one product image URL from the Amazon 2023 images list."""
    if isinstance(images, str):
        return images
    if not isinstance(images, list) or not images:
        return None
    order = [pref, "hi_res", "large", "thumb"]

    def url_of(img):
        if isinstance(img, str):
            return img
        if isinstance(img, dict):
            for key in order:
                u = img.get(key)
                if u:
                    return u
        return None

    # prefer the MAIN variant if present
    for img in images:
        if isinstance(img, dict) and str(img.get("variant", "")).upper() == "MAIN":
            u = url_of(img)
            if u:
                return u
    for img in images:
        u = url_of(img)
        if u:
            return u
    return None


def iter_meta(meta_dir):
    meta_dir = Path(meta_dir)
    files = sorted(meta_dir.glob("meta_*.jsonl")) or sorted(meta_dir.glob("*.jsonl"))
    if not files:
        # also accept .json / .jsonl.gz style; keep it simple here
        files = sorted(meta_dir.glob("*.json"))
    for mf in files:
        with open(mf, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def is_clothing(title, categories):
    blob = (title + " " + " ".join(categories)).lower()
    if any(x in blob for x in EXCLUDE_KW):
        return False
    return any(x in blob for x in CLOTHING_KW)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-dir",
                    default=os.environ.get(
                        "AMAZON_META_DIR",
                        "/home1/hjhj6411/pod_bench/data/amazon/fashion_only"))
    ap.add_argument("--out", default="./corpus/amazon_fashion_urls.csv")
    ap.add_argument("--image-pref", default="hi_res",
                    choices=["hi_res", "large", "thumb"])
    ap.add_argument("--max-items", type=int, default=0, help="0 = no cap")
    ap.add_argument("--clothing-only", action="store_true",
                    help="keep only garment-like items (drop shoes/bags/etc.)")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    n_in = n_out = 0
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["url", "caption", "asin", "parent_asin",
                    "main_category", "categories"])
        for item in iter_meta(args.meta_dir):
            n_in += 1
            title = (item.get("title") or "").strip()
            if not title:
                continue
            cats = item.get("categories") or item.get("category") or []
            if isinstance(cats, str):
                cats = [cats]
            cats = [str(c) for c in cats]

            if args.clothing_only and not is_clothing(title, cats):
                continue

            url = pick_image_url(item.get("images") or item.get("image"),
                                 pref=args.image_pref)
            if not url or url in seen:
                continue
            seen.add(url)

            w.writerow([
                url,
                title[:300],
                item.get("asin", "") or "",
                item.get("parent_asin", "") or "",
                item.get("main_category", "") or "",
                " > ".join(cats)[:300],
            ])
            n_out += 1
            if args.max_items and n_out >= args.max_items:
                break
            if n_out % 50000 == 0:
                print(f"  written {n_out:,} (scanned {n_in:,})")

    print(f"\n  ✓ scanned {n_in:,} items, wrote {n_out:,} unique image URLs")
    print(f"  ✓ {out}")
    print("\n  Next: feed this CSV to img2dataset (see header docstring).")


if __name__ == "__main__":
    main()