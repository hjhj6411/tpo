"""
오프라인 1회 실행. Amazon meta JSONL → FashionSigLIP title 임베딩 → FAISS 인덱스 저장.

Usage:
  python scripts/build_faiss_index.py \
    --meta-dir /home/hjhj6411/fashion/data/amazon/fashion_only \
    --output-dir /home/hjhj6411/fashion/data/amazon/fashion_only/faiss \
    --batch-size 512
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import open_clip
import faiss


def iter_meta(meta_dir: Path):
    """JSONL meta 파일에서 {title, image_url, asin, parent_asin} 스트리밍."""
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
                    if title and img_url:
                        yield {
                            "title": title,
                            "image_url": img_url,
                            "asin": item.get("asin"),
                            "parent_asin": item.get("parent_asin"),
                            "category": item.get("main_category", ""),
                        }
                except Exception:
                    continue


def build_index(meta_dir: Path, output_dir: Path, batch_size: int = 512):
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "amazon_fashion.index"
    ids_path = output_dir / "amazon_fashion_ids.json"

    if index_path.exists() and ids_path.exists():
        print(f"Index already exists at {output_dir}. Use --overwrite to rebuild.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading FashionSigLIP on {device}...")
    model, _, _ = open_clip.create_model_and_transforms("hf-hub:Marqo/marqo-fashionSigLIP")
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionSigLIP")

    # 스트리밍 수집
    print(f"Streaming meta from {meta_dir}...")
    all_items = list(iter_meta(meta_dir))
    print(f"  {len(all_items):,} items found")

    # 배치 인코딩
    all_embs = []
    titles = [it["title"][:77] for it in all_items]
    for i in range(0, len(titles), batch_size):
        batch = titles[i : i + batch_size]
        tokens = tokenizer(batch).to(device)
        with torch.no_grad():
            emb = model.encode_text(tokens, normalize=True).cpu().float().numpy()
        all_embs.append(emb)
        if (i // batch_size) % 20 == 0:
            print(f"  Encoded {min(i + batch_size, len(titles)):,} / {len(titles):,}")

    emb_matrix = np.concatenate(all_embs, axis=0).astype("float32")
    faiss.normalize_L2(emb_matrix)  # inner product = cosine

    # FAISS 인덱스 빌드
    dim = emb_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb_matrix)
    faiss.write_index(index, str(index_path))
    print(f"  FAISS index saved: {index_path}  ({index.ntotal:,} vectors, dim={dim})")

    # ID 매핑 저장
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False)
    print(f"  IDs saved: {ids_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meta-dir",
        type=Path,
        default=Path(os.environ.get("AMAZON_META_DIR",
                                    "/home/hjhj6411/fashion/data/amazon/fashion_only")),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or (args.meta_dir / "faiss")

    if args.overwrite:
        for f in output_dir.glob("amazon_fashion*"):
            f.unlink()

    build_index(args.meta_dir, output_dir, args.batch_size)


if __name__ == "__main__":
    main()