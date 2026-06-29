#!/usr/bin/env python3
"""Visualize QwenEmb clothing-vs-background decisions on adaptive square patches.

Purpose:
  Before adding a heavy segmentation backend, this script checks whether the
  embedding model can judge "does this square patch contain clothing/fabric?".
  It writes an overlay image similar to a mask visualization, patch-level scores,
  and a small HTML report for manual inspection.

Expected QwenEmb server route:
  POST /score-image-files

Example:
  python retrieval/visualize_qwenemb_patch_garment.py \
    --image data/retrieval/.../rank_001.jpg \
    --client-url http://127.0.0.1:1236/knn-service \
    --output-root data/retrieval/patch_garment_viz_smoke \
    --short-side-tiles 10 \
    --overlap 0.25 \
    --threshold 0.0 \
    --force
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval.square_patch_utils import SquarePatchRecord, make_adaptive_square_patch_records  # noqa: E402

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

DEFAULT_POSITIVE_TEXTS = [
    "visible clothing fabric",
    "fashion garment textile",
    "a patch of clothes or wearable garment",
]

DEFAULT_NEGATIVE_TEXTS = [
    "background without clothing",
    "skin face hair hand or body part, not clothing fabric",
    "non fashion object scenery package text logo, not clothing",
]


def endpoint(base: str, route: str) -> str:
    base = base.rstrip("/")
    for suffix in ["/knn-service", "/score-image-files", "/score-candidates"]:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base + route


def slugify(text: Any, max_len: int = 96) -> str:
    s = str(text or "")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")
    return (s or "image")[:max_len]


def relpath(path: Path, base: Path) -> str:
    try:
        return os.path.relpath(path, start=base)
    except ValueError:
        return str(path)


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def download_url(url: str, out_dir: Path, timeout: float, retries: int) -> Path:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        suffix = ".jpg"
    out = out_dir / (slugify(Path(parsed.path).stem or url, 80) + suffix)
    out_dir.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return out
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers=DEFAULT_HEADERS, stream=True, timeout=timeout)
            r.raise_for_status()
            tmp = out.with_suffix(out.suffix + ".tmp")
            with tmp.open("wb") as f:
                for chunk in r.iter_content(1024 * 128):
                    if chunk:
                        f.write(chunk)
            tmp.replace(out)
            return out
        except Exception as e:
            last_err = e
            if attempt >= retries:
                break
            time.sleep(min(2.0 ** attempt, 8.0))
    raise RuntimeError(f"failed to download {url}: {last_err}")


def load_images(args: argparse.Namespace) -> list[Path]:
    images: list[Path] = []
    for p in args.image or []:
        path = Path(p).expanduser()
        if path.exists():
            images.append(path.resolve())
        else:
            raise FileNotFoundError(f"--image path does not exist: {path}")

    download_dir = args.output_root / "downloaded_inputs"
    for url in args.image_url or []:
        images.append(download_url(url, download_dir, args.download_timeout, args.retries).resolve())

    if args.results_jsonl:
        with args.results_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if args.max_images > 0 and len(images) >= args.max_images:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                local = row.get("local_path")
                if local and Path(local).exists():
                    images.append(Path(local).resolve())

    # stable de-duplication
    out: list[Path] = []
    seen: set[str] = set()
    for p in images:
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    if args.max_images > 0:
        out = out[: args.max_images]
    if not out:
        raise SystemExit("No input images. Use --image, --image-url, or --results-jsonl.")
    return out


def aggregate(values: list[float], mode: str) -> float:
    if not values:
        return 0.0
    if mode == "max":
        return max(values)
    return sum(values) / len(values)


def post_scores(
    session: requests.Session,
    score_url: str,
    texts: dict[str, str],
    image_paths: list[Path],
    timeout: float,
    retries: int,
) -> list[dict[str, Any]]:
    payload = {"texts": texts, "image_paths": [str(p) for p in image_paths]}
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = session.post(score_url, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and data.get("error"):
                raise RuntimeError(str(data["error"]))
            rows = data.get("scores") if isinstance(data, dict) else None
            if not isinstance(rows, list):
                raise RuntimeError(f"expected scores list, got {type(data).__name__}: {data}")
            return [x if isinstance(x, dict) else {} for x in rows]
        except Exception as e:
            last_err = e
            if attempt >= retries:
                break
            time.sleep(min(2.0 ** attempt, 8.0))
    raise RuntimeError(f"score request failed: {last_err}")


def score_records(records: list[SquarePatchRecord], args: argparse.Namespace) -> list[dict[str, Any]]:
    pos_texts = args.pos_text or DEFAULT_POSITIVE_TEXTS
    neg_texts = args.neg_text or DEFAULT_NEGATIVE_TEXTS
    texts: dict[str, str] = {}
    for i, t in enumerate(pos_texts):
        texts[f"pos_{i}"] = t
    for i, t in enumerate(neg_texts):
        texts[f"neg_{i}"] = t

    session = requests.Session()
    score_url = args.score_url or endpoint(args.client_url, "/score-image-files")
    paths = [r.path for r in records]
    score_rows: list[dict[str, Any]] = []
    for start in range(0, len(paths), args.batch_size):
        chunk = paths[start : start + args.batch_size]
        score_rows.extend(post_scores(session, score_url, texts, chunk, args.timeout, args.retries))

    out: list[dict[str, Any]] = []
    for rec, row in zip(records, score_rows):
        pos_vals = []
        neg_vals = []
        for i in range(len(pos_texts)):
            try:
                pos_vals.append(float(row.get(f"pos_{i}", 0.0)))
            except Exception:
                pos_vals.append(0.0)
        for i in range(len(neg_texts)):
            try:
                neg_vals.append(float(row.get(f"neg_{i}", 0.0)))
            except Exception:
                neg_vals.append(0.0)
        pos = aggregate(pos_vals, args.aggregate)
        neg = aggregate(neg_vals, args.aggregate)
        margin = pos - neg
        present = margin >= args.threshold
        obj = rec.to_json()
        obj.update({
            "positive_score": pos,
            "negative_score": neg,
            "margin": margin,
            "present": bool(present),
            "threshold": args.threshold,
            "aggregate": args.aggregate,
            "positive_texts": pos_texts,
            "negative_texts": neg_texts,
            "raw_scores": row,
        })
        out.append(obj)
    return out


def make_overlay(image_path: Path, scored: list[dict[str, Any]], out_path: Path, args: argparse.Namespace) -> None:
    base = Image.open(image_path).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")

    pos_alpha = max(0, min(255, int(round(args.positive_alpha * 255))))
    neg_alpha = max(0, min(255, int(round(args.negative_alpha * 255))))
    outline_alpha = max(0, min(255, int(round(args.outline_alpha * 255))))

    for row in scored:
        x = int(row["x"])
        y = int(row["y"])
        s = int(row["size"])
        box = [x, y, x + s, y + s]
        if row.get("present"):
            draw.rectangle(box, fill=(255, 0, 0, pos_alpha), outline=(255, 0, 0, outline_alpha), width=args.outline_width)
        elif args.show_negative:
            draw.rectangle(box, fill=(80, 80, 80, neg_alpha), outline=(80, 80, 80, outline_alpha), width=args.outline_width)

    out = Image.alpha_composite(base, layer).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path, quality=95)


def write_report(image_reports: list[dict[str, Any]], output_root: Path) -> Path:
    css = """
    body { font-family: Arial, sans-serif; margin: 24px; background: #fafafa; color: #222; }
    .image-block { background: white; border: 1px solid #ddd; border-radius: 10px; padding: 16px; margin: 18px 0; }
    .pair { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; }
    img.main { width: 100%; max-height: 520px; object-fit: contain; background: #eee; border-radius: 8px; }
    .meta, .score { color: #555; font-size: 13px; }
    .patches { display: grid; grid-template-columns: repeat(auto-fill, minmax(110px, 1fr)); gap: 8px; margin-top: 12px; }
    .patch { border: 1px solid #ddd; border-radius: 6px; padding: 5px; background: #fff; font-size: 11px; }
    .patch.present { border-color: #e00000; background: #fff5f5; }
    .patch img { width: 100%; aspect-ratio: 1/1; object-fit: cover; border-radius: 4px; }
    code { background: #f2f2f2; padding: 1px 3px; border-radius: 3px; }
    """
    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>QwenEmb patch garment visualization</title>",
        f"<style>{css}</style></head><body>",
        "<h1>QwenEmb patch garment visualization</h1>",
    ]
    for rep in image_reports:
        parts.append("<section class='image-block'>")
        parts.append(f"<h2>{html.escape(rep['name'])}</h2>")
        parts.append(
            "<div class='meta'>"
            f"patches={rep['num_patches']} · clothing-positive={rep['num_present']} "
            f"({rep['coverage']:.3f}) · patch_size={rep['patch_size']} · "
            f"threshold={rep['threshold']}"
            "</div>"
        )
        parts.append("<div class='pair'>")
        parts.append(f"<div><h3>Original</h3><img class='main' src='{html.escape(rep['original_rel'])}'></div>")
        parts.append(f"<div><h3>Overlay</h3><img class='main' src='{html.escape(rep['overlay_rel'])}'></div>")
        parts.append("</div>")
        parts.append(f"<p class='meta'>Scores: <a href='{html.escape(rep['scores_rel'])}'>patch_scores.jsonl</a></p>")
        parts.append("<div class='patches'>")
        for p in rep["patches"]:
            cls = "patch present" if p.get("present") else "patch"
            parts.append(f"<div class='{cls}'>")
            parts.append(f"<img src='{html.escape(p['path_rel'])}'>")
            parts.append(
                "<div class='score'>"
                f"#{p['index']} · {'cloth' if p.get('present') else 'non'}<br>"
                f"pos={p['positive_score']:.3f}<br>"
                f"neg={p['negative_score']:.3f}<br>"
                f"m={p['margin']:.3f}"
                "</div>"
            )
            parts.append("</div>")
        parts.append("</div>")
        parts.append("</section>")
    parts.append("</body></html>")
    out = output_root / "index.html"
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def process_image(image_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    stem = slugify(image_path.stem, 80)
    out_dir = args.output_root / stem
    if out_dir.exists() and args.force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    original_copy = out_dir / ("original" + image_path.suffix.lower())
    if not original_copy.exists() or args.force:
        shutil.copy2(image_path, original_copy)

    tile_dir = out_dir / "patches"
    records = make_adaptive_square_patch_records(
        image_path=image_path,
        grid=args.short_side_tiles,
        patch_size=args.patch_size,
        max_tiles=args.max_patches,
        tile_dir=tile_dir,
        save_tiles=True,
        overlap=args.overlap,
    )
    scored = score_records(records, args)

    scores_path = out_dir / "patch_scores.jsonl"
    if scores_path.exists():
        scores_path.unlink()
    append_jsonl(scores_path, scored)

    overlay_path = out_dir / "overlay.png"
    make_overlay(image_path, scored, overlay_path, args)

    num_present = sum(1 for r in scored if r.get("present"))
    num_patches = len(scored)
    patch_size = int(scored[0]["size"]) if scored else 0
    summary = {
        "image": str(image_path),
        "num_patches": num_patches,
        "num_present": num_present,
        "coverage": num_present / max(1, num_patches),
        "patch_size": patch_size,
        "threshold": args.threshold,
        "overlay": str(overlay_path),
        "scores": str(scores_path),
    }
    write_json(out_dir / "summary.json", summary)

    patches_for_html = []
    for row in scored:
        item = dict(row)
        item["path_rel"] = relpath(Path(row["path"]), args.output_root)
        patches_for_html.append(item)

    return {
        "name": image_path.name,
        "original_rel": relpath(original_copy, args.output_root),
        "overlay_rel": relpath(overlay_path, args.output_root),
        "scores_rel": relpath(scores_path, args.output_root),
        "num_patches": num_patches,
        "num_present": num_present,
        "coverage": num_present / max(1, num_patches),
        "patch_size": patch_size,
        "threshold": args.threshold,
        "patches": patches_for_html,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--image", action="append", default=[], help="Local image path. Can be repeated.")
    p.add_argument("--image-url", action="append", default=[], help="Image URL to download then visualize. Can be repeated.")
    p.add_argument("--results-jsonl", type=Path, default=None, help="Optional scored/raw retrieval jsonl; uses rows with existing local_path.")
    p.add_argument("--max-images", type=int, default=0)
    p.add_argument("--client-url", default="http://127.0.0.1:1236/knn-service")
    p.add_argument("--score-url", default=None, help="Override full /score-image-files URL.")
    p.add_argument("--output-root", type=Path, default=Path("data/retrieval/patch_garment_viz"))

    p.add_argument("--short-side-tiles", type=int, default=10, help="Number of square patches along the shorter side when --patch-size is not set.")
    p.add_argument("--patch-size", type=int, default=0, help="Fixed square patch side in pixels. Overrides --short-side-tiles when >0.")
    p.add_argument("--overlap", type=float, default=0.0, help="Patch overlap ratio in [0, 0.9].")
    p.add_argument("--max-patches", type=int, default=0, help="0 means all patches; positive values cap patches for debugging only.")

    p.add_argument("--pos-text", action="append", default=[], help="Positive clothing text prompt. Can be repeated.")
    p.add_argument("--neg-text", action="append", default=[], help="Negative non-clothing text prompt. Can be repeated.")
    p.add_argument("--aggregate", choices=["mean", "max"], default="mean")
    p.add_argument("--threshold", type=float, default=0.0, help="Patch is clothing-positive if positive_score - negative_score >= threshold.")

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--download-timeout", type=float, default=30.0)
    p.add_argument("--retries", type=int, default=2)

    p.add_argument("--positive-alpha", type=float, default=0.45)
    p.add_argument("--negative-alpha", type=float, default=0.00)
    p.add_argument("--outline-alpha", type=float, default=0.85)
    p.add_argument("--outline-width", type=int, default=2)
    p.add_argument("--show-negative", action="store_true", help="Also draw faint non-clothing patches.")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    if args.force and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.patch_size <= 0:
        args.patch_size = None

    images = load_images(args)
    print("=" * 80)
    print("  QwenEmb patch clothing/background visualization")
    print("=" * 80)
    print(f"  images:      {len(images)}")
    print(f"  score_url:   {args.score_url or endpoint(args.client_url, '/score-image-files')}")
    print(f"  output_root: {args.output_root}")
    print(f"  tiling:      short_side_tiles={args.short_side_tiles}, patch_size={args.patch_size}, overlap={args.overlap}")
    print(f"  threshold:   {args.threshold}")

    reports = []
    for i, image_path in enumerate(images, start=1):
        print(f"[{i}/{len(images)}] {image_path}")
        rep = process_image(image_path, args)
        reports.append(rep)
        print(f"  patches={rep['num_patches']} clothing={rep['num_present']} coverage={rep['coverage']:.3f}")

    index = write_report(reports, args.output_root)
    write_json(args.output_root / "run_config.json", vars(args))
    print("\nDone.")
    print(f"  report: {index}")


if __name__ == "__main__":
    main()
