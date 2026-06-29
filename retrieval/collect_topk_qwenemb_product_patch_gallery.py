#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import math
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import requests
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.collect_topk_scored_gallery import (
    OPTION_LABELS,
    append_jsonl,
    build_gallery,
    collect_one,
    grouped_rerank,
    load_jsonl,
    make_tasks,
    write_json,
)
from retrieval.visualize_birefnet_patch_mask import (
    load_model,
    mask_coverage,
    pad_mask,
    predict_mask,
    square_grid_with_padding,
)


def endpoint(base: str, route: str) -> str:
    base = base.rstrip("/")
    for suffix in ["/knn-service", "/score-image-files", "/score-candidates"]:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base + route


def product_query(text: str, suffix: str) -> str:
    text = str(text or "").strip()
    suffix = str(suffix or "").strip()
    return f"{text}, {suffix}" if suffix else text


def request_json(
    session: requests.Session,
    url: str,
    payload: dict[str, Any],
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = session.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and data.get("error"):
                raise RuntimeError(str(data["error"]))
            if not isinstance(data, dict):
                raise RuntimeError(f"expected dict response, got {type(data).__name__}")
            return data
        except Exception as e:
            last_err = e
            if attempt >= retries:
                break
            time.sleep(min(2.0 ** attempt, 8.0))
    raise RuntimeError(f"request failed: {last_err}")


def faiss_index(rec: dict[str, Any]) -> int | None:
    for obj in [rec, rec.get("raw") if isinstance(rec.get("raw"), dict) else None]:
        if not isinstance(obj, dict):
            continue
        if obj.get("faiss_index") is not None:
            try:
                return int(obj["faiss_index"])
            except Exception:
                return None
    return None


def axis_scores(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    session: requests.Session,
) -> dict[int, dict[str, float]]:
    indices: list[int] = []
    for rec in rows:
        idx = faiss_index(rec)
        rec["faiss_index"] = idx
        if idx is not None:
            indices.append(idx)

    if not indices:
        return {}

    head = rows[0]
    color = str(head.get("target_color") or "").strip()
    garment = str(head.get("target_garment") or "").strip()

    data = request_json(
        session,
        args.axis_score_url,
        {
            "texts": {
                "color_score": color,
                "garment_score": garment,
            },
            "indices": indices,
        },
        args.score_timeout,
        args.retries,
    )

    out: dict[int, dict[str, float]] = {}
    for row in data.get("scores") or []:
        if not isinstance(row, dict) or row.get("faiss_index") is None:
            continue
        idx = int(row["faiss_index"])
        out[idx] = {
            "color_score": float(row.get("color_score", 0.0)),
            "garment_score": float(row.get("garment_score", 0.0)),
        }
    return out


def pattern_texts(pattern: str) -> tuple[str, str]:
    pattern = str(pattern or "").replace("_", " ").strip().lower()
    if not pattern:
        return "", ""
    if pattern == "solid":
        return "plain solid clothing fabric", "visible patterned clothing fabric"
    return f"{pattern} clothing fabric", "plain solid clothing fabric or background"


def make_clothing_patch_tiles(
    image_path: Path,
    mask: Image.Image,
    args: argparse.Namespace,
    out_dir: Path,
) -> list[dict[str, Any]]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    patches, padded_w, padded_h = square_grid_with_padding(
        width=width,
        height=height,
        short_side_tiles=args.short_side_tiles,
    )
    padded_mask = pad_mask(mask, padded_w, padded_h)

    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for i, (x, y, s) in enumerate(patches):
        cov = mask_coverage(padded_mask, x, y, s, args.mask_threshold)
        keep = cov >= args.min_coverage
        if not keep:
            continue

        # Intentional simplification: padded area is not saved as tiles.
        # Ceiling: right/bottom edge patches may be smaller after crop.
        # Upgrade path: save padded RGB image too if fixed-size edge tiles become necessary.
        tile = image.crop((x, y, min(x + s, width), min(y + s, height)))
        tile_path = out_dir / f"tile_{i:04d}__xy_{x}_{y}__s_{s}__cov_{cov:.3f}.jpg"
        tile.save(tile_path, quality=92)

        rows.append({
            "index": i,
            "tile": str(tile_path),
            "x": x,
            "y": y,
            "size": s,
            "mask_coverage": cov,
        })
    return rows


def score_pattern_with_qwenemb(
    image_path: Path,
    pattern: str,
    rec_id: str,
    birefnet_model,
    args: argparse.Namespace,
    session: requests.Session,
) -> dict[str, Any]:
    pattern = str(pattern or "").replace("_", " ").strip().lower()
    if not pattern:
        return {"score": 1.0, "mode": "missing_pattern_target", "tile_scores": []}

    image = Image.open(image_path).convert("RGB")
    mask = predict_mask(
        model=birefnet_model,
        image=image,
        device=args.device,
        input_size=args.input_size,
        fp16=args.fp16,
    )

    mask_dir = args.output_root / "masks" / rec_id
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask.save(mask_dir / "mask.png")

    tiles = make_clothing_patch_tiles(
        image_path=image_path,
        mask=mask,
        args=args,
        out_dir=args.output_root / "tiles" / rec_id,
    )

    if not tiles:
        return {
            "score": 0.0,
            "mode": "no_clothing_patch_after_birefnet_mask",
            "tile_scores": [],
        }

    pos_text, neg_text = pattern_texts(pattern)
    data = request_json(
        session,
        args.image_score_url,
        {
            "texts": {
                "pattern": pos_text,
                "negative": neg_text,
            },
            "image_paths": [t["tile"] for t in tiles],
        },
        args.image_score_timeout,
        args.retries,
    )

    tile_scores: list[dict[str, Any]] = []
    for tile, row in zip(tiles, data.get("scores") or []):
        if not isinstance(row, dict):
            row = {}
        pos = float(row.get("pattern", 0.0))
        neg = float(row.get("negative", 0.0))
        margin = pos - neg
        present = margin >= args.pattern_margin

        tile_scores.append({
            **tile,
            "pattern_text": pos_text,
            "negative_text": neg_text,
            "pattern_similarity": pos,
            "negative_similarity": neg,
            "margin": margin,
            "present": present,
        })

    denom = max(1, len(tile_scores))
    hits = sum(1 for t in tile_scores if t["present"])
    return {
        "score": hits / denom,
        "mode": "birefnet_masked_qwenemb_patch_coverage",
        "hits": hits,
        "num_tiles": len(tile_scores),
        "pattern_text": pos_text,
        "negative_text": neg_text,
        "margin_threshold": args.pattern_margin,
        "avg_margin": sum(float(t["margin"]) for t in tile_scores) / denom,
        "avg_pattern_similarity": sum(float(t["pattern_similarity"]) for t in tile_scores) / denom,
        "avg_negative_similarity": sum(float(t["negative_similarity"]) for t in tile_scores) / denom,
        "tile_scores": tile_scores,
    }


def score_group(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    birefnet_model,
) -> list[dict[str, Any]]:
    session = requests.Session()

    try:
        axes = axis_scores(rows, args, session)
        axis_error = None
    except Exception as e:
        axes = {}
        axis_error = str(e)

    scored: list[dict[str, Any]] = []
    for rec in rows:
        local = rec.get("local_path")
        idx = rec.get("faiss_index")
        one = axes.get(int(idx), {}) if idx is not None else {}

        color_score = float(one.get("color_score", 0.0))
        garment_score = float(one.get("garment_score", 0.0))

        if local and Path(local).exists():
            try:
                rec_id = (
                    f"{rec.get('query_id')}__{rec.get('option_label')}"
                    f"__rank_{int(rec.get('original_rank') or 0):03d}"
                )
                pinfo = score_pattern_with_qwenemb(
                    image_path=Path(local),
                    pattern=str(rec.get("target_pattern") or ""),
                    rec_id=rec_id,
                    birefnet_model=birefnet_model,
                    args=args,
                    session=session,
                )
                pattern_score = float(pinfo.get("score") or 0.0)
            except Exception as e:
                pinfo = {
                    "score": 0.0,
                    "mode": "pattern_score_failed",
                    "error": str(e),
                    "tile_scores": [],
                }
                pattern_score = 0.0
        else:
            pinfo = {
                "score": 0.0,
                "mode": "missing_local_image",
                "tile_scores": [],
            }
            pattern_score = 0.0

        combined = math.sqrt(max(0.0, pattern_score * color_score * garment_score))

        rec.update({
            "retrieval_query": rec.get("option_text"),
            "pattern_score": pattern_score,
            "color_score": color_score,
            "garment_score": garment_score,
            "combined_score": combined,
            "rerank_formula": "sqrt(max(0, qwenemb_pattern_patch_coverage * qwenemb_color_ip * qwenemb_garment_ip))",
            "pattern_score_source": "BiRefNet clothing-mask patches + QwenEmb pattern similarity coverage",
            "color_score_source": "QwenEmb text-image similarity",
            "garment_score_source": "QwenEmb text-image similarity",
            "pattern_score_raw": pinfo,
            "axis_score_error": axis_error,
        })
        scored.append(rec)

    return scored


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--plan-path", type=Path, default=Path("data/options/option_plans.jsonl"))
    p.add_argument("--client-url", default="http://127.0.0.1:1236/knn-service")
    p.add_argument("--axis-score-url", default=None)
    p.add_argument("--image-score-url", default=None)
    p.add_argument("--index-name", default="pod_qwenemb")
    p.add_argument("--output-root", type=Path, default=Path("data/retrieval/qwenemb_product_patch_gallery"))

    p.add_argument("--product-query-suffix", default="studio product shot, isolated clothing item, plain white background, product catalog photo")
    p.add_argument("--retrieval-k", type=int, default=100)
    p.add_argument("--score-top-n", type=int, default=100)
    p.add_argument("--show-k", type=int, default=100)

    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--options", default="A,B,C,D")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--score-workers", type=int, default=1)

    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--download-timeout", type=float, default=30.0)
    p.add_argument("--score-timeout", type=float, default=120.0)
    p.add_argument("--image-score-timeout", type=float, default=120.0)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--max-image-bytes", type=int, default=20_000_000)

    p.add_argument("--birefnet-model-id", default="ZhengPeng7/BiRefNet")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--input-size", type=int, default=1024)
    p.add_argument("--fp16", action="store_true")

    p.add_argument("--short-side-tiles", type=int, default=12)
    p.add_argument("--mask-threshold", type=int, default=32)
    p.add_argument("--min-coverage", type=float, default=0.05)
    p.add_argument("--pattern-margin", type=float, default=0.0)

    p.add_argument("--force", action="store_true")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.axis_score_url = args.axis_score_url or endpoint(args.client_url, "/score-candidates")
    args.image_score_url = args.image_score_url or endpoint(args.client_url, "/score-image-files")

    options = [x.strip().upper() for x in args.options.split(",") if x.strip()]
    bad = [x for x in options if x not in OPTION_LABELS]
    if bad:
        raise SystemExit(f"Unsupported option labels: {bad}; use subset of A,B,C,D")

    plans = load_jsonl(args.plan_path)
    if args.offset:
        plans = plans[args.offset:]
    if args.limit > 0:
        plans = plans[:args.limit]

    tasks = [
        replace(t, option_text=product_query(t.option_text, args.product_query_suffix))
        for t in make_tasks(plans, options)
    ]

    print("=" * 80)
    print("  QwenEmb product-shot retrieval + BiRefNet patch pattern coverage")
    print("=" * 80)
    print(f"  plans:        {len(plans)}")
    print(f"  option tasks: {len(tasks)}")
    print(f"  retrieval_k:  {args.retrieval_k}")
    print(f"  score_top_n:  {args.score_top_n or 'all'}")
    print(f"  show_k:       {args.show_k}")
    print(f"  query suffix: {args.product_query_suffix}")
    print(f"  endpoint:     {args.client_url}")
    print(f"  axis scores:  {args.axis_score_url}")
    print(f"  image scores: {args.image_score_url}")
    print(f"  output:       {args.output_root}")

    if args.dry_run:
        for t in tasks[:20]:
            print(f"{t.query_id} {t.option_label}: {t.option_text}")
        return

    if args.force and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    raw_log = args.output_root / "raw_topk_results.jsonl"
    scored_log = args.output_root / "scored_results.jsonl"
    for path in [raw_log, scored_log]:
        if path.exists():
            path.unlink()

    write_json(args.output_root / "run_config.json", vars(args) | {"options_list": options})

    raw_records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(collect_one, t, args): t for t in tasks}
        for done, fut in enumerate(cf.as_completed(futs), start=1):
            task = futs[fut]
            try:
                rows = fut.result()
                append_jsonl(raw_log, rows)
                raw_records.extend(rows)
            except Exception as e:
                errors.append({"query_id": task.query_id, "option_label": task.option_label, "error": str(e)})
            if done % 10 == 0 or done == len(tasks):
                print(f"  retrieve [{done}/{len(tasks)}] records={len(raw_records)} errors={len(errors)}")

    write_json(args.output_root / "retrieval_errors.json", errors)

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for rec in raw_records:
        groups.setdefault((str(rec.get("query_id")), str(rec.get("option_label"))), []).append(rec)

    print("  loading BiRefNet...")
    birefnet_model = load_model(args.birefnet_model_id, args.device, args.fp16)

    scored_records: list[dict[str, Any]] = []
    items = list(groups.values())

    # BiRefNet model is shared in-process. Keep this serial by default.
    if args.score_workers <= 1:
        for done, rows in enumerate(items, start=1):
            scored = score_group(rows, args, birefnet_model)
            append_jsonl(scored_log, scored)
            scored_records.extend(scored)
            if done % 4 == 0 or done == len(items):
                print(f"  score    [{done}/{len(items)}] option groups")
    else:
        raise SystemExit("--score-workers > 1 is intentionally disabled for shared BiRefNet model safety")

    scored_records = grouped_rerank(scored_records)
    if scored_log.exists():
        scored_log.unlink()
    append_jsonl(scored_log, scored_records)

    gallery = build_gallery(
        scored_records,
        args.output_root,
        "QwenEmb product-shot retrieval + BiRefNet patch pattern coverage",
        args.show_k,
    )

    print()
    print("Done.")
    print(f"  raw records:    {len(raw_records)}")
    print(f"  scored records: {len(scored_records)}")
    print(f"  retrieval errs: {len(errors)}")
    print(f"  raw log:        {raw_log}")
    print(f"  scored log:     {scored_log}")
    print(f"  gallery:        {gallery}")


if __name__ == "__main__":
    main()
