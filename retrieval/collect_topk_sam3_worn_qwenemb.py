#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import mimetypes
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import requests
import torch
from PIL import Image


OPTION_LABELS = ["A", "B", "C", "D"]
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
}


@dataclass(frozen=True)
class Task:
    plan_idx: int
    query_id: str
    user_id: str
    option_label: str
    option_semantic: str
    option_text: str
    target_color: str
    target_pattern: str
    target_garment: str
    attrs: dict[str, Any]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def slug(text: Any, max_len: int = 100) -> str:
    s = str(text or "")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")
    return (s or "unknown")[:max_len]


def norm(value: Any) -> str:
    return str(value or "").replace("_", " ").strip().lower()


def fnum(x: Any) -> float:
    try:
        return float(x or 0.0)
    except Exception:
        return 0.0


def option_to_text(opt: dict[str, Any]) -> str:
    if opt.get("search_query"):
        return str(opt["search_query"])

    attrs = opt.get("attributes") or {}
    parts = []
    if attrs.get("pattern") and attrs.get("pattern") != "solid":
        parts.append(norm(attrs.get("pattern")))
    if attrs.get("color"):
        parts.append(norm(attrs.get("color")))
    if attrs.get("garment_category"):
        parts.append(norm(attrs.get("garment_category")))

    return " ".join(parts).strip() or "unknown clothing item"


def make_tasks(plans: list[dict[str, Any]], options: list[str]) -> list[Task]:
    tasks: list[Task] = []

    for i, plan in enumerate(plans):
        opts = plan.get("options") or {}
        for label in options:
            opt = opts.get(label)
            if not isinstance(opt, dict):
                continue

            attrs = opt.get("attributes") or {}
            tasks.append(Task(
                plan_idx=i,
                query_id=str(plan.get("query_id") or f"plan_{i:05d}"),
                user_id=str(plan.get("user_id") or "unknown_user"),
                option_label=label,
                option_semantic=str(opt.get("label") or ""),
                option_text=option_to_text(opt),
                target_color=norm(attrs.get("color")),
                target_pattern=norm(attrs.get("pattern")),
                target_garment=norm(attrs.get("garment_category")),
                attrs=attrs,
            ))

    return tasks


def retrieval_query(task: Task, mode: str) -> str:
    if mode == "original":
        return task.option_text

    if mode == "garment_first":
        parts = []
        if task.target_garment:
            parts.append(task.target_garment)
        if task.target_pattern and task.target_pattern != "solid":
            parts.append(task.target_pattern)
        if task.target_color:
            parts.append(task.target_color)
        return ", ".join(parts) or task.option_text

    raise ValueError(f"unknown query mode: {mode}")


def endpoint(base: str, route: str) -> str:
    base = base.rstrip("/")
    for suffix in ["/knn-service", "/score-image-files", "/score-candidates"]:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base + route


def post_json(session: requests.Session, url: str, payload: dict[str, Any], timeout: float, retries: int) -> Any:
    last = None
    for attempt in range(retries + 1):
        try:
            resp = session.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and data.get("error"):
                raise RuntimeError(str(data["error"]))
            return data
        except Exception as e:
            last = e
            if attempt >= retries:
                break
            time.sleep(min(2 ** attempt, 8.0))

    raise RuntimeError(f"POST failed: {url}: {last}")


def query_knn(session: requests.Session, args: argparse.Namespace, text: str) -> list[dict[str, Any]]:
    data = post_json(
        session,
        args.client_url,
        {
            "text": text,
            "modality": "image",
            "num_images": args.retrieval_k,
            "indice_name": args.index_name,
        },
        args.timeout,
        args.retries,
    )

    if not isinstance(data, list):
        raise RuntimeError(f"KNN response must be list, got {type(data).__name__}: {data}")

    return [x for x in data if isinstance(x, dict)]


def choose_ext(url: str, content_type: str | None) -> str:
    if content_type:
        ct = content_type.split(";", 1)[0].strip().lower()
        ext = mimetypes.guess_extension(ct)
        if ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            return ".jpg" if ext == ".jpeg" else ext

    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        return ".jpg" if suffix == ".jpeg" else suffix

    return ".jpg"


def download_image(session: requests.Session, url: str, out_base: Path, args: argparse.Namespace) -> tuple[Path | None, str | None]:
    if not url:
        return None, "missing_url"

    last = None
    for attempt in range(args.retries + 1):
        try:
            with session.get(url, headers=DEFAULT_HEADERS, stream=True, timeout=args.download_timeout) as resp:
                resp.raise_for_status()
                ext = choose_ext(url, resp.headers.get("Content-Type"))
                out = out_base.with_suffix(ext)
                tmp = out.with_suffix(out.suffix + ".tmp")
                out.parent.mkdir(parents=True, exist_ok=True)

                total = 0
                with tmp.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 128):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if args.max_image_bytes > 0 and total > args.max_image_bytes:
                            raise RuntimeError("image too large")
                        f.write(chunk)

                tmp.replace(out)
                return out, None

        except Exception as e:
            last = e
            if attempt >= args.retries:
                break
            time.sleep(min(2 ** attempt, 8.0))

    return None, str(last)


def score_image_files(
    session: requests.Session,
    args: argparse.Namespace,
    texts: dict[str, str],
    paths: list[str],
    batch_size: int,
) -> list[dict[str, float]]:
    if not paths:
        return []

    out: list[dict[str, float]] = []

    for i in range(0, len(paths), batch_size):
        chunk = paths[i:i + batch_size]
        data = post_json(
            session,
            args.image_score_url,
            {"texts": texts, "image_paths": chunk},
            args.score_timeout,
            args.retries,
        )

        scores = data.get("scores") if isinstance(data, dict) else None
        if not isinstance(scores, list):
            scores = []

        for row in scores:
            if isinstance(row, dict):
                out.append({k: float(v) for k, v in row.items() if isinstance(v, (int, float))})
            else:
                out.append({})

        while len(out) < i + len(chunk):
            out.append({})

    return out


def garment_aliases(garment: str) -> list[str]:
    g = norm(garment)

    table = {
        "coat": ["coat", "overcoat", "long coat"],
        "jacket": ["jacket", "outerwear"],
        "cardigan": ["cardigan"],
        "hoodie": ["hoodie", "sweatshirt"],
        "sweater": ["sweater", "knitwear", "pullover"],
        "tank top": ["tank top", "sleeveless top", "camisole top"],
        "sleeveless top": ["sleeveless top", "tank top", "camisole top"],
        "camisole": ["camisole", "camisole top", "tank top"],
        "crop top": ["crop top", "tank top", "top"],
        "t shirt": ["t shirt", "shirt", "top"],
        "shirt": ["shirt", "top"],
        "blouse": ["blouse", "shirt", "top"],
        "skirt": ["skirt"],
        "dress": ["dress"],
        "pants": ["pants", "trousers"],
        "jeans": ["jeans", "denim pants"],
        "shorts": ["shorts"],
    }

    aliases = table.get(g, [g] if g else ["clothing item"])

    seen = set()
    out = []
    for a in aliases:
        a = norm(a)
        if a and a not in seen:
            seen.add(a)
            out.append(a)

    return out or ["clothing item"]


def load_sam3(device: str):
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model()
    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()

    processor = Sam3Processor(model)
    return processor


def to_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def normalize_masks(masks: Any) -> np.ndarray:
    arr = to_numpy(masks)

    if arr.size == 0:
        return np.zeros((0, 1, 1), dtype=bool)

    if arr.ndim == 4:
        if arr.shape[1] == 1:
            arr = arr[:, 0]
        elif arr.shape[0] == 1:
            arr = arr[0]

    if arr.ndim == 2:
        arr = arr[None, :, :]

    return arr.astype(bool)


def normalize_boxes(boxes: Any, n: int, masks: np.ndarray) -> np.ndarray:
    if boxes is None:
        return boxes_from_masks(masks)

    arr = to_numpy(boxes)

    if arr.size == 0:
        return boxes_from_masks(masks)

    arr = arr.reshape(-1, 4).astype(np.float32)

    if len(arr) < n:
        fallback = boxes_from_masks(masks)
        out = np.zeros((n, 4), dtype=np.float32)
        for i in range(n):
            out[i] = arr[i] if i < len(arr) else fallback[i]
        return out

    return arr[:n]


def normalize_scores(scores: Any, n: int) -> np.ndarray:
    if scores is None:
        return np.ones((n,), dtype=np.float32)

    arr = to_numpy(scores).reshape(-1).astype(np.float32)

    if len(arr) < n:
        out = np.ones((n,), dtype=np.float32)
        out[:len(arr)] = arr
        return out

    return arr[:n]


def boxes_from_masks(masks: np.ndarray) -> np.ndarray:
    boxes = []

    for m in masks:
        ys, xs = np.where(m)
        if len(xs) == 0 or len(ys) == 0:
            boxes.append([0, 0, 1, 1])
        else:
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])

    return np.asarray(boxes, dtype=np.float32)


def expand_box(box: np.ndarray, image_size: tuple[int, int], ratio: float) -> np.ndarray:
    w, h = image_size
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    return np.asarray([
        max(0.0, x1 - bw * ratio),
        max(0.0, y1 - bh * ratio),
        min(float(w), x2 + bw * ratio),
        min(float(h), y2 + bh * ratio),
    ], dtype=np.float32)


def box_to_int(box: np.ndarray, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    w, h = image_size

    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))

    return x1, y1, x2, y2


def sam3_detect_best(processor, image: Image.Image, garment: str, args: argparse.Namespace) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    state = processor.set_image(image)

    candidates: list[dict[str, Any]] = []

    for prompt in garment_aliases(garment):
        try:
            output = processor.set_text_prompt(state=state, prompt=prompt)
        except TypeError:
            output = processor.set_text_prompt(state, prompt)

        masks = normalize_masks(output.get("masks"))
        if len(masks) == 0:
            continue

        boxes = normalize_boxes(output.get("boxes"), len(masks), masks)
        scores = normalize_scores(output.get("scores"), len(masks))

        w, h = image.size

        for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
            mask_area_ratio = float(mask.sum()) / float(mask.size)
            x1, y1, x2, y2 = [float(v) for v in box]
            box_area_ratio = max(0.0, (x2 - x1) * (y2 - y1) / max(1.0, w * h))

            if mask_area_ratio < args.min_mask_area_ratio:
                continue
            if mask_area_ratio > args.max_mask_area_ratio:
                continue
            if box_area_ratio < args.min_box_area_ratio:
                continue
            if box_area_ratio > args.max_box_area_ratio:
                continue

            candidates.append({
                "prompt": prompt,
                "index": int(i),
                "mask": mask,
                "box": box,
                "score": float(score),
                "mask_area_ratio": mask_area_ratio,
                "box_area_ratio": box_area_ratio,
            })

    if not candidates:
        return None, None, {
            "status": "no_sam3_mask",
            "prompts": garment_aliases(garment),
        }

    # Do not prefer large masks. SAM3 score is the primary signal.
    candidates.sort(key=lambda r: r["score"], reverse=True)
    best = candidates[0]

    used_box = expand_box(best["box"], image.size, args.box_expand)

    return best["mask"], used_box, {
        "status": "ok",
        "selected_prompt": best["prompt"],
        "selected_index": best["index"],
        "selected_score": best["score"],
        "selected_mask_area_ratio": best["mask_area_ratio"],
        "selected_box_area_ratio": best["box_area_ratio"],
        "box_xyxy": [float(v) for v in used_box],
        "num_candidates": len(candidates),
        "top_candidates": [
            {
                "prompt": c["prompt"],
                "index": c["index"],
                "score": c["score"],
                "mask_area_ratio": c["mask_area_ratio"],
                "box_area_ratio": c["box_area_ratio"],
                "box_xyxy": [float(v) for v in c["box"]],
            }
            for c in candidates[:5]
        ],
    }


def crop_image_and_mask(image: Image.Image, mask: np.ndarray, box: np.ndarray) -> tuple[Image.Image, np.ndarray, tuple[int, int, int, int]]:
    x1, y1, x2, y2 = box_to_int(box, image.size)
    crop = image.crop((x1, y1, x2, y2)).convert("RGB")
    crop_mask = mask[y1:y2, x1:x2].copy()
    return crop, crop_mask, (x1, y1, x2, y2)


def ceil_to_multiple(x: int, m: int) -> int:
    return int(math.ceil(x / m) * m)


def patch_grid(w: int, h: int, short_side_tiles: int) -> list[tuple[int, int, int]]:
    side = max(1, int(math.ceil(min(w, h) / max(1, short_side_tiles))))
    pw = ceil_to_multiple(w, side)
    ph = ceil_to_multiple(h, side)
    return [(x, y, side) for y in range(0, ph, side) for x in range(0, pw, side)]


def make_patch_files(crop: Image.Image, mask: np.ndarray, args: argparse.Namespace, out_dir: Path) -> list[dict[str, Any]]:
    crop = crop.convert("RGB")
    w, h = crop.size

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for i, (x, y, s) in enumerate(patch_grid(w, h, args.short_side_tiles)):
        x2, y2 = min(x + s, w), min(y + s, h)
        if x2 <= x or y2 <= y:
            continue

        coverage = float(mask[y:y2, x:x2].sum()) / float(s * s)
        if coverage < args.min_coverage:
            continue

        tile = Image.new("RGB", (s, s), "white")
        tile.paste(crop.crop((x, y, x2, y2)), (0, 0))

        path = out_dir / f"tile_{i:04d}__x{x}_y{y}_s{s}_cov{coverage:.3f}.jpg"
        tile.save(path, quality=92)

        rows.append({
            "index": i,
            "tile": str(path),
            "x": x,
            "y": y,
            "size": s,
            "mask_coverage": coverage,
        })

    return rows


def pattern_texts(pattern: str) -> dict[str, str]:
    p = norm(pattern)

    if p == "solid":
        return {
            "positive": "plain solid clothing fabric",
            "negative_pattern": "striped checkered plaid floral polka dot patterned clothing fabric",
            "negative_nonclothing": "skin hair body background",
        }

    return {
        "positive": f"{p} clothing fabric pattern",
        "negative_plain": "plain solid clothing fabric",
        "negative_other_pattern": "different clothing fabric pattern",
        "negative_nonclothing": "skin hair body background",
    }


def score_pattern_on_crop(
    session: requests.Session,
    crop: Image.Image,
    crop_mask: np.ndarray,
    pattern: str,
    args: argparse.Namespace,
    tmp_dir: Path,
) -> dict[str, Any]:
    p = norm(pattern)
    if not p:
        return {
            "score": 1.0,
            "mode": "missing_pattern_target",
            "tile_scores": [],
        }

    tiles = make_patch_files(crop, crop_mask, args, tmp_dir / "patches")
    if not tiles:
        return {
            "score": 0.0,
            "mode": "no_patch_inside_mask",
            "tile_scores": [],
        }

    rows = score_image_files(
        session=session,
        args=args,
        texts=pattern_texts(p),
        paths=[t["tile"] for t in tiles],
        batch_size=args.patch_score_batch,
    )

    tile_scores = []
    weighted_hit = 0.0
    weighted_soft = 0.0
    total_weight = 0.0

    for tile, row in zip(tiles, rows):
        pos = float(row.get("positive", 0.0))
        neg = max([float(v) for k, v in row.items() if k != "positive"] or [0.0])
        margin = pos - neg
        present = margin >= args.pattern_margin
        weight = float(tile.get("mask_coverage") or 0.0)
        soft = pos / max(1e-8, pos + neg)

        weighted_hit += weight * (1.0 if present else 0.0)
        weighted_soft += weight * soft
        total_weight += weight

        tile_scores.append({
            "index": tile["index"],
            "x": tile["x"],
            "y": tile["y"],
            "size": tile["size"],
            "mask_coverage": weight,
            "positive_score": pos,
            "strongest_negative_score": neg,
            "margin": margin,
            "soft_score": soft,
            "present": present,
        })

    score = weighted_hit / total_weight if total_weight > 0 else 0.0
    soft_score = weighted_soft / total_weight if total_weight > 0 else 0.0

    return {
        "score": score,
        "soft_score": soft_score,
        "mode": "sam3_crop_weighted_patch_pattern_coverage",
        "pattern": p,
        "hits": sum(1 for t in tile_scores if t["present"]),
        "num_tiles": len(tile_scores),
        "total_mask_weight": total_weight,
        "pattern_margin": args.pattern_margin,
        "formula": "sum(mask_coverage_i * pattern_hit_i) / sum(mask_coverage_i)",
        "tile_scores": tile_scores,
    }


def score_axis_on_crop(session: requests.Session, crop_path: Path, task: Task, args: argparse.Namespace) -> tuple[float, float, dict[str, Any]]:
    texts = {}
    if task.target_color:
        texts["color_score"] = f"{task.target_color} clothing item"
    if task.target_garment:
        texts["garment_score"] = f"{task.target_garment} clothing item"

    if not texts:
        return 1.0, 1.0, {}

    rows = score_image_files(
        session=session,
        args=args,
        texts=texts,
        paths=[str(crop_path)],
        batch_size=1,
    )

    row = rows[0] if rows else {}
    color_score = 1.0 if not task.target_color else max(0.0, float(row.get("color_score", 0.0)))
    garment_score = 1.0 if not task.target_garment else max(0.0, float(row.get("garment_score", 0.0)))

    return color_score, garment_score, row


def save_winner_crops(records: list[dict[str, Any]], task: Task, out_root: Path, save_top_m: int) -> list[Path]:
    rows = [
        r for r in records
        if not r.get("skip_reason") and r.get("_temp_crop_path") and Path(str(r["_temp_crop_path"])).exists()
    ]
    rows = sorted(rows, key=lambda r: (fnum(r.get("combined_score")), fnum(r.get("similarity"))), reverse=True)

    odir = (
        out_root
        / "winner_crops"
        / slug(task.query_id, 80)
        / f"{task.option_label}_{slug(task.option_semantic, 40)}__{slug(task.option_text, 80)}"
    )
    odir.mkdir(parents=True, exist_ok=True)

    tsv = [
        f"query_id\t{task.query_id}",
        f"option\t{task.option_label}",
        f"option_text\t{task.option_text}",
        f"target\tcolor={task.target_color} pattern={task.target_pattern} garment={task.target_garment}",
        "",
        "rank\torig\tcombined\tpattern\tcolor\tgarment\tsimilarity\tsam3_prompt\tfile",
    ]

    saved = []

    for rank, r in enumerate(rows[:save_top_m], start=1):
        src = Path(str(r["_temp_crop_path"]))
        ext = src.suffix or ".jpg"

        name = (
            f"{rank:03d}"
            f"__orig_{int(r.get('original_rank') or 0):03d}"
            f"__comb_{fnum(r.get('combined_score')):.4f}"
            f"__pat_{fnum(r.get('pattern_score')):.4f}"
            f"__col_{fnum(r.get('color_score')):.4f}"
            f"__gar_{fnum(r.get('garment_score')):.4f}"
            f"__sim_{fnum(r.get('similarity')):.4f}"
            f"__sam3_{slug(r.get('sam3_selected_prompt'), 24)}"
            f"{ext}"
        )
        dst = odir / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()

        shutil.copy2(src, dst)
        r["saved_crop_path"] = str(dst)
        saved.append(dst)

        tsv.append(
            f"{rank:03d}\t{int(r.get('original_rank') or 0):03d}\t"
            f"{fnum(r.get('combined_score')):.6f}\t{fnum(r.get('pattern_score')):.6f}\t"
            f"{fnum(r.get('color_score')):.6f}\t{fnum(r.get('garment_score')):.6f}\t"
            f"{fnum(r.get('similarity')):.6f}\t{r.get('sam3_selected_prompt')}\t{name}"
        )

    if not saved:
        tsv.append("NO_VALID_CROP")

    (odir / "_ranking.tsv").write_text("\n".join(tsv) + "\n", encoding="utf-8")
    print(f"[winner] {odir}")

    return saved


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--plan-path", type=Path, default=Path("data/options/option_plans.jsonl"))
    p.add_argument("--client-url", default="http://127.0.0.1:1236/knn-service")
    p.add_argument("--image-score-url", default="")
    p.add_argument("--index-name", default="pod_qwenemb")
    p.add_argument("--output-root", type=Path, default=Path("data/retrieval/sam3_worn_qwenemb"))

    p.add_argument("--query-mode", choices=["original", "garment_first"], default="original")
    p.add_argument("--retrieval-k", type=int, default=100)
    p.add_argument("--score-top-n", type=int, default=100)
    p.add_argument("--save-top-m", type=int, default=1)

    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--options", default="A,B,C,D")

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--min-box-area-ratio", type=float, default=0.001)
    p.add_argument("--max-box-area-ratio", type=float, default=0.95)
    p.add_argument("--min-mask-area-ratio", type=float, default=0.001)
    p.add_argument("--max-mask-area-ratio", type=float, default=0.95)
    p.add_argument("--box-expand", type=float, default=0.02)

    p.add_argument("--short-side-tiles", type=int, default=8)
    p.add_argument("--min-coverage", type=float, default=0.20)
    p.add_argument("--pattern-margin", type=float, default=0.0)
    p.add_argument("--patch-score-batch", type=int, default=256)

    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--download-timeout", type=float, default=30.0)
    p.add_argument("--score-timeout", type=float, default=120.0)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--max-image-bytes", type=int, default=20_000_000)

    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.image_score_url = args.image_score_url or endpoint(args.client_url, "/score-image-files")

    options = [x.strip().upper() for x in args.options.split(",") if x.strip()]
    bad = [x for x in options if x not in OPTION_LABELS]
    if bad:
        raise SystemExit(f"bad option labels: {bad}")

    if args.force and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    plans = load_jsonl(args.plan_path)
    if args.offset:
        plans = plans[args.offset:]
    if args.limit > 0:
        plans = plans[:args.limit]

    tasks = make_tasks(plans, options)

    print("=" * 96)
    print("top-k retrieval -> SAM3 concept mask crop -> QwenEmb patch scoring -> winner crop")
    print("=" * 96)
    print(f"plans:        {len(plans)}")
    print(f"option tasks: {len(tasks)}")
    print(f"query_mode:   {args.query_mode}")
    print(f"retrieval_k:  {args.retrieval_k}")
    print(f"score_top_n:  {args.score_top_n}")
    print(f"save_top_m:   {args.save_top_m}")
    print(f"output:       {args.output_root}")
    print(f"QwenEmb:      {args.client_url}")
    print(f"image score:  {args.image_score_url}")

    write_json(args.output_root / "run_config.json", vars(args))

    print("loading SAM3...")
    sam3_processor = load_sam3(args.device)

    session = requests.Session()

    raw_log = args.output_root / "raw_topk_results.jsonl"
    scored_log = args.output_root / "scored_results.jsonl"
    for p in [raw_log, scored_log]:
        if p.exists():
            p.unlink()

    with tempfile.TemporaryDirectory(prefix="sam3_worn_run_") as run_tmp:
        tmp_root = Path(run_tmp)

        for task_i, task in enumerate(tasks, start=1):
            q = retrieval_query(task, args.query_mode)
            print()
            print(f"[{task_i}/{len(tasks)}] {task.query_id} {task.option_label} | query={q}")

            results = query_knn(session, args, q)
            if args.score_top_n > 0:
                results = results[: args.score_top_n]

            records: list[dict[str, Any]] = []

            for rank, item in enumerate(results, start=1):
                url = str(item.get("image_url") or item.get("url") or "")
                key = str(item.get("key") or f"rank_{rank:03d}")
                rec_tmp = tmp_root / slug(task.query_id, 80) / task.option_label / f"rank_{rank:03d}"
                img_base = rec_tmp / f"orig__{slug(key, 80)}"

                rec = {
                    "plan_idx": task.plan_idx,
                    "query_id": task.query_id,
                    "user_id": task.user_id,
                    "option_label": task.option_label,
                    "option_semantic": task.option_semantic,
                    "retrieval_query": q,
                    "option_text": task.option_text,
                    "target_color": task.target_color,
                    "target_pattern": task.target_pattern,
                    "target_garment": task.target_garment,
                    "original_rank": rank,
                    "similarity": item.get("similarity"),
                    "url": url,
                    "key": key,
                    "caption": item.get("caption") or item.get("title") or "",
                    "raw": item,
                }

                img_path, err = download_image(session, url, img_base, args)
                if err or img_path is None:
                    rec.update({
                        "skip_reason": f"download_failed: {err}",
                        "combined_score": 0.0,
                        "pattern_score": 0.0,
                        "color_score": 0.0,
                        "garment_score": 0.0,
                    })
                    records.append(rec)
                    continue

                try:
                    image = Image.open(img_path).convert("RGB")
                except Exception as e:
                    rec.update({
                        "skip_reason": f"image_open_failed: {e}",
                        "combined_score": 0.0,
                        "pattern_score": 0.0,
                        "color_score": 0.0,
                        "garment_score": 0.0,
                    })
                    records.append(rec)
                    continue

                mask, box, sam3_info = sam3_detect_best(
                    processor=sam3_processor,
                    image=image,
                    garment=task.target_garment,
                    args=args,
                )

                if mask is None or box is None:
                    rec.update({
                        "skip_reason": "no_sam3_mask",
                        "sam3_info": sam3_info,
                        "combined_score": 0.0,
                        "pattern_score": 0.0,
                        "color_score": 0.0,
                        "garment_score": 0.0,
                    })
                    records.append(rec)
                    continue

                crop, crop_mask, crop_xyxy = crop_image_and_mask(image, mask, box)
                crop_path = rec_tmp / "sam3_crop.jpg"
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                crop.save(crop_path, quality=94)

                pattern_info = score_pattern_on_crop(
                    session=session,
                    crop=crop,
                    crop_mask=crop_mask,
                    pattern=task.target_pattern,
                    args=args,
                    tmp_dir=rec_tmp,
                )
                pattern_score = float(pattern_info.get("score") or 0.0)

                color_score, garment_score, axis_raw = score_axis_on_crop(
                    session=session,
                    crop_path=crop_path,
                    task=task,
                    args=args,
                )

                combined = math.sqrt(max(0.0, pattern_score * color_score * garment_score))

                rec.update({
                    "skip_reason": None,
                    "sam3_info": sam3_info,
                    "sam3_selected_prompt": sam3_info.get("selected_prompt"),
                    "crop_xyxy": list(crop_xyxy),
                    "pattern_score": pattern_score,
                    "color_score": color_score,
                    "garment_score": garment_score,
                    "combined_score": combined,
                    "pattern_score_raw": pattern_info,
                    "axis_score_raw": axis_raw,
                    "rerank_formula": "sqrt(max(0, pattern_score * color_score * garment_score))",
                    "_temp_crop_path": str(crop_path),
                })
                records.append(rec)

                if rank % 10 == 0 or rank == len(results):
                    valid = sum(1 for r in records if not r.get("skip_reason"))
                    print(f"  processed {rank}/{len(results)} valid={valid}")

            records = sorted(
                records,
                key=lambda r: (fnum(r.get("combined_score")), fnum(r.get("similarity"))),
                reverse=True,
            )
            for i, r in enumerate(records, start=1):
                r["rerank"] = i

            saved = save_winner_crops(
                records=records,
                task=task,
                out_root=args.output_root,
                save_top_m=args.save_top_m,
            )

            for r in records:
                r.pop("_temp_crop_path", None)

            append_jsonl(raw_log, [{
                "query_id": task.query_id,
                "option_label": task.option_label,
                "retrieval_query": q,
                "num_retrieved": len(results),
                "num_valid_sam3": sum(1 for r in records if not r.get("skip_reason")),
            }])
            append_jsonl(scored_log, records)

            if saved:
                top = records[0]
                print(
                    f"  top1 orig={top.get('original_rank')} "
                    f"comb={fnum(top.get('combined_score')):.4f} "
                    f"pat={fnum(top.get('pattern_score')):.4f} "
                    f"col={fnum(top.get('color_score')):.4f} "
                    f"gar={fnum(top.get('garment_score')):.4f} "
                    f"prompt={top.get('sam3_selected_prompt')}"
                )
            else:
                print("  no valid SAM3 crop")

    print()
    print("Done.")
    print(f"raw log:     {raw_log}")
    print(f"scored log:  {scored_log}")
    print(f"winner dir:  {args.output_root / 'winner_crops'}")


if __name__ == "__main__":
    main()
