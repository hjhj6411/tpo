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
from contextlib import nullcontext
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
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


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
    tasks = []
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
            r = session.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and data.get("error"):
                raise RuntimeError(str(data["error"]))
            return data
        except Exception as e:
            last = e
            if attempt >= retries:
                break
            time.sleep(min(2 ** attempt, 8))
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
            with session.get(url, headers=DEFAULT_HEADERS, stream=True, timeout=args.download_timeout) as r:
                r.raise_for_status()
                ext = choose_ext(url, r.headers.get("Content-Type"))
                out = out_base.with_suffix(ext)
                tmp = out.with_suffix(out.suffix + ".tmp")
                out.parent.mkdir(parents=True, exist_ok=True)
                total = 0
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 128):
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
            time.sleep(min(2 ** attempt, 8))
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


def load_groundingdino(model_id: str, device: str):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model


def load_sam2(model_id: str, device: str):
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    predictor = SAM2ImagePredictor.from_pretrained(model_id)
    if hasattr(predictor, "model"):
        predictor.model.to(device)
        predictor.model.eval()
    return predictor


def clean_prompt(text: str) -> str:
    text = " ".join(str(text or "").lower().replace("_", " ").split())
    if not text.endswith("."):
        text += "."
    return text


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

    aliases = table.get(g, [g] if g else ["clothing"])
    seen = set()
    out = []
    for a in aliases:
        a = norm(a)
        if a and a not in seen:
            seen.add(a)
            out.append(a)
    return out or ["clothing"]


def post_process_groundingdino(processor, outputs, input_ids, image: Image.Image, threshold: float, text_threshold: float):
    target_sizes = [(image.height, image.width)]
    fn = processor.post_process_grounded_object_detection

    calls = [
        lambda: fn(outputs, input_ids, threshold=threshold, text_threshold=text_threshold, target_sizes=target_sizes),
        lambda: fn(outputs, input_ids=input_ids, threshold=threshold, text_threshold=text_threshold, target_sizes=target_sizes),
        lambda: fn(outputs, threshold=threshold, text_threshold=text_threshold, target_sizes=target_sizes),
        lambda: fn(outputs, input_ids, box_threshold=threshold, text_threshold=text_threshold, target_sizes=target_sizes),
        lambda: fn(outputs, input_ids=input_ids, box_threshold=threshold, text_threshold=text_threshold, target_sizes=target_sizes),
    ]

    last = None
    for call in calls:
        try:
            res = call()
            return res[0] if isinstance(res, list) else res
        except TypeError as e:
            last = e
    raise last


def detect_best_box(processor, model, image: Image.Image, garment: str, device: str, args: argparse.Namespace):
    prompts = [clean_prompt(x) for x in garment_aliases(garment)]

    candidates = []

    for prompt in prompts:
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        ctx = torch.autocast("cuda", dtype=torch.float16) if args.fp16 and str(device).startswith("cuda") else nullcontext()

        with torch.no_grad(), ctx:
            outputs = model(**inputs)

        result = post_process_groundingdino(
            processor=processor,
            outputs=outputs,
            input_ids=inputs.input_ids,
            image=image,
            threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )

        boxes = result.get("boxes")
        scores = result.get("scores")
        labels = result.get("text_labels") or result.get("labels") or []

        if boxes is None or len(boxes) == 0:
            continue

        boxes_np = boxes.detach().cpu().numpy() if torch.is_tensor(boxes) else np.asarray(boxes)
        scores_np = scores.detach().cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)

        w, h = image.size
        for i, box in enumerate(boxes_np):
            x1, y1, x2, y2 = [float(v) for v in box]
            area_ratio = max(0.0, (x2 - x1) * (y2 - y1) / max(1.0, w * h))
            if area_ratio < args.min_box_area_ratio:
                continue
            if area_ratio > args.max_box_area_ratio:
                continue

            score = float(scores_np[i]) if i < len(scores_np) else 0.0
            label = str(labels[i]) if i < len(labels) else prompt

            # Do not boost large boxes. Use detector confidence first.
            candidates.append({
                "prompt": prompt,
                "label": label,
                "box": np.asarray([x1, y1, x2, y2], dtype=np.float32),
                "score": score,
                "area_ratio": area_ratio,
            })

    candidates.sort(key=lambda r: r["score"], reverse=True)

    if not candidates:
        return None, {
            "status": "no_box",
            "prompts": prompts,
        }

    best = candidates[0]
    return best["box"], {
        "status": "ok",
        "prompts": prompts,
        "selected_prompt": best["prompt"],
        "selected_label": best["label"],
        "selected_score": best["score"],
        "selected_area_ratio": best["area_ratio"],
        "box_xyxy": [float(v) for v in best["box"]],
        "top_candidates": [
            {
                "prompt": c["prompt"],
                "label": c["label"],
                "score": c["score"],
                "area_ratio": c["area_ratio"],
                "box_xyxy": [float(v) for v in c["box"]],
            }
            for c in candidates[:5]
        ],
    }


def expand_box(box: np.ndarray, image_size: tuple[int, int], ratio: float) -> np.ndarray:
    w, h = image_size
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    return np.asarray([
        max(0.0, x1 - bw * ratio),
        max(0.0, y1 - bh * ratio),
        min(float(w - 1), x2 + bw * ratio),
        min(float(h - 1), y2 + bh * ratio),
    ], dtype=np.float32)


def box_to_int(box: np.ndarray, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    w, h = image_size
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))
    return x1, y1, x2, y2


def sam2_mask_or_box(predictor, image: Image.Image, box: np.ndarray, args: argparse.Namespace):
    expanded = expand_box(box, image.size, args.box_expand)
    arr = np.asarray(image.convert("RGB"))

    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if str(args.device).startswith("cuda") else nullcontext()

    try:
        with torch.inference_mode(), ctx:
            predictor.set_image(arr)
            masks, scores, _ = predictor.predict(
                box=expanded.astype(np.float32),
                multimask_output=True,
            )

        masks_np = masks.detach().cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks)
        scores_np = scores.detach().cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)

        if masks_np.ndim == 4:
            masks_np = masks_np[:, 0]
        if masks_np.ndim == 2:
            masks_np = masks_np[None, :, :]

        best = None
        for i, m in enumerate(masks_np):
            mask = m.astype(bool)
            score = float(scores_np[i]) if i < len(scores_np) else 0.0
            area_ratio = float(mask.sum()) / float(mask.size)

            if area_ratio < args.min_mask_area_ratio:
                continue
            if area_ratio > args.max_mask_area_ratio:
                continue

            row = {
                "mask": mask,
                "score": score,
                "area_ratio": area_ratio,
            }
            if best is None or row["score"] > best["score"]:
                best = row

        if best is not None:
            return best["mask"], expanded, {
                "mask_source": "groundingdino_sam2",
                "sam2_score": best["score"],
                "mask_area_ratio": best["area_ratio"],
            }

    except Exception as e:
        sam_error = str(e)
    else:
        sam_error = "no_valid_sam2_mask"

    if not args.box_mask_fallback:
        return None, expanded, {
            "mask_source": "sam2_failed",
            "sam2_error": sam_error,
        }

    w, h = image.size
    x1, y1, x2, y2 = box_to_int(expanded, image.size)
    mask = np.zeros((h, w), dtype=bool)
    mask[y1:y2, x1:x2] = True

    return mask, expanded, {
        "mask_source": "groundingdino_box_fallback",
        "sam2_error": sam_error,
        "mask_area_ratio": float(mask.sum()) / float(mask.size),
    }


def crop_image_and_mask(image: Image.Image, mask: np.ndarray, box: np.ndarray):
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

    rows = []
    for i, (x, y, s) in enumerate(patch_grid(w, h, args.short_side_tiles)):
        x2, y2 = min(x + s, w), min(y + s, h)
        if x2 <= x or y2 <= y:
            continue

        cov = float(mask[y:y2, x:x2].sum()) / float(s * s)
        if cov < args.min_coverage:
            continue

        tile = Image.new("RGB", (s, s), "white")
        tile.paste(crop.crop((x, y, x2, y2)), (0, 0))

        path = out_dir / f"tile_{i:04d}__x{x}_y{y}_s{s}_cov{cov:.3f}.jpg"
        tile.save(path, quality=92)

        rows.append({
            "index": i,
            "tile": str(path),
            "x": x,
            "y": y,
            "size": s,
            "mask_coverage": cov,
        })

    return rows


def pattern_texts(pattern: str) -> tuple[str, str]:
    p = norm(pattern)
    if not p:
        return "", ""
    if p == "solid":
        return "plain solid clothing fabric", "striped checkered plaid floral polka dot patterned clothing fabric"
    return f"{p} clothing fabric pattern", "plain solid clothing fabric or background skin hair"


def score_pattern_on_crop(session: requests.Session, crop: Image.Image, crop_mask: np.ndarray, pattern: str, args: argparse.Namespace, tmp_dir: Path):
    p = norm(pattern)
    if not p:
        return {"score": 1.0, "mode": "missing_pattern_target", "tile_scores": []}

    tiles = make_patch_files(crop, crop_mask, args, tmp_dir / "patches")
    if not tiles:
        return {"score": 0.0, "mode": "no_patch_inside_mask", "tile_scores": []}

    pos, neg = pattern_texts(p)
    rows = score_image_files(
        session=session,
        args=args,
        texts={"pattern": pos, "negative": neg},
        paths=[t["tile"] for t in tiles],
        batch_size=args.patch_score_batch,
    )

    tile_scores = []
    for tile, row in zip(tiles, rows):
        ps = float(row.get("pattern", 0.0))
        ns = float(row.get("negative", 0.0))
        margin = ps - ns
        present = margin >= args.pattern_margin
        tile_scores.append({
            "index": tile["index"],
            "x": tile["x"],
            "y": tile["y"],
            "size": tile["size"],
            "mask_coverage": tile["mask_coverage"],
            "pattern_similarity": ps,
            "negative_similarity": ns,
            "margin": margin,
            "present": present,
        })

    denom = max(1, len(tile_scores))
    hits = sum(1 for t in tile_scores if t["present"])

    return {
        "score": hits / denom,
        "mode": "crop_mask_patch_coverage",
        "hits": hits,
        "num_tiles": len(tile_scores),
        "pattern_margin": args.pattern_margin,
        "avg_margin": sum(float(t["margin"]) for t in tile_scores) / denom,
        "tile_scores": tile_scores,
    }


def score_axis_on_crop(session: requests.Session, crop_path: Path, task: Task, args: argparse.Namespace):
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


def save_winner_crops(records: list[dict[str, Any]], task: Task, out_root: Path, save_top_m: int, copy: bool):
    rows = [
        r for r in records
        if not r.get("skip_reason") and r.get("_temp_crop_path") and Path(str(r["_temp_crop_path"])).exists()
    ]
    rows = sorted(rows, key=lambda r: (fnum(r.get("combined_score")), fnum(r.get("similarity"))), reverse=True)

    odir = out_root / "winner_crops" / slug(task.query_id, 80) / f"{task.option_label}_{slug(task.option_semantic, 40)}__{slug(task.option_text, 80)}"
    odir.mkdir(parents=True, exist_ok=True)

    tsv = [
        f"query_id\t{task.query_id}",
        f"option\t{task.option_label}",
        f"option_text\t{task.option_text}",
        f"target\tcolor={task.target_color} pattern={task.target_pattern} garment={task.target_garment}",
        "",
        "rank\torig\tcombined\tpattern\tcolor\tgarment\tsimilarity\tmask_source\tfile",
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
            f"__mask_{slug(r.get('mask_source'), 24)}"
            f"{ext}"
        )
        dst = odir / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()

        if copy:
            shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)

        r["saved_crop_path"] = str(dst)
        saved.append(dst)

        tsv.append(
            f"{rank:03d}\t{int(r.get('original_rank') or 0):03d}\t"
            f"{fnum(r.get('combined_score')):.6f}\t{fnum(r.get('pattern_score')):.6f}\t"
            f"{fnum(r.get('color_score')):.6f}\t{fnum(r.get('garment_score')):.6f}\t"
            f"{fnum(r.get('similarity')):.6f}\t{r.get('mask_source')}\t{name}"
        )

    if len(tsv) == 6:
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
    p.add_argument("--output-root", type=Path, default=Path("data/retrieval/gdino_sam2_worn_qwenemb"))

    p.add_argument("--query-mode", choices=["original", "garment_first"], default="original")
    p.add_argument("--retrieval-k", type=int, default=100)
    p.add_argument("--score-top-n", type=int, default=100)
    p.add_argument("--save-top-m", type=int, default=1)

    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--options", default="A,B,C,D")

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--grounding-model", default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--sam2-model", default="facebook/sam2-hiera-tiny")

    p.add_argument("--box-threshold", type=float, default=0.15)
    p.add_argument("--text-threshold", type=float, default=0.15)
    p.add_argument("--min-box-area-ratio", type=float, default=0.001)
    p.add_argument("--max-box-area-ratio", type=float, default=0.95)
    p.add_argument("--box-expand", type=float, default=0.02)

    p.add_argument("--min-mask-area-ratio", type=float, default=0.001)
    p.add_argument("--max-mask-area-ratio", type=float, default=0.95)
    p.add_argument("--box-mask-fallback", action="store_true", default=True)

    p.add_argument("--short-side-tiles", type=int, default=8)
    p.add_argument("--min-coverage", type=float, default=0.05)
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
    print("top-k retrieval -> gDINO-tiny + SAM2-tiny box crop -> QwenEmb scoring -> winner crop")
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

    print("loading GroundingDINO...")
    processor, gdino = load_groundingdino(args.grounding_model, args.device)

    print("loading SAM2...")
    sam2 = load_sam2(args.sam2_model, args.device)

    session = requests.Session()

    raw_log = args.output_root / "raw_topk_results.jsonl"
    scored_log = args.output_root / "scored_results.jsonl"
    for p in [raw_log, scored_log]:
        if p.exists():
            p.unlink()

    with tempfile.TemporaryDirectory(prefix="gdino_sam2_worn_run_") as run_tmp:
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

                box, det_info = detect_best_box(
                    processor=processor,
                    model=gdino,
                    image=image,
                    garment=task.target_garment,
                    device=args.device,
                    args=args,
                )

                if box is None:
                    rec.update({
                        "skip_reason": "no_detection_box",
                        "mask_detection_info": det_info,
                        "combined_score": 0.0,
                        "pattern_score": 0.0,
                        "color_score": 0.0,
                        "garment_score": 0.0,
                    })
                    records.append(rec)
                    continue

                mask, used_box, mask_info = sam2_mask_or_box(sam2, image, box, args)
                if mask is None:
                    rec.update({
                        "skip_reason": "no_valid_sam2_mask",
                        "mask_detection_info": det_info,
                        "mask_info": mask_info,
                        "combined_score": 0.0,
                        "pattern_score": 0.0,
                        "color_score": 0.0,
                        "garment_score": 0.0,
                    })
                    records.append(rec)
                    continue

                crop, crop_mask, crop_xyxy = crop_image_and_mask(image, mask, used_box)
                crop_path = rec_tmp / "detected_crop.jpg"
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
                    "detected_box_xyxy": [float(v) for v in used_box],
                    "crop_xyxy": list(crop_xyxy),
                    "mask_source": mask_info.get("mask_source"),
                    "mask_detection_info": det_info,
                    "mask_info": mask_info,
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
                copy=True,
            )

            # Remove temp path before writing persistent JSON.
            for r in records:
                r.pop("_temp_crop_path", None)

            append_jsonl(raw_log, [
                {
                    "query_id": task.query_id,
                    "option_label": task.option_label,
                    "retrieval_query": q,
                    "num_retrieved": len(results),
                    "num_scored_valid": sum(1 for r in records if not r.get("skip_reason")),
                }
            ])
            append_jsonl(scored_log, records)

            if saved:
                top = records[0]
                print(
                    f"  top1 orig={top.get('original_rank')} "
                    f"comb={fnum(top.get('combined_score')):.4f} "
                    f"pat={fnum(top.get('pattern_score')):.4f} "
                    f"col={fnum(top.get('color_score')):.4f} "
                    f"gar={fnum(top.get('garment_score')):.4f}"
                )
            else:
                print("  no valid detected crop")

    print()
    print("Done.")
    print(f"raw log:     {raw_log}")
    print(f"scored log:  {scored_log}")
    print(f"winner dir:  {args.output_root / 'winner_crops'}")


if __name__ == "__main__":
    main()
