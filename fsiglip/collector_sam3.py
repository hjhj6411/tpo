#!/usr/bin/env python3
# Variant of collect_topk_sam3_fsiglip_patch_rank.py.
# Only the garment axis is changed: instead of a FashionSigLIP cosine similarity,
# the SAM3 crop is judged PASS/FAIL by a vision-language model served via vLLM
# (OpenAI-compatible /v1/chat/completions). garment_score = 1.0 (PASS) or 0.0 (FAIL).
# Pattern and color scoring use separate patch grids because patterns need
# larger visual context while colors benefit from denser local patches.


"""
# 터미널 3: 수집 실행
conda activate sam3
python fsiglip/collector_sam3.py \
  --plan-path data/options/option_plans.jsonl \
  --client-url http://127.0.0.1:1235/knn-service \
  --score-url http://127.0.0.1:1235/score-image-files \
  --vlm-url http://127.0.0.1:8001/v1/chat/completions \
  --vlm-model Qwen/Qwen3-VL-4B-Instruct \
  --output-root data/retrieval/sam3vlmfix \
  --pattern-short-side-tiles 5 \
  --color-short-side-tiles 15 \
  --patch-mask-fill local_mean \
  --limit 10 --options A,B,C,D --device cuda --force
"""
from __future__ import annotations

import argparse
import base64
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
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# ── variant-aware data root ───────────────────────────────────────────────
# POD_VARIANT redirects every data path to data_<variant>/ (configs/config.py).
# Collectors used to hardcode "data/", which silently read the OLD plans when a
# variant was active — no error, just the wrong dataset. Import the same config
# the generators and evaluators use so all of them agree.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from configs.config import OPTIONS_DIR, DATA_DIR
    DEFAULT_PLAN_PATH = OPTIONS_DIR / "option_plans.jsonl"
except Exception as _e:      # config import is best-effort; fall back to ./data
    print(f"  [warn] could not import configs.config ({_e}); defaulting to ./data")
    DATA_DIR = Path("data")
    DEFAULT_PLAN_PATH = DATA_DIR / "options" / "option_plans.jsonl"

import numpy as np
import requests
import torch
from PIL import Image, ImageDraw, ImageFont



OPTION_LABELS = ["A", "B", "C", "D"]

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
}

DEFAULT_COLORS = [
    "black", "white", "gray", "navy", "blue", "red", "pink",
    "orange", "yellow", "green", "brown", "beige", "purple",
]

DEFAULT_GARMENTS = [
    "blazer", "outerwear",
    "cardigan", "hoodie", "sweater", "sweatshirt",
    "tank top", "sleeveless top", "camisole top", "crop top",
    "t shirt", "tee", "tee shirt", "formal shirt", "dress shirt", "top",
    "windbreaker", "leather jacket", "puffer jacket", "fleece jacket", "trench coat",
    "mini skirt", "long skirt",
    "slacks", "pants", "trousers", "jeans", "shorts", "leggings",
    "dress", "gown",
]

TEXT_QUERY_GARMENT_VOCAB = [
    "t shirt", "tank top", "formal shirt",
    "sweatshirt", "sweater", "hoodie", "cardigan",
    "blazer", "windbreaker", "leather jacket", "puffer jacket",
    "fleece jacket", "trench coat",
    "jeans", "slacks", "shorts", "leggings",
    "dress", "mini skirt", "long skirt",
]

GARMENT_QUERY_TERMS = TEXT_QUERY_GARMENT_VOCAB + DEFAULT_GARMENTS


@dataclass(frozen=True)
class OptionTask:
    plan_idx: int
    plan_id: str
    query_id: str
    user_id: str
    scenario_id: str | None
    scenario_name: str | None
    active_axis: str | None
    query_type: str | None
    option_label: str
    option_semantic: str | None
    option_text: str
    option_attrs: dict[str, Any]
    target_color: str
    target_pattern: str
    target_garment: str
    query_garment: str
    vlm_garment_vocab: tuple[str, ...]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def slug(text: Any, max_len: int = 96) -> str:
    s = str(text or "")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")
    return (s or "unknown")[:max_len]


def norm(value: Any) -> str:
    return str(value or "").replace("_", " ").strip().lower()


def phrase_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", norm(value)).strip()


def unique_norm(items: list[Any] | tuple[Any, ...]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        v = norm(item)
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def extract_query_garment(text: str, fallback: str, candidates: list[str] | tuple[str, ...] = ()) -> str:
    """Extract only the garment axis from a composed query.

    Example: "plaid navy tank top" -> "tank top". This intentionally does not
    add umbrella prompts such as "top", "outerwear", "garment", or aliases not
    present in the query.
    """
    hay = f" {phrase_key(text)} "
    terms = unique_norm(list(candidates) + GARMENT_QUERY_TERMS + [fallback])
    terms = sorted(terms, key=lambda x: (len(phrase_key(x).split()), len(x)), reverse=True)
    for term in terms:
        needle = f" {phrase_key(term)} "
        if needle.strip() and needle in hay:
            return term
    return norm(fallback)


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


def make_tasks(plans: list[dict[str, Any]], options: list[str]) -> list[OptionTask]:
    tasks = []

    for plan_idx, plan in enumerate(plans):
        query_id = str(plan.get("query_id") or f"plan_{plan_idx:05d}")
        # one output folder per PLAN: a two-violation-axis query has two plans
        # whose options differ, so query_id folders would collide.
        plan_id = str(plan.get("plan_id") or query_id)
        user_id = str(plan.get("user_id") or "unknown_user")
        opts = plan.get("options") or {}
        garment_vocab = tuple(TEXT_QUERY_GARMENT_VOCAB)

        for label in options:
            opt = opts.get(label)
            if not isinstance(opt, dict):
                continue

            attrs = opt.get("attributes") or {}
            option_text = option_to_text(opt)
            target_garment = norm(attrs.get("garment_category"))
            query_garment = extract_query_garment(option_text, target_garment, garment_vocab)
            tasks.append(OptionTask(
                plan_idx=plan_idx,
                plan_id=plan_id,
                query_id=query_id,
                user_id=user_id,
                scenario_id=plan.get("scenario_id"),
                scenario_name=plan.get("scenario_name"),
                active_axis=plan.get("active_axis"),
                query_type=plan.get("query_type"),
                option_label=label,
                option_semantic=opt.get("label"),
                option_text=option_text,
                option_attrs=attrs,
                target_color=norm(attrs.get("color")),
                target_pattern=norm(attrs.get("pattern")),
                target_garment=target_garment,
                query_garment=query_garment,
                vlm_garment_vocab=garment_vocab,
            ))

    return tasks


def retrieval_query(task: OptionTask, mode: str) -> str:
    if mode == "original":
        return task.option_text

    if mode == "garment_first":
        parts = []
        # Prefer the alias-rendered garment from the search query (e.g.
        # "fleece jacket", not "fleece") — same order as sam3_garment_prompts.
        if task.query_garment or task.target_garment:
            parts.append(task.query_garment or task.target_garment)
        if task.target_pattern and task.target_pattern != "solid":
            parts.append(task.target_pattern)
        if task.target_color:
            parts.append(task.target_color)
        return ", ".join(parts) or task.option_text

    raise ValueError(f"unknown query mode: {mode}")


def endpoint(base: str, route: str) -> str:
    base = base.rstrip("/")
    for suffix in ["/knn-service", "/knn-service-ensemble", "/score-image-files"]:
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
            time.sleep(min(2.0 ** attempt, 8.0))
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
        timeout=args.timeout,
        retries=args.retries,
    )

    if not isinstance(data, list):
        raise RuntimeError(f"KNN response must be list, got {type(data).__name__}: {data}")

    return [x for x in data if isinstance(x, dict)]


def score_image_files(
    session: requests.Session,
    args: argparse.Namespace,
    texts: dict[str, str],
    image_paths: list[str],
    batch_size: int,
) -> list[dict[str, float]]:
    out = []

    for i in range(0, len(image_paths), batch_size):
        chunk = image_paths[i:i + batch_size]
        data = post_json(
            session,
            args.score_url,
            {
                "texts": texts,
                "image_paths": chunk,
            },
            timeout=args.score_timeout,
            retries=args.retries,
        )

        rows = data.get("scores") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            rows = []

        for row in rows:
            if isinstance(row, dict) and "error" not in row:
                out.append({k: float(v) for k, v in row.items() if isinstance(v, (int, float))})
            else:
                out.append({})

        while len(out) < i + len(chunk):
            out.append({})

    return out


def choose_extension(url: str, content_type: str | None) -> str:
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

    existing = [p for p in sorted(out_base.parent.glob(out_base.name + ".*")) if not p.name.endswith(".tmp")]
    if existing and not args.force:
        return existing[0], None

    last = None
    for attempt in range(args.retries + 1):
        try:
            with session.get(url, headers=DEFAULT_HEADERS, stream=True, timeout=args.download_timeout) as r:
                r.raise_for_status()
                ext = choose_extension(url, r.headers.get("Content-Type"))
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
                            raise RuntimeError(f"image exceeds max_image_bytes={args.max_image_bytes}")
                        f.write(chunk)

                tmp.replace(out)
                return out, None

        except Exception as e:
            last = e
            if attempt >= args.retries:
                break
            time.sleep(min(2.0 ** attempt, 8.0))

    return None, str(last)


def font(size: int):
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def load_sam3(device: str):
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(device=device, eval_mode=True)
    return Sam3Processor(model)


def to_numpy(x: Any) -> np.ndarray:
    if x is None:
        return np.asarray([])

    if torch.is_tensor(x):
        x = x.detach()
        if x.dtype in {torch.bfloat16, torch.float16}:
            x = x.float()
        return x.cpu().numpy()

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


def boxes_from_masks(masks: np.ndarray) -> np.ndarray:
    boxes = []

    for mask in masks:
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            boxes.append([0, 0, 1, 1])
        else:
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])

    return np.asarray(boxes, dtype=np.float32)


def normalize_boxes(boxes: Any, masks: np.ndarray) -> np.ndarray:
    if boxes is None:
        return boxes_from_masks(masks)

    arr = to_numpy(boxes)

    if arr.size == 0:
        return boxes_from_masks(masks)

    arr = arr.reshape(-1, 4).astype(np.float32)
    if len(arr) < len(masks):
        fallback = boxes_from_masks(masks)
        out = np.zeros((len(masks), 4), dtype=np.float32)
        for i in range(len(masks)):
            out[i] = arr[i] if i < len(arr) else fallback[i]
        return out

    return arr[: len(masks)]


def normalize_scores(scores: Any, n: int) -> np.ndarray:
    if scores is None:
        return np.ones((n,), dtype=np.float32)

    arr = to_numpy(scores).reshape(-1).astype(np.float32)
    if len(arr) < n:
        out = np.ones((n,), dtype=np.float32)
        out[: len(arr)] = arr
        return out

    return arr[:n]


def maybe_denormalize_box(box: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    w, h = image_size
    b = box.astype(np.float32).copy()
    if float(np.nanmax(b)) <= 1.5:
        b[[0, 2]] *= w
        b[[1, 3]] *= h
    return b


def ensure_mask_size(mask: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    w, h = image_size
    if mask.shape == (h, w):
        return mask.astype(bool)

    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    img = img.resize((w, h), resample=Image.NEAREST)
    return np.asarray(img) > 0


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


def sam3_garment_prompts(task: OptionTask) -> list[str]:
    g = norm(task.query_garment or task.target_garment)
    return [g] if g else ["clothing item"]


def run_sam3_all(processor, image: Image.Image, prompts: list[str], args: argparse.Namespace) -> list[dict[str, Any]]:
    amp_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if str(args.device).startswith("cuda") and torch.cuda.is_available()
        else nullcontext()
    )

    with torch.inference_mode(), amp_ctx:
        state = processor.set_image(image)

    candidates = []
    w, h = image.size

    for prompt in prompts:
        with torch.inference_mode(), amp_ctx:
            try:
                output = processor.set_text_prompt(state=state, prompt=prompt)
            except TypeError:
                output = processor.set_text_prompt(state, prompt)

        masks = normalize_masks(output.get("masks"))
        boxes = normalize_boxes(output.get("boxes"), masks)
        scores = normalize_scores(output.get("scores"), len(masks))

        for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
            mask = ensure_mask_size(mask, image.size)
            box = maybe_denormalize_box(box, image.size)

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

    candidates.sort(key=lambda r: r["score"], reverse=True)
    return candidates


def crop_image_and_mask(image: Image.Image, mask: np.ndarray, box: np.ndarray) -> tuple[Image.Image, np.ndarray, tuple[int, int, int, int]]:
    x1, y1, x2, y2 = box_to_int(box, image.size)
    crop = image.crop((x1, y1, x2, y2)).convert("RGB")
    crop_mask = mask[y1:y2, x1:x2].copy()
    return crop, crop_mask, (x1, y1, x2, y2)


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def ceil_to_multiple(x: int, m: int) -> int:
    return int(math.ceil(x / m) * m)


def patch_grid(w: int, h: int, short_side_tiles: int) -> list[tuple[int, int, int]]:
    side = max(1, int(math.ceil(min(w, h) / max(1, short_side_tiles))))
    padded_w = ceil_to_multiple(w, side)
    padded_h = ceil_to_multiple(h, side)
    return [(x, y, side) for y in range(0, padded_h, side) for x in range(0, padded_w, side)]


def fixed_patch_fill(fill_mode: str) -> tuple[int, int, int]:
    if fill_mode == "white":
        return (255, 255, 255)
    if fill_mode == "black":
        return (0, 0, 0)
    return (127, 127, 127)


def masked_tile(
    crop: Image.Image,
    crop_mask: np.ndarray,
    x: int,
    y: int,
    x2: int,
    y2: int,
    size: int,
    fill_mode: str,
) -> Image.Image:
    region = np.asarray(crop.crop((x, y, x2, y2)).convert("RGB"), dtype=np.uint8)
    region_mask = crop_mask[y:y2, x:x2].astype(bool)

    if fill_mode == "local_mean" and region_mask.any():
        fill = tuple(int(round(v)) for v in region[region_mask].mean(axis=0))
    else:
        fill = fixed_patch_fill(fill_mode)

    tile = np.full((size, size, 3), fill, dtype=np.uint8)
    tile_region = tile[: y2 - y, : x2 - x]
    tile_region[region_mask] = region[region_mask]
    return Image.fromarray(tile, mode="RGB")


def masked_crop_preview(crop: Image.Image, crop_mask: np.ndarray, fill_mode: str) -> Image.Image:
    crop = crop.convert("RGB")
    arr = np.asarray(crop, dtype=np.uint8)
    mask = crop_mask.astype(bool)

    if fill_mode == "local_mean" and mask.any():
        fill = tuple(int(round(v)) for v in arr[mask].mean(axis=0))
    else:
        fill = fixed_patch_fill(fill_mode)

    out = np.full(arr.shape, fill, dtype=np.uint8)
    out[mask] = arr[mask]
    return Image.fromarray(out, mode="RGB")


def make_patch_files(
    crop: Image.Image,
    crop_mask: np.ndarray,
    args: argparse.Namespace,
    patch_dir: Path,
    short_side_tiles: int,
    axis: str,
) -> list[dict[str, Any]]:
    crop = crop.convert("RGB")
    w, h = crop.size
    patch_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, (x, y, s) in enumerate(patch_grid(w, h, short_side_tiles)):
        x2 = min(x + s, w)
        y2 = min(y + s, h)
        if x2 <= x or y2 <= y:
            continue

        cov = float(crop_mask[y:y2, x:x2].sum()) / float(s * s)
        if cov < args.min_coverage:
            continue

        tile = masked_tile(
            crop=crop,
            crop_mask=crop_mask,
            x=x,
            y=y,
            x2=x2,
            y2=y2,
            size=s,
            fill_mode=args.patch_mask_fill,
        )

        path = patch_dir / f"tile_{i:04d}__x{x}_y{y}_s{s}_cov{cov:.3f}.jpg"
        tile.save(path, quality=92)

        rows.append({
            "index": i,
            "axis": axis,
            "x": x,
            "y": y,
            "size": s,
            "short_side_tiles": int(short_side_tiles),
            "mask_coverage": cov,
            "mask_fill": args.patch_mask_fill,
            "path": str(path),
        })

    return rows


def draw_patch_overlay(crop: Image.Image, crop_mask: np.ndarray, patches: list[dict[str, Any]], out_path: Path, alpha: int = 95) -> None:
    crop = crop.convert("RGB")
    w, h = crop.size

    base = crop.convert("RGBA")
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[crop_mask] = np.array([255, 0, 0, alpha], dtype=np.uint8)
    layer = Image.fromarray(rgba, mode="RGBA")
    out = Image.alpha_composite(base, layer)

    draw = ImageDraw.Draw(out)
    f = font(12)
    for p in patches:
        x = int(p["x"])
        y = int(p["y"])
        s = int(p["size"])
        cov = float(p["mask_coverage"])
        draw.rectangle([x, y, x + s, y + s], outline=(255, 255, 0, 255), width=2)
        draw.text((x + 3, y + 3), f"{cov:.2f}", fill=(255, 255, 0, 255), font=f)

    out.convert("RGB").save(out_path, quality=95)


PATTERN_VOCAB = [
    "plain solid-colored",
    "striped",
    "plaid",
    "checkered",
    "polka dot",
    "floral",
    "camouflage",
    "houndstooth",
    "animal print",
]


def canonical_pattern(pattern: str) -> str:
    p = norm(pattern)
    if p in {"plain", "solid"}:
        return "plain solid-colored"
    if p == "dotted":
        return "polka dot"
    if p == "checked":
        return "checkered"
    return p


def score_patch_vocab(
    session: requests.Session,
    args: argparse.Namespace,
    axis: str,
    target: str,
    vocab: list[str],
    prompt_template: str,
    margin: float,
    patches: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Closed-vocabulary patch voting; the same rule for every target:

        score = (# patches whose argmax anchor is the target) / (# patches)

    One anchor per vocabulary entry, no per-class aliases or hard negatives.
    """
    t = norm(target)
    if not patches:
        return {"score": 0.0, "mode": "no_valid_patches", "tile_scores": []}

    vocab = [norm(v) for v in vocab]
    if t not in vocab:
        vocab = [t] + vocab

    texts = {f"{axis}::{v}": prompt_template.format(v) for v in vocab}
    rows = score_image_files(
        session=session,
        args=args,
        texts=texts,
        image_paths=[x["path"] for x in patches],
        batch_size=args.score_batch,
    )

    target_key = f"{axis}::{t}"
    tile_scores = []

    for patch, row in zip(patches, rows):
        if not row:
            pred = None
            target_score = 0.0
            other_score = 0.0
            tile_margin = -1.0
            hit = False
        else:
            pred_key = max(row, key=lambda k: float(row[k]))
            pred = pred_key.split("::", 1)[-1]
            target_score = float(row.get(target_key, 0.0))
            other_score = max([float(v) for k, v in row.items() if k != target_key] or [0.0])
            tile_margin = target_score - other_score
            hit = (pred_key == target_key) and (tile_margin >= margin)

        tile_scores.append({
            "index": patch["index"],
            "x": patch["x"],
            "y": patch["y"],
            "size": patch["size"],
            "short_side_tiles": patch.get("short_side_tiles"),
            "mask_coverage": float(patch["mask_coverage"]),
            "mask_fill": patch.get("mask_fill"),
            "target": t,
            "pred": pred,
            "target_score": target_score,
            "strongest_other_score": other_score,
            "margin": tile_margin,
            "hit": bool(hit),
            "path": patch["path"],
        })

    hits = sum(1 for x in tile_scores if x["hit"])
    score = hits / len(tile_scores)

    from collections import Counter
    pred_counts = Counter(x["pred"] for x in tile_scores if x.get("pred"))

    return {
        "score": score,
        "mode": f"patch_vocab_argmax_{axis}",
        "target": t,
        "vocab": vocab,
        "margin": margin,
        "formula": "num_target_patches / num_patches",
        "hits": hits,
        "num_tiles": len(tile_scores),
        "pred_counts": dict(pred_counts),
        "top_pred": pred_counts.most_common(1)[0][0] if pred_counts else None,
        "tile_scores": tile_scores,
    }


def score_pattern_patches(session: requests.Session, args: argparse.Namespace, pattern: str, patches: list[dict[str, Any]]) -> dict[str, Any]:
    p = norm(pattern)
    if not p:
        return {"score": 1.0, "mode": "missing_pattern_target", "tile_scores": []}

    return score_patch_vocab(
        session=session,
        args=args,
        axis="pattern",
        target=canonical_pattern(p),
        vocab=PATTERN_VOCAB,
        prompt_template="a close-up of {} clothing fabric",
        margin=args.pattern_margin,
        patches=patches,
    )


def score_color_patches(
    session: requests.Session,
    args: argparse.Namespace,
    color: str,
    patches: list[dict[str, Any]],
) -> dict[str, Any]:
    c = norm(color)
    if not c:
        return {"score": 1.0, "mode": "missing_color_target", "tile_scores": []}

    return score_patch_vocab(
        session=session,
        args=args,
        axis="color",
        target=c,
        vocab=DEFAULT_COLORS,
        prompt_template="a close-up of {} clothing fabric",
        margin=args.color_margin,
        patches=patches,
    )


# Closed garment vocabulary used by the text-query generation pipeline.
# VLM classification uses this full list, not only the garments present in one
# option set.
GARMENT_VOCAB = TEXT_QUERY_GARMENT_VOCAB

# Symmetric equivalence classes: only true synonyms are merged, and every entry
# is either a valid target garment or a plausible VLM synonym for one. The
# visually confusable outerwear categories are intentionally NOT merged, so the
# VLM verdict actually discriminates them.
GARMENT_EQUIV_GROUPS = [
    {"tank top", "sleeveless top", "camisole top", "camisole", "crop top"},
    {"t shirt", "tee", "tee shirt"},
    {"formal shirt", "dress shirt"},
    {"jeans", "denim pants", "denim"},
    {"dress", "gown"},
    {"sweater", "pullover", "knit sweater", "cardigan", "knitwear"},
    {"hoodie", "sweatshirt"},
    {"fleece", "fleece jacket"},
    {"slacks", "pants", "trousers"},
]


def garment_equivalence_set(garment: str, strict: bool) -> set[str]:
    g = norm(garment)
    if strict:
        return {g}
    members = {g}
    for grp in GARMENT_EQUIV_GROUPS:
        if g in grp:
            members |= grp
    return members


def assert_vocab_covers_targets(tasks: list["OptionTask"]) -> None:
    """Fail loudly if a target garment is outside the text-query garment vocab."""
    uncovered: dict[str, int] = {}
    for t in tasks:
        g = norm(t.query_garment or t.target_garment)
        if not g:
            continue
        vocab = set(t.vlm_garment_vocab or GARMENT_VOCAB)
        if not (garment_equivalence_set(g, strict=False) & vocab):
            uncovered[g] = uncovered.get(g, 0) + 1
    if uncovered:
        detail = ", ".join(f"{g!r} x{n}" for g, n in sorted(uncovered.items(), key=lambda kv: -kv[1]))
        raise SystemExit(
            "VLM garment candidate pools are not unified with the text-query garment categories; "
            "these targets can never pass the garment axis: " + detail + ". "
            "Check option search_query/garment_category extraction."
        )


VLM_GARMENT_PROMPT = (
    "You are a fashion garment classifier. Look only at the single main clothing "
    "item worn on the body in the image, and ignore the background, the person, "
    "the pose, and any accessories.\n\n"
    "Classify that garment into exactly ONE of these categories:\n"
    "{options}\n\n"
    "Rules:\n"
    "- Outerwear, keep these distinct: a trench coat is a long belted "
    "raincoat-style coat; a fleece jacket is a soft brushed insulating jacket; a puffer jacket is "
    "a quilted insulated jacket; a windbreaker is a thin lightweight nylon "
    "shell; a leather jacket is a leather outerwear jacket; a blazer is a "
    "structured tailored jacket worn on its own.\n"
    "{other_rule}"
    "Answer with only the category name, exactly as written above, and nothing else."
)


def vlm_prompt_for_vocab(vocab: list[str]) -> str:
    options_block = "\n".join(f"- {c}" for c in vocab)
    other_rule = (
        "- If it clearly does not fit any listed category, answer \"other\".\n"
        if "other" in vocab else
        "- Choose from only the listed query garment categories; do not introduce another category.\n"
    )
    return VLM_GARMENT_PROMPT.format(options=options_block, other_rule=other_rule)


def encode_image_data_uri(path: Path) -> str:
    ext = Path(path).suffix.lower().lstrip(".") or "jpeg"
    if ext == "jpg":
        ext = "jpeg"
    b64 = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/{ext};base64,{b64}"


def parse_garment_category(text: str, vocab: list[str]) -> str | None:
    """Return the vocabulary entry the VLM named. Longest substring match wins
    so that 'tank top' beats 'top' and 'suit jacket' beats 'jacket'."""
    t = f" {phrase_key(text)} "
    if not t:
        return None
    matches = [v for v in vocab if f" {phrase_key(v)} " in t]
    if not matches:
        return None
    return max(matches, key=lambda v: (len(phrase_key(v).split()), len(v)))


def score_garment_vlm(
    session: requests.Session,
    args: argparse.Namespace,
    crop_path: Path,
    garment: str,
    vocab: list[str] | tuple[str, ...] | None = None,
) -> tuple[float, dict[str, Any]]:
    """
    Garment score via a VLM (served through vLLM's OpenAI-compatible chat
    endpoint). The model CLASSIFIES the SAM3 crop into one closed garment
    category; this code then decides PASS/FAIL by comparing that predicted
    category to the target's equivalence set.

    PASS -> 1.0, FAIL -> 0.0. On unparseable / failed calls the crop is treated
    as FAIL (0.0) so a wrong garment cannot survive on color/pattern alone; the
    raw reason is always recorded for auditing.
    """
    g = norm(garment)
    if not g:
        return 1.0, {"mode": "missing_garment_target"}

    vocab = unique_norm(list(vocab or [])) or GARMENT_VOCAB
    if g and g not in vocab:
        vocab = unique_norm(list(vocab) + [g])

    try:
        data_uri = encode_image_data_uri(crop_path)
    except Exception as e:
        return 0.0, {"mode": "vlm_garment_image_encode_failed", "target": g, "error": str(e)}

    prompt = vlm_prompt_for_vocab(vocab)
    payload = {
        "model": args.vlm_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
        "max_tokens": args.vlm_max_tokens,
        "temperature": args.vlm_temperature,
    }

    try:
        data = post_json(session, args.vlm_url, payload, timeout=args.vlm_timeout, retries=args.vlm_retries)
        content = data["choices"][0]["message"]["content"]
    except Exception as e:
        return 0.0, {"mode": "vlm_garment_request_failed", "target": g, "error": str(e)}

    pred = parse_garment_category(content, vocab)
    if pred is None:
        return 0.0, {
            "mode": "vlm_garment_unparseable",
            "target": g,
            "vocab": vocab,
            "raw_response": str(content)[:500],
            "model": args.vlm_model,
        }

    equiv = garment_equivalence_set(g, args.garment_strict)
    verdict = pred in equiv
    score = 1.0 if verdict else 0.0
    return score, {
        "mode": "vlm_garment_classify",
        "target": g,
        "predicted": pred,
        "verdict": "PASS" if verdict else "FAIL",
        "score": score,
        "strict": bool(args.garment_strict),
        "vocab": vocab,
        "equivalence_set": sorted(equiv),
        "raw_response": str(content)[:500],
        "model": args.vlm_model,
    }


def task_dirs(args: argparse.Namespace, task: OptionTask) -> tuple[Path, Path, Path]:
    q = slug(task.plan_id, 80)
    opt = f"{task.option_label}_{slug(task.option_semantic or 'option', 32)}__{slug(task.option_text, 72)}"
    return (
        args.output_root / "topk_images" / q / opt,
        args.output_root / "topk_patches" / q / opt,
        args.output_root / "ranked_images" / q / opt,
    )


def save_ranked_images(records: list[dict[str, Any]], task: OptionTask, args: argparse.Namespace) -> None:
    _, _, rdir = task_dirs(args, task)
    rdir.mkdir(parents=True, exist_ok=True)

    rows = sorted(
        records,
        key=lambda r: (fnum(r.get("combined_score")), fnum(r.get("similarity"))),
        reverse=True,
    )

    tsv = [
        f"plan_id\t{task.plan_id}",
        f"query_id\t{task.query_id}",
        f"option\t{task.option_label}",
        f"option_text\t{task.option_text}",
        f"target\tcolor={task.target_color} pattern={task.target_pattern} garment={task.target_garment} query_garment={task.query_garment}",
        f"vlm_garment_vocab\t{','.join(task.vlm_garment_vocab)}",
        "",
        "rank\torig_rank\tcombined\tpattern\tcolor\tgarment\tsimilarity\tsam3_prompt\tskip_reason\tfile",
    ]

    for rank, rec in enumerate(rows, start=1):
        src_path = rec.get("crop_path") or rec.get("local_path")
        if not src_path or not Path(src_path).exists():
            continue

        src = Path(src_path)
        ext = src.suffix or ".jpg"
        name = (
            f"{rank:03d}"
            f"__orig_{int(rec.get('original_rank') or 0):03d}"
            f"__comb_{fnum(rec.get('combined_score')):.4f}"
            f"__pat_{fnum(rec.get('pattern_score')):.4f}"
            f"__col_{fnum(rec.get('color_score')):.4f}"
            f"__gar_{fnum(rec.get('garment_score')):.4f}"
            f"__sim_{fnum(rec.get('similarity')):.4f}"
            f"__sam3_{slug(rec.get('sam3_prompt') or 'none', 20)}"
            f"__{slug(rec.get('skip_reason') or 'ok', 24)}"
            f"{ext}"
        )
        dst = rdir / name
        shutil.copy2(src, dst)

        tsv.append(
            f"{rank:03d}\t{int(rec.get('original_rank') or 0):03d}\t"
            f"{fnum(rec.get('combined_score')):.6f}\t"
            f"{fnum(rec.get('pattern_score')):.6f}\t"
            f"{fnum(rec.get('color_score')):.6f}\t"
            f"{fnum(rec.get('garment_score')):.6f}\t"
            f"{fnum(rec.get('similarity')):.6f}\t"
            f"{rec.get('sam3_prompt')}\t"
            f"{rec.get('skip_reason')}\t"
            f"{name}"
        )

    (rdir / "_ranking.tsv").write_text("\n".join(tsv) + "\n", encoding="utf-8")
    print(f"[ranked] {rdir}")


def process_one_record(
    session: requests.Session,
    sam3_processor,
    task: OptionTask,
    rec: dict[str, Any],
    args: argparse.Namespace,
    patch_root: Path,
) -> dict[str, Any]:
    local = rec.get("local_path")
    if not local or not Path(local).exists():
        rec.update({
            "skip_reason": "missing_local_image",
            "pattern_score": 0.0,
            "color_score": 0.0,
            "garment_score": 0.0,
            "combined_score": 0.0,
        })
        return rec

    rank = int(rec.get("original_rank") or 0)
    rank_dir = patch_root / f"rank_{rank:03d}"
    rank_dir.mkdir(parents=True, exist_ok=True)

    try:
        image = Image.open(local).convert("RGB")
    except Exception as e:
        rec.update({
            "skip_reason": f"image_open_failed: {e}",
            "pattern_score": 0.0,
            "color_score": 0.0,
            "garment_score": 0.0,
            "combined_score": 0.0,
        })
        return rec

    prompts = sam3_garment_prompts(task)
    try:
        candidates = run_sam3_all(sam3_processor, image, prompts, args)
    except Exception as e:
        rec.update({
            "skip_reason": f"sam3_failed: {e}",
            "pattern_score": 0.0,
            "color_score": 0.0,
            "garment_score": 0.0,
            "combined_score": 0.0,
            "sam3_prompts": prompts,
        })
        return rec

    cand_json = []
    for c in candidates[:10]:
        cand_json.append({
            "prompt": c["prompt"],
            "index": c["index"],
            "score": c["score"],
            "mask_area_ratio": c["mask_area_ratio"],
            "box_area_ratio": c["box_area_ratio"],
            "box_xyxy": [float(v) for v in c["box"]],
        })

    if not candidates:
        write_json(rank_dir / "skip.json", {
            "skip_reason": "no_sam3_mask",
            "prompts": prompts,
            "local_path": local,
        })
        rec.update({
            "skip_reason": "no_sam3_mask",
            "pattern_score": 0.0,
            "color_score": 0.0,
            "garment_score": 0.0,
            "combined_score": 0.0,
            "sam3_prompts": prompts,
            "sam3_candidates": [],
        })
        return rec

    best = candidates[0]
    box = expand_box(best["box"], image.size, args.box_expand)
    mask = best["mask"]

    crop, crop_mask, crop_xyxy = crop_image_and_mask(image, mask, box)

    crop_path = rank_dir / "crop.jpg"
    masked_crop_path = rank_dir / "masked_crop.jpg"
    mask_path = rank_dir / "mask.png"
    pattern_patch_overlay_path = rank_dir / "patch_overlay.jpg"
    color_patch_overlay_path = rank_dir / "color_patch_overlay.jpg"
    pattern_patch_dir = rank_dir / "patches"
    color_patch_dir = rank_dir / "color_patches"

    crop.save(crop_path, quality=95)
    masked_crop_preview(crop, crop_mask, args.patch_mask_fill).save(masked_crop_path, quality=95)
    save_mask(crop_mask, mask_path)

    pattern_patches = make_patch_files(
        crop=crop,
        crop_mask=crop_mask,
        args=args,
        patch_dir=pattern_patch_dir,
        short_side_tiles=args.pattern_short_side_tiles,
        axis="pattern",
    )
    color_patches = make_patch_files(
        crop=crop,
        crop_mask=crop_mask,
        args=args,
        patch_dir=color_patch_dir,
        short_side_tiles=args.color_short_side_tiles,
        axis="color",
    )
    draw_patch_overlay(crop, crop_mask, pattern_patches, pattern_patch_overlay_path)
    draw_patch_overlay(crop, crop_mask, color_patches, color_patch_overlay_path)

    pattern_info = score_pattern_patches(session, args, task.target_pattern, pattern_patches)
    pattern_score = float(pattern_info.get("score") or 0.0)

    color_info = score_color_patches(
        session=session,
        args=args,
        color=task.target_color,
        patches=color_patches,
    )
    color_score = float(color_info.get("score") or 0.0)

    garment_score, garment_info = score_garment_vlm(
        session=session,
        args=args,
        crop_path=crop_path,
        garment=task.query_garment or task.target_garment,
        vocab=task.vlm_garment_vocab,
    )
    garment_score = float(garment_score)

    # A VLM infra error (timeout / connection / unparseable) is NOT a genuine
    # "wrong garment" FAIL. Keep the two outcomes distinct so a server hiccup
    # cannot silently discard a correct garment.
    garment_error_modes = {
        "vlm_garment_request_failed",
        "vlm_garment_image_encode_failed",
        "vlm_garment_unparseable",
    }
    garment_is_error = garment_info.get("mode") in garment_error_modes

    if garment_is_error and args.vlm_error_policy == "uncertain":
        # Do not let a missing judgement kill the candidate; rank by pattern*color.
        garment_score = 1.0
        final_skip_reason = "garment_uncertain"
    elif garment_is_error:
        garment_score = 0.0
        final_skip_reason = "garment_vlm_error"
    elif garment_score <= 0.0:
        final_skip_reason = "garment_fail" if args.garment_fail_skip else None
    else:
        final_skip_reason = None

    combined = math.sqrt(max(0.0, pattern_score * color_score * garment_score))

    # Hard filter: an option whose pattern or color barely matches is a wrong
    # option no matter how good the other axes look. Force its score to 0 and
    # flag it as skipped (unless a garment reason already claimed the skip slot).
    attr_below = []
    if args.attr_min_score > 0.0:
        if pattern_score <= args.attr_min_score:
            attr_below.append("pattern")
        if color_score <= args.attr_min_score:
            attr_below.append("color")
    if attr_below:
        combined = 0.0
        if final_skip_reason is None:
            final_skip_reason = "_".join(attr_below) + "_below_min"

    write_json(rank_dir / "sam3_candidates.json", cand_json)
    write_json(rank_dir / "patch_scores.json", pattern_info)
    write_json(rank_dir / "pattern_patch_scores.json", pattern_info)
    write_json(rank_dir / "color_patch_scores.json", color_info)
    write_json(rank_dir / "score.json", {
        "pattern_score": pattern_score,
        "color_score": color_score,
        "garment_score": garment_score,
        "combined_score": combined,
        "sam3_prompt": best["prompt"],
        "query_garment": task.query_garment,
        "vlm_garment_vocab": list(task.vlm_garment_vocab),
        "pattern_short_side_tiles": args.pattern_short_side_tiles,
        "color_short_side_tiles": args.color_short_side_tiles,
        "pattern_num_patches": len(pattern_patches),
        "color_num_patches": len(color_patches),
        "patch_mask_fill": args.patch_mask_fill,
        "crop_xyxy": list(crop_xyxy),
        "masked_crop_path": str(masked_crop_path),
        "color_score_raw": color_info,
        "garment_score_raw": garment_info,
    })

    rec.update({
        "skip_reason": final_skip_reason,
        "sam3_prompt": best["prompt"],
        "sam3_score": best["score"],
        "sam3_prompts": prompts,
        "sam3_candidates": cand_json,
        "crop_xyxy": list(crop_xyxy),
        "crop_path": str(crop_path),
        "masked_crop_path": str(masked_crop_path),
        "mask_path": str(mask_path),
        "patch_overlay_path": str(pattern_patch_overlay_path),
        "patch_dir": str(pattern_patch_dir),
        "num_patches": len(pattern_patches),
        "pattern_patch_overlay_path": str(pattern_patch_overlay_path),
        "color_patch_overlay_path": str(color_patch_overlay_path),
        "pattern_patch_dir": str(pattern_patch_dir),
        "color_patch_dir": str(color_patch_dir),
        "pattern_num_patches": len(pattern_patches),
        "color_num_patches": len(color_patches),
        "pattern_short_side_tiles": args.pattern_short_side_tiles,
        "color_short_side_tiles": args.color_short_side_tiles,
        "patch_mask_fill": args.patch_mask_fill,
        "pattern_score": pattern_score,
        "color_score": color_score,
        "garment_score": garment_score,
        "combined_score": combined,
        "pattern_score_raw": pattern_info,
        "color_score_raw": color_info,
        "garment_score_raw": garment_info,
        "rerank_formula": "sqrt(pattern_score * color_score * garment_score)",
    })
    return rec


def collect_task(session: requests.Session, sam3_processor, task: OptionTask, args: argparse.Namespace) -> list[dict[str, Any]]:
    img_dir, patch_dir, _ = task_dirs(args, task)
    img_dir.mkdir(parents=True, exist_ok=True)
    patch_dir.mkdir(parents=True, exist_ok=True)

    q = retrieval_query(task, args.query_mode)
    results = query_knn(session, args, q)

    if args.score_top_n > 0:
        results = results[: args.score_top_n]

    records = []

    for rank, item in enumerate(results, start=1):
        url = str(item.get("image_url") or item.get("url") or "")
        key = str(item.get("key") or f"rank_{rank:03d}")
        sim = fnum(item.get("similarity"))

        out_base = img_dir / f"rank_{rank:03d}__sim_{sim:.4f}__{slug(key, 80)}"
        local_path, download_error = download_image(session, url, out_base, args)

        rec = {
            "plan_idx": task.plan_idx,
            "plan_id": task.plan_id,
            "query_id": task.query_id,
            "user_id": task.user_id,
            "scenario_id": task.scenario_id,
            "scenario_name": task.scenario_name,
            "active_axis": task.active_axis,
            "query_type": task.query_type,
            "option_label": task.option_label,
            "option_semantic": task.option_semantic,
            "option_text": task.option_text,
            "retrieval_query": q,
            "option_attrs": task.option_attrs,
            "target_color": task.target_color,
            "target_pattern": task.target_pattern,
            "target_garment": task.target_garment,
            "query_garment": task.query_garment,
            "vlm_garment_vocab": list(task.vlm_garment_vocab),
            "original_rank": rank,
            "similarity": item.get("similarity"),
            "url": url,
            "key": key,
            "caption": item.get("caption") or item.get("title") or "",
            "source": item.get("source"),
            "local_path": str(local_path) if local_path else None,
            "download_error": download_error,
            "raw": item,
        }

        if download_error or local_path is None:
            rec.update({
                "skip_reason": f"download_failed: {download_error}",
                "pattern_score": 0.0,
                "color_score": 0.0,
                "garment_score": 0.0,
                "combined_score": 0.0,
            })
        else:
            rec = process_one_record(session, sam3_processor, task, rec, args, patch_dir)

        records.append(rec)

        if rank % args.progress_every == 0 or rank == len(results):
            valid = sum(1 for r in records if not r.get("skip_reason"))
            print(f"  processed {rank}/{len(results)} valid={valid}")

    rows = sorted(
        records,
        key=lambda r: (fnum(r.get("combined_score")), fnum(r.get("similarity"))),
        reverse=True,
    )
    for rerank, rec in enumerate(rows, start=1):
        rec["rerank"] = rerank

    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--plan-path", type=Path, default=DEFAULT_PLAN_PATH,
                   help="option plans; default follows POD_VARIANT")
    p.add_argument("--client-url", default="http://127.0.0.1:1235/knn-service")
    p.add_argument("--score-url", default="")
    p.add_argument("--index-name", default="pod_fashion")
    p.add_argument("--output-root", type=Path, default=DATA_DIR / "retrieval/sam3_fsiglip_patch_rank",
                   help="output root; default follows POD_VARIANT")

    p.add_argument("--query-mode", choices=["original", "garment_first"], default="original")
    p.add_argument("--retrieval-k", type=int, default=50)
    p.add_argument("--score-top-n", type=int, default=20)

    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--options", default="A,B,C,D")

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--min-mask-area-ratio", type=float, default=0.000001)
    p.add_argument("--max-mask-area-ratio", type=float, default=1.0)
    p.add_argument("--min-box-area-ratio", type=float, default=0.000001)
    p.add_argument("--max-box-area-ratio", type=float, default=1.0)
    p.add_argument("--box-expand", type=float, default=0.02)

    p.add_argument(
        "--short-side-tiles",
        type=int,
        default=8,
        help="Backwards-compatible default patch density for both axes when axis-specific values are omitted.",
    )
    p.add_argument(
        "--pattern-short-side-tiles",
        type=int,
        default=None,
        help="Pattern patch grid density. Smaller value means larger patches; defaults to --short-side-tiles.",
    )
    p.add_argument(
        "--color-short-side-tiles",
        type=int,
        default=None,
        help="Color patch grid density. Larger value means smaller/denser patches; defaults to --short-side-tiles.",
    )
    p.add_argument("--min-coverage", type=float, default=0.20)
    p.add_argument(
        "--patch-mask-fill",
        choices=["local_mean", "gray", "white", "black"],
        default="local_mean",
        help="Fill color for pixels outside the SAM3 mask before saving patch tiles. "
             "local_mean uses each tile's masked garment pixels to avoid background-color bias.",
    )
    p.add_argument("--pattern-margin", type=float, default=0.0)
    p.add_argument("--color-margin", type=float, default=0.005)
    p.add_argument(
        "--color-mode",
        choices=["patch", "global", "both"],
        default="patch",
        help="unused (color is always patch-based now); accepted so existing launch scripts keep working.",
    )

    # VLM garment judge (vLLM OpenAI-compatible chat endpoint).
    p.add_argument("--vlm-url", default="http://127.0.0.1:8000/v1/chat/completions",
                   help="vLLM OpenAI-compatible chat-completions endpoint used for garment classification.")
    p.add_argument("--vlm-model", default="Qwen/Qwen2.5-VL-7B-Instruct",
                   help="Model name as served by vLLM (must match --served-model-name).")
    p.add_argument("--vlm-max-tokens", type=int, default=16)
    p.add_argument("--vlm-temperature", type=float, default=0.0)
    p.add_argument("--vlm-timeout", type=float, default=120.0)
    p.add_argument("--vlm-retries", type=int, default=4,
                   help="Retries (with backoff) for a VLM call before it counts as an error.")
    p.add_argument("--vlm-error-policy", choices=["skip", "uncertain"], default="skip",
                   help="How to treat a VLM infra error (timeout/connection/unparseable), which is "
                        "NOT a genuine wrong-garment FAIL. skip: drop with skip_reason=garment_vlm_error "
                        "(re-runnable, never a false PASS). uncertain: keep the candidate, rank it by "
                        "pattern*color only, skip_reason=garment_uncertain.")
    p.add_argument("--garment-strict", action="store_true",
                   help="PASS only if the VLM's predicted category equals the target exactly. "
                        "Default (off) also accepts true synonyms (e.g. tank top/camisole), "
                        "but never merges coat/jacket/blazer.")
    p.add_argument("--garment-fail-skip", action="store_true", default=True,
                   help="Mark garment-FAIL crops with skip_reason=garment_fail (default on).")
    p.add_argument("--no-garment-fail-skip", dest="garment_fail_skip", action="store_false")
    p.add_argument("--attr-min-score", type=float, default=0.2,
                   help="Hard filter: if the pattern or color score is <= this threshold, the "
                        "candidate is filtered out (combined_score forced to 0, "
                        "skip_reason=pattern/color_below_min). Set to 0 to disable.")

    p.add_argument("--score-batch", type=int, default=128)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--score-timeout", type=float, default=120.0)
    p.add_argument("--download-timeout", type=float, default=30.0)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--max-image-bytes", type=int, default=20_000_000)
    p.add_argument("--progress-every", type=int, default=10)

    p.add_argument(
        "--print-ranking-n",
        type=int,
        default=20,
        help="Print top-N reranked rows to terminal for each option. Use 0 to disable.",
    )

    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.score_url = args.score_url or endpoint(args.client_url, "/score-image-files")
    if args.pattern_short_side_tiles is None:
        args.pattern_short_side_tiles = args.short_side_tiles
    if args.color_short_side_tiles is None:
        args.color_short_side_tiles = args.short_side_tiles
    if args.pattern_short_side_tiles <= 0:
        raise SystemExit("--pattern-short-side-tiles must be > 0")
    if args.color_short_side_tiles <= 0:
        raise SystemExit("--color-short-side-tiles must be > 0")

    options = [x.strip().upper() for x in args.options.split(",") if x.strip()]
    bad = [x for x in options if x not in OPTION_LABELS]
    if bad:
        raise SystemExit(f"bad option labels: {bad}")

    if args.force and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    plans = load_jsonl(args.plan_path)

    if getattr(args, "query_id", ""):
        wanted = {x.strip() for x in str(args.query_id).split(",") if x.strip()}
        plans = [p for p in plans if str(p.get("query_id") or "") in wanted]
        if not plans:
            raise SystemExit(f"No plans matched --query-id={args.query_id!r}")
    else:
        if args.offset:
            plans = plans[args.offset:]
        if args.limit > 0:
            plans = plans[: args.limit]

    tasks = make_tasks(plans, options)
    assert_vocab_covers_targets(tasks)

    print("=" * 96)
    print("FashionSigLIP top-k -> SAM3 garment mask -> FSigLIP patch/global ranking")
    print("=" * 96)
    print(f"plans:       {len(plans)}")
    print(f"tasks:       {len(tasks)}")
    print(f"client_url:  {args.client_url}")
    print(f"score_url:   {args.score_url}")
    print(f"retrieval_k: {args.retrieval_k}")
    print(f"score_top_n: {args.score_top_n}")
    print(f"pattern_tiles: {args.pattern_short_side_tiles}")
    print(f"color_tiles:   {args.color_short_side_tiles}")
    print(f"patch_mask_fill: {args.patch_mask_fill}")
    print(f"output:      {args.output_root}")

    write_json(args.output_root / "run_config.json", vars(args))

    print("loading SAM3...")
    sam3_processor = load_sam3(args.device)

    session = requests.Session()

    raw_log = args.output_root / "raw_topk_results.jsonl"
    scored_log = args.output_root / "scored_results.jsonl"
    for path in [raw_log, scored_log]:
        if path.exists():
            path.unlink()

    for i, task in enumerate(tasks, start=1):
        q = retrieval_query(task, args.query_mode)
        print()
        print(
            f"[{i}/{len(tasks)}] {task.plan_id} {task.option_label} | "
            f"query={q} | mask={task.query_garment} | vlm={list(task.vlm_garment_vocab)}"
        )

        records = collect_task(session, sam3_processor, task, args)

        append_jsonl(raw_log, [{
            "plan_id": task.plan_id,
            "query_id": task.query_id,
            "option_label": task.option_label,
            "option_text": task.option_text,
            "retrieval_query": q,
            "query_garment": task.query_garment,
            "vlm_garment_vocab": list(task.vlm_garment_vocab),
            "num_records": len(records),
            "num_valid": sum(1 for r in records if not r.get("skip_reason")),
        }])
        append_jsonl(scored_log, records)

        save_ranked_images(records, task, args)

        if records:
            top = records[0]
            print(
                f"  top1 orig={top.get('original_rank')} "
                f"comb={fnum(top.get('combined_score')):.4f} "
                f"pat={fnum(top.get('pattern_score')):.4f} "
                f"col={fnum(top.get('color_score')):.4f} "
                f"gar={fnum(top.get('garment_score')):.4f} "
                f"sam3={top.get('sam3_prompt')} "
                f"skip={top.get('skip_reason')}"
            )

    print()
    print("Done.")
    print(f"top-k images: {args.output_root / 'topk_images'}")
    print(f"patches:      {args.output_root / 'topk_patches'}")
    print(f"ranked:       {args.output_root / 'ranked_images'}")
    print(f"scored log:   {scored_log}")


if __name__ == "__main__":
    main()
