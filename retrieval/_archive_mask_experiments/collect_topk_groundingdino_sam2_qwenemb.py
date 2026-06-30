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


def product_query(text: str, suffix: str) -> str:
    text = str(text or "").strip()
    suffix = str(suffix or "").strip()
    return f"{text}, {suffix}" if suffix else text


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

    existing = [p for p in sorted(out_base.parent.glob(out_base.name + ".*")) if not p.name.endswith(".tmp")]
    if existing and not args.force:
        return existing[0], None

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
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return {"processor": processor, "model": model, "device": device}


def load_sam2(model_id: str, device: str):
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    predictor = SAM2ImagePredictor.from_pretrained(model_id)
    if hasattr(predictor, "model"):
        predictor.model.to(device)
        predictor.model.eval()
    return predictor


def grounding_caption(garment: str, color: str = "", pattern: str = "") -> tuple[str, list[str]]:
    g = norm(garment) or "clothing item"
    c = norm(color)
    p = norm(pattern)

    phrases = [
        f"entire {g}",
        f"full {g}",
        f"main {g}",
        f"{g} clothing item",
        f"{g} garment",
    ]
    if c:
        phrases += [f"{c} {g}", f"entire {c} {g}"]
    if p and p != "solid":
        phrases += [f"{p} {g}", f"{p} clothing item"]
    phrases += [
        "main clothing item",
        "entire clothing garment",
    ]

    seen = set()
    clean = []
    for x in phrases:
        x = re.sub(r"\s+", " ", x.strip().lower())
        if x and x not in seen:
            seen.add(x)
            clean.append(x)

    # GroundingDINO prompt convention: categories separated by periods.
    return " . ".join(clean) + " .", clean


def expand_box(box: np.ndarray, w: int, h: int, ratio: float) -> np.ndarray:
    x1, y1, x2, y2 = [float(x) for x in box]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    return np.array([
        max(0, x1 - bw * ratio),
        max(0, y1 - bh * ratio),
        min(w - 1, x2 + bw * ratio),
        min(h - 1, y2 + bh * ratio),
    ], dtype=np.float32)


def post_process_groundingdino(
    processor,
    outputs,
    input_ids,
    target_sizes,
    box_threshold: float,
    text_threshold: float,
):
    fn = processor.post_process_grounded_object_detection

    attempts = [
        lambda: fn(
            outputs,
            input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        ),
        lambda: fn(
            outputs,
            input_ids=input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        ),
        lambda: fn(
            outputs,
            input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        ),
        lambda: fn(
            outputs,
            input_ids=input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        ),
        lambda: fn(
            outputs,
            target_sizes=target_sizes,
            threshold=box_threshold,
            text_threshold=text_threshold,
        ),
    ]

    last_err = None
    for call in attempts:
        try:
            result = call()
            if isinstance(result, list):
                return result[0]
            return result
        except TypeError as e:
            last_err = e

    raise last_err

def detect_groundingdino_box(
    gdino: dict[str, Any],
    image: Image.Image,
    caption: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    processor = gdino["processor"]
    model = gdino["model"]
    device = gdino["device"]

    inputs = processor(images=image, text=caption, return_tensors="pt")
    inputs = inputs.to(device)

    ctx = torch.autocast("cuda", dtype=torch.float16) if args.fp16 and str(device).startswith("cuda") else nullcontext()

    with torch.no_grad(), ctx:
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=device)

    results = post_process_groundingdino(
        processor=processor,
        outputs=outputs,
        input_ids=inputs.input_ids,
        target_sizes=target_sizes,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    boxes = results.get("boxes")
    scores = results.get("scores")
    labels = results.get("labels", [])

    if boxes is None or len(boxes) == 0:
        return None, {
            "caption": caption,
            "num_boxes": 0,
            "error": "no_groundingdino_box",
        }

    boxes_np = boxes.detach().cpu().numpy() if torch.is_tensor(boxes) else np.asarray(boxes)
    scores_np = scores.detach().cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)

    w, h = image.size
    candidates = []
    for i, box in enumerate(boxes_np):
        x1, y1, x2, y2 = [float(x) for x in box]
        area_ratio = max(0.0, (x2 - x1) * (y2 - y1) / max(1.0, w * h))
        if area_ratio < args.min_box_area_ratio:
            continue
        if area_ratio > args.max_box_area_ratio:
            continue

        score = float(scores_np[i]) if i < len(scores_np) else 0.0
        final_score = score * (area_ratio ** args.area_score_power)

        candidates.append({
            "idx": i,
            "box": expand_box(box, w=w, h=h, ratio=args.box_expand),
            "score": score,
            "area_ratio": area_ratio,
            "final_score": final_score,
            "label": str(labels[i]) if i < len(labels) else "",
        })

    if not candidates:
        return None, {
            "caption": caption,
            "num_boxes": int(len(boxes_np)),
            "error": "no_box_after_area_filter",
        }

    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    best = candidates[0]

    return best["box"], {
        "caption": caption,
        "num_boxes": int(len(boxes_np)),
        "selected_index": int(best["idx"]),
        "selected_score": float(best["score"]),
        "selected_area_ratio": float(best["area_ratio"]),
        "selected_final_score": float(best["final_score"]),
        "selected_label": best["label"],
        "box_xyxy": [float(x) for x in best["box"]],
        "top_candidates": [
            {
                "score": float(c["score"]),
                "area_ratio": float(c["area_ratio"]),
                "final_score": float(c["final_score"]),
                "label": c["label"],
                "box_xyxy": [float(x) for x in c["box"]],
            }
            for c in candidates[:5]
        ],
    }


def rectangle_mask(size: tuple[int, int], box: np.ndarray) -> np.ndarray:
    w, h = size
    mask = np.zeros((h, w), dtype=bool)
    x1, y1, x2, y2 = [int(round(x)) for x in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True
    return mask


def sam2_mask(predictor, image: Image.Image, box: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray | None, dict[str, Any]]:
    arr = np.asarray(image.convert("RGB"))
    try:
        with torch.inference_mode():
            predictor.set_image(arr)
            masks, scores, _ = predictor.predict(
                box=box.astype(np.float32),
                multimask_output=True,
            )

        if torch.is_tensor(masks):
            masks_np = masks.detach().cpu().numpy()
        else:
            masks_np = np.asarray(masks)

        if torch.is_tensor(scores):
            scores_np = scores.detach().cpu().numpy()
        else:
            scores_np = np.asarray(scores) if scores is not None else np.asarray([])

        if masks_np.ndim == 4:
            masks_np = masks_np[:, 0]
        if masks_np.ndim == 2:
            masks_np = masks_np[None, :, :]

        if len(masks_np) == 0:
            return None, {"error": "no_sam2_mask"}

        idx = int(np.argmax(scores_np)) if scores_np.size else 0
        mask = masks_np[idx].astype(bool)
        score = float(scores_np[idx]) if scores_np.size else None

        area_ratio = float(mask.sum()) / float(mask.size)
        if area_ratio < args.min_mask_area_ratio:
            return None, {
                "sam2_score": score,
                "mask_area_ratio": area_ratio,
                "error": "mask_too_small",
            }
        if area_ratio > args.max_mask_area_ratio:
            return None, {
                "sam2_score": score,
                "mask_area_ratio": area_ratio,
                "error": "mask_too_large",
            }

        return mask, {
            "sam2_score": score,
            "mask_area_ratio": area_ratio,
        }
    except Exception as e:
        return None, {"error": str(e)}


def ceil_to_multiple(x: int, m: int) -> int:
    return int(math.ceil(x / m) * m)


def patch_grid(w: int, h: int, short_side_tiles: int) -> list[tuple[int, int, int]]:
    side = max(1, int(math.ceil(min(w, h) / max(1, short_side_tiles))))
    pw = ceil_to_multiple(w, side)
    ph = ceil_to_multiple(h, side)
    return [(x, y, side) for y in range(0, ph, side) for x in range(0, pw, side)]


def make_masked_patch_files(
    image: Image.Image,
    mask: np.ndarray,
    args: argparse.Namespace,
    out_dir: Path,
) -> list[dict[str, Any]]:
    image = image.convert("RGB")
    w, h = image.size
    patches = patch_grid(w, h, args.short_side_tiles)

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for i, (x, y, s) in enumerate(patches):
        x2, y2 = min(x + s, w), min(y + s, h)
        if x2 <= x or y2 <= y:
            continue

        valid = mask[y:y2, x:x2]
        coverage = float(valid.sum()) / float(s * s)
        if coverage < args.min_coverage:
            continue

        tile = Image.new("RGB", (s, s), "white")
        tile.paste(image.crop((x, y, x2, y2)), (0, 0))

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


def pattern_texts(pattern: str) -> tuple[str, str]:
    p = norm(pattern)
    if not p:
        return "", ""
    if p == "solid":
        return "plain solid clothing fabric", "striped checkered plaid floral polka dot patterned clothing fabric"
    return f"{p} clothing fabric pattern", "plain solid clothing fabric or background skin hair"


def score_pattern(
    session: requests.Session,
    image_path: Path,
    pattern: str,
    mask: np.ndarray | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    p = norm(pattern)
    if not p:
        return {"score": 1.0, "mode": "missing_pattern_target", "tile_scores": []}
    if mask is None:
        return {"score": 0.0, "mode": "missing_query_garment_mask", "tile_scores": []}

    image = Image.open(image_path).convert("RGB")
    with tempfile.TemporaryDirectory(prefix="qwenemb_gdino_sam2_patch_") as tmp:
        tiles = make_masked_patch_files(image, mask, args, Path(tmp))
        if not tiles:
            return {"score": 0.0, "mode": "no_patch_inside_query_mask", "tile_scores": []}

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
        "mode": "groundingdino_sam2_query_mask_patch_coverage",
        "hits": hits,
        "num_tiles": len(tile_scores),
        "pattern_margin": args.pattern_margin,
        "avg_margin": sum(float(t["margin"]) for t in tile_scores) / denom,
        "tile_scores": tile_scores,
    }


def export_ranked_images(records: list[dict[str, Any]], out_dir: Path, show_k: int, copy: bool) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in records:
        groups.setdefault((str(r.get("query_id")), str(r.get("option_label"))), []).append(r)

    total = 0
    for (qid, label), rows in sorted(groups.items()):
        rows = sorted(
            rows,
            key=lambda r: (fnum(r.get("combined_score")), fnum(r.get("similarity"))),
            reverse=True,
        )[:show_k]

        if not rows:
            continue

        head = rows[0]
        odir = out_dir / slug(qid, 80) / f"{label}_{slug(head.get('option_text'), 80)}"
        odir.mkdir(parents=True, exist_ok=True)

        tsv = [
            f"query_id\t{qid}",
            f"option\t{label}",
            f"option_text\t{head.get('option_text')}",
            f"target\tcolor={head.get('target_color')} pattern={head.get('target_pattern')} garment={head.get('target_garment')}",
            "",
            "rank\torig\tcombined\tpattern\tcolor\tgarment\tsimilarity\tmask_source\tfile",
        ]

        for rank, r in enumerate(rows, start=1):
            src_raw = r.get("local_path")
            if not src_raw:
                continue
            src = Path(src_raw)
            if not src.exists():
                continue

            ext = src.suffix or ".jpg"
            orig = int(r.get("original_rank") or 0)
            comb = fnum(r.get("combined_score"))
            pat = fnum(r.get("pattern_score"))
            col = fnum(r.get("color_score"))
            gar = fnum(r.get("garment_score"))
            sim = fnum(r.get("similarity"))
            mask_source = str(r.get("mask_source") or "unknown")

            name = (
                f"{rank:03d}"
                f"__orig_{orig:03d}"
                f"__comb_{comb:.4f}"
                f"__pat_{pat:.4f}"
                f"__col_{col:.4f}"
                f"__gar_{gar:.4f}"
                f"__sim_{sim:.4f}"
                f"__mask_{slug(mask_source, 24)}"
                f"{ext}"
            )
            dst = odir / name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if copy:
                shutil.copy2(src, dst)
            else:
                rel = os.path.relpath(src.resolve(), start=dst.parent.resolve())
                os.symlink(rel, dst)

            tsv.append(
                f"{rank:03d}\t{orig:03d}\t{comb:.6f}\t{pat:.6f}\t"
                f"{col:.6f}\t{gar:.6f}\t{sim:.6f}\t{mask_source}\t{name}"
            )
            total += 1

        (odir / "_ranking.tsv").write_text("\n".join(tsv) + "\n", encoding="utf-8")
        print(f"[ranked] {odir}")

    print(f"exported ranked images: {total}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--plan-path", type=Path, default=Path("data/options/option_plans.jsonl"))
    p.add_argument("--client-url", default="http://127.0.0.1:1236/knn-service")
    p.add_argument("--image-score-url", default="")
    p.add_argument("--index-name", default="pod_qwenemb")
    p.add_argument("--output-root", type=Path, default=Path("data/retrieval/groundingdino_sam2_qwenemb"))

    p.add_argument("--product-query-suffix", default="studio product shot, isolated clothing item, plain white background, product catalog photo")
    p.add_argument("--retrieval-k", type=int, default=100)
    p.add_argument("--score-top-n", type=int, default=100)
    p.add_argument("--show-k", type=int, default=100)
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--options", default="A,B,C,D")

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fp16", action="store_true")

    p.add_argument("--grounding-model", default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--sam2-model", default="facebook/sam2-hiera-tiny")
    p.add_argument("--box-threshold", type=float, default=0.20)
    p.add_argument("--text-threshold", type=float, default=0.20)
    p.add_argument("--box-expand", type=float, default=0.03)
    p.add_argument("--min-box-area-ratio", type=float, default=0.003)
    p.add_argument("--max-box-area-ratio", type=float, default=0.98)
    p.add_argument("--area-score-power", type=float, default=0.20)

    p.add_argument("--min-mask-area-ratio", type=float, default=0.002)
    p.add_argument("--max-mask-area-ratio", type=float, default=0.98)
    p.add_argument("--box-mask-fallback", action="store_true", default=True)

    p.add_argument("--short-side-tiles", type=int, default=8)
    p.add_argument("--min-coverage", type=float, default=0.05)
    p.add_argument("--pattern-margin", type=float, default=0.0)
    p.add_argument("--score-batch", type=int, default=64)
    p.add_argument("--patch-score-batch", type=int, default=256)

    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--download-timeout", type=float, default=30.0)
    p.add_argument("--score-timeout", type=float, default=120.0)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--max-image-bytes", type=int, default=20_000_000)

    p.add_argument("--copy-ranked", action="store_true")
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
    print("GroundingDINO-Tiny + SAM2-Tiny + QwenEmb retrieval/scoring/ranking")
    print("=" * 96)
    print(f"plans:          {len(plans)}")
    print(f"option tasks:   {len(tasks)}")
    print(f"retrieval_k:    {args.retrieval_k}")
    print(f"score_top_n:    {args.score_top_n}")
    print(f"output:         {args.output_root}")
    print(f"QwenEmb:        {args.client_url}")
    print(f"image score:    {args.image_score_url}")
    print(f"grounding:      {args.grounding_model}")
    print(f"sam2:           {args.sam2_model}")

    print("loading GroundingDINO...")
    gdino = load_groundingdino(args.grounding_model, args.device)

    print("loading SAM2...")
    sam2 = load_sam2(args.sam2_model, args.device)

    session = requests.Session()

    raw_log = args.output_root / "raw_topk_results.jsonl"
    scored_log = args.output_root / "scored_results.jsonl"
    for p in [raw_log, scored_log]:
        if p.exists():
            p.unlink()

    all_records: list[dict[str, Any]] = []

    for task_i, task in enumerate(tasks, start=1):
        retrieval_text = product_query(task.option_text, args.product_query_suffix)
        print()
        print(f"[{task_i}/{len(tasks)}] {task.query_id} {task.option_label} | {retrieval_text}")

        results = query_knn(session, args, retrieval_text)
        if args.score_top_n > 0:
            results = results[: args.score_top_n]

        odir = (
            args.output_root
            / "images"
            / slug(task.query_id)
            / f"{task.option_label}_{slug(task.option_semantic, 30)}__{slug(retrieval_text, 80)}"
        )

        records = []
        for rank, item in enumerate(results, start=1):
            url = str(item.get("image_url") or item.get("url") or "")
            key = str(item.get("key") or f"rank_{rank:03d}")
            out_base = odir / f"rank_{rank:03d}__{slug(key, 80)}"
            local, err = download_image(session, url, out_base, args)

            rec = {
                "plan_idx": task.plan_idx,
                "query_id": task.query_id,
                "user_id": task.user_id,
                "option_label": task.option_label,
                "option_semantic": task.option_semantic,
                "option_text": retrieval_text,
                "target_color": task.target_color,
                "target_pattern": task.target_pattern,
                "target_garment": task.target_garment,
                "original_rank": rank,
                "similarity": item.get("similarity"),
                "url": url,
                "key": key,
                "caption": item.get("caption") or item.get("title") or "",
                "local_path": str(local) if local else None,
                "download_error": err,
                "raw": item,
            }
            records.append(rec)

        append_jsonl(raw_log, records)

        local_records = [r for r in records if r.get("local_path") and Path(str(r["local_path"])).exists()]
        image_paths = [str(r["local_path"]) for r in local_records]

        axis_texts = {}
        if task.target_color:
            axis_texts["color_score"] = f"{task.target_color} clothing item"
        if task.target_garment:
            axis_texts["garment_score"] = f"{task.target_garment} clothing item"

        axis_rows = score_image_files(
            session=session,
            args=args,
            texts=axis_texts,
            paths=image_paths,
            batch_size=args.score_batch,
        ) if image_paths and axis_texts else [{} for _ in image_paths]

        for r, row in zip(local_records, axis_rows):
            r["color_score"] = 1.0 if not task.target_color else max(0.0, float(row.get("color_score", 0.0)))
            r["garment_score"] = 1.0 if not task.target_garment else max(0.0, float(row.get("garment_score", 0.0)))

        caption, phrases = grounding_caption(task.target_garment, task.target_color, task.target_pattern)

        for j, rec in enumerate(records, start=1):
            local = rec.get("local_path")
            if not local or not Path(local).exists():
                rec.update({
                    "mask_source": "missing_image",
                    "pattern_score": 0.0,
                    "color_score": 0.0,
                    "garment_score": 0.0,
                    "combined_score": 0.0,
                })
                continue

            image_path = Path(local)
            image = Image.open(image_path).convert("RGB")

            box, gdino_info = detect_groundingdino_box(gdino, image, caption, args)
            mask = None
            sam2_info = {}

            if box is not None:
                mask, sam2_info = sam2_mask(sam2, image, box, args)
                if mask is not None:
                    mask_source = "groundingdino_sam2"
                elif args.box_mask_fallback:
                    mask = rectangle_mask(image.size, box)
                    mask_source = "groundingdino_box_fallback"
                else:
                    mask_source = "sam2_failed"
            else:
                mask_source = "no_groundingdino_box"

            pinfo = score_pattern(session, image_path, task.target_pattern, mask, args)
            pattern_score = float(pinfo.get("score") or 0.0)

            color_score = float(rec.get("color_score") or 0.0)
            garment_score = float(rec.get("garment_score") or 0.0)
            combined = math.sqrt(max(0.0, pattern_score * color_score * garment_score))

            rec.update({
                "mask_source": mask_source,
                "mask_query_caption": caption,
                "mask_query_phrases": phrases,
                "mask_groundingdino_info": gdino_info,
                "mask_sam2_info": sam2_info,
                "pattern_score": pattern_score,
                "pattern_score_raw": pinfo,
                "combined_score": combined,
                "rerank_formula": "sqrt(max(0, pattern_score * color_score * garment_score))",
            })

            if j % 10 == 0 or j == len(records):
                print(f"  scored {j}/{len(records)}")

        records = sorted(
            records,
            key=lambda r: (float(r.get("combined_score") or 0.0), float(r.get("similarity") or 0.0)),
            reverse=True,
        )
        for i, r in enumerate(records, start=1):
            r["rerank"] = i

        append_jsonl(scored_log, records)
        all_records.extend(records)

    export_ranked_images(
        all_records,
        args.output_root / "ranked_images",
        show_k=args.show_k,
        copy=args.copy_ranked,
    )

    print()
    print("Done.")
    print(f"raw log:     {raw_log}")
    print(f"scored log:  {scored_log}")
    print(f"ranked dir:  {args.output_root / 'ranked_images'}")


if __name__ == "__main__":
    main()
