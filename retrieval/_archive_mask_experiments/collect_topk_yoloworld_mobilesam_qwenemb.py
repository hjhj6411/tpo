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
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def slug(text: Any, max_len: int = 100) -> str:
    s = str(text or "")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")
    return (s or "unknown")[:max_len]


def norm(value: Any) -> str:
    return str(value or "").replace("_", " ").strip().lower()


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


def load_yolo_world(model_name: str):
    try:
        from ultralytics import YOLOWorld
        return YOLOWorld(model_name)
    except Exception:
        from ultralytics import YOLO
        return YOLO(model_name)


def load_mobilesam(checkpoint: Path, device: str):
    from mobile_sam import SamPredictor, sam_model_registry
    sam = sam_model_registry["vit_t"](checkpoint=str(checkpoint))
    sam.to(device=device)
    sam.eval()
    return SamPredictor(sam)


def garment_classes(garment: str) -> list[str]:
    g = norm(garment)
    table = {
        "tank top": ["tank top", "sleeveless top", "camisole top"],
        "sleeveless top": ["sleeveless top", "tank top", "camisole top"],
        "camisole": ["camisole", "camisole top", "tank top"],
        "t shirt": ["t shirt", "shirt", "top"],
        "shirt": ["shirt", "top"],
        "blouse": ["blouse", "shirt", "top"],
        "sweater": ["sweater", "knitwear", "pullover"],
        "hoodie": ["hoodie", "sweatshirt"],
        "jacket": ["jacket", "outerwear"],
        "coat": ["coat", "jacket", "outerwear"],
        "cardigan": ["cardigan", "sweater"],
        "skirt": ["skirt"],
        "dress": ["dress"],
        "pants": ["pants", "trousers"],
        "jeans": ["jeans", "denim pants"],
        "shorts": ["shorts"],
    }
    if g in table:
        return table[g]
    return [g] if g else ["clothing item"]


def expand_box(box: np.ndarray, w: int, h: int, ratio: float) -> np.ndarray:
    x1, y1, x2, y2 = [float(x) for x in box]
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * ratio
    y1 -= bh * ratio
    x2 += bw * ratio
    y2 += bh * ratio
    return np.array([
        max(0, x1),
        max(0, y1),
        min(w - 1, x2),
        min(h - 1, y2),
    ], dtype=np.float32)


def detect_best_box(yolo, image_path: Path, classes: list[str], args: argparse.Namespace) -> tuple[np.ndarray | None, dict[str, Any]]:
    try:
        if hasattr(yolo, "set_classes"):
            yolo.set_classes(classes)

        device_arg = 0 if str(args.device).startswith("cuda") else "cpu"
        results = yolo.predict(
            source=str(image_path),
            conf=args.yolo_conf,
            iou=args.yolo_iou,
            imgsz=args.yolo_imgsz,
            device=device_arg,
            half=bool(args.fp16 and str(args.device).startswith("cuda")),
            verbose=False,
        )
        if not results:
            return None, {"classes": classes, "error": "no_yolo_result"}

        res = results[0]
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None, {"classes": classes, "num_boxes": 0}

        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy()

        img = Image.open(image_path)
        w, h = img.size

        candidates = []
        for i, b in enumerate(xyxy):
            x1, y1, x2, y2 = b
            area_ratio = max(0.0, (x2 - x1) * (y2 - y1) / max(1.0, w * h))
            if area_ratio < args.min_box_area_ratio:
                continue
            if area_ratio > args.max_box_area_ratio:
                continue
            score = float(conf[i]) * (area_ratio ** 0.25)
            candidates.append((score, float(conf[i]), area_ratio, b, i))

        if not candidates:
            return None, {
                "classes": classes,
                "num_boxes": int(len(xyxy)),
                "error": "no_box_after_area_filter",
            }

        candidates.sort(key=lambda x: x[0], reverse=True)
        score, c, area_ratio, box, idx = candidates[0]
        box = expand_box(box, w=w, h=h, ratio=args.box_expand)

        return box, {
            "classes": classes,
            "num_boxes": int(len(xyxy)),
            "selected_index": int(idx),
            "selected_conf": c,
            "selected_area_ratio": area_ratio,
            "selected_score": score,
            "box_xyxy": [float(x) for x in box],
        }

    except Exception as e:
        return None, {"classes": classes, "error": str(e)}


def rectangle_mask(size: tuple[int, int], box: np.ndarray) -> np.ndarray:
    w, h = size
    mask = np.zeros((h, w), dtype=bool)
    x1, y1, x2, y2 = [int(round(x)) for x in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True
    return mask


def mobilesam_mask(predictor, image: Image.Image, box: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
    try:
        arr = np.asarray(image.convert("RGB"))
        predictor.set_image(arr)
        masks, scores, _ = predictor.predict(
            box=box.astype(np.float32),
            multimask_output=False,
        )
        if masks is None or len(masks) == 0:
            return None, {"error": "no_sam_mask"}
        mask = masks[0].astype(bool)
        score = float(scores[0]) if scores is not None and len(scores) else None
        return mask, {"sam_score": score}
    except Exception as e:
        return None, {"error": str(e)}


def ceil_to_multiple(x: int, m: int) -> int:
    return int(math.ceil(x / m) * m)


def patch_grid(w: int, h: int, short_side_tiles: int) -> tuple[list[tuple[int, int, int]], int, int]:
    side = max(1, int(math.ceil(min(w, h) / max(1, short_side_tiles))))
    pw = ceil_to_multiple(w, side)
    ph = ceil_to_multiple(h, side)
    return [(x, y, side) for y in range(0, ph, side) for x in range(0, pw, side)], pw, ph


def make_masked_patch_files(
    image: Image.Image,
    mask: np.ndarray,
    args: argparse.Namespace,
    out_dir: Path,
) -> list[dict[str, Any]]:
    image = image.convert("RGB")
    w, h = image.size
    patches, _, _ = patch_grid(w, h, args.short_side_tiles)

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
        return "plain solid clothing fabric", "striped checkered floral polka dot patterned clothing fabric"
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
    with tempfile.TemporaryDirectory(prefix="qwenemb_yw_patch_") as tmp:
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
        "mode": "yoloworld_mobilesam_query_mask_patch_coverage",
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
            key=lambda r: (float(r.get("combined_score") or 0.0), float(r.get("similarity") or 0.0)),
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
            comb = float(r.get("combined_score") or 0.0)
            pat = float(r.get("pattern_score") or 0.0)
            col = float(r.get("color_score") or 0.0)
            gar = float(r.get("garment_score") or 0.0)
            sim = float(r.get("similarity") or 0.0)
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
    p.add_argument("--output-root", type=Path, default=Path("data/retrieval/yoloworld_mobilesam_qwenemb"))

    p.add_argument("--product-query-suffix", default="studio product shot, isolated clothing item, plain white background, product catalog photo")
    p.add_argument("--retrieval-k", type=int, default=100)
    p.add_argument("--score-top-n", type=int, default=100)
    p.add_argument("--show-k", type=int, default=100)
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--options", default="A,B,C,D")

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fp16", action="store_true")

    p.add_argument("--yolo-model", default="yolov8s-world.pt")
    p.add_argument("--yolo-conf", type=float, default=0.03)
    p.add_argument("--yolo-iou", type=float, default=0.50)
    p.add_argument("--yolo-imgsz", type=int, default=640)
    p.add_argument("--box-expand", type=float, default=0.03)
    p.add_argument("--min-box-area-ratio", type=float, default=0.005)
    p.add_argument("--max-box-area-ratio", type=float, default=0.95)

    p.add_argument("--mobilesam-checkpoint", type=Path, default=Path("weights/mobile_sam.pt"))
    p.add_argument("--sam-fallback-box", action="store_true", default=True)

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

    if not args.mobilesam_checkpoint.exists():
        raise SystemExit(f"missing MobileSAM checkpoint: {args.mobilesam_checkpoint}")

    plans = load_jsonl(args.plan_path)
    if args.offset:
        plans = plans[args.offset:]
    if args.limit > 0:
        plans = plans[:args.limit]

    tasks = make_tasks(plans, options)

    print("=" * 88)
    print("YOLO-World-S + MobileSAM + QwenEmb retrieval/scoring/ranking")
    print("=" * 88)
    print(f"plans:        {len(plans)}")
    print(f"option tasks: {len(tasks)}")
    print(f"retrieval_k:  {args.retrieval_k}")
    print(f"score_top_n:  {args.score_top_n}")
    print(f"output:       {args.output_root}")
    print(f"QwenEmb:      {args.client_url}")
    print(f"image score:  {args.image_score_url}")

    print("loading YOLO-World...")
    yolo = load_yolo_world(args.yolo_model)

    print("loading MobileSAM...")
    predictor = load_mobilesam(args.mobilesam_checkpoint, args.device)

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

        axis_rows = score_image_files(
            session=session,
            args=args,
            texts={
                "color_score": f"{task.target_color} clothing item" if task.target_color else "",
                "garment_score": f"{task.target_garment} clothing item" if task.target_garment else "",
            },
            paths=image_paths,
            batch_size=args.score_batch,
        ) if image_paths else []

        for r, row in zip(local_records, axis_rows):
            r["color_score"] = max(0.0, float(row.get("color_score", 1.0 if not task.target_color else 0.0)))
            r["garment_score"] = max(0.0, float(row.get("garment_score", 1.0 if not task.target_garment else 0.0)))

        classes = garment_classes(task.target_garment)
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

            box, yolo_info = detect_best_box(yolo, image_path, classes, args)
            mask = None
            sam_info = {}

            if box is not None:
                mask, sam_info = mobilesam_mask(predictor, image, box)
                if mask is not None:
                    mask_source = "yoloworld_mobilesam"
                else:
                    mask = rectangle_mask(image.size, box)
                    mask_source = "yoloworld_box_fallback"
            else:
                mask_source = "no_yoloworld_box"

            pinfo = score_pattern(session, image_path, task.target_pattern, mask, args)
            pattern_score = float(pinfo.get("score") or 0.0)

            color_score = float(rec.get("color_score") or 0.0)
            garment_score = float(rec.get("garment_score") or 0.0)
            combined = math.sqrt(max(0.0, pattern_score * color_score * garment_score))

            rec.update({
                "mask_source": mask_source,
                "mask_query_classes": classes,
                "mask_yolo_info": yolo_info,
                "mask_sam_info": sam_info,
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
