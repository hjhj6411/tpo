#!/usr/bin/env python3
"""Collect top-k retrieval candidates, score attributes, rerank, and build a gallery.

Diagnostic collector for manually inspecting complex text-to-image retrieval such
as "black striped coat". It keeps the raw retrieval top-k list, adds three
attribute scores for every downloaded candidate, reranks by

    combined_score = sqrt(pattern_score * color_score * garment_score)

and writes both JSONL logs and an HTML gallery.

Scoring design:
  - color_score: global Qwen-VL score for whether the image matches the target color.
  - garment_score: global Qwen-VL score for whether the main item matches the target garment.
  - pattern_score: patch-wise Qwen-VL scoring. The image is split into a grid,
    likely non-background tiles are scored for the target pattern, and coverage is
    (# tiles with score >= threshold) / (# scored tiles). For pattern == solid, a
    global plain/solid score is used instead.

The script does not VLM-filter candidates out. It only logs scores and reranks them.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import html
import json
import math
import mimetypes
import os
import re
import shutil
import time
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

import requests
from PIL import Image, ImageStat


OPTION_LABELS = ["A", "B", "C", "D"]
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def slugify(text: Any, max_len: int = 96) -> str:
    text = str(text or "")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return (text or "unknown")[:max_len]


def norm_attr(value: Any) -> str:
    return str(value or "").replace("_", " ").strip().lower()


def option_to_text(opt: dict[str, Any]) -> str:
    text = opt.get("search_query")
    if text:
        return str(text)
    attrs = opt.get("attributes") or {}
    parts: list[str] = []
    if attrs.get("pattern") and attrs["pattern"] != "solid":
        parts.append(norm_attr(attrs["pattern"]))
    if attrs.get("color"):
        parts.append(norm_attr(attrs["color"]))
    if attrs.get("garment_category"):
        parts.append(norm_attr(attrs["garment_category"]))
    return " ".join(parts).strip() or "unknown item"


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


def make_tasks(plans: list[dict[str, Any]], options: list[str]) -> list[OptionTask]:
    tasks: list[OptionTask] = []
    for plan_idx, plan in enumerate(plans):
        query_id = str(plan.get("query_id") or f"plan_{plan_idx:05d}")
        # plan_id is the collection key: a dress-code query with two violation
        # axes emits two plans whose options differ, so one folder per query_id
        # would overwrite half the images.
        plan_id = str(plan.get("plan_id") or query_id)
        user_id = str(plan.get("user_id") or "unknown_user")
        opts = plan.get("options") or {}
        for label in options:
            opt = opts.get(label)
            if not isinstance(opt, dict):
                continue
            attrs = opt.get("attributes") or {}
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
                option_text=option_to_text(opt),
                option_attrs=attrs,
                target_color=norm_attr(attrs.get("color")),
                target_pattern=norm_attr(attrs.get("pattern")),
                target_garment=norm_attr(attrs.get("garment_category")),
            ))
    return tasks


def choose_extension(url: str, content_type: str | None) -> str:
    if content_type:
        content_type = content_type.split(";", 1)[0].strip().lower()
        ext = mimetypes.guess_extension(content_type)
        if ext in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}:
            return ".jpg" if ext == ".jpeg" else ext
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}:
        return ".jpg" if suffix == ".jpeg" else suffix
    return ".jpg"


def query_knn(session: requests.Session, client_url: str, text: str, retrieval_k: int,
              index_name: str, timeout: float, retries: int) -> list[dict[str, Any]]:
    payload = {
        "text": text,
        "modality": "image",
        "num_images": retrieval_k,
        "indice_name": index_name,
    }
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = session.post(client_url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and "error" in data:
                raise RuntimeError(str(data["error"]))
            if not isinstance(data, list):
                raise RuntimeError(f"Expected list response, got {type(data).__name__}: {data}")
            return [x for x in data if isinstance(x, dict)]
        except Exception as e:
            last_err = e
            if attempt >= retries:
                break
            time.sleep(min(2.0 ** attempt, 8.0))
    raise RuntimeError(f"KNN request failed for text={text!r}: {last_err}")


def download_image(session: requests.Session, url: str, out_base: Path, timeout: float,
                   retries: int, max_bytes: int, force: bool) -> tuple[Path | None, str | None]:
    if not url:
        return None, "missing_url"
    existing = [p for p in sorted(out_base.parent.glob(out_base.name + ".*")) if not p.name.endswith(".tmp")]
    if existing and not force:
        return existing[0], None

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with session.get(url, headers=DEFAULT_HEADERS, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                ext = choose_extension(url, resp.headers.get("Content-Type"))
                out_path = out_base.with_suffix(ext)
                tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                total = 0
                with tmp_path.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 128):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if max_bytes > 0 and total > max_bytes:
                            raise RuntimeError(f"image exceeds max_bytes={max_bytes}: {url}")
                        f.write(chunk)
                tmp_path.replace(out_path)
                return out_path, None
        except Exception as e:
            last_err = e
            if attempt >= retries:
                break
            time.sleep(min(2.0 ** attempt, 8.0))
    return None, str(last_err)


def collect_one(task: OptionTask, args: argparse.Namespace) -> list[dict[str, Any]]:
    session = requests.Session()
    results = query_knn(
        session=session,
        client_url=args.client_url,
        text=task.option_text,
        retrieval_k=args.retrieval_k,
        index_name=args.index_name,
        timeout=args.timeout,
        retries=args.retries,
    )
    if args.score_top_n > 0:
        results = results[:args.score_top_n]

    qdir = args.output_root / "images" / slugify(task.plan_id)
    odir_name = f"{task.option_label}_{slugify(task.option_semantic or 'option', 40)}__{slugify(task.option_text, 64)}"
    odir = qdir / odir_name
    records: list[dict[str, Any]] = []
    for rank, item in enumerate(results, start=1):
        url = str(item.get("image_url") or item.get("url") or "")
        key = str(item.get("key") or "")
        out_base = odir / f"rank_{rank:03d}__{slugify(key or 'image', 80)}"
        local_path: Path | None = None
        download_error: str | None = None
        if not args.no_download:
            local_path, download_error = download_image(
                session=session,
                url=url,
                out_base=out_base,
                timeout=args.download_timeout,
                retries=args.retries,
                max_bytes=args.max_image_bytes,
                force=args.force,
            )
        records.append({
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
            "option_attrs": task.option_attrs,
            "target_color": task.target_color,
            "target_pattern": task.target_pattern,
            "target_garment": task.target_garment,
            "original_rank": rank,
            "similarity": item.get("similarity"),
            "url": url,
            "key": key,
            "caption": item.get("caption") or item.get("title") or "",
            "source": item.get("source"),
            "local_path": str(local_path) if local_path else None,
            "download_error": download_error,
            "raw": item,
        })
    return records


def encode_image_b64(path: Path) -> tuple[str, str]:
    data = path.read_bytes()
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return base64.b64encode(data).decode("utf-8"), mime


def extract_score(text: str) -> tuple[float | None, Any]:
    text = (text or "").strip()
    if not text:
        return None, None
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I).strip()
    text = re.sub(r"\s*```$", "", text).strip()
    obj = None
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None
    if isinstance(obj, dict):
        for key in ["score", "confidence", "probability", "match_score"]:
            if key in obj:
                try:
                    val = float(obj[key])
                    if val > 1.0 and val <= 100.0:
                        val /= 100.0
                    return max(0.0, min(1.0, val)), obj
                except Exception:
                    pass
    nums = re.findall(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?|[1-9]\d?(?:\.\d+)?%)(?!\d)", text)
    if nums:
        raw = nums[0]
        try:
            val = float(raw[:-1]) / 100.0 if raw.endswith("%") else float(raw)
            if val > 1.0 and val <= 100.0:
                val /= 100.0
            return max(0.0, min(1.0, val)), {"raw_text": text}
        except Exception:
            pass
    return None, {"raw_text": text}


def call_vlm_score(image_path: Path, prompt: str, vlm_url: str, model: str,
                   timeout: float, retries: int, max_tokens: int) -> tuple[float | None, Any, str | None]:
    b64, mime = encode_image_b64(image_path)
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    url = vlm_url.rstrip("/") + "/chat/completions"
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            score, parsed = extract_score(content)
            if score is None:
                return None, parsed, f"could_not_parse_score: {content[:200]}"
            return score, parsed, None
        except Exception as e:
            last_err = e
            if attempt >= retries:
                break
            time.sleep(min(2.0 ** attempt, 8.0))
    return None, None, str(last_err)


def color_prompt(color: str) -> str:
    return f"""You are scoring a fashion retrieval candidate.
Target color: {color}
Look at the main clothing item. Return ONLY valid JSON: {{"score": <number from 0 to 1>}}
Score 1.0 if the main visible garment clearly matches the target color. Score 0.5 if partially present or ambiguous. Score 0.0 if different or not visible. Do not explain."""


def garment_prompt(garment: str) -> str:
    return f"""You are scoring a fashion retrieval candidate.
Target garment category: {garment}
Look at the main clothing item. Return ONLY valid JSON: {{"score": <number from 0 to 1>}}
Score 1.0 if the main garment is the target category. Score 0.5 if close/ambiguous. Score 0.0 if different or not visible. Do not explain."""


def solid_prompt() -> str:
    return """You are scoring a fashion retrieval candidate.
Target pattern: solid/plain fabric with no visible decorative pattern.
Look at the main clothing item. Return ONLY valid JSON: {"score": <number from 0 to 1>}
Score 1.0 if the main garment is plain/solid. Score 0.5 if mostly plain with small logos or weak texture. Score 0.0 if it has visible stripes, checks, florals, camouflage, polka dots, argyle, or another pattern. Do not explain."""


def pattern_tile_prompt(pattern: str) -> str:
    return f"""You are scoring a cropped image tile from a fashion retrieval candidate.
Target pattern: {pattern}
Return ONLY valid JSON: {{"score": <number from 0 to 1>}}
Score 1.0 if this crop clearly shows {pattern} fabric/clothing pattern. Score 0.5 if ambiguous or partial. Score 0.0 if background, skin, hair, object, text, plain fabric, or a different pattern. Do not explain."""


def likely_non_background_tile(tile: Image.Image, white_thresh: float, min_std: float) -> bool:
    rgb = tile.convert("RGB")
    stat = ImageStat.Stat(rgb)
    mean = sum(stat.mean) / 3.0
    std = sum(stat.stddev) / 3.0
    if mean >= white_thresh and std < 35.0:
        return False
    if std < min_std:
        return False
    return True


def make_pattern_tiles(image_path: Path, grid: int, max_tiles: int, tile_dir: Path,
                       save_tiles: bool, white_thresh: float, min_std: float) -> list[Path]:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    candidates: list[tuple[float, int, Image.Image]] = []
    for gy in range(grid):
        for gx in range(grid):
            left = int(w * gx / grid)
            upper = int(h * gy / grid)
            right = int(w * (gx + 1) / grid)
            lower = int(h * (gy + 1) / grid)
            tile = img.crop((left, upper, right, lower))
            if not likely_non_background_tile(tile, white_thresh=white_thresh, min_std=min_std):
                continue
            stat = ImageStat.Stat(tile)
            candidates.append((float(sum(stat.stddev) / 3.0), gy * grid + gx, tile))
    candidates.sort(key=lambda x: (-x[0], x[1]))
    selected = candidates[:max_tiles]
    tile_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for _, idx, tile in selected:
        out = tile_dir / f"tile_{idx:02d}.jpg"
        if save_tiles or not out.exists():
            tile.save(out, quality=92)
        paths.append(out)
    return paths


def score_pattern(image_path: Path, pattern: str, rec_id: str, args: argparse.Namespace, vlm_url: str) -> dict[str, Any]:
    if not pattern:
        return {"score": 1.0, "mode": "missing_pattern_target", "tile_scores": []}
    if pattern == "solid":
        score, parsed, err = call_vlm_score(
            image_path=image_path,
            prompt=solid_prompt(),
            vlm_url=vlm_url,
            model=args.vlm_model,
            timeout=args.score_timeout,
            retries=args.retries,
            max_tokens=args.vlm_max_tokens,
        )
        return {"score": score if score is not None else 0.0, "mode": "solid_global", "parsed": parsed, "error": err, "tile_scores": []}

    tiles = make_pattern_tiles(
        image_path=image_path,
        grid=args.tile_grid,
        max_tiles=args.pattern_max_tiles,
        tile_dir=args.output_root / "tiles" / rec_id,
        save_tiles=args.save_tiles,
        white_thresh=args.tile_white_thresh,
        min_std=args.tile_min_std,
    ) or [image_path]
    tile_scores: list[dict[str, Any]] = []
    for tile_path in tiles:
        score, parsed, err = call_vlm_score(
            image_path=tile_path,
            prompt=pattern_tile_prompt(pattern),
            vlm_url=vlm_url,
            model=args.vlm_model,
            timeout=args.score_timeout,
            retries=args.retries,
            max_tokens=args.vlm_max_tokens,
        )
        s = score if score is not None else 0.0
        tile_scores.append({
            "tile": str(tile_path),
            "score": s,
            "present": bool(s >= args.pattern_threshold),
            "parsed": parsed,
            "error": err,
        })
    denom = max(len(tile_scores), 1)
    coverage = sum(1 for t in tile_scores if t["present"]) / denom
    avg_score = sum(float(t["score"]) for t in tile_scores) / denom
    return {
        "score": coverage,
        "avg_tile_score": avg_score,
        "mode": "patch_coverage",
        "num_tiles": len(tile_scores),
        "threshold": args.pattern_threshold,
        "tile_scores": tile_scores,
    }


def score_candidate(args_tuple: tuple[int, dict[str, Any], argparse.Namespace]) -> dict[str, Any]:
    idx, rec, args = args_tuple
    local_path = rec.get("local_path")
    if not local_path or not Path(local_path).exists():
        rec.update({
            "color_score": 0.0,
            "pattern_score": 0.0,
            "garment_score": 0.0,
            "combined_score": 0.0,
            "score_error": "missing_local_image",
        })
        return rec

    image_path = Path(local_path)
    vlm_urls = [u.strip() for u in args.vlm_urls.split(",") if u.strip()]
    vlm_url = vlm_urls[idx % len(vlm_urls)]
    color = rec.get("target_color") or ""
    garment = rec.get("target_garment") or ""
    pattern = rec.get("target_pattern") or ""
    rec_id = f"{slugify(rec.get('plan_id'))}__{rec.get('option_label')}__rank_{int(rec.get('original_rank') or 0):03d}"

    color_score, color_parsed, color_err = call_vlm_score(
        image_path=image_path,
        prompt=color_prompt(color),
        vlm_url=vlm_url,
        model=args.vlm_model,
        timeout=args.score_timeout,
        retries=args.retries,
        max_tokens=args.vlm_max_tokens,
    ) if color else (1.0, None, None)
    garment_score, garment_parsed, garment_err = call_vlm_score(
        image_path=image_path,
        prompt=garment_prompt(garment),
        vlm_url=vlm_url,
        model=args.vlm_model,
        timeout=args.score_timeout,
        retries=args.retries,
        max_tokens=args.vlm_max_tokens,
    ) if garment else (1.0, None, None)
    pattern_info = score_pattern(image_path=image_path, pattern=pattern, rec_id=rec_id, args=args, vlm_url=vlm_url)

    c = float(color_score if color_score is not None else 0.0)
    g = float(garment_score if garment_score is not None else 0.0)
    p = float(pattern_info.get("score") or 0.0)
    combined = math.sqrt(max(0.0, c * g * p))
    rec.update({
        "color_score": c,
        "pattern_score": p,
        "garment_score": g,
        "combined_score": combined,
        "rerank_formula": "sqrt(pattern_score * color_score * garment_score)",
        "color_score_raw": color_parsed,
        "garment_score_raw": garment_parsed,
        "pattern_score_raw": pattern_info,
        "color_score_error": color_err,
        "garment_score_error": garment_err,
        "score_error": "; ".join(e for e in [color_err, garment_err] if e) or None,
    })
    return rec


def relpath(path: str | None, base: Path) -> str | None:
    if not path:
        return None
    try:
        return os.path.relpath(path, start=base)
    except ValueError:
        return path


def grouped_rerank(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for rec in records:
        groups.setdefault((str(rec.get("plan_id")), str(rec.get("option_label"))), []).append(rec)
    out: list[dict[str, Any]] = []
    for _key, rows in groups.items():
        rows.sort(key=lambda r: (float(r.get("combined_score") or 0.0), float(r.get("similarity") or 0.0)), reverse=True)
        for i, r in enumerate(rows, start=1):
            r["rerank"] = i
        out.extend(rows)
    return out


def build_gallery(records: list[dict[str, Any]], output_root: Path, title: str, show_k: int) -> Path:
    by_plan: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for rec in records:
        by_plan.setdefault(str(rec["plan_id"]), {}).setdefault(str(rec["option_label"]), []).append(rec)

    css = """
    body { font-family: Arial, sans-serif; margin: 24px; background: #fafafa; color: #222; }
    h1 { margin-bottom: 4px; }
    .meta { color: #666; margin-bottom: 20px; }
    .query { border: 1px solid #ddd; background: white; border-radius: 10px; padding: 16px; margin: 18px 0; }
    .qmeta { color: #666; font-size: 13px; margin-bottom: 10px; }
    .option { margin: 14px 0 24px 0; }
    .option h3 { margin: 8px 0; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 12px; }
    .card { border: 1px solid #ddd; border-radius: 8px; background: #fff; padding: 8px; overflow-wrap: anywhere; }
    .card img { width: 100%; height: 190px; object-fit: contain; background: #f1f1f1; border-radius: 6px; }
    .rank { font-weight: bold; margin-top: 6px; }
    .score { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; line-height: 1.35; }
    .caption { color: #444; font-size: 12px; margin-top: 4px; }
    .err { color: #b00020; font-size: 12px; }
    a { color: #0b57d0; text-decoration: none; }
    a:hover { text-decoration: underline; }
    """
    html_parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        f"<style>{css}</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
        f"<div class='meta'>records={len(records)} · showing top {show_k} after rerank per option · formula=sqrt(pattern*color*garment)</div>",
    ]
    for plan_id in sorted(by_plan):
        option_map = by_plan[plan_id]
        first = next(iter(next(iter(option_map.values()))), {})
        html_parts.append("<section class='query'>")
        html_parts.append(f"<h2>{html.escape(plan_id)}</h2>")
        html_parts.append(
            "<div class='qmeta'>"
            f"user={html.escape(str(first.get('user_id', '')))} · "
            f"scenario={html.escape(str(first.get('scenario_id', '')))} · "
            f"axis={html.escape(str(first.get('active_axis', '')))} · "
            f"qtype={html.escape(str(first.get('query_type', '')))}"
            "</div>"
        )
        for label in OPTION_LABELS:
            rows = sorted(option_map.get(label, []), key=lambda r: int(r.get("rerank") or 999999))[:show_k]
            if not rows:
                continue
            head = rows[0]
            target = f"target: color={head.get('target_color')} / pattern={head.get('target_pattern')} / garment={head.get('target_garment')}"
            html_parts.append("<div class='option'>")
            html_parts.append(
                f"<h3>{html.escape(label)} · {html.escape(str(head.get('option_semantic')))} · "
                f"{html.escape(str(head.get('option_text')))}</h3>"
            )
            html_parts.append(f"<div class='qmeta'>{html.escape(target)}</div>")
            html_parts.append("<div class='grid'>")
            for rec in rows:
                local = relpath(rec.get("local_path"), output_root)
                url = rec.get("url") or ""
                cap = str(rec.get("caption") or "")[:160]
                sim = rec.get("similarity")
                sim_text = f"{float(sim):.4f}" if isinstance(sim, (int, float)) else str(sim)
                html_parts.append("<div class='card'>")
                if local:
                    html_parts.append(
                        f"<a href='{html.escape(local)}' target='_blank'>"
                        f"<img src='{html.escape(local)}' loading='lazy'></a>"
                    )
                else:
                    html_parts.append("<div class='err'>no local image</div>")
                html_parts.append(f"<div class='rank'>rerank {rec.get('rerank')} · original rank {rec.get('original_rank')}</div>")
                html_parts.append(
                    "<div class='score'>"
                    f"combined={float(rec.get('combined_score') or 0):.3f}<br>"
                    f"pattern={float(rec.get('pattern_score') or 0):.3f} · "
                    f"color={float(rec.get('color_score') or 0):.3f} · "
                    f"garment={float(rec.get('garment_score') or 0):.3f}<br>"
                    f"retrieval_sim={html.escape(sim_text)}"
                    "</div>"
                )
                if url:
                    html_parts.append(f"<div><a href='{html.escape(url)}' target='_blank'>source url</a></div>")
                if rec.get("download_error") or rec.get("score_error"):
                    html_parts.append(f"<div class='err'>{html.escape(str(rec.get('download_error') or rec.get('score_error')))}</div>")
                if cap:
                    html_parts.append(f"<div class='caption'>{html.escape(cap)}</div>")
                html_parts.append("</div>")
            html_parts.append("</div></div>")
        html_parts.append("</section>")
    html_parts.append("</body></html>")
    gallery_path = output_root / "scored_gallery.html"
    gallery_path.write_text("\n".join(html_parts), encoding="utf-8")
    return gallery_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--plan-path", type=Path, default=DEFAULT_PLAN_PATH,
                   help="option plans; default follows POD_VARIANT")
    p.add_argument("--client-url", default="http://127.0.0.1:1236/knn-service")
    p.add_argument("--index-name", default="pod_qwenemb")
    p.add_argument("--vlm-urls", default="http://127.0.0.1:8002/v1")
    p.add_argument("--vlm-model", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--output-root", type=Path, default=DATA_DIR / "retrieval/qwenemb_scored_gallery",
                   help="output root; default follows POD_VARIANT")
    p.add_argument("--retrieval-k", type=int, default=48, help="Raw KNN candidates per option")
    p.add_argument("--score-top-n", type=int, default=48, help="Score only first N retrieved candidates; 0 means all")
    p.add_argument("--show-k", type=int, default=12, help="Show this many reranked candidates per option in HTML")
    p.add_argument("--limit", type=int, default=0, help="Number of option plans to process; 0 = all")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--options", default="A,B,C,D")
    p.add_argument("--workers", type=int, default=8, help="Retrieval/download workers")
    p.add_argument("--score-workers", type=int, default=2, help="VLM scoring workers")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--download-timeout", type=float, default=30.0)
    p.add_argument("--score-timeout", type=float, default=120.0)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--max-image-bytes", type=int, default=20_000_000)
    p.add_argument("--vlm-max-tokens", type=int, default=64)
    p.add_argument("--tile-grid", type=int, default=5)
    p.add_argument("--pattern-max-tiles", type=int, default=12)
    p.add_argument("--pattern-threshold", type=float, default=0.5)
    p.add_argument("--tile-white-thresh", type=float, default=242.0)
    p.add_argument("--tile-min-std", type=float, default=8.0)
    p.add_argument("--save-tiles", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--no-score", action="store_true", help="Collect/download only; do not run attribute scoring")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    options = [x.strip().upper() for x in args.options.split(",") if x.strip()]
    bad_options = [x for x in options if x not in OPTION_LABELS]
    if bad_options:
        raise SystemExit(f"Unsupported option labels: {bad_options}; use subset of A,B,C,D")
    if args.no_score and args.no_download:
        raise SystemExit("--no-score with --no-download gives no local images or scores; remove one of them")

    plans = load_jsonl(args.plan_path)
    if args.offset:
        plans = plans[args.offset:]
    if args.limit > 0:
        plans = plans[:args.limit]
    tasks = make_tasks(plans, options)

    print("=" * 76)
    print("  Attribute-scored Top-k Retrieval Gallery")
    print("=" * 76)
    print(f"  plans:        {len(plans)}")
    print(f"  option tasks: {len(tasks)}")
    print(f"  retrieval_k:  {args.retrieval_k}")
    print(f"  score_top_n:  {args.score_top_n or 'all'}")
    print(f"  expected raw: {len(tasks) * (args.score_top_n or args.retrieval_k)} records")
    print(f"  endpoint:     {args.client_url}")
    print(f"  VLM URLs:     {args.vlm_urls}")
    print(f"  formula:      sqrt(pattern_score * color_score * garment_score)")
    print(f"  output:       {args.output_root}")

    if args.dry_run:
        for t in tasks[:20]:
            print(f"  {t.plan_id} {t.option_label}: {t.option_text} | c={t.target_color} p={t.target_pattern} g={t.target_garment}")
        if len(tasks) > 20:
            print(f"  ... {len(tasks) - 20} more tasks")
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
    done = 0
    if args.workers <= 1:
        for task in tasks:
            try:
                rows = collect_one(task, args)
                append_jsonl(raw_log, rows)
                raw_records.extend(rows)
            except Exception as e:
                errors.append({"plan_id": task.plan_id, "query_id": task.query_id, "option_label": task.option_label, "error": str(e)})
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"  retrieve [{done}/{len(tasks)}] records={len(raw_records)} errors={len(errors)}")
    else:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(collect_one, task, args): task for task in tasks}
            for fut in cf.as_completed(futs):
                task = futs[fut]
                try:
                    rows = fut.result()
                    append_jsonl(raw_log, rows)
                    raw_records.extend(rows)
                except Exception as e:
                    errors.append({"plan_id": task.plan_id, "query_id": task.query_id, "option_label": task.option_label, "error": str(e)})
                done += 1
                if done % 10 == 0 or done == len(tasks):
                    print(f"  retrieve [{done}/{len(tasks)}] records={len(raw_records)} errors={len(errors)}")
    write_json(args.output_root / "retrieval_errors.json", errors)

    if args.no_score:
        scored_records = raw_records
        for r in scored_records:
            r.update({"color_score": None, "pattern_score": None, "garment_score": None, "combined_score": None})
    else:
        scored_records: list[dict[str, Any]] = []
        score_inputs = [(i, r, args) for i, r in enumerate(raw_records)]
        done = 0
        if args.score_workers <= 1:
            for item in score_inputs:
                rec = score_candidate(item)
                append_jsonl(scored_log, [rec])
                scored_records.append(rec)
                done += 1
                if done % 20 == 0 or done == len(score_inputs):
                    print(f"  score    [{done}/{len(score_inputs)}]")
        else:
            with cf.ThreadPoolExecutor(max_workers=args.score_workers) as ex:
                futs = [ex.submit(score_candidate, item) for item in score_inputs]
                for fut in cf.as_completed(futs):
                    rec = fut.result()
                    append_jsonl(scored_log, [rec])
                    scored_records.append(rec)
                    done += 1
                    if done % 20 == 0 or done == len(score_inputs):
                        print(f"  score    [{done}/{len(score_inputs)}]")

    scored_records = grouped_rerank(scored_records)
    if scored_log.exists():
        scored_log.unlink()
    append_jsonl(scored_log, scored_records)
    gallery_path = build_gallery(scored_records, args.output_root, "Attribute-scored top-k retrieval gallery", args.show_k)

    print("\nDone.")
    print(f"  raw records:    {len(raw_records)}")
    print(f"  scored records: {len(scored_records)}")
    print(f"  retrieval errs: {len(errors)}")
    print(f"  raw log:        {raw_log}")
    print(f"  scored log:     {scored_log}")
    print(f"  gallery:        {gallery_path}")


if __name__ == "__main__":
    main()
