#!/usr/bin/env python3
"""Collect raw top-k retrieval results and build an inspection gallery.

This script is intentionally retrieval-only: it does not run DPV gates, VLM
verification, deduplication, or option selection. For each option text in an
option_plans.jsonl file, it queries a clip-retrieval-compatible KNN endpoint,
downloads every returned top-k image, writes a candidate-level JSONL log, and
creates a simple HTML gallery for manual inspection.

Typical QwenEmb usage:

  conda activate pod
  python retrieval/collect_topk_gallery.py \
    --plan-path data/options/option_plans.jsonl \
    --client-url http://127.0.0.1:1236/knn-service \
    --index-name pod_qwenemb \
    --output-root data/retrieval/qwenemb_topk_gallery \
    --top-k 12 --limit 30 --workers 8 --force

For a full run, remove --limit.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import html
import json
import mimetypes
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests


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
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def slugify(text: str, max_len: int = 96) -> str:
    text = str(text or "")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    if not text:
        return "unknown"
    return text[:max_len]


def option_to_text(opt: dict[str, Any]) -> str:
    text = opt.get("search_query")
    if text:
        return str(text)

    attrs = opt.get("attributes") or {}
    parts: list[str] = []
    if attrs.get("pattern") and attrs["pattern"] != "solid":
        parts.append(str(attrs["pattern"]).replace("_", " "))
    if attrs.get("color"):
        parts.append(str(attrs["color"]))
    if attrs.get("garment_category"):
        parts.append(str(attrs["garment_category"]).replace("_", " "))
    return " ".join(parts).strip() or "unknown item"


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


def query_knn(
    session: requests.Session,
    client_url: str,
    text: str,
    top_k: int,
    index_name: str,
    timeout: float,
    retries: int,
) -> list[dict[str, Any]]:
    payload = {
        "text": text,
        "modality": "image",
        "num_images": top_k,
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
        except Exception as e:  # requests and JSON errors both need retry handling
            last_err = e
            if attempt >= retries:
                break
            time.sleep(min(2.0 ** attempt, 8.0))
    raise RuntimeError(f"KNN request failed for text={text!r}: {last_err}")


def download_image(
    session: requests.Session,
    url: str,
    out_base: Path,
    timeout: float,
    retries: int,
    max_bytes: int,
    force: bool,
) -> tuple[Path | None, str | None]:
    if not url:
        return None, "missing_url"

    existing = sorted(out_base.parent.glob(out_base.name + ".*"))
    if existing and not force:
        return existing[0], None

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with session.get(url, headers=DEFAULT_HEADERS, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                content_type = resp.headers.get("Content-Type")
                ext = choose_extension(url, content_type)
                out_path = out_base.with_suffix(ext)
                tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

                total = 0
                out_path.parent.mkdir(parents=True, exist_ok=True)
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


@dataclass(frozen=True)
class OptionTask:
    plan_idx: int
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


def make_tasks(plans: list[dict[str, Any]], options: list[str]) -> list[OptionTask]:
    tasks: list[OptionTask] = []
    for plan_idx, plan in enumerate(plans):
        query_id = str(plan.get("query_id") or f"plan_{plan_idx:05d}")
        user_id = str(plan.get("user_id") or "unknown_user")
        opts = plan.get("options") or {}
        for label in options:
            opt = opts.get(label)
            if not isinstance(opt, dict):
                continue
            tasks.append(OptionTask(
                plan_idx=plan_idx,
                query_id=query_id,
                user_id=user_id,
                scenario_id=plan.get("scenario_id"),
                scenario_name=plan.get("scenario_name"),
                active_axis=plan.get("active_axis"),
                query_type=plan.get("query_type"),
                option_label=label,
                option_semantic=opt.get("label"),
                option_text=option_to_text(opt),
                option_attrs=opt.get("attributes") or {},
            ))
    return tasks


def collect_one(task: OptionTask, args: argparse.Namespace) -> list[dict[str, Any]]:
    session = requests.Session()
    results = query_knn(
        session=session,
        client_url=args.client_url,
        text=task.option_text,
        top_k=args.top_k,
        index_name=args.index_name,
        timeout=args.timeout,
        retries=args.retries,
    )

    qdir = args.output_root / "images" / slugify(task.query_id)
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
            "rank": rank,
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


def relpath(path: str | None, base: Path) -> str | None:
    if not path:
        return None
    try:
        return os.path.relpath(path, start=base)
    except ValueError:
        return path


def build_gallery(records: list[dict[str, Any]], output_root: Path, title: str, max_caption_chars: int = 180) -> Path:
    by_query: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for rec in records:
        by_query.setdefault(rec["query_id"], {}).setdefault(rec["option_label"], []).append(rec)

    css = """
    body { font-family: Arial, sans-serif; margin: 24px; background: #fafafa; color: #222; }
    h1 { margin-bottom: 4px; }
    .meta { color: #666; margin-bottom: 20px; }
    .query { border: 1px solid #ddd; background: white; border-radius: 10px; padding: 16px; margin: 18px 0; }
    .qmeta { color: #666; font-size: 13px; margin-bottom: 10px; }
    .option { margin: 14px 0 22px 0; }
    .option h3 { margin: 8px 0; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; }
    .card { border: 1px solid #ddd; border-radius: 8px; background: #fff; padding: 8px; overflow-wrap: anywhere; }
    .card img { width: 100%; height: 180px; object-fit: contain; background: #f1f1f1; border-radius: 6px; }
    .rank { font-weight: bold; margin-top: 6px; }
    .sim { color: #555; font-size: 12px; }
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
        f"<div class='meta'>records={len(records)} · output={html.escape(str(output_root))}</div>",
    ]

    for query_id in sorted(by_query):
        option_map = by_query[query_id]
        first = next(iter(next(iter(option_map.values()))), {})
        html_parts.append("<section class='query'>")
        html_parts.append(f"<h2>{html.escape(query_id)}</h2>")
        html_parts.append(
            "<div class='qmeta'>"
            f"user={html.escape(str(first.get('user_id', '')))} · "
            f"scenario={html.escape(str(first.get('scenario_id', '')))} · "
            f"axis={html.escape(str(first.get('active_axis', '')))} · "
            f"qtype={html.escape(str(first.get('query_type', '')))}"
            "</div>"
        )

        for label in OPTION_LABELS:
            rows = sorted(option_map.get(label, []), key=lambda r: int(r.get("rank") or 0))
            if not rows:
                continue
            head = rows[0]
            html_parts.append("<div class='option'>")
            html_parts.append(
                f"<h3>{html.escape(label)} · {html.escape(str(head.get('option_semantic')))} · "
                f"{html.escape(str(head.get('option_text')))}</h3>"
            )
            html_parts.append("<div class='grid'>")
            for rec in rows:
                local = relpath(rec.get("local_path"), output_root)
                url = rec.get("url") or ""
                cap = str(rec.get("caption") or "")[:max_caption_chars]
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
                html_parts.append(f"<div class='rank'>rank {rec.get('rank')}</div>")
                html_parts.append(f"<div class='sim'>similarity: {html.escape(sim_text)}</div>")
                if url:
                    html_parts.append(f"<div><a href='{html.escape(url)}' target='_blank'>source url</a></div>")
                if rec.get("download_error"):
                    html_parts.append(f"<div class='err'>{html.escape(str(rec['download_error']))}</div>")
                if cap:
                    html_parts.append(f"<div class='caption'>{html.escape(cap)}</div>")
                html_parts.append("</div>")
            html_parts.append("</div></div>")
        html_parts.append("</section>")

    html_parts.append("</body></html>")
    gallery_path = output_root / "gallery.html"
    gallery_path.write_text("\n".join(html_parts), encoding="utf-8")
    return gallery_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--plan-path", type=Path, default=Path("data/options/option_plans.jsonl"))
    p.add_argument("--client-url", default="http://127.0.0.1:1236/knn-service")
    p.add_argument("--index-name", default="pod_qwenemb")
    p.add_argument("--output-root", type=Path, default=Path("data/retrieval/qwenemb_topk_gallery"))
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument("--limit", type=int, default=0, help="Number of option plans to process; 0 = all")
    p.add_argument("--offset", type=int, default=0, help="Skip this many option plans before --limit")
    p.add_argument("--options", default="A,B,C,D", help="Comma-separated option labels to collect")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--timeout", type=float, default=120.0, help="KNN request timeout")
    p.add_argument("--download-timeout", type=float, default=30.0)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--max-image-bytes", type=int, default=20_000_000)
    p.add_argument("--force", action="store_true", help="Delete existing output root before running")
    p.add_argument("--no-download", action="store_true", help="Only log URLs; do not download images")
    p.add_argument("--dry-run", action="store_true", help="Print planned tasks and exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    options = [x.strip().upper() for x in args.options.split(",") if x.strip()]
    bad_options = [x for x in options if x not in OPTION_LABELS]
    if bad_options:
        raise SystemExit(f"Unsupported option labels: {bad_options}; use subset of A,B,C,D")

    plans = load_jsonl(args.plan_path)
    if args.offset:
        plans = plans[args.offset:]
    if args.limit > 0:
        plans = plans[:args.limit]

    tasks = make_tasks(plans, options)
    print("=" * 72)
    print("  Raw Top-k Retrieval Gallery Collector")
    print("=" * 72)
    print(f"  plans:      {len(plans)}")
    print(f"  tasks:      {len(tasks)} options")
    print(f"  top_k:      {args.top_k}")
    print(f"  expected:   {len(tasks) * args.top_k} candidate records")
    print(f"  endpoint:   {args.client_url}")
    print(f"  index:      {args.index_name}")
    print(f"  output:     {args.output_root}")

    if args.dry_run:
        for t in tasks[:20]:
            print(f"  {t.query_id} {t.option_label} {t.option_semantic}: {t.option_text}")
        if len(tasks) > 20:
            print(f"  ... {len(tasks) - 20} more tasks")
        return

    if args.force and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    log_path = args.output_root / "topk_results.jsonl"
    if log_path.exists():
        log_path.unlink()

    write_json(args.output_root / "run_config.json", {
        "plan_path": str(args.plan_path),
        "client_url": args.client_url,
        "index_name": args.index_name,
        "output_root": str(args.output_root),
        "top_k": args.top_k,
        "limit": args.limit,
        "offset": args.offset,
        "options": options,
        "workers": args.workers,
        "no_download": args.no_download,
    })

    all_records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    done = 0

    if args.workers <= 1:
        for task in tasks:
            try:
                records = collect_one(task, args)
                append_jsonl(log_path, records)
                all_records.extend(records)
            except Exception as e:
                errors.append({"query_id": task.query_id, "option_label": task.option_label, "error": str(e)})
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"  [{done}/{len(tasks)}] records={len(all_records)} errors={len(errors)}")
    else:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(collect_one, task, args): task for task in tasks}
            for fut in cf.as_completed(futs):
                task = futs[fut]
                try:
                    records = fut.result()
                    append_jsonl(log_path, records)
                    all_records.extend(records)
                except Exception as e:
                    errors.append({"query_id": task.query_id, "option_label": task.option_label, "error": str(e)})
                done += 1
                if done % 10 == 0 or done == len(tasks):
                    print(f"  [{done}/{len(tasks)}] records={len(all_records)} errors={len(errors)}")

    write_json(args.output_root / "errors.json", errors)
    gallery_path = build_gallery(
        records=all_records,
        output_root=args.output_root,
        title=f"Top-{args.top_k} retrieval gallery",
    )

    print("\nDone.")
    print(f"  records: {len(all_records)}")
    print(f"  errors:  {len(errors)}")
    print(f"  log:     {log_path}")
    print(f"  gallery: {gallery_path}")
    if errors:
        print("  See errors.json for failed option tasks.")


if __name__ == "__main__":
    main()
