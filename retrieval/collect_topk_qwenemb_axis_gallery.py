#!/usr/bin/env python3
"""Collect top-k candidates and rerank with QwenEmb axis scores.

This is the corrected diagnostic variant for the reranking idea:

  - Retrieval: QwenEmb text -> image top-k for the full option query.
  - pattern_score: patch-wise pattern coverage, scored by a VLM over image tiles.
  - color_score: raw QwenEmb IP/cosine between the color-axis text and candidate image vector.
  - garment_score: raw QwenEmb IP/cosine between the garment-axis text and candidate image vector.
  - combined_score: sqrt(max(0, pattern_score * color_score * garment_score)).

Unlike VLM yes/no scoring, color_score and garment_score are embedding similarity
values. They are not coverage probabilities and are not expected to become 1.0.

Requires a QwenEmb server that exposes /score-candidates. Restart
QwenEmb/serve_qwenemb_knn.py after pulling the matching server patch.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import requests

# Allow both:
#   python retrieval/collect_topk_qwenemb_axis_gallery.py
# and:
#   python -m retrieval.collect_topk_qwenemb_axis_gallery
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval.collect_topk_scored_gallery import (  # noqa: E402
    OPTION_LABELS,
    append_jsonl,
    build_gallery,
    collect_one,
    grouped_rerank,
    load_jsonl,
    make_tasks,
    score_pattern,
    write_json,
)
from retrieval.print_scored_rankings import group_records, print_group  # noqa: E402


def derive_axis_score_url(client_url: str) -> str:
    if client_url.rstrip("/").endswith("/knn-service"):
        return client_url.rstrip("/")[: -len("/knn-service")] + "/score-candidates"
    return client_url.rstrip("/") + "/score-candidates"


def post_axis_scores(
    session: requests.Session,
    axis_score_url: str,
    color_text: str,
    garment_text: str,
    indices: list[int],
    timeout: float,
    retries: int,
) -> dict[int, dict[str, float]]:
    payload = {
        "texts": {
            "color_score": color_text,
            "garment_score": garment_text,
        },
        "indices": indices,
    }
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = session.post(axis_score_url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and "error" in data:
                raise RuntimeError(str(data["error"]))
            rows = data.get("scores") if isinstance(data, dict) else None
            if not isinstance(rows, list):
                raise RuntimeError(f"Expected {{scores: [...]}} response, got: {data}")
            out: dict[int, dict[str, float]] = {}
            for row in rows:
                if not isinstance(row, dict) or "faiss_index" not in row:
                    continue
                try:
                    idx = int(row["faiss_index"])
                    out[idx] = {
                        "color_score": float(row.get("color_score", 0.0)),
                        "garment_score": float(row.get("garment_score", 0.0)),
                    }
                except Exception:
                    continue
            return out
        except Exception as e:
            last_err = e
            if attempt >= retries:
                break
            time.sleep(min(2.0 ** attempt, 8.0))
    raise RuntimeError(f"Axis score request failed: {last_err}")


def get_faiss_index(rec: dict[str, Any]) -> int | None:
    for obj in [rec, rec.get("raw") if isinstance(rec.get("raw"), dict) else None]:
        if not isinstance(obj, dict):
            continue
        if obj.get("faiss_index") is not None:
            try:
                return int(obj["faiss_index"])
            except Exception:
                return None
    return None


def axis_texts(rec: dict[str, Any]) -> tuple[str, str]:
    color = str(rec.get("target_color") or "").strip()
    garment = str(rec.get("target_garment") or "").strip()
    # Keep these axis-isolated. Do not use the full phrase such as "black striped coat".
    return color, garment


def score_records_for_option(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if not rows:
        return rows
    color_text, garment_text = axis_texts(rows[0])
    session = requests.Session()

    indices: list[int] = []
    for rec in rows:
        idx = get_faiss_index(rec)
        rec["faiss_index"] = idx
        if idx is not None:
            indices.append(idx)

    axis_scores: dict[int, dict[str, float]] = {}
    axis_error: str | None = None
    if indices:
        try:
            axis_scores = post_axis_scores(
                session=session,
                axis_score_url=args.axis_score_url,
                color_text=color_text,
                garment_text=garment_text,
                indices=indices,
                timeout=args.axis_timeout,
                retries=args.retries,
            )
        except Exception as e:
            axis_error = str(e)

    out: list[dict[str, Any]] = []
    vlm_urls = [u.strip() for u in args.vlm_urls.split(",") if u.strip()]
    for i, rec in enumerate(rows):
        idx = rec.get("faiss_index")
        local_path = rec.get("local_path")
        vlm_url = vlm_urls[i % len(vlm_urls)] if vlm_urls else "http://127.0.0.1:8002/v1"

        if local_path and Path(local_path).exists() and not args.no_pattern_score:
            pattern_info = score_pattern(
                image_path=Path(local_path),
                pattern=str(rec.get("target_pattern") or ""),
                rec_id=f"{rec.get('query_id')}__{rec.get('option_label')}__rank_{int(rec.get('original_rank') or 0):03d}",
                args=args,
                vlm_url=vlm_url,
            )
            pattern_score = float(pattern_info.get("score") or 0.0)
        else:
            pattern_info = {"score": 0.0, "mode": "missing_local_image_or_disabled", "tile_scores": []}
            pattern_score = 0.0

        one_axis = axis_scores.get(int(idx), {}) if idx is not None else {}
        color_score = float(one_axis.get("color_score", 0.0))
        garment_score = float(one_axis.get("garment_score", 0.0))
        combined = math.sqrt(max(0.0, pattern_score * color_score * garment_score))

        rec.update({
            "pattern_score": pattern_score,
            "color_score": color_score,
            "garment_score": garment_score,
            "combined_score": combined,
            "rerank_formula": "sqrt(max(0, pattern_score * qwenemb_color_ip * qwenemb_garment_ip))",
            "pattern_score_source": "vlm_patch_coverage",
            "color_score_source": "qwenemb_axis_ip",
            "garment_score_source": "qwenemb_axis_ip",
            "axis_color_text": color_text,
            "axis_garment_text": garment_text,
            "pattern_score_raw": pattern_info,
            "axis_score_error": axis_error,
        })
        out.append(rec)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--plan-path", type=Path, default=Path("data/options/option_plans.jsonl"))
    p.add_argument("--client-url", default="http://127.0.0.1:1236/knn-service")
    p.add_argument("--axis-score-url", default=None)
    p.add_argument("--index-name", default="pod_qwenemb")
    p.add_argument("--vlm-urls", default="http://127.0.0.1:8002/v1")
    p.add_argument("--vlm-model", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--output-root", type=Path, default=Path("data/retrieval/qwenemb_axis_scored_gallery"))
    p.add_argument("--retrieval-k", type=int, default=48)
    p.add_argument("--score-top-n", type=int, default=48)
    p.add_argument("--show-k", type=int, default=12)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--options", default="A,B,C,D")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--score-workers", type=int, default=1, help="Parallel option-level scoring workers")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--download-timeout", type=float, default=30.0)
    p.add_argument("--axis-timeout", type=float, default=120.0)
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
    p.add_argument("--no-pattern-score", action="store_true")
    p.add_argument("--print-table", action="store_true", default=True)
    p.add_argument("--no-print-table", dest="print_table", action="store_false")
    p.add_argument("--print-sort-by", choices=["combined", "pattern", "color", "garment", "similarity"], default="combined")
    p.add_argument("--print-top-n", type=int, default=20)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.axis_score_url = args.axis_score_url or derive_axis_score_url(args.client_url)
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

    print("=" * 80)
    print("  QwenEmb axis-scored Top-k Retrieval Gallery")
    print("=" * 80)
    print(f"  plans:        {len(plans)}")
    print(f"  option tasks: {len(tasks)}")
    print(f"  retrieval_k:  {args.retrieval_k}")
    print(f"  score_top_n:  {args.score_top_n or 'all'}")
    print(f"  endpoint:     {args.client_url}")
    print(f"  axis scores:  {args.axis_score_url}")
    print(f"  VLM URLs:     {args.vlm_urls}")
    print(f"  formula:      sqrt(max(0, pattern_coverage * qwenemb_color_ip * qwenemb_garment_ip))")
    print(f"  output:       {args.output_root}")

    if args.dry_run:
        for t in tasks[:20]:
            print(f"  {t.query_id} {t.option_label}: {t.option_text} | color={t.target_color} pattern={t.target_pattern} garment={t.target_garment}")
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
                errors.append({"query_id": task.query_id, "option_label": task.option_label, "error": str(e)})
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
                    errors.append({"query_id": task.query_id, "option_label": task.option_label, "error": str(e)})
                done += 1
                if done % 10 == 0 or done == len(tasks):
                    print(f"  retrieve [{done}/{len(tasks)}] records={len(raw_records)} errors={len(errors)}")
    write_json(args.output_root / "retrieval_errors.json", errors)

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for rec in raw_records:
        groups.setdefault((str(rec.get("query_id")), str(rec.get("option_label"))), []).append(rec)

    scored_records: list[dict[str, Any]] = []
    items = list(groups.values())
    done = 0
    if args.score_workers <= 1:
        for rows in items:
            scored_records.extend(score_records_for_option(rows, args))
            done += 1
            if done % 4 == 0 or done == len(items):
                print(f"  score    [{done}/{len(items)}] option groups")
    else:
        with cf.ThreadPoolExecutor(max_workers=args.score_workers) as ex:
            futs = [ex.submit(score_records_for_option, rows, args) for rows in items]
            for fut in cf.as_completed(futs):
                scored_records.extend(fut.result())
                done += 1
                if done % 4 == 0 or done == len(items):
                    print(f"  score    [{done}/{len(items)}] option groups")

    scored_records = grouped_rerank(scored_records)
    append_jsonl(scored_log, scored_records)
    gallery_path = build_gallery(scored_records, args.output_root, "QwenEmb axis-scored top-k retrieval gallery", args.show_k)

    print("\nDone.")
    print(f"  raw records:    {len(raw_records)}")
    print(f"  scored records: {len(scored_records)}")
    print(f"  retrieval errs: {len(errors)}")
    print(f"  raw log:        {raw_log}")
    print(f"  scored log:     {scored_log}")
    print(f"  gallery:        {gallery_path}")

    if args.print_table:
        print()
        grouped = group_records(scored_records)
        for key in sorted(grouped):
            print_group(grouped[key], sort_by=args.print_sort_by, top_n=args.print_top_n, caption_width=58, show_score_ranks=True)


if __name__ == "__main__":
    main()
