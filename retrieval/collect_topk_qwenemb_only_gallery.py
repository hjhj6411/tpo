#!/usr/bin/env python3
"""QwenEmb-only top-k retrieval reranking diagnostic.

No VLM is used.

Scores:
  pattern_score = patch-wise QwenEmb coverage
  color_score   = QwenEmb(text=color, image=candidate) IP/cosine
  garment_score = QwenEmb(text=garment, image=candidate) IP/cosine
  combined      = sqrt(max(0, pattern * color * garment))

Requires QwenEmb/serve_qwenemb_knn.py with:
  /knn-service
  /score-candidates
  /score-image-files
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import requests

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
    make_pattern_tiles,
    make_tasks,
    write_json,
)
from retrieval.print_scored_rankings import group_records, print_group  # noqa: E402


def endpoint(base: str, route: str) -> str:
    base = base.rstrip("/")
    if base.endswith("/knn-service"):
        base = base[: -len("/knn-service")]
    return base + route


def req_json(session: requests.Session, url: str, payload: dict[str, Any], timeout: float, retries: int) -> dict[str, Any]:
    err: Exception | None = None
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
            err = e
            if attempt >= retries:
                break
            time.sleep(min(2.0 ** attempt, 8.0))
    raise RuntimeError(f"request failed url={url}: {err}")


def faiss_index(rec: dict[str, Any]) -> int | None:
    for obj in (rec, rec.get("raw") if isinstance(rec.get("raw"), dict) else None):
        if not isinstance(obj, dict):
            continue
        if obj.get("faiss_index") is not None:
            try:
                return int(obj["faiss_index"])
            except Exception:
                return None
    return None


def axis_scores(rows: list[dict[str, Any]], args: argparse.Namespace, session: requests.Session) -> dict[int, dict[str, float]]:
    indices = []
    for r in rows:
        idx = faiss_index(r)
        r["faiss_index"] = idx
        if idx is not None:
            indices.append(idx)
    if not indices:
        return {}
    head = rows[0]
    color = str(head.get("target_color") or "").strip()
    garment = str(head.get("target_garment") or "").strip()
    data = req_json(
        session,
        args.axis_score_url,
        {"texts": {"color_score": color, "garment_score": garment}, "indices": indices},
        args.axis_timeout,
        args.retries,
    )
    out: dict[int, dict[str, float]] = {}
    for row in data.get("scores") or []:
        if not isinstance(row, dict) or row.get("faiss_index") is None:
            continue
        try:
            out[int(row["faiss_index"])] = {
                "color_score": float(row.get("color_score", 0.0)),
                "garment_score": float(row.get("garment_score", 0.0)),
            }
        except Exception:
            pass
    return out


def pattern_texts(pattern: str) -> tuple[str, str]:
    p = str(pattern or "").replace("_", " ").strip().lower()
    if p == "solid":
        return "plain solid fabric", "visible patterned fabric"
    return f"{p} fabric", "plain fabric"


def pattern_score(image_path: Path, pattern: str, rec_id: str, args: argparse.Namespace, session: requests.Session) -> dict[str, Any]:
    p = str(pattern or "").replace("_", " ").strip().lower()
    if not p:
        return {"score": 1.0, "mode": "missing_pattern_target", "tile_scores": []}

    tiles = make_pattern_tiles(
        image_path=image_path,
        grid=args.tile_grid,
        max_tiles=args.pattern_max_tiles,
        tile_dir=args.output_root / "tiles" / rec_id,
        save_tiles=True,
        white_thresh=args.tile_white_thresh,
        min_std=args.tile_min_std,
    ) or [image_path]

    pos_text, neg_text = pattern_texts(p)
    data = req_json(
        session,
        args.image_score_url,
        {
            "texts": {"pos": pos_text, "neg": neg_text},
            "image_paths": [str(x) for x in tiles],
        },
        args.image_score_timeout,
        args.retries,
    )

    tile_scores = []
    for tile, row in zip(tiles, data.get("scores") or []):
        pos = float(row.get("pos", 0.0)) if isinstance(row, dict) else 0.0
        neg = float(row.get("neg", 0.0)) if isinstance(row, dict) else 0.0
        margin = pos - neg
        present = margin >= args.pattern_margin
        tile_scores.append({
            "tile": str(tile),
            "pattern_text": pos_text,
            "negative_text": neg_text,
            "pattern_similarity": pos,
            "negative_similarity": neg,
            "margin": margin,
            "present": present,
            "error": row.get("error") if isinstance(row, dict) else None,
        })

    denom = max(1, len(tile_scores))
    hits = sum(1 for x in tile_scores if x["present"])
    return {
        "score": hits / denom,
        "mode": "qwenemb_patch_coverage",
        "hits": hits,
        "num_tiles": len(tile_scores),
        "margin_threshold": args.pattern_margin,
        "pattern_text": pos_text,
        "negative_text": neg_text,
        "avg_margin": sum(float(x["margin"]) for x in tile_scores) / denom,
        "avg_pattern_similarity": sum(float(x["pattern_similarity"]) for x in tile_scores) / denom,
        "avg_negative_similarity": sum(float(x["negative_similarity"]) for x in tile_scores) / denom,
        "tile_scores": tile_scores,
    }


def score_group(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    session = requests.Session()
    try:
        axes = axis_scores(rows, args, session)
        axis_error = None
    except Exception as e:
        axes = {}
        axis_error = str(e)

    out = []
    for rec in rows:
        local = rec.get("local_path")
        if local and Path(local).exists():
            try:
                pinfo = pattern_score(
                    Path(local),
                    str(rec.get("target_pattern") or ""),
                    f"{rec.get('query_id')}__{rec.get('option_label')}__rank_{int(rec.get('original_rank') or 0):03d}",
                    args,
                    session,
                )
                pscore = float(pinfo.get("score") or 0.0)
            except Exception as e:
                pinfo = {"score": 0.0, "mode": "qwenemb_patch_failed", "error": str(e), "tile_scores": []}
                pscore = 0.0
        else:
            pinfo = {"score": 0.0, "mode": "missing_local_image", "tile_scores": []}
            pscore = 0.0

        idx = rec.get("faiss_index")
        one = axes.get(int(idx), {}) if idx is not None else {}
        cscore = float(one.get("color_score", 0.0))
        gscore = float(one.get("garment_score", 0.0))
        combined = math.sqrt(max(0.0, pscore * cscore * gscore))
        rec.update({
            "pattern_score": pscore,
            "color_score": cscore,
            "garment_score": gscore,
            "combined_score": combined,
            "rerank_formula": "sqrt(max(0, qwenemb_pattern_coverage * qwenemb_color_ip * qwenemb_garment_ip))",
            "pattern_score_source": "qwenemb_patch_coverage",
            "color_score_source": "qwenemb_axis_ip",
            "garment_score_source": "qwenemb_axis_ip",
            "pattern_score_raw": pinfo,
            "axis_score_error": axis_error,
        })
        out.append(rec)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--plan-path", type=Path, default=Path("data/options/option_plans.jsonl"))
    p.add_argument("--client-url", default="http://127.0.0.1:1236/knn-service")
    p.add_argument("--axis-score-url", default=None)
    p.add_argument("--image-score-url", default=None)
    p.add_argument("--index-name", default="pod_qwenemb")
    p.add_argument("--output-root", type=Path, default=Path("data/retrieval/qwenemb_only_gallery"))
    p.add_argument("--retrieval-k", type=int, default=48)
    p.add_argument("--score-top-n", type=int, default=48)
    p.add_argument("--show-k", type=int, default=12)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--options", default="A,B,C,D")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--score-workers", type=int, default=1)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--download-timeout", type=float, default=30.0)
    p.add_argument("--axis-timeout", type=float, default=120.0)
    p.add_argument("--image-score-timeout", type=float, default=120.0)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--max-image-bytes", type=int, default=20_000_000)
    p.add_argument("--tile-grid", type=int, default=5)
    p.add_argument("--pattern-max-tiles", type=int, default=12)
    p.add_argument("--pattern-margin", type=float, default=0.0)
    p.add_argument("--tile-white-thresh", type=float, default=242.0)
    p.add_argument("--tile-min-std", type=float, default=8.0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--print-table", action="store_true", default=True)
    p.add_argument("--no-print-table", dest="print_table", action="store_false")
    p.add_argument("--print-sort-by", choices=["combined", "pattern", "color", "garment", "similarity"], default="combined")
    p.add_argument("--print-top-n", type=int, default=20)
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
    tasks = make_tasks(plans, options)

    print("=" * 80)
    print("  QwenEmb-only patch/axis Top-k Retrieval Gallery")
    print("=" * 80)
    print(f"  plans:        {len(plans)}")
    print(f"  option tasks: {len(tasks)}")
    print(f"  retrieval_k:  {args.retrieval_k}")
    print(f"  score_top_n:  {args.score_top_n or 'all'}")
    print(f"  endpoint:     {args.client_url}")
    print(f"  axis scores:  {args.axis_score_url}")
    print(f"  image scores: {args.image_score_url}")
    print(f"  formula:      sqrt(max(0, qwenemb_pattern_coverage * qwenemb_color_ip * qwenemb_garment_ip))")
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
    for path in (raw_log, scored_log):
        if path.exists():
            path.unlink()
    write_json(args.output_root / "run_config.json", vars(args) | {"options_list": options})

    raw_records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    done = 0
    with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(collect_one, t, args): t for t in tasks}
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
    with cf.ThreadPoolExecutor(max_workers=max(1, args.score_workers)) as ex:
        futs = [ex.submit(score_group, rows, args) for rows in items]
        for fut in cf.as_completed(futs):
            scored_records.extend(fut.result())
            done += 1
            if done % 4 == 0 or done == len(items):
                print(f"  score    [{done}/{len(items)}] option groups")

    scored_records = grouped_rerank(scored_records)
    append_jsonl(scored_log, scored_records)
    gallery = build_gallery(scored_records, args.output_root, "QwenEmb-only patch/axis top-k retrieval gallery", args.show_k)

    print("\nDone.")
    print(f"  raw records:    {len(raw_records)}")
    print(f"  scored records: {len(scored_records)}")
    print(f"  retrieval errs: {len(errors)}")
    print(f"  raw log:        {raw_log}")
    print(f"  scored log:     {scored_log}")
    print(f"  gallery:        {gallery}")

    if args.print_table:
        print()
        grouped = group_records(scored_records)
        for key in sorted(grouped):
            print_group(grouped[key], sort_by=args.print_sort_by, top_n=args.print_top_n, caption_width=58, show_score_ranks=True)


if __name__ == "__main__":
    main()
