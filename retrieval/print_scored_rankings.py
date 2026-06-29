#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

SCORE_KEYS = {
    "combined": "combined_score",
    "pattern": "pattern_score",
    "color": "color_score",
    "garment": "garment_score",
    "similarity": "similarity",
}

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def group_records(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups = {}
    for r in rows:
        groups.setdefault((str(r.get("query_id")), str(r.get("option_label"))), []).append(r)
    return groups

def score_value(rec: dict[str, Any], key: str) -> float:
    try:
        v = float(rec.get(key) or 0.0)
        return 0.0 if math.isnan(v) else v
    except Exception:
        return 0.0

def fnum(x: Any, digits: int = 3) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "nan"

def short(text: Any, width: int) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= width else s[: max(0, width - 1)] + "…"

def pattern_count(rec: dict[str, Any]) -> str:
    info = rec.get("pattern_score_raw") or {}
    if not isinstance(info, dict):
        return "-"
    hits = info.get("hits")
    n = info.get("num_tiles")
    if hits is not None and n is not None:
        return f"{hits}/{n}"
    tiles = info.get("tile_scores") or []
    if isinstance(tiles, list) and tiles:
        return f"{sum(1 for t in tiles if isinstance(t, dict) and t.get('present'))}/{len(tiles)}"
    return "-"

def assign_score_ranks(rows: list[dict[str, Any]]) -> None:
    for rank_name, score_key in SCORE_KEYS.items():
        ordered = sorted(
            rows,
            key=lambda r: (
                score_value(r, score_key),
                score_value(r, "similarity"),
                -int(r.get("original_rank") or 10**9),
            ),
            reverse=True,
        )
        for i, rec in enumerate(ordered):
            rec[f"rank_{rank_name}"] = i

def sorted_by(rows: list[dict[str, Any]], sort_by: str) -> list[dict[str, Any]]:
    key = SCORE_KEYS[sort_by]
    return sorted(
        rows,
        key=lambda r: (
            score_value(r, key),
            score_value(r, "similarity"),
            -int(r.get("original_rank") or 10**9),
        ),
        reverse=True,
    )

def print_group(
    rows: list[dict[str, Any]],
    sort_by: str,
    top_n: int,
    caption_width: int = 58,
    show_score_ranks: bool = True,
) -> None:
    if not rows:
        return

    assign_score_ranks(rows)
    ordered = sorted_by(rows, sort_by)
    head = ordered[0]

    print("=" * 104)
    print(
        f"{head.get('query_id')}  opt {head.get('option_label')}  "
        f"target: pattern={head.get('target_pattern')} "
        f"color={head.get('target_color')} "
        f"garment={head.get('target_garment')}"
    )
    print(f"   query: {head.get('option_text')}")

    if show_score_ranks:
        print("   newrank retr  rP rC rG   comb   pat   col  garm  pat/val     sim  caption")
    else:
        print("   newrank retr   comb   pat   col  garm  pat/val     sim  caption")

    for new_rank, rec in enumerate(ordered[:top_n]):
        original_rank = int(rec.get("original_rank") or -1)
        marker = "  <- retr top1" if original_rank == 1 else ""

        if show_score_ranks:
            print(
                f"   {new_rank:7d} {original_rank:4d} "
                f"{int(rec.get('rank_pattern') or -1):3d} "
                f"{int(rec.get('rank_color') or -1):2d} "
                f"{int(rec.get('rank_garment') or -1):2d}  "
                f"{fnum(rec.get('combined_score')):>6} "
                f"{fnum(rec.get('pattern_score')):>5} "
                f"{fnum(rec.get('color_score')):>5} "
                f"{fnum(rec.get('garment_score')):>5} "
                f"{pattern_count(rec):>8} "
                f"{fnum(rec.get('similarity'), 4):>7}  "
                f"{short(rec.get('caption'), caption_width)}{marker}"
            )
        else:
            print(
                f"   {new_rank:7d} {original_rank:4d}  "
                f"{fnum(rec.get('combined_score')):>6} "
                f"{fnum(rec.get('pattern_score')):>5} "
                f"{fnum(rec.get('color_score')):>5} "
                f"{fnum(rec.get('garment_score')):>5} "
                f"{pattern_count(rec):>8} "
                f"{fnum(rec.get('similarity'), 4):>7}  "
                f"{short(rec.get('caption'), caption_width)}{marker}"
            )

    retrieval_top1 = min(rows, key=lambda r: int(r.get("original_rank") or 10**9))
    top = ordered[0]
    score_key = SCORE_KEYS[sort_by]
    differs = int(top.get("original_rank") or -1) != int(retrieval_top1.get("original_rank") or -1)
    status = "DIFFERS" if differs else "same as retrieval top-1"
    print(
        f"   -> {sort_by}-first top-1: retr_rank {top.get('original_rank')}, "
        f"{score_key} {fnum(top.get(score_key))}  "
        f"[{status}; retrieval top-1 {score_key}={fnum(retrieval_top1.get(score_key))}, "
        f"{sort_by}-rank={retrieval_top1.get('rank_' + sort_by)}]"
    )

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--sort-by", choices=sorted(SCORE_KEYS), default="combined")
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--query-id", default=None)
    p.add_argument("--option", default=None)
    p.add_argument("--caption-width", type=int, default=58)
    p.add_argument("--no-score-ranks", action="store_true")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.results)
    if args.query_id:
        rows = [r for r in rows if str(r.get("query_id")) == args.query_id]
    if args.option:
        rows = [r for r in rows if str(r.get("option_label")).upper() == args.option.upper()]
    groups = group_records(rows)
    if not groups:
        raise SystemExit("No matching scored records found.")
    for key in sorted(groups):
        print_group(
            groups[key],
            sort_by=args.sort_by,
            top_n=args.top_n,
            caption_width=args.caption_width,
            show_score_ranks=not args.no_score_ranks,
        )

if __name__ == "__main__":
    main()
