#!/usr/bin/env python3
"""Print terminal ranking reports from scored retrieval JSONL.

Reads retrieval/collect_topk_scored_gallery.py's scored_results.jsonl and prints
one terminal table per (query_id, option_label), including combined, pattern,
color, garment, retrieval similarity, original retrieval rank, and per-score
rank diagnostics.
"""

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
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fnum(x: Any, digits: int = 3, none: str = "nan") -> str:
    try:
        v = float(x)
        if math.isnan(v):
            return none
        return f"{v:.{digits}f}"
    except Exception:
        return none


def finteger(x: Any, none: str = "-") -> str:
    try:
        return str(int(x))
    except Exception:
        return none


def short(text: Any, width: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= width:
        return text
    return text[: max(0, width - 1)] + "…"


def score_value(rec: dict[str, Any], key: str) -> float:
    try:
        v = float(rec.get(key) or 0.0)
        if math.isnan(v):
            return 0.0
        return v
    except Exception:
        return 0.0


def pattern_count(rec: dict[str, Any]) -> str:
    info = rec.get("pattern_score_raw") or {}
    if not isinstance(info, dict):
        return "-"
    mode = info.get("mode")
    if mode == "patch_coverage":
        tiles = info.get("tile_scores") or []
        if isinstance(tiles, list) and tiles:
            hit = sum(1 for t in tiles if isinstance(t, dict) and t.get("present"))
            return f"{hit}/{len(tiles)}"
        n = info.get("num_tiles")
        if n:
            return f"?/{n}"
    if mode == "solid_global":
        return "solid"
    return "-"


def group_records(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((str(r.get("query_id")), str(r.get("option_label"))), []).append(r)
    return groups


def assign_score_ranks(rows: list[dict[str, Any]]) -> None:
    for rank_name, score_key in SCORE_KEYS.items():
        ordered = sorted(
            rows,
            key=lambda r: (score_value(r, score_key), score_value(r, "similarity"), -int(r.get("original_rank") or 10**9)),
            reverse=True,
        )
        for i, rec in enumerate(ordered):
            rec[f"rank_{rank_name}"] = i


def sort_rows(rows: list[dict[str, Any]], sort_by: str) -> list[dict[str, Any]]:
    score_key = SCORE_KEYS[sort_by]
    return sorted(
        rows,
        key=lambda r: (score_value(r, score_key), score_value(r, "similarity"), -int(r.get("original_rank") or 10**9)),
        reverse=True,
    )


def print_group(rows: list[dict[str, Any]], sort_by: str, top_n: int, caption_width: int, show_score_ranks: bool) -> None:
    if not rows:
        return
    assign_score_ranks(rows)
    ordered = sort_rows(rows, sort_by)
    head = ordered[0]

    query_id = str(head.get("query_id") or "")
    opt = str(head.get("option_label") or "")
    target_pattern = str(head.get("target_pattern") or "")
    target_color = str(head.get("target_color") or "")
    target_garment = str(head.get("target_garment") or "")
    query = str(head.get("option_text") or "")

    print("=" * 104)
    print(f"{query_id}  opt {opt}  target: pattern={target_pattern} color={target_color} garment={target_garment}")
    print(f"   query: {query}")

    if show_score_ranks:
        print("   newrank retr  rP rC rG   comb   pat   col  garm  pat/val     sim  caption")
    else:
        print("   newrank retr   comb   pat   col  garm  pat/val     sim  caption")

    for new_rank, rec in enumerate(ordered[:top_n]):
        marker = ""
        try:
            if int(rec.get("original_rank")) == 1:
                marker = "  <- retr top1"
        except Exception:
            pass

        retr = finteger(rec.get("original_rank"))
        comb = fnum(rec.get("combined_score"), 3)
        pat = fnum(rec.get("pattern_score"), 3)
        col = fnum(rec.get("color_score"), 3)
        gar = fnum(rec.get("garment_score"), 3)
        pval = pattern_count(rec)
        sim = fnum(rec.get("similarity"), 4)
        cap = short(rec.get("caption"), caption_width)

        if show_score_ranks:
            rp = finteger(rec.get("rank_pattern"))
            rc = finteger(rec.get("rank_color"))
            rg = finteger(rec.get("rank_garment"))
            print(f"   {new_rank:7d} {retr:>4} {rp:>3} {rc:>2} {rg:>2}  {comb:>6} {pat:>5} {col:>5} {gar:>5} {pval:>8} {sim:>7}  {cap}{marker}")
        else:
            print(f"   {new_rank:7d} {retr:>4}  {comb:>6} {pat:>5} {col:>5} {gar:>5} {pval:>8} {sim:>7}  {cap}{marker}")

    retrieval_top1 = min(rows, key=lambda r: int(r.get("original_rank") or 10**9))
    rerank_top1 = ordered[0]
    retr_rank = int(rerank_top1.get("original_rank") or -1)
    retr_top1_rank_after = int(retrieval_top1.get(f"rank_{sort_by}") or -1)
    score_key = SCORE_KEYS[sort_by]
    top_score = fnum(rerank_top1.get(score_key), 3)
    retr_top_score = fnum(retrieval_top1.get(score_key), 3)
    differs = retr_rank != int(retrieval_top1.get("original_rank") or -1)
    status = "DIFFERS" if differs else "same as retrieval top-1"
    print(
        f"   -> {sort_by}-first top-1: retr_rank {retr_rank}, {score_key} {top_score}"
        f"  [{status}; retrieval top-1 {score_key}={retr_top_score}, {sort_by}-rank={retr_top1_rank_after}]"
    )

    for name in ["pattern", "color", "garment", "combined", "similarity"]:
        if name == sort_by:
            continue
        best = sort_rows(rows, name)[0]
        print(
            f"      {name:>10} top-1: retr_rank {finteger(best.get('original_rank'))}, "
            f"{SCORE_KEYS[name]}={fnum(best.get(SCORE_KEYS[name]), 3)}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True, help="scored_results.jsonl path")
    p.add_argument("--sort-by", choices=sorted(SCORE_KEYS), default="combined")
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--query-id", default=None, help="Only print one query_id")
    p.add_argument("--option", default=None, help="Only print one option label, e.g. A")
    p.add_argument("--caption-width", type=int, default=58)
    p.add_argument("--no-score-ranks", action="store_true", help="Hide rP/rC/rG columns")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.results)
    if args.query_id:
        rows = [r for r in rows if str(r.get("query_id")) == args.query_id]
    if args.option:
        opt = args.option.strip().upper()
        rows = [r for r in rows if str(r.get("option_label")).upper() == opt]
    groups = group_records(rows)
    if not groups:
        raise SystemExit("No matching scored records found.")

    for key in sorted(groups):
        print_group(
            rows=groups[key],
            sort_by=args.sort_by,
            top_n=args.top_n,
            caption_width=args.caption_width,
            show_score_ranks=not args.no_score_ranks,
        )


if __name__ == "__main__":
    main()
