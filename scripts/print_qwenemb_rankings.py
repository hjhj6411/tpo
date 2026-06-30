#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


OPTION_LABELS = ["A", "B", "C", "D"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"missing file: {path}")
    return [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]


def fnum(x: Any) -> float:
    try:
        return float(x or 0.0)
    except Exception:
        return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", type=Path, required=True)
    ap.add_argument("--show-k", type=int, default=100)
    ap.add_argument("--query-id", default="")
    ap.add_argument("--option", default="")
    args = ap.parse_args()

    records = load_jsonl(args.scored)

    if args.query_id:
        records = [r for r in records if str(r.get("query_id")) == args.query_id]
    if args.option:
        opt = args.option.upper()
        records = [r for r in records if str(r.get("option_label")).upper() == opt]

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in records:
        groups.setdefault((str(r.get("query_id")), str(r.get("option_label"))), []).append(r)

    print("=" * 140)
    print(f"RANKING VIEW | records={len(records)} | scored={args.scored}")
    print("=" * 140)

    for query_id in sorted({k[0] for k in groups}):
        print()
        print(f"[QUERY] {query_id}")
        print("-" * 140)

        for label in OPTION_LABELS:
            rows = groups.get((query_id, label), [])
            if not rows:
                continue

            rows = sorted(
                rows,
                key=lambda r: (fnum(r.get("combined_score")), fnum(r.get("similarity"))),
                reverse=True,
            )

            metric_ranks: dict[str, dict[int, int]] = {}
            for metric in ["combined_score", "pattern_score", "color_score", "garment_score"]:
                ordered = sorted(
                    rows,
                    key=lambda r: (fnum(r.get(metric)), fnum(r.get("similarity"))),
                    reverse=True,
                )
                metric_ranks[metric] = {id(r): i for i, r in enumerate(ordered, start=1)}

            head = rows[0]
            print()
            print(f"[{label}] {head.get('option_semantic')} | {head.get('option_text')}")
            print(
                f"target: color={head.get('target_color')} | "
                f"pattern={head.get('target_pattern')} | "
                f"garment={head.get('target_garment')}"
            )
            print()

            for i, r in enumerate(rows[: args.show_k], start=1):
                img = str(r.get("local_path") or "NO_IMAGE")
                sim = fnum(r.get("similarity"))
                comb = fnum(r.get("combined_score"))
                pat = fnum(r.get("pattern_score"))
                col = fnum(r.get("color_score"))
                gar = fnum(r.get("garment_score"))
                orig = int(r.get("original_rank") or 0)

                print(
                    f"{i:03d} | "
                    f"orig={orig:03d} | "
                    f"comb={comb:.4f} #{metric_ranks['combined_score'].get(id(r), 0):03d} | "
                    f"pat={pat:.4f} #{metric_ranks['pattern_score'].get(id(r), 0):03d} | "
                    f"col={col:.4f} #{metric_ranks['color_score'].get(id(r), 0):03d} | "
                    f"gar={gar:.4f} #{metric_ranks['garment_score'].get(id(r), 0):03d} | "
                    f"sim={sim:.4f} | "
                    f"{img}"
                )

    print()
    print("=" * 140)


if __name__ == "__main__":
    main()
