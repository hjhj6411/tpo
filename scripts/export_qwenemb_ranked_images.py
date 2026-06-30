#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


OPTION_LABELS = ["A", "B", "C", "D"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"missing scored file: {path}")
    return [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]


def fnum(x: Any) -> float:
    try:
        return float(x or 0.0)
    except Exception:
        return 0.0


def slug(text: Any, max_len: int = 120) -> str:
    s = str(text or "")
    out = []
    for ch in s:
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_")[:max_len] or "unknown"


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if copy:
        shutil.copy2(src, dst)
    else:
        rel = os.path.relpath(src.resolve(), start=dst.parent.resolve())
        os.symlink(rel, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--show-k", type=int, default=100)
    ap.add_argument("--query-id", default="")
    ap.add_argument("--option", default="")
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlink")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    records = load_jsonl(args.scored)

    if args.query_id:
        records = [r for r in records if str(r.get("query_id")) == args.query_id]
    if args.option:
        records = [r for r in records if str(r.get("option_label")).upper() == args.option.upper()]

    root = args.scored.parent
    out_dir = args.out_dir or (root / "ranked_images")

    if args.force and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in records:
        groups.setdefault((str(r.get("query_id")), str(r.get("option_label"))), []).append(r)

    total = 0

    for (query_id, option_label), rows in sorted(groups.items()):
        rows = sorted(
            rows,
            key=lambda r: (
                fnum(r.get("combined_score")),
                fnum(r.get("similarity")),
            ),
            reverse=True,
        )[: args.show_k]

        if not rows:
            continue

        head = rows[0]
        option_text = slug(head.get("option_text"), 80)
        option_dir = out_dir / slug(query_id, 80) / f"{option_label}_{option_text}"
        option_dir.mkdir(parents=True, exist_ok=True)

        lines = []
        lines.append(f"query_id: {query_id}")
        lines.append(f"option: {option_label}")
        lines.append(f"option_text: {head.get('option_text')}")
        lines.append(
            f"target: color={head.get('target_color')} | "
            f"pattern={head.get('target_pattern')} | "
            f"garment={head.get('target_garment')}"
        )
        lines.append("")
        lines.append("rank\torig\tcombined\tpattern\tcolor\tgarment\tsimilarity\tfile")

        for rank, r in enumerate(rows, start=1):
            src_raw = r.get("local_path")
            if not src_raw:
                continue
            src = Path(src_raw)
            if not src.exists():
                continue

            ext = src.suffix.lower() or ".jpg"
            orig = int(r.get("original_rank") or 0)
            comb = fnum(r.get("combined_score"))
            pat = fnum(r.get("pattern_score"))
            col = fnum(r.get("color_score"))
            gar = fnum(r.get("garment_score"))
            sim = fnum(r.get("similarity"))

            name = (
                f"{rank:03d}"
                f"__orig_{orig:03d}"
                f"__comb_{comb:.4f}"
                f"__pat_{pat:.4f}"
                f"__col_{col:.4f}"
                f"__gar_{gar:.4f}"
                f"__sim_{sim:.4f}"
                f"{ext}"
            )
            dst = option_dir / name
            link_or_copy(src, dst, copy=args.copy)

            lines.append(
                f"{rank:03d}\t{orig:03d}\t{comb:.6f}\t{pat:.6f}\t"
                f"{col:.6f}\t{gar:.6f}\t{sim:.6f}\t{name}"
            )
            total += 1

        (option_dir / "_ranking.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(option_dir)

    print()
    print(f"exported ranked images: {total}")
    print(f"out_dir: {out_dir}")


if __name__ == "__main__":
    main()
