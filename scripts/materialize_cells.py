#!/usr/bin/env python3
"""materialize_cells.py — build a variant's image folder + images_manifest.json.

WHY THIS EXISTS
---------------
Two different sets were being confused with each other:

    annotation/attribute_library.json   "available"     a human approved the cell
    data_<variant>/images_manifest.json "materialized"  the jpg is in images/

Only the second one makes an option renderable, and the manifest that shipped
with `wacv_scenario_v5` was derived from the plan set that existed when it was
written: it lists exactly (cells used by those plans) ∩ (available). When the
plans were later rebuilt they moved onto 16 available cells that had never been
copied, and the evaluators silently skipped every item that used one. The
manifest was an untraceable by-product; this script makes it a rebuildable one.

WHAT IT DOES
------------
Copies one jpg per `color|garment_category|pattern` cell into

    <out>/images/<color>/<pattern>_<garment>.jpg

and writes `<out>/images_manifest.json` describing exactly what was copied.
Cells come from a base manifest (copied from that variant's image folder, so a
released image is never re-derived) plus, optionally, the cells referenced by a
plan file that the base manifest is missing (copied from the library's source
image under annotation/). Nothing is re-collected and nothing is re-annotated:
every byte is an existing file.

Deterministic, stdlib only, and it never writes inside the base variant.

    python -m scripts.materialize_cells \
        --library        annotation/attribute_library.json \
        --base-manifest  data_wacv_scenario_v5/images_manifest.json \
        --base-images    data_wacv_scenario_v5/images \
        --add-from-plans data_wacv_scenario_v5/options/option_plans.jsonl \
        --out            data_wacv_scenario_v5b \
        --variant        wacv_scenario_v5b
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS = ["A", "B", "C", "D"]


def cell_keys_of(plan) -> list[str]:
    """The four color|garment|pattern keys a plan needs, '' when unspecified."""
    out = []
    for k in LABELS:
        a = plan["options"][k].get("attributes", {})
        c, g, p = a.get("color"), a.get("garment_category"), a.get("pattern")
        out.append(f"{c}|{g}|{p}" if (c and g and p) else "")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", type=Path,
                    default=ROOT / "annotation" / "attribute_library.json")
    ap.add_argument("--base-manifest", type=Path, required=True,
                    help="manifest whose cells are copied as-is")
    ap.add_argument("--base-images", type=Path, required=True,
                    help="image folder the base manifest describes")
    ap.add_argument("--add-from-plans", type=Path, default=None,
                    help="also materialize every library-available cell these "
                         "plans reference that the base manifest lacks")
    ap.add_argument("--out", type=Path, required=True,
                    help="variant directory; images/ and images_manifest.json "
                         "are written under it")
    ap.add_argument("--variant", default=None,
                    help="value of the manifest's 'variant' field "
                         "(default: the --out directory name minus 'data_')")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    lib = json.loads(args.library.read_text())["cells"]
    base = json.loads(args.base_manifest.read_text())["cells"]
    variant = args.variant or args.out.name.removeprefix("data_")

    # ── which cells go in ────────────────────────────────────────────────
    entries = dict(base)                       # key -> {file, source, url_key}
    added, unavailable, missing_source = [], [], []
    if args.add_from_plans:
        wanted = set()
        with args.add_from_plans.open() as fh:
            for line in fh:
                if line.strip():
                    wanted.update(k for k in cell_keys_of(json.loads(line)) if k)
        for key in sorted(wanted - set(base)):
            entry = lib.get(key)
            if not entry or entry.get("status") != "available":
                unavailable.append(key)
                continue
            images = entry.get("images") or []
            if not images:
                missing_source.append(key)
                continue
            src_rel = min(im["path"] for im in images)   # 1 image/cell today
            img = next(im for im in images if im["path"] == src_rel)
            if not (args.library.parent / src_rel).is_file():
                missing_source.append(key)
                continue
            c, g, p = key.split("|")
            entries[key] = {"file": f"{c}/{p}_{g}.jpg", "source": src_rel,
                            "url_key": img.get("url_key")}
            added.append(key)

    print(f"  base manifest       : {len(base)} cells ({args.base_manifest})")
    if args.add_from_plans:
        print(f"  plans               : {args.add_from_plans}")
        print(f"  newly materialized  : {len(added)} cells")
        for key in added:
            print(f"      + {key:38s} <- annotation/{entries[key]['source']}")
        print(f"  wanted but NOT available in the library: {len(unavailable)}")
        for key in unavailable:
            print(f"      - {key:38s} "
                  f"status={lib.get(key, {}).get('status', '<not in library>')}")
        if missing_source:
            print(f"  available but source image missing: {len(missing_source)}")
            for key in missing_source:
                print(f"      ! {key}")
    print(f"  total               : {len(entries)} cells -> {args.out}/images")
    if args.dry_run:
        print("  (dry run — nothing written)")
        return 0

    # ── copy ─────────────────────────────────────────────────────────────
    images_dir = args.out / "images"
    n_base = n_new = 0
    for key, ent in sorted(entries.items()):
        dst = images_dir / ent["file"]
        if dst.is_file():
            continue
        src = (args.base_images / ent["file"] if key in base
               else args.library.parent / ent["source"])
        if not src.is_file():
            raise SystemExit(f"[error] source image not found: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        n_base += key in base
        n_new += key not in base
    print(f"  copied              : {n_base} from {args.base_images}, "
          f"{n_new} from {args.library.parent}")

    out_manifest = args.out / "images_manifest.json"
    out_manifest.write_text(json.dumps(
        {"variant": variant,
         "schema": "color|garment_category|pattern -> images/<color>/<pattern>_<garment>.jpg",
         "n_cells": len(entries),
         "cells": {k: entries[k] for k in sorted(entries)}},
        ensure_ascii=False, indent=1) + "\n")
    print(f"  -> {out_manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
