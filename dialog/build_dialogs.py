#!/usr/bin/env python3
"""대화 생성 진입점 — 계획(D3) → 렌더링(D4) → 저장.

    POD_VARIANT=wacv_scenario_v5 python -m dialog.build_dialogs
    POD_VARIANT=wacv_scenario_v5 python -m dialog.build_dialogs --users U001 U002

산출물:
    data_<variant>/dialogs/dlg__U001__image.json
    data_<variant>/dialogs/dlg__U001__text.json
    data_<variant>/dialogs/dialogs_manifest.json

검증은 별도다: python -m dialog.validate_dialogs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.config import DATA_DIR                              # noqa: E402
from configs.dialog_templates import (DIALOG_TEMPLATE_VERSION,    # noqa: E402
                                      EPISODES)
from dialog.dialog_planner import plan_dialog                     # noqa: E402
from dialog.image_picker import load_available_cells              # noqa: E402
from dialog.renderer import render, text_twin                     # noqa: E402

LIBRARY = ROOT / "annotation" / "attribute_library.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", nargs="*", default=None, help="기본: 전체")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=DATA_DIR / "dialogs")
    ap.add_argument("--library", type=Path, default=LIBRARY)
    args = ap.parse_args()

    cells = load_available_cells(args.library)
    lib_hash = hashlib.sha256(args.library.read_bytes()).hexdigest()[:16]

    profiles = []
    with open(DATA_DIR / "profiles" / "profiles.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            if args.users is None or r["user_id"] in args.users:
                profiles.append(r)

    args.out.mkdir(parents=True, exist_ok=True)
    episodes = [ep_id for ep_id, _, _ in EPISODES]

    manifest = {"variant": DATA_DIR.name, "seed": args.seed,
                "template_version": DIALOG_TEMPLATE_VERSION,
                "library_sha256_16": lib_hash, "episodes": episodes,
                "dialogs": []}
    failures = 0
    turns_tot = imgs_tot = 0

    for prof in profiles:
        plan = plan_dialog(prof, cells, episodes, args.seed)
        if plan.failures:
            failures += 1
            for f in plan.failures:
                print(f"  ! {prof['user_id']}: {f}")
        img_dlg = render(plan)
        txt_dlg = text_twin(img_dlg)
        n_img = sum(len(t["images"]) for t in img_dlg["turns"])
        turns_tot += len(img_dlg["turns"])
        imgs_tot += n_img

        for d in (img_dlg, txt_dlg):
            (args.out / f"{d['dialog_id']}.json").write_text(
                json.dumps(d, ensure_ascii=False, indent=1))
        manifest["dialogs"].append({
            "user_id": prof["user_id"], "turns": len(img_dlg["turns"]),
            "evidence": len(img_dlg["evidence"]), "images": n_img,
            "downgrades": plan.downgrades,
            "cells": sorted({f'{a["color"]}|{a["garment_category"]}|{a["pattern"]}'
                             for e in img_dlg["evidence"]
                             for a in e["image_attributes"]})})
        print(f"  {prof['user_id']}  turns={len(img_dlg['turns']):3d} "
              f"ev={len(img_dlg['evidence']):2d} imgs={n_img:3d}"
              + (f"  downgrades={len(plan.downgrades)}" if plan.downgrades else ""))

    (args.out / "dialogs_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1))
    n = len(profiles)
    print(f"\n{n} users -> {2 * n} files in {args.out}")
    if n:
        print(f"평균 턴 {turns_tot / n:.1f}  이미지 {imgs_tot / n:.1f}  "
              f"(턴당 {imgs_tot / turns_tot:.2f})")
    print(f"library {lib_hash}  templates {DIALOG_TEMPLATE_VERSION}")
    if failures:
        print(f"\n{failures} user(s) with unfillable preferences — 위 로그 확인")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
