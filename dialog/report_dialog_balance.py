#!/usr/bin/env python3
"""대화 분포 리포트 — report_track_balance 의 대화판.

    POD_VARIANT=wacv_scenario_v5 python -m dialog.report_dialog_balance

확인 항목:
  * 커버리지: 사용자당 evidence 18개, 전체 24x18=432
  * 방식 분포와 downgrade(요청 방식이 셀 부족으로 완화된 건수)
  * 축 / 극성 / 에피소드 분포
  * differ 위치 카운터밸런스 (evidence 이미지가 first/second 균등한가)
  * 이미지 파일 실재 여부 — 라이브러리가 available 이라 해도 파일이
    실제로 있는지는 별개다
  * 셀 재사용 빈도 (특정 셀에 쏠리면 대화가 단조로워진다)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from configs.config import DATA_DIR, IMAGES_DIR                    # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogs", type=Path, default=DATA_DIR / "dialogs")
    ap.add_argument("--images-root", type=Path, default=IMAGES_DIR)
    args = ap.parse_args()

    files = sorted(args.dialogs.glob("dlg__*__image.json"))
    if not files:
        raise SystemExit(f"no dialogs in {args.dialogs}")

    ty, req, ax, pol, ep = Counter(), Counter(), Counter(), Counter(), Counter()
    idx = Counter()
    cells = Counter()
    per_user = {}
    downgrades = []
    missing = Counter()
    n_ref = 0

    for path in files:
        d = json.loads(path.read_text())
        uid = d["user_id"]
        per_user[uid] = {"turns": len(d["turns"]), "ev": len(d["evidence"]),
                         "imgs": sum(len(t["images"]) for t in d["turns"])}
        downgrades += [f"{uid}: {x}" for x in d.get("downgrades", [])]
        for e in d["evidence"]:
            ty[e["expression_type"]] += 1
            req[e["requested_type"]] += 1
            ax[e["axis"]] += 1
            pol[e["polarity"]] += 1
            ep[e["episode_id"]] += 1
            if e["expression_type"] == "differ":
                idx[e["evidence_index"]] += 1
            for a in e["image_attributes"]:
                cells[f'{a["color"]}|{a["garment_category"]}|{a["pattern"]}'] += 1
        for t in d["turns"]:
            for f in t["images"]:
                n_ref += 1
                if not (args.images_root / f).is_file():
                    missing[f] += 1

    n = len(files)
    total_ev = sum(ty.values())
    print("=" * 62)
    print(f"  대화 {n}개  evidence {total_ev} (기대 {n * 18})"
          + ("  ✓" if total_ev == n * 18 else "  ← 커버리지 미달"))
    t = sum(v["turns"] for v in per_user.values())
    i = sum(v["imgs"] for v in per_user.values())
    print(f"  평균 턴 {t / n:.1f}  이미지 {i / n:.1f}  (턴당 {i / t:.2f})")

    print("\n  표현 방식 (실제 / 요청)")
    for k in ("single", "differ", "share2", "share3"):
        print(f"    {k:8s} {ty[k]:4d} / {req[k]:4d}"
              + (f"   {ty[k] - req[k]:+d}" if ty[k] != req[k] else ""))
    if downgrades:
        print(f"    downgrade {len(downgrades)}건 "
              f"({len(downgrades) / total_ev:.1%})")
        for x in downgrades[:5]:
            print(f"      - {x}")

    print("\n  축:", dict(ax), " 극성:", dict(pol))
    print("  에피소드:", dict(ep))
    if idx:
        tot = sum(idx.values())
        print(f"  differ 위치: first {idx[0]} / second {idx[1]} "
              f"({idx[0] / tot:.0%} vs {idx[1] / tot:.0%})"
              + ("  ← 불균형" if abs(idx[0] - idx[1]) > tot * 0.15 else "  ✓"))

    print(f"\n  고유 셀 {len(cells)}개 / 참조 {sum(cells.values())}회")
    print("  최다 사용:", ", ".join(f"{k}×{v}" for k, v in cells.most_common(3)))
    if missing:
        print(f"\n  ⚠ 파일 없는 이미지 {len(missing)}종 / 참조 {n_ref}회")
        for k, v in missing.most_common(5):
            print(f"      {k} ×{v}")
    else:
        print(f"  이미지 파일 확인: {n_ref}회 참조 전부 존재 ✓"
              if args.images_root.is_dir() and any(args.images_root.iterdir())
              else "  (이미지 디렉터리 비어 있음 — 파일 확인 생략)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
