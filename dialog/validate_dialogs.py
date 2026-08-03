#!/usr/bin/env python3
"""대화 검증 — 생성기와 독립 실행.

생성기 안에서만 검증하면 '생성기가 자기 산출물을 자기 기준으로 통과시키는'
순환이 생긴다. 별도 모듈로 두면 손으로 쓴 대화나 LLM 패러프레이즈판에도
같은 검증을 적용할 수 있다.

검사 항목:
  R1  시나리오·TPO 어휘 금지 — 대화는 취향만 전달한다
  R2  image 변형 턴 텍스트에 벤치마크 어휘(13색·23의류·6무늬)와 렌더링
      별칭이 등장하지 않는다. 축 단어(color/print/cut)는 허용
  R3  비-evidence 축 값은 사용자-중립 (expression.check 안에서 확인)
  R4  모든 이미지는 검수 완료 available 셀
  R5  방식별 연역 조건 (differ 단일축 차이, share 단일축 공유 등)
  C   커버리지 — 프로필의 취향 18개가 전부 노출됨
  P   극성 쿼터 — like/dislike 둘 다 존재

usage:
    python -m dialog.validate_dialogs
    python -m dialog.validate_dialogs --self-test   # 돌연변이 테스트
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
from pathlib import Path

from configs.config import DATA_DIR, FASHION_ATTRIBUTE_AXES
from dialog.dialog_planner import all_preferences
from dialog.expression import AXES, EXPRESSIONS
from dialog.image_picker import ImagePicker, load_available_cells

ROOT = Path(__file__).resolve().parent.parent
LIBRARY = ROOT / "annotation" / "attribute_library.json"

# 렌더링 별칭까지 포함한 금지 표면형 (R2)
RENDER_ALIASES = {
    "t_shirt": ["t-shirt", "t shirt", "tee"],
    "tank_top": ["tank top"],
    "formal_shirt": ["formal shirt", "dress shirt", "button-up", "button up"],
    "polo_shirt": ["polo shirt", "polo"],
    "suit_vest": ["suit vest", "vest", "waistcoat"],
    "leather_jacket": ["leather jacket"],
    "puffer_jacket": ["puffer jacket", "puffer"],
    "fleece_jacket": ["fleece jacket", "fleece"],
    "pea_coat": ["pea coat", "peacoat"],
    "long_coat": ["long coat", "overcoat", "wool coat"],
    "mini_skirt": ["mini skirt", "miniskirt"],
    "long_skirt": ["long skirt", "maxi skirt"],
    "polka_dot": ["polka dot", "polka-dot", "dots", "dotted"],
    "leopard": ["leopard print", "animal print", "leopard"],
    "checkered": ["checked", "check", "plaid", "checkered"],
    "floral": ["flower print", "flowers", "floral"],
    "striped": ["stripes", "stripe", "striped"],
}

SCENARIO_TERMS = [
    "blizzard", "wedding", "funeral", "interview", "gym", "hike", "beach",
    "court", "gala", "temple", "church", "meeting", "dress code", "ceremony",
    "reception", "safari", "yoga", "ski", "regatta", "opera",
]


def _leak_terms() -> list[str]:
    terms = set()
    for axis, values in FASHION_ATTRIBUTE_AXES.items():
        for v in values:
            if v == "solid":
                continue          # 무늬 축의 영 수준. 대화에 쓰지도 않는다
            terms.add(v.replace("_", " "))
            terms.update(RENDER_ALIASES.get(v, []))
    return sorted(terms, key=len, reverse=True)


LEAK_TERMS = _leak_terms()


def find_leaks(text: str) -> list[str]:
    low = f" {text.lower()} "
    hits = []
    for t in LEAK_TERMS:
        for suffix in (" ", "s ", ". ", ", ", "! ", "? ", "\u2014"):
            if f" {t}{suffix}" in low:
                hits.append(t)
                break
    return hits


def validate(dialog: dict, profile: dict, cells: set[str]) -> list[str]:
    errs: list[str] = []
    picker = ImagePicker(profile, cells, random.Random(0))
    is_image = dialog.get("variant") == "image"

    # C — 커버리지
    want = set(all_preferences(profile))
    got = {(e["axis"], e["value"], e["polarity"]) for e in dialog["evidence"]}
    if want != got:
        for miss in sorted(want - got):
            errs.append(f"[C] not exposed: {miss}")
        for extra in sorted(got - want):
            errs.append(f"[C] not in profile: {extra}")

    # P — 극성
    pols = {e["polarity"] for e in dialog["evidence"]}
    if not {"like", "dislike"} <= pols:
        errs.append(f"[P] polarity quota violated: {sorted(pols)}")

    for e in dialog["evidence"]:
        expr = EXPRESSIONS.get(e["expression_type"])
        if expr is None:
            errs.append(f"[R5] unknown expression {e['expression_type']}")
            continue
        imgs = e["image_attributes"]
        # R5 + R3
        for msg in expr.check(imgs, e["axis"], e["value"],
                              e.get("evidence_index"), picker.is_neutral):
            errs.append(f"[R5] {e['value']}: {msg}")
        # R4
        for im in imgs:
            key = f'{im["color"]}|{im["garment_category"]}|{im["pattern"]}'
            if key not in cells:
                errs.append(f"[R4] cell not available: {key}")
        # 턴-이미지 개수 정합
        turn = dialog["turns"][e["revealed_in_turn"]]
        if is_image and len(turn["images"]) != len(imgs):
            errs.append(f"[R5] turn {turn['turn_id']} image count mismatch")

    for turn in dialog["turns"]:
        low = turn["text"].lower()
        for term in SCENARIO_TERMS:                       # R1
            # 단어 경계로 본다. 'ski' 가 'skirt' 안에서 걸리면 안 된다.
            if re.search(rf"\b{re.escape(term)}\b", low):
                errs.append(f"[R1] scenario term '{term}' in turn {turn['turn_id']}")
        if is_image:                                      # R2
            hits = find_leaks(turn["text"])
            if hits:
                errs.append(f"[R2] vocab leak turn {turn['turn_id']}: {hits}")
    return errs


# ── 돌연변이 테스트: 일부러 위반을 주입해 검증기가 잡는지 확인 ─────────────
def _mutations(dialog, profile):
    muts = []

    def m(name, fn):
        d = copy.deepcopy(dialog)
        if fn(d):
            muts.append((name, d))

    def leak(d):
        for t in d["turns"]:
            if t["images"]:
                t["text"] += " I love blue."
                return True
    def drop(d):
        for i, e in enumerate(d["evidence"]):
            if e["expression_type"] != "single":
                d["evidence"].pop(i)
                return True
    def scenario(d):
        d["turns"][0]["text"] += " Also I have a wedding next week."
        return True
    def break_share(d):
        for e in d["evidence"]:
            if e["expression_type"].startswith("share"):
                a = e["axis"]
                other = next(x for x in AXES if x != a)
                v = e["image_attributes"][0][other]
                for im in e["image_attributes"]:
                    im[other] = v            # 두 축이 공통이 되어 연역 붕괴
                return True
    def break_differ(d):
        for e in d["evidence"]:
            if e["expression_type"] == "differ":
                other = next(x for x in AXES if x != e["axis"])
                e["image_attributes"][1][other] = "black"
                return True
    def bad_cell(d):
        d["evidence"][0]["image_attributes"][0]["color"] = "chartreuse"
        return True

    for name, fn in [("R2 leak", leak), ("C coverage", drop),
                     ("R1 scenario", scenario), ("R5 share", break_share),
                     ("R5 differ", break_differ), ("R4 cell", bad_cell)]:
        m(name, fn)
    return muts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogs", type=Path, default=DATA_DIR / "dialogs")
    ap.add_argument("--library", type=Path, default=LIBRARY)
    ap.add_argument("--self-test", action="store_true",
                    help="돌연변이 주입 후 검증기가 잡는지 확인")
    args = ap.parse_args()

    cells = load_available_cells(args.library)
    profiles = {}
    with open(DATA_DIR / "profiles" / "profiles.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            profiles[r["user_id"]] = r

    files = sorted(args.dialogs.glob("dlg__*.json"))
    if not files:
        raise SystemExit(f"no dialogs in {args.dialogs}")

    n_bad = 0
    for path in files:
        d = json.loads(path.read_text())
        if any("expression_type" not in e for e in d.get("evidence", [])):
            print(f"[SKIP] {path.name:28s} (다른 스키마 — 이 검증기 대상 아님)")
            continue
        errs = validate(d, profiles[d["user_id"]], cells)
        status = "OK " if not errs else "FAIL"
        n_bad += bool(errs)
        n_img = sum(len(t["images"]) for t in d["turns"])
        print(f"[{status}] {path.name:28s} turns={len(d['turns']):3d} "
              f"ev={len(d['evidence']):2d} imgs={n_img:3d}"
              + (f"  downgrades={len(d.get('downgrades', []))}"
                 if d.get("downgrades") else ""))
        for e in errs[:6]:
            print("     !", e)

    print(f"\n{len(files) - n_bad}/{len(files)} passed")

    if args.self_test:
        print("\n== mutation test ==")
        d = json.loads(files[0].read_text())
        prof = profiles[d["user_id"]]
        caught = 0
        muts = _mutations(d, prof)
        for name, mutant in muts:
            errs = validate(mutant, prof, cells)
            ok = bool(errs)
            caught += ok
            print(f"  [{'caught' if ok else 'MISSED'}] {name}")
        print(f"  {caught}/{len(muts)} mutations caught")
        if caught != len(muts):
            return 1
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
