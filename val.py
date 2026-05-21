"""
option_plans.jsonl 품질 검증 스크립트
"""
from collections import Counter, defaultdict
from src.utils import load_jsonl
from src.option_planner import tpo_compatible, TPO_ATTR_INCOMPATIBILITIES

plans = load_jsonl("data/options/option_plans.jsonl")
print(f"총 {len(plans)}개 plan 로드\n")

errors   = []
warnings = []

for p in plans:
    qid        = p["query_id"]
    axis       = p["active_axis"]
    qtype      = p["query_type"]
    scenario   = p.get("tpo_axis") and {}   # tpo_scenario는 plan에 없으므로 옵션 속성으로만 검증
    opts       = p["options"]
    attrs      = {k: opts[k]["attributes"] for k in "ABCD"}

    # ── 1. 라벨 4종이 모두 다른가 ─────────────────────────────────
    labels = [opts[k]["label"] for k in "ABCD"]
    if sorted(labels) != sorted(["tpo_and_preference","tpo_only","preference_only","neither"]):
        errors.append(f"{qid}: 라벨 누락/중복 {labels}")

    # ── 2. A/B/C/D 속성이 모두 다른가 (중복 옵션 없는가) ──────────
    attr_strs = [str(sorted(attrs[k].items())) for k in "ABCD"]
    if len(set(attr_strs)) < 4:
        errors.append(f"{qid}: 중복 옵션 존재 {attrs}")

    # ── 3. active_axis 기준 A==C (liked), B==D (disliked) ─────────
    if axis in ("color", "pattern"):
        if attrs["A"].get(axis) != attrs["C"].get(axis):
            errors.append(f"{qid}: A와 C의 {axis} 불일치")
        if attrs["B"].get(axis) != attrs["D"].get(axis):
            errors.append(f"{qid}: B와 D의 {axis} 불일치")
        # A/B는 garment_category가 같아야
        if attrs["A"].get("garment_category") != attrs["B"].get("garment_category"):
            errors.append(f"{qid}: A와 B의 garment_category 불일치")
        # C/D는 A와 garment_category가 달라야 (violation)
        if attrs["C"].get("garment_category") == attrs["A"].get("garment_category"):
            warnings.append(f"{qid}: C의 garment_category가 A와 동일 (violation 없음)")

    elif axis == "garment_category":
        if attrs["A"].get(axis) != attrs["C"].get(axis):
            errors.append(f"{qid}: A와 C의 garment_category 불일치")
        if attrs["B"].get(axis) != attrs["D"].get(axis):
            errors.append(f"{qid}: B와 D의 garment_category 불일치")

    # ── 4. A != B (liked vs disliked가 실제로 다른가) ─────────────
    if attrs["A"].get(axis) == attrs["B"].get(axis):
        errors.append(f"{qid}: A와 B의 {axis} 동일 (liked==disliked)")

    # ── 5. search_query가 비어있지 않은가 ─────────────────────────
    for k in "ABCD":
        if not opts[k].get("search_query", "").strip():
            warnings.append(f"{qid}-{k}: search_query 비어있음")

# ── 분포 통계 ──────────────────────────────────────────────────────
print("=== 분포 통계 ===")
print(f"axis  분포: {dict(Counter(p['active_axis']  for p in plans))}")
print(f"qtype 분포: {dict(Counter(p['query_type']   for p in plans))}")
print(f"user  분포: {dict(Counter(p['user_id']      for p in plans))}")

violation_axes = Counter(p["tpo_violation_axis"] for p in plans)
print(f"violation_axis 분포: {dict(violation_axes)}")

# ── 결과 요약 ──────────────────────────────────────────────────────
print(f"\n=== 검증 결과 ===")
if errors:
    print(f"❌ ERROR {len(errors)}건:")
    for e in errors:
        print(f"   {e}")
else:
    print("✅ ERROR 없음")

if warnings:
    print(f"⚠️  WARNING {len(warnings)}건:")
    for w in warnings[:20]:   # 너무 많으면 20개만
        print(f"   {w}")
else:
    print("✅ WARNING 없음")