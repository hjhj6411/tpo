"""
POD-Bench v2 — Data Quality Verification Script
================================================
실행: python scripts/verify_data.py
      python scripts/verify_data.py --sample 20   # 육안 검증용 샘플 출력
      python scripts/verify_data.py --spot N       # 특정 인스턴스 상세 출력

Layer 1: Structural integrity  (자동, 전수)
Layer 2: Semantic spot-check   (자동 생성 + 육안 검토용 출력)
Layer 3: Label consistency     (rule vs plan 불일치 탐지)
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from configs.scenarios import get_scenario_by_id, get_constrained_axes

# ── 파일 경로 ─────────────────────────────────────────────
PROFILES_PATH = ROOT / "data/profiles/profiles.jsonl"
QUERIES_PATH  = ROOT / "data/queries/queries.jsonl"
PLANS_PATH    = ROOT / "data/options/option_plans.jsonl"

def load_jsonl(path):
    if not Path(path).exists():
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


# ══════════════════════════════════════════════════════════
#  LAYER 1: STRUCTURAL INTEGRITY  (전수 자동 검증)
# ══════════════════════════════════════════════════════════

EXPECTED_LABELS = {"tpo_and_preference", "tpo_only", "preference_only", "neither"}

def check_2x2_structure(plan):
    """A/B/C/D 레이블이 정확히 2×2 행렬을 구성하는지 확인."""
    labels = [plan["options"][k]["label"] for k in "ABCD"]
    label_set = set(labels)
    missing = EXPECTED_LABELS - label_set
    duplicate = [l for l in labels if labels.count(l) > 1]
    errors = []
    if missing:
        errors.append(f"missing labels: {missing}")
    if duplicate:
        errors.append(f"duplicate labels: {set(duplicate)}")
    return errors

def check_active_axis_isolation(plan):
    """fixed_attrs가 active_axis 값을 포함하지 않는지 확인."""
    errors = []
    active = plan["active_axis"]
    fixed = plan.get("fixed_attrs", {})
    if active in fixed:
        errors.append(f"active_axis '{active}' found in fixed_attrs: {fixed}")
    return errors

def check_violation_value_in_incompatible(plan):
    """violation_value가 scenario의 incompatible에 실제로 있는지 확인."""
    errors = []
    scenario = get_scenario_by_id(plan.get("scenario_id", ""))
    if not scenario:
        return [f"scenario not found: {plan.get('scenario_id')}"]
    v_axis = plan.get("violation_axis")
    v_val  = plan.get("violation_value")
    if not v_axis or not v_val:
        return []
    constraint = scenario.get(v_axis)
    if constraint is None:
        # violation_axis는 scenario가 제약하지 않는 축일 수 있음 (fallback case)
        return []
    incompat = set(constraint.get("incompatible", []))
    if v_val not in incompat:
        errors.append(
            f"violation_value '{v_val}' NOT in scenario[{v_axis}].incompatible={incompat}"
        )
    return errors

def check_ab_values_differ(plan):
    """A와 B의 active_axis 값이 서로 다른지 확인."""
    errors = []
    active = plan["active_axis"]
    a_val = plan["options"]["A"]["attributes"].get(active)
    b_val = plan["options"]["B"]["attributes"].get(active)
    if a_val == b_val:
        errors.append(f"A and B have same active_axis value: {a_val}")
    return errors

def check_cd_violation(plan):
    """C/D 옵션이 실제로 TPO 위반 값을 가지는지 확인.

    Two modes:
    - Normal (violation_axis != active_axis): C and D both use the same
      violation_value on a secondary axis (e.g. garment when axis=color).
    - Same-axis (violation_axis == active_axis): C and D each use a
      DIFFERENT incompatible value. We verify both are in the scenario's
      incompatible set, not that they equal violation_value (D's value only).
    """
    errors = []
    v_axis  = plan.get("violation_axis")
    v_val   = plan.get("violation_value")
    active  = plan.get("active_axis")
    if not v_axis or not v_val:
        return []

    scenario = get_scenario_by_id(plan.get("scenario_id", ""))
    if not scenario:
        return []

    if v_axis != active:
        # ── Normal case: C and D must both carry violation_value ──────
        for k in ("C", "D"):
            val = plan["options"][k]["attributes"].get(v_axis)
            if val != v_val:
                errors.append(
                    f"Option {k} has {v_axis}={val}, expected violation={v_val}")
    else:
        # ── Same-axis case: C and D values must each be in incompatible ──
        constraint = scenario.get(v_axis)
        if constraint is None:
            return []
        incompat_set = set(constraint.get("incompatible", []))
        for k in ("C", "D"):
            val = plan["options"][k]["attributes"].get(v_axis)
            if val not in incompat_set:
                errors.append(
                    f"Option {k}: {v_axis}={val} not in scenario incompatible={incompat_set}")
        # Also verify C and D use different values (otherwise 2×2 collapses)
        c_val = plan["options"]["C"]["attributes"].get(v_axis)
        d_val = plan["options"]["D"]["attributes"].get(v_axis)
        if c_val and d_val and c_val == d_val:
            errors.append(f"C and D have same {v_axis}={c_val} in same-axis case")

    return errors

def run_layer1(plans):
    print("\n" + "═"*60)
    print("  LAYER 1: Structural Integrity (전수 검증)")
    print("═"*60)

    checkers = [
        ("2×2 label structure",        check_2x2_structure),
        ("active_axis isolation",       check_active_axis_isolation),
        ("violation_value in incompat", check_violation_value_in_incompatible),
        ("A≠B on active_axis",          check_ab_values_differ),
        ("C/D carry violation value",   check_cd_violation),
    ]

    total = len(plans)
    results = {name: [] for name, _ in checkers}

    for plan in plans:
        for name, fn in checkers:
            errs = fn(plan)
            if errs:
                results[name].append((plan["query_id"], errs))

    all_pass = True
    for name, failures in results.items():
        n_fail = len(failures)
        status = "✓" if n_fail == 0 else "✗"
        print(f"  {status} {name}: {total - n_fail}/{total} pass")
        if failures:
            all_pass = False
            for qid, errs in failures[:3]:
                print(f"      [{qid}] {errs[0]}")
            if len(failures) > 3:
                print(f"      ... and {len(failures)-3} more")

    if all_pass:
        print(f"\n  ✅ All structural checks passed ({total} plans)")
    else:
        total_fail = sum(len(v) for v in results.values())
        print(f"\n  ❌ {total_fail} structural issues found")
    return all_pass


# ══════════════════════════════════════════════════════════
#  LAYER 2: SEMANTIC SPOT-CHECK  (샘플링 + 육안 검토용 출력)
# ══════════════════════════════════════════════════════════

def check_query_scenario_alignment(query):
    """쿼리 텍스트가 시나리오 상황을 담는지 휴리스틱 검사.
    implicit 쿼리는 키워드를 의도적으로 숨기므로 낮은 기준 적용."""
    scenario = get_scenario_by_id(query.get("scenario_id", ""))
    if not scenario:
        return None
    query_text  = query.get("query_text", "").lower()
    query_type  = query.get("query_type", "explicit_tpo")
    scenario_name = scenario["name"].lower()
    stop = {"with", "and", "the", "for", "all", "day", "guest", "visit"}
    name_keywords = [w for w in scenario_name.split() if len(w) > 3 and w not in stop]
    tpo_keywords = []
    for axis, sub in scenario.get("tpo", {}).items():
        if isinstance(sub, dict):
            for v in sub.values():
                word = str(v).lower().replace("_", " ")
                if len(word) > 3:
                    tpo_keywords.append(word)
    all_keywords = list(set(name_keywords + tpo_keywords))
    matched      = [k for k in all_keywords if k in query_text]
    coverage     = len(matched) / max(len(all_keywords), 1)
    is_implicit  = (query_type == "implicit_tpo")
    threshold    = 0.15 if is_implicit else 0.40
    return {"coverage": coverage, "keywords": all_keywords, "matched": matched,
            "is_implicit": is_implicit, "low": coverage < threshold}


def check_profile_narrative_completeness(profile):
    """narrative가 likes/dislikes 키워드를 반영하는지 fuzzy 검사.
    trench_coat -> trench 또는 coat 하나라도 있으면 통과."""
    narrative = profile.get("narrative_profile", "").lower()
    likes    = profile.get("likes_keywords", [])
    dislikes = profile.get("dislikes_keywords", [])
    missing_likes = []
    missing_dislikes = []
    def fuzzy_in(kw, text):
        val = kw.split(":")[-1]
        tokens = val.replace("_", " ").split()
        exact  = val.replace("_", " ")
        if exact in text:
            return True
        if all(t in text for t in tokens):
            return True
        return any(t in text for t in tokens if len(t) > 3)
    for kw in likes:
        if not fuzzy_in(kw, narrative):
            missing_likes.append(kw.split(":")[-1])
    for kw in dislikes:
        if not fuzzy_in(kw, narrative):
            missing_dislikes.append(kw.split(":")[-1])
    return {"missing_likes": missing_likes, "missing_dislikes": missing_dislikes}

def run_layer2_stats(queries, profiles_map, n_sample=None):
    """시맨틱 검증 통계 (전수)."""
    print("\n" + "═"*60)
    print("  LAYER 2: Semantic Integrity (통계)")
    print("═"*60)

    # 2-1. Query-scenario alignment (implicit/explicit 분리 평가)
    res_exp, res_imp = [], []
    low_exp, low_imp = [], []
    for q in queries:
        r = check_query_scenario_alignment(q)
        if not r:
            continue
        if r["is_implicit"]:
            res_imp.append(r["coverage"])
            if r["low"]:
                low_imp.append((q["query_id"], r))
        else:
            res_exp.append(r["coverage"])
            if r["low"]:
                low_exp.append((q["query_id"], r))
    print(f"\n  Query-Scenario coverage (explicit, threshold≥40%):")
    if res_exp:
        avg_e = sum(res_exp) / len(res_exp)
        ok_e  = len(res_exp) - len(low_exp)
        print(f"    avg={avg_e:.2f}  ok={ok_e}/{len(res_exp)} ({ok_e/len(res_exp):.1%})")
        for qid, r in low_exp[:3]:
            print(f"    ⚠ [{qid}] matched={r['matched']}")
    print(f"\n  Query-Scenario coverage (implicit, threshold≥15%):")
    if res_imp:
        avg_i = sum(res_imp) / len(res_imp)
        ok_i  = len(res_imp) - len(low_imp)
        print(f"    avg={avg_i:.2f}  ok={ok_i}/{len(res_imp)} ({ok_i/len(res_imp):.1%})")
        for qid, r in low_imp[:3]:
            print(f"    ⚠ [{qid}] matched={r['matched']}")

    # 2-2. Profile narrative completeness (fuzzy match)
    missing_any = 0
    missing_details = []
    for uid, p in profiles_map.items():
        r = check_profile_narrative_completeness(p)
        if r["missing_likes"] or r["missing_dislikes"]:
            missing_any += 1
            missing_details.append((uid, r))
    n_profiles = len(profiles_map)
    print(f"\n  Profile narrative completeness (fuzzy):")
    print(f"    Profiles with ≥1 unmatched keyword: {missing_any}/{n_profiles}")
    for uid, r in missing_details[:3]:
        print(f"    [{uid}] missing likes={r['missing_likes']} dislikes={r['missing_dislikes']}")

    qtype_dist = Counter(q.get("query_type") for q in queries)
    print(f"\n  Query type distribution:")
    for qt, n in sorted(qtype_dist.items()):
        print(f"    {qt}: {n} ({n/len(queries):.1%})")

    # 2-4. Axis distribution
    axis_dist = Counter(q.get("active_axis") for q in queries)
    print(f"\n  Active axis distribution:")
    for ax, n in sorted(axis_dist.items()):
        print(f"    {ax}: {n} ({n/len(queries):.1%})")

    # 2-5. Archetype coverage
    arch_dist = Counter(q.get("scenario_archetype") for q in queries)
    print(f"\n  Scenario archetype distribution:")
    for arch, n in sorted(arch_dist.items()):
        print(f"    {arch:25s}: {n}")


# ══════════════════════════════════════════════════════════
#  LAYER 3: LABEL CONSISTENCY  (rule vs plan 불일치 탐지)
# ══════════════════════════════════════════════════════════

def infer_label_rule(opt_attrs, user_structured, scenario, active_axis, violation_axis):
    """rule_match 간소화 버전."""
    prefs = user_structured.get(active_axis, {})
    likes = set(prefs.get("likes", []))
    dislikes = set(prefs.get("dislikes", []))
    active_val = opt_attrs.get(active_axis)
    pref_ok = active_val in likes

    # TPO: violation_axis 값이 incompatible에 있으면 위반
    tpo_ok = True
    for ax in [violation_axis, active_axis, "garment_category", "color", "pattern"]:
        c = scenario.get(ax)
        if c is None:
            continue
        incompat = set(c.get("incompatible", []))
        if opt_attrs.get(ax) in incompat:
            tpo_ok = False
            break

    if tpo_ok and pref_ok:   return "tpo_and_preference"
    if tpo_ok:               return "tpo_only"
    if pref_ok:              return "preference_only"
    return "neither"

def run_layer3(plans, profiles_map):
    print("\n" + "═"*60)
    print("  LAYER 3: Label Consistency (rule vs plan)")
    print("═"*60)

    mismatches = []
    for plan in plans:
        uid = plan["user_id"]
        profile = profiles_map.get(uid)
        if not profile:
            continue
        scenario = get_scenario_by_id(plan.get("scenario_id", ""))
        if not scenario:
            continue
        active  = plan["active_axis"]
        v_axis  = plan.get("violation_axis", "garment_category")
        user_sa = profile["structured_attributes"]

        for k in "ABCD":
            opt = plan["options"][k]
            rule_label = infer_label_rule(
                opt["attributes"], user_sa, scenario, active, v_axis)
            plan_label = opt["label"]
            if rule_label != plan_label:
                mismatches.append({
                    "query_id": plan["query_id"],
                    "option": k,
                    "plan_label": plan_label,
                    "rule_label": rule_label,
                    "attrs": opt["attributes"],
                })

    total_decisions = len(plans) * 4
    n_ok = total_decisions - len(mismatches)
    print(f"\n  Rule vs plan agreement: {n_ok}/{total_decisions} = {n_ok/total_decisions:.1%}")

    if mismatches:
        print(f"  ⚠ {len(mismatches)} mismatches found (top 5):")
        for m in mismatches[:5]:
            print(f"    [{m['query_id']}] opt={m['option']} "
                  f"plan={m['plan_label']} rule={m['rule_label']} "
                  f"attrs={m['attrs']}")

        # 불일치 분포 분석
        mismatch_types = Counter(
            (m["plan_label"], m["rule_label"]) for m in mismatches)
        print(f"\n  Mismatch type breakdown:")
        for (p, r), n in mismatch_types.most_common(5):
            print(f"    plan={p} → rule={r}: {n}")
    else:
        print("  ✅ Perfect rule-plan agreement")

    return mismatches


# ══════════════════════════════════════════════════════════
#  육안 검토용 샘플 출력
# ══════════════════════════════════════════════════════════

def print_sample(plans, queries_map, profiles_map, n=10, seed=42):
    """n개의 인스턴스를 사람이 읽기 쉬운 형식으로 출력."""
    print("\n" + "═"*60)
    print(f"  SPOT CHECK: {n} random instances (육안 검토)")
    print("═"*60)

    rng = random.Random(seed)
    sample = rng.sample(plans, min(n, len(plans)))

    for i, plan in enumerate(sample):
        qid    = plan["query_id"]
        uid    = plan["user_id"]
        query  = queries_map.get(qid, {})
        profile = profiles_map.get(uid, {})
        scenario = get_scenario_by_id(plan.get("scenario_id", ""))

        print(f"\n{'─'*56}")
        print(f"  [{i+1}] {qid}")
        print(f"  User  : {uid} ({profile.get('preference_archetype','')})")
        print(f"  Scenario: {plan.get('scenario_id','')} "
              f"[{scenario['archetype'] if scenario else '?'}]")
        print(f"  Axis  : {plan['active_axis']}  |  violation: "
              f"{plan.get('violation_axis','')}={plan.get('violation_value','')}")
        print(f"\n  QUERY : \"{query.get('query_text','')}\"")
        print(f"  TYPE  : {query.get('query_type','')}")
        print(f"\n  Profile likes   : {profile.get('likes_keywords',[])}")
        print(f"  Profile dislikes: {profile.get('dislikes_keywords',[])}")
        print(f"\n  Options:")
        for k in "ABCD":
            opt = plan["options"][k]
            label = opt["label"]
            attrs = opt["attributes"]
            marker = " ← CORRECT" if label == "tpo_and_preference" else ""
            print(f"    {k} [{label:22s}] {attrs}{marker}")
        print(f"\n  Rationale A: {plan['options']['A'].get('rationale','')}")
        print(f"  Rationale C: {plan['options']['C'].get('rationale','')}")

    print(f"\n{'─'*56}")
    print(f"  위 {n}개 인스턴스를 검토하여 다음을 확인하세요:")
    print(f"  1. QUERY가 시나리오 상황을 자연스럽게 담는가?")
    print(f"  2. 옵션 A(정답)가 직관적으로 맞는 선택인가?")
    print(f"  3. 옵션 C(선호O + TPO위반)가 실제로 부적절한가?")
    print(f"  4. Profile likes/dislikes와 옵션이 논리적으로 연결되는가?")


# ══════════════════════════════════════════════════════════
#  특정 인스턴스 상세 출력
# ══════════════════════════════════════════════════════════

def print_spot(plans, queries_map, profiles_map, idx=0):
    plan = plans[idx]
    qid  = plan["query_id"]
    uid  = plan["user_id"]
    query   = queries_map.get(qid, {})
    profile = profiles_map.get(uid, {})
    scenario = get_scenario_by_id(plan.get("scenario_id", ""))

    print(f"\n{'═'*60}")
    print(f"  DETAILED VIEW: instance #{idx}  ({qid})")
    print(f"{'═'*60}")
    print(f"\n  ── Scenario ──────────────────────────────────────")
    if scenario:
        print(f"  name      : {scenario['name']}")
        print(f"  archetype : {scenario['archetype']}")
        print(f"  tpo       : {json.dumps(scenario['tpo'], ensure_ascii=False)}")
        for ax in ("garment_category", "color", "pattern"):
            c = scenario.get(ax)
            if c:
                print(f"  {ax:18s}: compat={c['compatible']}")
                print(f"  {'':18s}  incompat={c['incompatible']}")
        print(f"  justification: {scenario.get('justification','')}")

    print(f"\n  ── User Profile ──────────────────────────────────")
    print(f"  user_id   : {uid}  archetype={profile.get('preference_archetype','')}")
    print(f"  narrative : {profile.get('narrative_profile','')}")
    print(f"  likes     : {profile.get('likes_keywords',[])}")
    print(f"  dislikes  : {profile.get('dislikes_keywords',[])}")

    print(f"\n  ── Query ─────────────────────────────────────────")
    print(f"  text      : \"{query.get('query_text','')}\"")
    print(f"  type      : {query.get('query_type','')}")
    print(f"  active_axis: {plan['active_axis']}")
    print(f"  fixed_attrs: {plan.get('fixed_attrs',{})}")
    print(f"  violation  : {plan.get('violation_axis','')} = {plan.get('violation_value','')}")

    print(f"\n  ── Options ───────────────────────────────────────")
    for k in "ABCD":
        opt = plan["options"][k]
        marker = " ← CORRECT ANSWER" if opt["label"] == "tpo_and_preference" else ""
        print(f"  {k} [{opt['label']:22s}]{marker}")
        print(f"    attrs: {opt['attributes']}")
        print(f"    query: \"{opt['search_query']}\"")
        print(f"    why  : {opt.get('rationale','')}")


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="POD-Bench v2 data verification")
    parser.add_argument("--sample", type=int, default=0,
                        help="육안 검토용 인스턴스 수 (기본 0 = 출력 안 함)")
    parser.add_argument("--spot", type=int, default=-1,
                        help="특정 인스턴스 번호 상세 출력")
    parser.add_argument("--seed", type=int, default=42,
                        help="샘플링 랜덤 시드")
    parser.add_argument("--layer", type=int, default=0,
                        help="특정 레이어만 실행 (1/2/3, 기본 0 = 전체)")
    args = parser.parse_args()

    print(f"\n  Loading data...")
    profiles  = load_jsonl(PROFILES_PATH)
    queries   = load_jsonl(QUERIES_PATH)
    plans     = load_jsonl(PLANS_PATH)

    if not plans:
        print("  ❌ No plans found. Run pipeline first.")
        sys.exit(1)

    profiles_map = {p["user_id"]: p for p in profiles}
    queries_map  = {q["query_id"]: q for q in queries}

    print(f"  profiles={len(profiles)}, queries={len(queries)}, plans={len(plans)}")

    if args.spot >= 0:
        print_spot(plans, queries_map, profiles_map, args.spot)
        return

    run_all = args.layer == 0

    if run_all or args.layer == 1:
        run_layer1(plans)

    if run_all or args.layer == 2:
        run_layer2_stats(queries, profiles_map)

    if run_all or args.layer == 3:
        run_layer3(plans, profiles_map)

    if args.sample > 0:
        print_sample(plans, queries_map, profiles_map,
                     n=args.sample, seed=args.seed)

    print(f"\n  ────────────────────────────────────────────────")
    print(f"  사용법:")
    print(f"    python scripts/verify_data.py              # 전체 자동 검증")
    print(f"    python scripts/verify_data.py --sample 10 # 10개 인스턴스 육안검토")
    print(f"    python scripts/verify_data.py --spot 42   # 42번 인스턴스 상세")
    print(f"    python scripts/verify_data.py --layer 3   # label consistency만")
    print(f"    python scripts/verify_data.py --sample 5 --seed 99  # 다른 샘플")

if __name__ == "__main__":
    main()