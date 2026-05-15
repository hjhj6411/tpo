"""
Option Planner
===============
각 (user_profile, query) 쌍에 대해 4 선지의 속성 조합을 계산합니다.

선지 구성:
  A) TPO+선호 모두 충족 (정답)
  B) TPO 충족, 선호 위반
  C) 선호 충족, TPO 위반
  D) 둘 다 부분/전부 위반 (distractor)

핵심 설계:
  - 모든 4 선지는 *같은 메인 카테고리*에 있어야 함 (예: 모두 우비)
    → 단순한 카테고리 매칭으로 풀리지 않게.
  - 차별 속성은 fine-grained (색상, 디테일, 실루엣, 패턴)
  - LLM이 각 선지의 검색 쿼리(slot-fill)와 시각 디테일을 함께 생성

산출:
  data/options/option_plans.jsonl
"""

import argparse
from pathlib import Path

from .utils import (
    call_gpt5_mini, parse_json_response,
    save_jsonl, load_jsonl, log_step,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import (
    OPTIONS_DIR, PROFILES_DIR, QUERIES_DIR,
    OPTION_LABELS,
)


# ─────────────────────────────────────────────
# Option plan 생성 프롬프트
# ─────────────────────────────────────────────

OPTION_PLAN_SYSTEM = """\
You are designing 4-option multiple choice questions for a personalization benchmark.

Each instance has 4 image options:
  A) Matches BOTH the user's intrinsic preference AND the situational TPO context (correct answer)
  B) Matches TPO ONLY (violates user preference)
  C) Matches user preference ONLY (violates TPO requirements)
  D) Matches NEITHER (distractor)

CRITICAL DESIGN RULES:
1. All 4 options must be in the SAME main category (e.g., all rain jackets, all sofas).
   The differentiation must come from FINE-GRAINED visual attributes, not category.
2. Each option specifies:
   - main_category: shared category (e.g., "rain jacket", "sofa")
   - distinguishing_attributes: dict of attributes that differ
   - search_query_en: English search query for image collection
   - rationale: 1-sentence why this option has its label
3. Differentiation must be visually meaningful but text-caption-resistant.
   Prefer attributes like: silhouette shape, detail load, surface finish, texture
   Over: extreme color contrasts that captioners trivially capture
4. Avoid making D too obvious (e.g., don't pick wildly inappropriate items).
   D should be plausible but wrong on both axes.

Output ONLY a JSON object."""


def build_option_plan_prompt(profile: dict, query: dict) -> str:
    """선지 plan 생성 프롬프트."""
    user_id = profile["user_id"]
    domain = profile["domain"]
    narrative = profile["narrative_profile"]
    structured = profile["structured_attributes"]

    query_text = query["query_text"]
    query_type = query["query_type"]
    tpo = query.get("tpo_scenario", {})

    # structured 요약 (LLM이 정확한 매칭을 만들기 위해 필요)
    pref_summary = []
    avoid_summary = []
    pref_summary.append(f"primary_style: {structured.get('primary_style', '')}")
    for cat, vals in structured.items():
        if cat == "primary_style" or not isinstance(vals, dict):
            continue
        if vals.get("prefer"):
            pref_summary.append(f"{cat} prefer: {vals['prefer']}")
        if vals.get("avoid"):
            avoid_summary.append(f"{cat} avoid: {vals['avoid']}")

    pref_text = "\n  ".join(pref_summary)
    avoid_text = "\n  ".join(avoid_summary) if avoid_summary else "(none)"

    # TPO 요약
    tpo_text = "(no TPO; neutral query)"
    if tpo:
        tpo_parts = []
        for axis, sub in tpo.items():
            if isinstance(sub, dict):
                for k, v in sub.items():
                    tpo_parts.append(f"{axis}.{k} = {v}")
        tpo_text = "; ".join(tpo_parts)

    domain_kr = "fashion" if domain == "fashion" else "interior"

    return f"""Design 4 multiple-choice options for the following preference inference question.

DOMAIN: {domain_kr}

USER PROFILE (narrative shown to model):
  {narrative}

USER PROFILE (ground-truth structured, NOT shown to model — for your reference):
  Prefer:
  {pref_text}
  Avoid:
  {avoid_text}

QUERY ({query_type}):
  "{query_text}"

TPO CONTEXT:
  {tpo_text}

TASK:
Design 4 options A/B/C/D following this matrix:
  A: TPO_match=YES, Preference_match=YES   (correct)
  B: TPO_match=YES, Preference_match=NO
  C: TPO_match=NO,  Preference_match=YES
  D: TPO_match=NO,  Preference_match=NO

CONSTRAINTS:
- All 4 options must share the SAME main_category.
- For each option, specify 3-5 distinguishing_attributes (color, material, silhouette, detail, etc.).
- search_query_en should be 3-7 English words usable for catalog/Google image search.
- Make A, B, C, D visually distinguishable but plausibly similar in style.

OUTPUT JSON:
{{
  "main_category": "...",
  "options": {{
    "A": {{
      "label": "tpo_and_preference",
      "distinguishing_attributes": {{...}},
      "search_query_en": "...",
      "rationale": "..."
    }},
    "B": {{...}},
    "C": {{...}},
    "D": {{...}}
  }}
}}"""


# ─────────────────────────────────────────────
# 검증
# ─────────────────────────────────────────────

def validate_plan(plan: dict) -> tuple[bool, str]:
    """plan의 형식과 일관성 검증."""
    if "main_category" not in plan or not plan["main_category"]:
        return False, "missing main_category"

    if "options" not in plan:
        return False, "missing options"

    options = plan["options"]
    if set(options.keys()) != {"A", "B", "C", "D"}:
        return False, f"options keys should be A/B/C/D, got {list(options.keys())}"

    expected_labels = {
        "A": "tpo_and_preference",
        "B": "tpo_only",
        "C": "preference_only",
        "D": "neither",
    }
    for k, expected in expected_labels.items():
        opt = options[k]
        if opt.get("label") != expected:
            return False, f"option {k} label mismatch: {opt.get('label')} vs {expected}"
        if not opt.get("search_query_en"):
            return False, f"option {k} missing search_query_en"
        if not opt.get("distinguishing_attributes"):
            return False, f"option {k} missing distinguishing_attributes"

    return True, ""


# ─────────────────────────────────────────────
# 메인 파이프라인
# ─────────────────────────────────────────────

def plan_options_for_query(profile: dict, query: dict) -> dict:
    """단일 (profile, query)에 대해 option plan 생성."""
    prompt = build_option_plan_prompt(profile, query)

    response = call_gpt5_mini(
        prompt=prompt, system=OPTION_PLAN_SYSTEM,
        max_tokens=2048, temperature=0.5,
    )
    plan = parse_json_response(response)

    if plan is None:
        return None

    ok, msg = validate_plan(plan)
    if not ok:
        # 1회 재시도
        response = call_gpt5_mini(
            prompt=prompt + f"\n\nPREVIOUS ATTEMPT FAILED: {msg}\nPlease produce valid output.",
            system=OPTION_PLAN_SYSTEM,
            max_tokens=2048, temperature=0.3,
        )
        plan = parse_json_response(response)
        if plan is None:
            return None
        ok, msg = validate_plan(plan)
        if not ok:
            return None

    return {
        "query_id": query["query_id"],
        "user_id": query["user_id"],
        "domain": query["domain"],
        "query_type": query["query_type"],
        "main_category": plan["main_category"],
        "options": plan["options"],
    }


def run_pipeline(profile_path: Path, query_path: Path, output_path: Path,
                 force: bool = False, limit: int = 0):
    """전체 option planning 파이프라인 실행."""
    log_step("Option Planner")

    profiles = {p["user_id"]: p for p in load_jsonl(profile_path)}
    queries = load_jsonl(query_path)
    print(f"  Loaded {len(profiles)} profiles, {len(queries)} queries")

    if output_path.exists() and not force:
        existing = load_jsonl(output_path)
        done_qids = {p["query_id"] for p in existing}
        print(f"  Resuming: {len(done_qids)} queries already planned")
        plans = existing
        queries_to_do = [q for q in queries if q["query_id"] not in done_qids]
    else:
        plans = []
        queries_to_do = queries

    if limit > 0:
        queries_to_do = queries_to_do[:limit]

    n_failed = 0
    for i, query in enumerate(queries_to_do):
        if query["user_id"] not in profiles:
            print(f"  [{i+1}] Skip {query['query_id']}: profile not found")
            continue

        profile = profiles[query["user_id"]]
        print(f"\n  [{i+1}/{len(queries_to_do)}] {query['query_id']} ({query['query_type']})")
        print(f"    Q: {query['query_text']}")

        try:
            plan = plan_options_for_query(profile, query)
            if plan is None:
                print(f"    Failed validation")
                n_failed += 1
                continue
            plans.append(plan)
            print(f"    Category: {plan['main_category']}")
            for opt_key in ["A", "B", "C", "D"]:
                opt = plan["options"][opt_key]
                print(f"      {opt_key} ({opt['label']}): {opt['search_query_en']}")

            # 매 10개마다 저장
            if (i + 1) % 10 == 0:
                save_jsonl(plans, output_path)

        except Exception as e:
            print(f"    ERROR: {e}")
            n_failed += 1

    save_jsonl(plans, output_path)
    print(f"\n  ✓ Saved {len(plans)} plans to {output_path}")
    print(f"  Failed: {n_failed}")
    return plans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_path", type=Path,
                        default=PROFILES_DIR / "profiles.jsonl")
    parser.add_argument("--query_path", type=Path,
                        default=QUERIES_DIR / "queries.jsonl")
    parser.add_argument("--output", type=Path,
                        default=OPTIONS_DIR / "option_plans.jsonl")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of queries (0 = all)")
    args = parser.parse_args()

    run_pipeline(args.profile_path, args.query_path, args.output,
                 args.force, args.limit)


if __name__ == "__main__":
    main()
