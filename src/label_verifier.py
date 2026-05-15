"""
Label Verifier
===============
각 인스턴스의 4 선지 라벨을 검증합니다.

검증 방식 (2-tier):
  1. 규칙 기반: option plan의 distinguishing_attributes를
     user의 structured_attributes와 매칭해 자동 라벨링
  2. LLM judge ensemble:
     - GPT-5-mini (유료, 1개)
     - 로컬 Qwen2.5-7B (무료)
     - 로컬 Llama-3.1-8B (무료)
     3개 judge 중 최소 2개 합의 시 통과

신뢰도 메트릭:
  - Krippendorff's α (3 judges 간)
  - Cohen's κ (규칙 vs LLM 합의)

사용:
  python -m src.label_verifier

산출:
  data/labels/labels.jsonl
"""

import argparse
from pathlib import Path

import numpy as np

from .utils import (
    call_gpt5_mini, call_local_llm, parse_json_response,
    save_jsonl, load_jsonl, log_step,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import (
    LABELS_DIR, PROFILES_DIR, OPTIONS_DIR, IMAGES_DIR,
    FREE_MODELS, LABEL_QUALITY_THRESHOLDS, OPTION_LABELS,
)


# ─────────────────────────────────────────────
# 규칙 기반 라벨러
# ─────────────────────────────────────────────

def rule_based_label(option_attrs: dict, user_structured: dict,
                      tpo_scenario: dict) -> tuple[bool, bool]:
    """option_attrs가 (user 선호 매칭, TPO 매칭)인지 판단.

    Returns: (preference_match, tpo_match) — 두 boolean
    """
    # 1) 선호 매칭: option attrs가 user의 prefer와 얼마나 겹치는가
    pref_score = 0
    pref_violations = 0

    flat_opt = _flatten_attrs(option_attrs)
    flat_opt_lower = {str(v).lower() for v in flat_opt}

    for cat, vals in user_structured.items():
        if not isinstance(vals, dict):
            continue
        prefers = vals.get("prefer", [])
        avoids = vals.get("avoid", [])

        for p in prefers:
            if str(p).lower() in flat_opt_lower:
                pref_score += 1

        for a in avoids:
            if str(a).lower() in flat_opt_lower:
                pref_violations += 1

    preference_match = pref_score > pref_violations

    # 2) TPO 매칭: option attrs가 TPO 요구사항과 호환되는가
    if not tpo_scenario:
        tpo_match = True  # neutral query면 TPO 충족으로 간주
    else:
        # TPO 요구사항을 attrs로 변환하는 휴리스틱
        tpo_match = _check_tpo_compatibility(flat_opt, tpo_scenario)

    return preference_match, tpo_match


def _flatten_attrs(attrs: dict) -> list:
    """nested dict의 모든 leaf value를 수집."""
    result = []
    if isinstance(attrs, dict):
        for v in attrs.values():
            result.extend(_flatten_attrs(v))
    elif isinstance(attrs, list):
        for item in attrs:
            result.extend(_flatten_attrs(item))
    else:
        result.append(attrs)
    return result


def _check_tpo_compatibility(opt_attrs: list, tpo: dict) -> bool:
    """option attrs가 TPO 요구사항과 양립 가능한지 휴리스틱 판단.

    이는 보수적 매칭: TPO가 명시한 키워드와 정면 충돌하지 않으면 OK.
    완벽한 매칭은 LLM judge에 위임.
    """
    opt_text = " ".join(str(v).lower() for v in opt_attrs)

    # TPO에서 명시적 요구사항 추출
    tpo_flat = _flatten_attrs(tpo)
    tpo_keywords = [str(v).lower() for v in tpo_flat]

    # 양립 불가 keyword 쌍
    incompatible_pairs = [
        ({"rainy", "snowy", "winter"}, {"summer_cooling", "sleeveless", "shorts"}),
        ({"formal", "semi_formal", "wedding"}, {"very_casual", "sporty", "athleisure"}),
        ({"summer", "hot_humid", "beach"}, {"heavy_coat", "winter_wear", "wool"}),
        ({"exercise", "gym", "athletic"}, {"formal", "leather_dress_shoes", "silk"}),
    ]

    tpo_set = set(tpo_keywords)
    for tpo_keys, opt_violates in incompatible_pairs:
        if tpo_set & tpo_keys:
            for violation in opt_violates:
                if violation in opt_text:
                    return False
    return True


def derive_rule_labels(plan: dict, profile: dict) -> dict:
    """plan의 4 선지에 대해 규칙 기반 라벨 산출."""
    rule_results = {}
    for opt_key in ["A", "B", "C", "D"]:
        opt = plan["options"][opt_key]
        pref_match, tpo_match = rule_based_label(
            opt["distinguishing_attributes"],
            profile["structured_attributes"],
            plan.get("tpo_scenario", {}),
        )

        # (tpo, pref) → label
        if tpo_match and pref_match:
            inferred = "tpo_and_preference"
        elif tpo_match and not pref_match:
            inferred = "tpo_only"
        elif not tpo_match and pref_match:
            inferred = "preference_only"
        else:
            inferred = "neither"

        rule_results[opt_key] = {
            "inferred_label": inferred,
            "expected_label": opt["label"],
            "matches_plan": inferred == opt["label"],
            "preference_match": pref_match,
            "tpo_match": tpo_match,
        }

    return rule_results


# ─────────────────────────────────────────────
# LLM Judge
# ─────────────────────────────────────────────

JUDGE_SYSTEM = """\
You are a careful labeler for a personalization benchmark.

Given:
  - User's narrative profile (their enduring taste)
  - TPO context (Time/Place/Occasion situation)
  - 4 option items with their attributes

Your task: assign each of A/B/C/D one of these labels:
  - "tpo_and_preference": matches BOTH user taste AND TPO situation
  - "tpo_only":           matches TPO but violates user taste
  - "preference_only":    matches user taste but violates TPO
  - "neither":            violates both

CRITICAL: Each label should appear EXACTLY ONCE across the 4 options.
If you cannot assign unique labels, return labels as best you can but flag with "ambiguous": true.

Output ONLY a JSON object."""


def build_judge_prompt(plan: dict, profile: dict) -> str:
    """LLM judge에게 보낼 프롬프트 (이미지 없이 텍스트만; 라벨 검증용)."""

    options_text = []
    for opt_key in ["A", "B", "C", "D"]:
        opt = plan["options"][opt_key]
        attrs_str = ", ".join(
            f"{k}={v}" for k, v in opt["distinguishing_attributes"].items()
        )
        options_text.append(f"  {opt_key}: {attrs_str}")
    options_block = "\n".join(options_text)

    tpo = plan.get("tpo_scenario") or {}
    tpo_str = "(none, neutral query)"
    if tpo:
        tpo_parts = []
        for axis, sub in tpo.items():
            if isinstance(sub, dict):
                for k, v in sub.items():
                    tpo_parts.append(f"{axis}.{k}={v}")
        tpo_str = "; ".join(tpo_parts)

    return f"""USER NARRATIVE PROFILE:
  {profile['narrative_profile']}

QUERY: "{plan.get('query_text', '')}"
TPO CONTEXT: {tpo_str}

MAIN CATEGORY: {plan['main_category']}

OPTIONS (described by attributes only — no images):
{options_block}

Assign each option a label from {{"tpo_and_preference", "tpo_only", "preference_only", "neither"}}.
Each label should appear exactly once.

Output JSON:
{{
  "A": "...",
  "B": "...",
  "C": "...",
  "D": "...",
  "rationale": "1-sentence reasoning",
  "ambiguous": false
}}"""


def llm_judge(plan: dict, profile: dict, judge_name: str) -> dict:
    """단일 LLM judge 호출."""
    prompt = build_judge_prompt(plan, profile)

    if judge_name == "gpt5_mini":
        response = call_gpt5_mini(prompt, system=JUDGE_SYSTEM,
                                   max_tokens=512, temperature=0.1)
    elif judge_name == "local_qwen":
        cfg = FREE_MODELS["local_judge"]
        response = call_local_llm(prompt, system=JUDGE_SYSTEM,
                                   api_base=cfg["api_base"],
                                   model=cfg["model_name"],
                                   max_tokens=512, temperature=0.1)
    elif judge_name == "local_llama":
        cfg = FREE_MODELS["local_judge_alt"]
        response = call_local_llm(prompt, system=JUDGE_SYSTEM,
                                   api_base=cfg["api_base"],
                                   model=cfg["model_name"],
                                   max_tokens=512, temperature=0.1)
    else:
        raise ValueError(f"Unknown judge: {judge_name}")

    parsed = parse_json_response(response)
    if parsed is None:
        return {"A": None, "B": None, "C": None, "D": None,
                "error": "parse_failed"}
    return parsed


# ─────────────────────────────────────────────
# Krippendorff's α (간단 구현, 명목 데이터용)
# ─────────────────────────────────────────────

def krippendorff_alpha_nominal(ratings_matrix: np.ndarray) -> float:
    """ratings_matrix: (n_units, n_raters) of integer category labels (-1 = missing).

    Krippendorff's α for nominal data.
    """
    n_units, n_raters = ratings_matrix.shape

    # 카테고리 추출
    valid = ratings_matrix >= 0
    if not valid.any():
        return 0.0

    categories = np.unique(ratings_matrix[valid])
    if len(categories) < 2:
        return 1.0

    # observed disagreement
    n_pairs_observed = 0
    n_disagree_observed = 0
    for u in range(n_units):
        raters = ratings_matrix[u][valid[u]]
        n_r = len(raters)
        if n_r < 2:
            continue
        for i in range(n_r):
            for j in range(i + 1, n_r):
                n_pairs_observed += 1
                if raters[i] != raters[j]:
                    n_disagree_observed += 1

    if n_pairs_observed == 0:
        return 0.0
    Do = n_disagree_observed / n_pairs_observed

    # expected disagreement (marginal frequencies)
    all_ratings = ratings_matrix[valid]
    counts = np.array([np.sum(all_ratings == c) for c in categories])
    total = counts.sum()
    if total < 2:
        return 0.0

    De = 1 - np.sum((counts / total) ** 2)

    if De == 0:
        return 1.0
    alpha = 1 - Do / De
    return float(alpha)


# ─────────────────────────────────────────────
# Cohen's κ
# ─────────────────────────────────────────────

def cohens_kappa(rater1: list, rater2: list) -> float:
    """두 평가자 간 Cohen's κ."""
    assert len(rater1) == len(rater2)
    if len(rater1) == 0:
        return 0.0

    labels = sorted(set(rater1) | set(rater2))
    n = len(rater1)
    obs_agree = sum(1 for a, b in zip(rater1, rater2) if a == b) / n

    p1 = {l: rater1.count(l) / n for l in labels}
    p2 = {l: rater2.count(l) / n for l in labels}
    exp_agree = sum(p1[l] * p2[l] for l in labels)

    if exp_agree == 1.0:
        return 1.0
    return (obs_agree - exp_agree) / (1 - exp_agree)


# ─────────────────────────────────────────────
# 메인 파이프라인
# ─────────────────────────────────────────────

def verify_instance(plan: dict, profile: dict,
                     enable_local_judges: bool = True,
                     enable_paid_judge: bool = True) -> dict:
    """단일 인스턴스의 라벨 검증."""

    # 1) Rule-based
    rule_results = derive_rule_labels(plan, profile)

    # 2) LLM judges
    judges = {}
    if enable_paid_judge:
        try:
            judges["gpt5_mini"] = llm_judge(plan, profile, "gpt5_mini")
        except Exception as e:
            judges["gpt5_mini"] = {"error": str(e)}

    if enable_local_judges:
        try:
            judges["local_qwen"] = llm_judge(plan, profile, "local_qwen")
        except Exception as e:
            judges["local_qwen"] = {"error": str(e)}

        try:
            judges["local_llama"] = llm_judge(plan, profile, "local_llama")
        except Exception as e:
            judges["local_llama"] = {"error": str(e)}

    # 3) Consensus 분석
    consensus = {}
    plan_labels = {k: plan["options"][k]["label"] for k in ["A", "B", "C", "D"]}

    for opt_key in ["A", "B", "C", "D"]:
        judge_votes = []
        for j_name, j_result in judges.items():
            if "error" not in j_result:
                v = j_result.get(opt_key)
                if v:
                    judge_votes.append(v)

        rule_label = rule_results[opt_key]["inferred_label"]
        plan_label = plan_labels[opt_key]

        # 다수결
        from collections import Counter
        vote_counts = Counter(judge_votes)
        if vote_counts:
            majority_label, majority_count = vote_counts.most_common(1)[0]
        else:
            majority_label, majority_count = None, 0

        # 최종 라벨: judge 합의 >= 2면 majority, 아니면 plan label 채택
        threshold = LABEL_QUALITY_THRESHOLDS["judge_agreement_min"]
        if majority_count >= threshold:
            final_label = majority_label
        else:
            final_label = plan_label

        consensus[opt_key] = {
            "plan_label": plan_label,
            "rule_label": rule_label,
            "judge_votes": judge_votes,
            "majority": majority_label,
            "majority_count": majority_count,
            "final_label": final_label,
            "all_agree": (
                rule_label == plan_label == majority_label
                and majority_count == len(judge_votes)
                and len(judge_votes) >= threshold
            ),
        }

    return {
        "query_id": plan["query_id"],
        "user_id": plan["user_id"],
        "rule_results": rule_results,
        "judges": judges,
        "consensus": consensus,
        "all_options_agree": all(c["all_agree"] for c in consensus.values()),
    }


def run_pipeline(plan_path: Path, profile_path: Path, query_path: Path,
                 output_path: Path, limit: int = 0,
                 enable_local_judges: bool = True,
                 enable_paid_judge: bool = True):
    """전체 라벨 검증 파이프라인 실행."""
    log_step("Label Verifier")

    plans = load_jsonl(plan_path)
    profiles = {p["user_id"]: p for p in load_jsonl(profile_path)}

    # query text를 plan에 보강
    queries = {q["query_id"]: q for q in load_jsonl(query_path)}
    for p in plans:
        if p["query_id"] in queries:
            q = queries[p["query_id"]]
            p["query_text"] = q["query_text"]
            p["tpo_scenario"] = q.get("tpo_scenario", {})

    print(f"  Loaded {len(plans)} plans, {len(profiles)} profiles")

    if output_path.exists():
        existing = load_jsonl(output_path)
        done = {r["query_id"] for r in existing}
        results = existing
        plans_to_do = [p for p in plans if p["query_id"] not in done]
        print(f"  Resuming: {len(done)} already verified")
    else:
        results = []
        plans_to_do = plans

    if limit > 0:
        plans_to_do = plans_to_do[:limit]

    for i, plan in enumerate(plans_to_do):
        if plan["user_id"] not in profiles:
            continue

        profile = profiles[plan["user_id"]]
        print(f"\n  [{i+1}/{len(plans_to_do)}] {plan['query_id']}")
        try:
            result = verify_instance(plan, profile,
                                      enable_local_judges,
                                      enable_paid_judge)
            results.append(result)

            n_agree = sum(1 for c in result["consensus"].values() if c["all_agree"])
            print(f"    Options with full agreement: {n_agree}/4")

            if (i + 1) % 10 == 0:
                save_jsonl(results, output_path)
        except Exception as e:
            print(f"    ERROR: {e}")

    save_jsonl(results, output_path)

    # 신뢰도 메트릭 계산
    _compute_reliability_metrics(results, output_path.parent)

    print(f"\n  ✓ Saved {len(results)} labels to {output_path}")


def _compute_reliability_metrics(results: list[dict], out_dir: Path):
    """Krippendorff α, Cohen κ 계산 및 저장."""
    label_to_int = {
        "tpo_and_preference": 0, "tpo_only": 1,
        "preference_only": 2, "neither": 3,
    }

    # judge 별 라벨 시퀀스 수집
    judges_list = ["gpt5_mini", "local_qwen", "local_llama"]
    sequences = {j: [] for j in judges_list}
    rule_seq = []
    plan_seq = []

    for r in results:
        for opt_key in ["A", "B", "C", "D"]:
            c = r["consensus"][opt_key]
            rule_seq.append(label_to_int.get(c["rule_label"], -1))
            plan_seq.append(label_to_int.get(c["plan_label"], -1))
            for j in judges_list:
                v = r["judges"].get(j, {})
                if "error" in v:
                    sequences[j].append(-1)
                else:
                    label = v.get(opt_key)
                    sequences[j].append(label_to_int.get(label, -1))

    # Krippendorff α among 3 judges
    matrix = np.array([sequences[j] for j in judges_list]).T  # (n_units, 3)
    alpha = krippendorff_alpha_nominal(matrix)

    # Cohen κ between rule and judge majority
    judge_majority = []
    for i in range(len(rule_seq)):
        from collections import Counter
        votes = [sequences[j][i] for j in judges_list if sequences[j][i] >= 0]
        if votes:
            judge_majority.append(Counter(votes).most_common(1)[0][0])
        else:
            judge_majority.append(-1)

    valid_idx = [i for i in range(len(rule_seq))
                 if rule_seq[i] >= 0 and judge_majority[i] >= 0]
    rule_valid = [rule_seq[i] for i in valid_idx]
    judge_valid = [judge_majority[i] for i in valid_idx]

    if rule_valid:
        kappa = cohens_kappa(rule_valid, judge_valid)
    else:
        kappa = 0.0

    metrics = {
        "n_instances": len(results),
        "n_option_decisions": len(rule_seq),
        "krippendorff_alpha_judges": alpha,
        "cohen_kappa_rule_vs_judge": kappa,
        "thresholds": {
            "alpha_min": LABEL_QUALITY_THRESHOLDS["krippendorff_alpha_min"],
            "kappa_min": LABEL_QUALITY_THRESHOLDS["cohen_kappa_min"],
        },
        "passed": (
            alpha >= LABEL_QUALITY_THRESHOLDS["krippendorff_alpha_min"]
            and kappa >= LABEL_QUALITY_THRESHOLDS["cohen_kappa_min"]
        ),
    }

    save_json(metrics, out_dir / "reliability_metrics.json")
    print(f"\n  Reliability metrics:")
    print(f"    Krippendorff α (judges): {alpha:.3f} (threshold {metrics['thresholds']['alpha_min']})")
    print(f"    Cohen κ (rule vs judge): {kappa:.3f} (threshold {metrics['thresholds']['kappa_min']})")
    print(f"    Overall passed: {metrics['passed']}")


def save_json(obj, path: Path):
    """Local helper."""
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan_path", type=Path,
                        default=OPTIONS_DIR / "option_plans.jsonl")
    parser.add_argument("--profile_path", type=Path,
                        default=PROFILES_DIR / "profiles.jsonl")
    parser.add_argument("--query_path", type=Path,
                        default=Path("data/queries/queries.jsonl"))
    parser.add_argument("--output", type=Path,
                        default=LABELS_DIR / "labels.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no_local", action="store_true",
                        help="Skip local judges (require only GPT-5-mini)")
    parser.add_argument("--no_paid", action="store_true",
                        help="Skip GPT-5-mini (free only — for testing)")
    args = parser.parse_args()

    run_pipeline(args.plan_path, args.profile_path, args.query_path,
                 args.output, args.limit,
                 enable_local_judges=not args.no_local,
                 enable_paid_judge=not args.no_paid)


if __name__ == "__main__":
    main()
