"""
Label Verifier v2 — Rule + 3-judge LLM ensemble.
Updated: rule_match uses scenario constraints instead of TPO_ATTR_INCOMPATIBILITIES.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from .utils import (
    call_llm, parse_json_response, save_jsonl, load_jsonl, save_json, log_step,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import (
    LABELS_DIR, PROFILES_DIR, OPTIONS_DIR, QUERIES_DIR,
    LABEL_QUALITY_THRESHOLDS, PROVIDERS,
)
from configs.scenarios import get_scenario_by_id


LABEL_TO_INT = {"tpo_and_preference": 0, "tpo_only": 1,
                "preference_only": 2, "neither": 3}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}
JUDGE_STAGES = ["label_judge_primary", "label_judge_secondary", "label_judge_tertiary"]


def _is_tpo_compatible(attrs, scenario, active_axis, violation_axis):
    """Check if attrs violate TPO based on scenario constraints."""
    # Check violation axis: if the attr value is in scenario's incompatible set
    for axis_name in [active_axis, violation_axis, "garment_category", "color", "pattern"]:
        constraint = scenario.get(axis_name)
        if constraint is None:
            continue
        incompat = set(constraint.get("incompatible", []))
        val = attrs.get(axis_name)
        if val and val in incompat:
            return False
    return True


def rule_match(opt_attrs, user_structured, scenario, active_axis, violation_axis):
    """Rule-based label inference using scenario constraints."""
    # Preference match: check active_axis value against user prefs
    prefs = user_structured.get(active_axis, {})
    likes = set(prefs.get("likes", []))
    dislikes = set(prefs.get("dislikes", []))
    active_val = opt_attrs.get(active_axis)

    if active_val in likes:
        preference_match = True
    elif active_val in dislikes:
        preference_match = False
    else:
        preference_match = False  # neutral = non-preferred

    # TPO match: check if any attr is in scenario's incompatible set
    tpo_ok = _is_tpo_compatible(opt_attrs, scenario, active_axis, violation_axis)

    return preference_match, tpo_ok


def derive_rule_labels(plan, profile, scenario):
    out = {}
    active_axis = plan["active_axis"]
    violation_axis = plan.get("violation_axis", "garment_category")
    for k in "ABCD":
        opt = plan["options"][k]
        pref_m, tpo_m = rule_match(
            opt["attributes"], profile["structured_attributes"],
            scenario, active_axis, violation_axis)
        if tpo_m and pref_m:
            inferred = "tpo_and_preference"
        elif tpo_m:
            inferred = "tpo_only"
        elif pref_m:
            inferred = "preference_only"
        else:
            inferred = "neither"
        out[k] = {
            "inferred_label": inferred, "plan_label": opt["label"],
            "matches_plan": inferred == opt["label"],
            "preference_match": pref_m, "tpo_match": tpo_m,
        }
    return out


JUDGE_SYSTEM = """You are a careful labeler for a fashion personalization benchmark.

Given a user profile, a query with TPO context, and 4 option items described
by attributes, assign each option ONE label:
  - "tpo_and_preference": matches BOTH user taste AND TPO situation
  - "tpo_only":           matches TPO, does not match user taste
  - "preference_only":    matches user taste, violates TPO
  - "neither":            violates both

Each label MUST appear exactly once across A/B/C/D.

Output ONLY a JSON object:
{"A": "...", "B": "...", "C": "...", "D": "...", "rationale": "one sentence"}
"""


def build_judge_prompt(plan, profile, query_text, scenario):
    likes_str = ", ".join(profile["likes_keywords"]) or "(none)"
    dislikes_str = ", ".join(profile["dislikes_keywords"]) or "(none)"
    tpo_parts = []
    tpo = scenario.get("tpo", {})
    for axis, sub in tpo.items():
        if isinstance(sub, dict):
            for k, v in sub.items():
                tpo_parts.append(f"{axis}.{k}={v}")
    tpo_str = "; ".join(tpo_parts) or "(no TPO context)"

    opts_block = []
    for k in "ABCD":
        attrs = plan["options"][k]["attributes"]
        opts_block.append(f"  {k}: " + ", ".join(f"{a}={v}" for a, v in attrs.items()))

    return f"""USER PROFILE
  narrative: {profile['narrative_profile']}
  likes:     {likes_str}
  dislikes:  {dislikes_str}

QUERY: "{query_text}"
TPO:   {tpo_str}
Scenario: {scenario['name']}

OPTIONS:
{chr(10).join(opts_block)}

Assign each of A/B/C/D one of:
tpo_and_preference, tpo_only, preference_only, neither.
Each label must appear exactly once. Output JSON only."""


def call_one_judge(stage, prompt):
    response = call_llm(prompt, stage=stage, system=JUDGE_SYSTEM)
    parsed = parse_json_response(response)
    if parsed is None:
        return {"A": None, "B": None, "C": None, "D": None, "error": "parse_fail"}
    return parsed


def krippendorff_alpha_nominal(matrix):
    n_units, n_raters = matrix.shape
    valid = matrix >= 0
    if not valid.any():
        return 0.0
    cats = np.unique(matrix[valid])
    if len(cats) < 2:
        return 1.0
    n_pairs = n_dis = 0
    for u in range(n_units):
        raters = matrix[u][valid[u]]
        n_r = len(raters)
        if n_r < 2:
            continue
        for i in range(n_r):
            for j in range(i+1, n_r):
                n_pairs += 1
                if raters[i] != raters[j]:
                    n_dis += 1
    if n_pairs == 0:
        return 0.0
    Do = n_dis / n_pairs
    all_r = matrix[valid]
    counts = np.array([np.sum(all_r == c) for c in cats])
    total = counts.sum()
    if total < 2:
        return 0.0
    De = 1 - np.sum((counts / total) ** 2)
    return 1.0 if De == 0 else float(1 - Do / De)


def cohens_kappa(r1, r2):
    assert len(r1) == len(r2)
    n = len(r1)
    if n == 0:
        return 0.0
    labels = sorted(set(r1) | set(r2))
    obs = sum(1 for a, b in zip(r1, r2) if a == b) / n
    p1 = {l: r1.count(l) / n for l in labels}
    p2 = {l: r2.count(l) / n for l in labels}
    exp = sum(p1[l] * p2[l] for l in labels)
    if exp == 1.0:
        return 1.0
    return (obs - exp) / (1 - exp)


def verify_instance(plan, profile, query, enable_judges=None):
    if enable_judges is None:
        enable_judges = list(JUDGE_STAGES)

    scenario = get_scenario_by_id(plan.get("scenario_id", ""))
    if scenario is None:
        # Fallback: construct minimal scenario from query TPO
        scenario = {"tpo": query.get("tpo_scenario", {}),
                    "name": query.get("scenario_name", "unknown"),
                    "garment_category": None, "color": None, "pattern": None}

    rule_results = derive_rule_labels(plan, profile, scenario)
    prompt = build_judge_prompt(plan, profile, query["query_text"], scenario)

    judges = {}
    for stage in enable_judges:
        provider = PROVIDERS.get(stage, {}).get("provider", "?")
        try:
            judges[stage] = call_one_judge(stage, prompt)
            judges[stage]["_provider"] = provider
        except Exception as e:
            judges[stage] = {"error": str(e), "_provider": provider}

    consensus = {}
    th = LABEL_QUALITY_THRESHOLDS["judge_agreement_min"]
    for k in "ABCD":
        votes = []
        for stage, jres in judges.items():
            if "error" in jres:
                continue
            v = jres.get(k)
            if v in LABEL_TO_INT:
                votes.append(v)
        if votes:
            majority, count = Counter(votes).most_common(1)[0]
        else:
            majority, count = None, 0
        plan_label = plan["options"][k]["label"]
        rule_label = rule_results[k]["inferred_label"]
        final = majority if count >= th else plan_label
        all_agree = (count >= th and majority == plan_label
                     and rule_label == plan_label)
        consensus[k] = {
            "plan_label": plan_label, "rule_label": rule_label,
            "judge_votes": votes, "majority": majority,
            "majority_count": count, "final_label": final,
            "all_agree": all_agree,
        }

    return {
        "query_id": plan["query_id"], "user_id": plan["user_id"],
        "scenario_id": plan.get("scenario_id"),
        "active_axis": plan["active_axis"],
        "rule_results": rule_results, "judges": judges,
        "consensus": consensus,
        "all_options_agree": all(c["all_agree"] for c in consensus.values()),
    }


def run_pipeline(plan_path, profile_path, query_path, output_path,
                 limit=0, judge_stages=None):
    log_step(f"Label Verifier v2 (judges: {judge_stages or JUDGE_STAGES})")
    plans = load_jsonl(plan_path)
    profiles = {p["user_id"]: p for p in load_jsonl(profile_path)}
    queries = {q["query_id"]: q for q in load_jsonl(query_path)}
    print(f"  {len(plans)} plans, {len(profiles)} profiles, {len(queries)} queries")

    if output_path.exists():
        existing = load_jsonl(output_path)
        done = {r["query_id"] for r in existing}
        results = existing
        todo = [p for p in plans if p["query_id"] not in done]
        print(f"  Resuming: {len(done)} already verified")
    else:
        results = []
        todo = plans
    if limit > 0:
        todo = todo[:limit]

    for i, plan in enumerate(todo):
        if plan["user_id"] not in profiles or plan["query_id"] not in queries:
            continue
        prof = profiles[plan["user_id"]]
        q = queries[plan["query_id"]]
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(todo)}]")
        try:
            res = verify_instance(plan, prof, q, judge_stages)
            results.append(res)
            if (i + 1) % 50 == 0:
                save_jsonl(results, output_path)
        except Exception as e:
            print(f"    ERROR: {e}")

    save_jsonl(results, output_path)
    _compute_reliability(results, output_path.parent, judge_stages or JUDGE_STAGES)


def _compute_reliability(results, out_dir, judge_stages):
    if not results:
        return
    seqs = {j: [] for j in judge_stages}
    rule_seq = []
    for r in results:
        for k in "ABCD":
            c = r["consensus"][k]
            rule_seq.append(LABEL_TO_INT.get(c["rule_label"], -1))
            for j in judge_stages:
                jres = r["judges"].get(j, {})
                if "error" in jres:
                    seqs[j].append(-1)
                else:
                    seqs[j].append(LABEL_TO_INT.get(jres.get(k), -1))

    matrix = np.array([seqs[j] for j in judge_stages]).T
    alpha = krippendorff_alpha_nominal(matrix)
    judge_maj = []
    for i in range(len(rule_seq)):
        votes = [seqs[j][i] for j in judge_stages if seqs[j][i] >= 0]
        if votes:
            judge_maj.append(Counter(votes).most_common(1)[0][0])
        else:
            judge_maj.append(-1)
    valid = [i for i in range(len(rule_seq))
             if rule_seq[i] >= 0 and judge_maj[i] >= 0]
    kappa = cohens_kappa([rule_seq[i] for i in valid],
                         [judge_maj[i] for i in valid]) if valid else 0.0

    metrics = {
        "n_instances": len(results), "n_decisions": len(rule_seq),
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
    print(f"\n  Reliability:")
    print(f"    α (judges):        {alpha:.3f} (≥{metrics['thresholds']['alpha_min']})")
    print(f"    κ (rule vs judge): {kappa:.3f} (≥{metrics['thresholds']['kappa_min']})")
    print(f"    passed: {metrics['passed']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan_path", type=Path,
                        default=OPTIONS_DIR / "option_plans.jsonl")
    parser.add_argument("--profile_path", type=Path,
                        default=PROFILES_DIR / "profiles.jsonl")
    parser.add_argument("--query_path", type=Path,
                        default=QUERIES_DIR / "queries.jsonl")
    parser.add_argument("--output", type=Path,
                        default=LABELS_DIR / "labels.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--judges", nargs="+", default=None)
    args = parser.parse_args()
    run_pipeline(args.plan_path, args.profile_path, args.query_path,
                 args.output, args.limit, args.judges)


if __name__ == "__main__":
    main()
