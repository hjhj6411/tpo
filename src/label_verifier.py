"""
Label Verifier (English)
=========================

For each instance, verify the 4-option labels via two channels:

  1. Rule-based: match option's closed-set attributes against the user's
     structured_attributes and against the TPO scenario requirements.
  2. LLM judge ensemble (3 judges):
       - GPT-5-mini (paid)
       - Qwen2.5-7B (free, local)
       - Llama-3.1-8B (free, local)

Reliability metrics:
  - Krippendorff's α across the 3 judges
  - Cohen's κ between rule-based and judge majority
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .utils import (
    call_gpt5_mini, call_local_llm, parse_json_response,
    save_jsonl, load_jsonl, save_json, log_step,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import (
    LABELS_DIR, PROFILES_DIR, OPTIONS_DIR, QUERIES_DIR,
    FREE_MODELS, GPT5_MINI, LABEL_QUALITY_THRESHOLDS,
)


LABEL_TO_INT = {
    "tpo_and_preference": 0,
    "tpo_only": 1,
    "preference_only": 2,
    "neither": 3,
}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


# ─────────────────────────────────────────────
# Rule-based labeler (closed-set vocabulary)
# ─────────────────────────────────────────────

def rule_match(option_attrs: dict, user_structured: dict,
                tpo_scenario: dict) -> tuple[bool, bool, dict]:
    """Decide preference_match and tpo_match for one option.

    Returns:
      (preference_match, tpo_match, detail_dict)
    """
    # 1) Preference matching
    pref_hits = 0
    pref_violations = 0
    pref_detail = []

    for axis, prefs in user_structured.items():
        opt_val = option_attrs.get(axis)
        if opt_val is None:
            continue
        if opt_val in prefs.get("likes", []):
            pref_hits += 1
            pref_detail.append(f"+{axis}:{opt_val}")
        if opt_val in prefs.get("dislikes", []):
            pref_violations += 1
            pref_detail.append(f"-{axis}:{opt_val}")

    preference_match = pref_hits > pref_violations

    # 2) TPO matching
    tpo_compatible = _check_tpo(option_attrs, tpo_scenario or {})

    return preference_match, tpo_compatible, {
        "pref_hits": pref_hits,
        "pref_violations": pref_violations,
        "pref_detail": pref_detail,
        "tpo_compatible": tpo_compatible,
    }


# Curated incompatibility rules for TPO ⇄ attribute conflicts.
# These are conservative — they catch clear violations but don't enforce
# perfect matching (LLM judges fill that gap).
TPO_ATTR_INCOMPATIBILITIES = [
    # (tpo_keyword_set, axis, forbidden_values)
    ({"rainy", "snowy", "winter", "cold"}, "garment_category", ["shorts", "t_shirt"]),
    ({"rainy", "snowy"}, "material", ["silk", "linen", "suede"]),
    ({"summer", "hot_humid", "beach"}, "garment_category",
        ["trench_coat", "coat"]),
    ({"summer", "hot_humid"}, "material", ["wool", "cashmere", "fleece"]),
    ({"summer", "hot_humid"}, "sleeve_length", ["long_sleeve"]),
    ({"formal", "wedding_venue", "ceremony_attendance", "business_casual"},
        "garment_category", ["t_shirt", "shorts", "hoodie"]),
    ({"formal", "wedding_venue"}, "pattern", ["graphic_print", "camouflage"]),
    ({"exercise", "gym"}, "garment_category",
        ["suit", "blazer", "coat", "trench_coat"]),
    ({"exercise", "gym"}, "material", ["silk", "wool", "leather"]),
    ({"office", "work_meeting", "business_casual"}, "garment_category",
        ["shorts", "hoodie"]),
    ({"office", "work_meeting"}, "pattern", ["graphic_print", "camouflage"]),
    ({"beach", "park"}, "garment_category", ["suit", "blazer", "trench_coat"]),
]


def _flatten_tpo(tpo: dict) -> set:
    flat = set()
    for axis, sub in tpo.items():
        if isinstance(sub, dict):
            for v in sub.values():
                flat.add(str(v).lower())
    return flat


def _check_tpo(option_attrs: dict, tpo: dict) -> bool:
    """Return True if option attributes are *compatible* with TPO.

    Compatibility = no curated incompatibility rule is violated.
    For neutral queries (empty tpo), always returns True.
    """
    if not tpo:
        return True
    tpo_flat = _flatten_tpo(tpo)
    for tpo_keys, axis, forbidden in TPO_ATTR_INCOMPATIBILITIES:
        if tpo_flat & tpo_keys:
            val = option_attrs.get(axis)
            if val and val in forbidden:
                return False
    return True


def derive_rule_labels(plan: dict, profile: dict,
                        tpo_scenario: dict) -> dict:
    """Run rule labeler on all 4 options."""
    rule_out = {}
    for k in "ABCD":
        opt = plan["options"][k]
        pref_m, tpo_m, detail = rule_match(
            opt["attributes"], profile["structured_attributes"], tpo_scenario,
        )
        if tpo_m and pref_m:
            inferred = "tpo_and_preference"
        elif tpo_m and not pref_m:
            inferred = "tpo_only"
        elif (not tpo_m) and pref_m:
            inferred = "preference_only"
        else:
            inferred = "neither"
        rule_out[k] = {
            "inferred_label": inferred,
            "plan_label": opt["label"],
            "matches_plan": inferred == opt["label"],
            "preference_match": pref_m,
            "tpo_match": tpo_m,
            "detail": detail,
        }
    return rule_out


# ─────────────────────────────────────────────
# LLM judge (English)
# ─────────────────────────────────────────────

JUDGE_SYSTEM = """\
You are a careful labeler for a fashion personalization benchmark.

Given a user's narrative + keyword profile, a query and its TPO context,
and 4 option items described by attributes, assign each option ONE label:
  - "tpo_and_preference": matches BOTH the user's taste AND the TPO situation
  - "tpo_only":           matches TPO situation, violates the user's taste
  - "preference_only":    matches the user's taste, violates the TPO situation
  - "neither":            violates both

Each label MUST appear exactly once across the 4 options.

Output ONLY a JSON object with the form:
{
  "A": "...", "B": "...", "C": "...", "D": "...",
  "rationale": "one sentence",
  "ambiguous": false
}
"""


def build_judge_prompt(plan: dict, profile: dict, query_text: str,
                        tpo_scenario: dict) -> str:
    likes_str = ", ".join(profile["likes_keywords"]) or "(none)"
    dislikes_str = ", ".join(profile["dislikes_keywords"]) or "(none)"

    tpo_str = "(none, neutral query)"
    if tpo_scenario:
        parts = []
        for axis, sub in tpo_scenario.items():
            if isinstance(sub, dict):
                for k, v in sub.items():
                    parts.append(f"{axis}.{k}={v}")
        tpo_str = "; ".join(parts)

    opts_block = []
    for k in "ABCD":
        attrs = plan["options"][k]["attributes"]
        opts_block.append(f"  {k}: " + ", ".join(f"{a}={v}" for a, v in attrs.items()))
    opts_text = "\n".join(opts_block)

    return f"""USER PROFILE
  narrative: {profile['narrative_profile']}
  likes:     {likes_str}
  dislikes:  {dislikes_str}

QUERY: "{query_text}"
TPO:   {tpo_str}
MAIN CATEGORY: {plan['main_category']}

OPTIONS (attribute descriptions only — no images):
{opts_text}

Assign each of A/B/C/D one of: tpo_and_preference, tpo_only, preference_only, neither.
Each label must appear exactly once.

Output JSON only.
"""


def call_judge(judge_name: str, prompt: str) -> dict:
    if judge_name == "gpt5_mini":
        response = call_gpt5_mini(
            prompt, system=JUDGE_SYSTEM,
            max_completion_tokens=GPT5_MINI["max_completion_tokens_default"],
        )
    elif judge_name == "local_qwen":
        cfg = FREE_MODELS["local_judge"]
        response = call_local_llm(
            prompt, system=JUDGE_SYSTEM, api_base=cfg["api_base"],
            model=cfg["model_name"], max_tokens=512, temperature=0.1,
        )
    elif judge_name == "local_llama":
        cfg = FREE_MODELS["local_judge_alt"]
        response = call_local_llm(
            prompt, system=JUDGE_SYSTEM, api_base=cfg["api_base"],
            model=cfg["model_name"], max_tokens=512, temperature=0.1,
        )
    else:
        raise ValueError(f"Unknown judge {judge_name}")

    parsed = parse_json_response(response)
    if parsed is None:
        return {"A": None, "B": None, "C": None, "D": None, "error": "parse_fail"}
    return parsed


# ─────────────────────────────────────────────
# Reliability metrics
# ─────────────────────────────────────────────

def krippendorff_alpha_nominal(matrix: np.ndarray) -> float:
    """Krippendorff's α for nominal data. matrix: (n_units, n_raters); -1 missing."""
    n_units, n_raters = matrix.shape
    valid = matrix >= 0
    if not valid.any():
        return 0.0
    cats = np.unique(matrix[valid])
    if len(cats) < 2:
        return 1.0

    n_pairs_obs = 0
    n_dis_obs = 0
    for u in range(n_units):
        raters = matrix[u][valid[u]]
        n_r = len(raters)
        if n_r < 2:
            continue
        for i in range(n_r):
            for j in range(i+1, n_r):
                n_pairs_obs += 1
                if raters[i] != raters[j]:
                    n_dis_obs += 1
    if n_pairs_obs == 0:
        return 0.0
    Do = n_dis_obs / n_pairs_obs
    all_r = matrix[valid]
    counts = np.array([np.sum(all_r == c) for c in cats])
    total = counts.sum()
    if total < 2:
        return 0.0
    De = 1 - np.sum((counts/total) ** 2)
    return 1.0 if De == 0 else float(1 - Do/De)


def cohens_kappa(r1: list, r2: list) -> float:
    assert len(r1) == len(r2)
    n = len(r1)
    if n == 0:
        return 0.0
    labels = sorted(set(r1) | set(r2))
    obs_agree = sum(1 for a, b in zip(r1, r2) if a == b) / n
    p1 = {l: r1.count(l)/n for l in labels}
    p2 = {l: r2.count(l)/n for l in labels}
    exp_agree = sum(p1[l]*p2[l] for l in labels)
    if exp_agree == 1.0:
        return 1.0
    return (obs_agree - exp_agree) / (1 - exp_agree)


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def verify_instance(plan: dict, profile: dict, query: dict,
                     enable_paid: bool = True,
                     enable_local: bool = True) -> dict:
    tpo = query.get("tpo_scenario") or {}
    rule_results = derive_rule_labels(plan, profile, tpo)

    prompt = build_judge_prompt(plan, profile, query["query_text"], tpo)
    judges = {}

    if enable_paid:
        try:
            judges["gpt5_mini"] = call_judge("gpt5_mini", prompt)
        except Exception as e:
            judges["gpt5_mini"] = {"error": str(e)}

    if enable_local:
        for jname in ["local_qwen", "local_llama"]:
            try:
                judges[jname] = call_judge(jname, prompt)
            except Exception as e:
                judges[jname] = {"error": str(e)}

    # Consensus per option
    consensus = {}
    threshold = LABEL_QUALITY_THRESHOLDS["judge_agreement_min"]
    for k in "ABCD":
        votes = []
        for j_name, j_result in judges.items():
            if "error" in j_result:
                continue
            v = j_result.get(k)
            if v in LABEL_TO_INT:
                votes.append(v)

        if votes:
            majority, count = Counter(votes).most_common(1)[0]
        else:
            majority, count = None, 0

        plan_label = plan["options"][k]["label"]
        rule_label = rule_results[k]["inferred_label"]

        # Final label: judge majority if reached threshold, else plan_label.
        if count >= threshold:
            final = majority
        else:
            final = plan_label

        all_agree = (
            count >= threshold
            and majority == plan_label
            and rule_label == plan_label
        )

        consensus[k] = {
            "plan_label": plan_label,
            "rule_label": rule_label,
            "judge_votes": votes,
            "majority": majority,
            "majority_count": count,
            "final_label": final,
            "all_agree": all_agree,
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
                 enable_paid: bool = True, enable_local: bool = True):
    log_step("Label Verifier")

    plans = load_jsonl(plan_path)
    profiles = {p["user_id"]: p for p in load_jsonl(profile_path)}
    queries = {q["query_id"]: q for q in load_jsonl(query_path)}

    print(f"  Loaded {len(plans)} plans, {len(profiles)} profiles, {len(queries)} queries")

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
        print(f"\n  [{i+1}/{len(todo)}] {plan['query_id']}")
        try:
            res = verify_instance(plan, prof, q, enable_paid, enable_local)
            results.append(res)
            n_agree = sum(1 for c in res["consensus"].values() if c["all_agree"])
            print(f"    options with full agreement: {n_agree}/4")
            if (i + 1) % 10 == 0:
                save_jsonl(results, output_path)
        except Exception as e:
            print(f"    ERROR: {e}")

    save_jsonl(results, output_path)
    _compute_reliability(results, output_path.parent)
    print(f"\n  ✓ Saved {len(results)} verification records")


def _compute_reliability(results: list, out_dir: Path):
    if not results:
        return

    judges_list = ["gpt5_mini", "local_qwen", "local_llama"]
    seqs = {j: [] for j in judges_list}
    rule_seq, plan_seq = [], []

    for r in results:
        for k in "ABCD":
            c = r["consensus"][k]
            rule_seq.append(LABEL_TO_INT.get(c["rule_label"], -1))
            plan_seq.append(LABEL_TO_INT.get(c["plan_label"], -1))
            for j in judges_list:
                jres = r["judges"].get(j, {})
                if "error" in jres:
                    seqs[j].append(-1)
                else:
                    seqs[j].append(LABEL_TO_INT.get(jres.get(k), -1))

    matrix = np.array([seqs[j] for j in judges_list]).T
    alpha = krippendorff_alpha_nominal(matrix)

    # Judge majority for κ
    judge_maj = []
    for i in range(len(rule_seq)):
        votes = [seqs[j][i] for j in judges_list if seqs[j][i] >= 0]
        if votes:
            judge_maj.append(Counter(votes).most_common(1)[0][0])
        else:
            judge_maj.append(-1)

    valid = [i for i in range(len(rule_seq))
             if rule_seq[i] >= 0 and judge_maj[i] >= 0]
    if valid:
        kappa = cohens_kappa([rule_seq[i] for i in valid],
                              [judge_maj[i] for i in valid])
    else:
        kappa = 0.0

    metrics = {
        "n_instances": len(results),
        "n_decisions": len(rule_seq),
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
    print(f"    α (judges):       {alpha:.3f} (≥{metrics['thresholds']['alpha_min']})")
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
    parser.add_argument("--no_paid", action="store_true")
    parser.add_argument("--no_local", action="store_true")
    args = parser.parse_args()
    run_pipeline(args.plan_path, args.profile_path, args.query_path,
                 args.output, args.limit,
                 not args.no_paid, not args.no_local)


if __name__ == "__main__":
    main()
