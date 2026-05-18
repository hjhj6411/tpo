"""
Option Planner (English, Fashion)
==================================

For each (profile, query) pair, plan 4 image options as attribute specs:

  A: TPO_match=YES, Preference_match=YES   (correct)
  B: TPO_match=YES, Preference_match=NO
  C: TPO_match=NO,  Preference_match=YES
  D: TPO_match=NO,  Preference_match=NO

All 4 options share the SAME garment_category. Differentiation comes from
fine-grained attributes (color, material, pattern, detail) — captioner-resistant.

Attribute values are drawn ONLY from the closed-set vocabulary in
configs/config.py to make automatic rule-based labeling reliable.
"""

import argparse
import random
from pathlib import Path

from .utils import (
    call_gpt5_mini, parse_json_response,
    save_jsonl, load_jsonl, log_step,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import (
    OPTIONS_DIR, PROFILES_DIR, QUERIES_DIR,
    FASHION_ATTRIBUTE_AXES, GPT5_MINI,
)


# ─────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────

OPTION_PLAN_SYSTEM = """\
You design 4-option multiple-choice items for a fashion personalization benchmark.

For each user profile + query, output exactly 4 options A/B/C/D following this 2x2 matrix:
  A: matches BOTH user preference AND TPO situation   (correct)
  B: matches TPO situation, VIOLATES user preference
  C: matches user preference, VIOLATES TPO situation
  D: violates BOTH

CRITICAL DESIGN RULES:
1. ALL 4 options must share the same garment_category. They must look like
   sibling alternatives in the same category, not different garment types.
2. Each option's attribute values MUST come from the controlled vocabulary
   I will provide. Do not invent attribute values.
3. Differentiation across options should mix coarse attributes (color, material)
   with fine-grained ones (pattern, fit, sleeve_length, neckline). Avoid
   trivially captionable extreme contrasts (e.g., neon vs black).
4. Each option also has a 3-7 word English `search_query` usable for catalog/
   web image search.
5. Each option has a `rationale` (1 sentence) explaining its label.

Output ONLY a JSON object.
"""


def render_attribute_vocabulary() -> str:
    """Render the closed-set vocabulary for the LLM."""
    lines = ["Controlled vocabulary (use ONLY these values for each axis):"]
    for axis, values in FASHION_ATTRIBUTE_AXES.items():
        lines.append(f"  - {axis}: {values}")
    return "\n".join(lines)


def build_option_plan_prompt(profile: dict, query: dict) -> str:
    """Build the option-planning prompt."""

    # Both keyword and narrative representations are shown to the planner,
    # since we want the option labels to be correct under both profile variants.
    likes_str = ", ".join(profile["likes_keywords"]) or "(none)"
    dislikes_str = ", ".join(profile["dislikes_keywords"]) or "(none)"

    tpo = query.get("tpo_scenario") or {}
    tpo_str = "(no TPO; neutral query)"
    if tpo:
        parts = []
        for axis, sub in tpo.items():
            if isinstance(sub, dict):
                for k, v in sub.items():
                    parts.append(f"{axis}.{k}={v}")
        tpo_str = "; ".join(parts)

    vocab = render_attribute_vocabulary()

    return f"""Design 4 options for the following instance.

USER PROFILE
  narrative: {profile['narrative_profile']}
  likes:    {likes_str}
  dislikes: {dislikes_str}

QUERY ({query['query_type']})
  text: "{query['query_text']}"
  TPO:  {tpo_str}

{vocab}

OUTPUT JSON:
{{
  "main_category": "<one value from garment_category vocabulary>",
  "options": {{
    "A": {{
      "label": "tpo_and_preference",
      "attributes": {{"color": "...", "material": "...", "pattern": "...", ...}},
      "search_query": "<3-7 English words>",
      "rationale": "<one sentence>"
    }},
    "B": {{"label": "tpo_only", "attributes": {{...}}, "search_query": "...", "rationale": "..."}},
    "C": {{"label": "preference_only", "attributes": {{...}}, "search_query": "...", "rationale": "..."}},
    "D": {{"label": "neither", "attributes": {{...}}, "search_query": "...", "rationale": "..."}}
  }}
}}"""


# ─────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────

EXPECTED_LABELS = {
    "A": "tpo_and_preference",
    "B": "tpo_only",
    "C": "preference_only",
    "D": "neither",
}


def validate_plan(plan: dict) -> tuple[bool, str]:
    """Check plan structure and vocabulary compliance."""
    if "main_category" not in plan or not plan["main_category"]:
        return False, "missing main_category"
    if plan["main_category"] not in FASHION_ATTRIBUTE_AXES["garment_category"]:
        return False, f"main_category '{plan['main_category']}' not in vocabulary"

    options = plan.get("options", {})
    if set(options.keys()) != {"A", "B", "C", "D"}:
        return False, f"options keys should be A/B/C/D, got {list(options.keys())}"

    for k, expected_label in EXPECTED_LABELS.items():
        opt = options[k]
        if opt.get("label") != expected_label:
            return False, f"option {k} label mismatch: {opt.get('label')} vs {expected_label}"
        if not opt.get("search_query"):
            return False, f"option {k} missing search_query"
        attrs = opt.get("attributes") or {}
        if not attrs:
            return False, f"option {k} missing attributes"

        # Validate each attribute value is in vocabulary
        for ax, val in attrs.items():
            if ax not in FASHION_ATTRIBUTE_AXES:
                return False, f"option {k} unknown axis '{ax}'"
            if val not in FASHION_ATTRIBUTE_AXES[ax]:
                return False, f"option {k}.{ax}='{val}' not in vocabulary"

        # Ensure garment_category matches main_category
        if attrs.get("garment_category") and attrs["garment_category"] != plan["main_category"]:
            return False, f"option {k} garment_category mismatches main_category"

    return True, ""


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def plan_options_for_query(profile: dict, query: dict) -> dict:
    prompt = build_option_plan_prompt(profile, query)
    response = call_gpt5_mini(
        prompt=prompt, system=OPTION_PLAN_SYSTEM,
        max_completion_tokens=GPT5_MINI["max_completion_tokens_long"],
    )
    plan = parse_json_response(response)
    if plan is None:
        return None

    ok, msg = validate_plan(plan)
    if not ok:
        # Single retry with explicit error feedback
        retry_prompt = (
            prompt
            + f"\n\nPrevious attempt failed validation: {msg}\n"
            + "Please regenerate, strictly using ONLY the controlled vocabulary."
        )
        response = call_gpt5_mini(
            prompt=retry_prompt, system=OPTION_PLAN_SYSTEM,
            max_completion_tokens=GPT5_MINI["max_completion_tokens_long"],
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
        "domain": "fashion",
        "query_type": query["query_type"],
        "main_category": plan["main_category"],
        "options": plan["options"],
    }


def run_pipeline(profile_path: Path, query_path: Path, output_path: Path,
                 force: bool = False, limit: int = 0):
    log_step("Option Planner")

    profiles = {p["user_id"]: p for p in load_jsonl(profile_path)}
    queries = load_jsonl(query_path)
    print(f"  Loaded {len(profiles)} profiles, {len(queries)} queries")

    if output_path.exists() and not force:
        existing = load_jsonl(output_path)
        done = {p["query_id"] for p in existing}
        plans = existing
        queries_todo = [q for q in queries if q["query_id"] not in done]
        print(f"  Resuming: {len(done)} already planned")
    else:
        plans = []
        queries_todo = queries

    if limit > 0:
        queries_todo = queries_todo[:limit]

    n_fail = 0
    for i, query in enumerate(queries_todo):
        if query["user_id"] not in profiles:
            continue
        prof = profiles[query["user_id"]]
        print(f"\n  [{i+1}/{len(queries_todo)}] {query['query_id']} ({query['query_type']})")
        print(f"    Q: {query['query_text']}")
        try:
            plan = plan_options_for_query(prof, query)
            if plan is None:
                print("    Failed validation")
                n_fail += 1
                continue
            plans.append(plan)
            print(f"    Category: {plan['main_category']}")
            for k in "ABCD":
                attrs = plan["options"][k]["attributes"]
                print(f"      {k} ({plan['options'][k]['label']}): {attrs}")
            if (i + 1) % 10 == 0:
                save_jsonl(plans, output_path)
        except Exception as e:
            print(f"    ERROR: {e}")
            n_fail += 1

    save_jsonl(plans, output_path)
    print(f"\n  ✓ Saved {len(plans)} plans, failed {n_fail}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_path", type=Path,
                        default=PROFILES_DIR / "profiles.jsonl")
    parser.add_argument("--query_path", type=Path,
                        default=QUERIES_DIR / "queries.jsonl")
    parser.add_argument("--output", type=Path,
                        default=OPTIONS_DIR / "option_plans.jsonl")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    run_pipeline(args.profile_path, args.query_path, args.output,
                 args.force, args.limit)


if __name__ == "__main__":
    main()
