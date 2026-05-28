"""
2 STAGE

Query Generator v2 — Scenario-based matching.

For each compatible (user, scenario, active_axis) triple:
  1. Pick a query seed from the scenario's templates (explicit/implicit)
  2. Optionally use LLM to rephrase for natural variation
  3. Emit the instance record with full scenario metadata

No more random TPO sampling — all TPO context comes from the canonical catalog.
"""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

from .utils import call_llm, parse_json_list_response, save_jsonl, load_jsonl, log_step
from .compatibility import get_compatible_instances

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import (
    QUERIES_DIR, PROFILES_DIR, PHASE1_CONFIG, FASHION_ATTRIBUTE_AXES,
)
from configs.scenarios import CANONICAL_SCENARIOS, get_scenario_by_id


QUERY_SYSTEM = """You are a fashion query rewriter. Given an original query and context,
rephrase it in a different natural style while preserving ALL situational details.
Output ONLY a JSON list of 1 string: ["rephrased query"]"""


def _pick_query_type(rng):
    dist = PHASE1_CONFIG["query_type_distribution"]
    r = rng.random()
    cumulative = 0.0
    for qt, prob in dist.items():
        cumulative += prob
        if r < cumulative:
            return qt
    return list(dist.keys())[-1]


def _pick_seed_query(scenario, query_type, rng):
    """Pick a seed query text from the scenario's templates."""
    seeds = scenario.get("query_seeds", {})
    if query_type == "explicit_tpo":
        pool = seeds.get("explicit", [])
    else:
        pool = seeds.get("implicit", [])
    if not pool:
        pool = [f"I need an outfit for {scenario['name']}. What should I wear?"]
    return rng.choice(pool)


def _rephrase_query(seed_text, scenario_name, provider_override=None):
    """Use LLM to rephrase a seed query for variation. Returns original on failure."""
    prompt = f"""Rephrase this fashion recommendation query in a different natural style.
Keep ALL situational details (weather, venue, occasion) intact.

Original: "{seed_text}"
Context: {scenario_name}

Output JSON list: ["rephrased query"]"""
    try:
        response = call_llm(prompt=prompt, stage="query_generation",
                            system=QUERY_SYSTEM,
                            provider_override=provider_override)
        parsed = parse_json_list_response(response)
        if parsed and isinstance(parsed, list) and len(parsed) > 0:
            return str(parsed[0])
    except Exception:
        pass
    return seed_text


def _pick_liked_color(structured, rng):
    """
    Pick a color that is in user's likes.
    Fallback: anything not in dislikes.
    fixed_attrs color MUST come from user likes so the narrative profile
    is fully consistent with the option text.
    """
    col_likes = list(structured.get("color", {}).get("likes", []))
    col_dislikes = set(structured.get("color", {}).get("dislikes", []))

    # 1st: user likes (narrative consistent)
    liked_safe = [c for c in col_likes if c not in col_dislikes]
    if liked_safe:
        return rng.choice(liked_safe)

    # 2nd: anything not disliked
    safe = [c for c in ["black", "white", "navy", "gray", "beige"]
            if c not in col_dislikes]
    return rng.choice(safe or ["black"])


def _pick_liked_garment(structured, sc_compat_set, rng):
    """
    Pick a garment_category that is in user's likes AND scenario compatible.
    Fallback chain:
      1. user likes ∩ scenario compatible
      2. user likes only (ignore scenario compat)
      3. scenario compatible not in dislikes
      4. any scenario compatible
    fixed_attrs garment MUST come from user likes so the narrative profile
    is fully consistent with the option text.
    """
    gc_likes = list(structured.get("garment_category", {}).get("likes", []))
    gc_dislikes = set(structured.get("garment_category", {}).get("dislikes", []))

    # 1st: liked ∩ scenario compatible
    liked_compat = [g for g in gc_likes if g in sc_compat_set and g not in gc_dislikes]
    if liked_compat:
        return rng.choice(liked_compat)

    # 2nd: liked only (scenario compat may not cover user's preferred garments)
    liked_safe = [g for g in gc_likes if g not in gc_dislikes]
    if liked_safe:
        return rng.choice(liked_safe)

    # 3rd: scenario compatible not in dislikes
    safe_compat = [g for g in sc_compat_set if g not in gc_dislikes]
    if safe_compat:
        return rng.choice(safe_compat)

    # 4th: any scenario compatible
    return rng.choice(sorted(sc_compat_set) or ["shirt"])


def sample_fixed_attrs(active_axis, profile, scenario, rng):
    """
    Choose fixed attributes for axes that are NOT the active_axis.

    INVARIANT: fixed_attrs values must be in user's likes for every axis,
    so the narrative profile is fully consistent with the option text and
    LLMs reason only about the active_axis signal.
    """
    structured = profile["structured_attributes"]
    fixed = {}

    if active_axis == "garment_category":
        # Fix color (from user likes) and pattern
        fixed["color"] = _pick_liked_color(structured, rng)
        fixed["pattern"] = "solid"

    elif active_axis == "color":
        # Fix garment_category (user likes ∩ scenario compatible) and pattern
        sc_compat = set(scenario["garment_category"]["compatible"])
        fixed["garment_category"] = _pick_liked_garment(structured, sc_compat, rng)
        fixed["pattern"] = "solid"

    elif active_axis == "pattern":
        # Fix garment_category (user likes ∩ scenario compatible) and color (user likes)
        sc_compat = set(scenario["garment_category"]["compatible"])
        fixed["garment_category"] = _pick_liked_garment(structured, sc_compat, rng)
        fixed["color"] = _pick_liked_color(structured, rng)

    return fixed


def generate_instances(profiles, limit=0, provider_override=None,
                       rephrase=True):
    """Generate all query instances from compatible (user, scenario, axis) triples."""
    compatible, stats = get_compatible_instances(profiles)
    print(f"  Compatible instances: {len(compatible)}")

    if limit > 0:
        compatible = compatible[:limit]

    profiles_map = {p["user_id"]: p for p in profiles}
    instances = []

    for i, triple in enumerate(compatible):
        uid = triple["user_id"]
        sid = triple["scenario_id"]
        axis = triple["active_axis"]
        profile = profiles_map[uid]
        scenario = get_scenario_by_id(sid)

        rng = random.Random(abs(hash(f"{uid}_{sid}_{axis}")) % 100000)
        query_type = _pick_query_type(rng)
        seed_text = _pick_seed_query(scenario, query_type, rng)

        # Optionally rephrase for variation
        if rephrase and rng.random() < 0.5:
            query_text = _rephrase_query(seed_text, scenario["name"],
                                         provider_override)
        else:
            query_text = seed_text

        fixed = sample_fixed_attrs(axis, profile, scenario, rng)

        query_id = f"{uid}_{sid}_{axis[:2]}"
        instances.append({
            "query_id": query_id,
            "user_id": uid,
            "scenario_id": sid,
            "domain": "fashion",
            "query_type": query_type,
            "query_text": query_text,
            "tpo_scenario": scenario["tpo"],
            "active_axis": axis,
            "fixed_attrs": fixed,
            "scenario_archetype": scenario["archetype"],
            "scenario_name": scenario["name"],
            # Carry forward compatibility info for option planner
            "liked_compatible": triple["liked_compatible"],
            "disliked_compatible": triple["disliked_compatible"],
            "neutral_compatible": triple["neutral_compatible"],
            "liked_incompatible": triple["liked_incompatible"],
            "disliked_incompatible": triple["disliked_incompatible"],
        })

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(compatible)}] generated")

    return instances, stats


def run_pipeline(profile_path, output_path, force=False, limit=0,
                 provider=None, rephrase=True):
    log_step(f"Query+Scenario Generator v2 (provider={provider or 'default'})")

    profiles = load_jsonl(profile_path)
    print(f"  Loaded {len(profiles)} profiles")

    if output_path.exists() and not force:
        existing = load_jsonl(output_path)
        print(f"  Already have {len(existing)} queries. Use --force to regenerate.")
        return

    instances, stats = generate_instances(
        profiles, limit=limit, provider_override=provider, rephrase=rephrase
    )

    save_jsonl(instances, output_path)
    print(f"\n  ✓ Saved {len(instances)} query instances")
    print(f"  axis dist:     {dict(Counter(q['active_axis'] for q in instances))}")
    print(f"  qtype dist:    {dict(Counter(q['query_type'] for q in instances))}")
    print(f"  archetype dist: {dict(Counter(q['scenario_archetype'] for q in instances))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_path", type=Path,
                        default=PROFILES_DIR / "profiles.jsonl")
    parser.add_argument("--output", type=Path,
                        default=QUERIES_DIR / "queries.jsonl")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--no_rephrase", action="store_true")
    args = parser.parse_args()
    run_pipeline(args.profile_path, args.output, args.force, args.limit,
                 args.provider, not args.no_rephrase)


if __name__ == "__main__":
    main()
