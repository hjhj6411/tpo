"""
Query Generator v2 — Scenario-based matching.

Benchmark 2×2 structure:
  A: preferred   color/pattern + TPO-compatible garment
  B: nonpref     color/pattern + TPO-compatible garment
  C: preferred   color/pattern + TPO-incompatible garment
  D: nonpref     color/pattern + TPO-incompatible garment

active_axis ∈ {color, pattern} ONLY.
violation_axis = garment_category ALWAYS.

fixed_attrs contains:
  - garment_category: user_liked ∩ TPO-compatible  (used in A/B)
  - the other non-active axis (color or pattern): from user likes
"""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

from .utils import call_llm, parse_json_list_response, save_jsonl, load_jsonl, log_step
from .compatibility import get_compatible_instances

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import QUERIES_DIR, PROFILES_DIR, PHASE1_CONFIG
from configs.scenarios import get_scenario_by_id


QUERY_SYSTEM = """You are a fashion query rewriter. Given an original query and context,
rephrase it in a different natural style while preserving ALL situational details.
Output ONLY a JSON list of 1 string: [\"rephrased query\"]"""


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
    seeds = scenario.get("query_seeds", {})
    pool = seeds.get("explicit" if query_type == "explicit_tpo" else "implicit", [])
    if not pool:
        pool = [f"I need an outfit for {scenario['name']}. What should I wear?"]
    return rng.choice(pool)


def _rephrase_query(seed_text, scenario_name, provider_override=None):
    prompt = (
        f'Rephrase this fashion recommendation query in a different natural style.\n'
        f'Keep ALL situational details intact.\n\n'
        f'Original: "{seed_text}"\nContext: {scenario_name}\n\n'
        f'Output JSON list: ["rephrased query"]'
    )
    try:
        response = call_llm(prompt=prompt, stage="query_generation",
                            system=QUERY_SYSTEM, provider_override=provider_override)
        parsed = parse_json_list_response(response)
        if parsed and isinstance(parsed, list) and parsed:
            return str(parsed[0])
    except Exception:
        pass
    return seed_text


def _pick_from_likes(structured, axis, rng, fallbacks=None):
    """
    Pick a value from user's likes for the given axis.
    Falls back to anything not in dislikes.
    """
    likes    = list(structured.get(axis, {}).get("likes", []))
    dislikes = set(structured.get(axis, {}).get("dislikes", []))
    safe = [v for v in likes if v not in dislikes]
    if safe:
        return rng.choice(safe)
    if fallbacks:
        safe2 = [v for v in fallbacks if v not in dislikes]
        if safe2:
            return rng.choice(safe2)
    return None


def sample_fixed_attrs(active_axis, instance, profile, rng):
    """
    Build fixed_attrs for A/B options.

    Always contains:
      garment_category: user_liked ∩ TPO-compatible  (from compatibility check)
      <other non-active axis>: from user likes

    INVARIANT: every value in fixed_attrs is in user's likes.
    """
    structured = profile["structured_attributes"]
    fixed = {}

    # garment_category: user liked AND TPO-compatible (pre-computed in compatibility)
    liked_garments = instance["liked_garments"]
    fixed["garment_category"] = rng.choice(liked_garments)

    # the other non-active axis
    if active_axis == "color":
        fixed["pattern"] = _pick_from_likes(
            structured, "pattern", rng,
            fallbacks=["solid", "striped", "plaid"]
        )
    elif active_axis == "pattern":
        fixed["color"] = _pick_from_likes(
            structured, "color", rng,
            fallbacks=["black", "white", "navy", "gray", "beige"]
        )

    return fixed


def generate_instances(profiles, limit=0, provider_override=None, rephrase=True):
    compatible, stats = get_compatible_instances(profiles)
    print(f"  Compatible instances: {len(compatible)}")

    if limit > 0:
        compatible = compatible[:limit]

    profiles_map = {p["user_id"]: p for p in profiles}
    instances = []

    for i, triple in enumerate(compatible):
        uid  = triple["user_id"]
        sid  = triple["scenario_id"]
        axis = triple["active_axis"]
        profile  = profiles_map[uid]
        scenario = get_scenario_by_id(sid)

        rng = random.Random(abs(hash(f"{uid}_{sid}_{axis}")) % 100000)
        query_type = _pick_query_type(rng)
        seed_text  = _pick_seed_query(scenario, query_type, rng)

        if rephrase and rng.random() < 0.5:
            query_text = _rephrase_query(seed_text, scenario["name"], provider_override)
        else:
            query_text = seed_text

        fixed = sample_fixed_attrs(axis, triple, profile, rng)

        query_id = f"{uid}_{sid}_{axis[:2]}"
        instances.append({
            "query_id":            query_id,
            "user_id":             uid,
            "scenario_id":         sid,
            "domain":              "fashion",
            "query_type":          query_type,
            "query_text":          query_text,
            "tpo_scenario":        scenario["tpo"],
            "active_axis":         axis,
            "fixed_attrs":         fixed,
            "scenario_archetype":  scenario["archetype"],
            "scenario_name":       scenario["name"],
            # Values for option planner
            "liked_compatible":    triple["liked_compatible"],
            "disliked_compatible": triple["disliked_compatible"],
            "neutral_compatible":  triple["neutral_compatible"],
            "incompat_garments":   triple["incompat_garments"],
        })

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(compatible)}] generated")

    return instances, stats


def run_pipeline(profile_path, output_path, force=False, limit=0,
                 provider=None, rephrase=True):
    log_step(f"Query Generator v2 (provider={provider or 'default'})")
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
    print(f"  axis dist:      {dict(Counter(q['active_axis'] for q in instances))}")
    print(f"  qtype dist:     {dict(Counter(q['query_type'] for q in instances))}")
    print(f"  archetype dist: {dict(Counter(q['scenario_archetype'] for q in instances))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_path", type=Path, default=PROFILES_DIR / "profiles.jsonl")
    parser.add_argument("--output",       type=Path, default=QUERIES_DIR / "queries.jsonl")
    parser.add_argument("--force",        action="store_true")
    parser.add_argument("--limit",        type=int, default=0)
    parser.add_argument("--provider",     type=str, default=None)
    parser.add_argument("--no_rephrase",  action="store_true")
    args = parser.parse_args()
    run_pipeline(args.profile_path, args.output, args.force, args.limit,
                 args.provider, not args.no_rephrase)


if __name__ == "__main__":
    main()
