"""
Query Generator — clean 2×2 version.

Build query records only for active_axis in {color, pattern}.
Garment category is not fixed here; planner chooses a neutral compatible/incompatible pair.
"""

import argparse
import random
import sys
from pathlib import Path

from .utils import save_jsonl, load_jsonl, log_step

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import PROFILES_DIR, QUERIES_DIR
from configs.scenarios import CANONICAL_SCENARIOS, get_scenario_by_id
from .compatibility import get_compatible_instances, print_compatibility_report

ALLOWED_ACTIVE_AXES = {"color", "pattern"}

FALLBACK_AXIS_VALUES = {
    "color": ["black", "white", "navy", "gray", "beige"],
    "pattern": ["solid", "striped", "plaid"],
}


def _profile_prefs(profile, axis):
    sa = profile.get("structured_attributes", {})
    ax_prefs = sa.get(axis, {})
    return set(ax_prefs.get("likes", [])), set(ax_prefs.get("dislikes", []))


def _neutral_values(values, likes, dislikes):
    return sorted(set(values) - likes - dislikes)


def _sample_preference_neutral_value(profile, scenario, axis, rng):
    if axis == "garment_category":
        return None

    likes, dislikes = _profile_prefs(profile, axis)
    constraint = scenario.get(axis)

    if constraint and constraint.get("compatible"):
        pool = _neutral_values(constraint["compatible"], likes, dislikes)
        if pool:
            return rng.choice(pool)

    fallback_pool = _neutral_values(FALLBACK_AXIS_VALUES.get(axis, []), likes, dislikes)
    if fallback_pool:
        return rng.choice(fallback_pool)

    raw_pool = FALLBACK_AXIS_VALUES.get(axis, [])
    if raw_pool:
        return rng.choice(raw_pool)

    return None


def _extract_seed_pool(scenario, query_type):
    seeds = scenario.get("query_seeds", {})
    if isinstance(seeds, dict):
        pool = seeds.get(query_type, [])
        if pool:
            return pool
        alt = seeds.get("explicit", []) + seeds.get("implicit", [])
        if alt:
            return alt
    elif isinstance(seeds, list):
        if seeds:
            return seeds
    return []


def _render_fallback_query(scenario, query_type):
    name = scenario.get("name", "").strip()
    if query_type == "implicit":
        if name:
            return f"I have an event coming up: {name}. Any outfit suggestion?"
        return "I have an event coming up. What should I wear?"
    if name:
        return f"For this situation — {name} — what should I wear?"
    return "What should I wear for this situation?"


def _build_query_text(scenario, rng, explicit_ratio=0.5):
    query_type = "explicit" if rng.random() < explicit_ratio else "implicit"
    pool = _extract_seed_pool(scenario, query_type)
    if not pool:
        pool = _extract_seed_pool(scenario, "explicit" if query_type == "implicit" else "implicit")
    if pool:
        return query_type, rng.choice(pool)
    return query_type, _render_fallback_query(scenario, query_type)


def _build_fixed_attrs(profile, scenario, active_axis, rng):
    fixed_attrs = {}
    for axis in ("color", "pattern"):
        if axis == active_axis:
            continue
        val = _sample_preference_neutral_value(profile, scenario, axis, rng)
        if val is not None:
            fixed_attrs[axis] = val
    return fixed_attrs


def _make_query_id(user_id, scenario_id, active_axis, idx):
    return f"{user_id}__{scenario_id}__{active_axis}__{idx:04d}"


def build_queries(profiles, scenarios=None, seed=42, per_instance=1, explicit_ratio=0.5):
    if scenarios is None:
        scenarios = CANONICAL_SCENARIOS

    scenario_map = {s["scenario_id"]: s for s in scenarios}
    compatible_instances, stats = get_compatible_instances(profiles, scenarios)
    queries = []

    rng = random.Random(seed)
    running_idx = 0

    for inst in compatible_instances:
        profile = next(p for p in profiles if p["user_id"] == inst["user_id"])
        scenario = scenario_map[inst["scenario_id"]]
        active_axis = inst["active_axis"]

        if active_axis not in ALLOWED_ACTIVE_AXES:
            continue

        for _ in range(per_instance):
            running_idx += 1
            local_rng = random.Random((seed, running_idx).__hash__())

            query_type, query_text = _build_query_text(
                scenario, local_rng, explicit_ratio=explicit_ratio
            )
            fixed_attrs = _build_fixed_attrs(profile, scenario, active_axis, local_rng)
            query_id = _make_query_id(
                inst["user_id"], inst["scenario_id"], active_axis, running_idx
            )

            record = {
                "query_id": query_id,
                "user_id": inst["user_id"],
                "scenario_id": inst["scenario_id"],
                "scenario_archetype": scenario.get("archetype"),   # ← 추가
                "scenario_name": scenario.get("name"),   
                "active_axis": active_axis,
                "liked_compatible": inst["liked_compatible"],
                "disliked_compatible": inst["disliked_compatible"],
                "neutral_compatible": inst["neutral_compatible"],
                "compatible_garments": inst["compatible_garments"],
                "incompatible_garments": inst["incompatible_garments"],
                "query_type": query_type,
                "query_text": query_text,
                "fixed_attrs": fixed_attrs,
            }
            queries.append(record)

    return queries, stats


def run_pipeline(profile_path, output_path, force=False, seed=42, per_instance=1,
                 explicit_ratio=0.5, limit=0):
    log_step("Query Generator (clean 2×2)")
    profiles = load_jsonl(profile_path)
    print(f"  {len(profiles)} profiles loaded")

    queries, stats = build_queries(
        profiles,
        scenarios=CANONICAL_SCENARIOS,
        seed=seed,
        per_instance=per_instance,
        explicit_ratio=explicit_ratio,
    )

    print_compatibility_report(stats)

    if limit > 0:
        queries = queries[:limit]

    if output_path.exists() and not force:
        print(f"  Output exists: {output_path}")
        print("  Use --force to overwrite.")
        return

    save_jsonl(queries, output_path)
    print(f"\n  ✓ Saved {len(queries)} queries to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_path", type=Path,
                        default=PROFILES_DIR / "profiles.jsonl")
    parser.add_argument("--output", type=Path,
                        default=QUERIES_DIR / "queries.jsonl")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per_instance", type=int, default=1)
    parser.add_argument("--explicit_ratio", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    run_pipeline(
        profile_path=args.profile_path,
        output_path=args.output,
        force=args.force,
        seed=args.seed,
        per_instance=args.per_instance,
        explicit_ratio=args.explicit_ratio,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()