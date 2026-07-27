"""
STAGE 2 — Query Generator (clean 2×2).

Builds one query record per compatible (user, scenario, active_axis) with
active_axis in {color, pattern, garment_category} (see ALLOWED_ACTIVE_AXES
below) — garment-active instances exist only for dress-code scenarios, where
color or pattern can carry the TPO norm instead.

The violation axis is NOT fixed here — the planner (STAGE 3) picks a neutral
compatible/incompatible pair on some *other* axis, so that axis alone carries
the TPO contrast while the selected active axis carries the like/dislike
preference contrast. For physical scenarios that other axis is always the
garment.

Updated for the cleaned pattern vocabulary:
- fallback pattern values use only canonical, stable patterns
- only cleaned canonical fallback pattern values are used
- local RNG seed is process-independent
"""

import argparse
import hashlib
import random
import sys
from collections import Counter
from pathlib import Path

from .utils import save_jsonl, load_jsonl, log_step

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import PROFILES_DIR, QUERIES_DIR, FASHION_ATTRIBUTE_AXES
from configs.scenarios import CANONICAL_SCENARIOS
from .compatibility import get_compatible_instances, print_compatibility_report

# Value axes plus garment: garment-active queries exist only for coded
# (dress-code) scenarios — compatibility.py emits those instances only there.
ALLOWED_ACTIVE_AXES = {"color", "pattern", "garment_category"}

# Niche-knowledge scenarios where an implicit query would test domain trivia
# instead of constraint application (design doc §4.3 / §6.2): force explicit
# queries so the rule is always stated in the query text.
EXPLICIT_ONLY_SCENARIOS = {
    "field_wildlife_hide", "stage_greenscreen_shoot",
    # festive_bright / garden_floral norms ("wear bright colors" / "florals")
    # are not universal common sense, so the rule must be stated in the query
    # (§6.3): this makes the dark/leopard violation defensible and turns the
    # item into instruction-application + preference rather than a knowledge test.
    "festive_holiday_party", "festive_bright_theme_party", "festive_street_fiesta",
    "garden_spring_party", "garden_afternoon_tea", "garden_bridal_shower",
    # This event uses a stated university guest dress guide rather than a
    # universal graduation color rule.
    "celebration_graduation",
}

# Safe neutral fallbacks. Deliberately excludes high-salience patterns such as
# floral, polka_dot, and leopard to avoid injecting a second strong visual cue
# into the non-active axis. All values must exist in FASHION_ATTRIBUTE_AXES.
FALLBACK_AXIS_VALUES = {
    "color": ["black", "white", "navy", "gray", "beige"],
    "pattern": ["solid", "striped", "checkered"],
}

for _axis, _vals in FALLBACK_AXIS_VALUES.items():
    _unknown = set(_vals) - set(FASHION_ATTRIBUTE_AXES[_axis])
    if _unknown:
        raise ValueError(f"Fallback values not in config vocabulary for {_axis}: {_unknown}")


def _stable_seed(*parts) -> int:
    payload = "::".join(str(p) for p in parts)
    return int(hashlib.md5(payload.encode("utf-8")).hexdigest()[:8], 16)


def _profile_prefs(profile, axis):
    sa = profile.get("structured_attributes", {})
    ax_prefs = sa.get(axis, {})
    return set(ax_prefs.get("likes", [])), set(ax_prefs.get("dislikes", []))


def _neutral_values(values, likes, dislikes):
    return sorted(set(values) - likes - dislikes)


def _sample_preference_neutral_value(profile, scenario, axis, rng,
                                     usage=None, user_id=None):
    """A preference-NEUTRAL, TPO-compatible value for a NON-active axis.

    Returns None if none exists — the caller then leaves that axis unfixed,
    which is safer than injecting a liked/disliked value (a second preference
    signal) or a TPO-incompatible value (a second TPO signal).

    When the scenario does not constrain a COLOR axis, any color is
    situation-appropriate, so the pool is the full vocabulary minus the
    user's likes/dislikes (the narrow dress-code-safe fallback list would
    collapse to a single value for users whose preferences cover the rest).
    Pattern keeps the low-salience fallback set: a high-salience fixed
    pattern (leopard/floral/polka_dot) would inject a second strong visual
    cue into the non-active axis.

    `usage` is a per-(user, axis, value) counter; when given, the least-used
    value is picked first so one user does not repeat the same fixed value
    across their whole query set.
    """
    if axis == "garment_category":
        return None

    likes, dislikes = _profile_prefs(profile, axis)
    constraint = scenario.get(axis)
    tpo_bad = set(constraint.get("incompatible", [])) if constraint else set()
    axis_constrained = bool(constraint and constraint.get("incompatible"))

    # 1) scenario-compatible AND preference-neutral
    pool = []
    if constraint and constraint.get("compatible"):
        pool = _neutral_values(constraint["compatible"], likes, dislikes)

    # 2) constrained axis with no compatible-neutral value left -> UNFIXED.
    # A fallback here would inject a value the scenario does not declare
    # compatible (e.g. checkered at a funeral), so options could no longer
    # claim every non-violation attribute is TPO-appropriate.
    if not pool and axis_constrained:
        return None

    # 3) unconstrained axis: generic preference-neutral fallback
    if not pool:
        if axis == "color":
            base = FASHION_ATTRIBUTE_AXES["color"]
        else:
            base = FALLBACK_AXIS_VALUES.get(axis, [])
        pool = [v for v in _neutral_values(base, likes, dislikes)
                if v not in tpo_bad]

    # 4) give up — leave this axis unfixed rather than leak a bad value
    if not pool:
        return None

    if usage is None:
        return rng.choice(pool)
    min_use = min(usage[(user_id, axis, v)] for v in pool)
    least_used = [v for v in pool if usage[(user_id, axis, v)] == min_use]
    pick = rng.choice(least_used)
    usage[(user_id, axis, pick)] += 1
    return pick


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
    if scenario.get("scenario_id") in EXPLICIT_ONLY_SCENARIOS:
        query_type = "explicit"
    else:
        query_type = "explicit" if rng.random() < explicit_ratio else "implicit"
    pool = _extract_seed_pool(scenario, query_type)
    if not pool:
        pool = _extract_seed_pool(scenario, "explicit" if query_type == "implicit" else "implicit")
    if pool:
        return query_type, rng.choice(pool)
    return query_type, _render_fallback_query(scenario, query_type)


def _build_fixed_attrs(profile, scenario, active_axis, rng, usage=None, user_id=None):
    fixed_attrs = {}
    fixed_bg = scenario.get("fixed_background") or {}
    for axis in ("color", "pattern"):
        if axis == active_axis:
            continue
        if axis in fixed_bg:
            # axis pinned to a constant background value for this scenario
            # (e.g. sports jerseys are always solid); no preference signal.
            fixed_attrs[axis] = fixed_bg[axis]
            continue
        val = _sample_preference_neutral_value(
            profile, scenario, axis, rng, usage=usage, user_id=user_id)
        if val is not None:
            fixed_attrs[axis] = val
    return fixed_attrs


def _make_query_id(user_id, scenario_id, active_axis, idx):
    # global index FIRST + zero-padded, so records / image folders sort
    # numerically by the global instance number (not by user_id alphabetically).
    return f"q{idx:05d}__{user_id}__{scenario_id}__{active_axis}"


def build_queries(profiles, scenarios=None, seed=42, per_instance=1, explicit_ratio=0.5):
    if scenarios is None:
        scenarios = CANONICAL_SCENARIOS

    scenario_map = {s["scenario_id"]: s for s in scenarios}
    profile_map = {p["user_id"]: p for p in profiles}
    compatible_instances, stats = get_compatible_instances(profiles, scenarios)
    queries = []

    # Shared per-(user, axis, value) counter so fixed non-active values rotate
    # per user instead of repeating one value across all of a user's queries.
    fixed_usage = Counter()

    running_idx = 0
    for inst in compatible_instances:
        profile = profile_map[inst["user_id"]]
        scenario = scenario_map[inst["scenario_id"]]
        active_axis = inst["active_axis"]

        if active_axis not in ALLOWED_ACTIVE_AXES:
            continue

        for _ in range(per_instance):
            running_idx += 1
            local_rng = random.Random(_stable_seed(seed, running_idx))

            query_type, query_text = _build_query_text(
                scenario, local_rng, explicit_ratio=explicit_ratio
            )
            fixed_attrs = _build_fixed_attrs(
                profile, scenario, active_axis, local_rng,
                usage=fixed_usage, user_id=inst["user_id"],
            )
            query_id = _make_query_id(
                inst["user_id"], inst["scenario_id"], active_axis, running_idx
            )

            queries.append({
                "query_id": query_id,
                "user_id": inst["user_id"],
                "scenario_id": inst["scenario_id"],
                "scenario_archetype": scenario.get("archetype"),
                "scenario_name": scenario.get("name"),
                "track": scenario.get("track"),
                "active_axis": active_axis,
                "liked_compatible": inst["liked_compatible"],
                "disliked_compatible": inst["disliked_compatible"],
                "neutral_compatible": inst["neutral_compatible"],
                "compatible_garments": inst["compatible_garments"],
                "incompatible_garments": inst["incompatible_garments"],
                "violation_options": inst.get("violation_options", {}),
                "query_type": query_type,
                "query_text": query_text,
                "fixed_attrs": fixed_attrs,
            })

    return queries, stats


def run_pipeline(profile_path, output_path, force=False, seed=42, per_instance=1,
                 explicit_ratio=0.5, limit=0):
    log_step("STAGE 2 — Query Generator (clean 2×2)")

    if output_path.exists() and not force:
        print(f"  Output exists: {output_path}\n  Use --force to overwrite.")
        return

    profiles = load_jsonl(profile_path)
    print(f"  {len(profiles)} profiles loaded")

    queries, stats = build_queries(
        profiles, scenarios=CANONICAL_SCENARIOS, seed=seed,
        per_instance=per_instance, explicit_ratio=explicit_ratio,
    )
    print_compatibility_report(stats)

    if limit > 0:
        queries = queries[:limit]

    save_jsonl(queries, output_path)
    print(f"\n  ✓ Saved {len(queries)} queries → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_path", type=Path, default=PROFILES_DIR / "profiles.jsonl")
    parser.add_argument("--output", type=Path, default=QUERIES_DIR / "queries.jsonl")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per_instance", type=int, default=1)
    parser.add_argument("--explicit_ratio", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    run_pipeline(
        profile_path=args.profile_path, output_path=args.output, force=args.force,
        seed=args.seed, per_instance=args.per_instance,
        explicit_ratio=args.explicit_ratio, limit=args.limit,
    )


if __name__ == "__main__":
    main()
