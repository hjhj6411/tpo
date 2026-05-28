"""
Compatibility — user × scenario matching.

Benchmark 2×2 structure:
  A: preferred   color/pattern + TPO-compatible garment
  B: nonpref     color/pattern + TPO-compatible garment
  C: preferred   color/pattern + TPO-incompatible garment
  D: nonpref     color/pattern + TPO-incompatible garment

Therefore:
  - active_axis ∈ {color, pattern} ONLY
  - violation_axis = garment_category ALWAYS

A (user, scenario, active_axis) triple is compatible iff:
  1. active_axis ∈ {color, pattern}
  2. user has ≥1 liked value in scenario's compatible set for active_axis  → option A
  3. user has ≥1 non-preferred value in scenario's compatible set          → option B
  4. scenario has ≥1 incompatible garment_category value                   → options C/D
  5. user has ≥1 liked garment in scenario's compatible garment set        → fixed garment
"""

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.scenarios import CANONICAL_SCENARIOS

ACTIVE_AXES = ("color", "pattern")


def _profile_prefs(profile, axis):
    sa = profile.get("structured_attributes", {})
    ax_prefs = sa.get(axis, {})
    return set(ax_prefs.get("likes", [])), set(ax_prefs.get("dislikes", []))


def check_axis_compatibility(profile, scenario, axis):
    """
    Check if (profile, scenario, axis) can form a valid 2×2 instance.

    Returns dict with compatible=True/False and supporting value lists.
    """
    # Only color and pattern are active axes
    if axis not in ACTIVE_AXES:
        return {"compatible": False, "reason": "axis_not_active"}

    # active axis must be constrained in scenario
    active_constraint = scenario.get(axis)
    if not active_constraint or not active_constraint.get("incompatible"):
        return {"compatible": False, "reason": "active_axis_not_constrained"}

    # garment_category must have incompatible values (for C/D)
    gc_constraint = scenario.get("garment_category")
    if not gc_constraint or not gc_constraint.get("incompatible"):
        return {"compatible": False, "reason": "garment_not_constrained"}

    compat_set = set(active_constraint["compatible"])

    likes, dislikes = _profile_prefs(profile, axis)

    liked_compat    = sorted(likes & compat_set)
    disliked_compat = sorted(dislikes & compat_set)
    neutral_compat  = sorted(compat_set - likes - dislikes)

    # A: need liked + TPO-compatible
    has_A = len(liked_compat) > 0
    # B: need non-preferred (disliked or neutral) + TPO-compatible
    has_B = len(disliked_compat) > 0 or len(neutral_compat) > 0

    # fixed garment: user must like at least one TPO-compatible garment
    gc_compat_set = set(gc_constraint["compatible"])
    gc_likes, _ = _profile_prefs(profile, "garment_category")
    liked_garments = sorted(gc_likes & gc_compat_set)
    has_garment = len(liked_garments) > 0

    compatible = has_A and has_B and has_garment

    return {
        "compatible": compatible,
        "liked_compatible":    liked_compat,
        "disliked_compatible": disliked_compat,
        "neutral_compatible":  neutral_compat,
        "liked_garments":      liked_garments,           # for fixed_attrs
        "incompat_garments":   sorted(gc_constraint["incompatible"]),  # for C/D
    }


def build_compatibility_matrix(profiles, scenarios=None):
    if scenarios is None:
        scenarios = CANONICAL_SCENARIOS

    matrix = {}
    stats = {
        "total_slots": 0, "compatible_slots": 0,
        "by_axis":      defaultdict(lambda: {"total": 0, "compat": 0}),
        "by_archetype": defaultdict(lambda: {"total": 0, "compat": 0}),
        "by_user":      defaultdict(lambda: {"total": 0, "compat": 0}),
    }

    for profile in profiles:
        uid = profile["user_id"]
        matrix[uid] = {}

        for sc in scenarios:
            sid = sc["scenario_id"]
            matrix[uid][sid] = {}

            for axis in ACTIVE_AXES:
                result = check_axis_compatibility(profile, sc, axis)
                matrix[uid][sid][axis] = result

                stats["total_slots"] += 1
                stats["by_axis"][axis]["total"] += 1
                stats["by_archetype"][sc["archetype"]]["total"] += 1
                stats["by_user"][uid]["total"] += 1

                if result.get("compatible"):
                    stats["compatible_slots"] += 1
                    stats["by_axis"][axis]["compat"] += 1
                    stats["by_archetype"][sc["archetype"]]["compat"] += 1
                    stats["by_user"][uid]["compat"] += 1

    stats["compatibility_rate"] = (
        stats["compatible_slots"] / max(stats["total_slots"], 1)
    )
    return matrix, dict(stats)


def get_compatible_instances(profiles, scenarios=None):
    matrix, stats = build_compatibility_matrix(profiles, scenarios)
    instances = []

    for uid, sc_map in matrix.items():
        for sid, axis_map in sc_map.items():
            for axis, result in axis_map.items():
                if result.get("compatible"):
                    instances.append({
                        "user_id":             uid,
                        "scenario_id":         sid,
                        "active_axis":         axis,
                        "liked_compatible":    result["liked_compatible"],
                        "disliked_compatible": result["disliked_compatible"],
                        "neutral_compatible":  result["neutral_compatible"],
                        "liked_garments":      result["liked_garments"],
                        "incompat_garments":   result["incompat_garments"],
                    })

    return instances, stats


def print_compatibility_report(stats):
    total  = stats["total_slots"]
    compat = stats["compatible_slots"]
    rate   = stats["compatibility_rate"]
    print(f"\n  Compatibility: {compat}/{total} = {rate:.1%}")
    print(f"\n  By axis:")
    for ax, s in stats["by_axis"].items():
        r = s["compat"] / max(s["total"], 1)
        print(f"    {ax:20s}: {s['compat']}/{s['total']} = {r:.1%}")
    print(f"\n  By archetype:")
    for arch, s in stats["by_archetype"].items():
        r = s["compat"] / max(s["total"], 1)
        print(f"    {arch:25s}: {s['compat']}/{s['total']} = {r:.1%}")
    print(f"\n  By user:")
    for uid, s in sorted(stats["by_user"].items()):
        r = s["compat"] / max(s["total"], 1)
        print(f"    {uid}: {s['compat']}/{s['total']} = {r:.1%}")
