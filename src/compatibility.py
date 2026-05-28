"""
Compatibility — clean 2×2 version.

Only color/pattern can be active axes.
Garment category is always the TPO axis and must be preference-neutral.
"""

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.scenarios import CANONICAL_SCENARIOS, get_constrained_axes

ALLOWED_ACTIVE_AXES = {"color", "pattern"}


def _profile_prefs(profile, axis):
    sa = profile.get("structured_attributes", {})
    ax_prefs = sa.get(axis, {})
    return set(ax_prefs.get("likes", [])), set(ax_prefs.get("dislikes", []))


def _neutral_values(values, likes, dislikes):
    return sorted(set(values) - likes - dislikes)


def check_axis_compatibility(profile, scenario, axis):
    if axis not in ALLOWED_ACTIVE_AXES:
        return {"compatible": False, "reason": "axis_not_supported"}

    active_constraint = scenario.get(axis)
    garment_constraint = scenario.get("garment_category")

    if active_constraint is None or not active_constraint.get("incompatible"):
        return {"compatible": False, "reason": "active_axis_not_constrained"}

    if garment_constraint is None:
        return {"compatible": False, "reason": "garment_constraint_missing"}

    if not garment_constraint.get("compatible") or not garment_constraint.get("incompatible"):
        return {"compatible": False, "reason": "garment_constraint_incomplete"}

    likes, dislikes = _profile_prefs(profile, axis)
    compat_set = set(active_constraint["compatible"])

    liked_compatible = sorted(likes & compat_set)
    disliked_compatible = sorted(dislikes & compat_set)
    neutral_compatible = _neutral_values(active_constraint["compatible"], likes, dislikes)

    garment_likes, garment_dislikes = _profile_prefs(profile, "garment_category")
    compatible_garments = _neutral_values(
        garment_constraint["compatible"], garment_likes, garment_dislikes
    )
    incompatible_garments = _neutral_values(
        garment_constraint["incompatible"], garment_likes, garment_dislikes
    )

    has_A = len(liked_compatible) > 0
    has_B = len(disliked_compatible) > 0 or len(neutral_compatible) > 0
    has_clean_tpo_pair = len(compatible_garments) > 0 and len(incompatible_garments) > 0

    return {
        "compatible": has_A and has_B and has_clean_tpo_pair,
        "liked_compatible": liked_compatible,
        "disliked_compatible": disliked_compatible,
        "neutral_compatible": neutral_compatible,
        "compatible_garments": compatible_garments,
        "incompatible_garments": incompatible_garments,
    }


def build_compatibility_matrix(profiles, scenarios=None):
    if scenarios is None:
        scenarios = CANONICAL_SCENARIOS

    matrix = {}
    stats = {
        "total_slots": 0,
        "compatible_slots": 0,
        "by_axis": defaultdict(lambda: {"total": 0, "compat": 0}),
        "by_archetype": defaultdict(lambda: {"total": 0, "compat": 0}),
        "by_user": defaultdict(lambda: {"total": 0, "compat": 0}),
    }

    for profile in profiles:
        uid = profile["user_id"]
        matrix[uid] = {}

        for sc in scenarios:
            sid = sc["scenario_id"]
            constrained = [ax for ax in get_constrained_axes(sc) if ax in ALLOWED_ACTIVE_AXES]
            matrix[uid][sid] = {}

            for axis in constrained:
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
                        "user_id": uid,
                        "scenario_id": sid,
                        "active_axis": axis,
                        "liked_compatible": result["liked_compatible"],
                        "disliked_compatible": result["disliked_compatible"],
                        "neutral_compatible": result["neutral_compatible"],
                        "compatible_garments": result["compatible_garments"],
                        "incompatible_garments": result["incompatible_garments"],
                    })

    return instances, stats


def print_compatibility_report(stats):
    total = stats["total_slots"]
    compat = stats["compatible_slots"]
    rate = stats["compatibility_rate"]

    print(f"\n  Compatibility: {compat}/{total} = {rate:.1%}")
    print("\n  By axis:")
    for ax, s in stats["by_axis"].items():
        r = s["compat"] / max(s["total"], 1)
        print(f"    {ax:20s}: {s['compat']}/{s['total']} = {r:.1%}")

    print("\n  By archetype:")
    for arch, s in stats["by_archetype"].items():
        r = s["compat"] / max(s["total"], 1)
        print(f"    {arch:25s}: {s['compat']}/{s['total']} = {r:.1%}")

    print("\n  By user:")
    for uid, s in sorted(stats["by_user"].items()):
        r = s["compat"] / max(s["total"], 1)
        print(f"    {uid}: {s['compat']}/{s['total']} = {r:.1%}")