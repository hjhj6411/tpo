"""
Compatibility — clean 2×2 version (RELAXED so all 15 archetypes contribute).

Only color/pattern can be active axes.
Garment category is always the TPO axis and must be preference-neutral.

CHANGE vs the original (and WHY)
--------------------------------
Original behaviour: a scenario could only host an active axis (color/pattern) if
that axis was itself TPO-constrained (had a non-empty `incompatible` set). That
silently excluded every PHYSICAL archetype (extreme_cold, extreme_heat,
athletic_*, aquatic, rugged, severe_weather, casual_leisure) — none constrain
color/pattern, so they produced 0 instances. That is the entire reason adding
scenarios did not increase the count.

Relaxed behaviour: color/pattern can be the active axis for ANY scenario whose
GARMENT is fully constrained (every scenario). When the active axis is NOT
TPO-constrained, all of its values are situation-appropriate, so:
  - the active axis is a *pure preference* probe (liked vs disliked), and
  - the garment alone carries TPO (compatible vs incompatible),
which is the cleanest possible 2×2 instance.

When the active axis IS TPO-constrained (dress-coded archetypes), the original
behaviour is preserved exactly: A/B values are restricted to the scenario's
compatible set so that B is "disliked but still TPO-OK", never "neither".

Strict preference contrast: B/D require a profile-disliked active-axis value.
If the profile's disliked value is not TPO-compatible for this scenario, that
(user, scenario, axis) slot is skipped instead of falling back to neutral.
"""

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import FASHION_ATTRIBUTE_AXES
from configs.scenarios import CANONICAL_SCENARIOS

ALLOWED_ACTIVE_AXES = {"color", "pattern"}


def _profile_prefs(profile, axis):
    sa = profile.get("structured_attributes", {})
    ax_prefs = sa.get(axis, {})
    return set(ax_prefs.get("likes", [])), set(ax_prefs.get("dislikes", []))


def _neutral_values(values, likes, dislikes):
    return sorted(set(values) - likes - dislikes)


def _active_compatible_values(scenario, axis):
    """The set of axis values that are TPO-appropriate in this scenario.

    If the scenario constrains the axis -> its declared `compatible` set.
    Otherwise -> the full attribute vocabulary (axis carries no TPO meaning).
    Returns (compatible_value_list, axis_is_tpo_constrained: bool).
    """
    constraint = scenario.get(axis)
    if constraint is not None and constraint.get("incompatible"):
        return list(constraint.get("compatible", [])), True
    # unconstrained active axis: every value is situation-appropriate
    return list(FASHION_ATTRIBUTE_AXES[axis]), False


def check_axis_compatibility(profile, scenario, axis):
    if axis not in ALLOWED_ACTIVE_AXES:
        return {"compatible": False, "reason": "axis_not_supported"}

    garment_constraint = scenario.get("garment_category")
    if garment_constraint is None:
        return {"compatible": False, "reason": "garment_constraint_missing"}
    if not garment_constraint.get("compatible") or not garment_constraint.get("incompatible"):
        return {"compatible": False, "reason": "garment_constraint_incomplete"}

    compat_values, axis_tpo_constrained = _active_compatible_values(scenario, axis)
    compat_set = set(compat_values)

    likes, dislikes = _profile_prefs(profile, axis)
    liked_compatible = sorted(likes & compat_set)
    disliked_compatible = sorted(dislikes & compat_set)
    neutral_compatible = _neutral_values(compat_values, likes, dislikes)

    garment_likes, garment_dislikes = _profile_prefs(profile, "garment_category")
    compatible_garments = _neutral_values(
        garment_constraint["compatible"], garment_likes, garment_dislikes)
    incompatible_garments = _neutral_values(
        garment_constraint["incompatible"], garment_likes, garment_dislikes)

    has_A = len(liked_compatible) > 0
    has_B = len(disliked_compatible) > 0
    has_clean_tpo_pair = len(compatible_garments) > 0 and len(incompatible_garments) > 0

    return {
        "compatible": has_A and has_B and has_clean_tpo_pair,
        "axis_tpo_constrained": axis_tpo_constrained,
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
            gc = sc.get("garment_category")
            if gc is None or not gc.get("compatible") or not gc.get("incompatible"):
                # scenario can't host the TPO axis at all
                continue

            matrix[uid][sid] = {}
            # RELAXED: every garment-constrained scenario can host color & pattern
            for axis in ("color", "pattern"):
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
        stats["compatible_slots"] / max(stats["total_slots"], 1))
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
                        "axis_tpo_constrained": result.get("axis_tpo_constrained", False),
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
