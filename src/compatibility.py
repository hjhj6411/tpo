"""
Compatibility — user × scenario matching.

A (user, scenario, active_axis) triple is compatible iff:
  1. user has ≥1 liked value IN the scenario's compatible set for that axis
  2. user has ≥1 disliked (or non-preferred) value IN the same compatible set
     (so that option B = "non-preferred + TPO-compatible" can be built)
  3. scenario constrains that axis (has non-empty incompatible set)

For pattern axis, "non-preferred" includes values not in likes (neutral).
"""

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.scenarios import CANONICAL_SCENARIOS, get_constrained_axes


def _profile_prefs(profile, axis):
    """Extract (likes_set, dislikes_set) for an axis from a profile."""
    sa = profile.get("structured_attributes", {})
    ax_prefs = sa.get(axis, {})
    return set(ax_prefs.get("likes", [])), set(ax_prefs.get("dislikes", []))


def check_axis_compatibility(profile, scenario, axis):
    """
    Returns:
      {"compatible": bool,
       "liked_compatible": [...],      # user likes ∩ scenario compatible
       "disliked_compatible": [...],   # user dislikes ∩ scenario compatible
       "neutral_compatible": [...],    # (not liked, not disliked) ∩ scenario compatible
       "liked_incompatible": [...],    # user likes ∩ scenario incompatible
       "disliked_incompatible": [...]} # user dislikes ∩ scenario incompatible
    """
    constraint = scenario.get(axis)
    if constraint is None or not constraint.get("incompatible"):
        return {"compatible": False, "reason": "axis_not_constrained"}

    compat_set = set(constraint["compatible"])
    incompat_set = set(constraint["incompatible"])

    likes, dislikes = _profile_prefs(profile, axis)

    liked_compat = sorted(likes & compat_set)
    disliked_compat = sorted(dislikes & compat_set)
    neutral_compat = sorted(compat_set - likes - dislikes)

    liked_incompat = sorted(likes & incompat_set)
    disliked_incompat = sorted(dislikes & incompat_set)

    # A option: need liked + compatible
    has_A = len(liked_compat) > 0
    # B option: need non-preferred + compatible
    # "non-preferred" = disliked OR neutral (not in likes)
    has_B = len(disliked_compat) > 0 or len(neutral_compat) > 0

    return {
        "compatible": has_A and has_B,
        "liked_compatible": liked_compat,
        "disliked_compatible": disliked_compat,
        "neutral_compatible": neutral_compat,
        "liked_incompatible": liked_incompat,
        "disliked_incompatible": disliked_incompat,
    }


def build_compatibility_matrix(profiles, scenarios=None):
    """
    Build full compatibility matrix.

    Returns:
      matrix[user_id][scenario_id] = {
          axis: check_result for each constrained axis
      }

    And summary stats.
    """
    if scenarios is None:
        scenarios = CANONICAL_SCENARIOS

    matrix = {}
    stats = {"total_slots": 0, "compatible_slots": 0,
             "by_axis": defaultdict(lambda: {"total": 0, "compat": 0}),
             "by_archetype": defaultdict(lambda: {"total": 0, "compat": 0}),
             "by_user": defaultdict(lambda: {"total": 0, "compat": 0})}

    for profile in profiles:
        uid = profile["user_id"]
        matrix[uid] = {}

        for sc in scenarios:
            sid = sc["scenario_id"]
            constrained = get_constrained_axes(sc)
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
    """
    Enumerate all compatible (user_id, scenario_id, active_axis) triples.
    Each triple becomes one benchmark instance.
    """
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
                        "liked_incompatible": result["liked_incompatible"],
                        "disliked_incompatible": result["disliked_incompatible"],
                    })

    return instances, stats


def print_compatibility_report(stats):
    total = stats["total_slots"]
    compat = stats["compatible_slots"]
    rate = stats["compatibility_rate"]
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
