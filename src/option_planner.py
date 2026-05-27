"""
Option Planner v2 — Scenario-based deterministic 4-option construction.

Given (profile, query) with scenario and active_axis:
  A = liked active value + fixed attrs               (tpo_and_preference)
  B = non-preferred active value + fixed attrs        (tpo_only)
  C = liked active value + TPO-violated fixed attr    (preference_only)
  D = non-preferred active value + TPO-violated fixed (neither)

TPO violation:
  - For active_axis ∈ {color, pattern}: swap garment_category to scenario incompatible
  - For active_axis = garment_category: swap color to scenario incompatible (if constrained)
    or use a neutral fallback violation garment
"""

import argparse
import random
import sys
from pathlib import Path

from .utils import save_jsonl, load_jsonl, log_step

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import OPTIONS_DIR, PROFILES_DIR, QUERIES_DIR
from configs.scenarios import get_scenario_by_id


def _pick_liked_value(query, rng):
    """Pick a liked value that is in scenario's compatible set."""
    liked_compat = query["liked_compatible"]
    if liked_compat:
        return rng.choice(liked_compat)
    return None


def _pick_nonpreferred_value(query, rng):
    """Pick a non-preferred value in scenario's compatible set.
    Prefer disliked; fall back to neutral."""
    disliked_compat = query["disliked_compatible"]
    neutral_compat = query["neutral_compatible"]
    if disliked_compat:
        return rng.choice(disliked_compat)
    if neutral_compat:
        return rng.choice(neutral_compat)
    return None


def _pick_violation_value(query, scenario, violation_axis, rng):
    """Pick a value for violation_axis from the scenario's incompatible set."""
    constraint = scenario.get(violation_axis)
    if constraint and constraint.get("incompatible"):
        return rng.choice(constraint["incompatible"])
    # Fallback: for garment_category, use generally informal items
    if violation_axis == "garment_category":
        # These are almost always TPO-incompatible in formal/cold scenarios
        fallbacks = ["shorts", "tank_top", "t_shirt"]
        return rng.choice(fallbacks)
    if violation_axis == "color":
        fallbacks = ["orange", "yellow"]
        return rng.choice(fallbacks)
    if violation_axis == "pattern":
        fallbacks = ["graphic_print", "camouflage"]
        return rng.choice(fallbacks)
    return None


def _choose_violation_axis(active_axis, scenario):
    """Decide which axis to violate for C/D options."""
    if active_axis in ("color", "pattern"):
        return "garment_category"
    # active_axis == garment_category → prefer color if constrained, else garment fallback
    if scenario.get("color") and scenario["color"].get("incompatible"):
        return "color"
    if scenario.get("pattern") and scenario["pattern"].get("incompatible"):
        return "pattern"
    # No secondary constraint → use garment_category itself with incompatible items
    return "garment_category"


def attrs_to_search_query(attrs):
    parts = []
    pattern = attrs.get("pattern")
    color = attrs.get("color")
    garment = attrs.get("garment_category", "item")
    if pattern and pattern != "solid":
        parts.append(pattern.replace("_", " "))
    if color:
        parts.append(color)
    parts.append(garment.replace("_", " "))
    return " ".join(parts)


def plan_options_for_query(profile, query):
    """Build deterministic 4-option plan from scenario constraints."""
    scenario = get_scenario_by_id(query["scenario_id"])
    if scenario is None:
        return None

    active_axis = query["active_axis"]
    fixed_attrs = dict(query["fixed_attrs"])
    rng = random.Random(abs(hash(query["query_id"])) % 100000)

    # Pick A and B values on the active axis
    liked_v = _pick_liked_value(query, rng)
    nonpref_v = _pick_nonpreferred_value(query, rng)
    if liked_v is None or nonpref_v is None:
        return None
    if liked_v == nonpref_v:
        return None

    # Determine violation axis and violation value
    violation_axis = _choose_violation_axis(active_axis, scenario)
    violation_value = _pick_violation_value(query, scenario, violation_axis, rng)
    if violation_value is None:
        return None

    # Build A/B/C/D attribute dicts
    attrs_a = {**fixed_attrs, active_axis: liked_v}
    attrs_b = {**fixed_attrs, active_axis: nonpref_v}

    if violation_axis == active_axis:
        # For garment_category with no secondary color/pattern constraint:
        # C/D use liked/nonpref garment but itself is from incompatible list
        # This shouldn't happen normally — but handle gracefully
        liked_incompat = query.get("liked_incompatible", [])
        disliked_incompat = query.get("disliked_incompatible", [])
        c_val = rng.choice(liked_incompat) if liked_incompat else violation_value
        d_val = rng.choice(disliked_incompat) if disliked_incompat else violation_value
        attrs_c = {**fixed_attrs, active_axis: c_val}
        attrs_d = {**fixed_attrs, active_axis: d_val}
    else:
        attrs_c = {**fixed_attrs, active_axis: liked_v,
                   violation_axis: violation_value}
        attrs_d = {**fixed_attrs, active_axis: nonpref_v,
                   violation_axis: violation_value}

    label_map = {"A": "tpo_and_preference", "B": "tpo_only",
                 "C": "preference_only",     "D": "neither"}
    options = {}
    for k, attrs in [("A", attrs_a), ("B", attrs_b),
                     ("C", attrs_c), ("D", attrs_d)]:
        options[k] = {
            "label": label_map[k],
            "attributes": attrs,
            "search_query": attrs_to_search_query(attrs),
            "rationale": _rationale(k, active_axis, attrs, violation_axis),
        }

    return {
        "query_id": query["query_id"],
        "user_id": query["user_id"],
        "scenario_id": query["scenario_id"],
        "domain": "fashion",
        "query_type": query["query_type"],
        "active_axis": active_axis,
        "fixed_attrs": fixed_attrs,
        "violation_axis": violation_axis,
        "violation_value": violation_value,
        "main_category": fixed_attrs.get("garment_category"),
        "options": options,
    }


def _rationale(k, active_axis, attrs, violation_axis):
    val = attrs.get(active_axis)
    if k == "A":
        return f"liked {active_axis}={val}, TPO-compatible"
    if k == "B":
        return f"non-preferred {active_axis}={val}, TPO-compatible"
    if k == "C":
        return (f"liked {active_axis}={val}, but {violation_axis}="
                f"{attrs.get(violation_axis)} violates TPO")
    if k == "D":
        return (f"non-preferred {active_axis}={val}, AND {violation_axis}="
                f"{attrs.get(violation_axis)} violates TPO")
    return ""


def run_pipeline(profile_path, query_path, output_path, force=False, limit=0):
    log_step("Option Planner v2 (scenario-based deterministic)")
    profiles = {p["user_id"]: p for p in load_jsonl(profile_path)}
    queries = load_jsonl(query_path)
    print(f"  {len(profiles)} profiles, {len(queries)} queries")

    if output_path.exists() and not force:
        existing = load_jsonl(output_path)
        done = {p["query_id"] for p in existing}
        plans = existing
        todo = [q for q in queries if q["query_id"] not in done]
        print(f"  Resuming: {len(done)} already planned")
    else:
        plans = []
        todo = queries
    if limit > 0:
        todo = todo[:limit]

    n_fail = 0
    for i, query in enumerate(todo):
        if query["user_id"] not in profiles:
            continue
        prof = profiles[query["user_id"]]
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(todo)}] planning...")
        try:
            plan = plan_options_for_query(prof, query)
            if plan is None:
                n_fail += 1
                continue
            plans.append(plan)
            if (i + 1) % 100 == 0:
                save_jsonl(plans, output_path)
        except Exception as e:
            print(f"    ERROR {query['query_id']}: {e}")
            n_fail += 1

    save_jsonl(plans, output_path)
    print(f"\n  ✓ Saved {len(plans)} plans, {n_fail} failed")


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
