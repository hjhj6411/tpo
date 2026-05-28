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
    or use liked_incompatible garment for C, disliked_incompatible for D

v2.1 fix:
  When violation_axis == active_axis (garment_category with no secondary constraint):
  - Option C requires liked_incompatible non-empty (user likes something that is TPO-incompatible)
    → if empty, return None (skip instance; label would be wrong)
  - Option D requires d_val not in user's likes
    → prefer disliked_incompat; if falling back to violation_value, verify it is not liked
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
    # active_axis == garment_category → prefer color if constrained, else pattern, else self
    if scenario.get("color") and scenario["color"].get("incompatible"):
        return "color"
    if scenario.get("pattern") and scenario["pattern"].get("incompatible"):
        return "pattern"
    # No secondary constraint → violation on garment_category itself
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
    """
    Build deterministic 4-option plan from scenario constraints.

    Returns None if a valid 4-option 2×2 structure cannot be constructed.
    """
    scenario = get_scenario_by_id(query["scenario_id"])
    if scenario is None:
        return None

    active_axis = query["active_axis"]
    fixed_attrs = dict(query["fixed_attrs"])
    rng = random.Random(abs(hash(query["query_id"])) % 100000)

    # ── Pick A and B values on the active axis ────────────
    liked_v   = _pick_liked_value(query, rng)
    nonpref_v = _pick_nonpreferred_value(query, rng)
    if liked_v is None or nonpref_v is None:
        return None
    if liked_v == nonpref_v:
        return None

    # ── Determine violation axis and mode ────────────────
    violation_axis = _choose_violation_axis(active_axis, scenario)

    # ── Build C/D depending on whether violation_axis == active_axis ──
    if violation_axis != active_axis:
        # ── Normal case: C/D use different axis for violation ──
        # e.g. active=color, violation=garment_category
        violation_value = _pick_violation_value(query, scenario, violation_axis, rng)
        if violation_value is None:
            return None

        attrs_a = {**fixed_attrs, active_axis: liked_v}
        attrs_b = {**fixed_attrs, active_axis: nonpref_v}
        attrs_c = {**fixed_attrs, active_axis: liked_v,   violation_axis: violation_value}
        attrs_d = {**fixed_attrs, active_axis: nonpref_v, violation_axis: violation_value}

    else:
        # ── Same-axis case: active_axis == violation_axis == garment_category ──
        liked_incompat    = query.get("liked_incompatible", [])
        disliked_incompat = query.get("disliked_incompatible", [])

        if not liked_incompat:
            return None

        c_val = rng.choice(liked_incompat)

        user_likes_set = set(query.get("liked_compatible", []) + liked_incompat)

        if disliked_incompat:
            d_candidates = disliked_incompat
        else:
            sc_incompat = scenario.get(active_axis, {}).get("incompatible", [])
            d_candidates = [v for v in sc_incompat if v not in user_likes_set]

        if not d_candidates:
            return None

        d_candidates = sorted(d_candidates)
        rng.shuffle(d_candidates)
        d_val = d_candidates[0]

        if c_val == d_val:
            remaining = [v for v in d_candidates if v != c_val]
            if not remaining:
                return None
            d_val = remaining[0]

        violation_value = d_val

        attrs_a = {**fixed_attrs, active_axis: liked_v}
        attrs_b = {**fixed_attrs, active_axis: nonpref_v}
        attrs_c = {**fixed_attrs, active_axis: c_val}
        attrs_d = {**fixed_attrs, active_axis: d_val}

    # ── Assemble plan ─────────────────────────────────────
    label_map = {
        "A": "tpo_and_preference",
        "B": "tpo_only",
        "C": "preference_only",
        "D": "neither",
    }
    options = {}
    for k, attrs in [("A", attrs_a), ("B", attrs_b),
                     ("C", attrs_c), ("D", attrs_d)]:
        options[k] = {
            "label":        label_map[k],
            "attributes":   attrs,
            "search_query": attrs_to_search_query(attrs),
            "rationale":    _rationale(k, active_axis, attrs, violation_axis),
        }

    return {
        "query_id":       query["query_id"],
        "user_id":        query["user_id"],
        "scenario_id":    query["scenario_id"],
        "domain":         "fashion",
        "query_type":     query["query_type"],
        "active_axis":    active_axis,
        "fixed_attrs":    fixed_attrs,
        "violation_axis": violation_axis,
        "violation_value": violation_value,
        "main_category":  fixed_attrs.get("garment_category"),
        "options":        options,
    }


def _rationale(k, active_axis, attrs, violation_axis):
    val = attrs.get(active_axis)
    if k == "A":
        return f"liked {active_axis}={val}, TPO-compatible"
    if k == "B":
        return f"non-preferred {active_axis}={val}, TPO-compatible"
    if k == "C":
        vax_val = attrs.get(violation_axis)
        if violation_axis == active_axis:
            return f"liked {active_axis}={val}, but this garment violates TPO"
        return f"liked {active_axis}={val}, but {violation_axis}={vax_val} violates TPO"
    if k == "D":
        vax_val = attrs.get(violation_axis)
        if violation_axis == active_axis:
            return f"non-preferred {active_axis}={val}, AND this garment violates TPO"
        return (f"non-preferred {active_axis}={val}, "
                f"AND {violation_axis}={vax_val} violates TPO")
    return ""


def run_pipeline(profile_path, query_path, output_path, force=False, limit=0):
    log_step("Option Planner v2 (scenario-based deterministic)")
    profiles = {p["user_id"]: p for p in load_jsonl(profile_path)}
    queries  = load_jsonl(query_path)
    print(f"  {len(profiles)} profiles, {len(queries)} queries")

    if output_path.exists() and not force:
        existing = load_jsonl(output_path)
        done  = {p["query_id"] for p in existing}
        plans = existing
        todo  = [q for q in queries if q["query_id"] not in done]
        print(f"  Resuming: {len(done)} already planned")
    else:
        plans = []
        todo  = queries
    if limit > 0:
        todo = todo[:limit]

    n_fail = 0
    n_skip_no_liked_incompat = 0
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
                if (query.get("active_axis") == "garment_category"
                        and not query.get("liked_incompatible")):
                    n_skip_no_liked_incompat += 1
                continue
            plans.append(plan)
            if (i + 1) % 100 == 0:
                save_jsonl(plans, output_path)
        except Exception as e:
            print(f"    ERROR {query['query_id']}: {e}")
            n_fail += 1

    save_jsonl(plans, output_path)
    print(f"\n  ✓ Saved {len(plans)} plans")
    print(f"  Skipped: {n_fail} total "
          f"(of which {n_skip_no_liked_incompat} skipped: "
          f"same-axis, liked_incompat empty)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_path", type=Path,
                        default=PROFILES_DIR / "profiles.jsonl")
    parser.add_argument("--query_path",   type=Path,
                        default=QUERIES_DIR / "queries.jsonl")
    parser.add_argument("--output",       type=Path,
                        default=OPTIONS_DIR / "option_plans.jsonl")
    parser.add_argument("--force",  action="store_true")
    parser.add_argument("--limit",  type=int, default=0)
    args = parser.parse_args()
    run_pipeline(args.profile_path, args.query_path, args.output,
                 args.force, args.limit)


if __name__ == "__main__":
    main()
