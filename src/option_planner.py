"""
Option Planner — clean 2×2 version.

Only color/pattern can be active axes.
Garment category is always the sole TPO violation axis.
All garments used in A/B/C/D are preference-neutral w.r.t. the user.
"""

import argparse
import random
import sys
from pathlib import Path

from .utils import save_jsonl, load_jsonl, log_step

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import OPTIONS_DIR, PROFILES_DIR, QUERIES_DIR
from configs.scenarios import get_scenario_by_id

ALLOWED_ACTIVE_AXES = {"color", "pattern"}


def _pick_liked_value(query, rng):
    vals = query.get("liked_compatible", [])
    return rng.choice(vals) if vals else None


def _pick_nonpreferred_value(query, rng):
    disliked = query.get("disliked_compatible", [])
    neutral = query.get("neutral_compatible", [])
    if disliked:
        return rng.choice(disliked)
    if neutral:
        return rng.choice(neutral)
    return None


def _pick_garment(query, field, rng):
    vals = query.get(field, [])
    return rng.choice(vals) if vals else None


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


def _rationale(k, active_axis, attrs):
    val = attrs.get(active_axis)
    garment = attrs.get("garment_category")

    if k == "A":
        return f"preferred {active_axis}={val}, TPO-compatible garment={garment}"
    if k == "B":
        return f"non-preferred {active_axis}={val}, TPO-compatible garment={garment}"
    if k == "C":
        return f"preferred {active_axis}={val}, but garment={garment} violates TPO"
    if k == "D":
        return f"non-preferred {active_axis}={val}, and garment={garment} violates TPO"
    return ""


def plan_options_for_query(profile, query):
    scenario = get_scenario_by_id(query["scenario_id"])
    if scenario is None:
        return None

    active_axis = query["active_axis"]
    if active_axis not in ALLOWED_ACTIVE_AXES:
        return None

    rng = random.Random(abs(hash(query["query_id"])) % 100000)

    liked_v = _pick_liked_value(query, rng)
    nonpref_v = _pick_nonpreferred_value(query, rng)
    compat_garment = _pick_garment(query, "compatible_garments", rng)
    incompat_garment = _pick_garment(query, "incompatible_garments", rng)

    if not liked_v or not nonpref_v:
        return None
    if liked_v == nonpref_v:
        return None
    if not compat_garment or not incompat_garment:
        return None
    if compat_garment == incompat_garment:
        return None

    fixed_attrs = dict(query.get("fixed_attrs", {}))
    fixed_attrs.pop("garment_category", None)

    attrs_a = {**fixed_attrs, active_axis: liked_v, "garment_category": compat_garment}
    attrs_b = {**fixed_attrs, active_axis: nonpref_v, "garment_category": compat_garment}
    attrs_c = {**fixed_attrs, active_axis: liked_v, "garment_category": incompat_garment}
    attrs_d = {**fixed_attrs, active_axis: nonpref_v, "garment_category": incompat_garment}

    label_map = {
        "A": "tpo_and_preference",
        "B": "tpo_only",
        "C": "preference_only",
        "D": "neither",
    }

    options = {}
    for k, attrs in [("A", attrs_a), ("B", attrs_b), ("C", attrs_c), ("D", attrs_d)]:
        options[k] = {
            "label": label_map[k],
            "attributes": attrs,
            "search_query": attrs_to_search_query(attrs),
            "rationale": _rationale(k, active_axis, attrs),
        }

    return {
        "query_id": query["query_id"],
        "user_id": query["user_id"],
        "scenario_id": query["scenario_id"],
        "domain": "fashion",
        "query_type": query["query_type"],
        "active_axis": active_axis,
        "fixed_attrs": fixed_attrs,
        "violation_axis": "garment_category",
        "violation_value": incompat_garment,
        "main_category": compat_garment,
        "options": options,
    }


def run_pipeline(profile_path, query_path, output_path, force=False, limit=0):
    log_step("Option Planner (clean 2×2)")
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
    print(f"\n  ✓ Saved {len(plans)} plans")
    print(f"  Skipped: {n_fail} total")


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