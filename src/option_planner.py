"""
Option Planner v2 — Deterministic 2×2 option construction.

Benchmark structure:
  A: preferred   color/pattern + TPO-compatible garment   (tpo_and_preference)
  B: nonpref     color/pattern + TPO-compatible garment   (tpo_only)
  C: preferred   color/pattern + TPO-incompatible garment (preference_only)
  D: nonpref     color/pattern + TPO-incompatible garment (neither)

Rules:
  - active_axis ∈ {color, pattern} ONLY
  - violation_axis = garment_category ALWAYS
  - fixed_attrs[garment_category] = user_liked ∩ TPO-compatible  → used in A/B
  - C/D: same active value as A/B, but garment replaced with TPO-incompatible value
  - A≠C and B≠D guaranteed by different garment values (compat vs incompat)
  - A≠B and C≠D guaranteed by different active_axis values (liked vs nonpref)

No same-axis violation logic. No _choose_violation_axis. Clean 2×2.
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
    """Option A/C: pick a liked value in scenario's compatible set."""
    pool = query["liked_compatible"]
    return rng.choice(pool) if pool else None


def _pick_nonpref_value(query, rng):
    """Option B/D: pick a non-preferred value in scenario's compatible set.
    Prefer disliked; fall back to neutral."""
    disliked = query["disliked_compatible"]
    neutral  = query["neutral_compatible"]
    if disliked:
        return rng.choice(disliked)
    if neutral:
        return rng.choice(neutral)
    return None


def _pick_incompat_garment(query, fixed_garment, rng):
    """
    Pick a TPO-incompatible garment for C/D.
    Must differ from the fixed (TPO-compatible) garment used in A/B.
    """
    pool = [g for g in query["incompat_garments"] if g != fixed_garment]
    return rng.choice(pool) if pool else None


def attrs_to_search_query(attrs):
    parts = []
    pattern = attrs.get("pattern")
    color   = attrs.get("color")
    garment = attrs.get("garment_category", "item")
    if pattern and pattern != "solid":
        parts.append(pattern.replace("_", " "))
    if color:
        parts.append(color)
    parts.append(garment.replace("_", " "))
    return " ".join(parts)


def plan_options_for_query(query):
    """
    Build deterministic 2×2 option plan.
    Returns None if a valid plan cannot be constructed.
    """
    active_axis  = query["active_axis"]
    fixed_attrs  = dict(query["fixed_attrs"])   # garment + other non-active axis
    fixed_garment = fixed_attrs["garment_category"]

    rng = random.Random(abs(hash(query["query_id"])) % 100000)

    # ── Active axis values ────────────────────────────────
    liked_v   = _pick_liked_value(query, rng)
    nonpref_v = _pick_nonpref_value(query, rng)
    if liked_v is None or nonpref_v is None:
        return None
    if liked_v == nonpref_v:
        return None

    # ── Violation garment for C/D ────────────────────────
    incompat_garment = _pick_incompat_garment(query, fixed_garment, rng)
    if incompat_garment is None:
        return None

    # ── Build 4 options ──────────────────────────────────
    # A/B: fixed garment (TPO-compatible)
    # C/D: incompat garment (TPO-incompatible)
    attrs_a = {**fixed_attrs,                                    active_axis: liked_v}
    attrs_b = {**fixed_attrs,                                    active_axis: nonpref_v}
    attrs_c = {**fixed_attrs, "garment_category": incompat_garment, active_axis: liked_v}
    attrs_d = {**fixed_attrs, "garment_category": incompat_garment, active_axis: nonpref_v}

    label_map = {
        "A": "tpo_and_preference",
        "B": "tpo_only",
        "C": "preference_only",
        "D": "neither",
    }

    options = {}
    for key, attrs in [("A", attrs_a), ("B", attrs_b), ("C", attrs_c), ("D", attrs_d)]:
        options[key] = {
            "label":        label_map[key],
            "attributes":   attrs,
            "search_query": attrs_to_search_query(attrs),
            "rationale":    _rationale(key, active_axis, liked_v, nonpref_v,
                                       fixed_garment, incompat_garment, attrs),
        }

    return {
        "query_id":         query["query_id"],
        "user_id":          query["user_id"],
        "scenario_id":      query["scenario_id"],
        "domain":           "fashion",
        "query_type":       query["query_type"],
        "active_axis":      active_axis,
        "violation_axis":   "garment_category",
        "fixed_attrs":      fixed_attrs,
        "liked_value":      liked_v,
        "nonpref_value":    nonpref_v,
        "compat_garment":   fixed_garment,
        "incompat_garment": incompat_garment,
        "options":          options,
    }


def _rationale(key, active_axis, liked_v, nonpref_v, compat_g, incompat_g, attrs):
    active_val = attrs.get(active_axis)
    garment    = attrs.get("garment_category")
    if key == "A":
        return (f"preferred {active_axis}={liked_v}, "
                f"TPO-compatible garment={compat_g}")
    if key == "B":
        return (f"non-preferred {active_axis}={nonpref_v}, "
                f"TPO-compatible garment={compat_g}")
    if key == "C":
        return (f"preferred {active_axis}={liked_v}, "
                f"TPO-incompatible garment={incompat_g}")
    if key == "D":
        return (f"non-preferred {active_axis}={nonpref_v}, "
                f"TPO-incompatible garment={incompat_g}")
    return ""


def run_pipeline(profile_path, query_path, output_path, force=False, limit=0):
    log_step("Option Planner v2 (2x2 deterministic)")
    queries = load_jsonl(query_path)
    print(f"  {len(queries)} queries")

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
    for i, query in enumerate(todo):
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(todo)}] planning...")
        try:
            plan = plan_options_for_query(query)
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
    print(f"  Skipped: {n_fail}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_path", type=Path, default=PROFILES_DIR / "profiles.jsonl")
    parser.add_argument("--query_path",   type=Path, default=QUERIES_DIR  / "queries.jsonl")
    parser.add_argument("--output",       type=Path, default=OPTIONS_DIR  / "option_plans.jsonl")
    parser.add_argument("--force",  action="store_true")
    parser.add_argument("--limit",  type=int, default=0)
    args = parser.parse_args()
    run_pipeline(args.profile_path, args.query_path, args.output,
                 args.force, args.limit)


if __name__ == "__main__":
    main()
