"""
Option Planner — clean 2x2 with GLOBAL COUNTERBALANCING.

Drop-in replacement for src/option_planner.py.

Difference from the original:
  The original picked A's value (liked) and B's value (non-preferred) with a
  per-query RNG. Because most users' liked-compatible set was {solid}/{a few dark
  colors}, A's value distribution was skewed (liked_rate(solid)=0.90), letting a
  preference-blind model exploit a value prior (~82–87% on pattern).

  This version assigns A/B values in a SECOND, GLOBAL pass that drives every
  value's (#used-as-A − #used-as-B) toward zero per axis. Result: each value
  appears as the "liked/correct" option about as often as the "non-preferred"
  option, so the value-frequency prior carries ~0 signal (blind exploit ~0.50).
  Garment selection and the 2x2 structure are unchanged.

Also adds `scenario_archetype` and `scenario_name` to each plan record so the
downstream per-archetype analysis (quality_audit / evaluator) is not null.

Usage (same CLI as before):
  python -m src.option_planner --profile_path ... --query_path ... --output ...
"""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

from .utils import save_jsonl, load_jsonl, log_step

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import OPTIONS_DIR, PROFILES_DIR, QUERIES_DIR
from configs.scenarios import get_scenario_by_id

ALLOWED_ACTIVE_AXES = {"color", "pattern"}


# ── value pools per query ──────────────────────────────────

def _liked_pool(query):
    return list(query.get("liked_compatible", []))


def _nonpref_pool(query):
    # non-preferred = disliked first, then neutral (matches original B semantics)
    return list(query.get("disliked_compatible", [])) + list(query.get("neutral_compatible", []))


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
    g = attrs.get("garment_category")
    return {
        "A": f"preferred {active_axis}={val}, TPO-compatible garment={g}",
        "B": f"non-preferred {active_axis}={val}, TPO-compatible garment={g}",
        "C": f"preferred {active_axis}={val}, but garment={g} violates TPO",
        "D": f"non-preferred {active_axis}={val}, and garment={g} violates TPO",
    }.get(k, "")


# ── global counterbalanced (A,B) value assignment ─────────

def assign_ab_values(queries, seed=42):
    """Choose (a_val, b_val) for every query so that, within each active axis,
    every value's usage as A vs B is balanced. Greedy, deterministic given seed.

    Returns: dict query_id -> (a_val, b_val); queries with no valid pair omitted.
    """
    rng = random.Random(seed)
    assignment = {}
    # process per axis so balancing is within-axis
    by_axis = {}
    for q in queries:
        if q["active_axis"] in ALLOWED_ACTIVE_AXES:
            by_axis.setdefault(q["active_axis"], []).append(q)

    for axis, qs in by_axis.items():
        net = Counter()  # value -> (#A - #B) so far
        # order: fewest degrees of freedom first (small liked/nonpref pools),
        # so constrained queries are placed before the easy ones.
        def dof(q):
            return (len(set(_liked_pool(q))), len(set(_nonpref_pool(q))))
        for q in sorted(qs, key=lambda q: (dof(q), q["query_id"])):
            liked = list(dict.fromkeys(_liked_pool(q)))
            nonpref = [v for v in dict.fromkeys(_nonpref_pool(q)) if v not in set(liked)]
            if not liked or not nonpref:
                continue
            # A = liked value most over-represented as B so far (net most negative)
            a = min(liked, key=lambda v: (net[v], rng.random()))
            cand = [v for v in nonpref if v != a] or nonpref
            # B = non-preferred value most over-represented as A so far (net most positive)
            b = max(cand, key=lambda v: (net[v], rng.random()))
            if a == b:
                continue
            net[a] += 1
            net[b] -= 1
            assignment[q["query_id"]] = (a, b)
    return assignment


# ── per-query plan build (uses pre-assigned a/b) ──────────

def plan_options_for_query(profile, query, ab_values):
    scenario = get_scenario_by_id(query["scenario_id"])
    if scenario is None:
        return None
    active_axis = query["active_axis"]
    if active_axis not in ALLOWED_ACTIVE_AXES:
        return None

    ab = ab_values.get(query["query_id"])
    if ab is None:
        return None
    liked_v, nonpref_v = ab
    if not liked_v or not nonpref_v or liked_v == nonpref_v:
        return None

    rng = random.Random(abs(hash(query["query_id"])) % 100000)
    compat_garment = _pick_garment(query, "compatible_garments", rng)
    incompat_garment = _pick_garment(query, "incompatible_garments", rng)
    if not compat_garment or not incompat_garment or compat_garment == incompat_garment:
        return None

    fixed_attrs = dict(query.get("fixed_attrs", {}))
    fixed_attrs.pop("garment_category", None)

    attrs_a = {**fixed_attrs, active_axis: liked_v, "garment_category": compat_garment}
    attrs_b = {**fixed_attrs, active_axis: nonpref_v, "garment_category": compat_garment}
    attrs_c = {**fixed_attrs, active_axis: liked_v, "garment_category": incompat_garment}
    attrs_d = {**fixed_attrs, active_axis: nonpref_v, "garment_category": incompat_garment}

    label_map = {"A": "tpo_and_preference", "B": "tpo_only",
                 "C": "preference_only", "D": "neither"}
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
        # propagate archetype/name so downstream per-archetype analysis is populated
        "scenario_archetype": scenario.get("archetype"),
        "scenario_name": scenario.get("name"),
        "domain": "fashion",
        "query_type": query["query_type"],
        "query_text": query.get("query_text", ""),
        "active_axis": active_axis,
        "fixed_attrs": fixed_attrs,
        "violation_axis": "garment_category",
        "violation_value": incompat_garment,
        "main_category": compat_garment,
        "options": options,
    }


def run_pipeline(profile_path, query_path, output_path, force=False, limit=0, seed=42):
    log_step("Option Planner (clean 2x2 + global counterbalancing)")
    profiles = {p["user_id"]: p for p in load_jsonl(profile_path)}
    queries = load_jsonl(query_path)
    print(f"  {len(profiles)} profiles, {len(queries)} queries")

    # GLOBAL pass: balanced (A,B) value assignment across the whole query set.
    # (Always computed over ALL queries so balance holds even with --limit.)
    ab_values = assign_ab_values(queries, seed=seed)
    print(f"  Counterbalanced A/B value assignment for {len(ab_values)} queries")

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
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(todo)}] planning...")
        try:
            plan = plan_options_for_query(prof, query, ab_values)
            if plan is None:
                n_fail += 1
                continue
            plans.append(plan)
        except Exception as e:
            print(f"    ERROR {query['query_id']}: {e}")
            n_fail += 1

    save_jsonl(plans, output_path)
    print(f"\n  ✓ Saved {len(plans)} plans   (skipped {n_fail})")

    # quick self-report of residual A/B value bias
    _report_balance(plans)


def _report_balance(plans):
    from collections import defaultdict
    fa = defaultdict(Counter)
    fb = defaultdict(Counter)
    for p in plans:
        ax = p["active_axis"]
        fa[ax][p["options"]["A"]["attributes"].get(ax)] += 1
        fb[ax][p["options"]["B"]["attributes"].get(ax)] += 1
    for ax in fa:
        lr = {v: (fa[ax][v] / (fa[ax][v] + fb[ax][v]) if (fa[ax][v] + fb[ax][v]) else .5)
              for v in set(fa[ax]) | set(fb[ax])}
        # blind exploit upper bound = pick higher-liked-rate value
        # (here approximated by reporting max skew)
        worst = max(lr.items(), key=lambda kv: abs(kv[1] - 0.5)) if lr else (None, .5)
        print(f"  [{ax}] residual max value-skew: {worst[0]}={worst[1]:.2f} "
              f"(0.50 == confound-free)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_path", type=Path, default=PROFILES_DIR / "profiles.jsonl")
    parser.add_argument("--query_path", type=Path, default=QUERIES_DIR / "queries.jsonl")
    parser.add_argument("--output", type=Path, default=OPTIONS_DIR / "option_plans.jsonl")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_pipeline(args.profile_path, args.query_path, args.output,
                 args.force, args.limit, args.seed)


if __name__ == "__main__":
    main()
