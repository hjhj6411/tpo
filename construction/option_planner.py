"""
STAGE 3 — Option Planner (clean 2×2 with strict disliked B/D values).

For every query, emits the 4-option (A/B/C/D) *specification* (attributes +
search_query), not images. The 2×2:

    active_axis value:      A,C = liked        B,D = disliked
    violation_axis (TPO):   A,B = compatible   C,D = incompatible

    A tpo_and_preference | B tpo_only | C preference_only | D neither

Violation axis: the axis carrying the TPO contrast follows the axes the
scenario actually constrains. Physical scenarios constrain only garment, so
C/D always violate via a TPO-incompatible garment. Dress-coded scenarios also
constrain color/pattern; there the violation rotates between garment and each
feasible constrained axis (balanced per scenario), with C/D taking a
preference-NEUTRAL TPO-incompatible value on that axis while the garment stays
compatible-neutral and shared across all four options. Slots where no
color/pattern violation is preference-neutral fall back to garment, so
coverage never shrinks.

Updated for the cleaned garment/pattern vocabulary:
- formal_shirt renders as "formal shirt"
- leopard renders as "leopard print" in retrieval queries
- polka_dot renders as "polka dot"
- search queries use the cleaned canonical vocabulary and retrieval-friendly aliases

Confusability-aware assignment:
- (A, B) active-axis values, (compat, incompat) garment pairs, and violation
  value pairs avoid visually-confusable pairs (e.g. black/navy, jeans/slacks)
  via a soft penalty whenever the profile/scenario pools offer an alternative.

Balance priorities (per assignment): confusable-pair avoidance first, then
PER-USER variety (least-used values for that user), then global A/B
counterbalance as the tie-breaker — global balance must never force one user
onto a single value.
"""

import argparse
import hashlib
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .utils import save_jsonl, load_jsonl, log_step

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import OPTIONS_DIR, PROFILES_DIR, QUERIES_DIR
from configs.scenarios import get_scenario_by_id
from .compatibility import is_pattern_garment_compatible

ALLOWED_ACTIVE_AXES = {"color", "pattern"}

# Visually-confusable value pairs. A/B (and compat/incompat garments) differing
# only by such a pair are hard to separate in CLIP retrieval and VLM judging,
# so they carry a large soft penalty. Penalty (not exclusion): dress-coded
# scenarios whose compatible colors are all dark neutrals must still emit a
# pair, and no (user, scenario, axis) slot should become a new coverage hole.
CONFUSABLE_PAIR_PENALTY = 10_000
CONFUSABLE_ACTIVE_PAIRS = {
    "color": {frozenset(p) for p in [
        ("black", "navy"), ("black", "gray"), ("gray", "navy"),
        ("gray", "white"), ("navy", "blue"), ("navy", "purple"),
        ("white", "beige"), ("brown", "beige"),
        ("red", "orange"), ("red", "pink"),
    ]},
    "pattern": set(),
}
CONFUSABLE_GARMENT_PAIRS = {frozenset(p) for p in [
    ("jeans", "slacks"), ("leggings", "slacks"),
    ("fleece_jacket", "sweater"), ("fleece_jacket", "hoodie"),
    ("sweater", "sweatshirt"), ("hoodie", "sweatshirt"),
    ("sweater", "cardigan"), ("windbreaker", "puffer_jacket"),
    ("dress", "long_skirt"), ("mini_skirt", "shorts"),
]}

PATTERN_QUERY_ALIAS = {
    "solid": "solid",
    "striped": "striped",
    "checkered": "checkered",
    "floral": "floral print",
    "polka_dot": "polka dot",
    "leopard": "leopard print",
}

GARMENT_QUERY_ALIAS = {
    "t_shirt": "t-shirt",
    "long_sleeve_t_shirt": "long-sleeve t-shirt",
    "tank_top": "tank top",
    "formal_shirt": "formal shirt",
    "sweatshirt": "sweatshirt",
    "sweater": "sweater",
    "hoodie": "hoodie",
    "cardigan": "cardigan",
    "blazer": "blazer",
    "windbreaker": "windbreaker",
    "leather_jacket": "leather jacket",
    "puffer_jacket": "puffer jacket",
    "fleece_jacket": "fleece jacket",
    "trench_coat": "trench coat",
    "jeans": "jeans",
    "slacks": "slacks",
    "shorts": "shorts",
    "leggings": "leggings",
    "dress": "dress",
    "mini_skirt": "mini skirt",
    "long_skirt": "long skirt",
}


def _stable_seed(s: str) -> int:
    """Process-independent seed from a string."""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


def _render_value(axis: str, value: str) -> str:
    if axis == "pattern":
        return PATTERN_QUERY_ALIAS.get(value, value.replace("_", " "))
    if axis == "garment_category":
        return GARMENT_QUERY_ALIAS.get(value, value.replace("_", " "))
    return value.replace("_", " ")


# ── value pools per query ──────────────────────────────────

def _liked_pool(query):
    return list(query.get("liked_compatible", []))


def _disliked_pool(query):
    return list(query.get("disliked_compatible", []))


def _pick_garment(query, field, rng, patterns=()):
    vals = [
        garment for garment in query.get(field, [])
        if all(is_pattern_garment_compatible(pattern, garment) for pattern in patterns)
    ]
    return rng.choice(vals) if vals else None


def _option_patterns(query, ab_values):
    if query.get("active_axis") == "pattern":
        return {value for value in ab_values.get(query["query_id"], ()) if value}
    pattern = (query.get("fixed_attrs") or {}).get("pattern")
    return {pattern} if pattern else set()


def attrs_to_search_query(attrs):
    parts = []
    pattern = attrs.get("pattern")
    color = attrs.get("color")
    garment = attrs.get("garment_category", "item")

    # Do not add "solid" as a retrieval adjective; it often hurts recall.
    if pattern and pattern != "solid":
        parts.append(_render_value("pattern", pattern))
    if color:
        parts.append(_render_value("color", color))
    parts.append(_render_value("garment_category", garment))
    return " ".join(parts)


def _rationale(k, active_axis, attrs, violation_axis="garment_category"):
    val = _render_value(active_axis, attrs.get(active_axis, ""))
    vname = "garment" if violation_axis == "garment_category" else violation_axis
    vv = _render_value(violation_axis, attrs.get(violation_axis, ""))
    return {
        "A": f"preferred {active_axis}={val}, TPO-compatible {vname}={vv}",
        "B": f"disliked {active_axis}={val}, TPO-compatible {vname}={vv}",
        "C": f"preferred {active_axis}={val}, but {vname}={vv} violates TPO",
        "D": f"disliked {active_axis}={val}, and {vname}={vv} violates TPO",
    }.get(k, "")


# ── global counterbalanced (A,B) value assignment ─────────

def assign_ab_values(queries, seed=42):
    """Choose liked/disliked (a_val, b_val) per query.

    Lexicographic priority: avoid confusable pairs, then PER-USER variety
    (least-used A and B values for this user), then global A/B counterbalance
    as the tie-breaker. Global balance intentionally comes last: with net
    (#A - #B) as the leading term, values many users dislike (large negative
    net) get force-assigned as A to every user who likes them, collapsing
    per-user diversity (observed: users with 59-60/60 identical A values).
    B/D never use neutral values.

    Returns: dict query_id -> (a_val, b_val); queries with no valid pair omitted.
    """
    rng = random.Random(seed)
    assignment = {}
    by_axis = {}
    for q in queries:
        if q["active_axis"] in ALLOWED_ACTIVE_AXES:
            by_axis.setdefault(q["active_axis"], []).append(q)

    for axis, qs in by_axis.items():
        net = Counter()  # value -> (#A - #B) so far
        user_a = Counter()
        user_b = Counter()

        def dof(q):
            # fewest degrees of freedom first → place constrained queries before easy ones
            return (len(set(_liked_pool(q))), len(set(_disliked_pool(q))))

        confusable = CONFUSABLE_ACTIVE_PAIRS.get(axis, set())

        for q in sorted(qs, key=lambda q: (dof(q), q["query_id"])):
            uid = q.get("user_id", "?")
            liked = list(dict.fromkeys(_liked_pool(q)))
            disliked = [v for v in dict.fromkeys(_disliked_pool(q)) if v not in set(liked)]
            if not liked or not disliked:
                continue
            best = None
            for av in liked:
                for bv in disliked:
                    if av == bv:
                        continue
                    key = (
                        frozenset((av, bv)) in confusable,
                        user_a[(uid, av)] + user_b[(uid, bv)],
                        net[av] - net[bv],
                        rng.random(),
                    )
                    if best is None or key < best[0]:
                        best = (key, av, bv)
            if best is None:
                continue
            _, a, b = best
            net[a] += 1
            net[b] -= 1
            user_a[(uid, a)] += 1
            user_b[(uid, b)] += 1
            assignment[q["query_id"]] = (a, b)
    return assignment


def _value_violation_feasible(query, axis, ab_values):
    """Whether `axis` (color/pattern) can carry this query's TPO violation.

    Requires neutral value pools on both sides plus at least one
    compatible-neutral garment that passes the pattern-garment exclusion for
    every pattern the options would contain.
    """
    pools = (query.get("violation_options") or {}).get(axis)
    if not pools or not pools.get("compatible_neutral") or not pools.get("incompatible_neutral"):
        return False
    garments = list(dict.fromkeys(query.get("compatible_garments", [])))
    if axis == "color":
        patterns = _option_patterns(query, ab_values)
        return any(
            all(is_pattern_garment_compatible(p, g) for p in patterns)
            for g in garments
        )
    # axis == "pattern": violation values replace the fixed pattern
    for g in garments:
        for cv in pools["compatible_neutral"]:
            for iv in pools["incompatible_neutral"]:
                if (is_pattern_garment_compatible(cv, g)
                        and is_pattern_garment_compatible(iv, g)):
                    return True
    return False


def assign_violation_axes(queries, ab_values, seed=42):
    """Choose the TPO-violation axis per query.

    Physical scenarios offer only garment_category. Dress-coded scenarios
    additionally offer each scenario-constrained color/pattern axis (other
    than the active axis) whenever the profile leaves preference-neutral
    values on both sides. Axis usage is balanced within each scenario, then
    per user; infeasible slots fall back to garment_category so no coverage
    is lost.
    """
    rng = random.Random(seed + 2027)
    assignment = {}
    scen_axis_use = Counter()
    user_axis_use = Counter()
    for q in sorted(queries, key=lambda q: q["query_id"]):
        qid = q["query_id"]
        if qid not in ab_values:
            continue
        sid = q.get("scenario_id", "?")
        uid = q.get("user_id", "?")
        candidates = ["garment_category"]
        for axis in sorted(q.get("violation_options") or {}):
            if _value_violation_feasible(q, axis, ab_values):
                candidates.append(axis)
        pick = min(candidates, key=lambda ax: (
            scen_axis_use[(sid, ax)], user_axis_use[(uid, ax)], rng.random()))
        assignment[qid] = pick
        scen_axis_use[(sid, pick)] += 1
        user_axis_use[(uid, pick)] += 1
    return assignment


def assign_violation_values(queries, ab_values, violation_axes, seed=42):
    """For color/pattern-violation queries, choose the shared compatible
    garment and the (TPO-compatible, TPO-incompatible) neutral value pair.

    Balanced like the garment pairs: confusable pairs penalized, then value
    pairs and garments rotated per user and per scenario.
    """
    rng = random.Random(seed + 3041)
    assignment = {}
    val_use = Counter()
    user_val_use = Counter()
    inc_use = Counter()
    g_use = Counter()
    user_g_use = Counter()

    for q in sorted(queries, key=lambda q: q["query_id"]):
        qid = q["query_id"]
        axis = violation_axes.get(qid)
        if axis in (None, "garment_category"):
            continue
        uid = q.get("user_id", "?")
        sid = q.get("scenario_id", "?")
        pools = (q.get("violation_options") or {}).get(axis) or {}
        confusable = CONFUSABLE_ACTIVE_PAIRS.get(axis, set())
        base_patterns = _option_patterns(q, ab_values) if axis == "color" else set()

        candidates = []
        for g in dict.fromkeys(q.get("compatible_garments", [])):
            for cv in pools.get("compatible_neutral", []):
                for iv in pools.get("incompatible_neutral", []):
                    if cv == iv:
                        continue
                    patterns = base_patterns | ({cv, iv} if axis == "pattern" else set())
                    if not all(is_pattern_garment_compatible(p, g) for p in patterns):
                        continue
                    score = (
                        CONFUSABLE_PAIR_PENALTY * (frozenset((cv, iv)) in confusable)
                        + 8 * user_val_use[(uid, axis, cv, iv)]
                        + 4 * val_use[(sid, axis, cv, iv)]
                        + 2 * user_g_use[(uid, g)]
                        + 2 * inc_use[(axis, iv)]
                        + g_use[(sid, g)]
                    )
                    candidates.append((score, rng.random(), g, cv, iv))
        if not candidates:
            continue

        _, _, g, cv, iv = min(candidates)
        assignment[qid] = (g, cv, iv)
        val_use[(sid, axis, cv, iv)] += 1
        user_val_use[(uid, axis, cv, iv)] += 1
        inc_use[(axis, iv)] += 1
        g_use[(sid, g)] += 1
        user_g_use[(uid, g)] += 1
    return assignment


def assign_garment_pairs(queries, ab_values, seed=42):
    """Choose neutral TPO garment pairs while reducing repeated option sets."""
    rng = random.Random(seed + 1009)
    assignment = {}
    pair_use = Counter()
    comp_use = Counter()
    inc_use = Counter()
    user_pair_use = Counter()
    user_scenario_pair_use = Counter()
    user_comp_use = Counter()
    user_inc_use = Counter()

    def dof(q):
        comp = list(dict.fromkeys(q.get("compatible_garments", [])))
        inc = list(dict.fromkeys(q.get("incompatible_garments", [])))
        return (len(comp) * len(inc), len(comp), len(inc), q["query_id"])

    for q in sorted(queries, key=dof):
        uid = q.get("user_id", "?")
        sid = q.get("scenario_id", "?")
        axis = q.get("active_axis", "?")
        patterns = _option_patterns(q, ab_values)
        comp = list(dict.fromkeys(q.get("compatible_garments", [])))
        inc = list(dict.fromkeys(q.get("incompatible_garments", [])))
        candidates = []
        for cg in comp:
            for ig in inc:
                if cg == ig:
                    continue
                if not all(
                    is_pattern_garment_compatible(pattern, garment)
                    for pattern in patterns
                    for garment in (cg, ig)
                ):
                    continue
                score = (
                    CONFUSABLE_PAIR_PENALTY * (frozenset((cg, ig)) in CONFUSABLE_GARMENT_PAIRS)
                    + 20 * user_scenario_pair_use[(uid, sid, cg, ig)]
                    + 10 * user_pair_use[(uid, axis, cg, ig)]
                    + 4 * pair_use[(axis, cg, ig)]
                    + 2 * user_comp_use[(uid, axis, cg)]
                    + 2 * user_inc_use[(uid, axis, ig)]
                    + comp_use[(axis, cg)]
                    + inc_use[(axis, ig)]
                )
                candidates.append((score, rng.random(), cg, ig))
        if not candidates:
            continue

        _, _, compat_garment, incompat_garment = min(candidates)
        assignment[q["query_id"]] = (compat_garment, incompat_garment)
        pair_use[(axis, compat_garment, incompat_garment)] += 1
        comp_use[(axis, compat_garment)] += 1
        inc_use[(axis, incompat_garment)] += 1
        user_pair_use[(uid, axis, compat_garment, incompat_garment)] += 1
        user_scenario_pair_use[(uid, sid, compat_garment, incompat_garment)] += 1
        user_comp_use[(uid, axis, compat_garment)] += 1
        user_inc_use[(uid, axis, incompat_garment)] += 1
    return assignment


# ── per-query plan build (uses pre-assigned a/b) ──────────

def plan_options_for_query(profile, query, ab_values, violation_axes=None,
                           garment_pairs=None, violation_values=None):
    """Returns (plan_dict, None) on success, or (None, reason) on skip."""
    scenario = get_scenario_by_id(query["scenario_id"])
    if scenario is None:
        return None, "scenario_not_found"
    active_axis = query["active_axis"]
    if active_axis not in ALLOWED_ACTIVE_AXES:
        return None, "active_axis_not_allowed"

    ab = ab_values.get(query["query_id"])
    if ab is None:
        return None, "no_counterbalanced_ab_pair"   # liked or disliked pool empty
    liked_v, disliked_v = ab
    if not liked_v or not disliked_v or liked_v == disliked_v:
        return None, "degenerate_ab_pair"
    if disliked_v not in set(query.get("disliked_compatible", [])):
        return None, "active_value_not_disliked"

    violation_axis = (violation_axes or {}).get(query["query_id"], "garment_category")

    fixed_attrs = dict(query.get("fixed_attrs", {}))
    fixed_attrs.pop("garment_category", None)

    if violation_axis == "garment_category":
        pair = (garment_pairs or {}).get(query["query_id"])
        option_patterns = _option_patterns(query, ab_values)
        if pair is None:
            rng = random.Random(_stable_seed(query["query_id"]))
            compat_garment = _pick_garment(
                query, "compatible_garments", rng, patterns=option_patterns
            )
            incompat_garment = _pick_garment(
                query, "incompatible_garments", rng, patterns=option_patterns
            )
        else:
            compat_garment, incompat_garment = pair
        if not compat_garment or not incompat_garment or compat_garment == incompat_garment:
            return None, "no_neutral_garment_pair"

        tpo_compatible_value = compat_garment
        violation_value = incompat_garment
        attrs_a = {**fixed_attrs, active_axis: liked_v, "garment_category": compat_garment}
        attrs_b = {**fixed_attrs, active_axis: disliked_v, "garment_category": compat_garment}
        attrs_c = {**fixed_attrs, active_axis: liked_v, "garment_category": incompat_garment}
        attrs_d = {**fixed_attrs, active_axis: disliked_v, "garment_category": incompat_garment}
    else:
        # Color/pattern violation: one compatible-neutral garment is shared by
        # all four options; A/B vs C/D differ on the violation-axis value.
        triple = (violation_values or {}).get(query["query_id"])
        if triple is None:
            return None, "no_violation_value_pair"
        compat_garment, tpo_compatible_value, violation_value = triple
        fixed_attrs.pop(violation_axis, None)

        attrs_a = {**fixed_attrs, active_axis: liked_v,
                   violation_axis: tpo_compatible_value, "garment_category": compat_garment}
        attrs_b = {**fixed_attrs, active_axis: disliked_v,
                   violation_axis: tpo_compatible_value, "garment_category": compat_garment}
        attrs_c = {**fixed_attrs, active_axis: liked_v,
                   violation_axis: violation_value, "garment_category": compat_garment}
        attrs_d = {**fixed_attrs, active_axis: disliked_v,
                   violation_axis: violation_value, "garment_category": compat_garment}

    label_map = {"A": "tpo_and_preference", "B": "tpo_only",
                 "C": "preference_only", "D": "neither"}
    options = {}
    for k, attrs in [("A", attrs_a), ("B", attrs_b), ("C", attrs_c), ("D", attrs_d)]:
        options[k] = {
            "label": label_map[k],
            "attributes": attrs,
            "search_query": attrs_to_search_query(attrs),
            "rationale": _rationale(k, active_axis, attrs, violation_axis),
        }

    plan = {
        "query_id": query["query_id"],
        "user_id": query["user_id"],
        "scenario_id": query["scenario_id"],
        "scenario_archetype": scenario.get("archetype"),
        "scenario_name": scenario.get("name"),
        "domain": "fashion",
        "query_type": query["query_type"],
        "query_text": query.get("query_text", ""),
        "active_axis": active_axis,
        "fixed_attrs": fixed_attrs,
        "violation_axis": violation_axis,
        "violation_value": violation_value,
        "tpo_compatible_value": tpo_compatible_value,
        "main_category": compat_garment,
        "options": options,
    }
    return plan, None


def run_pipeline(profile_path, query_path, output_path, force=False, limit=0, seed=42):
    log_step("STAGE 3 — Option Planner (clean 2×2 + strict disliked B/D)")
    profiles = {p["user_id"]: p for p in load_jsonl(profile_path)}
    queries = load_jsonl(query_path)
    print(f"  {len(profiles)} profiles, {len(queries)} queries")

    # GLOBAL pass over ALL queries so balance holds even with --limit.
    ab_values = assign_ab_values(queries, seed=seed)
    print(f"  Counterbalanced A/B value assignment for {len(ab_values)} queries")
    violation_axes = assign_violation_axes(queries, ab_values, seed=seed)
    axis_counts = Counter(violation_axes.values())
    print(f"  Violation-axis assignment: {dict(axis_counts)}")
    garment_queries = [
        q for q in queries
        if violation_axes.get(q["query_id"]) == "garment_category"
    ]
    garment_pairs = assign_garment_pairs(garment_queries, ab_values, seed=seed)
    print(f"  Diverse neutral garment-pair assignment for {len(garment_pairs)} queries")
    violation_values = assign_violation_values(queries, ab_values, violation_axes, seed=seed)
    print(f"  Violation value-pair assignment for {len(violation_values)} queries")

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
    fail_reasons = Counter()
    for i, query in enumerate(todo):
        if query["user_id"] not in profiles:
            n_fail += 1
            fail_reasons["user_not_in_profiles"] += 1
            continue
        prof = profiles[query["user_id"]]
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(todo)}] planning...")
        try:
            plan, reason = plan_options_for_query(
                prof, query, ab_values, violation_axes, garment_pairs, violation_values)
            if plan is None:
                n_fail += 1
                fail_reasons[reason] += 1
                continue
            plans.append(plan)
        except Exception as e:
            print(f"    ERROR {query['query_id']}: {e}")
            n_fail += 1
            fail_reasons["exception"] += 1

    save_jsonl(plans, output_path)
    print(f"\n  ✓ Saved {len(plans)} plans   (skipped {n_fail})")

    # failure breakdown (filter %)
    total = len(plans) + n_fail
    if n_fail:
        print("  skip reasons (share of all attempted):")
        for r, n in fail_reasons.most_common():
            print(f"    {r:28s}: {n:5d}  ({n / max(total, 1):.1%})")

    _report_balance(plans)
    _report_per_user_variety(plans)
    _report_violation_axes(plans)
    _report_option_diversity(plans)


def _report_balance(plans):
    fa = defaultdict(Counter)
    fb = defaultdict(Counter)
    for p in plans:
        ax = p["active_axis"]
        fa[ax][p["options"]["A"]["attributes"].get(ax)] += 1
        fb[ax][p["options"]["B"]["attributes"].get(ax)] += 1
    for ax in fa:
        lr = {v: (fa[ax][v] / (fa[ax][v] + fb[ax][v]) if (fa[ax][v] + fb[ax][v]) else .5)
              for v in set(fa[ax]) | set(fb[ax])}
        worst = max(lr.items(), key=lambda kv: abs(kv[1] - 0.5)) if lr else (None, .5)
        print(f"  [{ax}] residual max value-skew: {worst[0]}={worst[1]:.2f} "
              f"(0.50 target; strict dislike pools may make this unattainable)")


def _report_per_user_variety(plans):
    """Per-user monotony check: max share of one value in a user's A slots,
    B slots, and fixed non-active attrs. 1.00 = user always sees one value."""
    for slot, getter in [
        ("A", lambda p, ax: p["options"]["A"]["attributes"].get(ax)),
        ("B", lambda p, ax: p["options"]["B"]["attributes"].get(ax)),
    ]:
        for ax in ("color", "pattern"):
            per_user = defaultdict(Counter)
            for p in plans:
                if p["active_axis"] == ax:
                    per_user[p["user_id"]][getter(p, ax)] += 1
            shares = [c.most_common(1)[0][1] / sum(c.values())
                      for c in per_user.values() if c]
            if shares:
                print(f"  [{ax}] per-user {slot}-value max-share: "
                      f"avg={sum(shares)/len(shares):.2f}, worst={max(shares):.2f}")

    per_user_fixed = defaultdict(Counter)
    for p in plans:
        fx = p.get("fixed_attrs") or {}
        for ax, v in fx.items():
            per_user_fixed[(p["user_id"], ax)][v] += 1
    shares = [c.most_common(1)[0][1] / sum(c.values())
              for c in per_user_fixed.values() if sum(c.values()) >= 5]
    if shares:
        print(f"  fixed-attr per-user max-share: "
              f"avg={sum(shares)/len(shares):.2f}, worst={max(shares):.2f}")


def _report_violation_axes(plans):
    dist = Counter(p.get("violation_axis") for p in plans)
    total = len(plans)
    parts = ", ".join(f"{ax}={n} ({n/max(total,1):.0%})" for ax, n in dist.most_common())
    print(f"  violation-axis mix: {parts}")
    # dress-coded scenarios only (those that could rotate at all)
    by_arch = defaultdict(Counter)
    for p in plans:
        by_arch[p.get("scenario_archetype")][p.get("violation_axis")] += 1
    rotated = {a: c for a, c in by_arch.items()
               if set(c) - {"garment_category"}}
    for arch in sorted(rotated):
        c = rotated[arch]
        t = sum(c.values())
        parts = ", ".join(f"{ax}={n}" for ax, n in c.most_common())
        print(f"    {arch:25s}: {parts}  (n={t})")


def _report_option_diversity(plans):
    by_user_axis = defaultdict(Counter)
    for p in plans:
        opts = p["options"]
        pair = (
            opts["A"]["attributes"].get("garment_category"),
            opts["C"]["attributes"].get("garment_category"),
        )
        by_user_axis[(p.get("user_id"), p.get("active_axis"))][pair] += 1

    ratios = []
    worst = None
    for key, counts in by_user_axis.items():
        total = sum(counts.values())
        if total <= 0:
            continue
        ratio = len(counts) / total
        ratios.append(ratio)
        item = (ratio, key, len(counts), total, counts.most_common(1)[0])
        if worst is None or item < worst:
            worst = item

    if ratios and worst:
        ratio, key, uniq, total, top = worst
        avg = sum(ratios) / len(ratios)
        print(f"  garment-pair diversity per user-axis: avg={avg:.2f}, "
              f"min={ratio:.2f} ({key[0]}/{key[1]}: {uniq}/{total}, "
              f"top_repeat={top[1]})")


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
