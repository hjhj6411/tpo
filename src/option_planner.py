"""
Option Planner v2 — DETERMINISTIC 4-option construction (no LLM at label time).

Given (profile, query) with active_axis + fixed_attrs + tpo_scenario:
  A = liked active value + fixed                              (tpo_and_preference)
  B = disliked active value + fixed                           (tpo_only)
  C = liked active value + fixed (one fixed axis → TPO-bad)   (preference_only)
  D = disliked active value + fixed (one fixed axis → TPO-bad) (neither)

For active_axis ∈ {color, pattern}, TPO violation is achieved by swapping
garment_category to a TPO-incompatible value. For active_axis=garment_category,
TPO violation is by swapping color (secondary axis).
"""

import argparse
import random
import sys
from pathlib import Path

from .utils import save_jsonl, load_jsonl, log_step

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import (
    OPTIONS_DIR, PROFILES_DIR, QUERIES_DIR, FASHION_ATTRIBUTE_AXES,
)

# neutral 쿼리 전용 fallback violation 후보
NEUTRAL_VIOLATION_GARMENTS = ["shorts", "tank_top", "t_shirt"]
NEUTRAL_VIOLATION_COLORS   = ["orange", "yellow"]

# Curated TPO incompatibility rules (used by both planner and label_verifier)
TPO_ATTR_INCOMPATIBILITIES = [
    ({"rainy", "snowy", "winter", "cold"}, "garment_category",
        {"shorts", "t_shirt", "tank_top", "skirt"}),
    ({"summer", "hot_humid"}, "garment_category",
        {"trench_coat", "coat", "parka", "sweater"}),
    ({"formal", "wedding_venue", "ceremony_attendance"}, "garment_category",
        {"t_shirt", "shorts", "hoodie", "tank_top", "windbreaker"}),
    ({"office", "work_meeting", "business_casual"}, "garment_category",
        {"shorts", "hoodie", "tank_top"}),
    ({"exercise", "gym"}, "garment_category",
        {"suit_jacket", "blazer", "coat", "trench_coat", "dress"}),
    ({"beach"}, "garment_category",
        {"suit_jacket", "blazer", "trench_coat", "coat", "parka"}),
    ({"formal", "wedding_venue", "ceremony_attendance", "business_casual",
       "office", "work_meeting"}, "pattern",
        {"graphic_print", "camouflage", "animal_print", "floral"}),
    ({"formal", "wedding_venue", "ceremony_attendance"}, "color",
        {"orange", "yellow", "pink", "green", "red", "purple"}),
    ({"office", "work_meeting", "business_casual"}, "color",
        {"orange", "yellow", "pink"}),
]


def _flat_tpo(scenario):
    flat = set()
    if not scenario:
        return flat
    for axis, sub in scenario.items():
        if isinstance(sub, dict):
            for v in sub.values():
                flat.add(str(v).lower())
    return flat


def tpo_compatible(attrs, scenario):
    if not scenario:
        return True
    flat = _flat_tpo(scenario)
    for keys, axis, forbidden in TPO_ATTR_INCOMPATIBILITIES:
        if flat & keys:
            v = attrs.get(axis)
            if v and v in forbidden:
                return False
    return True


def axis_values_tpo_incompatible(axis, fixed_attrs, scenario):
    out = []
    for v in FASHION_ATTRIBUTE_AXES[axis]:
        attrs = dict(fixed_attrs)
        attrs[axis] = v
        if not tpo_compatible(attrs, scenario):
            out.append(v)
    return out


def build_options_color_or_pattern(active_axis, fixed_attrs, structured,
                                   scenario, rng):
    prefs    = structured.get(active_axis, {})
    likes    = list(prefs.get("likes",    []))
    dislikes = list(prefs.get("dislikes", []))
    if not likes or not dislikes:
        return None

    # ── A: liked + TPO-compatible ──────────────────────────────────
    compatible_likes = [v for v in likes
                        if tpo_compatible({**fixed_attrs, active_axis: v}, scenario)]
    if not compatible_likes:
        return None                          # liked 값이 모두 TPO-incompatible → 포기
    liked_v = rng.choice(compatible_likes)

    # ── B: disliked + TPO-compatible ───────────────────────────────
    compatible_dislikes = [v for v in dislikes
                           if tpo_compatible({**fixed_attrs, active_axis: v}, scenario)]
    if not compatible_dislikes:
        # Fallback: 전체 어휘에서 liked가 아니고 TPO-compatible한 값
        compatible_dislikes = [
            v for v in FASHION_ATTRIBUTE_AXES[active_axis]
            if v not in likes
            and tpo_compatible({**fixed_attrs, active_axis: v}, scenario)
        ]
    if not compatible_dislikes:
        return None
    disliked_v = rng.choice(compatible_dislikes)

    # ── C/D: TPO-violation garment ─────────────────────────────────
    incompat_g = axis_values_tpo_incompatible(
        "garment_category", {**fixed_attrs, active_axis: liked_v}, scenario
    )
    if not incompat_g:
        # Fallback: neutral 쿼리 — 현재 garment와 다른 violation 후보
        current_g = fixed_attrs.get("garment_category", "")
        incompat_g = [g for g in NEUTRAL_VIOLATION_GARMENTS if g != current_g]
    if not incompat_g:
        return None
    bad_g = rng.choice(incompat_g)

    return {
        "A": {**fixed_attrs, active_axis: liked_v},
        "B": {**fixed_attrs, active_axis: disliked_v},
        "C": {**fixed_attrs, active_axis: liked_v,    "garment_category": bad_g},
        "D": {**fixed_attrs, active_axis: disliked_v, "garment_category": bad_g},
        "main_category":      fixed_attrs.get("garment_category"),
        "tpo_violation_axis": "garment_category",
    }


def build_options_garment_category(fixed_attrs, structured, scenario,
                                   secondary_axis, rng):
    prefs    = structured.get("garment_category", {})
    likes    = list(prefs.get("likes",    []))
    dislikes = list(prefs.get("dislikes", []))
    if not likes or not dislikes:
        return None

    # ── A: liked garment + TPO-compatible ─────────────────────────
    compatible_likes = [v for v in likes
                        if tpo_compatible({**fixed_attrs, "garment_category": v}, scenario)]
    if not compatible_likes:
        return None
    liked_g = rng.choice(compatible_likes)

    # ── B: disliked garment + TPO-compatible ──────────────────────
    compatible_dislikes = [v for v in dislikes
                           if tpo_compatible({**fixed_attrs, "garment_category": v}, scenario)]
    if not compatible_dislikes:
        compatible_dislikes = [
            v for v in FASHION_ATTRIBUTE_AXES["garment_category"]
            if v not in likes
            and tpo_compatible({**fixed_attrs, "garment_category": v}, scenario)
        ]
    if not compatible_dislikes:
        return None
    disliked_g = rng.choice(compatible_dislikes)

    # ── C/D: TPO-violation secondary value ────────────────────────
    incompat_sec = axis_values_tpo_incompatible(
        secondary_axis, {**fixed_attrs, "garment_category": liked_g}, scenario
    )
    if not incompat_sec:
        # Fallback: neutral 쿼리
        current_sec = fixed_attrs.get(secondary_axis, "")
        fallback_pool = (NEUTRAL_VIOLATION_COLORS if secondary_axis == "color"
                         else ["graphic_print", "camouflage"])
        incompat_sec = [v for v in fallback_pool if v != current_sec]
    if not incompat_sec:
        return None
    bad_sec = rng.choice(incompat_sec)

    return {
        "A": {**fixed_attrs, "garment_category": liked_g},
        "B": {**fixed_attrs, "garment_category": disliked_g},
        "C": {**fixed_attrs, "garment_category": liked_g,    secondary_axis: bad_sec},
        "D": {**fixed_attrs, "garment_category": disliked_g, secondary_axis: bad_sec},
        "main_category":      None,
        "tpo_violation_axis": secondary_axis,
    }


def attrs_to_search_query(attrs):
    parts = []
    color   = attrs.get("color")
    pattern = attrs.get("pattern")
    garment = attrs.get("garment_category", "item")
    if pattern and pattern != "solid":
        parts.append(pattern.replace("_", " "))
    if color:
        parts.append(color)
    parts.append(garment.replace("_", " "))
    return " ".join(parts)


def _rationale(k, active_axis, attrs, violation_axis):
    val = attrs.get(active_axis)
    if k == "A":
        return f"liked {active_axis}={val}, TPO-compatible"
    if k == "B":
        return f"disliked {active_axis}={val}, TPO-compatible"
    if k == "C":
        return (f"liked {active_axis}={val}, but {violation_axis}={attrs.get(violation_axis)} "
                f"violates TPO")
    if k == "D":
        return (f"disliked {active_axis}={val}, AND {violation_axis}={attrs.get(violation_axis)} "
                f"violates TPO")
    return ""


def _ensure_axis_prefs(structured, axis, rng):
    """프로필에 axis가 없거나 likes/dislikes가 비어 있으면 즉석 생성."""
    prefs = structured.get(axis, {})
    likes    = list(prefs.get("likes",    []))
    dislikes = list(prefs.get("dislikes", []))
    all_vals = FASHION_ATTRIBUTE_AXES[axis]

    if not likes:
        likes = rng.sample(all_vals, min(2, len(all_vals)))
    if not dislikes:
        remaining = [v for v in all_vals if v not in likes]
        dislikes = rng.sample(remaining, min(1, len(remaining))) if remaining else []

    return {"likes": likes, "dislikes": dislikes}

def plan_options_for_query(profile, query):
    active_axis = query["active_axis"]
    fixed_attrs = query["fixed_attrs"]
    scenario    = query.get("tpo_scenario") or {}
    rng         = random.Random(abs(hash(query["query_id"])) % 100000)
    structured = dict(profile["structured_attributes"])
    structured[active_axis] = _ensure_axis_prefs(structured, active_axis, rng)

    if active_axis in ("color", "pattern"):
        result = build_options_color_or_pattern(
            active_axis, fixed_attrs, structured, scenario, rng,
        )
    elif active_axis == "garment_category":
        sec_axes = query.get("secondary_differentiation_axis", ["color"])
        sec_axis = sec_axes[0] if sec_axes else "color"
        result = build_options_garment_category(
            fixed_attrs, structured, scenario, sec_axis, rng,
        )
    else:
        return None

    if result is None:
        return None

    options   = {}
    label_map = {"A": "tpo_and_preference", "B": "tpo_only",
                 "C": "preference_only",     "D": "neither"}
    for k in "ABCD":
        attrs = result[k]
        options[k] = {
            "label":        label_map[k],
            "attributes":   attrs,
            "search_query": attrs_to_search_query(attrs),
            "rationale":    _rationale(k, active_axis, attrs, result["tpo_violation_axis"]),
        }

    return {
        "query_id":           query["query_id"],
        "user_id":            query["user_id"],
        "domain":             "fashion",
        "query_type":         query["query_type"],
        "active_axis":        active_axis,
        "tpo_axis":           query["tpo_axis"],
        "fixed_attrs":        fixed_attrs,
        "main_category":      result["main_category"],
        "tpo_violation_axis": result["tpo_violation_axis"],
        "options":            options,
    }


def run_pipeline(profile_path, query_path, output_path, force=False, limit=0,
                 provider=None):
    log_step("Option Planner v2 (deterministic, One Axis Per Instance)")
    profiles = {p["user_id"]: p for p in load_jsonl(profile_path)}
    queries  = load_jsonl(query_path)
    print(f"  {len(profiles)} profiles, {len(queries)} queries")

    if output_path.exists() and not force:
        existing = load_jsonl(output_path)
        done     = {p["query_id"] for p in existing}
        plans    = existing
        todo     = [q for q in queries if q["query_id"] not in done]
        print(f"  Resuming: {len(done)} already planned")
    else:
        plans = []
        todo  = queries
    if limit > 0:
        todo = todo[:limit]

    n_fail = 0
    for i, query in enumerate(todo):
        if query["user_id"] not in profiles:
            continue
        prof = profiles[query["user_id"]]
        print(f"\n  [{i+1}/{len(todo)}] {query['query_id']} "
              f"(axis={query['active_axis']}, qtype={query['query_type']})")
        try:
            plan = plan_options_for_query(prof, query)
            if plan is None:
                print("    Could not build valid options")
                n_fail += 1
                continue
            plans.append(plan)
            for k in "ABCD":
                print(f"    {k} ({plan['options'][k]['label']}): "
                      f"{plan['options'][k]['attributes']}")
            if (i + 1) % 10 == 0:
                save_jsonl(plans, output_path)
        except Exception as e:
            print(f"    ERROR: {e}")
            n_fail += 1

    save_jsonl(plans, output_path)
    print(f"\n  ✓ Saved {len(plans)} plans, {n_fail} failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_path", type=Path, default=PROFILES_DIR / "profiles.jsonl")
    parser.add_argument("--query_path",   type=Path, default=QUERIES_DIR   / "queries.jsonl")
    parser.add_argument("--output",       type=Path, default=OPTIONS_DIR   / "option_plans.jsonl")
    parser.add_argument("--force",    action="store_true")
    parser.add_argument("--limit",    type=int, default=0)
    parser.add_argument("--provider", type=str, default=None)
    args = parser.parse_args()
    run_pipeline(args.profile_path, args.query_path, args.output,
                 args.force, args.limit, args.provider)


if __name__ == "__main__":
    main()