#!/usr/bin/env python3
"""
Verify whether every option follows the actual option_planner 2x2 semantics.

Actual planner semantics:
  A = active_axis liked value     + TPO-compatible garment
  B = active_axis non-preferred   + TPO-compatible garment
  C = active_axis liked value     + TPO-incompatible garment
  D = active_axis non-preferred   + TPO-incompatible garment

Important:
  - active_axis is color or pattern.
  - TPO is represented by garment_category, not by active_axis.
  - liked/non-preferred values should be checked against query pools, not directly
    against scenario color/pattern compat fields.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from configs.config import OPTIONS_DIR, PROFILES_DIR, QUERIES_DIR
from scripts.utils import load_jsonl


EXPECTED = {
    "A": {"tpo": True,  "profile": True},
    "B": {"tpo": True,  "profile": False},
    "C": {"tpo": False, "profile": True},
    "D": {"tpo": False, "profile": False},
}


def as_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def as_set(x):
    return set(as_list(x))


def option_attrs(option):
    return option.get("attributes", {}) or {}


def liked_pool(query):
    return as_set(query.get("liked_compatible", []))


def nonpref_pool(query):
    # Same semantics as construction/option_planner.py:
    # non-preferred = disliked first, then neutral.
    return as_set(query.get("disliked_compatible", [])) | as_set(query.get("neutral_compatible", []))


def check_plan(plan, queries_map):
    errors = []

    qid = plan.get("query_id")
    query = queries_map.get(qid)
    if not query:
        return [{
            "kind": "missing_query",
            "query_id": qid,
            "user_id": plan.get("user_id"),
            "scenario_id": plan.get("scenario_id"),
        }]

    active_axis = plan.get("active_axis")
    if active_axis not in {"color", "pattern"}:
        return [{
            "kind": "unexpected_active_axis",
            "query_id": qid,
            "active_axis": active_axis,
        }]

    compatible_garments = as_set(query.get("compatible_garments", []))
    incompatible_garments = as_set(query.get("incompatible_garments", []))
    liked_values = liked_pool(query)
    nonpref_values = nonpref_pool(query)

    options = plan.get("options", {}) or {}

    for label in ["A", "B", "C", "D"]:
        if label not in options:
            errors.append({
                "kind": "missing_option",
                "query_id": qid,
                "option": label,
            })
            continue

        attrs = option_attrs(options[label])
        garment = attrs.get("garment_category")
        active_value = attrs.get(active_axis)

        expected = EXPECTED[label]

        if expected["tpo"]:
            tpo_ok = garment in compatible_garments
        else:
            tpo_ok = garment in incompatible_garments

        if expected["profile"]:
            profile_ok = active_value in liked_values
        else:
            profile_ok = active_value in nonpref_values

        if not tpo_ok or not profile_ok:
            errors.append({
                "kind": "semantic_mismatch",
                "query_id": qid,
                "user_id": plan.get("user_id"),
                "scenario_id": plan.get("scenario_id"),
                "active_axis": active_axis,
                "option": label,

                "expected_tpo": expected["tpo"],
                "expected_profile": expected["profile"],

                "tpo_ok": tpo_ok,
                "profile_ok": profile_ok,

                "garment_category": garment,
                "active_value": active_value,

                "compatible_garments": sorted(compatible_garments),
                "incompatible_garments": sorted(incompatible_garments),
                "liked_values": sorted(liked_values),
                "nonpref_values": sorted(nonpref_values),

                "search_query": options[label].get("search_query"),
                "attributes": attrs,
            })

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", type=Path, default=OPTIONS_DIR / "option_plans.jsonl")
    parser.add_argument("--queries", type=Path, default=QUERIES_DIR / "queries.jsonl")
    parser.add_argument("--profiles", type=Path, default=PROFILES_DIR / "profiles.jsonl")
    parser.add_argument("--out", type=Path, default=OPTIONS_DIR / "option_semantic_errors.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--show", type=int, default=20)
    args = parser.parse_args()

    plans = load_jsonl(args.plans)
    queries = load_jsonl(args.queries)
    _profiles = load_jsonl(args.profiles)  # loaded only to confirm path consistency

    if args.limit > 0:
        plans = plans[:args.limit]

    queries_map = {q["query_id"]: q for q in queries}

    all_errors = []
    for plan in plans:
        all_errors.extend(check_plan(plan, queries_map))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for e in all_errors:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    by_kind = Counter(e["kind"] for e in all_errors)
    by_axis = Counter(e.get("active_axis", "unknown") for e in all_errors)
    by_option = Counter(e.get("option", "unknown") for e in all_errors)

    print("=" * 80)
    print("Option semantic verification")
    print("=" * 80)
    print(f"plans checked: {len(plans)}")
    print(f"errors:        {len(all_errors)}")
    print(f"saved:         {args.out}")

    print("\n-- by kind --")
    for k, v in by_kind.most_common():
        print(f"{k:24s} {v}")

    print("\n-- by active_axis --")
    for k, v in by_axis.most_common():
        print(f"{k:24s} {v}")

    print("\n-- by option --")
    for k, v in by_option.most_common():
        print(f"{k:24s} {v}")

    if all_errors:
        print(f"\n-- first {min(args.show, len(all_errors))} errors --")
        for e in all_errors[:args.show]:
            print(
                f"[{e.get('kind')}] "
                f"{e.get('query_id')} "
                f"axis={e.get('active_axis')} "
                f"opt={e.get('option')} "
                f"garment={e.get('garment_category')} "
                f"value={e.get('active_value')} "
                f"tpo_ok={e.get('tpo_ok')} "
                f"profile_ok={e.get('profile_ok')}"
            )

    if all_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()