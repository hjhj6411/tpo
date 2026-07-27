#!/usr/bin/env python3
"""Mutation test for the option-plan validator.

A validator that reports "0 failures" is only evidence if it can actually fail.
This injects known-bad edits into real plans and asserts the validator catches
every one. 6 mutations x 8 plan samples = 48 injections, all of which must be
detected; anything less means the release audit is blind to that defect class.

The 8 samples are drawn deterministically across both tracks and all three
active axes, so a mutation cannot pass merely because it was tried on a plan
shape that does not exercise the relevant check.

Run:
    POD_VARIANT=wacv_scenario_v2 python -m tests.test_option_validator_mutations
    # or: python tests/test_option_validator_mutations.py
Exits 1 if any injected defect goes undetected.
"""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.validate_options import (          # noqa: E402
    check_structure_and_pref, check_tpo, axis_roles, load_jsonl,
)
from configs.config import OPTIONS_DIR, PROFILES_DIR                # noqa: E402

N_SAMPLES = 8


# ── mutations: each returns a mutated copy, or None if inapplicable ────────
def mut_swap_labels(plan, prof):
    """B and C swap their label strings -> the 2x2 no longer means what it says."""
    p = copy.deepcopy(plan)
    p["options"]["B"]["label"], p["options"]["C"]["label"] = (
        p["options"]["C"]["label"], p["options"]["B"]["label"])
    return p


def mut_break_active_pair(plan, prof):
    """C's active value stops matching A's -> preference axis is degenerate."""
    act, _vio, _fix = axis_roles(plan)
    p = copy.deepcopy(plan)
    p["options"]["C"]["attributes"][act] = p["options"]["B"]["attributes"][act]
    return p


def mut_break_violation_pair(plan, prof):
    """B takes C's violation value -> A/B no longer share the TPO-compatible value."""
    act, vio, _fix = axis_roles(plan)
    p = copy.deepcopy(plan)
    p["options"]["B"]["attributes"][vio] = p["options"]["C"]["attributes"][vio]
    return p


def mut_disliked_A(plan, prof):
    """A's active value becomes one the user DISLIKES -> answer key is wrong."""
    act, _vio, _fix = axis_roles(plan)
    dis = (prof.get("structured_attributes", {}).get(act, {}) or {}).get("dislikes", [])
    if not dis:
        return None
    p = copy.deepcopy(plan)
    p["options"]["A"]["attributes"][act] = dis[0]
    p["options"]["C"]["attributes"][act] = dis[0]
    return p


def mut_non_neutral_violation(plan, prof):
    """The violation axis takes a user-liked value -> TPO confounded with preference."""
    _act, vio, _fix = axis_roles(plan)
    likes = (prof.get("structured_attributes", {}).get(vio, {}) or {}).get("likes", [])
    if not likes:
        return None
    p = copy.deepcopy(plan)
    for k in "AB":
        p["options"][k]["attributes"][vio] = likes[0]
    return p


def mut_tpo_flip(plan, prof):
    """C/D take the TPO-COMPATIBLE violation value -> no situation contrast left."""
    _act, vio, _fix = axis_roles(plan)
    p = copy.deepcopy(plan)
    compat = p["options"]["A"]["attributes"][vio]
    for k in "CD":
        p["options"][k]["attributes"][vio] = compat
    return p


MUTATIONS = [
    ("swap_labels",           mut_swap_labels),
    ("break_active_pair",     mut_break_active_pair),
    ("break_violation_pair",  mut_break_violation_pair),
    ("disliked_A",            mut_disliked_A),
    ("non_neutral_violation", mut_non_neutral_violation),
    ("tpo_flip",              mut_tpo_flip),
]


def validate(plan, profile):
    """Run the release validator exactly as scripts/validate_options.py does."""
    s_errs, _info = check_structure_and_pref(plan, profile)
    return s_errs + check_tpo(plan)


def pick_samples(plans, n=N_SAMPLES):
    """Deterministic spread over (track, active_axis), then plan_id order."""
    buckets = {}
    for p in sorted(plans, key=lambda x: x.get("plan_id") or x["query_id"]):
        if axis_roles(p) is None:
            continue
        buckets.setdefault((p.get("track"), p.get("active_axis")), []).append(p)
    out, i = [], 0
    keys = sorted(buckets, key=lambda k: (str(k[0]), str(k[1])))
    while len(out) < n and keys:
        for k in keys:
            if i < len(buckets[k]):
                out.append(buckets[k][i])
                if len(out) == n:
                    break
        i += 1
        if i > 1000:
            break
    return out[:n]


def main():
    plans = load_jsonl(OPTIONS_DIR / "option_plans.jsonl")
    profiles = {p["user_id"]: p for p in load_jsonl(PROFILES_DIR / "profiles.jsonl")}
    if not plans or not profiles:
        sys.exit("  [error] could not load plans/profiles — set POD_VARIANT "
                 "(e.g. POD_VARIANT=wacv_scenario_v2)")

    samples = pick_samples(plans)
    if len(samples) < N_SAMPLES:
        sys.exit(f"  [error] only found {len(samples)} usable sample plans, "
                 f"need {N_SAMPLES}")

    # Sanity: the unmutated samples must PASS, else "detected" is meaningless.
    for p in samples:
        errs = validate(p, profiles[p["user_id"]])
        if errs:
            sys.exit(f"  [error] baseline plan already fails validation: "
                     f"{p.get('plan_id')} -> {errs}")

    print(f"  baseline: {len(samples)}/{len(samples)} unmutated plans pass\n")

    detected = attempted = 0
    misses = []
    for name, fn in MUTATIONS:
        hits = 0
        for p in samples:
            prof = profiles[p["user_id"]]
            mutated = fn(p, prof)
            if mutated is None:
                # inapplicable to this plan: count it as an attempt that the
                # suite must not silently drop
                misses.append((name, p.get("plan_id"), "inapplicable"))
                attempted += 1
                continue
            attempted += 1
            if validate(mutated, prof):
                detected += 1
                hits += 1
            else:
                misses.append((name, p.get("plan_id"), "UNDETECTED"))
        print(f"  {name:24s} {hits}/{len(samples)} detected")

    print(f"\n  {detected}/{attempted} detected")
    if misses:
        print("\n  PROBLEMS:")
        for name, pid, why in misses:
            print(f"    {why:12s} {name} on {pid}")
        sys.exit(1)
    print("  OK — the validator catches every injected defect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
