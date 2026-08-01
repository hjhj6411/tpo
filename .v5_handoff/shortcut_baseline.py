"""Shortcut baselines: how far does a model get WITHOUT understanding anything?

Motivation
----------
In the normative track the TPO-compatible pattern value is structurally almost
always `solid`, because formal dress codes admit essentially no other pattern.
A model could therefore raise its TPO score by always preferring the plain
option, with no understanding of either the situation or the user.

This script quantifies the ceiling of that shortcut so the paper can report it
next to real model accuracy. It reads option plans directly and needs no API
calls.

Baselines
---------
  random            uniform over A-D (sanity floor)
  always_solid      prefer options whose pattern is `solid`; tie -> first
  never_solid       inverse control; if `always_solid` helps, this must hurt
  majority_pattern  pick the pattern value shared by most options
  first_option      positional control (options are shuffled at eval time, so
                    this should sit at chance)

Scoring follows scripts/text_only_eval.py:
  strict     = picked A
  tpo        = picked A or B   (TPO_SCORE)
  preference = picked A or C   (PROFILE_SCORE)

Usage
    POD_VARIANT=wacv_scenario_v5 python -m scripts.shortcut_baseline
    POD_VARIANT=wacv_scenario_v5 python -m scripts.shortcut_baseline --json out.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from configs.config import OPTIONS_DIR

TPO_SCORE = {"A": 1, "B": 1, "C": 0, "D": 0}
PROFILE_SCORE = {"A": 1, "B": 0, "C": 1, "D": 0}
LABELS = ["A", "B", "C", "D"]

CELL_LIBRARY = Path(__file__).resolve().parent.parent / "annotation" / "attribute_library.json"


def load_available_cells():
    if not CELL_LIBRARY.exists():
        return None
    lib = json.loads(CELL_LIBRARY.read_text())
    return {k for k, v in lib.get("cells", {}).items() if v.get("status") == "available"}


def is_realizable(plan, available):
    """True when all four options sit on an annotated, available image cell."""
    if available is None:
        return True
    for lab in LABELS:
        a = plan["options"][lab]["attributes"]
        c, g, p = a.get("color"), a.get("garment_category"), a.get("pattern")
        if not (c and g and p) or f"{c}|{g}|{p}" not in available:
            return False
    return True


def pick_always_solid(plan, rng):
    hits = [l for l in LABELS if plan["options"][l]["attributes"].get("pattern") == "solid"]
    return rng.choice(hits) if hits else rng.choice(LABELS)


def pick_never_solid(plan, rng):
    hits = [l for l in LABELS if plan["options"][l]["attributes"].get("pattern") != "solid"]
    return rng.choice(hits) if hits else rng.choice(LABELS)


def pick_majority_pattern(plan, rng):
    counts = defaultdict(list)
    for l in LABELS:
        counts[plan["options"][l]["attributes"].get("pattern")].append(l)
    best = max(counts.values(), key=len)
    return rng.choice(best)


BASELINES = {
    "random":           lambda plan, rng: rng.choice(LABELS),
    "first_option":     lambda plan, rng: "A",
    "always_solid":     pick_always_solid,
    "never_solid":      pick_never_solid,
    "majority_pattern": pick_majority_pattern,
}


def evaluate(plans, seed=42, trials=200):
    """Average over `trials` tie-breaking draws so the numbers are stable."""
    out = {}
    for name, fn in BASELINES.items():
        by_track = defaultdict(lambda: {"strict": 0.0, "tpo": 0.0, "pref": 0.0, "n": 0})
        by_vaxis = defaultdict(lambda: {"tpo": 0.0, "n": 0})
        for plan in plans:
            s = t = p = 0.0
            reps = 1 if name in ("first_option",) else trials
            rng = random.Random(seed + hash(plan["plan_id"]) % 10_000)
            for _ in range(reps):
                pick = fn(plan, rng)
                s += pick == "A"
                t += TPO_SCORE.get(pick, 0)
                p += PROFILE_SCORE.get(pick, 0)
            s, t, p = s / reps, t / reps, p / reps
            for key in (plan["track"], "ALL"):
                d = by_track[key]
                d["strict"] += s; d["tpo"] += t; d["pref"] += p; d["n"] += 1
            k = f"{plan['track']}/violation={plan['violation_axis']}"
            by_vaxis[k]["tpo"] += t; by_vaxis[k]["n"] += 1
        out[name] = {
            "track": {k: {m: v[m] / v["n"] for m in ("strict", "tpo", "pref")} | {"n": v["n"]}
                      for k, v in by_track.items()},
            "violation_axis": {k: {"tpo": v["tpo"] / v["n"], "n": v["n"]}
                               for k, v in by_vaxis.items()},
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plans", type=Path, default=OPTIONS_DIR / "option_plans.jsonl")
    ap.add_argument("--all-plans", action="store_true",
                    help="score every plan, not just image-realizable ones")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--json", type=Path, help="also write raw numbers here")
    args = ap.parse_args()

    plans = [json.loads(l) for l in args.plans.read_text().splitlines() if l.strip()]
    available = None if args.all_plans else load_available_cells()
    kept = [p for p in plans if is_realizable(p, available)]

    print(f"\n  plans: {len(kept)} scored"
          f"{'' if args.all_plans else f' (of {len(plans)}; image-realizable only)'}")

    res = evaluate(kept, trials=args.trials)

    print("\n" + "=" * 72)
    print("  SHORTCUT BASELINES — accuracy without any understanding")
    print("=" * 72)
    tracks = sorted({t for r in res.values() for t in r["track"] if t != "ALL"}) + ["ALL"]
    for track in tracks:
        n = res["random"]["track"][track]["n"]
        print(f"\n  [{track}]  n={n}")
        print(f"    {'baseline':18s} {'strict':>8s} {'tpo':>8s} {'preference':>11s}")
        for name in BASELINES:
            d = res[name]["track"].get(track)
            if not d:
                continue
            print(f"    {name:18s} {d['strict']:8.1%} {d['tpo']:8.1%} {d['pref']:11.1%}")

    print("\n" + "-" * 72)
    print("  TPO accuracy by violation axis (where the shortcut can bite)")
    print("-" * 72)
    keys = sorted(res["always_solid"]["violation_axis"])
    print(f"    {'slice':38s} {'n':>6s} {'random':>8s} {'always_solid':>13s}")
    for k in keys:
        a = res["always_solid"]["violation_axis"][k]
        r = res["random"]["violation_axis"][k]
        print(f"    {k:38s} {a['n']:6d} {r['tpo']:8.1%} {a['tpo']:13.1%}")

    gap = (res["always_solid"]["track"]["ALL"]["tpo"]
           - res["random"]["track"]["ALL"]["tpo"])
    print(f"\n  always_solid TPO advantage over chance (all plans): {gap:+.1%}")
    inv = (res["never_solid"]["track"]["ALL"]["tpo"]
           - res["random"]["track"]["ALL"]["tpo"])
    print(f"  never_solid  TPO advantage over chance (control):    {inv:+.1%}")

    if args.json:
        args.json.write_text(json.dumps(res, indent=2))
        print(f"\n  raw numbers → {args.json}")
    print()


if __name__ == "__main__":
    main()
