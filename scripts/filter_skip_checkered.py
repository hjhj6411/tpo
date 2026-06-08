#!/usr/bin/env python3
"""
filter_skip_checkered.py — drop any query whose options contain a target pattern
--------------------------------------------------------------------------------
FashionSigLIP retrieves most patterns acceptably but struggles specifically with
`checkered`. For a fast v1 image-based benchmark we simply SKIP any query that
has `checkered` as the target pattern in ANY of its four options (A/B/C/D), so
those queries never enter collection / grading / eval.

Filtering at the PLAN level (not the collector) means every downstream step
inherits the skip automatically and the original file is untouched.

  python filter_skip_checkered.py \
    --in  data/options/option_plans.jsonl \
    --out data/options/option_plans_nochk.jsonl
    # default skips: checkered. add more with --skip-patterns checkered,plaid

Prints what was removed and the active-axis breakdown of the survivors so you can
see how much the pattern axis shrinks.
"""
import argparse, json
from collections import Counter
from pathlib import Path


def load_jsonl(p):
    out = []
    for l in open(p):
        l = l.strip()
        if l:
            out.append(json.loads(l))
    return out


def norm(p):
    return (p or "solid").strip().lower()


def option_patterns(plan):
    out = []
    for k in "ABCD":
        a = ((plan.get("options") or {}).get(k) or {}).get("attributes", {}) or {}
        out.append(norm(a.get("pattern")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-patterns", default="checkered",
                    help="comma-separated patterns; a query is dropped if ANY option targets one")
    args = ap.parse_args()

    skip = {s.strip().lower() for s in args.skip_patterns.split(",") if s.strip()}
    plans = load_jsonl(args.inp)

    kept, dropped = [], []
    for p in plans:
        pats = option_patterns(p)
        if any(pat in skip for pat in pats):
            dropped.append(p)
        else:
            kept.append(p)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for p in kept:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"  skip patterns: {sorted(skip)}")
    print(f"  total={len(plans)}  kept={len(kept)}  dropped={len(dropped)} "
          f"({len(dropped)/max(len(plans),1):.1%})")

    ax_all = Counter(p.get("active_axis") for p in plans)
    ax_keep = Counter(p.get("active_axis") for p in kept)
    print("\n  active_axis    before -> after")
    for ax in sorted(ax_all):
        print(f"    {str(ax):10s}  {ax_all[ax]:5d} -> {ax_keep.get(ax,0):5d}")

    # which axis the dropped queries were on (checkered should mostly hit pattern axis)
    drop_ax = Counter(p.get("active_axis") for p in dropped)
    if dropped:
        print("\n  dropped queries by active_axis:")
        for ax, n in drop_ax.most_common():
            print(f"    {str(ax):10s} {n}")
    print(f"\n  -> {args.out}")


if __name__ == "__main__":
    main()
