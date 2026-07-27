"""Track-separated balance report (redesign lever A).

Physical and Dress-code are two SEPARATE datasets. Sample sizes and axis shares
are never compared across them, so this report prints no cross-track percentage.
Within each track the axes (garment / color / pattern) are reported separately:

  * garment dominating the *violation* axis is an accepted structural limit
    (you can wear orange in any season, but not a tank top in winter), handled
    by per-axis reporting rather than forced 1:1:1 balance;
  * within a single axis, no A-vs-B value pair may dominate (e.g. color must not
    collapse to red-vs-blue while green-vs-red is <1%) — the vs-pair balance
    this report surfaces per track and per axis.

WHY RAW max% IS NOT THE METRIC
------------------------------
An axis whose vocabulary admits only k pairs has a uniform share of 1/k, so a
small-k axis looks "concentrated" at a share that is in fact near-uniform. Every
concentration figure is therefore printed against its uniform baseline as a
ratio. Worked example: dress-code pattern violations admit 6 pairs (4 values,
C(4,2)=6, all realized), so max=30% is 1.8x uniform — the *flattest* axis in the
benchmark — whereas dress-code garment preference spreads over 44 pairs yet its
top-3 reach 45% against a 6.8% uniform expectation, i.e. 6.6x.

PER-USER CONCENTRATION
----------------------
Pooled pair distributions hide repetition inside a single user's items. Because
each profile fixes one liked and one disliked value per anchor tier, a user's
A/B pair can repeat across many scenarios; this is the cost of guaranteeing that
every user always has a valid A/B. For a personalization benchmark the
per-user distinct-pair count is the number reviewers recompute first, so it is
reported alongside the pooled view.

Usage:
    python -m scripts.report_track_balance                 # default data dir
    python -m scripts.report_track_balance --plans PATH
    python -m scripts.report_track_balance --full          # every pair, not top-3
    POD_VARIANT=wacv_scenario_v2 python -m scripts.report_track_balance
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import OPTIONS_DIR
from configs.scenarios import TRACK_PHYSICAL, TRACK_DRESS_CODE, scenario_track, get_scenario_by_id

AXES = ("garment_category", "color", "pattern")
TRACKS = (TRACK_PHYSICAL, TRACK_DRESS_CODE)


def _plan_track(p):
    t = p.get("track")
    if t:
        return t
    sc = get_scenario_by_id(p["scenario_id"])
    return scenario_track(sc) if sc else "unknown"


def _active_pair(p):
    ax = p["active_axis"]
    a = p["options"]["A"]["attributes"].get(ax)
    b = p["options"]["B"]["attributes"].get(ax)
    return tuple(sorted((str(a), str(b))))


def _violation_pair(p):
    return tuple(sorted((str(p.get("tpo_compatible_value")),
                         str(p.get("violation_value")))))


def _fmt_pairs(pairs, full=False):
    """Concentration against the uniform baseline for this pair count.

    With k pairs, uniform gives each 1/k and the top-3 min(3,k)/k; dividing by
    those makes axes with different k comparable.
    """
    tot = sum(pairs.values())
    if not tot:
        return "  (none)"
    k = len(pairs)
    unif = 1 / k
    unif3 = min(3, k) / k
    hi = max(pairs.values()) / tot
    top3 = sum(v for _, v in pairs.most_common(3)) / tot
    lo = min(pairs.values()) / tot
    head = (f"pairs={k:2d}  max={hi:5.1%} ({hi/unif:.1f}x unif)  "
            f"top3={top3:5.1%} ({top3/unif3:.1f}x unif)  min={lo:.1%}")
    listed = pairs.most_common() if full else pairs.most_common(3)
    body = ", ".join(f"{'/'.join(kk)}={v}({v/tot:.0%})" for kk, v in listed)
    label = "all" if full else "top"
    return f"{head}\n         {label}: {body}"


def _per_user_active(bp, ax):
    """(plans/user, distinct-pair min/med/max, top-pair share avg/worst) or None."""
    per = defaultdict(Counter)
    for p in bp:
        if p["active_axis"] == ax:
            per[p["user_id"]][_active_pair(p)] += 1
    if not per:
        return None
    counts = sorted(len(c) for c in per.values())
    tops = [c.most_common(1)[0][1] / sum(c.values()) for c in per.values()]
    n_per = [sum(c.values()) for c in per.values()]
    return (sum(n_per) / len(n_per), counts[0], counts[len(counts) // 2], counts[-1],
            sum(tops) / len(tops), max(tops))


def report(plans, full=False):
    by_track = defaultdict(list)
    for p in plans:
        by_track[_plan_track(p)].append(p)

    print("=" * 78)
    print("TRACK-SEPARATED BALANCE REPORT")
    print("Physical and Dress-code are separate datasets; no figure below is")
    print("comparable across them. Concentration is stated relative to the")
    print("uniform baseline for that axis's pair count.")
    print("=" * 78)

    skew = []
    for tr in TRACKS:
        bp = by_track.get(tr, [])
        print("\n" + "=" * 78)
        print(f"{tr.upper()}   —   {len(bp)} plans, "
              f"{len({p['user_id'] for p in bp})} users, "
              f"{len({p['scenario_id'] for p in bp})} scenarios")
        print("=" * 78)

        act = Counter(p["active_axis"] for p in bp)
        vio = Counter(p["violation_axis"] for p in bp)
        s_a, s_v = sum(act.values()), sum(vio.values())
        print("  active-axis (preference probed): " +
              ", ".join(f"{a}={n}({n/max(s_a,1):.1%})" for a, n in act.most_common()))
        print("  violation-axis (TPO tested)    : " +
              ", ".join(f"{a}={n}({n/max(s_v,1):.1%})" for a, n in vio.most_common()))

        for role, pair_fn, key in (
                ("ACTIVE (preference A-vs-B)", _active_pair, "active_axis"),
                ("VIOLATION (compatible-vs-incompatible)", _violation_pair, "violation_axis")):
            print(f"\n  -- within-axis vs-pair balance: {role} --")
            for ax in AXES:
                sub = [p for p in bp if p[key] == ax]
                if not sub:
                    continue
                pairs = Counter(pair_fn(p) for p in sub)
                print(f"    {ax:16s} n={sum(pairs.values()):4d}  {_fmt_pairs(pairs, full)}")
                k = len(pairs)
                top3 = sum(v for _, v in pairs.most_common(3)) / sum(pairs.values())
                skew.append((top3 / (min(3, k) / k), tr, role.split()[0].lower(), ax, k, top3))

        print("\n  -- per-user concentration on the ACTIVE (preference) axis --")
        print(f"    {'axis':16s} {'plans/user':>11} {'distinct pairs':>22} "
              f"{'top-pair share':>20}")
        for ax in AXES:
            st = _per_user_active(bp, ax)
            if st is None:
                continue
            npu, lo, med, hi, avg_top, worst_top = st
            print(f"    {ax:16s} {npu:11.1f} "
                  f"{f'min {lo} / med {med} / max {hi}':>22} "
                  f"{f'avg {avg_top:.0%} / worst {worst_top:.0%}':>20}")

        pu = Counter(p["user_id"] for p in bp)
        if pu:
            vals = sorted(pu.values())
            print(f"\n  per-user plans: min={vals[0]} max={vals[-1]} "
                  f"avg={sum(vals)/len(vals):.1f}  (n_users={len(pu)})")

    print("\n" + "=" * 78)
    print("AXES RANKED BY TOP-3 CONCENTRATION RELATIVE TO UNIFORM")
    print("(the disclosure order for a limitations section: most skewed first)")
    print("=" * 78)
    for ratio, tr, role, ax, k, top3 in sorted(skew, reverse=True):
        print(f"  {ratio:5.2f}x  {tr:11s} {role:10s} {ax:17s} "
              f"pairs={k:3d}  top3={top3:5.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", type=Path, default=OPTIONS_DIR / "option_plans.jsonl")
    ap.add_argument("--full", action="store_true",
                    help="list every realized pair instead of the top 3")
    args = ap.parse_args()
    if not args.plans.exists():
        sys.exit(f"plans file not found: {args.plans}")
    plans = [json.loads(line) for line in open(args.plans)]
    report(plans, full=args.full)


if __name__ == "__main__":
    main()
