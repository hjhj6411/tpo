#!/usr/bin/env python3
"""
validate_options.py — Construction validity & confound audit for POD-Bench v2 (clean 2x2).

Paths default to the POD_VARIANT-aware data root (configs.config), so run it the
same way you run the generators:

    POD_VARIANT=wacv_scenario_v1 python -m scripts.validate_options
    # or override every path explicitly
    python scripts/validate_options.py \
        --profiles data_wacv_scenario_v1/profiles/profiles.jsonl \
        --queries  data_wacv_scenario_v1/queries/queries.jsonl \
        --plans    data_wacv_scenario_v1/options/option_plans.jsonl \
        --out      data_wacv_scenario_v1/options/validation_report.json

AXIS ROLES (generalized 2x2)
----------------------------
Each plan names two distinct axes; the remaining third axis is held fixed.
  active_axis     — carries PREFERENCE (A/C take the liked value, B/D do not)
  violation_axis  — carries TPO        (A/B take the compatible value, C/D do not)
  fixed axis      — constant across all four options, preference-neutral
Both roles may be any of color / pattern / garment_category. In particular
garment may be the active axis (dress-code garment preference) and color or
pattern may be the violation axis (dress-code color/pattern norms); an earlier
version of this script assumed violation==garment and active in {color,pattern}
and therefore reported those plans as failures.

WHAT IT CHECKS
--------------
Per instance:
  [S] Structural 2x2 integrity
        labels A/B/C/D == tpo_and_preference/tpo_only/preference_only/neither
        active_axis != violation_axis, both in the canonical axis set
        A.active == C.active (liked), B.active == D.active (non-preferred), A.active != B.active
        A.violation == B.violation (TPO-compat), C.violation == D.violation (incompat), A != C
        the fixed axis is identical across all four
        violation_value / tpo_compatible_value agree with the options
  [P] Preference correctness (needs profile)
        A.active in user likes; B.active NOT in user likes
        every violation-axis value is preference-neutral (not in likes, not in dislikes)
        fixed-axis value is preference-neutral
  [T] TPO correctness vs scenario compatible/incompatible sets (needs configs.scenarios)
        A/B violation value in scenario compatible; C/D in scenario incompatible
        A/B active value not in scenario incompatible; fixed value not in scenario incompatible
  [U] Uniqueness: the four attribute tuples are distinct

Dataset-level confound diagnostics (explains the WITHOUT-profile strict > 50%):
  - per-axis value frequency table as A(liked) vs B(non-preferred)
  - preference-blind exploit accuracy: pick the value with the higher global liked-rate;
    ~50% == balanced, >>50% == exploitable value-frequency bias
  - "always pick solid" blind baseline (pattern)
  - balance counts by axis / scenario_archetype / preference_archetype / user / query_type
  - value-balanced subset: instances whose A-value and B-value each appear in BOTH roles
    -> ids written to <out>.balanced_ids.json for a clean strict-accuracy re-analysis
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ── optional scenario import (for TPO checks) ──────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from configs.scenarios import get_scenario_by_id
    HAVE_SCENARIOS = True
except Exception as e:  # pragma: no cover
    print(f"  [warn] could not import configs.scenarios ({e}); skipping TPO checks")
    HAVE_SCENARIOS = False
    def get_scenario_by_id(_sid):  # type: ignore
        return None

try:
    from configs.config import PROFILES_DIR, QUERIES_DIR, OPTIONS_DIR
except Exception as e:  # pragma: no cover
    print(f"  [warn] could not import configs.config ({e}); falling back to ./data")
    PROFILES_DIR = Path("data/profiles")
    QUERIES_DIR = Path("data/queries")
    OPTIONS_DIR = Path("data/options")

LABELS = {"A": "tpo_and_preference", "B": "tpo_only",
          "C": "preference_only", "D": "neither"}

AXES = ("color", "pattern", "garment_category")


def axis_roles(plan):
    """(active, violation, fixed) for a plan, or None if the pair is unusable."""
    act, vio = plan.get("active_axis"), plan.get("violation_axis")
    if act not in AXES or vio not in AXES or act == vio:
        return None
    return act, vio, next(a for a in AXES if a not in (act, vio))


def load_jsonl(path):
    path = Path(path)
    out = []
    if not path.exists():
        print(f"  [error] missing file: {path}")
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def prefs(profile, axis):
    sa = profile.get("structured_attributes", {})
    ax = sa.get(axis, {})
    return set(ax.get("likes", [])), set(ax.get("dislikes", []))


# ── per-instance structural + preference checks ────────────────────────

def check_structure_and_pref(plan, profile):
    errs = []
    opts = plan.get("options", {})

    if set(opts.keys()) != {"A", "B", "C", "D"}:
        return ["missing_options"], None
    for k in "ABCD":
        if opts[k].get("label") != LABELS[k]:
            errs.append(f"label_{k}={opts[k].get('label')}")

    roles = axis_roles(plan)
    if roles is None:
        errs.append(f"bad_axis_roles=active:{plan.get('active_axis')}"
                    f"/violation:{plan.get('violation_axis')}")
        return errs, None
    axis, vaxis, faxis = roles

    attrs = {k: opts[k]["attributes"] for k in "ABCD"}
    a = {k: attrs[k].get(axis) for k in "ABCD"}     # preference axis
    v = {k: attrs[k].get(vaxis) for k in "ABCD"}    # TPO axis

    # structural 2x2 — preference axis pairs A/C and B/D, TPO axis pairs A/B and C/D
    if a["A"] != a["C"]: errs.append("A.active!=C.active")
    if a["B"] != a["D"]: errs.append("B.active!=D.active")
    if a["A"] == a["B"]: errs.append("A.active==B.active(degenerate)")
    if v["A"] != v["B"]: errs.append("A.violation!=B.violation")
    if v["C"] != v["D"]: errs.append("C.violation!=D.violation")
    if v["A"] == v["C"]: errs.append("A.violation==C.violation(degenerate)")

    fvals = {attrs[k].get(faxis) for k in "ABCD"}
    if len(fvals) > 1:
        errs.append(f"fixed_{faxis}_not_constant:{sorted(map(str, fvals))}")
    fval = next(iter(fvals)) if fvals else None

    # declared summary fields must agree with the options they summarize
    if plan.get("tpo_compatible_value") not in (None, v["A"]):
        errs.append("tpo_compatible_value_mismatch")
    if plan.get("violation_value") not in (None, v["C"]):
        errs.append("violation_value_mismatch")

    # preference correctness
    if profile is not None:
        likes, _ = prefs(profile, axis)
        if a["A"] not in likes:
            errs.append(f"A.active({a['A']})_not_in_likes")
        if a["B"] in likes:
            errs.append(f"B.active({a['B']})_in_likes")

        vlikes, vdis = prefs(profile, vaxis)
        for k in "ABCD":
            if v[k] in vlikes or v[k] in vdis:
                errs.append(f"{k}.violation({v[k]})_not_neutral")

        flikes, fdis = prefs(profile, faxis)
        if fval is not None and (fval in flikes or fval in fdis):
            errs.append(f"fixed_{faxis}({fval})_not_neutral")

    # uniqueness
    tuples = [tuple(sorted(attrs[k].items())) for k in "ABCD"]
    if len(set(tuples)) < 4:
        errs.append("duplicate_options")

    info = {"axis": axis, "a_val": a["A"], "b_val": a["B"],
            "violation_axis": vaxis, "compat_val": v["A"], "incompat_val": v["C"],
            "fixed_axis": faxis, "fixed_val": fval}
    return errs, info


def check_tpo(plan):
    if not HAVE_SCENARIOS:
        return []
    sc = get_scenario_by_id(plan.get("scenario_id", ""))
    if sc is None:
        return ["scenario_not_found"]
    roles = axis_roles(plan)
    if roles is None:
        return []  # already reported by the structural check
    axis, vaxis, faxis = roles
    errs = []
    opts = plan["options"]

    # the violation axis is the one the scenario must constrain
    vc = sc.get(vaxis) or {}
    vcomp, vinc = set(vc.get("compatible", [])), set(vc.get("incompatible", []))
    vv = {k: opts[k]["attributes"].get(vaxis) for k in "ABCD"}
    if not vinc:
        errs.append(f"scenario_has_no_{vaxis}_constraint")
    else:
        for k in "AB":
            if vv[k] not in vcomp:
                errs.append(f"{k}.violation({vv[k]})_not_TPO_compat")
        for k in "CD":
            if vv[k] not in vinc:
                errs.append(f"{k}.violation({vv[k]})_not_TPO_incompat")

    # the preference and fixed axes must stay clear of the scenario's bans
    ainc = set((sc.get(axis) or {}).get("incompatible", []))
    for k in "AB":
        av = opts[k]["attributes"].get(axis)
        if ainc and av in ainc:
            errs.append(f"{k}.active({av})_in_TPO_incompat")

    finc = set((sc.get(faxis) or {}).get("incompatible", []))
    fv = opts["A"]["attributes"].get(faxis)
    if finc and fv in finc:
        errs.append(f"fixed_{faxis}({fv})_in_TPO_incompat")
    return errs


# ── main ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default=str(PROFILES_DIR / "profiles.jsonl"))
    ap.add_argument("--queries", default=str(QUERIES_DIR / "queries.jsonl"))
    ap.add_argument("--plans", default=str(OPTIONS_DIR / "option_plans.jsonl"))
    ap.add_argument("--out", default=str(OPTIONS_DIR / "validation_report.json"))
    args = ap.parse_args()

    print(f"\n  data root: {OPTIONS_DIR.parent}"
          f"  (POD_VARIANT={os.environ.get('POD_VARIANT', '') or '<unset>'})")

    profiles = {p["user_id"]: p for p in load_jsonl(args.profiles)}
    queries = {q["query_id"]: q for q in load_jsonl(args.queries)}
    plans = load_jsonl(args.plans)
    print(f"\n  profiles={len(profiles)}  queries={len(queries)}  plans={len(plans)}")

    # downstream-readiness warning (per-archetype analysis depends on these)
    if queries:
        sample_q = next(iter(queries.values()))
        if "scenario_archetype" not in sample_q or "scenario_name" not in sample_q:
            print("  [warn] queries are missing scenario_archetype / scenario_name -> "
                  "quality_audit & evaluator per-archetype slices will be null.")

    n = len(plans)
    n_struct_fail = n_tpo_fail = 0
    err_counter = Counter()
    failures = []

    # confound accumulators
    freqA = defaultdict(Counter)   # axis -> Counter(value used as A/liked)
    freqB = defaultdict(Counter)   # axis -> Counter(value used as B/non-preferred)
    by = {k: Counter() for k in
          ("axis", "scenario_archetype", "preference_archetype", "user", "query_type")}
    inst_meta = []  # (axis, a_val, b_val, query_id)
    # Physical and Dress-code are reported as two separate datasets.
    track_stats = defaultdict(lambda: {"n": 0, "struct": 0, "tpo": 0,
                                       "active": Counter(), "violation": Counter()})

    for plan in plans:
        qid = plan.get("query_id")
        prof = profiles.get(plan.get("user_id"))
        q = queries.get(qid, {})

        s_errs, info = check_structure_and_pref(plan, prof)
        t_errs = check_tpo(plan)
        all_errs = s_errs + t_errs
        if s_errs:
            n_struct_fail += 1
        if t_errs:
            n_tpo_fail += 1

        ts = track_stats[plan.get("track", "?")]
        ts["n"] += 1
        ts["struct"] += bool(s_errs)
        ts["tpo"] += bool(t_errs)
        ts["active"][plan.get("active_axis", "?")] += 1
        ts["violation"][plan.get("violation_axis", "?")] += 1
        for e in all_errs:
            err_counter[e.split(":")[0].split("(")[0]] += 1
        if all_errs:
            failures.append({"query_id": qid, "errors": all_errs})

        if info:
            ax = info["axis"]
            freqA[ax][info["a_val"]] += 1
            freqB[ax][info["b_val"]] += 1
            inst_meta.append((ax, info["a_val"], info["b_val"], qid))
            by["axis"][ax] += 1
            by["scenario_archetype"][q.get("scenario_archetype", "?")] += 1
            by["preference_archetype"][
                (prof or {}).get("preference_archetype", "?")] += 1
            by["user"][plan.get("user_id", "?")] += 1
            by["query_type"][q.get("query_type", "?")] += 1

    # ── report: integrity ──────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  CONSTRUCTION INTEGRITY")
    print("=" * 64)
    print(f"  structural/preference failures: {n_struct_fail}/{n}")
    print(f"  TPO failures:                   {n_tpo_fail}/{n}"
          + ("" if HAVE_SCENARIOS else "  (skipped: no scenarios import)"))
    if err_counter:
        print("  error types (count):")
        for e, c in err_counter.most_common():
            print(f"    {e:42s} {c}")
    else:
        print("  ✓ no construction errors")

    # per-track view: the two tracks are separate datasets, never pooled
    print("\n  per track (reported as independent datasets):")
    for tname in sorted(track_stats):
        ts = track_stats[tname]
        share = lambda c: ", ".join(f"{k}={v} ({v / ts['n']:.1%})"
                                    for k, v in c.most_common())
        print(f"    [{tname}]  {ts['n']} plans   "
              f"structural={ts['struct']}  TPO={ts['tpo']}")
        print(f"       active axis    : {share(ts['active'])}")
        print(f"       violation axis : {share(ts['violation'])}")

    # ── report: balance ────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  DISTRIBUTION")
    print("=" * 64)
    for key in ("axis", "query_type", "scenario_archetype",
                "preference_archetype"):
        print(f"  by {key}:")
        for v, c in by[key].most_common():
            print(f"    {str(v):28s} {c}")

    # ── report: value-frequency confound ───────────────────────────────
    print("\n" + "=" * 64)
    print("  CONFOUND AUDIT  (active-axis value bias)")
    print("=" * 64)
    print("  Per value:  nA = times used as A(liked), nB = times used as B(non-pref),")
    print("              liked_rate = nA / (nA + nB).  ~0.5 is balanced.\n")

    liked_rate = {}  # axis -> {value: rate}
    for ax in sorted(set(freqA) | set(freqB)):
        liked_rate[ax] = {}
        vals = sorted(set(freqA[ax]) | set(freqB[ax]))
        print(f"  [{ax}]")
        for v in sorted(vals, key=lambda x: -(freqA[ax][x] + freqB[ax][x])):
            nA, nB = freqA[ax][v], freqB[ax][v]
            rate = nA / (nA + nB) if (nA + nB) else 0.0
            liked_rate[ax][v] = rate
            flag = "  <-- only-A (fully exploitable)" if nB == 0 and nA else (
                   "  <-- only-B" if nA == 0 and nB else "")
            print(f"    {str(v):16s} nA={nA:4d} nB={nB:4d} liked_rate={rate:.2f}{flag}")
        print()

    # preference-blind exploit accuracy
    print("  Preference-blind exploit accuracy")
    print("  (predict the value with higher global liked_rate as the answer A;")
    print("   compare to the WITHOUT-profile strict numbers you observed):")
    blind_by_axis = defaultdict(lambda: [0.0, 0])
    solid_hit = solid_tot = 0
    for ax, a_val, b_val, _qid in inst_meta:
        ra = liked_rate[ax].get(a_val, 0.0)
        rb = liked_rate[ax].get(b_val, 0.0)
        hit = 1.0 if ra > rb else (0.5 if ra == rb else 0.0)
        blind_by_axis[ax][0] += hit
        blind_by_axis[ax][1] += 1
        if ax == "pattern":
            solid_tot += 1
            if a_val == "solid" and b_val != "solid":
                solid_hit += 1
            elif b_val == "solid" and a_val != "solid":
                solid_hit += 0
            else:
                solid_hit += 0.5
    tot_hit = tot_n = 0
    for ax, (h, c) in blind_by_axis.items():
        tot_hit += h; tot_n += c
        print(f"    {ax:10s} blind_acc = {h / max(c,1):.3f}  (n={c})")
    print(f"    {'overall':10s} blind_acc = {tot_hit / max(tot_n,1):.3f}  (n={tot_n})")
    if solid_tot:
        print(f"    pattern 'always-solid' baseline = {solid_hit / solid_tot:.3f}")

    # ── counterbalanced subset (matched-orientation pairs) ─────────────
    # Presence in both roles is NOT enough (solid has nB=35>0 yet liked_rate=0.90).
    # A value-prior is only neutralized when each value plays A and B EQUALLY.
    # For each unordered pair {x,y}, keep min(#(A=x,B=y), #(A=y,B=x)) of each
    # orientation -> blind value-prior == 50% on the resulting subset by construction.
    orient = defaultdict(lambda: defaultdict(list))   # axis -> frozenset{x,y} -> [(qid, a_val)]
    for ax, a, b, qid in inst_meta:
        orient[ax][frozenset((a, b))].append((qid, a))
    cb_ids = []
    for ax, pairs in orient.items():
        for pair, items in pairs.items():
            if len(pair) < 2:
                continue
            x, y = tuple(pair)
            xy = [qid for qid, a in items if a == x]   # A == x
            yx = [qid for qid, a in items if a == y]   # A == y
            m = min(len(xy), len(yx))
            cb_ids += xy[:m] + yx[:m]
    print("\n  Counterbalanced subset (each value plays A and B equally;")
    print("  blind value-prior == 50% by construction -> confound-free strict pool):")
    print(f"    {len(cb_ids)}/{len(inst_meta)} instances "
          f"({len(cb_ids)/max(len(inst_meta),1):.0%})")

    # ── write report ───────────────────────────────────────────────────
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "n_plans": n,
        "n_structural_failures": n_struct_fail,
        "n_tpo_failures": n_tpo_fail,
        "error_types": dict(err_counter),
        "failures": failures[:500],
        "blind_exploit_acc": {ax: blind_by_axis[ax][0] / max(blind_by_axis[ax][1], 1)
                              for ax in blind_by_axis},
        "liked_rate": liked_rate,
        "counterbalanced_subset_size": len(cb_ids),
        "distribution": {k: dict(v) for k, v in by.items()},
        "by_track": {t: {"n": s["n"], "structural_failures": s["struct"],
                         "tpo_failures": s["tpo"],
                         "active_axis": dict(s["active"]),
                         "violation_axis": dict(s["violation"])}
                     for t, s in track_stats.items()},
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(out.with_suffix(".counterbalanced_ids.json"), "w", encoding="utf-8") as f:
        json.dump(cb_ids, f)
    print(f"\n  ✓ report -> {out}")
    print(f"  ✓ counterbalanced ids -> {out.with_suffix('.counterbalanced_ids.json')}\n")


if __name__ == "__main__":
    main()