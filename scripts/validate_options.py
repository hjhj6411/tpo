#!/usr/bin/env python3
"""
validate_options.py — Construction validity & confound audit for POD-Bench v2 (clean 2x2).

Drop into  tpo/scripts/  and run from the repo root:

    python -m scripts.validate_options
    # or
    python scripts/validate_options.py \
        --profiles data/profiles/profiles.jsonl \
        --queries  data/queries/queries.jsonl \
        --plans    data/options/option_plans.jsonl \
        --out      data/options/validation_report.json

WHAT IT CHECKS
--------------
Per instance:
  [S] Structural 2x2 integrity
        labels A/B/C/D == tpo_and_preference/tpo_only/preference_only/neither
        A.active == C.active (liked), B.active == D.active (non-preferred), A.active != B.active
        A.garment == B.garment (TPO-compat), C.garment == D.garment (TPO-incompat), A.g != C.g
        non-active axis (color<->pattern) identical across all four
  [P] Preference correctness (needs profile)
        A.active in user likes; B.active NOT in user likes
        every garment is preference-neutral (not in likes, not in dislikes)
        fixed non-active value is preference-neutral
  [T] TPO correctness vs scenario compatible/incompatible sets (needs configs.scenarios)
        A/B garment in scenario compatible; C/D garment in scenario incompatible
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

LABELS = {"A": "tpo_and_preference", "B": "tpo_only",
          "C": "preference_only", "D": "neither"}


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
    axis = plan.get("active_axis")
    opts = plan.get("options", {})

    if set(opts.keys()) != {"A", "B", "C", "D"}:
        return ["missing_options"], None
    for k in "ABCD":
        if opts[k].get("label") != LABELS[k]:
            errs.append(f"label_{k}={opts[k].get('label')}")
    if axis not in ("color", "pattern"):
        errs.append(f"active_axis_not_allowed={axis}")
        return errs, None

    A, B, C, D = (opts[k]["attributes"] for k in "ABCD")
    aA, aB, aC, aD = (x.get(axis) for x in (A, B, C, D))
    gA, gB, gC, gD = (x.get("garment_category") for x in (A, B, C, D))

    # structural 2x2
    if aA != aC: errs.append("A.active!=C.active")
    if aB != aD: errs.append("B.active!=D.active")
    if aA == aB: errs.append("A.active==B.active(degenerate)")
    if gA != gB: errs.append("A.garment!=B.garment")
    if gC != gD: errs.append("C.garment!=D.garment")
    if gA == gC: errs.append("A.garment==C.garment(degenerate)")

    other = "pattern" if axis == "color" else "color"
    ovals = {x.get(other) for x in (A, B, C, D)}
    if len(ovals) > 1:
        errs.append(f"fixed_{other}_not_constant:{sorted(map(str, ovals))}")
    oval = next(iter(ovals)) if ovals else None

    # preference correctness
    if profile is not None:
        likes, dislikes = prefs(profile, axis)
        if aA not in likes:
            errs.append(f"A.active({aA})_not_in_likes")
        if aB in likes:
            errs.append(f"B.active({aB})_in_likes")

        glikes, gdis = prefs(profile, "garment_category")
        for k, g in zip("ABCD", (gA, gB, gC, gD)):
            if g in glikes or g in gdis:
                errs.append(f"{k}.garment({g})_not_neutral")

        olikes, odis = prefs(profile, other)
        if oval is not None and (oval in olikes or oval in odis):
            errs.append(f"fixed_{other}({oval})_not_neutral")

    # uniqueness
    tuples = [tuple(sorted(x.items())) for x in (A, B, C, D)]
    if len(set(tuples)) < 4:
        errs.append("duplicate_options")

    info = {"axis": axis, "a_val": aA, "b_val": aB,
            "compat_garment": gA, "incompat_garment": gC,
            "fixed_axis": other, "fixed_val": oval}
    return errs, info


def check_tpo(plan):
    if not HAVE_SCENARIOS:
        return []
    sc = get_scenario_by_id(plan.get("scenario_id", ""))
    if sc is None:
        return ["scenario_not_found"]
    errs = []
    axis = plan["active_axis"]
    opts = plan["options"]

    gc = sc.get("garment_category") or {}
    gcomp, ginc = set(gc.get("compatible", [])), set(gc.get("incompatible", []))
    gv = {k: opts[k]["attributes"].get("garment_category") for k in "ABCD"}
    if gv["A"] not in gcomp: errs.append(f"A.garment({gv['A']})_not_TPO_compat")
    if gv["B"] not in gcomp: errs.append(f"B.garment({gv['B']})_not_TPO_compat")
    if gv["C"] not in ginc: errs.append(f"C.garment({gv['C']})_not_TPO_incompat")
    if gv["D"] not in ginc: errs.append(f"D.garment({gv['D']})_not_TPO_incompat")

    ac = sc.get(axis) or {}
    ainc = set(ac.get("incompatible", []))
    for k in "AB":
        v = opts[k]["attributes"].get(axis)
        if ainc and v in ainc:
            errs.append(f"{k}.active({v})_in_TPO_incompat")

    other = "pattern" if axis == "color" else "color"
    oc = sc.get(other) or {}
    oinc = set(oc.get("incompatible", []))
    ov = opts["A"]["attributes"].get(other)
    if oinc and ov in oinc:
        errs.append(f"fixed_{other}({ov})_in_TPO_incompat")
    return errs


# ── main ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default="data/profiles/profiles.jsonl")
    ap.add_argument("--queries", default="data/queries/queries.jsonl")
    ap.add_argument("--plans", default="data/options/option_plans.jsonl")
    ap.add_argument("--out", default="data/options/validation_report.json")
    args = ap.parse_args()

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

    # ── value-balanced subset ──────────────────────────────────────────
    balanced_vals = {ax: {v for v in (set(freqA[ax]) | set(freqB[ax]))
                          if freqA[ax][v] > 0 and freqB[ax][v] > 0}
                     for ax in set(freqA) | set(freqB)}
    balanced_ids = [qid for ax, a, b, qid in inst_meta
                    if a in balanced_vals[ax] and b in balanced_vals[ax]]
    print("\n  Value-balanced subset (both A-val and B-val occur in BOTH roles):")
    print(f"    {len(balanced_ids)}/{len(inst_meta)} instances "
          f"({len(balanced_ids)/max(len(inst_meta),1):.0%}) -> cleaner strict-accuracy pool")

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
        "value_balanced_subset_size": len(balanced_ids),
        "distribution": {k: dict(v) for k, v in by.items()},
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(out.with_suffix(".balanced_ids.json"), "w", encoding="utf-8") as f:
        json.dump(balanced_ids, f)
    print(f"\n  ✓ report -> {out}")
    print(f"  ✓ balanced ids -> {out.with_suffix('.balanced_ids.json')}\n")


if __name__ == "__main__":
    main()