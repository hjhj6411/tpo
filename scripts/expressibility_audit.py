#!/usr/bin/env python3
"""
expressibility_audit.py — how much of what the benchmark DECLARES it measures
is actually realized as items?

This is a MEASUREMENT tool, not a fix. It reads released artifacts only and is
fully deterministic (no RNG, no GPU, no network, no writes outside the single
new report file).

THE QUESTION
------------
Every item carries two axis roles: an ACTIVE axis (preference: A/C liked,
B/D non-preferred) and a VIOLATION axis (TPO: A/B compatible, C/D incompatible).
The remaining third axis is the BACKGROUND and is held constant.

When scenario S declares a compatible/incompatible split on axis v, the catalog
is asserting "in S, v can be wrong" — i.e. the benchmark claims it can ask that
question. This tool counts how many of those claims became items.

THREE SETS (see --help for the CLI)
-----------------------------------
  (A) DECLARED  — catalog only: (S, a, v) for every a != v and every v that S
                  constrains. Universe = |scenarios| x 6 ordered (a, v) pairs.
  (B) FEASIBLE  — (U, S, a, v) constructible for user U:
                    1) a: U has >=1 liked and >=1 disliked value, both
                       S-compatible on a
                    2) v: >=1 S-compatible and >=1 S-incompatible value, both
                       preference-neutral for U
                    3) background axis b: S-compatible n U-neutral non-empty
                       (when empty the axis stays unfixed — PROCESS.md §6 —
                       counted separately as `bg_unfixed_users`)
                    4) [separate column] some choice of the four option cells
                       is human-annotated "available" in the cell library
                  Conditions 1-3 = availability OFF; 1-4 = availability ON.
                  Both are reported.
  (C) REALIZED  — (U, S, a, v) actually present in option_plans.jsonl.

DEFECT CLASSES (the point of the tool)
--------------------------------------
  D0  not declared              — v is not a constrained axis of S. By design.
  D1  declared, 0 feasible users — catalog x profile interaction defect.
  D2  declared, >=1 feasible user, 0 realized plans — planner/emit defect.
  D3  realized > 0, but 0 plans have all four option images — realization defect.
  OK  realized > 0 and image-complete > 0.

FEASIBILITY IS IMPLEMENTED INDEPENDENTLY of construction/compatibility.py, on
purpose: an audit that reuses the pipeline's own predicate cannot detect a
defect in that predicate. The only shared inputs are the raw catalog
(configs/scenarios.py) and the attribute vocabulary (configs/config.py).

VARIANT RESOLUTION
------------------
POD_VARIANT selects the data root exactly as the rest of the pipeline does
(data_<variant>/). If that directory holds no option_plans.jsonl, the archived
release _old_data_<variant>/ is used instead and the substitution is recorded
in the report. --data-dir overrides both.

Note on the import below: configs/config.py creates the POD_VARIANT data
directories as an import side effect. This script must not create a stub data
directory for a variant it is only READING, so POD_VARIANT is neutralized for
the duration of the import (the DATA_DIR that import computes is never used —
this script resolves the data root itself).

USAGE
    POD_VARIANT=wacv_scenario_v5 python -m scripts.expressibility_audit \
        --cell-library annotation/attribute_library.json
    POD_VARIANT=wacv_scenario_v4 python -m scripts.expressibility_audit \
        --cell-library annotation/attribute_library.json
    # 2-row before/after table against an earlier report
    POD_VARIANT=wacv_scenario_v5 python -m scripts.expressibility_audit \
        --compare-report data_wacv_scenario_v4/options/expressibility_report.json
"""

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_saved_variant = os.environ.get("POD_VARIANT", "")
os.environ["POD_VARIANT"] = ""          # see module docstring
try:
    from configs.config import FASHION_ATTRIBUTE_AXES
    from configs.scenarios import CANONICAL_SCENARIOS, scenario_track
finally:
    if _saved_variant:
        os.environ["POD_VARIANT"] = _saved_variant
    else:
        os.environ.pop("POD_VARIANT", None)

AXES = ("color", "pattern", "garment_category")
CLASSES = ("OK", "D0", "D1", "D2", "D3")


# ── local helpers (deliberately not shared with construction/) ─────────────

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def resolve_data_dir(variant, override):
    """POD_VARIANT convention + archived-release fallback. Never creates."""
    if override:
        return Path(override).resolve(), None
    primary = ROOT / (f"data_{variant}" if variant else "data")
    if (primary / "options" / "option_plans.jsonl").is_file():
        return primary, None
    if variant:
        archived = ROOT / f"_old_data_{variant}"
        if (archived / "options" / "option_plans.jsonl").is_file():
            return archived, (
                f"{primary.name}/ has no option_plans.jsonl; "
                f"read the archived release {archived.name}/ instead")
    sys.exit(f"[error] no option_plans.jsonl under {primary} "
             f"(POD_VARIANT={variant!r}); pass --data-dir")


def axis_sets(scenario, axis):
    """(compatible, incompatible, is_constrained) for one axis of one scenario.

    An axis the scenario does not constrain carries no TPO meaning, so every
    vocabulary value is situation-appropriate.
    """
    constraint = scenario.get(axis)
    if constraint is not None and constraint.get("incompatible"):
        return (set(constraint.get("compatible") or ()),
                set(constraint["incompatible"]), True)
    return set(FASHION_ATTRIBUTE_AXES[axis]), set(), False


def constrained_axes(scenario):
    return [ax for ax in AXES if axis_sets(scenario, ax)[2]]


def profile_prefs(profile, axis):
    ax = (profile.get("structured_attributes") or {}).get(axis) or {}
    return set(ax.get("likes") or ()), set(ax.get("dislikes") or ())


def third_axis(a, v):
    (rest,) = [ax for ax in AXES if ax not in (a, v)]
    return rest


def cell_key(color, garment, pattern):
    return f"{color}|{garment}|{pattern}"


def images_possible(available, active, violation, background,
                    a_liked, a_disliked, v_comp, v_incomp, bg_pool):
    """Does SOME legal choice put all four options on available cells?

    The four options are the cross product {liked, disliked} x {compatible,
    incompatible} at one fixed background value, so the search is: fix the
    background, fix an (A-value, B-value) pair, then ask whether the set of
    violation values available for BOTH of them meets the compatible side and
    the incompatible side. Returns a witness dict or None.
    """
    def ok(av, vv, bv):
        attrs = {active: av, violation: vv, background: bv}
        return cell_key(attrs["color"], attrs["garment_category"],
                        attrs["pattern"]) in available

    for bv in sorted(bg_pool):
        for al in sorted(a_liked):
            for ad in sorted(a_disliked):
                comp_hit = next((vv for vv in sorted(v_comp)
                                 if ok(al, vv, bv) and ok(ad, vv, bv)), None)
                if comp_hit is None:
                    continue
                inc_hit = next((vv for vv in sorted(v_incomp)
                                if ok(al, vv, bv) and ok(ad, vv, bv)), None)
                if inc_hit is not None:
                    return {"background": bv, "active_liked": al,
                            "active_disliked": ad,
                            "violation_compatible": comp_hit,
                            "violation_incompatible": inc_hit}
    return None


def evaluate_cell(profiles, scenario, active, violation, available):
    """Per-user feasibility of one (S, a, v) cell. Returns a summary dict."""
    background = third_axis(active, violation)
    comp_a = axis_sets(scenario, active)[0]
    comp_v, incomp_v, _ = axis_sets(scenario, violation)
    comp_b = axis_sets(scenario, background)[0]

    out = {
        "feasible_users": [], "feasible_users_with_images": [],
        "bg_unfixed_users": [], "blocked_active_preference": [],
        "blocked_violation_neutral": [], "blocked_images": [],
    }
    for profile in profiles:
        uid = profile["user_id"]
        likes_a, dis_a = profile_prefs(profile, active)
        cond1_liked = sorted(likes_a & comp_a)
        cond1_disliked = sorted(dis_a & comp_a)
        if not (cond1_liked and cond1_disliked):
            out["blocked_active_preference"].append(uid)
            continue

        likes_v, dis_v = profile_prefs(profile, violation)
        v_comp_neutral = sorted(comp_v - likes_v - dis_v)
        v_incomp_neutral = sorted(incomp_v - likes_v - dis_v)
        if not (v_comp_neutral and v_incomp_neutral):
            out["blocked_violation_neutral"].append(uid)
            continue

        likes_b, dis_b = profile_prefs(profile, background)
        bg_pool = sorted(comp_b - likes_b - dis_b)
        if not bg_pool:
            # PROCESS.md §6: the axis stays unfixed rather than being filled
            out["bg_unfixed_users"].append(uid)
            continue

        out["feasible_users"].append(uid)
        witness = images_possible(available, active, violation, background,
                                  cond1_liked, cond1_disliked,
                                  v_comp_neutral, v_incomp_neutral, bg_pool)
        if witness is None:
            out["blocked_images"].append(uid)
        else:
            out["feasible_users_with_images"].append(uid)
    return out


def plan_is_image_complete(plan, available):
    for opt in plan["options"].values():
        attrs = opt.get("attributes") or {}
        color = attrs.get("color")
        garment = attrs.get("garment_category")
        pattern = attrs.get("pattern")
        if not (color and garment and pattern):
            return False
        if cell_key(color, garment, pattern) not in available:
            return False
    return True


def classify(declared, feasible_users, realized_plans, image_complete):
    if not declared:
        return "D0"
    if feasible_users == 0:
        return "D1"
    if realized_plans == 0:
        return "D2"
    if image_complete == 0:
        return "D3"
    return "OK"


# ── report assembly ────────────────────────────────────────────────────────

def build_report(args):
    variant = os.environ.get("POD_VARIANT", "").strip()
    data_dir, fallback_note = resolve_data_dir(variant, args.data_dir)

    paths = {
        "profiles": data_dir / "profiles" / "profiles.jsonl",
        "queries": data_dir / "queries" / "queries.jsonl",
        "option_plans": data_dir / "options" / "option_plans.jsonl",
        "cell_library": Path(args.cell_library).resolve(),
    }
    for name, path in paths.items():
        if not path.is_file():
            sys.exit(f"[error] missing input {name}: {path}")

    profiles = sorted(load_jsonl(paths["profiles"]), key=lambda p: p["user_id"])
    plans = load_jsonl(paths["option_plans"])
    library = json.loads(paths["cell_library"].read_text(encoding="utf-8"))
    available = {k for k, v in (library.get("cells") or {}).items()
                 if v.get("status") == "available"}

    scenarios = list(CANONICAL_SCENARIOS)
    if args.limit:
        scenarios = scenarios[:args.limit]
    scenario_ids = {s["scenario_id"] for s in scenarios}

    # (C) realized, straight off the released plans
    realized = defaultdict(list)
    for plan in plans:
        key = (plan["scenario_id"], plan["active_axis"], plan["violation_axis"])
        realized[key].append(plan)

    cells = []
    for scenario in scenarios:
        sid = scenario["scenario_id"]
        declared_v = set(constrained_axes(scenario))
        track = scenario_track(scenario)
        for violation in AXES:
            for active in AXES:
                if active == violation:
                    continue
                declared = violation in declared_v
                if declared:
                    ev = evaluate_cell(profiles, scenario, active, violation,
                                       available)
                else:
                    ev = {k: [] for k in (
                        "feasible_users", "feasible_users_with_images",
                        "bg_unfixed_users", "blocked_active_preference",
                        "blocked_violation_neutral", "blocked_images")}
                got = realized[(sid, active, violation)]
                complete = [p for p in got if plan_is_image_complete(p, available)]
                n_feas = len(ev["feasible_users"])
                n_feas_img = len(ev["feasible_users_with_images"])
                cells.append({
                    "scenario_id": sid,
                    "archetype": scenario["archetype"],
                    "track": track,
                    "active_axis": active,
                    "violation_axis": violation,
                    "background_axis": third_axis(active, violation),
                    "declared": declared,
                    "feasible_users": n_feas,
                    "feasible_users_availability_on": n_feas_img,
                    "bg_unfixed_users": len(ev["bg_unfixed_users"]),
                    "blocked_active_preference": len(ev["blocked_active_preference"]),
                    "blocked_violation_neutral": len(ev["blocked_violation_neutral"]),
                    "blocked_images": len(ev["blocked_images"]),
                    "realized_plans": len(got),
                    "realized_users": len({p["user_id"] for p in got}),
                    "image_complete_plans": len(complete),
                    "class": classify(declared, n_feas, len(got), len(complete)),
                    "class_availability_on": classify(
                        declared, n_feas_img, len(got), len(complete)),
                })

    def summarize(class_field):
        counts = {c: 0 for c in CLASSES}
        for cell in cells:
            counts[cell[class_field]] += 1
        declared_cells = [c for c in cells if c["declared"]]
        return {
            "universe_cells": len(cells),
            "declared_cells": len(declared_cells),
            "declared_with_realized": sum(1 for c in declared_cells
                                          if c["realized_plans"] > 0),
            "declared_with_image_complete": sum(
                1 for c in declared_cells if c["image_complete_plans"] > 0),
            "D0": counts["D0"], "D1": counts["D1"], "D2": counts["D2"],
            "D3": counts["D3"], "OK": counts["OK"],
        }

    def rollup(keyfn):
        agg = defaultdict(lambda: {"declared": 0, "feasible_user_cells": 0,
                                   "realized_cells": 0, "realized_plans": 0,
                                   "image_complete_plans": 0,
                                   "D1": 0, "D2": 0, "D3": 0})
        for cell in cells:
            row = agg[keyfn(cell)]
            row["declared"] += int(cell["declared"])
            row["feasible_user_cells"] += cell["feasible_users"]
            row["realized_cells"] += int(cell["realized_plans"] > 0)
            row["realized_plans"] += cell["realized_plans"]
            row["image_complete_plans"] += cell["image_complete_plans"]
            if cell["class"] in ("D1", "D2", "D3"):
                row[cell["class"]] += 1
        return {k: agg[k] for k in sorted(agg)}

    off_scenarios = [k for k in sorted(realized) if k[0] not in scenario_ids]

    return {
        "tool": "scripts/expressibility_audit.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "variant": variant or "(unset -> data/)",
        "data_dir": str(data_dir),
        "data_dir_fallback": fallback_note,
        "partial": bool(args.limit),
        "limit_scenarios": args.limit or None,
        "feasibility_implementation":
            "independent — construction/compatibility.py is NOT imported; "
            "only configs/scenarios.py and configs/config.py are read",
        "inputs": {name: {"path": str(path), "sha256": sha256_of(path)}
                   for name, path in sorted(paths.items())},
        "catalog": {
            "source": "configs/scenarios.py",
            "sha256": sha256_of(ROOT / "configs" / "scenarios.py"),
            "scenarios_total": len(CANONICAL_SCENARIOS),
            "scenarios_audited": len(scenarios),
        },
        "cell_library": {
            "available_cells": len(available),
            "total_cells": len(library.get("cells") or {}),
        },
        "plans": {
            "total": len(plans),
            "in_audited_scenarios": sum(len(v) for k, v in realized.items()
                                        if k[0] in scenario_ids),
            "plans_outside_audited_scenarios": sum(
                len(realized[k]) for k in off_scenarios),
        },
        "summary": {
            "availability_off": summarize("class"),
            "availability_on": summarize("class_availability_on"),
        },
        "cells": cells,
        "rollups": {
            "by_scenario": rollup(lambda c: c["scenario_id"]),
            "by_archetype": rollup(lambda c: c["archetype"]),
            "by_track_active_violation": rollup(
                lambda c: f"{c['track']}|{c['active_axis']}|{c['violation_axis']}"),
        },
    }


# ── stdout tables ──────────────────────────────────────────────────────────

def short(axis):
    return {"color": "color", "pattern": "pattern",
            "garment_category": "garment"}[axis]


def print_report(report):
    print("=" * 96)
    print(f"EXPRESSIBILITY AUDIT — {report['variant']}")
    print("=" * 96)
    print(f"  data dir       : {report['data_dir']}")
    if report["data_dir_fallback"]:
        print(f"                   ! {report['data_dir_fallback']}")
    print(f"  generated      : {report['generated_at']}")
    print(f"  feasibility    : {report['feasibility_implementation']}")
    for name, info in report["inputs"].items():
        print(f"  {name:<14} : {info['sha256'][:16]}…  {info['path']}")
    print(f"  catalog        : {report['catalog']['sha256'][:16]}…  "
          f"{report['catalog']['scenarios_audited']}/"
          f"{report['catalog']['scenarios_total']} scenarios")
    print(f"  cell library   : {report['cell_library']['available_cells']} available "
          f"/ {report['cell_library']['total_cells']} cells")
    print(f"  plans          : {report['plans']['total']} total, "
          f"{report['plans']['in_audited_scenarios']} in audited scenarios")
    if report["partial"]:
        print(f"  ** PARTIAL RUN (--limit {report['limit_scenarios']}) **")

    for label, field, summary_key in (
            ("availability OFF (conditions 1-3)", "class", "availability_off"),
            ("availability ON  (conditions 1-4)", "class_availability_on",
             "availability_on")):
        s = report["summary"][summary_key]
        print(f"\n  {label}:  declared {s['declared_cells']}/{s['universe_cells']}"
              f"   realized>0 {s['declared_with_realized']}"
              f"   OK {s['OK']}   D1 {s['D1']}   D2 {s['D2']}   D3 {s['D3']}")

    # ── 12-cell matrix ──
    print("\n" + "-" * 96)
    print("(track x active x violation) — declared cells / realized cells "
          "[realized plans]")
    print("-" * 96)
    rows = report["rollups"]["by_track_active_violation"]
    header = f"{'track':<11}{'active':<10}" + "".join(
        f"{'v=' + short(v):>22}" for v in AXES)
    print(header)
    for track in ("physical", "dress_code"):
        for active in AXES:
            line = f"{track:<11}{short(active):<10}"
            for violation in AXES:
                if active == violation:
                    line += f"{'—':>22}"
                    continue
                row = rows.get(f"{track}|{active}|{violation}")
                if row is None:
                    line += f"{'0 / 0 [0]':>22}"
                else:
                    line += (f"{row['declared']} / {row['realized_cells']} "
                             f"[{row['realized_plans']}]").rjust(22)
            print(line)

    # ── defect lists ──
    for klass, title in (
            ("D1", "D1 — declared but 0 feasible users "
                   "(catalog x profile interaction)"),
            ("D2", "D2 — declared, feasible, but 0 plans emitted "
                   "(planner/emit)"),
            ("D3", "D3 — plans exist but none has four images (realization)")):
        listed = [c for c in report["cells"] if c["class"] == klass]
        print("\n" + "-" * 96)
        print(f"{title}: {len(listed)} cells")
        print("-" * 96)
        if not listed:
            print("  (none)")
            continue
        for c in sorted(listed, key=lambda c: (c["track"], c["scenario_id"],
                                               c["active_axis"],
                                               c["violation_axis"])):
            detail = ""
            if klass == "D1":
                detail = (f"blocked: active-pref {c['blocked_active_preference']}"
                          f", violation-neutral {c['blocked_violation_neutral']}"
                          f", background {c['bg_unfixed_users']}")
            elif klass == "D2":
                detail = (f"feasible {c['feasible_users']} users "
                          f"({c['feasible_users_availability_on']} with images)")
            else:
                detail = (f"{c['realized_plans']} plans, "
                          f"{c['feasible_users_availability_on']} users have images")
            print(f"  {c['track']:<10} {c['scenario_id']:<28} "
                  f"a={short(c['active_axis']):<8} v={short(c['violation_axis']):<8} "
                  f"{detail}")

    # ── archetype rollup ──
    print("\n" + "-" * 96)
    print("by archetype")
    print("-" * 96)
    print(f"{'archetype':<26}{'decl':>6}{'real':>6}{'plans':>8}{'imgOK':>8}"
          f"{'D1':>5}{'D2':>5}{'D3':>5}")
    for arch, row in report["rollups"]["by_archetype"].items():
        print(f"{arch:<26}{row['declared']:>6}{row['realized_cells']:>6}"
              f"{row['realized_plans']:>8}{row['image_complete_plans']:>8}"
              f"{row['D1']:>5}{row['D2']:>5}{row['D3']:>5}")


def print_comparison(other, this):
    print("\n" + "=" * 96)
    print("BEFORE / AFTER")
    print("=" * 96)
    print(f"{'variant':<24}{'declared':>10}{'realized>0':>12}{'D1':>6}"
          f"{'D2':>6}{'D3':>6}{'plans':>9}{'imgOK':>9}")
    for rep in (other, this):
        s = rep["summary"]["availability_off"]
        plans = sum(c["realized_plans"] for c in rep["cells"])
        img = sum(c["image_complete_plans"] for c in rep["cells"])
        print(f"{rep['variant']:<24}{s['declared_cells']:>10}"
              f"{s['declared_with_realized']:>12}{s['D1']:>6}{s['D2']:>6}"
              f"{s['D3']:>6}{plans:>9}{img:>9}")


def main():
    ap = argparse.ArgumentParser(
        description="Count how much of the benchmark's declared contrast space "
                    "is realized as items (read-only, deterministic).")
    ap.add_argument("--cell-library", default=str(
        ROOT / "annotation" / "attribute_library.json"),
        help="human-annotated color|garment|pattern availability library")
    ap.add_argument("--data-dir", default=None,
                    help="override the POD_VARIANT-resolved data root")
    ap.add_argument("--out", default=None,
                    help="report path (default: <data>/options/"
                         "expressibility_report.json)")
    ap.add_argument("--limit", type=int, default=0,
                    help="audit only the first N scenarios (smoke run; writes "
                         "expressibility_report.partial.json)")
    ap.add_argument("--compare-report", default=None,
                    help="an earlier expressibility_report.json to print a "
                         "2-row before/after table against")
    ap.add_argument("--no-write", action="store_true",
                    help="print only, do not write the report file")
    args = ap.parse_args()

    report = build_report(args)
    print_report(report)

    if args.compare_report:
        other = json.loads(Path(args.compare_report).read_text(encoding="utf-8"))
        print_comparison(other, report)

    if not args.no_write:
        name = ("expressibility_report.partial.json" if args.limit
                else "expressibility_report.json")
        out = Path(args.out) if args.out else \
            Path(report["data_dir"]) / "options" / name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n",
                       encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
