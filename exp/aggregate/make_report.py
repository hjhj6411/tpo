"""
aggregate/make_report.py — cross-experiment result aggregation.

Objective : walk the results root, collect every run's manifest + analysis,
            and emit one markdown report + one flat csv for the paper.
Input     : --results-root (default data/exp_results) containing
            <exp_name>/<run>/{manifest.json, analysis.json}.
Output    : <results-root>/report.md and <results-root>/combined.csv
            (one row per run: exp, run, model, tag, key headline numbers).
Run       :
  python -m aggregate.make_report --results-root data/exp_results
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _g(d, *path):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def headline(exp, an):
    """Per-experiment headline numbers for the flat csv."""
    if exp == "e1_position_bias":
        return {"strict": _g(an, "overall", "strict", "acc"),
                "spread": _g(an, "acc_by_gold_position", "spread"),
                "luck_gap": _g(an, "robust", "luck_gap"),
                "pos_chi2_p": _g(an, "position_choice", "p")}
    if exp == "e2_bc_tradeoff":
        return {"p_tpo_both": _g(an, "by_condition", "both", "overall",
                                 "p_tpo"),
                "p_tpo_no_profile": _g(an, "by_condition", "scenario_only",
                                       "overall", "p_tpo")}
    if exp == "e3_input_conditions":
        return {"strict_both": _g(an, "by_condition", "both", "strict", "acc"),
                "pref_profile_only": _g(an, "by_condition", "profile_only",
                                        "pref", "acc"),
                "tpo_profile_only": _g(an, "by_condition", "profile_only",
                                       "tpo", "acc"),
                "tpo_at_chance": _g(an, "profile_only_sanity",
                                    "tpo_at_chance")}
    if exp == "e5_input_order":
        return {"strict_text_first": _g(an, "by_layout", "text_first",
                                        "strict", "acc"),
                "strict_img_first": _g(an, "by_layout", "img_first",
                                       "strict", "acc"),
                "mcnemar_p": _g(an, "mcnemar_strict", "p")}
    if exp == "e6_dialogue_preference":
        return {"strict_p0": _g(an, "by_variant", "dialogue_p0", "strict",
                                "acc"),
                "strict_p1": _g(an, "by_variant", "dialogue_p1", "strict",
                                "acc")}
    return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", default="data/exp_results")
    a = ap.parse_args()
    root = Path(a.results_root)
    rows = []
    for an_path in sorted(root.glob("*/*/analysis.json")):
        run_dir = an_path.parent
        exp = run_dir.parent.name
        an = json.loads(an_path.read_text())
        man = {}
        mp = run_dir / "manifest.json"
        if mp.exists():
            man = json.loads(mp.read_text())
        rows.append({"exp": exp, "run": run_dir.name,
                     "model": _g(man, "config", "model"),
                     "tag": _g(man, "config", "tag"),
                     **headline(exp, an)})
    if not rows:
        print(f"no analysis.json under {root}")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(root / "combined.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(root / "report.md", "w") as f:
        f.write("# POD-Bench experiment report\n\n")
        cur = None
        for r in rows:
            if r["exp"] != cur:
                cur = r["exp"]
                f.write(f"\n## {cur}\n\n")
            nums = {k: v for k, v in r.items()
                    if k not in ("exp", "run", "model", "tag") and v is not None}
            f.write(f"- `{r['run']}` model={r['model']} tag={r['tag']}: "
                    f"{json.dumps(nums)}\n")
    print(f"runs={len(rows)} -> {root/'report.md'}, {root/'combined.csv'}")


if __name__ == "__main__":
    main()
