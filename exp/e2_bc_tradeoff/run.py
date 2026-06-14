"""
e2_bc_tradeoff/run.py — forced trade-off between TPO and preference (item 3).

Objective : remove gold (A) and escape (D); present canonical B (tpo_only) vs
            C (pref_only). With no correct answer, P(choose B) IS the model's
            TPO-vs-preference priority. Side of B is seeded 50:50 so e1-style
            position bias cannot masquerade as a priority.
Sanity    : --conditions both scenario_only -> under scenario_only (profile
            absent) P(B) should approach 1.0; if not, the option pair leaks.
Cross-tab : --main-results <e1 results.jsonl> joins strict-WRONG items from the
            4-choice run with this 2-choice preference: a consistent direction
            across protocols = intrinsic priority, not protocol artifact.
Input     : --items; optional --main-results.
Output    : <results>/e2_bc_tradeoff/<ts>_<tag>/{results.jsonl, analysis.json,
            sample_prompt.json}
Expected  : per-model P(B) + Wilson CI, split by active_axis (does the
            trade-off flip when preference is color vs pattern?) and by
            condition; cross-tab agreement rate if --main-results given.
Run       :
  python -m e2_bc_tradeoff.run --items data/bench/items.jsonl \
      --model qwen3-vl-30b --conditions both scenario_only --tag qwen
  python -m e2_bc_tradeoff.run --model gpt --limit 60 \
      --main-results data/exp_results/e1_position_bias/<run>/results.jsonl
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import metrics
from common.clients import ChatClient, FakeClient, execute_tasks
from common.config import add_common_args, load_cfg, resolve_model
from common.data import load_items
from common.io_utils import (make_run_dir, read_jsonl, save_manifest,
                             write_json, write_jsonl)
from common.permute import item_rng
from common.prompts import build_bc_messages, serialize_for_dump

EXP = "e2_bc_tradeoff"
LETTERS = ("A", "B")


def build_tasks(items, cfg):
    tasks = []
    for cond in cfg["conditions"]:
        for it in items:
            rng = item_rng(cfg["seed"], f"{cond}:{it['query_id']}")
            b_first = rng.random() < 0.5
            msgs, pair = build_bc_messages(
                it, b_first, condition=cond,
                profile_mode=cfg["profile_mode"], layout=cfg["layout"])
            tasks.append({"query_id": it["query_id"],
                          "active_axis": it.get("active_axis"),
                          "condition": cond, "b_first": b_first, "pair": pair,
                          "letters": LETTERS, "messages": msgs})
    return tasks


def attach(records):
    for r in records:
        if r["choice_display"] is None:
            r["choice_canon"] = None
            r["chose_tpo"] = None
        else:
            pos = list(r["letters"]).index(r["choice_display"])
            r["choice_canon"] = r["pair"][pos]
            r["chose_tpo"] = (r["choice_canon"] == "B")
    return records


def _pb(records):
    rs = [r for r in records if r["chose_tpo"] is not None]
    k = sum(int(r["chose_tpo"]) for r in rs)
    p, lo, hi = metrics.wilson(k, len(rs))
    return {"p_tpo": p, "ci95": [lo, hi], "n": len(rs)}


def analyze(records, main_results=None):
    out = {"side_balance": {
        "b_first_rate": round(sum(int(r["b_first"]) for r in records)
                              / max(len(records), 1), 3)}}
    by_cond = defaultdict(list)
    for r in records:
        by_cond[r["condition"]].append(r)
    out["by_condition"] = {}
    for cond, rs in sorted(by_cond.items()):
        block = {"overall": _pb(rs), "by_axis": {}}
        by_ax = defaultdict(list)
        for r in rs:
            by_ax[r.get("active_axis")].append(r)
        for ax, axr in sorted(by_ax.items()):
            block["by_axis"][str(ax)] = _pb(axr)
        out["by_condition"][cond] = block
    out["parse_fail_rate"] = round(
        sum(1 for r in records if r["chose_tpo"] is None)
        / max(len(records), 1), 4)

    if main_results:
        main = {r["query_id"]: r for r in main_results
                if r.get("parsed") and not r.get("strict")}
        bc = {r["query_id"]: r for r in records
              if r["condition"] == "both" and r["chose_tpo"] is not None}
        joint = sorted(set(main) & set(bc))
        agree = sum(1 for q in joint
                    if (main[q].get("choice_canon") == "B") == bc[q]["chose_tpo"])
        out["cross_tab_vs_main_wrong"] = {
            "n_joint_wrong": len(joint),
            "direction_agreement": round(agree / max(len(joint), 1), 4),
            "note": "4-choice wrong answers' B/C direction vs 2-choice "
                    "preference; high agreement = protocol-robust priority"}
    return out


def run(cfg, items, client, main_results=None):
    records = attach(execute_tasks(client, build_tasks(items, cfg)))
    return records, analyze(records, main_results)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--conditions", nargs="+", default=["both"],
                    choices=["both", "scenario_only", "profile_only", "none"])
    ap.add_argument("--main-results", type=str, default=None,
                    help="e1 results.jsonl for the wrong-answer cross-tab")
    cfg = load_cfg(ap.parse_args())
    items = load_items(cfg["items"], cfg["limit"],
                       need_options=("B", "C"))
    run_dir = make_run_dir(cfg["results_root"], EXP, cfg["tag"])
    if cfg.get("dry_run"):
        client = FakeClient()
    else:
        m = resolve_model(cfg)
        client = ChatClient(m["base_url"], m["model_id"], m.get("api_key"),
                            m["guided"], cache_dir=run_dir / "cache"
                            if cfg["cache"] else None,
                            max_tokens=cfg["max_tokens"],
                            temperature=cfg["temperature"],
                            image_max_side=cfg["image_max_side"])
    main_res = read_jsonl(cfg["main_results"]) if cfg.get("main_results") else None
    save_manifest(run_dir, cfg, {"n_items": len(items)})
    records, analysis = run(cfg, items, client, main_res)
    write_jsonl(run_dir / "results.jsonl", records)
    write_json(run_dir / "analysis.json", analysis)
    write_json(run_dir / "sample_prompt.json",
               serialize_for_dump(build_tasks(items[:1], cfg)[0]["messages"]))

    print(f"\n[{EXP}] {cfg['model']}")
    for cond, blk in analysis["by_condition"].items():
        o = blk["overall"]
        print(f"  [{cond}] P(choose TPO-side)={o['p_tpo']} ci={o['ci95']} "
              f"n={o['n']}  axis={ {a: b['p_tpo'] for a, b in blk['by_axis'].items()} }")
    if "cross_tab_vs_main_wrong" in analysis:
        ct = analysis["cross_tab_vs_main_wrong"]
        print(f"  cross-tab: joint_wrong={ct['n_joint_wrong']} "
              f"agreement={ct['direction_agreement']}")
    print(f"  -> {run_dir}")


if __name__ == "__main__":
    main()
