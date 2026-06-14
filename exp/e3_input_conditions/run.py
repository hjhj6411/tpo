"""
e3_input_conditions/run.py — input-format analysis (list B, all three items).

Objective : compare none / scenario_only / profile_only / both ("all") under a
            SHARED display order per item (paired across conditions; only the
            context blocks differ).
Sanity hypothesis (B-3): under profile_only the model has zero TPO signal, so
            counterbalanced construction predicts
              pref accuracy  -> high (~ text-profile ceiling)
              TPO accuracy   -> ~0.50 (coin flip between TPO-yes/no pairs)
            The analysis block prints this verdict with CIs; deviation from
            0.5 means the option pairs leak TPO information -> construction bug.
Input     : --items; --conditions subset; --profile-mode {text,image,both}
            (text vs image profile IS the headline axis — run both).
Output    : <results>/e3_input_conditions/<ts>_<tag>/{results.jsonl,
            analysis.json, sample_prompt_<cond>.json}
Expected  : per-condition strict/tpo/pref + CI, McNemar(both vs each) on
            strict, profile_only sanity verdict.
Run       :
  python -m e3_input_conditions.run --items data/bench/items.jsonl \
      --model qwen3-vl-30b --profile-mode text --tag qwen_text
  python -m e3_input_conditions.run --model qwen3-vl-30b \
      --profile-mode image --tag qwen_image
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
from common.io_utils import make_run_dir, save_manifest, write_json, write_jsonl
from common.permute import base_perm, item_rng
from common.prompts import build_mcq_messages, serialize_for_dump

EXP = "e3_input_conditions"
LETTERS = ("A", "B", "C", "D")
ALL_CONDITIONS = ("both", "scenario_only", "profile_only", "none")


def build_tasks(items, cfg):
    tasks = []
    for it in items:
        order = base_perm(item_rng(cfg["seed"], it["query_id"]))
        for cond in cfg["conditions"]:           # SAME order across conditions
            msgs = build_mcq_messages(it, order, condition=cond,
                                      profile_mode=cfg["profile_mode"],
                                      layout=cfg["layout"], letters=LETTERS)
            tasks.append({"query_id": it["query_id"],
                          "active_axis": it.get("active_axis"),
                          "condition": cond, "order": order,
                          "letters": LETTERS, "messages": msgs})
    return tasks


def analyze(records):
    by_cond = defaultdict(list)
    for r in records:
        by_cond[r["condition"]].append(r)
    out = {"by_condition": {c: metrics.acc_summary(rs)
                            for c, rs in sorted(by_cond.items())}}
    if "both" in by_cond:
        out["mcnemar_vs_both"] = {
            c: metrics.paired_mcnemar(by_cond["both"], rs, "strict")
            for c, rs in sorted(by_cond.items()) if c != "both"}
    if "profile_only" in by_cond:
        s = out["by_condition"]["profile_only"]
        tpo_in_band = abs(s["tpo"]["acc"] - 0.5) <= 0.05 or \
            (s["tpo"]["ci95"][0] <= 0.5 <= s["tpo"]["ci95"][1])
        out["profile_only_sanity"] = {
            "pref_acc": s["pref"], "tpo_acc": s["tpo"],
            "tpo_at_chance": bool(tpo_in_band),
            "note": "tpo_at_chance False => option pairs leak TPO info; "
                    "inspect construction before trusting decomposed metrics"}
    return out


def run(cfg, items, client):
    records = execute_tasks(client, build_tasks(items, cfg))
    metrics.attach_scores(records)
    return records, analyze(records)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--conditions", nargs="+", default=list(ALL_CONDITIONS),
                    choices=list(ALL_CONDITIONS))
    ap.add_argument("--profile-mode", dest="profile_mode",
                    choices=["text", "image", "both"], default=None)
    cfg = load_cfg(ap.parse_args())
    items = load_items(cfg["items"], cfg["limit"])
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
    save_manifest(run_dir, cfg, {"n_items": len(items)})
    records, analysis = run(cfg, items, client)
    write_jsonl(run_dir / "results.jsonl", records)
    write_json(run_dir / "analysis.json", analysis)
    for cond in cfg["conditions"]:
        t = [t for t in build_tasks(items[:1], cfg) if t["condition"] == cond]
        write_json(run_dir / f"sample_prompt_{cond}.json",
                   serialize_for_dump(t[0]["messages"]))

    print(f"\n[{EXP}] {cfg['model']} profile_mode={cfg['profile_mode']}")
    for cond, s in analysis["by_condition"].items():
        print(f"  [{cond:>13}] strict={s['strict']['acc']:.3f} "
              f"tpo={s['tpo']['acc']:.3f} pref={s['pref']['acc']:.3f} "
              f"(n={s['n']}, parse_fail={s['parse_fail_rate']})")
    if "profile_only_sanity" in analysis:
        sv = analysis["profile_only_sanity"]
        print(f"  sanity(profile_only): tpo_at_chance={sv['tpo_at_chance']} "
              f"tpo={sv['tpo_acc']['acc']} pref={sv['pref_acc']['acc']}")
    print(f"  -> {run_dir}")


if __name__ == "__main__":
    main()
