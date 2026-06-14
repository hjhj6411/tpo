"""
e1_position_bias/run.py — answer-position bias (choice-list items 1, 2, 4).

Objective : decouple option content from display position and measure
  (a) which display position models prefer under shuffling      [item 1]
  (b) accuracy when the gold sits at A / B / C / D               [item 2]
  (c) whether per-model position bias exists, with significance  [item 4]
  ONE cyclic latin-square protocol yields all three analyses; --mode
  fixed_gold reproduces item 2 literally (only the gold moves).
Input     : --items (schema: common/data.py), model registry entry.
Output    : <results>/e1_position_bias/<ts>_<tag>/
              results.jsonl   one record per (item x run)
              analysis.json   position matrix + chi2, acc-by-gold-pos +
                              Cochran's Q + spread, robust vs mean accuracy
              sample_prompt.json (serialized first prompt, template check)
Expected  : unbiased model -> chi2 p > .05, spread ~ 0, luck_gap ~ 0.
            position-biased model -> concentrated choice column; acc@gold-pos
            spikes where its preferred slot is; luck_gap > 0.
Run       :
  python -m e1_position_bias.run --items data/bench/items.jsonl \
      --model qwen3-vl-30b --mode cyclic --n-runs 4 --tag qwen_cyc
  python -m e1_position_bias.run --mode fixed_gold --model gemma3-27b
  # closed models: subsample for cost
  python -m e1_position_bias.run --model gpt --limit 60 --mode cyclic
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import metrics
from common.clients import ChatClient, FakeClient, execute_tasks
from common.config import add_common_args, load_cfg, resolve_model
from common.data import load_items
from common.io_utils import make_run_dir, save_manifest, write_json, write_jsonl
from common.permute import base_perm, cyclic_orders, fixed_gold_order, item_rng
from common.prompts import build_mcq_messages, serialize_for_dump

EXP = "e1_position_bias"
LETTERS = ("A", "B", "C", "D")


def build_tasks(items, cfg):
    tasks = []
    for it in items:
        rng = item_rng(cfg["seed"], it["query_id"])
        base = base_perm(rng)
        if cfg["mode"] == "cyclic":
            orders = cyclic_orders(base, cfg["n_runs"])
        else:                                   # fixed_gold: gold idx 0 -> pos
            orders = [fixed_gold_order(base, 0, pos) for pos in range(4)]
        for run, order in enumerate(orders):
            msgs = build_mcq_messages(
                it, order, condition=cfg["condition"],
                profile_mode=cfg["profile_mode"], layout=cfg["layout"],
                letters=LETTERS)
            tasks.append({"query_id": it["query_id"],
                          "active_axis": it.get("active_axis"),
                          "mode": cfg["mode"], "run": run, "order": order,
                          "letters": LETTERS, "messages": msgs})
    return tasks


def analyze(records):
    parsed = [r for r in records if r["parsed"]]
    return {
        "overall": metrics.acc_summary(records),
        "position_choice": metrics.position_matrix(parsed, LETTERS),
        "wrong_only_position": metrics.position_matrix(
            [r for r in parsed if not r["strict"]], LETTERS),
        "acc_by_gold_position": metrics.acc_by_gold_position(records, LETTERS),
        "robust": metrics.robust_accuracy(records),
    }


def run(cfg, items, client):
    tasks = build_tasks(items, cfg)
    records = execute_tasks(client, tasks)
    metrics.attach_scores(records)
    return records, analyze(records)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--mode", choices=["cyclic", "fixed_gold"],
                    default="cyclic")
    ap.add_argument("--n-runs", type=int, default=4,
                    help="cyclic rotations (4 = full latin square)")
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
    write_json(run_dir / "sample_prompt.json",
               serialize_for_dump(build_tasks(items[:1], cfg)[0]["messages"]))

    a = analysis
    print(f"\n[{EXP}] {cfg['model']} mode={cfg['mode']} n={a['overall']['n']}")
    print(f"  strict={a['overall']['strict']['acc']:.3f} "
          f"tpo={a['overall']['tpo']['acc']:.3f} "
          f"pref={a['overall']['pref']['acc']:.3f} "
          f"parse_fail={a['overall']['parse_fail_rate']:.3f}")
    pc = a["position_choice"]
    print(f"  choice by position {pc['letters']}: {pc['counts']} "
          f"(chi2={pc['chi2']}, p={pc['p']})")
    g = a["acc_by_gold_position"]
    print(f"  acc by gold pos: "
          f"{ {k: v['acc'] for k, v in g['by_gold_pos'].items()} } "
          f"spread={g['spread']} cochranQ={g['cochran_q']}")
    print(f"  mean={a['robust']['mean_acc']} robust={a['robust']['robust_acc']}"
          f" luck_gap={a['robust']['luck_gap']}")
    print(f"  -> {run_dir}")


if __name__ == "__main__":
    main()
