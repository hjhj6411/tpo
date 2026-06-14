"""
e5_input_order/run.py — image->query vs query->image ordering (list D-1).

Objective : swap ONLY the macro block order (context text vs option images);
            display order, label-image binding, condition all held fixed.
            Paired over identical items -> exact McNemar on strict.
TEMPLATE PITFALL (read before trusting results): some VLM chat templates
            re-anchor images regardless of message order. This runner dumps
            sample_prompt_text_first.json / sample_prompt_img_first.json —
            VERIFY the serialized prompts actually differ for your model
            (vLLM: check the rendered prompt in server logs). Models whose
            template cancels the swap must be excluded or footnoted.
Input     : --items.
Output    : <results>/e5_input_order/<ts>_<tag>/{results.jsonl, analysis.json,
            sample_prompt_<layout>.json}
Expected  : per-layout strict/tpo/pref + McNemar; small effect expected ->
            robustness appendix material.
Run       :
  python -m e5_input_order.run --items data/bench/items.jsonl \
      --model qwen3-vl-30b --tag qwen
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

EXP = "e5_input_order"
LETTERS = ("A", "B", "C", "D")
LAYOUTS = ("text_first", "img_first")


def build_tasks(items, cfg):
    tasks = []
    for it in items:
        order = base_perm(item_rng(cfg["seed"], it["query_id"]))
        for layout in LAYOUTS:                  # SAME order across layouts
            msgs = build_mcq_messages(it, order, condition=cfg["condition"],
                                      profile_mode=cfg["profile_mode"],
                                      layout=layout, letters=LETTERS)
            tasks.append({"query_id": it["query_id"],
                          "active_axis": it.get("active_axis"),
                          "layout": layout, "order": order,
                          "letters": LETTERS, "messages": msgs})
    return tasks


def analyze(records):
    by = defaultdict(list)
    for r in records:
        by[r["layout"]].append(r)
    out = {"by_layout": {l: metrics.acc_summary(rs)
                         for l, rs in sorted(by.items())}}
    if all(l in by for l in LAYOUTS):
        out["mcnemar_strict"] = metrics.paired_mcnemar(
            by["text_first"], by["img_first"], "strict")
    return out


def run(cfg, items, client):
    records = execute_tasks(client, build_tasks(items, cfg))
    metrics.attach_scores(records)
    return records, analyze(records)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
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
    for layout in LAYOUTS:
        t = [t for t in build_tasks(items[:1], cfg) if t["layout"] == layout]
        write_json(run_dir / f"sample_prompt_{layout}.json",
                   serialize_for_dump(t[0]["messages"]))

    print(f"\n[{EXP}] {cfg['model']}")
    for l, s in analysis["by_layout"].items():
        print(f"  [{l:>10}] strict={s['strict']['acc']:.3f} "
              f"tpo={s['tpo']['acc']:.3f} pref={s['pref']['acc']:.3f}")
    if "mcnemar_strict" in analysis:
        print(f"  mcnemar(strict): {analysis['mcnemar_strict']}")
    print("  !! verify sample_prompt_*.json actually differ for this model's "
          "chat template before citing this result")
    print(f"  -> {run_dir}")


if __name__ == "__main__":
    main()
