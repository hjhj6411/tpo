"""
e6_dialogue_preference/run.py — preference understanding from multimodal
dialogue history (list D-2 + the implicit/explicit design).

Objective : preference enters ONLY through multi-turn dialogue (final question
            carries scenario + options, NO profile block). Each profile entry
            becomes one user turn (sentence + image); with prob p the sentence
            is the IMPLICIT variant (attribute word removed -> the model must
            SEE the image to recover the preference).
              explicit p=0  ~ dialogue twin of the text-attribute profile
              implicit p=1  ~ dialogue twin of the image-instance profile
Hypothesis: acc(p=0) ~ text-profile ceiling; acc(p=1) ~ image-profile floor
            (~56-59%) — if confirmed, the perception bottleneck generalizes
            from profile FORMAT to input MODALITY.
Controls  : assistant turns fixed "Got it." (no attribute leakage from the
            model's own turns); turn count = entry count for every p.
Input     : --items with profile_entries [{explicit, implicit, image}].
Output    : <results>/e6_dialogue_preference/<ts>_<tag>/{results.jsonl,
            analysis.json, sample_prompt_p<p>.json}
Expected  : per-p strict/tpo/pref + realized implicit fraction; optional text
            baseline (--include-text-baseline) for the ceiling reference.
Run       :
  python -m e6_dialogue_preference.run --items data/bench/items_dialogue.jsonl \
      --model qwen3-vl-30b --p-list 0 0.5 1 --include-text-baseline --tag pilot
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
from common.prompts import (build_dialogue_messages, build_mcq_messages,
                            serialize_for_dump)

EXP = "e6_dialogue_preference"
LETTERS = ("A", "B", "C", "D")


def build_tasks(items, cfg):
    tasks = []
    for it in items:
        rng = item_rng(cfg["seed"], it["query_id"])
        order = base_perm(rng)
        for p in cfg["p_list"]:                 # SAME order across p
            prng = item_rng(cfg["seed"], f"p{p}:{it['query_id']}")
            msgs, frac = build_dialogue_messages(
                it, p, prng, order, layout=cfg["layout"], letters=LETTERS)
            tasks.append({"query_id": it["query_id"],
                          "active_axis": it.get("active_axis"),
                          "p_implicit": p, "realized_implicit_frac": frac,
                          "n_turns": len(it["profile_entries"]),
                          "order": order, "letters": LETTERS,
                          "messages": msgs, "variant": f"dialogue_p{p:g}"})
        if cfg.get("include_text_baseline"):
            msgs = build_mcq_messages(it, order, condition="both",
                                      profile_mode="text",
                                      layout=cfg["layout"], letters=LETTERS)
            tasks.append({"query_id": it["query_id"],
                          "active_axis": it.get("active_axis"),
                          "p_implicit": None, "realized_implicit_frac": None,
                          "n_turns": 0, "order": order, "letters": LETTERS,
                          "messages": msgs, "variant": "text_profile_baseline"})
    return tasks


def analyze(records):
    by = defaultdict(list)
    for r in records:
        by[r["variant"]].append(r)
    out = {"by_variant": {}}
    for v, rs in sorted(by.items()):
        s = metrics.acc_summary(rs)
        fr = [r["realized_implicit_frac"] for r in rs
              if r["realized_implicit_frac"] is not None]
        s["realized_implicit_frac"] = round(sum(fr) / len(fr), 3) if fr else None
        out["by_variant"][v] = s
    base = out["by_variant"].get("text_profile_baseline")
    p0 = out["by_variant"].get("dialogue_p0")
    if base and p0:
        out["explicit_vs_text_gap"] = round(
            base["strict"]["acc"] - p0["strict"]["acc"], 4)
    return out


def run(cfg, items, client):
    records = execute_tasks(client, build_tasks(items, cfg))
    metrics.attach_scores(records)
    return records, analyze(records)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--p-list", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    ap.add_argument("--include-text-baseline", action="store_true")
    cfg = load_cfg(ap.parse_args())
    items = load_items(cfg["items"], cfg["limit"], need_entries=True)
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
    for p in cfg["p_list"]:
        t = [t for t in build_tasks(items[:1], cfg)
             if t["variant"] == f"dialogue_p{p:g}"]
        write_json(run_dir / f"sample_prompt_p{p:g}.json",
                   serialize_for_dump(t[0]["messages"]))

    print(f"\n[{EXP}] {cfg['model']}")
    for v, s in analysis["by_variant"].items():
        print(f"  [{v:>22}] strict={s['strict']['acc']:.3f} "
              f"tpo={s['tpo']['acc']:.3f} pref={s['pref']['acc']:.3f} "
              f"impl_frac={s['realized_implicit_frac']}")
    if "explicit_vs_text_gap" in analysis:
        print(f"  explicit-dialogue vs text-profile gap "
              f"(packaging cost): {analysis['explicit_vs_text_gap']}")
    print(f"  -> {run_dir}")


if __name__ == "__main__":
    main()
