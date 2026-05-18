"""
Evaluator
==========

Run a VLM on the assembled POD-Bench and produce:
  - Overall accuracy
  - Per-query-type accuracy
  - Confusion matrix (correct label vs predicted label)
  - Wrong-pick diagnostic:
      * high B (tpo_only) selection → TPO-overweighting (PCogAlign failure mode)
      * high C (preference_only) selection → preference-overweighting (Whose Boat? / FSPO failure mode)
  - Position consistency across 2 shuffle seeds

Profile-variant ablation:
  --profile_variant {keyword_only, narrative_only, combined}
"""

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

from .utils import (
    call_local_vlm, save_jsonl, load_jsonl, save_json, log_step,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import FINAL_DIR, FREE_MODELS


EVAL_DIR = FINAL_DIR.parent / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


EVAL_SYSTEM = """\
You are a fashion personalization assistant. The user shows you their profile,
a query, and 4 candidate items as images.

Choose the option (A/B/C/D) that best matches BOTH:
  1. The user's enduring taste from their profile
  2. The situational requirement of the query

Output ONLY the letter A, B, C, or D.
"""


def build_prompt(instance: dict, profile_variant: str) -> str:
    return f"""USER PROFILE:
  {instance['profile_variants'][profile_variant]}

QUERY: "{instance['query']['text']}"

The 4 images you see are options A, B, C, D in that order.
Which option (A/B/C/D) is the best choice?
Answer with only the letter."""


def evaluate_instance(instance: dict, model_cfg: dict,
                      shuffle_seed: int, profile_variant: str) -> dict:
    keys_original = list("ABCD")
    rng = random.Random(shuffle_seed)
    shuffled = list(keys_original)
    rng.shuffle(shuffled)

    image_paths = [instance["options"][k]["image_path"] for k in shuffled]
    if any(p is None for p in image_paths):
        return {"error": "missing image"}

    # mapping from letter shown → original key
    display_to_orig = {chr(ord("A") + i): shuffled[i] for i in range(4)}
    prompt = build_prompt(instance, profile_variant)

    try:
        response = call_local_vlm(
            prompt=prompt, image_paths=image_paths, system=EVAL_SYSTEM,
            api_base=model_cfg["api_base"], model=model_cfg["model_name"],
            max_tokens=8, temperature=0.0,
        )
    except Exception as e:
        return {"error": str(e)}

    pred_letter = None
    for c in (response or "").upper():
        if c in "ABCD":
            pred_letter = c
            break
    if not pred_letter:
        return {"error": "no letter", "raw": (response or "")[:100]}

    pred_orig = display_to_orig[pred_letter]
    correct = instance["correct_option"]
    return {
        "instance_id": instance["instance_id"],
        "shuffled_order": shuffled,
        "predicted_displayed": pred_letter,
        "predicted_original": pred_orig,
        "correct_original": correct,
        "correct": pred_orig == correct,
        "predicted_label": instance["options"][pred_orig]["label"],
        "raw": (response or "")[:120],
    }


def compute_metrics(results: list, instances: dict) -> dict:
    valid = [r for r in results if "error" not in r]
    n_valid = len(valid)
    if n_valid == 0:
        return {"error": "no valid results"}

    n_correct = sum(1 for r in valid if r["correct"])
    overall = n_correct / n_valid

    qtype_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    confusion = Counter()
    wrong_picks = Counter()

    for r in valid:
        inst = instances[r["instance_id"]]
        qt = inst["query"]["type"]
        qtype_stats[qt]["total"] += 1
        if r["correct"]:
            qtype_stats[qt]["correct"] += 1

        correct_label = inst["options"][r["correct_original"]]["label"]
        pred_label = r["predicted_label"]
        confusion[(correct_label, pred_label)] += 1
        if not r["correct"]:
            wrong_picks[pred_label] += 1

    n_wrong = n_valid - n_correct
    diagnostic = {}
    if n_wrong > 0:
        for lbl in ["tpo_only", "preference_only", "neither"]:
            diagnostic[f"wrong_picked_{lbl}_ratio"] = wrong_picks.get(lbl, 0) / n_wrong

    qtype_acc = {q: s["correct"] / s["total"] for q, s in qtype_stats.items()}

    return {
        "n_total": len(results),
        "n_valid": n_valid,
        "n_errors": len(results) - n_valid,
        "overall_accuracy": overall,
        "query_type_accuracy": qtype_acc,
        "confusion_matrix": {f"{k[0]}->{k[1]}": v for k, v in confusion.items()},
        "wrong_pick_diagnostic": diagnostic,
        "interpretation": _interpret(diagnostic),
    }


def _interpret(diag: dict) -> str:
    tpo_r = diag.get("wrong_picked_tpo_only_ratio", 0)
    pref_r = diag.get("wrong_picked_preference_only_ratio", 0)
    neither_r = diag.get("wrong_picked_neither_ratio", 0)
    notes = []
    if tpo_r > 0.45:
        notes.append("TPO-overweighting (PCogAlign-style failure)")
    if pref_r > 0.45:
        notes.append("Preference-overweighting (Whose Boat? / SynthesizeMe / FSPO-style failure, 'mechanical FAE')")
    if neither_r > 0.30:
        notes.append("Random / confused predictions")
    return "; ".join(notes) or "No strong directional bias"


def run_pipeline(benchmark_path: Path, model_key: str, output_dir: Path,
                 profile_variant: str = "combined",
                 limit: int = 0, n_seeds: int = 2):
    log_step(f"Evaluator — model={model_key}, variant={profile_variant}")

    if model_key in FREE_MODELS:
        cfg = FREE_MODELS[model_key]
    else:
        cfg = {"model_name": model_key, "api_base": "http://localhost:8000/v1"}
    print(f"  model: {cfg['model_name']}, api: {cfg['api_base']}")

    instances_list = load_jsonl(benchmark_path)
    if limit > 0:
        instances_list = instances_list[:limit]
    instances = {i["instance_id"]: i for i in instances_list}
    print(f"  loaded {len(instances_list)} instances")

    sub_dir = output_dir / f"{cfg['model_name'].replace('/', '_')}__{profile_variant}"
    sub_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for seed in range(n_seeds):
        print(f"\n  --- seed {seed} ---")
        seed_results = []
        for i, inst in enumerate(instances_list):
            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(instances_list)}]")
            try:
                r = evaluate_instance(inst, cfg, seed * 1000 + i, profile_variant)
                r["seed"] = seed
                seed_results.append(r)
            except Exception as e:
                seed_results.append({"instance_id": inst["instance_id"],
                                      "error": str(e), "seed": seed})
            if (i + 1) % 50 == 0:
                save_jsonl(seed_results, sub_dir / f"results_seed{seed}.jsonl")

        save_jsonl(seed_results, sub_dir / f"results_seed{seed}.jsonl")
        all_results.extend(seed_results)
        m = compute_metrics(seed_results, instances)
        save_json(m, sub_dir / f"metrics_seed{seed}.json")
        print(f"  seed {seed} acc: {m['overall_accuracy']:.3f}")
        print(f"  diagnostic: {m['interpretation']}")

    overall = compute_metrics(all_results, instances)

    # Position consistency
    if n_seeds >= 2:
        per_inst = defaultdict(list)
        for r in all_results:
            if "error" not in r:
                per_inst[r["instance_id"]].append(r["correct"])
        consistent = sum(1 for v in per_inst.values()
                          if len(v) >= 2 and len(set(v)) == 1)
        total = sum(1 for v in per_inst.values() if len(v) >= 2)
        overall["position_consistency"] = consistent / max(total, 1)

    save_json(overall, sub_dir / "metrics_overall.json")
    print(f"\n  ✓ Saved to {sub_dir}")
    print(f"  Overall: {overall['overall_accuracy']:.3f}")
    print(f"  Position consistency: {overall.get('position_consistency', 0):.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_path", type=Path,
                        default=FINAL_DIR / "pod_bench.jsonl")
    parser.add_argument("--model", type=str, default="vlm_evaluator")
    parser.add_argument("--profile_variant", type=str, default="combined",
                        choices=["keyword_only", "narrative_only", "combined"])
    parser.add_argument("--output_dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--n_seeds", type=int, default=2)
    args = parser.parse_args()
    run_pipeline(args.benchmark_path, args.model, args.output_dir,
                 args.profile_variant, args.limit, args.n_seeds)


if __name__ == "__main__":
    main()
