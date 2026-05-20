"""
Evaluator v2 — per-axis OW indices, position shuffle, profile-variant ablation.
"""

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .utils import call_vlm, save_jsonl, load_jsonl, save_json, log_step

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import FINAL_DIR, PROVIDERS


EVAL_DIR = FINAL_DIR.parent / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


EVAL_SYSTEM = """You are a fashion personalization assistant. The user shows you their profile,
a query, and 4 candidate items as images.

Choose the option (A/B/C/D) that best matches BOTH:
  1. The user's enduring taste from their profile
  2. The situational requirement of the query

Output ONLY the letter A, B, C, or D.
"""


def evaluate_instance(instance, shuffle_seed, profile_variant,
                       provider_override=None):
    keys = list("ABCD")
    rng = random.Random(shuffle_seed)
    shuffled = list(keys)
    rng.shuffle(shuffled)
    paths = [instance["options"][k]["image_path"] for k in shuffled]
    if any(p is None for p in paths):
        return {"error": "missing image"}

    display_to_orig = {chr(ord("A") + i): shuffled[i] for i in range(4)}
    prompt = f"""USER PROFILE:
  {instance['profile_variants'][profile_variant]}

QUERY: "{instance['query']['text']}"

The 4 images you see are options A, B, C, D in that order.
Which option (A/B/C/D) is the best choice?
Answer with only the letter."""

    try:
        response = call_vlm(prompt, image_paths=paths, stage="vlm_evaluator",
                              system=EVAL_SYSTEM, max_tokens=8, temperature=0.0,
                              provider_override=provider_override)
    except Exception as e:
        return {"error": str(e)}

    pred_letter = None
    for c in (response or "").upper():
        if c in "ABCD":
            pred_letter = c
            break
    if not pred_letter:
        return {"error": "no letter", "raw": (response or "")[:120]}

    pred_orig = display_to_orig[pred_letter]
    correct = instance["correct_option"]
    return {
        "instance_id": instance["instance_id"],
        "active_axis": instance["active_axis"],
        "shuffled_order": shuffled,
        "predicted_displayed": pred_letter,
        "predicted_original": pred_orig,
        "correct_original": correct,
        "correct": pred_orig == correct,
        "predicted_label": instance["options"][pred_orig]["label"],
    }


def compute_metrics(results, instances):
    valid = [r for r in results if "error" not in r]
    n_valid = len(valid)
    if n_valid == 0:
        return {"error": "no valid results"}

    n_correct = sum(1 for r in valid if r["correct"])
    overall_acc = n_correct / n_valid

    by_axis = defaultdict(lambda: {"total": 0, "correct": 0, "wrong_picks": Counter()})
    qtype_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    confusion = Counter()

    for r in valid:
        inst = instances[r["instance_id"]]
        axis = r["active_axis"]
        qt = inst["query"]["type"]
        pred_label = r["predicted_label"]

        by_axis[axis]["total"] += 1
        if r["correct"]:
            by_axis[axis]["correct"] += 1
        else:
            by_axis[axis]["wrong_picks"][pred_label] += 1

        qtype_stats[qt]["total"] += 1
        if r["correct"]:
            qtype_stats[qt]["correct"] += 1

        correct_label = inst["options"][r["correct_original"]]["label"]
        confusion[(correct_label, pred_label)] += 1

    per_axis = {}
    for axis, st in by_axis.items():
        n_wrong = st["total"] - st["correct"]
        ow = {}
        for lbl in ("tpo_only", "preference_only", "neither"):
            ow[f"OW_{lbl}"] = (st["wrong_picks"].get(lbl, 0) / n_wrong) if n_wrong else 0.0
        per_axis[axis] = {
            "n": st["total"],
            "accuracy": st["correct"] / st["total"] if st["total"] else 0.0,
            "n_wrong": n_wrong, **ow,
            "interpretation": _interpret(ow),
        }

    qtype_acc = {q: s["correct"] / s["total"] for q, s in qtype_stats.items()}

    return {
        "n_total": len(results), "n_valid": n_valid,
        "n_errors": len(results) - n_valid,
        "overall_accuracy": overall_acc,
        "per_axis": per_axis,
        "query_type_accuracy": qtype_acc,
        "confusion_matrix": {f"{k[0]}->{k[1]}": v for k, v in confusion.items()},
    }


def _interpret(ow):
    notes = []
    if ow["OW_tpo_only"] > 0.45:
        notes.append("TPO-overweighting (PCogAlign-style)")
    if ow["OW_preference_only"] > 0.45:
        notes.append("Preference-overweighting / mechanical FAE")
    if ow["OW_neither"] > 0.30:
        notes.append("Cognitive collapse on this axis")
    return "; ".join(notes) or "No strong directional bias"


def run_pipeline(benchmark_path, output_dir, profile_variant="combined",
                  limit=0, n_seeds=2, provider_override=None):
    log_step(f"Evaluator v2 — variant={profile_variant}, provider={provider_override or 'default'}")
    stage_provider = PROVIDERS.get("vlm_evaluator", {}).get("provider", "?")
    print(f"  vlm_evaluator stage → {provider_override or stage_provider}")

    instances_list = load_jsonl(benchmark_path)
    if limit > 0:
        instances_list = instances_list[:limit]
    instances = {i["instance_id"]: i for i in instances_list}
    print(f"  loaded {len(instances_list)} instances")

    eff_provider = provider_override or stage_provider
    sub_dir = output_dir / f"{eff_provider}__{profile_variant}"
    sub_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for seed in range(n_seeds):
        print(f"\n  --- seed {seed} ---")
        seed_results = []
        for i, inst in enumerate(instances_list):
            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(instances_list)}]")
            try:
                r = evaluate_instance(inst, seed * 1000 + i, profile_variant,
                                        provider_override)
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
        for axis, am in m["per_axis"].items():
            print(f"    [{axis}] acc={am['accuracy']:.3f} "
                  f"OW_tpo={am['OW_tpo_only']:.2f} "
                  f"OW_pref={am['OW_preference_only']:.2f} "
                  f"OW_neither={am['OW_neither']:.2f}")

    overall = compute_metrics(all_results, instances)
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
    print(f"  Overall acc: {overall['overall_accuracy']:.3f}")
    print(f"  Position consistency: {overall.get('position_consistency', 0):.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_path", type=Path, default=FINAL_DIR / "pod_bench.jsonl")
    parser.add_argument("--profile_variant", type=str, default="combined",
                        choices=["keyword_only", "narrative_only", "combined"])
    parser.add_argument("--output_dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--n_seeds", type=int, default=2)
    parser.add_argument("--provider", type=str, default=None)
    args = parser.parse_args()
    run_pipeline(args.benchmark_path, args.output_dir,
                  args.profile_variant, args.limit, args.n_seeds, args.provider)


if __name__ == "__main__":
    main()
