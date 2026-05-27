"""
Evaluator v2 — Multi-axis evaluation with per-archetype breakdown.

Changes from v1:
  - Per scenario_archetype metrics (not just per-axis)
  - Per preference_archetype metrics
  - Confusion matrix per archetype
  - Cross-archetype difficulty analysis
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .utils import (
    call_vlm, parse_json_response,
    save_jsonl, load_jsonl, save_json, log_step,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import FINAL_DIR


VLM_EVAL_SYSTEM = """You are evaluating a Vision-Language Model on a fashion personalization task.

Given:
- A user profile with fashion preferences
- A query describing a situation (weather, occasion, place, etc.)
- 4 product images labeled A, B, C, D

Pick the option that BEST satisfies BOTH:
  1. The user's personal taste (likes and dislikes)
  2. The situational requirements (TPO — Time, Place, Occasion)

Output ONLY JSON: {"answer": "A" or "B" or "C" or "D", "rationale": "..."}"""


def build_eval_prompt(record):
    profile = record["narrative_profile"]
    query = record["query_text"]
    likes = ", ".join(record.get("likes_keywords", [])) or "(none)"
    dislikes = ", ".join(record.get("dislikes_keywords", [])) or "(none)"

    return f"""USER PROFILE: {profile}
  likes: {likes}
  dislikes: {dislikes}

QUERY: "{query}"

The 4 images show options A, B, C, D (in order).
Which option BEST matches both the user's taste and the situation?
Output JSON: {{"answer": "A"/"B"/"C"/"D", "rationale": "..."}}"""


def evaluate_single(record, provider_override=None):
    """Run VLM evaluation on a single record."""
    if not record.get("images_complete"):
        return {"instance_id": record["instance_id"], "skipped": True,
                "reason": "images_incomplete"}

    image_paths = []
    for k in "ABCD":
        ip = record["options"][k].get("image_path")
        if not ip or not Path(ip).exists():
            return {"instance_id": record["instance_id"], "skipped": True,
                    "reason": f"missing_image_{k}"}
        image_paths.append(ip)

    prompt = build_eval_prompt(record)
    try:
        resp = call_vlm(prompt, image_paths=image_paths,
                        stage="vlm_evaluator", system=VLM_EVAL_SYSTEM,
                        provider_override=provider_override)
        parsed = parse_json_response(resp)
        if parsed and parsed.get("answer") in ("A", "B", "C", "D"):
            pred = parsed["answer"]
            gt = record["correct_answer"]
            return {
                "instance_id": record["instance_id"],
                "query_id": record["query_id"],
                "user_id": record["user_id"],
                "scenario_id": record.get("scenario_id"),
                "scenario_archetype": record.get("scenario_archetype"),
                "preference_archetype": record.get("preference_archetype"),
                "active_axis": record["active_axis"],
                "query_type": record.get("query_type"),
                "predicted": pred,
                "correct": gt,
                "is_correct": pred == gt,
                "rationale": parsed.get("rationale", ""),
                "skipped": False,
            }
    except Exception as e:
        return {"instance_id": record["instance_id"], "skipped": True,
                "reason": str(e)[:200]}

    return {"instance_id": record["instance_id"], "skipped": True,
            "reason": "parse_fail"}


def compute_metrics(eval_results):
    """Compute overall and per-slice accuracy metrics."""
    valid = [r for r in eval_results if not r.get("skipped")]
    if not valid:
        return {"n_total": len(eval_results), "n_valid": 0, "overall_acc": 0.0}

    n_correct = sum(1 for r in valid if r["is_correct"])
    overall_acc = n_correct / len(valid)

    # Per-slice breakdowns
    slices = {
        "by_active_axis": defaultdict(lambda: {"n": 0, "correct": 0}),
        "by_query_type": defaultdict(lambda: {"n": 0, "correct": 0}),
        "by_scenario_archetype": defaultdict(lambda: {"n": 0, "correct": 0}),
        "by_preference_archetype": defaultdict(lambda: {"n": 0, "correct": 0}),
    }

    for r in valid:
        for slice_key, field in [
            ("by_active_axis", "active_axis"),
            ("by_query_type", "query_type"),
            ("by_scenario_archetype", "scenario_archetype"),
            ("by_preference_archetype", "preference_archetype"),
        ]:
            val = r.get(field, "unknown")
            slices[slice_key][val]["n"] += 1
            if r["is_correct"]:
                slices[slice_key][val]["correct"] += 1

    # Confusion matrix (predicted label distribution)
    confusion = defaultdict(lambda: defaultdict(int))
    for r in valid:
        pred_label = _get_label(r, "predicted")
        gt_label = _get_label(r, "correct")
        confusion[gt_label][pred_label] += 1

    # Finalize slice accuracies
    metrics = {
        "n_total": len(eval_results),
        "n_valid": len(valid),
        "n_skipped": len(eval_results) - len(valid),
        "overall_acc": overall_acc,
        "slices": {},
        "confusion_matrix": dict(confusion),
    }

    for slice_key, data in slices.items():
        metrics["slices"][slice_key] = {}
        for k, v in data.items():
            acc = v["correct"] / max(v["n"], 1)
            metrics["slices"][slice_key][k] = {
                "n": v["n"], "correct": v["correct"], "acc": acc
            }

    return metrics


def _get_label(result, key):
    """Map predicted/correct option letter to its label type."""
    # We don't have the full record here, just the letter
    return result.get(key, "?")


def run_pipeline(benchmark_path, output_path, metrics_path,
                 limit=0, provider=None):
    log_step(f"Evaluator v2 (provider={provider or 'default'})")
    records = load_jsonl(benchmark_path)
    print(f"  Loaded {len(records)} benchmark records")

    if limit > 0:
        records = records[:limit]

    if output_path.exists():
        existing = load_jsonl(output_path)
        done = {r["instance_id"] for r in existing}
        results = existing
        todo = [r for r in records if r["instance_id"] not in done]
        print(f"  Resuming: {len(done)} already evaluated")
    else:
        results = []
        todo = records

    for i, rec in enumerate(todo):
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(todo)}]")
        try:
            res = evaluate_single(rec, provider_override=provider)
            results.append(res)
            if (i + 1) % 50 == 0:
                save_jsonl(results, output_path)
        except Exception as e:
            print(f"    ERROR: {e}")

    save_jsonl(results, output_path)
    print(f"\n  ✓ {len(results)} evaluation results")

    metrics = compute_metrics(results)
    save_json(metrics, metrics_path)

    print(f"\n  Overall accuracy: {metrics['overall_acc']:.3f} "
          f"({metrics['n_valid']} valid / {metrics['n_total']} total)")

    for slice_key in ["by_active_axis", "by_query_type",
                      "by_scenario_archetype", "by_preference_archetype"]:
        print(f"\n  {slice_key}:")
        for k, v in sorted(metrics["slices"].get(slice_key, {}).items()):
            print(f"    {k:30s}: {v['acc']:.3f} (n={v['n']})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_path", type=Path,
                        default=FINAL_DIR / "benchmark.jsonl")
    parser.add_argument("--output", type=Path,
                        default=FINAL_DIR / "eval_results.jsonl")
    parser.add_argument("--metrics", type=Path,
                        default=FINAL_DIR / "eval_metrics.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--provider", type=str, default=None)
    args = parser.parse_args()
    run_pipeline(args.benchmark_path, args.output, args.metrics,
                 args.limit, args.provider)


if __name__ == "__main__":
    main()
