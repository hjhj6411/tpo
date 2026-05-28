#!/usr/bin/env python3
"""
Text-Only No-Profile LLM Baseline Evaluation
--------------------------------------------
이미지 없이, 사용자 profile도 없이,
query_text + option 텍스트(attributes/search_query)만 보고
LLM이 어떤 옵션을 고르는지 평가합니다.

실행:
  python -m scripts.text_only_no_profile_eval
  python -m scripts.text_only_no_profile_eval --limit 50 --provider openai
  python -m scripts.text_only_no_profile_eval --limit 50 --verbose
"""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import call_llm, load_jsonl, save_jsonl, log_step
from configs.config import OPTIONS_DIR, QUERIES_DIR


SYSTEM_PROMPT = """\
You are a fashion advisor.

You will be given:
1. A fashion query describing a situation or occasion
2. Four clothing options (A, B, C, D) described by text only

Your task:
Select the single BEST option for the query.

Rules:
- Consider occasion / TPO appropriateness carefully
- Use only the information in the query and options
- Output ONLY a single letter: A, B, C, or D
- Do not explain. Just output the letter.
"""


def build_prompt(query, shuffled_options):
    option_lines = []
    for label, opt in shuffled_options:
        text = opt.get("search_query")
        if not text:
            a = opt.get("attributes", {})
            parts = []
            if a.get("color"):
                parts.append(a["color"])
            if a.get("pattern") and a["pattern"] != "solid":
                parts.append(a["pattern"].replace("_", " "))
            if a.get("garment_category"):
                parts.append(a["garment_category"].replace("_", " "))
            text = " ".join(parts).strip() or "unknown item"

        option_lines.append(f"  {label}. {text}")

    option_block = "\n".join(option_lines)

    prompt = f"""\
=== QUERY ===
{query.get('query_text', '').strip()}

=== OPTIONS ===
{option_block}

Which option best fits the query? Answer with only the letter (A/B/C/D):"""
    return prompt


def parse_answer(response: str):
    response = (response or "").strip()
    for ch in response:
        if ch in "ABCDabcd":
            return ch.upper()
    return None


def evaluate(plans, queries_map, limit=0, provider=None, seed=42, verbose=False):
    rng = random.Random(seed)

    if limit > 0:
        plans = plans[:limit]

    results = []
    correct = 0
    total = 0
    breakdown = defaultdict(lambda: {"correct": 0, "total": 0})

    for i, plan in enumerate(plans):
        qid = plan["query_id"]
        query = queries_map.get(qid)
        if query is None:
            continue

        option_items = list(plan["options"].items())   # [("A", {...}), ...]
        rng.shuffle(option_items)

        display_labels = ["A", "B", "C", "D"]
        shuffled = list(zip(display_labels, [opt for _, opt in option_items]))

        correct_display = None
        for disp_label, (orig_label, _) in zip(display_labels, option_items):
            if orig_label == "A":
                correct_display = disp_label
                break

        response = None
        predicted = None

        prompt = build_prompt(query, shuffled)
        try:
            response = call_llm(
                prompt=prompt,
                stage="text_only_no_profile_eval",
                system=SYSTEM_PROMPT,
                provider_override=provider,
            )
            predicted = parse_answer(response)
        except Exception as e:
            print(f"  [ERROR] {qid}: {e}")

        is_correct = (predicted == correct_display)
        if is_correct:
            correct += 1
        total += 1

        axis = plan.get("active_axis", "unknown")
        qtype = query.get("query_type", "unknown")

        breakdown[f"axis:{axis}"]["total"] += 1
        breakdown[f"qtype:{qtype}"]["total"] += 1
        if is_correct:
            breakdown[f"axis:{axis}"]["correct"] += 1
            breakdown[f"qtype:{qtype}"]["correct"] += 1

        results.append({
            "query_id": qid,
            "user_id": plan.get("user_id"),
            "active_axis": axis,
            "query_type": qtype,
            "scenario_id": plan.get("scenario_id"),
            "correct_display": correct_display,
            "predicted": predicted,
            "is_correct": is_correct,
            "raw_response": response,
        })

        if verbose or (i + 1) % 10 == 0:
            status = "✓" if is_correct else "✗"
            print(
                f"  [{i+1:3d}/{len(plans)}] {status} "
                f"pred={predicted} ans={correct_display} | "
                f"axis={axis} qtype={qtype}"
            )

    return results, correct, total, breakdown


def print_report(correct, total, breakdown):
    acc = correct / total * 100 if total > 0 else 0
    print("\n" + "=" * 50)
    print(f"  OVERALL ACCURACY: {correct}/{total} = {acc:.1f}%")
    print(f"  Random baseline:  25.0%")
    print(f"  Delta vs random:  {acc - 25.0:+.1f}%")
    print("=" * 50)

    print("\n  ── By active_axis ──")
    for key in sorted(k for k in breakdown if k.startswith("axis:")):
        d = breakdown[key]
        a = d["correct"] / d["total"] * 100 if d["total"] > 0 else 0
        print(f"  {key:25s}  {d['correct']}/{d['total']} = {a:.1f}%")

    print("\n  ── By query_type ──")
    for key in sorted(k for k in breakdown if k.startswith("qtype:")):
        d = breakdown[key]
        a = d["correct"] / d["total"] * 100 if d["total"] > 0 else 0
        print(f"  {key:25s}  {d['correct']}/{d['total']} = {a:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", type=Path,
                        default=OPTIONS_DIR / "option_plans.jsonl")
    parser.add_argument("--queries", type=Path,
                        default=QUERIES_DIR / "queries.jsonl")
    parser.add_argument("--output", type=Path,
                        default=OPTIONS_DIR / "text_only_no_profile_results.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="0 = all")
    parser.add_argument("--provider", type=str, default=None,
                        help="openai / anthropic / gemini")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    log_step("Text-Only No-Profile LLM Baseline Eval")

    plans = load_jsonl(args.plans)
    queries = load_jsonl(args.queries)
    queries_map = {q["query_id"]: q for q in queries}

    print(f"  Plans: {len(plans)}, Queries: {len(queries)}")
    if args.limit:
        print(f"  Limit: {args.limit}")

    results, correct, total, breakdown = evaluate(
        plans,
        queries_map,
        limit=args.limit,
        provider=args.provider,
        seed=args.seed,
        verbose=args.verbose,
    )

    save_jsonl(results, args.output)
    print(f"\n  Saved {len(results)} result records → {args.output}")

    print_report(correct, total, breakdown)


if __name__ == "__main__":
    main()