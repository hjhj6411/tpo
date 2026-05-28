#!/usr/bin/env python3
"""
Text-Only LLM Baseline Evaluation
-----------------------------------
이미지 없이, option_plans의 텍스트 메타데이터(attributes + search_query)만 보고
LLM이 올바른 옵션(tpo_and_preference = A)을 고를 수 있는지 측정합니다.

실행:
  python -m scripts.text_only_eval
  python -m scripts.text_only_eval --limit 50 --provider openai
  python -m scripts.text_only_eval --limit 50 --verbose
"""

import argparse
import random
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import call_llm, load_jsonl, save_jsonl, log_step
from configs.config import OPTIONS_DIR, QUERIES_DIR, PROFILES_DIR


# ── Prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a fashion advisor. You will be given:
1. A user's fashion query (situation or preference)
2. The user's fashion preferences (likes / dislikes)
3. Four clothing options (A, B, C, D) described by text attributes only

Your task: select the single BEST option that satisfies BOTH the occasion/TPO requirement AND the user's personal preferences.

Rules:
- Consider garment_category, color, and pattern appropriateness for the context
- Consider whether the option matches what the user likes
- Output ONLY a single letter: A, B, C, or D
- Do not explain. Just output the letter."""


def build_prompt(query, profile, shuffled_options):
    narrative = profile.get("narrative", "(no profile description)")

    option_lines = []
    for label, opt in shuffled_options:
        a = opt["attributes"]
        line = f"  {label}. {opt.get('search_query', '?')}"
        option_lines.append(line)

    option_block = "\n".join(option_lines)

    prompt = f"""\

=== QUERY ===
{query['query_text']}

=== USER PROFILE ===
{narrative}

=== OPTIONS ===
{option_block}

Which option is best? Output ONLY the letter (A/B/C/D):"""

    return prompt


def parse_answer(response: str) -> str | None:
    """LLM 응답에서 A/B/C/D 추출"""
    response = response.strip()
    for ch in response:
        if ch in "ABCDabcd":
            return ch.upper()
    return None


def evaluate(plans, queries_map, profiles_map,
             limit=0, provider=None, seed=42, verbose=False):
    rng = random.Random(seed)

    if limit > 0:
        plans = plans[:limit]

    results = []
    correct = 0
    total = 0

    # 통계 분류용
    breakdown = defaultdict(lambda: {"correct": 0, "total": 0})

    for i, plan in enumerate(plans):
        qid = plan["query_id"]
        uid = plan["user_id"]
        query   = queries_map.get(qid)
        profile = profiles_map.get(uid)
        if query is None or profile is None:
            continue

        # ── ABCD를 랜덤 순서로 섞기 ──────────────────
        # 원래 A = tpo_and_preference (정답)
        option_items = list(plan["options"].items())  # [("A", {...}), ...]
        rng.shuffle(option_items)

        display_labels = ["A", "B", "C", "D"]
        shuffled = list(zip(display_labels, [opt for _, opt in option_items]))

        # 정답 레이블이 shuffle 후 어디로 갔는지 추적
        correct_display = None
        for disp_label, (orig_label, _) in zip(display_labels, option_items):
            if orig_label == "A":  # "A" = tpo_and_preference
                correct_display = disp_label
                break

        # ── LLM 호출 ─────────────────────────────────
        prompt = build_prompt(query, profile, shuffled)
        try:
            response = call_llm(
                prompt=prompt,
                stage="text_only_eval",
                system=SYSTEM_PROMPT,
                provider_override=provider,
            )
            predicted = parse_answer(response)
        except Exception as e:
            print(f"  [ERROR] {qid}: {e}")
            predicted = None

        is_correct = (predicted == correct_display)
        if is_correct:
            correct += 1
        total += 1

        # breakdown
        axis = plan.get("active_axis", "unknown")
        qtype = query.get("query_type", "unknown")
        breakdown[f"axis:{axis}"]["total"] += 1
        breakdown[f"qtype:{qtype}"]["total"] += 1
        if is_correct:
            breakdown[f"axis:{axis}"]["correct"] += 1
            breakdown[f"qtype:{qtype}"]["correct"] += 1

        result_rec = {
            "query_id":        qid,
            "user_id":         uid,
            "active_axis":     axis,
            "query_type":      qtype,
            "scenario_id":     plan.get("scenario_id"),
            "correct_display": correct_display,
            "predicted":       predicted,
            "is_correct":      is_correct,
            "raw_response":    response if "response" in dir() else None,
        }
        results.append(result_rec)

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
    parser.add_argument("--plans",    type=Path,
                        default=OPTIONS_DIR / "option_plans.jsonl")
    parser.add_argument("--queries",  type=Path,
                        default=QUERIES_DIR / "queries.jsonl")
    parser.add_argument("--profiles", type=Path,
                        default=PROFILES_DIR / "profiles.jsonl")
    parser.add_argument("--output",   type=Path,
                        default=OPTIONS_DIR / "text_only_results.jsonl")
    parser.add_argument("--limit",    type=int, default=0,
                        help="0 = all")
    parser.add_argument("--provider", type=str, default=None,
                        help="openai / anthropic / gemini")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args()

    log_step("Text-Only LLM Baseline Eval")

    plans    = load_jsonl(args.plans)
    queries  = load_jsonl(args.queries)
    profiles = load_jsonl(args.profiles)

    queries_map  = {q["query_id"]: q for q in queries}
    profiles_map = {p["user_id"]:  p for p in profiles}

    print(f"  Plans: {len(plans)}, Queries: {len(queries)}, Profiles: {len(profiles)}")
    if args.limit:
        print(f"  Limit: {args.limit}")

    results, correct, total, breakdown = evaluate(
        plans, queries_map, profiles_map,
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