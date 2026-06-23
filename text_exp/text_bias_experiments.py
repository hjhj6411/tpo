#!/usr/bin/env python3
"""
Text choice-bias experiments built on text_exp/text_eval.py.

Experiments:
  1) position_bias
     Randomly shuffle original A/B/C/D into displayed A/B/C/D and measure which
     displayed position the model prefers.

  2) fixed_correct
     Force the original correct option A to appear as displayed A, B, C, or D and
     compare accuracy by correct answer position.

  3) bc_tradeoff
     Remove original A (both TPO+preference) and original D (neither), leaving only:
       original B = TPO-only
       original C = preference-only
     Randomize their displayed positions and measure whether the model chooses
     scenario/TPO or user preference.

Examples:
  python -m text_exp.text_bias_experiments --experiment all --model vllm --concurrency 32
  python -m text_exp.text_bias_experiments --experiment fixed_correct --input-format all+query --model vllm
  python -m text_exp.text_bias_experiments --experiment bc_tradeoff --model Qwen/Qwen2.5-7B-Instruct
  python -m text_exp.text_bias_experiments --experiment bc_tradeoff --see --limit 1
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import OPTIONS_DIR, QUERIES_DIR, PROFILES_DIR
from src.utils import call_llm, load_jsonl, log_step, save_jsonl
from text_exp.text_eval import (
    INPUT_FORMATS,
    SUMMARY_GROUPS,
    TPO_SCORE,
    PROFILE_SCORE,
    build_prompt,
    format_uses_query,
    option_to_text,
    parse_answer,
    profile_to_all_kv_text,
    profile_to_narrative,
    provider_alias_for_model,
    resolve_model_name,
    stage_name_for,
    system_prompt_for,
)

EXPERIMENTS = ["position_bias", "fixed_correct", "bc_tradeoff"]

SYSTEM_PROMPT_TWO_CHOICE = """\
You are a fashion advisor.

You may be given:
1. A user's fashion query (situation or preference)
2. User profile information
3. Two clothing options (A, B) described by text attributes only

Your task:
Select the single BETTER option from A and B.

OUTPUT FORMAT — CRITICAL:
You MUST output EXACTLY one character: A or B.
Do NOT output any explanation, reasoning, punctuation, or whitespace.
Do NOT write sentences. Do NOT write words.
If you write anything other than a single letter, your response is INVALID.
Your ENTIRE response must be one of: A  B
"""

SYSTEM_PROMPT_TWO_CHOICE_QUERY_ONLY = """\
You are a fashion advisor.

You will be given:
1. A fashion query describing a situation or occasion
2. Two clothing options (A, B) described by text attributes only

Your task:
Select the single BETTER option from A and B.

OUTPUT FORMAT — CRITICAL:
You MUST output EXACTLY one character: A or B.
Do NOT output any explanation, reasoning, punctuation, or whitespace.
Do NOT write sentences. Do NOT write words.
If you write anything other than a single letter, your response is INVALID.
Your ENTIRE response must be one of: A  B
"""


# ── Prompt construction ────────────────────────────────────────────────
def build_two_choice_prompt(query, profile, displayed_options, input_format):
    """Build a B-vs-C prompt where displayed labels are only A/B."""
    option_block = "\n".join(
        f"  {display_label}. {option_to_text(option)}"
        for display_label, _original_label, option in displayed_options
    )

    sections = []
    if format_uses_query(input_format):
        qtext = (query or {}).get("query_text", "").strip()
        sections.append(("QUERY", qtext))

    if input_format in {"narrative", "narrative+query"}:
        sections.append(("USER PROFILE", profile_to_narrative(profile)))
    elif input_format in {"all", "all+query"}:
        sections.append(("USER PROFILE", profile_to_all_kv_text(profile)))
    elif input_format != "query":
        raise ValueError(f"Unknown input_format: {input_format}")

    sections.append(("OPTIONS", option_block))

    if input_format == "query":
        task_line = "Select the better option for the query."
    elif format_uses_query(input_format):
        task_line = "Select the better option considering both the query and the user's preferences."
    else:
        task_line = "Select the better option for the user's preferences."

    body = "\n\n".join(f"=== {title} ===\n{text}" for title, text in sections)
    return f"""\
{body}

=== INSTRUCTION ===
{task_line}
Respond with ONE letter only: A or B.
Do NOT write any explanation or reasoning.
Do NOT write anything before or after the letter.
Your complete response must be a single character.

Answer:"""


def two_choice_system_prompt(input_format):
    if input_format == "query":
        return SYSTEM_PROMPT_TWO_CHOICE_QUERY_ONLY
    return SYSTEM_PROMPT_TWO_CHOICE


# ── Job builders ────────────────────────────────────────────────────────
def _limit_plans(plans, limit):
    return plans[:limit] if limit and limit > 0 else plans


def make_position_bias_jobs(plans, queries_map, profiles_map, input_format, seed, limit, trials):
    rng = random.Random(seed)
    jobs = []
    for plan in _limit_plans(plans, limit):
        query = queries_map.get(plan["query_id"], {})
        profile = profiles_map.get(plan["user_id"], {})
        if format_uses_query(input_format) and not query:
            continue

        original_items = list(plan["options"].items())
        for trial in range(trials):
            items = original_items[:]
            rng.shuffle(items)
            display_labels = ["A", "B", "C", "D"]
            displayed = list(zip(display_labels, [opt for _, opt in items]))
            display_to_original = {d: orig for d, (orig, _opt) in zip(display_labels, items)}
            original_to_display = {orig: d for d, orig in display_to_original.items()}
            prompt = build_prompt(query, profile, displayed, input_format=input_format)
            jobs.append({
                "experiment": "position_bias",
                "trial": trial,
                "plan": plan,
                "query": query,
                "prompt": prompt,
                "display_to_original": display_to_original,
                "correct_display": original_to_display.get("A"),
                "condition": "random_shuffle",
            })
    return jobs


def make_fixed_correct_jobs(plans, queries_map, profiles_map, input_format, seed, limit, trials):
    rng = random.Random(seed)
    jobs = []
    for plan in _limit_plans(plans, limit):
        query = queries_map.get(plan["query_id"], {})
        profile = profiles_map.get(plan["user_id"], {})
        if format_uses_query(input_format) and not query:
            continue

        options = plan["options"]
        for fixed_label in ["A", "B", "C", "D"]:
            for trial in range(trials):
                remaining_display = [x for x in ["A", "B", "C", "D"] if x != fixed_label]
                remaining_original = [x for x in ["B", "C", "D"]]
                rng.shuffle(remaining_original)

                display_to_original = {fixed_label: "A"}
                display_to_original.update(dict(zip(remaining_display, remaining_original)))

                displayed = [(d, options[display_to_original[d]]) for d in ["A", "B", "C", "D"]]
                prompt = build_prompt(query, profile, displayed, input_format=input_format)
                jobs.append({
                    "experiment": "fixed_correct",
                    "trial": trial,
                    "plan": plan,
                    "query": query,
                    "prompt": prompt,
                    "display_to_original": display_to_original,
                    "correct_display": fixed_label,
                    "condition": fixed_label,
                })
    return jobs


def make_bc_tradeoff_jobs(plans, queries_map, profiles_map, input_format, seed, limit, trials):
    rng = random.Random(seed)
    jobs = []
    for plan in _limit_plans(plans, limit):
        query = queries_map.get(plan["query_id"], {})
        profile = profiles_map.get(plan["user_id"], {})
        if format_uses_query(input_format) and not query:
            continue

        options = plan["options"]
        for trial in range(trials):
            originals = ["B", "C"]  # B = TPO-only, C = preference-only
            rng.shuffle(originals)
            display_labels = ["A", "B"]
            display_to_original = dict(zip(display_labels, originals))
            displayed = [(d, display_to_original[d], options[display_to_original[d]]) for d in display_labels]
            prompt = build_two_choice_prompt(query, profile, displayed, input_format)
            jobs.append({
                "experiment": "bc_tradeoff",
                "trial": trial,
                "plan": plan,
                "query": query,
                "prompt": prompt,
                "display_to_original": display_to_original,
                "correct_display": None,
                "condition": "B_vs_C",
            })
    return jobs


def make_jobs(experiment, plans, queries_map, profiles_map, input_format, seed, limit, trials):
    if experiment == "position_bias":
        return make_position_bias_jobs(plans, queries_map, profiles_map, input_format, seed, limit, trials)
    if experiment == "fixed_correct":
        return make_fixed_correct_jobs(plans, queries_map, profiles_map, input_format, seed, limit, trials)
    if experiment == "bc_tradeoff":
        return make_bc_tradeoff_jobs(plans, queries_map, profiles_map, input_format, seed, limit, trials)
    raise ValueError(f"Unknown experiment: {experiment}")


# ── Running ─────────────────────────────────────────────────────────────
def call_model(prompt, input_format, model, two_choice=False):
    provider_alias = provider_alias_for_model(model)
    if two_choice:
        return call_llm(
            prompt=prompt,
            stage=stage_name_for(input_format),
            system=two_choice_system_prompt(input_format),
            provider_override=provider_alias,
        )
    return call_llm(
        prompt=prompt,
        stage=stage_name_for(input_format),
        system=system_prompt_for(input_format),
        provider_override=provider_alias,
    )


def process_job(idx, job, input_format, model):
    plan = job["plan"]
    experiment = job["experiment"]
    two_choice = experiment == "bc_tradeoff"
    response = predicted = predicted_original = None

    try:
        response = call_model(job["prompt"], input_format, model, two_choice=two_choice)
        predicted = parse_answer(response)
        predicted_original = job["display_to_original"].get(predicted)
    except Exception as e:
        print(f"  [ERROR] {experiment} {plan.get('query_id')}: {e}")

    strict_hit = int(predicted_original == "A") if predicted_original else 0
    tpo_hit = TPO_SCORE.get(predicted_original, 0) if predicted_original else 0
    profile_hit = PROFILE_SCORE.get(predicted_original, 0) if predicted_original else 0

    if experiment == "bc_tradeoff":
        tradeoff_choice = None
        if predicted_original == "B":
            tradeoff_choice = "scenario_tpo"
        elif predicted_original == "C":
            tradeoff_choice = "preference"
    else:
        tradeoff_choice = None

    return {
        "_idx": idx,
        "experiment": experiment,
        "input_format": input_format,
        "condition": job["condition"],
        "trial": job["trial"],
        "query_id": plan.get("query_id"),
        "user_id": plan.get("user_id"),
        "scenario_id": plan.get("scenario_id"),
        "active_axis": plan.get("active_axis"),
        "query_type": plan.get("query_type"),
        "display_to_original": job["display_to_original"],
        "correct_display": job.get("correct_display"),
        "predicted": predicted,
        "predicted_original": predicted_original,
        "strict_correct": bool(strict_hit),
        "tpo_score": tpo_hit,
        "profile_score": profile_hit,
        "tradeoff_choice": tradeoff_choice,
        "raw_response": response,
    }


def run_experiment(experiment, plans, queries_map, profiles_map, input_format,
                   model, seed, limit, trials, concurrency, verbose=False):
    jobs = make_jobs(experiment, plans, queries_map, profiles_map, input_format, seed, limit, trials)
    results = []

    if concurrency > 1:
        with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {ex.submit(process_job, i, job, input_format, model): i for i, job in enumerate(jobs)}
            done = 0
            for fut in cf.as_completed(futures):
                rec = fut.result()
                results.append(rec)
                done += 1
                if verbose or done % 500 == 0:
                    print(f"  [{done}/{len(jobs)}] pred={rec['predicted']} orig={rec['predicted_original']}")
    else:
        for i, job in enumerate(jobs):
            rec = process_job(i, job, input_format, model)
            results.append(rec)
            if verbose or (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(jobs)}] pred={rec['predicted']} orig={rec['predicted_original']}")

    results.sort(key=lambda r: r["_idx"])
    for r in results:
        r.pop("_idx")
    return results


# ── Reporting ──────────────────────────────────────────────────────────
def pct(num, den):
    return num / den * 100 if den else 0.0


def print_position_bias_report(results):
    n = len(results)
    pred_display = Counter(r["predicted"] or "INVALID" for r in results)
    pred_original = Counter(r["predicted_original"] or "INVALID" for r in results)
    strict = sum(1 for r in results if r["predicted_original"] == "A")

    print("\n  ── Position bias: displayed choice distribution ──")
    for label in ["A", "B", "C", "D", "INVALID"]:
        c = pred_display.get(label, 0)
        print(f"  display {label:7s}: {c:5d} / {n:5d} = {pct(c, n):5.1f}%")

    print("\n  ── Position bias: original semantic choice distribution ──")
    for label in ["A", "B", "C", "D", "INVALID"]:
        c = pred_original.get(label, 0)
        semantic = {
            "A": "both",
            "B": "tpo_only",
            "C": "preference_only",
            "D": "neither",
        }.get(label, "invalid")
        print(f"  original {label:7s} ({semantic:15s}): {c:5d} / {n:5d} = {pct(c, n):5.1f}%")
    print(f"\n  strict accuracy after random shuffling: {strict}/{n} = {pct(strict, n):.1f}%")


def print_fixed_correct_report(results):
    print("\n  ── Fixed correct-position accuracy ──")
    print(f"  {'correct_pos':<12} {'strict':>8} {'tpo':>8} {'profile':>9} {'n':>8}")
    print("  " + "-" * 52)
    for pos in ["A", "B", "C", "D"]:
        rows = [r for r in results if r["condition"] == pos]
        n = len(rows)
        strict = sum(1 for r in rows if r["predicted_original"] == "A")
        tpo = sum(r["tpo_score"] for r in rows)
        prof = sum(r["profile_score"] for r in rows)
        print(f"  {pos:<12} {pct(strict, n):7.1f}% {pct(tpo, n):7.1f}% {pct(prof, n):8.1f}% {n:8d}")

    print("\n  ── Fixed correct-position predicted display distribution ──")
    for pos in ["A", "B", "C", "D"]:
        rows = [r for r in results if r["condition"] == pos]
        n = len(rows)
        counts = Counter(r["predicted"] or "INVALID" for r in rows)
        dist = "  ".join(f"{k}:{pct(counts.get(k,0), n):4.1f}%" for k in ["A", "B", "C", "D", "INVALID"])
        print(f"  correct={pos}: {dist}")


def print_bc_tradeoff_report(results):
    n = len(results)
    scenario = sum(1 for r in results if r["tradeoff_choice"] == "scenario_tpo")
    pref = sum(1 for r in results if r["tradeoff_choice"] == "preference")
    invalid = n - scenario - pref

    print("\n  ── B vs C tradeoff ──")
    print(f"  scenario/TPO choice      : {scenario:5d} / {n:5d} = {pct(scenario, n):5.1f}%")
    print(f"  preference choice        : {pref:5d} / {n:5d} = {pct(pref, n):5.1f}%")
    print(f"  invalid/other            : {invalid:5d} / {n:5d} = {pct(invalid, n):5.1f}%")

    print("\n  ── B vs C by active_axis ──")
    for axis in sorted(set(r["active_axis"] for r in results)):
        rows = [r for r in results if r["active_axis"] == axis]
        nn = len(rows)
        s = sum(1 for r in rows if r["tradeoff_choice"] == "scenario_tpo")
        p = sum(1 for r in rows if r["tradeoff_choice"] == "preference")
        print(f"  {axis:<12} scenario={pct(s, nn):5.1f}%  preference={pct(p, nn):5.1f}%  (n={nn})")

    print("\n  ── B vs C by query_type ──")
    for qt in sorted(set(r["query_type"] for r in results)):
        rows = [r for r in results if r["query_type"] == qt]
        nn = len(rows)
        s = sum(1 for r in rows if r["tradeoff_choice"] == "scenario_tpo")
        p = sum(1 for r in rows if r["tradeoff_choice"] == "preference")
        print(f"  {qt:<12} scenario={pct(s, nn):5.1f}%  preference={pct(p, nn):5.1f}%  (n={nn})")


def print_report(experiment, input_format, model_name, results):
    print("\n" + "=" * 78)
    print(f"  EXPERIMENT:   {experiment}")
    print(f"  INPUT FORMAT: {input_format}")
    print(f"  MODEL:        {model_name}")
    print(f"  N:            {len(results)}")
    print("=" * 78)

    if experiment == "position_bias":
        print_position_bias_report(results)
    elif experiment == "fixed_correct":
        print_fixed_correct_report(results)
    elif experiment == "bc_tradeoff":
        print_bc_tradeoff_report(results)


def summarize_for_final(experiment, input_format, results):
    n = len(results)
    out = {"experiment": experiment, "input_format": input_format, "n": n}
    if experiment == "position_bias":
        out["pred_display"] = dict(Counter(r["predicted"] or "INVALID" for r in results))
        out["pred_original"] = dict(Counter(r["predicted_original"] or "INVALID" for r in results))
        out["strict_acc"] = pct(sum(1 for r in results if r["predicted_original"] == "A"), n)
    elif experiment == "fixed_correct":
        out["strict_by_correct_pos"] = {}
        for pos in ["A", "B", "C", "D"]:
            rows = [r for r in results if r["condition"] == pos]
            out["strict_by_correct_pos"][pos] = pct(sum(1 for r in rows if r["predicted_original"] == "A"), len(rows))
    elif experiment == "bc_tradeoff":
        out["scenario_rate"] = pct(sum(1 for r in results if r["tradeoff_choice"] == "scenario_tpo"), n)
        out["preference_rate"] = pct(sum(1 for r in results if r["tradeoff_choice"] == "preference"), n)
    return out


def print_final_summary(rows):
    print("\n" + "=" * 90)
    print("  FINAL SUMMARY")
    print("=" * 90)
    for r in rows:
        exp = r["experiment"]
        fmt = r["input_format"]
        if exp == "position_bias":
            dist = r["pred_display"]
            print(f"  {exp:<15} {fmt:<18} strict={r['strict_acc']:5.1f}%  "
                  f"display A/B/C/D={dist.get('A',0)}/{dist.get('B',0)}/{dist.get('C',0)}/{dist.get('D',0)}")
        elif exp == "fixed_correct":
            d = r["strict_by_correct_pos"]
            print(f"  {exp:<15} {fmt:<18} acc when correct A/B/C/D="
                  f"{d.get('A',0):5.1f}/{d.get('B',0):5.1f}/{d.get('C',0):5.1f}/{d.get('D',0):5.1f}")
        elif exp == "bc_tradeoff":
            print(f"  {exp:<15} {fmt:<18} scenario={r['scenario_rate']:5.1f}%  preference={r['preference_rate']:5.1f}%")


# ── Preview ─────────────────────────────────────────────────────────────
def print_preview(experiment, plans, queries_map, profiles_map, input_format, seed, limit, trials):
    jobs = make_jobs(experiment, plans, queries_map, profiles_map, input_format, seed, limit or 1, trials)
    if not jobs:
        print("No preview job.")
        return
    job = jobs[0]
    print("=" * 78)
    print(f"PREVIEW experiment={experiment} input_format={input_format}")
    print(f"query_id={job['plan'].get('query_id')} user_id={job['plan'].get('user_id')}")
    print(f"display_to_original={job['display_to_original']}")
    print("=" * 78)
    print(job["prompt"])


# ── Main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=EXPERIMENTS + ["all"], default="all")
    parser.add_argument("--input-format", choices=INPUT_FORMATS, default=None,
                        help="If omitted, run all input formats.")
    parser.add_argument("--plans", type=Path, default=OPTIONS_DIR / "option_plans.jsonl")
    parser.add_argument("--queries", type=Path, default=QUERIES_DIR / "queries.jsonl")
    parser.add_argument("--profiles", type=Path, default=PROFILES_DIR / "profiles.jsonl")
    parser.add_argument("--output-dir", type=Path, default=OPTIONS_DIR / "text_bias")
    parser.add_argument("--model", type=str, default=None,
                        help="Provider alias such as vllm/gpt5_mini, or raw model name on the vllm endpoint.")
    parser.add_argument("--limit", type=int, default=0, help="0 = all plans")
    parser.add_argument("--trials", type=int, default=1,
                        help="Random shuffles per plan/condition. fixed_correct runs 4*trials per plan.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--see", action="store_true")
    args = parser.parse_args()

    log_step("Text Choice-Bias Experiments")
    plans = load_jsonl(args.plans)
    queries = load_jsonl(args.queries)
    profiles = load_jsonl(args.profiles)
    queries_map = {q["query_id"]: q for q in queries}
    profiles_map = {p["user_id"]: p for p in profiles}

    experiments = EXPERIMENTS if args.experiment == "all" else [args.experiment]
    input_formats = INPUT_FORMATS if args.input_format is None else [args.input_format]
    model_name = resolve_model_name(args.model, input_formats[0])

    print(f"  Plans: {len(plans)}, Queries: {len(queries)}, Profiles: {len(profiles)}")
    print(f"  Experiments: {experiments}")
    print(f"  Input formats: {input_formats}")
    print(f"  Model: {model_name}")
    print(f"  Limit: {args.limit or 'all'}, Trials: {args.trials}, Concurrency: {args.concurrency}")

    if args.see:
        for exp in experiments:
            for fmt in input_formats:
                print_preview(exp, plans, queries_map, profiles_map, fmt, args.seed, args.limit, args.trials)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_rows = []

    for exp in experiments:
        for fmt in input_formats:
            print(f"\n{'─' * 78}")
            print(f"  ▶ experiment={exp} | input_format={fmt} | model={model_name}")
            print(f"{'─' * 78}")
            results = run_experiment(
                exp, plans, queries_map, profiles_map, fmt,
                model=args.model,
                seed=args.seed,
                limit=args.limit,
                trials=args.trials,
                concurrency=args.concurrency,
                verbose=args.verbose,
            )
            out_path = args.output_dir / f"{exp}__{fmt.replace('+', '_plus_')}.jsonl"
            save_jsonl(results, out_path)
            print(f"  Saved {len(results)} records → {out_path}")
            print_report(exp, fmt, model_name, results)
            final_rows.append(summarize_for_final(exp, fmt, results))

    print_final_summary(final_rows)


if __name__ == "__main__":
    main()
