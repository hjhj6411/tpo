#!/usr/bin/env python3
"""
Text experiment variants preserving scripts/text_only_eval.py format.

Input formats:
  - all             : all profile key-value fields, no query
  - all+query       : all profile key-value fields + query
  - query           : query only, no profile
  - narrative       : narrative profile only, no query
  - narrative+query : narrative profile + query

Default report grouping:
  all / all+query / query /// narrative / narrative+query / query

Examples:
  python -m text_exp.text_eval --model vllm --limit 50 --concurrency 32
  python -m text_exp.text_eval --model gpt5_mini --input-format all+query --limit 50
  python -m text_exp.text_eval --see --limit 1
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import call_llm, load_jsonl, log_step, save_jsonl
from configs.config import OPTIONS_DIR, QUERIES_DIR, PROFILES_DIR, PROVIDERS, PROVIDER_ENDPOINTS


SYSTEM_PROMPT_QUERY_ONLY = """\
You are a fashion advisor.

You will be given:
1. A fashion query describing a situation or occasion
2. Four clothing options (A, B, C, D) described by text attributes only

Your task:
Select the single BEST option that best fits the query.

OUTPUT FORMAT — CRITICAL:
You MUST output EXACTLY one character: A, B, C, or D.
Do NOT output any explanation, reasoning, punctuation, or whitespace.
Do NOT write sentences. Do NOT write words.
If you write anything other than a single letter, your response is INVALID.
Your ENTIRE response must be one of: A  B  C  D
"""

SYSTEM_PROMPT_WITH_PROFILE = """\
You are a fashion advisor.

You may be given:
1. A user's fashion query (situation or preference)
2. User profile information
3. Four clothing options (A, B, C, D) described by text attributes only

Your task:
Select the single BEST option that best fits the available query and/or the user's preferences.

OUTPUT FORMAT — CRITICAL:
You MUST output EXACTLY one character: A, B, C, or D.
Do NOT output any explanation, reasoning, punctuation, or whitespace.
Do NOT write sentences. Do NOT write words.
If you write anything other than a single letter, your response is INVALID.
Your ENTIRE response must be one of: A  B  C  D
"""


TPO_SCORE = {"A": 1, "B": 1, "C": 0, "D": 0}
PROFILE_SCORE = {"A": 1, "B": 0, "C": 1, "D": 0}

INPUT_FORMATS = ["all", "all+query", "query", "narrative", "narrative+query"]
SUMMARY_GROUPS = [
    ["all", "all+query", "query"],
    ["narrative", "narrative+query", "query"],
]
NARRATIVE_KEYS = {
    "narrative_profile", "narrative", "profile_text", "description",
    "user_profile", "profile", "text",
}


# ── Profile formatting ──────────────────────────────────────────────────
def profile_to_narrative(profile):
    for key in [
        "narrative_profile", "narrative", "profile_text", "description",
        "user_profile", "profile", "text",
    ]:
        val = profile.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    meta = profile.get("metadata", {})
    for key in [
        "narrative_profile", "narrative", "profile_text", "description",
        "user_profile", "profile", "text",
    ]:
        val = meta.get(key) if isinstance(meta, dict) else None
        if isinstance(val, str) and val.strip():
            return val.strip()

    return ""


def _drop_narrative_fields(value):
    if isinstance(value, dict):
        return {
            k: _drop_narrative_fields(v)
            for k, v in value.items()
            if k not in NARRATIVE_KEYS
        }
    if isinstance(value, list):
        return [_drop_narrative_fields(v) for v in value]
    return value


def profile_to_all_kv_text(profile):
    """All profile in key-value form, excluding narrative text fields."""
    parts = []
    ordered_keys = [
        "user_id", "domain", "preference_archetype", "variant_index",
        "structured_attributes", "likes_keywords", "dislikes_keywords",
        "metadata",
    ]

    for key in ordered_keys:
        if key in profile and key not in NARRATIVE_KEYS:
            val = _drop_narrative_fields(profile[key])
            if val not in ({}, [], None, ""):
                parts.append(f"{key}: {json.dumps(val, ensure_ascii=False)}")

    for key, val in profile.items():
        if key in ordered_keys or key in NARRATIVE_KEYS:
            continue
        val = _drop_narrative_fields(val)
        if val not in ({}, [], None, ""):
            parts.append(f"{key}: {json.dumps(val, ensure_ascii=False)}")

    return "\n".join(parts)


# ── Option rendering ────────────────────────────────────────────────────
def option_to_text(opt):
    text = opt.get("search_query")
    if text:
        return text

    attrs = opt.get("attributes", {})
    parts = []
    if attrs.get("color"):
        parts.append(attrs["color"])
    if attrs.get("pattern") and attrs["pattern"] != "solid":
        parts.append(attrs["pattern"].replace("_", " "))
    if attrs.get("garment_category"):
        parts.append(attrs["garment_category"].replace("_", " "))
    return " ".join(parts).strip() or "unknown item"


# ── Prompt building ─────────────────────────────────────────────────────
def format_uses_query(input_format):
    return input_format in {"query", "all+query", "narrative+query"}


def format_uses_profile(input_format):
    return input_format in {"all", "all+query", "narrative", "narrative+query"}


def build_prompt(query, profile, shuffled_options, input_format="narrative+query"):
    option_lines = []
    for label, opt in shuffled_options:
        option_lines.append(f"  {label}. {option_to_text(opt)}")
    option_block = "\n".join(option_lines)

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
        task_line = "Select the single BEST option for the query."
    elif format_uses_query(input_format):
        task_line = "Select the single BEST option for both the query and the user's preferences."
    else:
        task_line = "Select the single BEST option for the user's preferences."

    body = "\n\n".join(f"=== {title} ===\n{text}" for title, text in sections)
    return f"""\
{body}

=== INSTRUCTION ===
{task_line}
Respond with ONE letter only: A, B, C, or D.
Do NOT write any explanation or reasoning.
Do NOT write anything before or after the letter.
Your complete response must be a single character.

Answer:"""


def system_prompt_for(input_format):
    if input_format == "query":
        return SYSTEM_PROMPT_QUERY_ONLY
    return SYSTEM_PROMPT_WITH_PROFILE


def stage_name_for(input_format):
    if input_format == "query":
        return "text_only_no_profile_eval"
    return "text_only_eval"


# ── Answer parsing / model resolution ───────────────────────────────────
def parse_answer(response: str):
    response = (response or "").strip()
    if re.fullmatch(r"[ABCDabcd]", response):
        return response.upper()
    first_line = response.splitlines()[0].strip() if response else ""
    if re.fullmatch(r"[ABCDabcd]", first_line):
        return first_line.upper()
    return None


def provider_alias_for_model(model_arg):
    """Return provider alias. Raw model names are mapped onto the vllm endpoint in-memory."""
    if not model_arg:
        return None
    if model_arg in PROVIDER_ENDPOINTS:
        return model_arg

    alias = "__text_exp_model__"
    base = dict(PROVIDER_ENDPOINTS.get("vllm", {}))
    if not base:
        raise ValueError("Cannot resolve raw --model because PROVIDER_ENDPOINTS['vllm'] is missing.")
    base["model_name"] = model_arg
    PROVIDER_ENDPOINTS[alias] = base
    return alias


def resolve_model_name(model_arg, input_format):
    if model_arg:
        if model_arg in PROVIDER_ENDPOINTS:
            return PROVIDER_ENDPOINTS[model_arg].get("model_name", model_arg)
        return model_arg
    stage = stage_name_for(input_format)
    provider_alias = PROVIDERS.get(stage, {}).get("provider", "vllm")
    endpoint = PROVIDER_ENDPOINTS.get(provider_alias, {})
    return endpoint.get("model_name", provider_alias)


def _call_one(prompt, system_prompt, stage_name, provider_alias):
    return call_llm(
        prompt=prompt,
        stage=stage_name,
        system=system_prompt,
        provider_override=provider_alias,
    )


# ── Job preparation / evaluation ────────────────────────────────────────
def prepare_jobs(plans, queries_map, profiles_map, input_format, seed=42, limit=0):
    rng = random.Random(seed)
    if limit > 0:
        plans = plans[:limit]

    jobs = []
    for plan in plans:
        qid = plan["query_id"]
        uid = plan["user_id"]
        query = queries_map.get(qid, {})
        profile = profiles_map.get(uid, {})
        if format_uses_query(input_format) and not query:
            continue

        option_items = list(plan["options"].items())
        rng.shuffle(option_items)

        display_labels = ["A", "B", "C", "D"]
        shuffled = list(zip(display_labels, [opt for _, opt in option_items]))
        display_to_original = {
            display: original for display, (original, _) in zip(display_labels, option_items)
        }
        correct_display = next(
            (display for display, original in display_to_original.items() if original == "A"), None
        )
        prompt = build_prompt(query, profile, shuffled, input_format=input_format)
        jobs.append((plan, query, prompt, display_to_original, correct_display))
    return jobs


def evaluate(plans, queries_map, profiles_map,
             input_format="narrative+query", limit=0, model=None,
             seed=42, verbose=False, concurrency=1):
    provider_alias = provider_alias_for_model(model)
    system_prompt = system_prompt_for(input_format)
    stage_name = stage_name_for(input_format)
    jobs = prepare_jobs(plans, queries_map, profiles_map, input_format, seed=seed, limit=limit)

    results = []
    strict_correct = tpo_correct = profile_correct = total = 0
    breakdown = defaultdict(lambda: {
        "strict_correct": 0, "tpo_correct": 0, "profile_correct": 0, "total": 0,
    })

    def process(idx, job):
        plan, query, prompt, display_to_original, correct_display = job
        qid = plan["query_id"]
        uid = plan["user_id"]
        axis = plan.get("active_axis", "unknown")
        qtype = (query or {}).get("query_type", "unknown")

        response = predicted = predicted_original = None
        try:
            response = _call_one(prompt, system_prompt, stage_name, provider_alias)
            predicted = parse_answer(response)
            predicted_original = display_to_original.get(predicted)
        except Exception as e:
            print(f"  [ERROR] {qid}: {e}")

        strict_hit = int(predicted_original == "A") if predicted_original else 0
        tpo_hit = TPO_SCORE.get(predicted_original, 0) if predicted_original else 0
        profile_hit = PROFILE_SCORE.get(predicted_original, 0) if predicted_original else 0

        rec = {
            "_idx": idx,
            "query_id": qid,
            "user_id": uid,
            "active_axis": axis,
            "query_type": qtype,
            "scenario_id": plan.get("scenario_id"),
            "input_format": input_format,
            "correct_display": correct_display,
            "predicted": predicted,
            "predicted_original": predicted_original,
            "strict_correct": bool(strict_hit),
            "tpo_score": tpo_hit,
            "profile_score": profile_hit,
            "raw_response": response,
        }
        return rec, strict_hit, tpo_hit, profile_hit, axis, qtype

    if concurrency > 1:
        with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {ex.submit(process, i, job): i for i, job in enumerate(jobs)}
            done_count = 0
            for fut in cf.as_completed(futures):
                rec, sh, th, ph, axis, qtype = fut.result()
                results.append(rec)
                strict_correct += sh; tpo_correct += th; profile_correct += ph; total += 1
                for key in [f"axis:{axis}", f"qtype:{qtype}"]:
                    breakdown[key]["total"] += 1
                    breakdown[key]["strict_correct"] += sh
                    breakdown[key]["tpo_correct"] += th
                    breakdown[key]["profile_correct"] += ph
                done_count += 1
                if verbose or done_count % 500 == 0:
                    print_progress(done_count, len(jobs), rec, sh, axis, qtype)
    else:
        for i, job in enumerate(jobs):
            rec, sh, th, ph, axis, qtype = process(i, job)
            results.append(rec)
            strict_correct += sh; tpo_correct += th; profile_correct += ph; total += 1
            for key in [f"axis:{axis}", f"qtype:{qtype}"]:
                breakdown[key]["total"] += 1
                breakdown[key]["strict_correct"] += sh
                breakdown[key]["tpo_correct"] += th
                breakdown[key]["profile_correct"] += ph
            if verbose or (i + 1) % 10 == 0:
                print_progress(i + 1, len(jobs), rec, sh, axis, qtype)

    results.sort(key=lambda x: x["_idx"])
    for rec in results:
        rec.pop("_idx")
    return results, strict_correct, tpo_correct, profile_correct, total, breakdown


def print_progress(done, total, rec, strict_hit, axis, qtype):
    status = "✓" if strict_hit else "✗"
    print(f"  [{done:3d}/{total}] {status} pred={rec['predicted']} "
          f"orig={rec['predicted_original']} ans={rec['correct_display']} | "
          f"axis={axis} qtype={qtype}")


# ── Reporting ───────────────────────────────────────────────────────────
def print_report(strict_correct, tpo_correct, profile_correct, total, breakdown,
                 input_format=None, model_name=None):
    strict_acc = strict_correct / total * 100 if total else 0
    tpo_acc = tpo_correct / total * 100 if total else 0
    profile_acc = profile_correct / total * 100 if total else 0

    print("\n" + "=" * 60)
    if input_format:
        print(f"  INPUT FORMAT:      {input_format}")
    if model_name:
        print(f"  MODEL:             {model_name}")
    print(f"  STRICT ACCURACY:   {strict_correct}/{total} = {strict_acc:.1f}%")
    print(f"  TPO ACCURACY:      {tpo_correct}/{total} = {tpo_acc:.1f}%")
    print(f"  PROFILE ACCURACY:  {profile_correct}/{total} = {profile_acc:.1f}%")
    print(f"  Random baseline:   25.0% strict")
    print("=" * 60)

    print("\n  ── By active_axis ──")
    for key in sorted(k for k in breakdown if k.startswith("axis:")):
        d = breakdown[key]
        n = d["total"]
        s = d["strict_correct"] / n * 100 if n else 0
        t = d["tpo_correct"] / n * 100 if n else 0
        p = d["profile_correct"] / n * 100 if n else 0
        print(f"  {key:18s} strict={s:5.1f}%  tpo={t:5.1f}%  profile={p:5.1f}%  (n={n})")

    print("\n  ── By query_type ──")
    for key in sorted(k for k in breakdown if k.startswith("qtype:")):
        d = breakdown[key]
        n = d["total"]
        s = d["strict_correct"] / n * 100 if n else 0
        t = d["tpo_correct"] / n * 100 if n else 0
        p = d["profile_correct"] / n * 100 if n else 0
        print(f"  {key:18s} strict={s:5.1f}%  tpo={t:5.1f}%  profile={p:5.1f}%  (n={n})")


def _summary_row(summary):
    total = summary["total"]
    strict = summary["strict_correct"] / total * 100 if total else 0
    tpo = summary["tpo_correct"] / total * 100 if total else 0
    prof = summary["profile_correct"] / total * 100 if total else 0
    return strict, tpo, prof


def print_grouped_summary(summaries):
    print("\n" + "=" * 78)
    print("  ══ GROUPED SUMMARY: all / all+query / query /// narrative / narrative+query / query ══")
    print(f"  {'format':<18} {'model':<32} {'strict':>7} {'tpo':>7} {'profile':>9}")
    print("  " + "-" * 74)
    for gi, group in enumerate(SUMMARY_GROUPS):
        if gi > 0:
            print("  " + "/" * 74)
        for fmt in group:
            s = summaries.get(fmt)
            if not s:
                continue
            model = (s["model_name"] or "?")[:31]
            strict, tpo, prof = _summary_row(s)
            print(f"  {fmt:<18} {model:<32} {strict:6.1f}%  {tpo:6.1f}%  {prof:7.1f}%")
    print("=" * 78)


# ── See prompts ─────────────────────────────────────────────────────────
def print_prompt_preview(plans, queries_map, profiles_map, seed=42):
    print("\n" + "=" * 78)
    print("  PROMPT PREVIEW: all / all+query / query /// narrative / narrative+query / query")
    print("=" * 78)

    preview_order = ["all", "all+query", "query", "narrative", "narrative+query", "query"]
    for i, fmt in enumerate(preview_order):
        jobs = prepare_jobs(plans, queries_map, profiles_map, fmt, seed=seed, limit=1)
        if not jobs:
            print(f"\n[WARN] no preview job for input_format={fmt}")
            continue
        plan, query, prompt, _, _ = jobs[0]
        print("\n" + "─" * 78)
        print(f"INPUT FORMAT: {fmt}   query_id={plan.get('query_id')}   user_id={plan.get('user_id')}")
        print("─" * 78)
        print("[SYSTEM]")
        print(system_prompt_for(fmt).rstrip())
        print("\n[USER]")
        print(prompt)
        if i == 2:
            print("\n" + "/" * 78)


# ── Main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", type=Path,
                        default=OPTIONS_DIR / "option_plans.jsonl")
    parser.add_argument("--queries", type=Path,
                        default=QUERIES_DIR / "queries.jsonl")
    parser.add_argument("--profiles", type=Path,
                        default=PROFILES_DIR / "profiles.jsonl")
    parser.add_argument("--output", type=Path,
                        default=OPTIONS_DIR / "text_eval_results.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="0 = all")
    parser.add_argument("--model", type=str, default=None,
                        help="provider alias from config, e.g. vllm/gpt5_mini, or raw model name on the vllm endpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--input-format", type=str, default=None, choices=INPUT_FORMATS,
                        help="생략 시 all / all+query / query / narrative / narrative+query 실행")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="병렬 요청 수. vllm은 32, OpenAI API는 1~8 권장")
    parser.add_argument("--see", action="store_true",
                        help="실험을 실행하지 않고 실제 model input prompt만 출력")
    args = parser.parse_args()

    log_step("Text Input-Format Eval")

    plans = load_jsonl(args.plans)
    queries = load_jsonl(args.queries)
    profiles = load_jsonl(args.profiles)

    queries_map = {q["query_id"]: q for q in queries}
    profiles_map = {p["user_id"]: p for p in profiles}

    print(f"  Plans: {len(plans)}, Queries: {len(queries)}, Profiles: {len(profiles)}")
    print(f"  Concurrency: {args.concurrency}")
    if args.limit:
        print(f"  Limit: {args.limit}")

    if args.see:
        print_prompt_preview(plans, queries_map, profiles_map, seed=args.seed)
        return

    formats_to_run = [args.input_format] if args.input_format else INPUT_FORMATS
    run_all = args.input_format is None
    summaries = {}

    for fmt in formats_to_run:
        model_name = resolve_model_name(args.model, fmt)
        print(f"\n{'─' * 60}")
        print(f"  ▶ input-format = {fmt}  |  model = {model_name}")
        print(f"{'─' * 60}")

        if run_all or args.output == OPTIONS_DIR / "text_eval_results.jsonl":
            suffix = fmt.replace("+", "_plus_")
            out_path = OPTIONS_DIR / f"text_eval_results_{suffix}.jsonl"
        else:
            out_path = args.output

        results, strict_correct, tpo_correct, profile_correct, total, breakdown = evaluate(
            plans, queries_map, profiles_map,
            input_format=fmt,
            limit=args.limit,
            model=args.model,
            seed=args.seed,
            verbose=args.verbose,
            concurrency=args.concurrency,
        )

        save_jsonl(results, out_path)
        print(f"\n  Saved {len(results)} result records → {out_path}")
        print_report(strict_correct, tpo_correct, profile_correct, total, breakdown,
                     input_format=fmt, model_name=model_name)

        summaries[fmt] = {
            "input_format": fmt,
            "model_name": model_name,
            "strict_correct": strict_correct,
            "tpo_correct": tpo_correct,
            "profile_correct": profile_correct,
            "total": total,
        }

    if run_all:
        print_grouped_summary(summaries)


if __name__ == "__main__":
    main()
