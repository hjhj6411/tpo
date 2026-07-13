#!/usr/bin/env python3
"""
Text experiment variants preserving scripts/text_only_eval.py format.

Input formats:
  - all             : compact key-value profile, no query
  - all+query       : compact key-value profile + query
  - query           : query only, no profile
  - narrative       : narrative profile only, no query
  - narrative+query : narrative profile + query

Default report grouping:
  all / all+query / query /// narrative / narrative+query / query

Examples:
  python -m text_exp.text_eval --model vllm --concurrency 32
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
"""

SYSTEM_PROMPT_WITH_PROFILE = """\
You are a fashion advisor.

You may be given:
1. A user's fashion query (situation or preference)
2. User profile information
3. Four clothing options (A, B, C, D) described by text attributes only

Your task:
Select the single BEST option that best fits the available query and/or the user's preferences.
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


def _join_values(values):
    if not values:
        return "none"
    return ", ".join(str(v).replace("_", " ") for v in values)


def _axis_preferences(profile, axis):
    attrs = profile.get("structured_attributes") or {}
    axis_info = attrs.get(axis) if isinstance(attrs, dict) else None
    if not isinstance(axis_info, dict):
        return [], []
    likes = axis_info.get("likes") or []
    dislikes = axis_info.get("dislikes") or []
    return list(likes), list(dislikes)


def profile_to_all_kv_text(profile):
    """Compact all-profile K-V text without narrative or duplicated keyword lists."""
    lines = []
    archetype = profile.get("preference_archetype")
    if archetype:
        lines.append(f"style: {str(archetype).replace('_', ' ')}")

    for axis, label in [
        ("garment_category", "garment"),
        ("color", "color"),
        ("pattern", "pattern"),
    ]:
        likes, dislikes = _axis_preferences(profile, axis)
        lines.append(f"likes.{label}: {_join_values(likes)}")
        lines.append(f"dislikes.{label}: {_join_values(dislikes)}")

    return "\n".join(lines)


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
    if not response:
        return None

    if "</think>" in response:
        response = response.split("</think>", 1)[1].strip()

    # Ideal case: exactly one choice.
    if re.fullmatch(r"[ABCDabcd]", response):
        return response.upper()

    first_line = response.splitlines()[0].strip() if response else ""
    if re.fullmatch(r"[ABCDabcd][\s\.)\]:-]*", first_line):
        return first_line[0].upper()

    # Common short completions when max_tokens is small: "Answer: C", "The answer is B".
    m = re.search(r"(?i)(?:answer|option|choice|select(?:ion)?|final)\s*[:\-]?\s*([ABCD])\b", response)
    if m:
        return m.group(1).upper()

    # Last-resort fallback for short outputs only. Avoid parsing arbitrary long reasoning text.
    if len(response) <= 80:
        m = re.search(r"\b([ABCDabcd])\b", response)
        if m:
            return m.group(1).upper()

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
    ok = "✓" if strict_hit else "✗"
    print(f"  [{done}/{total}] {ok} pred={rec['predicted']} orig={rec['predicted_original']} "
          f"ans={rec['correct_display']} | axis={axis} qtype={qtype}")


# ── Reporting ───────────────────────────────────────────────────────────
def pct(num, den):
    return num / den * 100 if den else 0.0


def summarize_breakdown(breakdown):
    return {
        key: {
            "strict": pct(v["strict_correct"], v["total"]),
            "tpo": pct(v["tpo_correct"], v["total"]),
            "profile": pct(v["profile_correct"], v["total"]),
            "n": v["total"],
        }
        for key, v in sorted(breakdown.items())
    }


def print_table(rows, title):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)
    print(f"  {'format':<18} {'model':<32} {'strict':>8} {'tpo':>7} {'profile':>9} {'n':>8}")
    print("  " + "-" * 72)
    for r in rows:
        print(f"  {r['input_format']:<18} {r['model'][:32]:<32} "
              f"{r['strict_acc']:7.1f}% {r['tpo_acc']:6.1f}% {r['profile_acc']:8.1f}% {r['n']:8d}")


def print_grouped_summary(rows, title):
    by_fmt = {r["input_format"]: r for r in rows}
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)
    print(f"  {'format':<18} {'model':<32} {'strict':>8} {'tpo':>7} {'profile':>9}")
    print("  " + "-" * 72)
    for gi, group in enumerate(SUMMARY_GROUPS):
        if gi > 0:
            print("  " + "/" * 72)
        for fmt in group:
            r = by_fmt.get(fmt)
            if not r:
                continue
            print(f"  {fmt:<18} {r['model'][:32]:<32} "
                  f"{r['strict_acc']:7.1f}% {r['tpo_acc']:6.1f}% {r['profile_acc']:8.1f}%")
    print("=" * 78)


def print_breakdown_comparison(rows, prefix, title):
    by_fmt = {r["input_format"]: r for r in rows}
    keys = sorted({
        key for r in rows for key in r.get("breakdown", {})
        if key.startswith(prefix)
    })
    if not keys:
        return

    print("\n" + "=" * 100)
    print(f"  ══ {title} ══")
    print("  cell = strict / tpo / profile")
    print("=" * 100)
    for key in keys:
        label = key.split(":", 1)[1]
        print(f"\n  ── {prefix}{label} ──")
        print(f"  {'format':<18} {'strict':>8} {'tpo':>7} {'profile':>9} {'n':>10}")
        print("  " + "-" * 52)
        for gi, group in enumerate(SUMMARY_GROUPS):
            if gi > 0:
                print("  " + "/" * 52)
            for fmt in group:
                r = by_fmt.get(fmt)
                if not r:
                    continue
                cell = r.get("breakdown", {}).get(key)
                if not cell:
                    continue
                print(f"  {fmt:<18} {cell['strict']:7.1f}% {cell['tpo']:6.1f}% "
                      f"{cell['profile']:8.1f}% {cell['n']:10d}")


# ── Preview ─────────────────────────────────────────────────────────────
def print_preview(plans, queries_map, profiles_map, input_format, seed=42, limit=1):
    jobs = prepare_jobs(plans, queries_map, profiles_map, input_format, seed=seed, limit=limit)
    if not jobs:
        print("No preview job.")
        return
    plan, _query, prompt, display_to_original, correct_display = jobs[0]
    print("=" * 78)
    print(f"PREVIEW input_format={input_format} query_id={plan['query_id']} user_id={plan['user_id']}")
    print(f"display_to_original={display_to_original} correct_display={correct_display}")
    print("=" * 78)
    print(prompt)


# ── Main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", type=Path, default=OPTIONS_DIR / "option_plans.jsonl")
    parser.add_argument("--queries", type=Path, default=QUERIES_DIR / "queries.jsonl")
    parser.add_argument("--profiles", type=Path, default=PROFILES_DIR / "profiles.jsonl")
    parser.add_argument("--output-dir", type=Path, default=OPTIONS_DIR / "text_eval")
    parser.add_argument("--input-format", choices=INPUT_FORMATS, default=None,
                        help="If omitted, run all formats.")
    parser.add_argument("--model", type=str, default=None,
                        help="Provider alias such as vllm/gpt5_mini, or raw model name on the vllm endpoint.")
    parser.add_argument("--limit", type=int, default=0, help="0 = all plans")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--see", action="store_true", help="Print one prompt preview and exit.")
    args = parser.parse_args()

    log_step("Text Input-Format Eval")
    plans = load_jsonl(args.plans)
    queries = load_jsonl(args.queries)
    profiles = load_jsonl(args.profiles)
    queries_map = {q["query_id"]: q for q in queries}
    profiles_map = {p["user_id"]: p for p in profiles}
    input_formats = INPUT_FORMATS if args.input_format is None else [args.input_format]
    model_name = resolve_model_name(args.model, input_formats[0])

    print(f"  Plans: {len(plans)}, Queries: {len(queries)}, Profiles: {len(profiles)}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Limit: {args.limit or 'all'}")

    if args.see:
        for fmt in input_formats:
            print_preview(plans, queries_map, profiles_map, fmt, seed=args.seed, limit=args.limit or 1)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for fmt in input_formats:
        print(f"\n{'─' * 78}")
        print(f"  ▶ input-format = {fmt}  |  model = {model_name}")
        print(f"{'─' * 78}")
        results, strict, tpo, profile, total, breakdown = evaluate(
            plans, queries_map, profiles_map,
            input_format=fmt,
            limit=args.limit,
            model=args.model,
            seed=args.seed,
            verbose=args.verbose,
            concurrency=args.concurrency,
        )
        safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
        out_path = args.output_dir / f"{fmt.replace('+', '_plus_')}__{safe_model}.jsonl"
        save_jsonl(results, out_path)
        row = {
            "input_format": fmt,
            "model": model_name,
            "strict_acc": pct(strict, total),
            "tpo_acc": pct(tpo, total),
            "profile_acc": pct(profile, total),
            "n": total,
            "out": str(out_path),
            "breakdown": summarize_breakdown(breakdown),
        }
        rows.append(row)
        print_table([row], title=f"RESULT: {fmt}")
        print(f"  Saved → {out_path}")

    print_table(rows, title="FINAL SUMMARY")
    print_grouped_summary(rows, title="══ GROUPED SUMMARY: all / all+query / query /// narrative / narrative+query / query ══")
    print_breakdown_comparison(rows, "axis:", "ACTIVE_AXIS COMPARISON")
    print_breakdown_comparison(rows, "qtype:", "QUERY_TYPE COMPARISON")

    summary_path = args.output_dir / f"summary__{re.sub(r'[^A-Za-z0-9_.-]+', '_', model_name)}.json"
    summary_rows = [{k: v for k, v in r.items() if k != "breakdown"} | {"breakdown": r["breakdown"]} for r in rows]
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)
    print(f"\n  Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
