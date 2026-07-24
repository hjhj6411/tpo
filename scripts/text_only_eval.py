#!/usr/bin/env python3
"""
Unified Text-Only LLM Baseline Evaluation
----------------------------------------
세 가지 모드를 하나로 통합:
- no         : profile 없이 query + options만
- narrative  : narrative profile만
- all        : profile.jsonl의 주요 요소를 모두 노출

동시에 아래 3개 점수를 모두 계산:
- Strict accuracy   : 원래 정답 A를 맞췄는가
- TPO accuracy      : TPO 축만 맞췄는가
- Profile accuracy  : preference 축만 맞췄는가

--concurrency 32
실행:
  # 단일 모드
  python -m scripts.text_only_eval --profile-mode narrative --concurrency 32
  python -m scripts.text_only_eval --profile-mode no
  python -m scripts.text_only_eval --profile-mode all --limit 50 --provider gpt5_mini

  # --profile-mode 생략 → no / narrative / all 3개 모두 순차 실행
  python -m scripts.text_only_eval --concurrency 32
  python -m scripts.text_only_eval --provider gpt5_mini --limit 50
"""

import argparse
import json
import random
import re
import sys
import concurrent.futures as cf
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import call_llm, load_jsonl, save_jsonl, log_step
from configs.config import OPTIONS_DIR, QUERIES_DIR, PROFILES_DIR, PROVIDERS, PROVIDER_ENDPOINTS


SYSTEM_PROMPT_NO = """\
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
Select the single BEST option that best fits both the query and the user's preferences.

OUTPUT FORMAT — CRITICAL:
You MUST output EXACTLY one character: A, B, C, or D.
Do NOT output any explanation, reasoning, punctuation, or whitespace.
Do NOT write sentences. Do NOT write words.
If you write anything other than a single letter, your response is INVALID.
Your ENTIRE response must be one of: A  B  C  D
"""


# ── Scoring semantics from original labels ──────────────────────────────
TPO_SCORE = {
    "A": 1, "B": 1, "C": 0, "D": 0
}

PROFILE_SCORE = {
    "A": 1, "B": 0, "C": 1, "D": 0
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
        val = meta.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    return ""


def profile_to_all_text(profile):
    parts = []

    # WHITELIST: only the user definition reaches the evaluated model.
    # Generation-internal metadata (style pool / archetype, variant index)
    # is a summary shortcut and must never leak into the prompt.
    ordered_keys = [
        "user_id", "domain",
        "structured_attributes", "likes_keywords", "dislikes_keywords",
        "narrative_profile",
    ]

    for key in ordered_keys:
        if key in profile:
            parts.append(f"{key}: {json.dumps(profile[key], ensure_ascii=False)}")

    return "\n".join(parts)


# ── Option rendering ────────────────────────────────────────────────────
def option_to_text(opt):
    text = opt.get("search_query")
    if text:
        return text

    a = opt.get("attributes", {})
    parts = []
    if a.get("color"):
        parts.append(a["color"])
    if a.get("pattern") and a["pattern"] != "solid":
        parts.append(a["pattern"].replace("_", " "))
    if a.get("garment_category"):
        parts.append(a["garment_category"].replace("_", " "))

    return " ".join(parts).strip() or "unknown item"


# ── Prompt ──────────────────────────────────────────────────────────────
def build_prompt(query, profile, shuffled_options, profile_mode="narrative"):
    option_lines = []
    for label, opt in shuffled_options:
        line = f"  {label}. {option_to_text(opt)}"
        option_lines.append(line)
    option_block = "\n".join(option_lines)

    qtext = query.get("query_text", "").strip()

    if profile_mode == "no":
        prompt = f"""\
=== QUERY ===
{qtext}

=== OPTIONS ===
{option_block}

=== INSTRUCTION ===
Respond with ONE letter only: A, B, C, or D.
Do NOT write any explanation or reasoning.
Do NOT write anything before or after the letter.
Your complete response must be a single character.

Answer:"""
        return prompt

    if profile_mode == "narrative":
        profile_text = profile_to_narrative(profile)
    elif profile_mode == "all":
        profile_text = profile_to_all_text(profile)
    else:
        raise ValueError(f"Unknown profile_mode: {profile_mode}")

    prompt = f"""\
=== QUERY ===
{qtext}

=== USER PROFILE ===
{profile_text}

=== OPTIONS ===
{option_block}

=== INSTRUCTION ===
Respond with ONE letter only: A, B, C, or D.
Do NOT write any explanation or reasoning.
Do NOT write anything before or after the letter.
Your complete response must be a single character.

Answer:"""
    return prompt


def parse_answer(response: str):
    """엄격 파서: 첫 줄이 정확히 한 글자인 경우만 인정."""
    response = (response or "").strip()

    # 전체가 한 글자
    if re.fullmatch(r"[ABCDabcd]", response):
        return response.upper()

    # 첫 줄이 한 글자 (thinking 토큰 제거 후에도 대비)
    first_line = response.splitlines()[0].strip() if response else ""
    if re.fullmatch(r"[ABCDabcd]", first_line):
        return first_line.upper()

    return None


def _call_one(prompt, system_prompt, stage_name, provider):
    """call_llm 원본 시그니처 그대로 호출. 추가 kwargs 없음."""
    return call_llm(
        prompt=prompt,
        stage=stage_name,
        system=system_prompt,
        provider_override=provider,
    )


# ── Model name resolution ───────────────────────────────────────────────
def resolve_model_name(provider_override, profile_mode):
    """현재 실험에 사용되는 모델명을 config에서 읽어 반환."""
    # provider_override가 명시된 경우 직접 조회
    if provider_override:
        ep = PROVIDER_ENDPOINTS.get(provider_override, {})
        return ep.get("model_name", provider_override)

    # stage 이름으로 PROVIDERS → PROVIDER_ENDPOINTS 체인 조회
    stage = "text_only_no_profile_eval" if profile_mode == "no" else "text_only_eval"
    provider_alias = PROVIDERS.get(stage, {}).get("provider", "vllm")
    ep = PROVIDER_ENDPOINTS.get(provider_alias, {})
    return ep.get("model_name", provider_alias)


def evaluate(plans, queries_map, profiles_map,
             limit=0, provider=None, seed=42, verbose=False,
             profile_mode="narrative", concurrency=1):
    rng = random.Random(seed)

    if limit > 0:
        plans = plans[:limit]

    if profile_mode == "no":
        system_prompt = SYSTEM_PROMPT_NO
        stage_name = "text_only_no_profile_eval"
    else:
        system_prompt = SYSTEM_PROMPT_WITH_PROFILE
        stage_name = "text_only_eval"

    # ── 모든 job 미리 준비 ─────────────────────────────────────────────
    jobs = []
    for plan in plans:
        qid = plan["query_id"]
        uid = plan["user_id"]
        query = queries_map.get(qid)
        profile = profiles_map.get(uid, {})
        if query is None:
            continue

        option_items = list(plan["options"].items())
        rng.shuffle(option_items)

        display_labels = ["A", "B", "C", "D"]
        shuffled = list(zip(display_labels, [opt for _, opt in option_items]))
        display_to_original = {
            d: o for d, (o, _) in zip(display_labels, option_items)
        }
        correct_display = next(
            (d for d, o in display_to_original.items() if o == "A"), None
        )
        prompt = build_prompt(query, profile, shuffled, profile_mode=profile_mode)

        jobs.append((plan, query, prompt, display_to_original, correct_display))

    results = []
    strict_correct = tpo_correct = profile_correct = total = 0
    breakdown = defaultdict(lambda: {
        "strict_correct": 0, "tpo_correct": 0, "profile_correct": 0, "total": 0
    })

    # ── 병렬(concurrency>1) 또는 순차(concurrency=1) 실행 ─────────────
    def process(idx, job):
        plan, query, prompt, display_to_original, correct_display = job
        qid = plan["query_id"]
        uid = plan["user_id"]
        axis = plan.get("active_axis", "unknown")
        qtype = query.get("query_type", "unknown")

        response = predicted = predicted_original = None
        try:
            response = _call_one(prompt, system_prompt, stage_name, provider)
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
            "profile_mode": profile_mode,
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
                    status = "✓" if sh else "✗"
                    print(f"  [{done_count:3d}/{len(jobs)}] {status} pred={rec['predicted']} "
                          f"orig={rec['predicted_original']} ans={rec['correct_display']} | "
                          f"axis={axis} qtype={qtype}")
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
                status = "✓" if sh else "✗"
                print(f"  [{i+1:3d}/{len(jobs)}] {status} pred={rec['predicted']} "
                      f"orig={rec['predicted_original']} ans={rec['correct_display']} | "
                      f"axis={axis} qtype={qtype}")

    # 원래 순서 복원 후 _idx 제거
    results.sort(key=lambda x: x["_idx"])
    for r in results:
        r.pop("_idx")

    return results, strict_correct, tpo_correct, profile_correct, total, breakdown


def print_report(strict_correct, tpo_correct, profile_correct, total, breakdown,
                 profile_mode=None, model_name=None):
    strict_acc = strict_correct / total * 100 if total > 0 else 0
    tpo_acc = tpo_correct / total * 100 if total > 0 else 0
    profile_acc = profile_correct / total * 100 if total > 0 else 0

    print("\n" + "=" * 60)
    if profile_mode:
        print(f"  PROFILE MODE:      {profile_mode}")
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


def print_combined_summary(all_summaries):
    """3개 모드 실행 후 비교 summary table 출력."""
    print("\n" + "=" * 60)
    print("  ══ COMBINED SUMMARY ══")
    print(f"  {'mode':<12} {'model':<32} {'strict':>7} {'tpo':>7} {'profile':>9}")
    print("  " + "-" * 56)
    for s in all_summaries:
        mode   = s["profile_mode"]
        model  = (s["model_name"] or "?")[:31]
        total  = s["total"]
        strict = s["strict_correct"] / total * 100 if total else 0
        tpo    = s["tpo_correct"]    / total * 100 if total else 0
        prof   = s["profile_correct"]/ total * 100 if total else 0
        print(f"  {mode:<12} {model:<32} {strict:6.1f}%  {tpo:6.1f}%  {prof:7.1f}%")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", type=Path,
                        default=OPTIONS_DIR / "option_plans.jsonl")
    parser.add_argument("--queries", type=Path,
                        default=QUERIES_DIR / "queries.jsonl")
    parser.add_argument("--profiles", type=Path,
                        default=PROFILES_DIR / "profiles.jsonl")
    parser.add_argument("--output", type=Path,
                        default=OPTIONS_DIR / "text_only_results.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="0 = all")
    parser.add_argument("--provider", type=str, default=None,
                        help="provider alias from config, e.g. vllm or gpt5_mini")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--profile-mode", type=str, default=None,
                        choices=["no", "narrative", "all"],
                        help="생략 시 no / narrative / all 3개 모두 순차 실행")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="병렬 요청 수. vllm은 32, gpt5_mini는 1~8 권장")
    args = parser.parse_args()

    log_step("Text-Only LLM Baseline Eval")

    plans    = load_jsonl(args.plans)
    queries  = load_jsonl(args.queries)
    profiles = load_jsonl(args.profiles)

    queries_map  = {q["query_id"]: q for q in queries}
    profiles_map = {p["user_id"]:  p for p in profiles}

    print(f"  Plans: {len(plans)}, Queries: {len(queries)}, Profiles: {len(profiles)}")
    print(f"  Concurrency: {args.concurrency}")
    if args.limit:
        print(f"  Limit: {args.limit}")

    # 실행할 모드 목록 결정
    modes_to_run = [args.profile_mode] if args.profile_mode else ["no", "narrative", "all"]
    run_all = (args.profile_mode is None)

    all_summaries = []

    for mode in modes_to_run:
        model_name = resolve_model_name(args.provider, mode)

        print(f"\n{'─' * 60}")
        print(f"  ▶ profile-mode = {mode}  |  model = {model_name}")
        print(f"{'─' * 60}")

        # output 경로: 단일 모드면 기존 로직, 전체 실행이면 항상 mode별 파일
        if run_all or args.output == OPTIONS_DIR / "text_only_results.jsonl":
            out_path = OPTIONS_DIR / f"text_only_results_{mode}.jsonl"
        else:
            out_path = args.output

        results, strict_correct, tpo_correct, profile_correct, total, breakdown = evaluate(
            plans, queries_map, profiles_map,
            limit=args.limit,
            provider=args.provider,
            seed=args.seed,
            verbose=args.verbose,
            profile_mode=mode,
            concurrency=args.concurrency,
        )

        save_jsonl(results, out_path)
        print(f"\n  Saved {len(results)} result records → {out_path}")

        print_report(strict_correct, tpo_correct, profile_correct, total, breakdown,
                     profile_mode=mode, model_name=model_name)

        all_summaries.append({
            "profile_mode":    mode,
            "model_name":      model_name,
            "strict_correct":  strict_correct,
            "tpo_correct":     tpo_correct,
            "profile_correct": profile_correct,
            "total":           total,
        })

    # 3개 모두 돌렸을 때만 combined summary 출력
    if run_all:
        print_combined_summary(all_summaries)


if __name__ == "__main__":
    main()
