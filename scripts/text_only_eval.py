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


실행:
  python -m scripts.text_only_eval --profile-mode narrative
  python -m scripts.text_only_eval --profile-mode no
  python -m scripts.text_only_eval --profile-mode all --limit 50 --provider gpt5_mini
"""


import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))


from src.utils import call_llm, load_jsonl, save_jsonl, log_step
from configs.config import OPTIONS_DIR, QUERIES_DIR, PROFILES_DIR



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
        "narrative_profile",
        "narrative",
        "profile_text",
        "description",
        "user_profile",
        "profile",
        "text",
    ]:
        val = profile.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()


    meta = profile.get("metadata", {})
    for key in [
        "narrative_profile",
        "narrative",
        "profile_text",
        "description",
        "user_profile",
        "profile",
        "text",
    ]:
        val = meta.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()


    return ""



def profile_to_all_text(profile):
    parts = []


    ordered_keys = [
        "user_id",
        "domain",
        "preference_archetype",
        "variant_index",
        "structured_attributes",
        "likes_keywords",
        "dislikes_keywords",
        "narrative_profile",
    ]


    for key in ordered_keys:
        if key in profile:
            parts.append(f"{key}: {json.dumps(profile[key], ensure_ascii=False)}")


    for key, val in profile.items():
        if key not in ordered_keys:
            parts.append(f"{key}: {json.dumps(val, ensure_ascii=False)}")


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
    response = (response or "").strip()
    for ch in response:
        if ch in "ABCDabcd":
            return ch.upper()
    return None



def evaluate(plans, queries_map, profiles_map,
             limit=0, provider=None, seed=42, verbose=False,
             profile_mode="narrative"):
    rng = random.Random(seed)


    if limit > 0:
        plans = plans[:limit]


    results = []
    strict_correct = 0
    tpo_correct = 0
    profile_correct = 0
    total = 0


    breakdown = defaultdict(lambda: {
        "strict_correct": 0,
        "tpo_correct": 0,
        "profile_correct": 0,
        "total": 0
    })


    if profile_mode == "no":
        system_prompt = SYSTEM_PROMPT_NO
        stage_name = "text_only_no_profile_eval"
    else:
        system_prompt = SYSTEM_PROMPT_WITH_PROFILE
        stage_name = "text_only_eval"


    for i, plan in enumerate(plans):
        qid = plan["query_id"]
        uid = plan["user_id"]


        query = queries_map.get(qid)
        profile = profiles_map.get(uid, {})


        if query is None:
            continue


        option_items = list(plan["options"].items())   # [("A", {...}), ...]
        rng.shuffle(option_items)


        display_labels = ["A", "B", "C", "D"]
        shuffled = list(zip(display_labels, [opt for _, opt in option_items]))


        display_to_original = {
            disp_label: orig_label
            for disp_label, (orig_label, _) in zip(display_labels, option_items)
        }


        correct_display = None
        for disp_label, orig_label in display_to_original.items():
            if orig_label == "A":
                correct_display = disp_label
                break


        response = None
        predicted = None
        predicted_original = None


        prompt = build_prompt(query, profile, shuffled, profile_mode=profile_mode)


        try:
            response = call_llm(
                prompt=prompt,
                stage=stage_name,
                system=system_prompt,
                provider_override=provider,
            )
            predicted = parse_answer(response)
            predicted_original = display_to_original.get(predicted)
        except Exception as e:
            print(f"  [ERROR] {qid}: {e}")


        strict_hit = int(predicted_original == "A") if predicted_original else 0
        tpo_hit = TPO_SCORE.get(predicted_original, 0) if predicted_original else 0
        profile_hit = PROFILE_SCORE.get(predicted_original, 0) if predicted_original else 0


        strict_correct += strict_hit
        tpo_correct += tpo_hit
        profile_correct += profile_hit
        total += 1


        axis = plan.get("active_axis", "unknown")
        qtype = query.get("query_type", "unknown")


        for key in [f"axis:{axis}", f"qtype:{qtype}"]:
            breakdown[key]["total"] += 1
            breakdown[key]["strict_correct"] += strict_hit
            breakdown[key]["tpo_correct"] += tpo_hit
            breakdown[key]["profile_correct"] += profile_hit


        result_rec = {
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
        results.append(result_rec)


        if verbose or (i + 1) % 10 == 0:
            status = "✓" if strict_hit else "✗"
            print(
                f"  [{i+1:3d}/{len(plans)}] {status} "
                f"pred={predicted} orig={predicted_original} ans={correct_display} | "
                f"axis={axis} qtype={qtype}"
            )


    return results, strict_correct, tpo_correct, profile_correct, total, breakdown



def print_report(strict_correct, tpo_correct, profile_correct, total, breakdown):
    strict_acc = strict_correct / total * 100 if total > 0 else 0
    tpo_acc = tpo_correct / total * 100 if total > 0 else 0
    profile_acc = profile_correct / total * 100 if total > 0 else 0


    print("\n" + "=" * 50)
    print(f"  STRICT ACCURACY:   {strict_correct}/{total} = {strict_acc:.1f}%")
    print(f"  TPO ACCURACY:      {tpo_correct}/{total} = {tpo_acc:.1f}%")
    print(f"  PROFILE ACCURACY:  {profile_correct}/{total} = {profile_acc:.1f}%")
    print(f"  Random baseline:   25.0% strict")
    print("=" * 50)


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
    parser.add_argument("--profile-mode", type=str, default="narrative",
                        choices=["no", "narrative", "all"])
    args = parser.parse_args()


    if args.output == OPTIONS_DIR / "text_only_results.jsonl":
        args.output = OPTIONS_DIR / f"text_only_results_{args.profile_mode}.jsonl"


    log_step("Text-Only LLM Baseline Eval")


    plans = load_jsonl(args.plans)
    queries = load_jsonl(args.queries)
    profiles = load_jsonl(args.profiles)


    queries_map = {q["query_id"]: q for q in queries}
    profiles_map = {p["user_id"]: p for p in profiles}


    print(f"  Plans: {len(plans)}, Queries: {len(queries)}, Profiles: {len(profiles)}")
    print(f"  Profile mode: {args.profile_mode}")
    if args.limit:
        print(f"  Limit: {args.limit}")


    results, strict_correct, tpo_correct, profile_correct, total, breakdown = evaluate(
        plans, queries_map, profiles_map,
        limit=args.limit,
        provider=args.provider,
        seed=args.seed,
        verbose=args.verbose,
        profile_mode=args.profile_mode,
    )


    save_jsonl(results, args.output)
    print(f"\n  Saved {len(results)} result records → {args.output}")


    print_report(strict_correct, tpo_correct, profile_correct, total, breakdown)



if __name__ == "__main__":
    main()
