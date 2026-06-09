#!/usr/bin/env python3
"""
multimodal_eval_fast.py — POD-Bench v2 image-MCQ eval (standalone, fast)
------------------------------------------------------------------------
multimodal_eval.py와 입출력 형태 동일. 차이는:
  - config 의존 제거: --model / --base-urls 로 vLLM 엔드포인트 직접 지정
  - 속도 개선:
      (1) 엔드포인트 라운드로빈  — 워커가 여러 vLLM에 분산 (가장 큰 이득)
      (2) base64 캐싱           — 같은 이미지 중복 인코딩 제거 (3모드 반복 등)
      (3) 이미지 리사이즈        — 긴 변 --img-size 로 축소 후 인코딩
      (4) max_tokens 축소       — 단일 글자 출력이라 기본 4

점수 3종(strict/tpo/profile), 셔플, 리포트, jsonl 출력은 원본과 동일.

vllm serve OpenGVLab/InternVL3_5-8B \
  --port 8002 --max-model-len 32768 \
  --tensor-parallel-size 4


python scripts/multimodal_eval.py \
  --plans data/options/option_plans.jsonl \
  --queries data/queries/queries.jsonl \
  --profiles data/profiles/profiles.jsonl \
  --image-root data/images_siglip_cov_q \
  --base-urls http://127.0.0.1:8002/v1 \
  --model Qwen/Qwen2.5-VL-72B-Instruct-AWQ \
  --concurrency 8 \
  --enable-thinking \
  --limit 1000 \
  --out-dir data/eval_multimodal_it \
  --enable-thinking --image-only-options

  # 단일 모드
  python multimodal_eval_fast.py --profile-mode no --limit 20 --verbose ...

  # 1-1 위치 선호 실험: 셔플 끄고 정답을 특정 위치로 고정 (--fix-correct A|B|C|D)
  python multimodal_eval_fast.py --profile-mode no --fix-correct A ...
"""
import argparse
import base64
import io
import json
import mimetypes
import random
import re
import sys
import threading
import concurrent.futures as cf
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

try:
    import openai
except ImportError:
    sys.exit("pip install openai  (this script calls an OpenAI-compatible vLLM endpoint)")

try:
    from PIL import Image
    _HAVE_PIL = True
except ImportError:
    _HAVE_PIL = False


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp"]

SYSTEM_PROMPT_NO = """You are a fashion advisor.

You will be given:
1. A fashion query describing a situation or occasion
2. Four clothing option images (A, B, C, D)

Your task:
Select the single BEST option image that best fits the query.

OUTPUT FORMAT — CRITICAL:
You MUST output EXACTLY one character: A, B, C, or D.
Do NOT output any explanation, reasoning, punctuation, or whitespace.
Do NOT write sentences. Do NOT write words.
If you write anything other than a single letter, your response is INVALID.
Your ENTIRE response must be one of: A  B  C  D
"""

SYSTEM_PROMPT_WITH_PROFILE = """You are a fashion advisor.

You will be given:
1. A fashion query (situation or occasion)
2. A user profile describing clothing preferences
3. Four clothing option images (A, B, C, D)

Your task:
Select the single BEST option image that best fits both the query and the user's preferences.

OUTPUT FORMAT — CRITICAL:
You MUST output EXACTLY one character: A, B, C, or D.
Do NOT output any explanation, reasoning, punctuation, or whitespace.
Do NOT write sentences. Do NOT write words.
If you write anything other than a single letter, your response is INVALID.
Your ENTIRE response must be one of: A  B  C  D
"""

TPO_SCORE     = {"A": 1, "B": 1, "C": 0, "D": 0}
PROFILE_SCORE = {"A": 1, "B": 0, "C": 1, "D": 0}


# ── io (standalone, no src.utils) ─────────────────────────────────────────
def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ── profile formatting ────────────────────────────────────────────────────
def profile_to_narrative(profile):
    for key in ["narrative_profile", "narrative", "profile_text", "description",
                "user_profile", "profile", "text"]:
        val = profile.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    meta = profile.get("metadata", {})
    for key in ["narrative_profile", "narrative", "profile_text", "description",
                "user_profile", "profile", "text"]:
        val = meta.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def profile_to_all_text(profile):
    ordered_keys = ["user_id", "domain", "preference_archetype", "variant_index",
                    "structured_attributes", "likes_keywords", "dislikes_keywords",
                    "narrative_profile"]
    parts = []
    for key in ordered_keys:
        if key in profile:
            parts.append(f"{key}: {json.dumps(profile[key], ensure_ascii=False)}")
    for key, val in profile.items():
        if key not in ordered_keys:
            parts.append(f"{key}: {json.dumps(val, ensure_ascii=False)}")
    return "\n".join(parts)


# ── image loading (cached + resized) ──────────────────────────────────────
_IMG_SIZE = 768          # set from args in main()
_b64_lock = threading.Lock()


@lru_cache(maxsize=8192)
def _encode_b64(path_str: str):
    """Read, optionally downscale (long edge <= _IMG_SIZE), JPEG-encode, base64.
    Cached by path so repeated modes/experiments don't re-encode. Thread-safe via
    lru_cache (GIL-atomic) + a lock around PIL work to be safe."""
    if _HAVE_PIL:
        with _b64_lock:
            img = Image.open(path_str).convert("RGB")
            if _IMG_SIZE and max(img.size) > _IMG_SIZE:
                img.thumbnail((_IMG_SIZE, _IMG_SIZE))
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=85)
            return "image/jpeg", base64.b64encode(buf.getvalue()).decode()
    # PIL 없으면 원본 바이트 그대로
    data = Path(path_str).read_bytes()
    mime = mimetypes.guess_type(path_str)[0] or "image/jpeg"
    return mime, base64.b64encode(data).decode()


@lru_cache(maxsize=8192)
def _resolve_path_cached(image_root_str: str, qid: str, opt_key: str,
                         explicit: str | None):
    """Resolve image path with the original priority, cached by (root,qid,opt)."""
    image_root = Path(image_root_str)
    if explicit:
        p = Path(explicit)
        if p.exists():
            return str(p)
        p2 = image_root / explicit
        if p2.exists():
            return str(p2)
    for ext in IMAGE_EXTS:
        p = image_root / qid / f"{opt_key}{ext}"
        if p.exists():
            return str(p)
    for ext in IMAGE_EXTS:
        p = image_root / f"{qid}_{opt_key}{ext}"
        if p.exists():
            return str(p)
    return None


def resolve_image_path(plan, opt_key, image_root):
    opt = plan.get("options", {}).get(opt_key, {})
    explicit = opt.get("image_path") or opt.get("image_url")
    if not explicit:
        plan_images = plan.get("images", {})
        explicit = plan_images.get(opt_key)
    qid = plan.get("query_id", "")
    res = _resolve_path_cached(str(image_root), qid, opt_key, explicit)
    return Path(res) if res else None


def option_to_fallback_text(opt):
    text = opt.get("search_query")
    if text:
        return text
    a = opt.get("attributes", {})
    parts = []
    if a.get("color"):       parts.append(a["color"])
    if a.get("pattern") and a["pattern"] != "solid":
        parts.append(a["pattern"].replace("_", " "))
    if a.get("garment_category"):
        parts.append(a["garment_category"].replace("_", " "))
    return " ".join(parts).strip() or "unknown item"


# ── prompt building ───────────────────────────────────────────────────────
def build_multimodal_messages(query, profile, shuffled_options, plan, image_root,
                              profile_mode="narrative",
                              system_prompt=SYSTEM_PROMPT_NO,
                              image_only_options=False):
    qtext = query.get("query_text", "").strip()

    profile_text = ""
    if profile_mode == "narrative":
        profile_text = profile_to_narrative(profile)
    elif profile_mode == "all":
        profile_text = profile_to_all_text(profile)

    content = []
    intro_parts = [f"=== QUERY ===\n{qtext}"]
    if profile_text and profile_mode != "no":
        intro_parts.append(f"\n=== USER PROFILE ===\n{profile_text}")
    intro_parts.append("\n=== OPTIONS ===")
    intro_parts.append("Below are four clothing option images labeled A, B, C, D.")
    content.append({"type": "text", "text": "\n".join(intro_parts)})

    load_failures = []
    for display_label, opt in shuffled_options:
        opt_key = opt.get("_original_key", display_label)
        img_path = resolve_image_path(plan, opt_key, image_root)
        content.append({"type": "text", "text": f"Option {display_label}:"})
        if img_path is not None:
            try:
                mime, b64 = _encode_b64(str(img_path))
                content.append({"type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{b64}"}})
            except Exception as e:
                if image_only_options:
                    content.append({"type": "text",
                                    "text": f"[Image unavailable for option {display_label}]"})
                else:
                    content.append({"type": "text",
                                    "text": f"[Image unavailable: {option_to_fallback_text(opt)}]"})
                load_failures.append((display_label, str(e)))
        else:
            if image_only_options:
                content.append({"type": "text",
                                "text": f"[Image not found for option {display_label}]"})
            else:
                content.append({"type": "text",
                                "text": f"[Image not found: {option_to_fallback_text(opt)}]"})
            load_failures.append((display_label, "path not found"))

    content.append({"type": "text", "text": (
        "\n=== INSTRUCTION ===\n"
        "Respond with ONE letter only: A, B, C, or D.\n"
        "Do NOT write any explanation or reasoning.\n"
        "Do NOT write anything before or after the letter.\n"
        "Your complete response must be a single character.\n\n"
        "Answer:")})

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": content}]
    return messages, load_failures


# ── VLM call (round-robin clients) ────────────────────────────────────────
_CLIENTS = []          # list of (client, model) ; set in main()
_GUIDED = True         # set from args in main()
_GUIDED_CHOICES = ["A", "B", "C", "D"]
_THINKING = False      # set from args in main()


def call_vlm(messages, endpoint_idx, model, max_tokens=8):
    client, _model = _CLIENTS[endpoint_idx % len(_CLIENTS)]
    kwargs = dict(model=model or _model, messages=messages,
                  max_tokens=max_tokens, temperature=0.0)
    extra = {}
    if _THINKING:
        # enable the model's reasoning; guided_choice is incompatible with a
        # think block (the model must emit reasoning tokens first), so it is NOT
        # set here. Need a large max_tokens so reasoning + final answer fit.
        extra["chat_template_kwargs"] = {"enable_thinking": True}
    else:
        if _GUIDED:
            extra["guided_choice"] = _GUIDED_CHOICES
        # explicitly disable thinking for Qwen3-style models when not thinking
        extra["chat_template_kwargs"] = {"enable_thinking": False}
    if extra:
        kwargs["extra_body"] = extra
    resp = client.chat.completions.create(**kwargs)
    msg = resp.choices[0].message
    content = msg.content or ""
    # some vLLM builds split reasoning into msg.reasoning and leave content as the
    # final answer; if content is empty but reasoning exists, fall back to reasoning
    if not content:
        content = getattr(msg, "reasoning", None) or ""
    return content


def parse_answer(response):
    response = (response or "").strip()
    if re.fullmatch(r"[ABCDabcd]", response):
        return response.upper()
    first = response.splitlines()[0].strip() if response else ""
    if re.fullmatch(r"[ABCDabcd]", first):
        return first.upper()
    return None


def parse_answer_thinking(response):
    """Extract the final A/B/C/D from a reasoning model's output, which may contain
    a long <think>...</think> block followed by the answer, or end with the choice.
    Strategy (most to least reliable):
      1. strip <think>...</think>, then try strict parse on what remains
      2. look for explicit 'Answer: X' / 'answer is X' / 'final answer X'
      3. take the LAST standalone A-D token (a letter on its own, not inside a word)
    """
    if not response:
        return None
    text = response.strip()

    # 1. remove a think block if present (closed or unclosed)
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    elif "<think>" in text:
        # unclosed think (truncated mid-reasoning) -> nothing usable after it
        text = text.split("<think>")[-1].strip()
    after = text

    # try strict on the post-think remainder
    s = parse_answer(after)
    if s:
        return s

    # 2. explicit answer phrases (case-insensitive)
    m = re.search(r"(?:answer|choice|option|select(?:ion)?|pick)\s*(?:is|:|=|->)?\s*"
                  r"\(?\*?\*?([ABCD])\*?\*?\)?", after, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"final\s+answer\s*(?:is|:)?\s*\(?([ABCD])\)?", after, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 3. last standalone A-D token (word boundary, uppercase only to avoid 'a'/'an')
    matches = re.findall(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", after)
    if matches:
        return matches[-1].upper()
    # fall back to scanning the FULL text's last standalone uppercase letter
    matches = re.findall(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", response)
    if matches:
        return matches[-1].upper()
    return None


# ── evaluate ──────────────────────────────────────────────────────────────
def evaluate(plans, queries_map, profiles_map, image_root,
             model, limit=0, seed=42, verbose=False,
             profile_mode="narrative", concurrency=1,
             image_only_options=False, max_tokens=4,
             fix_correct=None):
    """fix_correct: None -> random shuffle (1-1 position-pref experiment).
       'A'/'B'/'C'/'D' -> place the TRUE answer (orig key 'A') at that display
       position, others fill remaining slots in order (1-2 fixed-position exp)."""
    rng = random.Random(seed)
    if limit > 0:
        plans = plans[:limit]

    if profile_mode == "no":
        system_prompt = SYSTEM_PROMPT_NO
    else:
        system_prompt = SYSTEM_PROMPT_WITH_PROFILE

    jobs = []
    for plan in plans:
        qid = plan["query_id"]; uid = plan["user_id"]
        query = queries_map.get(qid)
        profile = profiles_map.get(uid, {})
        if query is None:
            continue
        option_items = list(plan["options"].items())   # [("A",{...}),...]

        if fix_correct in ("A", "B", "C", "D"):
            # place true answer (orig 'A') at the fixed display slot
            true_item = next((it for it in option_items if it[0] == "A"), None)
            others = [it for it in option_items if it[0] != "A"]
            rng.shuffle(others)
            display_labels = ["A", "B", "C", "D"]
            slot = display_labels.index(fix_correct)
            arranged = []
            oi = 0
            for i in range(4):
                if i == slot:
                    arranged.append(true_item)
                else:
                    arranged.append(others[oi]); oi += 1
            option_items = arranged
        else:
            rng.shuffle(option_items)

        display_labels = ["A", "B", "C", "D"]
        shuffled = []
        for disp, (orig_key, opt_dict) in zip(display_labels, option_items):
            shuffled.append((disp, dict(opt_dict, _original_key=orig_key)))
        display_to_original = {d: orig for d, (orig, _) in zip(display_labels, option_items)}
        correct_display = next((d for d, o in display_to_original.items() if o == "A"), None)
        jobs.append((plan, query, profile, shuffled, display_to_original, correct_display))

    results = []
    strict_correct = tpo_correct = profile_correct = total = 0
    breakdown = defaultdict(lambda: {"strict_correct": 0, "tpo_correct": 0,
                                     "profile_correct": 0, "total": 0})
    # answer position histogram (1-1): how often each DISPLAY slot is chosen
    pos_hist = defaultdict(int)

    def process(idx, job):
        plan, query, profile, shuffled, display_to_original, correct_display = job
        qid = plan["query_id"]; uid = plan["user_id"]
        axis = plan.get("active_axis", "unknown")
        qtype = query.get("query_type", "unknown")
        response = predicted = predicted_original = None
        img_failures = []
        try:
            messages, img_failures = build_multimodal_messages(
                query, profile, shuffled, plan, image_root,
                profile_mode=profile_mode, system_prompt=system_prompt,
                image_only_options=image_only_options)
            response = call_vlm(messages, idx, model, max_tokens=max_tokens)
            predicted = parse_answer_thinking(response) if _THINKING else parse_answer(response)
            predicted_original = display_to_original.get(predicted)
        except Exception as e:
            print(f"  [ERROR] {qid}: {e}")
        strict_hit  = int(predicted_original == "A") if predicted_original else 0
        tpo_hit     = TPO_SCORE.get(predicted_original, 0) if predicted_original else 0
        profile_hit = PROFILE_SCORE.get(predicted_original, 0) if predicted_original else 0
        rec = {"_idx": idx, "query_id": qid, "user_id": uid, "active_axis": axis,
               "query_type": qtype, "scenario_id": plan.get("scenario_id"),
               "profile_mode": profile_mode, "correct_display": correct_display,
               "predicted": predicted, "predicted_original": predicted_original,
               "strict_correct": bool(strict_hit), "tpo_score": tpo_hit,
               "profile_score": profile_hit, "raw_response": response,
               "image_failures": img_failures}
        return rec, strict_hit, tpo_hit, profile_hit, axis, qtype, predicted

    def tally(rec, sh, th, ph, axis, qtype, predicted):
        nonlocal strict_correct, tpo_correct, profile_correct, total
        results.append(rec)
        strict_correct += sh; tpo_correct += th; profile_correct += ph; total += 1
        if predicted:
            pos_hist[predicted] += 1
        for key in [f"axis:{axis}", f"qtype:{qtype}"]:
            breakdown[key]["total"] += 1
            breakdown[key]["strict_correct"]  += sh
            breakdown[key]["tpo_correct"]     += th
            breakdown[key]["profile_correct"] += ph

    if concurrency > 1:
        with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {ex.submit(process, i, job): i for i, job in enumerate(jobs)}
            done = 0
            for fut in cf.as_completed(futures):
                rec, sh, th, ph, axis, qtype, pred = fut.result()
                tally(rec, sh, th, ph, axis, qtype, pred)
                done += 1
                if verbose or done % 100 == 0:
                    st = "✓" if sh else "✗"
                    fn = f" [img_fail:{len(rec['image_failures'])}]" if rec["image_failures"] else ""
                    print(f"  [{done:3d}/{len(jobs)}] {st} pred={rec['predicted']} "
                          f"orig={rec['predicted_original']} ans={rec['correct_display']} | "
                          f"axis={axis} qtype={qtype}{fn}")
    else:
        for i, job in enumerate(jobs):
            rec, sh, th, ph, axis, qtype, pred = process(i, job)
            tally(rec, sh, th, ph, axis, qtype, pred)
            if verbose or (i + 1) % 10 == 0:
                st = "✓" if sh else "✗"
                fn = f" [img_fail:{len(rec['image_failures'])}]" if rec["image_failures"] else ""
                print(f"  [{i+1:3d}/{len(jobs)}] {st} pred={rec['predicted']} "
                      f"orig={rec['predicted_original']} ans={rec['correct_display']} | "
                      f"axis={axis} qtype={qtype}{fn}")

    results.sort(key=lambda x: x["_idx"])
    for r in results:
        r.pop("_idx")
    return (results, strict_correct, tpo_correct, profile_correct, total,
            breakdown, dict(pos_hist))


# ── report ────────────────────────────────────────────────────────────────
def print_report(strict_c, tpo_c, prof_c, total, breakdown, pos_hist,
                 profile_mode=None, model_name=None, fix_correct=None):
    sa = strict_c / total * 100 if total else 0
    ta = tpo_c / total * 100 if total else 0
    pa = prof_c / total * 100 if total else 0
    print("\n" + "=" * 60)
    if profile_mode: print(f"  PROFILE MODE:      {profile_mode}")
    if model_name:   print(f"  MODEL:             {model_name}")
    if fix_correct:  print(f"  FIXED CORRECT POS: {fix_correct}")
    print(f"  STRICT ACCURACY:   {strict_c}/{total} = {sa:.1f}%")
    print(f"  TPO ACCURACY:      {tpo_c}/{total}    = {ta:.1f}%")
    print(f"  PROFILE ACCURACY:  {prof_c}/{total} = {pa:.1f}%")
    print(f"  Random baseline:   25.0% strict")
    print("=" * 60)

    # position-preference histogram (1-1)
    if pos_hist:
        tot = sum(pos_hist.values())
        print("\n  ── Answer position preference (chosen display slot) ──")
        for slot in ["A", "B", "C", "D"]:
            n = pos_hist.get(slot, 0)
            pct = n / tot * 100 if tot else 0
            bar = "█" * int(pct / 2)
            print(f"  {slot}: {n:4d} ({pct:5.1f}%)  {bar}")
        print(f"  (uniform would be 25.0% each; deviation = position bias)")

    print("\n  ── By active_axis ──")
    for key in sorted(k for k in breakdown if k.startswith("axis:")):
        d = breakdown[key]; n = d["total"]
        s = d["strict_correct"]/n*100 if n else 0
        t = d["tpo_correct"]/n*100 if n else 0
        p = d["profile_correct"]/n*100 if n else 0
        print(f"  {key:18s} strict={s:5.1f}%  tpo={t:5.1f}%  profile={p:5.1f}%  (n={n})")
    print("\n  ── By query_type ──")
    for key in sorted(k for k in breakdown if k.startswith("qtype:")):
        d = breakdown[key]; n = d["total"]
        s = d["strict_correct"]/n*100 if n else 0
        t = d["tpo_correct"]/n*100 if n else 0
        p = d["profile_correct"]/n*100 if n else 0
        print(f"  {key:18s} strict={s:5.1f}%  tpo={t:5.1f}%  profile={p:5.1f}%  (n={n})")


def print_combined(all_summaries):
    print("\n" + "=" * 60)
    print("  ══ COMBINED SUMMARY ══")
    print(f"  {'mode':<12} {'strict':>7} {'tpo':>7} {'profile':>9}")
    print("  " + "-" * 40)
    for s in all_summaries:
        t = s["total"]
        sa = s["strict_correct"]/t*100 if t else 0
        ta = s["tpo_correct"]/t*100 if t else 0
        pa = s["profile_correct"]/t*100 if t else 0
        print(f"  {s['profile_mode']:<12} {sa:6.1f}%  {ta:6.1f}%  {pa:7.1f}%")
    print("=" * 60)


def sanitize(s):
    s = (s or "model").strip().replace("/", "__")
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("._-")
    return s or "model"


# ── main ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--profiles", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("multimodal_results.jsonl"),
                    help="base output path; filename gets _{mode}_{model}[_fix] suffix")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output DIRECTORY; filenames auto-named "
                         "multimodal_results_{mode}_{model}[_fix].jsonl inside it "
                         "(overrides --output's directory)")
    ap.add_argument("--image-root", type=Path, required=True)
    ap.add_argument("--base-urls", type=str, default="http://127.0.0.1:8002/v1",
                    help="comma-separated OpenAI-compat endpoints; round-robined across workers")
    ap.add_argument("--api-key", type=str, default="token-abc123")
    ap.add_argument("--model", type=str, required=True,
                    help="model name served by the endpoints, e.g. Qwen/Qwen3-VL-30B-A3B-Instruct")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--profile-mode", choices=["no", "narrative", "all"], default=None,
                    help="omit to run no/narrative/all sequentially")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--image-only-options", action="store_true")
    ap.add_argument("--img-size", type=int, default=768,
                    help="downscale long edge to this before encoding (0 = no resize)")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="0 = auto (8 normally, 2048 with --enable-thinking)")
    ap.add_argument("--enable-thinking", action="store_true",
                    help="enable the model's reasoning/think block. Auto-disables "
                         "guided_choice (incompatible) and raises max_tokens; uses a "
                         "reasoning-aware parser to extract the final A/B/C/D.")
    ap.add_argument("--guided", dest="guided", action="store_true", default=True,
                    help="constrain vLLM decoding to exactly A/B/C/D (default on; "
                         "ignored when --enable-thinking)")
    ap.add_argument("--no-guided", dest="guided", action="store_false",
                    help="disable guided_choice")
    ap.add_argument("--fix-correct", choices=["A", "B", "C", "D"], default=None,
                    help="1-2 experiment: pin the TRUE answer at this display slot "
                         "(default: random shuffle, the 1-1 experiment)")
    args = ap.parse_args()

    global _IMG_SIZE, _CLIENTS, _GUIDED, _THINKING
    _IMG_SIZE = args.img_size if args.img_size > 0 else None
    _THINKING = args.enable_thinking
    # thinking is incompatible with guided_choice -> force guided off when thinking
    _GUIDED = args.guided and not _THINKING
    # auto max_tokens: small for guided/instruct, large for reasoning
    if args.max_tokens and args.max_tokens > 0:
        max_tokens = args.max_tokens
    else:
        max_tokens = 2048 if _THINKING else 8

    base_urls = [u.strip() for u in args.base_urls.split(",") if u.strip()]
    _CLIENTS = [(openai.OpenAI(base_url=u, api_key=args.api_key), args.model) for u in base_urls]
    print(f"  Endpoints ({len(_CLIENTS)}): {base_urls}")
    print(f"  Model: {args.model}  (guided_choice={_GUIDED}, thinking={_THINKING}, "
          f"max_tokens={max_tokens})")
    print(f"  Image root: {args.image_root}  (resize long edge -> {_IMG_SIZE})")
    if not args.image_root.exists():
        print(f"  [WARN] image_root does not exist: {args.image_root}")

    plans = load_jsonl(args.plans)
    queries = load_jsonl(args.queries)
    profiles = load_jsonl(args.profiles)
    queries_map = {q["query_id"]: q for q in queries}
    profiles_map = {p["user_id"]: p for p in profiles}
    print(f"  Plans: {len(plans)}, Queries: {len(queries)}, Profiles: {len(profiles)}")
    print(f"  Concurrency: {args.concurrency} | image-only: {args.image_only_options} "
          f"| fix-correct: {args.fix_correct}")
    if args.limit:
        print(f"  Limit: {args.limit}")

    modes = [args.profile_mode] if args.profile_mode else ["no", "narrative", "all"]
    run_all = args.profile_mode is None
    all_summaries = []
    model_tag = sanitize(args.model)

    for mode in modes:
        print(f"\n{'─'*60}\n  ▶ profile-mode = {mode}  |  model = {args.model}\n{'─'*60}")
        (results, sc, tc, pc, total, breakdown, pos_hist) = evaluate(
            plans, queries_map, profiles_map, args.image_root,
            model=args.model, limit=args.limit, seed=args.seed, verbose=args.verbose,
            profile_mode=mode, concurrency=args.concurrency,
            image_only_options=args.image_only_options, max_tokens=max_tokens,
            fix_correct=args.fix_correct)

        suffix = f"_{mode}_{model_tag}"
        if args.fix_correct:
            suffix += f"_fix{args.fix_correct}"
        if args.out_dir:
            out_path = args.out_dir / f"multimodal_results{suffix}.jsonl"
        elif run_all or args.output == Path("multimodal_results.jsonl"):
            out_path = args.output.with_name(f"multimodal_results{suffix}.jsonl")
        else:
            out_path = args.output.with_name(f"{args.output.stem}{suffix}{args.output.suffix or '.jsonl'}")
        save_jsonl(results, out_path)   # save_jsonl already mkdirs parents
        print(f"\n  Saved {len(results)} records → {out_path}")
        print_report(sc, tc, pc, total, breakdown, pos_hist,
                     profile_mode=mode, model_name=args.model, fix_correct=args.fix_correct)
        all_summaries.append({"profile_mode": mode, "strict_correct": sc,
                              "tpo_correct": tc, "profile_correct": pc, "total": total})

    if run_all:
        print_combined(all_summaries)


if __name__ == "__main__":
    main()