#!/usr/bin/env python3
"""POD-Bench text-only A/B/C/D evaluation — eval_multimodal.py의 무이미지 판.

WHAT CHANGES
------------
옵션이 이미지가 아니라 정식 속성 텍스트로 제시된다.

    Option A: navy checkered blazer

그 외 모든 것 — 다섯 입력 형식, 프롬프트 문구, 채점(strict/tpo/preference),
plan당 1회 셔플, (plan_id, input_format) 단위 resume, results.jsonl 스키마 —
은 eval_multimodal.py와 동일하다. 따라서 같은 --seed로 돌리면 **문항별 선택지
순서까지 같아서** 이미지 판과 문항 단위로 짝지어 비교할 수 있고, 기존 집계·
트랙 분해 스크립트가 그대로 읽는다. 구분은 레코드의 modality 필드로 한다.

WHY IT MATTERS
--------------
이미지 판과의 차이가 곧 "이 문항을 푸는 데 시각 인지가 얼마나 필요한가"의
측정치다. 텍스트 판이 훨씬 높다면 병목은 추론이 아니라 지각이고, 비슷하다면
지각은 이미 충분하며 트레이드오프는 순수하게 추론 층의 현상이다.

주의: 텍스트 판에서 옵션과 프로필은 같은 정식 어휘를 쓰므로 preference 축이
부분적으로 문자열 대조로 풀린다. 이는 결함이 아니라 이 조건의 정의이며,
논문에서 이미지 판의 상한(ceiling)으로 보고해야지 같은 과제로 취급하면 안 된다.

DEFAULT ITEM SET
----------------
기본값은 이미지 판과 **정확히 같은 문항 집합**이다(셀 이미지가 네 장 모두
존재하는 plan만). 비교 가능성이 목적이므로 이게 기본이고, 이미지 제약 없이
전량을 쓰려면 --no-image-filter.

USAGE
-----
    # 로컬 vLLM (텍스트 전용이므로 --limit-mm-per-prompt 불필요)
    python -m exp.vlm_eval.eval_text --port 8001 --model Qwen/Qwen3-VL-4B-Instruct

    # 호스티드
    python -m exp.vlm_eval.eval_text --base-url https://api.openai.com/v1 \
        --model gpt-5-mini --api-key-env OPENAI_API_KEY
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
from exp.vlm_eval.option_order import assign_orders
from exp.vlm_eval.eval_multimodal import missing_images_error       # noqa: E402
from configs.scenarios import (EVAL_FRAME_CLAUSE, EVAL_PRIORITY_CLAUSE_SITUATION,
                               EVAL_PRIORITY_CLAUSE_PREFERENCE, PROMPT_VERSION)

INPUT_FORMATS = ["query", "narrative", "narrative+query", "all", "all+query"]
TPO_SCORE        = {"A": 1, "B": 1, "C": 0, "D": 0}
PREFERENCE_SCORE = {"A": 1, "B": 0, "C": 1, "D": 0}
LABELS = ["A", "B", "C", "D"]
MODALITY = "text"


# ══ prompts ═══════════════════════════════════════════════════════════════
# eval_multimodal.py와 같은 문구. "images" → "described by text attributes"만
# 바뀐다. 조항은 configs/scenarios.py에서 오므로 카탈로그와 절대 어긋나지 않는다.
_OUTPUT_RULE = """
OUTPUT FORMAT — CRITICAL:
You MUST output EXACTLY one character: A, B, C, or D.
Do NOT output any explanation, reasoning, punctuation, or whitespace.
Do NOT write sentences. Do NOT write words.
If you write anything other than a single letter, your response is INVALID.
Your ENTIRE response must be one of: A  B  C  D
"""

SYSTEM_QUERY_ONLY = f"""You are a fashion advisor.

You will be given:
1. A fashion query describing a situation or occasion
2. Four clothing options (A, B, C, D) described by text attributes only

Your task:
Select the single BEST option that best fits the query.
{EVAL_FRAME_CLAUSE}
{EVAL_PRIORITY_CLAUSE_SITUATION}
{_OUTPUT_RULE}"""

SYSTEM_PROFILE_ONLY = f"""You are a fashion advisor.

You will be given:
1. A user profile describing clothing preferences
2. Four clothing options (A, B, C, D) described by text attributes only

Your task:
Select the single BEST option that best matches the user's preferences.
{EVAL_FRAME_CLAUSE}
{_OUTPUT_RULE}"""

SYSTEM_WITH_PROFILE = f"""You are a fashion advisor.

You will be given:
1. A fashion query (situation or occasion)
2. A user profile describing clothing preferences
3. Four clothing options (A, B, C, D) described by text attributes only

Your task:
Select the single BEST option that best fits both the query and the user's preferences.
{EVAL_FRAME_CLAUSE}
{EVAL_PRIORITY_CLAUSE_SITUATION}
{EVAL_PRIORITY_CLAUSE_PREFERENCE}
{_OUTPUT_RULE}"""


def uses_query(fmt):    return fmt in {"query", "all+query", "narrative+query"}
def uses_profile(fmt):  return fmt in {"all", "all+query", "narrative", "narrative+query"}


def system_prompt_for(fmt):
    if not uses_profile(fmt):
        return SYSTEM_QUERY_ONLY
    return SYSTEM_WITH_PROFILE if uses_query(fmt) else SYSTEM_PROFILE_ONLY


# ══ profile rendering (eval_multimodal.py와 동일) ══════════════════════════
_NARRATIVE_KEYS = ["narrative_profile", "narrative", "profile_text",
                   "description", "user_profile", "profile", "text"]


def profile_to_narrative(profile):
    for src in (profile, profile.get("metadata") if isinstance(profile.get("metadata"), dict) else {}):
        for key in _NARRATIVE_KEYS:
            val = (src or {}).get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return ""


def profile_to_all_kv_text(profile):
    attrs = profile.get("structured_attributes") or {}
    lines = []
    for axis, label in [("garment_category", "garment"), ("color", "color"),
                        ("pattern", "pattern")]:
        info = attrs.get(axis) if isinstance(attrs, dict) else None
        likes = (info or {}).get("likes") or []
        dislikes = (info or {}).get("dislikes") or []
        fmt = lambda vs: ", ".join(str(v).replace("_", " ") for v in vs) or "none"
        lines.append(f"likes.{label}: {fmt(likes)}")
        lines.append(f"dislikes.{label}: {fmt(dislikes)}")
    return "\n".join(lines)


# ══ option rendering ══════════════════════════════════════════════════════
# 정식 어휘만 쓴다. 별칭·상위어 금지(작업 원칙 §8): 'wool coat' 같은 상위어가
# 다시 들어오면 pea_coat와 long_coat가 텍스트 층에서 구별 불가능해진다.
# solid는 무늬 축의 영(zero) 수준이므로 표기하지 않는다 — 프로필에서도 전원
# 중립인 값이라, 적는 순간 없는 대비를 만든다.
_ORTHOGRAPHY = {"t_shirt": "t-shirt", "polka_dot": "polka-dot"}


def _word(v):
    return _ORTHOGRAPHY.get(v, str(v).replace("_", " "))


def option_to_text(opt, style="attributes"):
    if style == "search_query" and opt.get("search_query"):
        return opt["search_query"]
    a = opt.get("attributes", {})
    parts = []
    if a.get("color"):
        parts.append(_word(a["color"]))
    if a.get("pattern") and a["pattern"] != "solid":
        parts.append(_word(a["pattern"]))
    if a.get("garment_category"):
        parts.append(_word(a["garment_category"]))
    return " ".join(parts).strip() or "unknown item"


# ══ message building ══════════════════════════════════════════════════════
# 모든 모델이 같은 문자열을 보도록, system 역할을 쓰지 않고 지시문을
# user 메시지 맨 앞에 합친다. 채팅 템플릿이 모델마다 달라 (Qwen 은 system 을
# 별도 블록으로, Gemma 는 system 을 지원하지 않아 첫 user 턴에 병합) 같은
# messages 를 보내도 최종 시퀀스가 달라지기 때문이다. PersonaVLM(CVPR'26),
# CoViP 등 멀티모달 대화 데이터셋도 같은 방식을 쓴다.
def build_messages(query, profile, shuffled, fmt, style="attributes"):
    # 순서: 프로필 → 질의 → 선택지 → 지시문 (eval_multimodal / eval_dialog 와 동일)
    parts = []
    if fmt in {"narrative", "narrative+query"}:
        parts.append(f"=== USER PROFILE ===\n{profile_to_narrative(profile)}")
    elif fmt in {"all", "all+query"}:
        parts.append(f"=== USER PROFILE ===\n{profile_to_all_kv_text(profile)}")
    if uses_query(fmt):
        pre = "\n" if parts else ""
        parts.append(f"{pre}=== QUERY ===\n"
                     f"{(query or {}).get('query_text','').strip()}")

    parts.append("\n=== OPTIONS ===")
    parts.append("Below are four clothing options labeled A, B, C, D.")
    for label, opt in shuffled:
        parts.append(f"Option {label}: {option_to_text(opt, style)}")

    if fmt == "query":
        task = "Select the single BEST option for the query."
    elif uses_query(fmt):
        task = "Select the single BEST option for both the query and the user's preferences."
    else:
        task = "Select the single BEST option for the user's preferences."

    parts.append("\n=== INSTRUCTION ===\n"
                 f"{task}\n"
                 "Respond with ONE letter only: A, B, C, or D.\n"
                 "Do NOT write any explanation or reasoning.\n"
                 "Do NOT write anything before or after the letter.\n"
                 "Your complete response must be a single character.\n\n"
                 "Answer:")
    return [{"role": "user",
             "content": system_prompt_for(fmt) + "\n" + "\n".join(parts)}]


def parse_answer(text):
    t = (text or "").strip()
    if not t:
        return None
    for ch in t:                      # first standalone letter wins
        if ch.upper() in LABELS:
            return ch.upper()
    return None


# ══ model call ════════════════════════════════════════════════════════════
def _is_reasoning(model):
    return model.split("/")[-1].startswith(("gpt-5", "o1", "o3", "o4"))


def call_model(session, url, model, messages, max_tokens, temperature,
               timeout, retries, api_key=None):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    payload = {"model": model, "messages": messages}
    if _is_reasoning(model):
        # 추론 토큰이 응답 예산을 선점하므로 4로는 빈 응답이 나온다.
        payload["max_completion_tokens"] = max(max_tokens, 2048)
        payload["reasoning_effort"] = "minimal"
    else:
        payload["max_tokens"] = max_tokens
        payload["temperature"] = temperature
    last = None
    for attempt in range(retries):
        try:
            r = session.post(url, json=payload, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            last = e
            resp = getattr(e, "response", None)
            if resp is not None:
                # 400/404/422는 재시도해도 같으므로 본문을 남기고 즉시 중단
                last = RuntimeError(f"{e} :: {resp.text[:300]}")
                if resp.status_code in (400, 404, 422):
                    break
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"{type(last).__name__}: {str(last)[:300]}")


# ══ jobs ══════════════════════════════════════════════════════════════════
def cell_image_exists(attrs, images_root: Path) -> bool:
    c, g, p = attrs.get("color"), attrs.get("garment_category"), attrs.get("pattern")
    if not (c and g and p):
        return False
    return any((images_root / c / f"{p}_{g}{ext}").is_file()
               for ext in (".jpg", ".jpeg", ".png", ".webp"))


def make_jobs(plans, queries, profiles, fmts, seed, done,
              images_root=None, orders=None, allow_missing=False):
    """셔플은 plan당 1회. eval_multimodal.py와 같은 키(f"{seed}|{pid}")를 쓰므로
    같은 seed면 이미지 판과 선택지 순서가 동일하다 — 문항 단위 짝비교가 된다."""
    jobs, skipped = [], []
    for plan in plans:
        pid = plan.get("plan_id") or plan["query_id"]
        missing = [] if images_root is None else [
            (k, plan["options"][k].get("attributes", {})) for k in LABELS
            if not cell_image_exists(plan["options"][k].get("attributes", {}),
                                     images_root)]
        if missing:
            # 텍스트 판은 이미지 판과 같은 문항 집합 위에서 돌아야 짝비교가
            # 성립한다. 조용히 빼면 두 집합이 갈라지고 그 사실이 어디에도
            # 남지 않는다 — 이미지 판과 같은 이유로 여기서 멈춘다.
            if not allow_missing:
                raise missing_images_error(pid, missing, images_root)
            skipped.append(pid); continue
        order = orders[pid] if orders else None
        if order is None:
            rng = random.Random(f"{seed}|{pid}")
            order = list(LABELS); rng.shuffle(order)
        shuffled = [(d, plan["options"][k]) for d, k in zip(LABELS, order)]
        d2o = {d: k for d, k in zip(LABELS, order)}
        correct = next(d for d, k in d2o.items() if k == "A")
        for fmt in fmts:
            if (pid, fmt) in done:
                continue
            jobs.append({"plan": plan, "pid": pid, "fmt": fmt,
                         "query": queries.get(plan["query_id"]),
                         "profile": profiles.get(plan["user_id"], {}),
                         "shuffled": shuffled, "d2o": d2o, "correct": correct})
    return jobs, skipped


# ══ reporting ═════════════════════════════════════════════════════════════
def pct(n, d):
    return f"{n / d:6.1%}" if d else "     -"


def print_block(rows, title, fmts):
    buckets = sorted({b for b, _ in rows})
    if not buckets:
        return
    w = max(len(str(b)) for b in buckets) + 2
    print(f"\n  {title}")
    print(f"  {'':<{w}}{'n':>6}{'strict':>9}{'tpo':>9}{'preference':>12}")
    print("  " + "-" * (w + 36))
    for b in buckets:
        for fmt in fmts:
            d = rows.get((b, fmt))
            if not d:
                continue
            print(f"  {str(b) + '/' + fmt:<{w + 16}}{d['n']:>6}"
                  f"{pct(d['strict'], d['n']):>9}{pct(d['tpo'], d['n']):>9}"
                  f"{pct(d['pref'], d['n']):>12}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="wacv_scenario_v5")
    ap.add_argument("--plans", type=Path)
    ap.add_argument("--queries", type=Path)
    ap.add_argument("--profiles", type=Path)
    ap.add_argument("--images-root", type=Path,
                    help="문항 집합을 이미지 판과 맞추는 데만 쓴다 (인코딩 없음)")
    ap.add_argument("--no-image-filter", action="store_true",
                    help="이미지 없는 plan도 포함 (이미지 판과 문항 집합이 달라짐)")
    ap.add_argument("--allow-missing-images", action="store_true",
                    help="이미지 없는 plan을 중단 대신 조용히 제외한다. "
                         "기본값은 중단: 제외는 문항 집합을 말없이 줄인다.")

    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int,
                    help="shortcut for --base-url http://127.0.0.1:PORT/v1")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key-env", default=None)

    ap.add_argument("--input-formats", default=",".join(INPUT_FORMATS))
    ap.add_argument("--track", default="both",
                    choices=["both", "physical", "dress_code"])
    ap.add_argument("--option-style", default="attributes",
                    choices=["attributes", "search_query"],
                    help="attributes=정식 어휘 렌더링(기본), search_query=코퍼스 질의 문자열")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int,
                    default=int(os.environ.get("POD_SEED", 1)),
                    help="문항 생성과 같은 시드여야 세 평가 스크립트의 셔플이 "
                         "일치해 문항 단위 짝비교가 성립한다. 기본값은 "
                         "$POD_SEED (없으면 1).")
    ap.add_argument("--option-order", default="balanced",
                    choices=["balanced", "random",
                             "fixed:A", "fixed:B", "fixed:C", "fixed:D"],
                    help="정답 위치 배정. balanced(기본)=정확히 4등분, "
                         "random=문항별 독립 셔플, fixed:X=정답을 항상 X 에 "
                         "고정(위치 편향 진단용). 세 평가 스크립트에 같은 값을 "
                         "줘야 문항 단위 짝비교가 성립한다.")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    data = REPO / f"data_{args.variant}"
    plans_path    = args.plans    or data / "options" / "option_plans.jsonl"
    queries_path  = args.queries  or data / "queries" / "queries.jsonl"
    profiles_path = args.profiles or data / "profiles" / "profiles.jsonl"
    images_root   = None if args.no_image_filter else (args.images_root or data / "images")
    base_url = args.base_url or (f"http://127.0.0.1:{args.port}/v1" if args.port
                                 else "http://127.0.0.1:8001/v1")
    url = base_url.rstrip("/") + "/chat/completions"
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if args.api_key_env and not api_key:
        raise SystemExit(f"[error] ${args.api_key_env} is empty")

    fmts = [f.strip() for f in args.input_formats.split(",") if f.strip()]
    for f in fmts:
        if f not in INPUT_FORMATS:
            raise SystemExit(f"[error] unknown format {f!r}; have {INPUT_FORMATS}")

    load = lambda p: [json.loads(l) for l in open(p) if l.strip()]
    plans = load(plans_path)
    queries = {q["query_id"]: q for q in load(queries_path)}
    profiles = {p["user_id"]: p for p in load(profiles_path)}
    all_plans = list(plans)          # 필터 전 전체 목록
    if args.track != "both":
        plans = [p for p in plans if p.get("track") == args.track]
    if args.limit > 0:
        plans = plans[:args.limit]

    # 이미지 판과 결과가 섞이지 않도록 출력 디렉터리에 __text 접미사
    # 정답 위치를 고정하거나 무작위로 돌린 결과는 기본(balanced) 결과와
    # 섞이면 안 된다. 같은 (plan_id, format) 인데 배치가 다르므로 resume 이
    # 이미 끝난 것으로 착각한다. 그래서 경로에 모드를 새긴다.
    _pos = ("" if args.option_order == "balanced"
            else "_" + args.option_order.replace(":", ""))
    out_dir = args.out or (data / ("eval_text" + _pos) / f"{args.model.replace('/', '_')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "results.jsonl"

    done = set()
    if res_path.exists() and not args.fresh:
        for line in open(res_path):
            try:
                r = json.loads(line)
                if not r.get("error"):        # 실패한 행은 완료로 치지 않는다
                    done.add((r["plan_id"], r["input_format"]))
            except Exception:
                pass
    elif args.fresh and res_path.exists():
        res_path.unlink()

    # 선택지 순서는 전체 plan 목록 위에서 한 번에 정한다.
    # --limit / --track 으로 부분만 돌려도 배치가 달라지지 않게 하기 위해서다.
    all_pids = [p.get("plan_id") or p["query_id"] for p in all_plans]
    orders = assign_orders(all_pids, args.seed, args.option_order)
    from collections import Counter as _C
    from exp.vlm_eval.option_order import correct_display as _cd
    _d = _C(_cd(orders[p]) for p in all_pids)
    print(f"  positions  : {args.option_order} (seed {args.seed}) "
          f"{dict(sorted(_d.items()))}")

    jobs, skipped = make_jobs(plans, queries, profiles, fmts, args.seed, done,
                              images_root, orders,
                              allow_missing=args.allow_missing_images)

    print("=" * 74)
    print("  POD-Bench TEXT-ONLY A/B/C/D evaluation")
    print(f"  model      : {args.model}")
    print(f"  endpoint   : {url}")
    print(f"  variant    : {args.variant}   track: {args.track}")
    print(f"  options    : text ({args.option_style})")
    print(f"  item set   : " + ("all plans (--no-image-filter)" if images_root is None
                                else f"same as image run ({images_root})"))
    print(f"  formats    : {', '.join(fmts)}")
    print(f"  plans      : {len(plans)} loaded, {len(skipped)} excluded")
    print(f"  jobs       : {len(jobs)}"
          + (f"  ({len(done)} already done, resuming)" if done else ""))
    print(f"  prompt_ver : {PROMPT_VERSION}")
    print("=" * 74)
    if not jobs:
        print("  nothing to do")
        return 0

    session = requests.Session()
    for scheme in ("http://", "https://"):
        session.mount(scheme, requests.adapters.HTTPAdapter(
            pool_connections=args.concurrency * 2, pool_maxsize=args.concurrency * 2))

    write_lock = threading.Lock()
    fh = open(res_path, "a", encoding="utf-8")
    counter = {"n": 0}

    def run(job):
        plan, fmt = job["plan"], job["fmt"]
        raw = err = None
        try:
            msgs = build_messages(job["query"], job["profile"], job["shuffled"],
                                  fmt, args.option_style)
            raw = call_model(session, url, args.model, msgs, args.max_tokens,
                             args.temperature, args.timeout, args.retries, api_key)
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:300]}"
        disp = parse_answer(raw)
        orig = job["d2o"].get(disp)
        rec = {"plan_id": job["pid"], "query_id": plan["query_id"],
               "user_id": plan["user_id"], "input_format": fmt,
               "modality": MODALITY,
               "track": plan.get("track"), "scenario_id": plan.get("scenario_id"),
               "active_axis": plan.get("active_axis"),
               "violation_axis": plan.get("violation_axis"),
               "query_type": (job["query"] or {}).get("query_type"),
               "correct_display": job["correct"], "predicted_display": disp,
               "predicted_original": orig,
               "strict": int(orig == "A"),
               "tpo": TPO_SCORE.get(orig, 0),
               "preference": PREFERENCE_SCORE.get(orig, 0),
               "raw_response": raw, "error": err,
               "model": args.model, "prompt_version": PROMPT_VERSION,
               "option_style": args.option_style}
        with write_lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            counter["n"] += 1
            if counter["n"] % 50 == 0 or counter["n"] == len(jobs):
                print(f"  [{counter['n']:5d}/{len(jobs)}]")
        return rec

    t0 = time.time()
    try:
        with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            list(ex.map(run, jobs))
    except KeyboardInterrupt:
        print("\n  [interrupt] finished items are saved; re-run to resume")
    finally:
        fh.close()
    print(f"\n  {counter['n']} items in {time.time() - t0:.0f}s")

    rows = [json.loads(l) for l in open(res_path) if l.strip()]
    rows = [r for r in rows if r["input_format"] in fmts]

    def bucket(keyfn):
        agg = defaultdict(lambda: {"strict": 0, "tpo": 0, "pref": 0, "n": 0})
        for r in rows:
            d = agg[(keyfn(r), r["input_format"])]
            d["strict"] += r["strict"]; d["tpo"] += r["tpo"]
            d["pref"] += r["preference"]; d["n"] += 1
        return agg

    print("\n" + "=" * 74)
    print("  RESULTS (text-only)")
    print("=" * 74)
    print_block(bucket(lambda r: "ALL"), "overall", fmts)
    print_block(bucket(lambda r: r.get("track") or "?"),
                "by track (independent datasets — do NOT pool)", fmts)
    print_block(bucket(lambda r: f"act:{r.get('active_axis')}"),
                "by ACTIVE axis (which preference is probed)", fmts)
    print_block(bucket(lambda r: f"vio:{r.get('violation_axis')}"),
                "by VIOLATION axis (which TPO rule is tested)", fmts)
    print_block(bucket(lambda r: f"{r.get('track')}/{r.get('active_axis')}"),
                "track x active axis", fmts)

    errs = sum(1 for r in rows if r.get("error"))
    unp = sum(1 for r in rows if not r.get("error") and r["predicted_display"] is None)
    print(f"\n  request errors {errs}   unparseable answers {unp}")
    pos = Counter(r["predicted_display"] for r in rows if r["predicted_display"])
    print(f"  answer position histogram: {dict(sorted(pos.items()))}"
          "   (flat is good; a spike is position bias)")

    summary = {"model": args.model, "variant": args.variant, "modality": MODALITY,
               "option_style": args.option_style,
               "prompt_version": PROMPT_VERSION, "formats": fmts, "n": len(rows),
               "by_track_format": {f"{k[0]}|{k[1]}": v
                                   for k, v in bucket(lambda r: r.get("track")).items()},
               "by_axis_format": {f"{k[0]}|{k[1]}": v
                                  for k, v in bucket(lambda r: r.get("active_axis")).items()}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\n  wrote {res_path}")
    print(f"  wrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())