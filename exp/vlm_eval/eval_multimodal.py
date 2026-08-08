#!/usr/bin/env python3
"""POD-Bench multimodal A/B/C/D evaluation.

WHAT IT SCORES
--------------
Every item shows four garment IMAGES. Exactly one (original key "A") satisfies
both the situation and the user's preferences; the other three fail one or both.
Three numbers come out of the same answer:

    strict      picked A                      both axes right
    tpo         picked A or B                 situation-appropriate
    preference  picked A or C                 preference-matching

FIVE INPUT FORMATS (the ablation)
---------------------------------
    query             situation only, no profile   -> isolates the TPO axis
    narrative         profile only, no query       -> isolates the preference axis
    narrative+query   both
    all               profile only (structured K-V)
    all+query         both

`query` is the shared control of both groups, so the two summary groups are
    [all, all+query, query]  and  [narrative, narrative+query, query].

IMAGES
------
One file per (colour, garment, pattern) CELL, not per plan:

    <images-root>/<colour>/<pattern>_<garment>.jpg

The same cell is reused by ~9 plans, so the base64 encoding is cached per file.

MODEL
-----
Any OpenAI-compatible chat endpoint. A local vLLM needs nothing but a port:

# 터미널 1
Qwen2.5-VL-7B-Instruct

CUDA_VISIBLE_DEVICES=0 vllm serve OpenGVLab/InternVL3_5-8B-Instruct --port 8001 \
  --gpu-memory-utilization 0.90 --max-model-len 16384 \
  --limit-mm-per-prompt '{"image":4}'


    python -m exp.vlm_eval.eval_multimodal --port 8001 \
        --model Qwen/Qwen3-VL-4B-Instruct --concurrency 16

and a hosted model needs a base URL and a key:


    python -m exp.vlm_eval.eval_multimodal \
        --base-url https://api.openai.com/v1 --model gpt5-mini \
        --api-key-env OPENAI_API_KEY --limit 100

Nothing is read from configs/config.py's provider tables; the endpoint is
whatever the flags say.

RESUME
------
Results append to <out>/results.jsonl and finished (plan_id, input_format)
pairs are skipped on the next run, so an interrupted sweep costs nothing.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import json
import mimetypes
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
from configs.scenarios import (EVAL_FRAME_CLAUSE, EVAL_PRIORITY_CLAUSE_SITUATION,
                               EVAL_PRIORITY_CLAUSE_PREFERENCE, PROMPT_VERSION)

INPUT_FORMATS = ["query", "narrative", "narrative+query", "all", "all+query"]
SUMMARY_GROUPS = [["all", "all+query", "query"],
                  ["narrative", "narrative+query", "query"]]

TPO_SCORE        = {"A": 1, "B": 1, "C": 0, "D": 0}
PREFERENCE_SCORE = {"A": 1, "B": 0, "C": 1, "D": 0}
LABELS = ["A", "B", "C", "D"]


# ══ prompts ═══════════════════════════════════════════════════════════════
# Carried over verbatim from scripts/multimodal_eval.py (PROMPT_VERSION 2).
# The clauses live in configs/scenarios.py so the catalog and the prompt can
# never drift apart. A third variant is added for the profile-only formats,
# where there is no situation to reason about.
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
2. Four clothing option images (A, B, C, D)

Your task:
Select the single BEST option image that best fits the query.
{EVAL_FRAME_CLAUSE}
{EVAL_PRIORITY_CLAUSE_SITUATION}
{_OUTPUT_RULE}"""

SYSTEM_PROFILE_ONLY = f"""You are a fashion advisor.

You will be given:
1. A user profile describing clothing preferences
2. Four clothing option images (A, B, C, D)

Your task:
Select the single BEST option image that best matches the user's preferences.
{EVAL_FRAME_CLAUSE}
{_OUTPUT_RULE}"""

SYSTEM_WITH_PROFILE = f"""You are a fashion advisor.

You will be given:
1. A fashion query (situation or occasion)
2. A user profile describing clothing preferences
3. Four clothing option images (A, B, C, D)

Your task:
Select the single BEST option image that best fits both the query and the user's preferences.
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


# ══ profile rendering (mirrors exp/llm_eval/text_eval.py) ═════════════════
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


# ══ images ════════════════════════════════════════════════════════════════
def cell_image_path(attrs, images_root: Path) -> Path | None:
    """<images-root>/<colour>/<pattern>_<garment>.jpg — one file per CELL."""
    c, g, p = (attrs.get("color"), attrs.get("garment_category"), attrs.get("pattern"))
    if not (c and g and p):
        return None
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        cand = images_root / c / f"{p}_{g}{ext}"
        if cand.is_file():
            return cand
    return None


_enc_cache: dict[str, str] = {}
_enc_lock = threading.Lock()


def data_uri(path: Path) -> str:
    """Cached: 1,119 distinct cells back ~10,200 option slots."""
    key = str(path)
    with _enc_lock:
        hit = _enc_cache.get(key)
    if hit:
        return hit
    mime = mimetypes.guess_type(key)[0] or "image/jpeg"
    uri = f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()
    with _enc_lock:
        _enc_cache[key] = uri
    return uri


# ══ message building ══════════════════════════════════════════════════════
# 모든 모델이 같은 문자열을 보도록, system 역할을 쓰지 않고 지시문을
# user 메시지 맨 앞에 합친다. 채팅 템플릿이 모델마다 달라 (Qwen 은 system 을
# 별도 블록으로, Gemma 는 system 을 지원하지 않아 첫 user 턴에 병합) 같은
# messages 를 보내도 최종 시퀀스가 달라지기 때문이다. PersonaVLM(CVPR'26),
# CoViP 등 멀티모달 대화 데이터셋도 같은 방식을 쓴다.
def build_messages(query, profile, shuffled, fmt, images_root):
    # 순서: 프로필 → 질의 → 선택지 → 지시문.
    # eval_dialog.py 의 대화 → 질의 → 선택지 → 지시문과 같은 배열이다.
    # 프로필 전달 형식(서술문 / 키-값 / 대화)만 바뀌고 나머지 구조는 동일해야
    # 조건 간 차이가 배치 차이와 섞이지 않는다. 사용자 정보를 앞에 두면
    # 같은 사용자의 문항들이 접두사를 공유해 캐싱에도 유리하다.
    intro = []
    if fmt in {"narrative", "narrative+query"}:
        intro.append(f"=== USER PROFILE ===\n{profile_to_narrative(profile)}")
    elif fmt in {"all", "all+query"}:
        intro.append(f"=== USER PROFILE ===\n{profile_to_all_kv_text(profile)}")
    if uses_query(fmt):
        pre = "\n" if intro else ""
        intro.append(f"{pre}=== QUERY ===\n"
                     f"{(query or {}).get('query_text','').strip()}")

    intro.append("\n=== OPTIONS ===")
    intro.append("Below are four clothing option images labeled A, B, C, D.")

    content = [{"type": "text",
                "text": system_prompt_for(fmt) + "\n" + "\n".join(intro)}]
    for label, opt in shuffled:
        path = cell_image_path(opt.get("attributes", {}), images_root)
        content.append({"type": "text", "text": f"Option {label}:"})
        if path is None:
            raise FileNotFoundError(f"no image for option {label}: {opt.get('attributes')}")
        content.append({"type": "image_url", "image_url": {"url": data_uri(path)}})

    if fmt == "query":
        task = "Select the single BEST option for the query."
    elif uses_query(fmt):
        task = "Select the single BEST option for both the query and the user's preferences."
    else:
        task = "Select the single BEST option for the user's preferences."

    content.append({"type": "text", "text":
                    "\n=== INSTRUCTION ===\n"
                    f"{task}\n"
                    "Respond with ONE letter only: A, B, C, or D.\n"
                    "Do NOT write any explanation or reasoning.\n"
                    "Do NOT write anything before or after the letter.\n"
                    "Your complete response must be a single character.\n\n"
                    "Answer:"})
    return [{"role": "user", "content": content}]


def parse_answer(text):
    t = (text or "").strip()
    if not t:
        return None
    for ch in t:                      # first standalone letter wins
        if ch.upper() in LABELS:
            return ch.upper()
    return None


# ══ model call ════════════════════════════════════════════════════════════
def call_model(session, url, model, messages, max_tokens, temperature,
               timeout, retries, api_key=None):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    reasoning = model.split("/")[-1].startswith(("gpt-5", "o1", "o3", "o4"))
    payload = {"model": model, "messages": messages}
    if reasoning:
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
            if hasattr(e, "response") and e.response is not None:
                last = RuntimeError(f"{e} :: {e.response.text[:300]}")
                break
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"{type(last).__name__}: {str(last)[:200]}")


# ══ jobs ══════════════════════════════════════════════════════════════════
def missing_option_images(plan, images_root):
    """[(label, attributes)] for the options whose cell has no image file."""
    return [(k, plan["options"][k].get("attributes", {})) for k in LABELS
            if cell_image_path(plan["options"][k].get("attributes", {}),
                               images_root) is None]


def usable(plan, images_root):
    return not missing_option_images(plan, images_root)


def missing_images_error(pid, missing, images_root):
    """A plan without four images is a BUILD defect, not an eval condition:
    silently skipping it changes the item set without changing the reported
    denominator, so every number computed afterwards is over a set nobody
    chose. Fail here and fix the dataset (or opt in with the flag)."""
    cells = "\n".join(
        f"      {k}: {a.get('color')}|{a.get('garment_category')}|{a.get('pattern')}"
        f"  ->  {images_root}/{a.get('color')}/{a.get('pattern')}_{a.get('garment_category')}.jpg"
        for k, a in missing)
    return SystemExit(
        f"[error] plan {pid} has no image for {len(missing)} of its 4 options:\n"
        f"{cells}\n"
        f"    The item set and the image folder disagree. Rebuild the plans "
        f"against the materialized cells\n"
        f"      python -m construction.option_planner --force --cell-library "
        f"annotation/attribute_library.json \\\n"
        f"          --solid-baseline --image-manifest <variant>/images_manifest.json\n"
        f"    or materialize the missing cells "
        f"(python -m scripts.materialize_cells --help).\n"
        f"    To score the image-complete subset anyway, pass "
        f"--allow-missing-images.")


def make_jobs(plans, queries, profiles, fmts, images_root, seed, done,
              orders=None, allow_missing=False):
    """Option order is shuffled ONCE per plan and reused across formats, so a
    format comparison is not also a comparison of different shuffles."""
    jobs, skipped = [], []
    for plan in plans:
        pid = plan.get("plan_id") or plan["query_id"]
        missing = missing_option_images(plan, images_root)
        if missing:
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
    """rows: {(bucket, fmt): {strict,tpo,pref,n}}"""
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
    ap.add_argument("--images-root", type=Path)

    ap.add_argument("--model", required=True,
                    help="e.g. Qwen/Qwen3-VL-4B-Instruct, gpt-4o")
    ap.add_argument("--port", type=int,
                    help="shortcut for --base-url http://127.0.0.1:PORT/v1")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key-env", default=None,
                    help="env var holding the key (hosted APIs only)")

    ap.add_argument("--input-formats", default=",".join(INPUT_FORMATS))
    ap.add_argument("--track", default="both",
                    choices=["both", "physical", "dress_code"])
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
    ap.add_argument("--fresh", action="store_true", help="ignore previous results")
    ap.add_argument("--allow-missing-images", action="store_true",
                    help="score the image-complete subset instead of failing "
                         "when a plan has no image for one of its options. "
                         "Off by default: a silent skip shrinks the item set "
                         "without saying so.")
    args = ap.parse_args()

    data = REPO / f"data_{args.variant}"
    plans_path    = args.plans    or data / "options" / "option_plans.jsonl"
    queries_path  = args.queries  or data / "queries" / "queries.jsonl"
    profiles_path = args.profiles or data / "profiles" / "profiles.jsonl"
    images_root   = args.images_root or data / "images"
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

    # 정답 위치를 고정하거나 무작위로 돌린 결과는 기본(balanced) 결과와
    # 섞이면 안 된다. 같은 (plan_id, format) 인데 배치가 달라 resume 이 이미
    # 끝난 것으로 착각한다. 그래서 경로에 모드를 새긴다.
    _pos = ("" if args.option_order == "balanced"
            else "_" + args.option_order.replace(":", ""))
    out_dir = args.out or (data / ("eval" + _pos) / f"{args.model.replace('/', '_')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "results.jsonl"

    done = set()
    if res_path.exists() and not args.fresh:
        for line in open(res_path):
            try:
                r = json.loads(line)
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

    jobs, skipped = make_jobs(plans, queries, profiles, fmts, images_root,
                              args.seed, done, orders,
                              allow_missing=args.allow_missing_images)

    print("=" * 74)
    print("  POD-Bench multimodal A/B/C/D evaluation")
    print(f"  model      : {args.model}")
    print(f"  endpoint   : {url}")
    print(f"  variant    : {args.variant}   track: {args.track}")
    print(f"  images     : {images_root}")
    print(f"  formats    : {', '.join(fmts)}")
    print(f"  plans      : {len(plans)} loaded, {len(skipped)} without images")
    print(f"  jobs       : {len(jobs)}"
          + (f"  ({len(done)} already done, resuming)" if done else ""))
    print(f"  prompt_ver : {PROMPT_VERSION}")
    print("=" * 74)
    if not jobs:
        print("  nothing to do")
        return 0

    session = requests.Session()
    session.mount("http://", requests.adapters.HTTPAdapter(
        pool_connections=args.concurrency * 2, pool_maxsize=args.concurrency * 2))
    session.mount("https://", requests.adapters.HTTPAdapter(
        pool_connections=args.concurrency * 2, pool_maxsize=args.concurrency * 2))

    write_lock = threading.Lock()
    fh = open(res_path, "a", encoding="utf-8")
    counter = {"n": 0}

    def run(job):
        plan, fmt = job["plan"], job["fmt"]
        raw = err = None
        try:
            msgs = build_messages(job["query"], job["profile"], job["shuffled"],
                                  fmt, images_root)
            raw = call_model(session, url, args.model, msgs, args.max_tokens,
                             args.temperature, args.timeout, args.retries, api_key)
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:200]}"
        disp = parse_answer(raw)
        orig = job["d2o"].get(disp)
        rec = {"plan_id": job["pid"], "query_id": plan["query_id"],
               "user_id": plan["user_id"], "input_format": fmt,
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
               "model": args.model, "prompt_version": PROMPT_VERSION}
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

    # ── summarise every result on disk, not just this run ────────────────
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
    print("  RESULTS")
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

    summary = {"model": args.model, "variant": args.variant,
               "prompt_version": PROMPT_VERSION, "formats": fmts,
               "n": len(rows),
               "by_track_format": {f"{k[0]}|{k[1]}": v
                                   for k, v in bucket(lambda r: r.get("track")).items()},
               "by_axis_format": {f"{k[0]}|{k[1]}": v
                                  for k, v in bucket(lambda r: r.get("active_axis")).items()}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\n  -> {res_path}")
    print(f"  -> {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())