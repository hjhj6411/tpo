#!/usr/bin/env python3
"""POD-Bench A/B/C/D evaluation conditioned on a MULTIMODAL DIALOGUE.

WHAT CHANGES vs eval_multimodal.py
----------------------------------
프로필이 서술문(narrative)이나 키-값(all)이 아니라 **다중 턴 대화**로 주어진다.
그 외 모든 것 — 선택지 4장 이미지, 채점(strict/tpo/preference), plan 당 1회
셔플, (plan_id, input_format) resume, results.jsonl 스키마 — 은 동일하다.
같은 --seed 로 돌리면 선택지 순서까지 같아서 문항 단위 짝비교가 된다.

    dialog        대화만 (상황 없음)  → preference 축을 잰다
    dialog+query  대화 + 상황         → 두 축의 경쟁을 잰다

query-only 는 eval_multimodal.py 에서 이미 측정했으므로 기본 목록에 없다.
필요하면 --input-formats query 로 재현할 수 있다.

DIALOG VARIANT
--------------
    --dialog-variant image   값이 이미지에만 있음 (기본)
    --dialog-variant text    같은 대화, 이미지 자리에 '{blue solid hoodie}'

두 변형은 같은 계획에서 렌더링되어 턴 수·순서·evidence 배치가 동일하다.
따라서 두 실행의 차이가 곧 "이 대화를 이해하는 데 시각 인지가 얼마나
필요한가"이다.

PROMPT ORDER
------------
대화를 query·옵션보다 **앞에** 둔다. 대화는 사용자당 하나이므로 같은 사용자의
문항들이 동일한 접두사를 공유하고, vLLM 접두사 캐싱과 API 캐시 단가가 걸린다.
문항당 이미지가 30장 넘게 늘어난 상황에서 이 순서는 비용 문제다.

IMAGE BUDGET
------------
대화 이미지(~33) + 선택지 4 = 문항당 ~37장. 서빙을 다시 띄워야 한다:

    vllm serve <model> --port 8001 --max-model-len 65536 \
        --limit-mm-per-prompt '{"image":40}'

USAGE
-----
    python -m exp.vlm_eval.eval_dialog --port 8001 \
        --model Qwen/Qwen3-VL-4B-Instruct --concurrency 8
    python -m exp.vlm_eval.eval_dialog --dialog-variant text \
        --base-url https://api.openai.com/v1 --model gpt-5-mini \
        --api-key-env OPENAI_API_KEY
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
from configs.scenarios import (EVAL_FRAME_CLAUSE,                    # noqa: E402
                               EVAL_PRIORITY_CLAUSE_PREFERENCE,
                               EVAL_PRIORITY_CLAUSE_SITUATION,
                               PROMPT_VERSION)

INPUT_FORMATS = ["dialog", "dialog+query"]
ALL_FORMATS = ["query", "dialog", "dialog+query"]
TPO_SCORE = {"A": 1, "B": 1, "C": 0, "D": 0}
PREFERENCE_SCORE = {"A": 1, "B": 0, "C": 1, "D": 0}
LABELS = ["A", "B", "C", "D"]

_enc_cache: dict[str, str] = {}
_enc_lock = threading.Lock()


# ══ prompts ═══════════════════════════════════════════════════════════════
# eval_multimodal.py 와 같은 골격. "user profile" 이 "past conversation" 으로
# 바뀌고, 대화에서 취향을 읽어내라는 한 줄이 추가될 뿐이다. 조항은
# configs/scenarios.py 에서 오므로 카탈로그와 절대 어긋나지 않는다.
_OUTPUT_RULE = """
OUTPUT FORMAT — CRITICAL:
You MUST output EXACTLY one character: A, B, C, or D.
Do NOT output any explanation, reasoning, punctuation, or whitespace.
Do NOT write sentences. Do NOT write words.
If you write anything other than a single letter, your response is INVALID.
Your ENTIRE response must be one of: A  B  C  D
"""

_DIALOG_NOTE = """The conversation is a real past exchange with this user.
Their clothing preferences are never stated as a list — you must infer them
from what they said and from the images they shared."""

SYSTEM_QUERY_ONLY = f"""You are a fashion advisor.

You will be given:
1. A fashion query describing a situation or occasion
2. Four clothing options (A, B, C, D) shown as images

Your task:
Select the single BEST option that best fits the query.
{EVAL_FRAME_CLAUSE}
{EVAL_PRIORITY_CLAUSE_SITUATION}
{_OUTPUT_RULE}"""

SYSTEM_DIALOG_ONLY = f"""You are a fashion advisor.

You will be given:
1. A past conversation with the user
2. Four clothing options (A, B, C, D) shown as images

Your task:
Select the single BEST option that best matches the user's preferences.
{_DIALOG_NOTE}
{EVAL_FRAME_CLAUSE}
{_OUTPUT_RULE}"""

SYSTEM_DIALOG_QUERY = f"""You are a fashion advisor.

You will be given:
1. A past conversation with the user
2. A fashion query (situation or occasion)
3. Four clothing options (A, B, C, D) shown as images

Your task:
Select the single BEST option that best fits both the query and the user's
preferences.
{_DIALOG_NOTE}
{EVAL_FRAME_CLAUSE}
{EVAL_PRIORITY_CLAUSE_SITUATION}
{EVAL_PRIORITY_CLAUSE_PREFERENCE}
{_OUTPUT_RULE}"""


def uses_query(fmt):   return fmt in {"query", "dialog+query"}
def uses_dialog(fmt):  return fmt in {"dialog", "dialog+query"}


def system_prompt_for(fmt):
    if not uses_dialog(fmt):
        return SYSTEM_QUERY_ONLY
    return SYSTEM_DIALOG_QUERY if uses_query(fmt) else SYSTEM_DIALOG_ONLY


# ══ images ════════════════════════════════════════════════════════════════
def cell_image_path(attrs, images_root: Path) -> Path | None:
    c, g, p = attrs.get("color"), attrs.get("garment_category"), attrs.get("pattern")
    if not (c and g and p):
        return None
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        cand = images_root / c / f"{p}_{g}{ext}"
        if cand.is_file():
            return cand
    return None


def dialog_image_path(rel: str, images_root: Path) -> Path | None:
    p = images_root / rel
    if p.is_file():
        return p
    stem = p.with_suffix("")
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        if stem.with_suffix(ext).is_file():
            return stem.with_suffix(ext)
    return None


def data_uri(path: Path) -> str:
    """캐시 필수. 대화 이미지는 문항마다 다시 등장하므로 (사용자당 33장 x
    100여 문항) 캐시가 없으면 인코딩이 실행 시간을 지배한다."""
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
def dialog_blocks(dialog, images_root: Path, variant: str):
    """대화를 content 블록 목록으로. 이미지 변형이면 턴 안에 이미지를 끼운다."""
    blocks = [{"type": "text", "text": "=== CONVERSATION HISTORY ==="}]
    for turn in dialog["turns"]:
        who = "User" if turn["speaker"] == "user" else "Assistant"
        blocks.append({"type": "text",
                       "text": f"[{turn['time']}] {who}: {turn['text']}"})
        for rel in turn["images"]:
            path = dialog_image_path(rel, images_root)
            if path is None:
                raise FileNotFoundError(f"dialog image missing: {rel}")
            blocks.append({"type": "image_url",
                           "image_url": {"url": data_uri(path)}})
    return blocks


def build_messages(query, dialog, shuffled, fmt, images_root, variant):
    content = []
    # 대화를 맨 앞에 — 같은 사용자의 문항들이 접두사를 공유해 캐싱이 걸린다
    if uses_dialog(fmt):
        content += dialog_blocks(dialog, images_root, variant)
    if uses_query(fmt):
        qt = (query or {}).get("query_text", "").strip()
        content.append({"type": "text", "text": f"\n=== QUERY ===\n{qt}"})

    content.append({"type": "text", "text":
                    "\n=== OPTIONS ===\n"
                    "Below are four clothing option images labeled A, B, C, D."})
    for label, opt in shuffled:
        path = cell_image_path(opt.get("attributes", {}), images_root)
        if path is None:
            raise FileNotFoundError(f"no image for option {label}")
        content.append({"type": "text", "text": f"Option {label}:"})
        content.append({"type": "image_url", "image_url": {"url": data_uri(path)}})

    if fmt == "query":
        task = "Select the single BEST option for the query."
    elif uses_query(fmt):
        task = ("Select the single BEST option for both the query and the "
                "user's preferences as shown in the conversation.")
    else:
        task = ("Select the single BEST option for the user's preferences as "
                "shown in the conversation.")

    content.append({"type": "text", "text":
                    "\n=== INSTRUCTION ===\n"
                    f"{task}\n"
                    "Respond with ONE letter only: A, B, C, or D.\n"
                    "Do NOT write any explanation or reasoning.\n"
                    "Do NOT write anything before or after the letter.\n"
                    "Your complete response must be a single character.\n\n"
                    "Answer:"})
    return [{"role": "system", "content": system_prompt_for(fmt)},
            {"role": "user", "content": content}]


def parse_answer(text):
    for ch in (text or "").strip():
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
                last = RuntimeError(f"{e} :: {resp.text[:300]}")
                if resp.status_code in (400, 404, 422):
                    break
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"{type(last).__name__}: {str(last)[:300]}")


# ══ evidence linkage ══════════════════════════════════════════════════════
def dialog_cells(dialog):
    """대화에 등장한 셀 집합. 선택지와의 재등장을 표시하기 위한 것."""
    return {f'{a["color"]}|{a["garment_category"]}|{a["pattern"]}'
            for e in dialog["evidence"] for a in e["image_attributes"]}


def recurrence(plan, cells):
    """대화에서 본 그 이미지가 선택지로 다시 나오는가.

    전체 슬롯의 7%, 문항의 25%에서 발생하며 A/B/C/D 에 고르게 퍼져 있어
    정답 쪽으로 치우치지 않는다. 다만 취향 축은 추론 대신 회상으로 풀릴 수
    있으므로, 숨은 교란요인으로 두지 말고 플래그로 남겨 분해 집계한다.
    """
    labs = [lab for lab, o in plan["options"].items()
            if f'{o["attributes"].get("color")}|'
               f'{o["attributes"].get("garment_category")}|'
               f'{o["attributes"].get("pattern")}' in cells]
    return {"recurs": bool(labs), "recur_labels": "".join(sorted(labs))}


def evidence_index(dialog):
    """(axis, value) -> evidence. 문항의 활성축 값이 대화에서 어떤 방식으로
    노출됐는지 붙이기 위한 것. 방식별 정확도 분해의 열쇠다."""
    return {(e["axis"], e["value"]): e for e in dialog["evidence"]}


def item_evidence(plan, ev_idx):
    """활성축의 선호값(A)과 비선호값(B)이 각각 어떤 방식으로 노출됐는가."""
    axis = plan.get("active_axis")
    opts = plan["options"]
    pref = opts["A"]["attributes"].get(axis)
    disp = opts["B"]["attributes"].get(axis)
    e_pref = ev_idx.get((axis, pref))
    e_disp = ev_idx.get((axis, disp))
    return {
        "pref_value": pref, "dispref_value": disp,
        "pref_expression": (e_pref or {}).get("expression_type"),
        "dispref_expression": (e_disp or {}).get("expression_type"),
        "pref_turn": (e_pref or {}).get("revealed_in_turn"),
        "pref_exposed": e_pref is not None,
        "dispref_exposed": e_disp is not None,
    }


# ══ jobs ══════════════════════════════════════════════════════════════════
def usable(plan, images_root):
    return all(cell_image_path(plan["options"][k].get("attributes", {}),
                               images_root) for k in LABELS)


def make_jobs(plans, queries, dialogs, fmts, images_root, seed, done):
    """셔플은 plan 당 1회. eval_multimodal.py 와 같은 키라 이미지 실행과
    선택지 순서가 일치한다."""
    jobs, skipped = [], []
    for plan in plans:
        pid = plan.get("plan_id") or plan["query_id"]
        dialog = dialogs.get(plan["user_id"])
        if dialog is None:
            skipped.append((pid, "no dialog")); continue
        if not usable(plan, images_root):
            skipped.append((pid, "missing option image")); continue
        rng = random.Random(f"{seed}|{pid}")
        items = list(plan["options"].items())
        rng.shuffle(items)
        shuffled = [(d, o) for d, (k, o) in zip(LABELS, items)]
        d2o = {d: k for d, (k, _) in zip(LABELS, items)}
        correct = next(d for d, k in d2o.items() if k == "A")
        ev = item_evidence(plan, evidence_index(dialog))
        ev.update(recurrence(plan, dialog_cells(dialog)))
        for fmt in fmts:
            if (pid, fmt) in done:
                continue
            jobs.append({"plan": plan, "pid": pid, "fmt": fmt,
                         "query": queries.get(plan["query_id"]),
                         "dialog": dialog, "shuffled": shuffled,
                         "d2o": d2o, "correct": correct, "ev": ev})
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
    print(f"  {'':<{w + 16}}{'n':>6}{'strict':>9}{'tpo':>9}{'preference':>12}")
    print("  " + "-" * (w + 52))
    for b in buckets:
        for fmt in fmts:
            d = rows.get((b, fmt))
            if not d:
                continue
            print(f"  {str(b) + '/' + fmt:<{w + 16}}{d['n']:>6}"
                  f"{pct(d['strict'], d['n']):>9}{pct(d['tpo'], d['n']):>9}"
                  f"{pct(d['pref'], d['n']):>12}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="wacv_scenario_v5")
    ap.add_argument("--plans", type=Path)
    ap.add_argument("--queries", type=Path)
    ap.add_argument("--dialogs", type=Path)
    ap.add_argument("--images-root", type=Path)
    ap.add_argument("--dialog-variant", default="image", choices=["image", "text"])

    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key-env", default=None)

    ap.add_argument("--input-formats", default=",".join(INPUT_FORMATS))
    ap.add_argument("--track", default="both",
                    choices=["both", "physical", "dress_code"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42,
                    help="eval_multimodal.py 와 같은 값이어야 셔플이 일치한다")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    data = REPO / f"data_{args.variant}"
    plans_path = args.plans or data / "options" / "option_plans.jsonl"
    queries_path = args.queries or data / "queries" / "queries.jsonl"
    dialogs_dir = args.dialogs or data / "dialogs"
    images_root = args.images_root or data / "images"
    base_url = args.base_url or (f"http://127.0.0.1:{args.port}/v1" if args.port
                                 else "http://127.0.0.1:8001/v1")
    url = base_url.rstrip("/") + "/chat/completions"
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if args.api_key_env and not api_key:
        raise SystemExit(f"[error] ${args.api_key_env} is empty")

    fmts = [f.strip() for f in args.input_formats.split(",") if f.strip()]
    for f in fmts:
        if f not in ALL_FORMATS:
            raise SystemExit(f"[error] unknown format {f!r}; have {ALL_FORMATS}")

    load = lambda p: [json.loads(l) for l in open(p) if l.strip()]
    plans = load(plans_path)
    queries = {q["query_id"]: q for q in load(queries_path)}
    dialogs = {}
    for p in sorted(dialogs_dir.glob(f"dlg__*__{args.dialog_variant}.json")):
        d = json.loads(p.read_text())
        dialogs[d["user_id"]] = d
    if not dialogs:
        raise SystemExit(f"[error] no {args.dialog_variant} dialogs in {dialogs_dir}")

    if args.track != "both":
        plans = [p for p in plans if p.get("track") == args.track]
    if args.limit > 0:
        plans = plans[:args.limit]

    tag = f"{args.model.replace('/', '_')}__{args.dialog_variant}"
    out_dir = args.out or (data / "eval_dialog" / tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "results.jsonl"

    done = set()
    if res_path.exists() and not args.fresh:
        for line in open(res_path):
            try:
                r = json.loads(line)
                if not r.get("error"):
                    done.add((r["plan_id"], r["input_format"]))
            except Exception:
                pass
    elif args.fresh and res_path.exists():
        res_path.unlink()

    jobs, skipped = make_jobs(plans, queries, dialogs, fmts, images_root,
                              args.seed, done)
    n_dlg_img = (0 if args.dialog_variant == "text" else
                 int(sum(sum(len(t["images"]) for t in d["turns"])
                         for d in dialogs.values()) / len(dialogs)))

    print("=" * 74)
    print("  POD-Bench DIALOG-CONDITIONED A/B/C/D evaluation")
    print(f"  model      : {args.model}")
    print(f"  endpoint   : {url}")
    print(f"  variant    : {args.variant}   track: {args.track}")
    print(f"  dialogs    : {len(dialogs)} users, {args.dialog_variant} variant, "
          f"~{n_dlg_img} images each")
    print(f"  images/req : ~{n_dlg_img + 4}  ← --limit-mm-per-prompt 확인")
    print(f"  formats    : {', '.join(fmts)}")
    print(f"  plans      : {len(plans)} loaded, {len(skipped)} skipped")
    print(f"  jobs       : {len(jobs)}"
          + (f"  ({len(done)} already done, resuming)" if done else ""))
    print(f"  prompt_ver : {PROMPT_VERSION}")
    print("=" * 74)
    if skipped:
        why = Counter(w for _, w in skipped)
        print("  skipped:", dict(why))
    if not jobs:
        print("  nothing to do")
        return 0

    session = requests.Session()
    for scheme in ("http://", "https://"):
        session.mount(scheme, requests.adapters.HTTPAdapter(
            pool_connections=args.concurrency * 2,
            pool_maxsize=args.concurrency * 2))

    write_lock = threading.Lock()
    fh = open(res_path, "a", encoding="utf-8")
    counter = {"n": 0}

    def run(job):
        plan, fmt = job["plan"], job["fmt"]
        raw = err = None
        try:
            msgs = build_messages(job["query"], job["dialog"], job["shuffled"],
                                  fmt, images_root, args.dialog_variant)
            raw = call_model(session, url, args.model, msgs, args.max_tokens,
                             args.temperature, args.timeout, args.retries, api_key)
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:300]}"
        disp = parse_answer(raw)
        orig = job["d2o"].get(disp)
        rec = {"plan_id": job["pid"], "query_id": plan["query_id"],
               "user_id": plan["user_id"], "input_format": fmt,
               "modality": args.dialog_variant, "profile_source": "dialog",
               "dialog_id": job["dialog"]["dialog_id"],
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
               **{f"evidence_{k}": v for k, v in job["ev"].items()}}
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
    print(f"  RESULTS (dialog / {args.dialog_variant})")
    print("=" * 74)
    print_block(bucket(lambda r: "ALL"), "overall", fmts)
    print_block(bucket(lambda r: r.get("track") or "?"),
                "by track (independent datasets — do NOT pool)", fmts)
    print_block(bucket(lambda r: f"act:{r.get('active_axis')}"),
                "by ACTIVE axis", fmts)
    print_block(bucket(lambda r: f"expr:{r.get('evidence_pref_expression')}"),
                "by EXPRESSION TYPE of the preferred value in the dialogue",
                fmts)
    print_block(bucket(lambda r: f"vio:{r.get('violation_axis')}"),
                "by VIOLATION axis", fmts)
    print_block(bucket(lambda r: f"recur:{bool(r.get('evidence_recurs'))}"),
                "by IMAGE RECURRENCE (a dialogue image reappears as an option)",
                fmts)

    errs = sum(1 for r in rows if r.get("error"))
    unp = sum(1 for r in rows if not r.get("error") and r["predicted_display"] is None)
    print(f"\n  request errors {errs}   unparseable answers {unp}")
    pos = Counter(r["predicted_display"] for r in rows if r["predicted_display"])
    print(f"  answer position histogram: {dict(sorted(pos.items()))}")
    unexposed = sum(1 for r in rows if not r.get("evidence_pref_exposed"))
    if unexposed:
        print(f"  ⚠ {unexposed} rows whose preferred value was NOT in the dialogue")

    summary = {"model": args.model, "variant": args.variant,
               "profile_source": "dialog", "dialog_variant": args.dialog_variant,
               "prompt_version": PROMPT_VERSION, "formats": fmts, "n": len(rows),
               "by_track_format": {f"{k[0]}|{k[1]}": v for k, v
                                   in bucket(lambda r: r.get("track")).items()},
               "by_expression_format": {
                   f"{k[0]}|{k[1]}": v for k, v
                   in bucket(lambda r: r.get("evidence_pref_expression")).items()}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\n  wrote {res_path}")
    print(f"  wrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())