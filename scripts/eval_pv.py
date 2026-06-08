#!/usr/bin/env python3
"""
eval_vlm_judge_r5.py  —  VLM-as-judge with a detailed 1-5 per-axis rubric
-------------------------------------------------------------------------
Stricter successor to eval_vlm_judge.py. Instead of one coarse 0-3 score, the
judge grades THREE axes independently on a 1-5 scale with behavioral anchors
(Prometheus-style rubric), which curbs the "everything looks ~3, give it a 4"
leniency. Code combines them; pattern coverage is baked into the pattern score.

Per option the judge returns:
  garment 1-5, color 1-5, pattern 1-5   (each with explicit anchors)
  pattern_full : yes/no/na   (yes only if pattern score >= 4; na if target solid)
  set_ok       : yes/no      (no if child model / uninterpretable crop)
  notes        : <=10 words

gpt-5 fix: reasoning models spend the token budget on hidden reasoning first, so
a small max_completion_tokens yields an EMPTY content (finish_reason="length").
We set max_completion_tokens=512 and reasoning_effort="minimal" (this is a narrow
perceptual judgment; minimal effort is appropriate and keeps cost/latency low).

Backends: gpt5_nano (OpenAI gpt-5-nano, vision) or vllm (local OpenAI-compatible).

Usage:
  export OPENAI_API_KEY=sk-...
  python scripts/eval_vlm_judge_r5.py \
    --logs dpv:data/images_dpv/collection_log.jsonl,off:data/images_siglip_top1/collection_log.jsonl \
    --plans data/options/option_plans.jsonl \
    --provider gpt5_nano --n 300 --oversample-pattern \
    --out data/eval/judge_r5.jsonl --report data/eval/judge_r5_report.json --force
"""
import argparse
import base64
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter

_SESSION = requests.Session()
_SESSION.mount("https://", HTTPAdapter(pool_connections=16, pool_maxsize=32, max_retries=0))
_SESSION.mount("http://", HTTPAdapter(pool_connections=16, pool_maxsize=32, max_retries=0))

_SOLID_LIKE = {"", "solid", "plain", "none"}

JUDGE_SYSTEM = """You are a STRICT fashion product-image grader for an academic benchmark.
You are given ONE image and the TARGET attributes it was supposed to depict. Judge the
MAIN garment only; ignore background, accessories, and any secondary draped/tied garment.
Grade THREE axes independently on a 1-5 scale using the exact anchors below. Do NOT be
lenient: if an axis is ambiguous or you are unsure, give the LOWER score.

Return ONLY compact JSON:
{"garment":1-5,"color":1-5,"pattern":1-5,"pattern_full":"yes"|"no"|"na","set_ok":"yes"|"no","notes":"<=10 words"}

garment (does the main garment TYPE match the target type?):
  5 = exactly the target garment type.
  4 = same functional category, minor sub-type difference (e.g. coat vs trench coat).
  3 = related but clearly different cut/type.
  2 = wrong garment but same broad area (a top when a top was wanted, etc.).
  1 = different garment entirely, or no identifiable garment.

color (does the dominant BODY color match the target color?):
  5 = the target color clearly dominates the garment body.
  4 = target color present and dominant but the shade is slightly off.
  3 = target color is only one of several, not clearly dominant.
  2 = a neighboring/confusable color dominates (e.g. navy vs blue, beige vs white).
  1 = a clearly different color dominates.

pattern (does the target PATTERN appear AND cover the body? judge the BODY only,
EXCLUDING sleeves/collar/cuffs/trim/hem/pocket):
  if the target pattern is solid/plain:
    5 = body is plain solid; 3 = minor unwanted texture; 1 = an unwanted pattern covers the body.
  if the target pattern is non-solid (plaid/striped/floral/polka_dot/checkered/
  graphic_print/camouflage/animal_print):
    5 = the target pattern, covering ALMOST THE ENTIRE body (upper AND lower torso).
    4 = the target pattern over a clear majority of the body.
    3 = the target pattern on only about half the body, OR the right pattern family but
        coverage is ambiguous.
    2 = pattern only LOCALIZED (sleeves / collar / trim / one panel / only-upper or
        only-lower), OR a different but related pattern.
    1 = no such pattern on the body, or a clearly wrong pattern.

pattern_full = "na" if the target is solid; otherwise "yes" ONLY if the pattern score is
>= 4 (covers most/all of the body), else "no". Be strict: when in doubt, "no".
set_ok = "no" if the model is a child/baby, or the garment is so cropped/draped that you
cannot tell what it is; otherwise "yes".
A repeating heart / star / character / novelty / logo / text motif counts as graphic_print,
NOT polka_dot and NOT checkered."""


def load_jsonl(path):
    out = []
    p = Path(path)
    if not p.exists():
        return out
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def parse_logs_arg(s):
    out = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk and not chunk.split(":", 1)[1].startswith("//"):
            name, path = chunk.split(":", 1)
        else:
            path = chunk
            name = Path(path).parent.name or "default"
        out.append((name.strip(), path.strip()))
    return out


def b64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def target_text(attrs):
    g = (attrs.get("garment_category") or "").replace("_", " ")
    c = attrs.get("color") or ""
    p = (attrs.get("pattern") or "solid").replace("_", " ")
    return f"garment={g}; color={c}; pattern={p}"


def _as_int_1_5(v):
    try:
        n = int(round(float(v)))
        return min(5, max(1, n))
    except Exception:
        return None


def call_judge(provider, api_base, model, image_path, attrs, api_key=None,
               timeout=90, retries=3):
    user_text = (f"TARGET attributes: {target_text(attrs)}\n"
                 f"Grade the three axes with the rubric. Return the JSON.")
    content = [
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{b64_image(image_path)}"}},
        {"type": "text", "text": user_text},
    ]
    payload = {"model": model,
               "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                            {"role": "user", "content": content}]}
    if provider == "gpt5_nano":
        payload["max_completion_tokens"] = 512        # room for hidden reasoning + JSON
        payload["reasoning_effort"] = "minimal"       # narrow judgment; minimal is enough
    else:
        payload["max_tokens"] = 200
        payload["temperature"] = 0.0
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = f"{api_base.rstrip('/')}/chat/completions"

    last = "unparseable"
    for attempt in range(retries):
        try:
            r = _SESSION.post(url, json=payload, headers=headers, timeout=timeout)
            if r.status_code != 200:
                if 400 <= r.status_code < 500 and r.status_code != 429:
                    return {"_error": f"HTTP {r.status_code}: {r.text[:160]}"}
                last = f"HTTP {r.status_code}"
                time.sleep(1.5 * (attempt + 1)); continue
            data = r.json()
            msg = data["choices"][0].get("message", {})
            txt = (msg.get("content") or "")
            fr = data["choices"][0].get("finish_reason")
            usage = data.get("usage", {})
            if "</think>" in txt:
                txt = txt.split("</think>", 1)[1]
            s, e = txt.find("{"), txt.rfind("}")
            if s >= 0 and e > s:
                d = json.loads(txt[s:e + 1])
                d["_usage"] = {"in": usage.get("prompt_tokens", 0),
                               "out": usage.get("completion_tokens", 0)}
                return d
            last = f"empty_content(finish={fr})"
            time.sleep(1.0 * (attempt + 1))
        except Exception as ex:
            last = str(ex)[:160]
            time.sleep(1.5 * (attempt + 1))
    return {"_error": last}


def collect_jobs(logs, plans_by_qid, done_keys):
    jobs = []
    for method, path in logs:
        for rec in load_jsonl(path):
            qid = rec.get("query_id"); axis = rec.get("active_axis")
            plan = plans_by_qid.get(qid)
            if not plan:
                continue
            for k in "ABCD":
                o = (rec.get("options") or {}).get(k) or {}
                ip = o.get("image_path")
                if not ip:
                    continue
                key = f"{method}|{qid}|{k}"
                if key in done_keys:
                    continue
                attrs = ((plan.get("options") or {}).get(k) or {}).get("attributes", {}) or {}
                tp = attrs.get("pattern")
                jobs.append({"method": method, "query_id": qid, "option": k,
                             "active_axis": axis, "image_path": ip, "attrs": attrs,
                             "pattern_nonsolid": (axis == "pattern"
                                                  and str(tp).strip().lower() not in _SOLID_LIKE)})
    return jobs


def sample_jobs(jobs, n, oversample_pattern, pattern_frac, seed):
    if n <= 0 or n >= len(jobs):
        return jobs
    rng = random.Random(seed)
    if oversample_pattern:
        pat = [j for j in jobs if j["pattern_nonsolid"]]
        oth = [j for j in jobs if not j["pattern_nonsolid"]]
        rng.shuffle(pat); rng.shuffle(oth)
        n_pat = min(len(pat), int(round(n * pattern_frac)))
        chosen = pat[:n_pat] + oth[:max(0, n - n_pat)]
    else:
        chosen = jobs[:]; rng.shuffle(chosen); chosen = chosen[:n]
    rng.shuffle(chosen)
    return chosen


def run(args):
    logs = parse_logs_arg(args.logs)
    plans_by_qid = {p["query_id"]: p for p in load_jsonl(args.plans)}

    if args.provider == "gpt5_nano":
        api_base = "https://api.openai.com/v1"
        model = args.vlm_model or "gpt-5-nano"
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY not set")
    else:
        api_base = args.vlm_url
        model = args.vlm_model or "Qwen/Qwen3-VL-30B-A3B-Instruct"
        api_key = None

    out = Path(args.out)
    existing = load_jsonl(out)
    # only treat SUCCESSFUL rows as done -> failed rows auto-retry on rerun
    done = {f"{r['method']}|{r['query_id']}|{r['option']}" for r in existing
            if isinstance(r.get("judge"), dict) and "_error" not in r["judge"]
            and "garment" in r["judge"]}
    jobs = collect_jobs(logs, plans_by_qid, done)
    jobs = sample_jobs(jobs, args.n, args.oversample_pattern, args.pattern_frac, args.seed)
    print(f"  methods={[m for m,_ in logs]}  to judge: {len(jobs)}  "
          f"(already done {len(done)})  provider={args.provider} model={model}")

    # keep only successful prior rows in the output (drop stale error rows)
    results = [r for r in existing if f"{r['method']}|{r['query_id']}|{r['option']}" in done]
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    lock = threading.Lock()
    tok_in = tok_out = 0

    def worker(job):
        d = call_judge(args.provider, api_base, model, job["image_path"],
                       job["attrs"], api_key=api_key)
        return job, d

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(worker, j) for j in jobs]
        n = 0
        for fut in as_completed(futs):
            job, d = fut.result()
            with lock:
                u = d.pop("_usage", {}) if isinstance(d, dict) else {}
                tok_in += u.get("in", 0); tok_out += u.get("out", 0)
                results.append({"method": job["method"], "query_id": job["query_id"],
                                "option": job["option"], "active_axis": job["active_axis"],
                                "pattern_nonsolid": job["pattern_nonsolid"],
                                "target": target_text(job["attrs"]), "judge": d})
                n += 1
                if n % 25 == 0:
                    g = d.get("garment", "?"); c = d.get("color", "?"); p = d.get("pattern", "?")
                    print(f"  [{n}/{len(jobs)}] last g/c/p={g}/{c}/{p} pf={d.get('pattern_full','?')}")
                if n % 50 == 0:
                    _save(results, out)
    _save(results, out)
    report(results, tok_in, tok_out, args.provider, args.report)


def _save(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def report(results, tok_in, tok_out, provider, report_path):
    by = defaultdict(lambda: {"n": 0, "err": 0,
                              "g_sum": 0, "c_sum": 0, "p_sum": 0, "axis_n": 0,
                              "overall_sum": 0.0,
                              "pf_yes": 0, "pf_n": 0, "setbad": 0})

    def bucket(key, r):
        b = by[key]; b["n"] += 1
        j = r["judge"]
        if not isinstance(j, dict) or "_error" in j or "garment" not in j:
            b["err"] += 1; return
        g = _as_int_1_5(j.get("garment")); c = _as_int_1_5(j.get("color")); p = _as_int_1_5(j.get("pattern"))
        if None in (g, c, p):
            b["err"] += 1; return
        b["axis_n"] += 1
        b["g_sum"] += g; b["c_sum"] += c; b["p_sum"] += p
        b["overall_sum"] += (g + c + p) / 3.0
        if str(j.get("set_ok")).lower() == "no":
            b["setbad"] += 1
        if r["pattern_nonsolid"] and str(j.get("pattern_full")).lower() in ("yes", "no"):
            b["pf_n"] += 1
            if str(j.get("pattern_full")).lower() == "yes":
                b["pf_yes"] += 1

    for r in results:
        bucket(f"method:{r['method']}", r)
        bucket(f"axis:{r['active_axis']}", r)
        bucket("ALL", r)

    def avg(s, n):
        return (s / n) if n else float("nan")

    def line(key):
        b = by[key]; an = b["axis_n"]
        g = avg(b["g_sum"], an); c = avg(b["c_sum"], an); p = avg(b["p_sum"], an)
        ov = avg(b["overall_sum"], an)
        pf = avg(b["pf_yes"], b["pf_n"])
        pfs = "  n/a" if pf != pf else f"{pf*100:5.1f}%"
        return (f"  {key:20s} n={b['n']:4d} ok={an:4d} err={b['err']:3d} | "
                f"garment={g:4.2f} color={c:4.2f} pattern={p:4.2f} overall={ov:4.2f}/5 | "
                f"pattern_full={pfs}(pf_n={b['pf_n']}) set_bad={b['setbad']}")

    print("\n" + "=" * 104)
    print("  VLM-AS-JUDGE (1-5 rubric)   per-axis means + pattern_full accuracy")
    print("=" * 104)
    for key in [k for k in by if k.startswith("method:")] + \
               [k for k in by if k.startswith("axis:")] + ["ALL"]:
        print(line(key))

    cost = None
    if provider == "gpt5_nano" and (tok_in or tok_out):
        cost = tok_in / 1e6 * 0.05 + tok_out / 1e6 * 0.40
        print(f"\n  tokens in={tok_in:,} out={tok_out:,}  est_cost=${cost:.3f} (gpt-5-nano)")

    rep = {"by": {k: dict(v) for k, v in by.items()},
           "tokens": {"in": tok_in, "out": tok_out}, "est_cost_usd": cost}
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  -> {report_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--plans", type=Path, default=Path("data/options/option_plans.jsonl"))
    ap.add_argument("--provider", choices=["gpt5_nano", "vllm"], default="gpt5_nano")
    ap.add_argument("--vlm-url", default="http://127.0.0.1:8002/v1")
    ap.add_argument("--vlm-model", default=None)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--oversample-pattern", action="store_true")
    ap.add_argument("--pattern-frac", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("data/eval/judge_r5.jsonl"))
    ap.add_argument("--report", type=Path, default=Path("data/eval/judge_r5_report.json"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.force and Path(args.out).exists():
        Path(args.out).unlink()
    run(args)


if __name__ == "__main__":
    main()