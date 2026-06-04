#!/usr/bin/env python3
"""
eval_vlm_judge.py  —  simple VLM-as-judge for collection quality
-----------------------------------------------------------------
The lean alternative to the 3-layer eval. For each collected option image, show
the judge VLM the image AND the plan's target attributes, and ask it to rate how
well the image matches the target. No extraction step, no rule matching, no
canonicalization — the model judges directly.

Per option it returns:
  match    : 0..3 integer  (0 wrong, 1 weak, 2 mostly, 3 perfect match of
             garment+color+pattern to the target)
  pattern_full : "yes"/"no"/"na"  (na unless target pattern is non-solid) —
             does the target pattern cover the WHOLE garment body, not just sleeves/trim
  notes    : one short phrase (why, if not perfect)

Sampling: by default judges a random ~N images across one or more collection logs
(stratified to over-sample pattern-axis non-solid options, which carry the key
claim). Resumable. Reports mean match score and pattern_full accuracy per method
and per active axis, plus a rough $ cost estimate.

Judge backends (pick with --provider):
  gpt5_nano : OpenAI gpt-5-nano (vision, ~$0.05/M in, $0.40/M out — 500 imgs < $1).
              needs OPENAI_API_KEY. Uses max_completion_tokens, no temperature.
  vllm      : any local OpenAI-compatible endpoint (--vlm-url, --vlm-model).

Usage:
  export OPENAI_API_KEY=sk-...
  python src/eval_vlm_judge.py \
    --logs dpv:data/images_dpv/collection_log.jsonl,off:data/images_siglip_top1/collection_log.jsonl \
    --plans data/options/option_plans.jsonl \
    --provider gpt5_nano --n 500 --oversample-pattern \
    --out data/eval/judge.jsonl --report data/eval/judge_report.json

  # local model instead:
  python src/eval_vlm_judge.py --logs dpv:data/images_dpv/collection_log.jsonl \
    --plans data/options/option_plans.jsonl \
    --provider vllm --vlm-url http://127.0.0.1:8002/v1 --vlm-model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --n 500 --out data/eval/judge.jsonl --report data/eval/judge_report.json
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

JUDGE_SYSTEM = (
    "You are a strict fashion product-image grader for a benchmark. You are given "
    "ONE image and the TARGET attributes the image was supposed to depict. Judge "
    "the MAIN garment only. Ignore background and accessories.\n\n"
    "Return ONLY compact JSON:\n"
    '{"match": 0|1|2|3, "pattern_full": "yes"|"no"|"na", "notes": "<=8 words"}\n\n'
    "match = how well the image matches ALL THREE target attributes (garment type, "
    "color, pattern):\n"
    "  3 = perfect: garment, color, and pattern all clearly match the target.\n"
    "  2 = mostly: two of the three match; one is slightly off.\n"
    "  1 = weak: only one attribute matches, or the garment is hard to see / draped / "
    "the image is badly framed.\n"
    "  0 = wrong: the main garment does not match the target garment, or it is a "
    "different gender/child model that breaks the set, or no garment is visible.\n"
    "pattern_full (judge the BODY only, EXCLUDING sleeves/collar/cuffs/trim/hem):\n"
    "  if target pattern is solid/none -> \"na\".\n"
    "  else \"yes\" only if the target pattern covers ALMOST THE ENTIRE body "
    "(upper+lower torso); \"no\" if it is only on sleeves/trim/one panel/half/localized. "
    "Be strict: when in doubt, \"no\"."
)


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


def call_judge(provider, api_base, model, image_path, attrs, api_key=None,
               timeout=90, retries=3):
    """Return parsed judge dict, or {} on failure."""
    user_text = (f"TARGET attributes: {target_text(attrs)}\n"
                 f"Grade how well the image matches the target. Return the JSON.")
    content = [
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{b64_image(image_path)}"}},
        {"type": "text", "text": user_text},
    ]
    payload = {"model": model,
               "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                            {"role": "user", "content": content}]}
    if provider == "gpt5_nano":
        payload["max_completion_tokens"] = 60        # gpt-5 family: no temperature, use this
    else:
        payload["max_tokens"] = 60
        payload["temperature"] = 0.0
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = f"{api_base.rstrip('/')}/chat/completions"

    for attempt in range(retries):
        try:
            r = _SESSION.post(url, json=payload, headers=headers, timeout=timeout)
            if r.status_code != 200:
                if 400 <= r.status_code < 500 and r.status_code != 429:
                    return {"_error": f"HTTP {r.status_code}: {r.text[:160]}"}
                time.sleep(1.5 * (attempt + 1)); continue
            data = r.json()
            txt = (data["choices"][0]["message"].get("content") or "")
            usage = data.get("usage", {})
            if "</think>" in txt:
                txt = txt.split("</think>", 1)[1]
            s, e = txt.find("{"), txt.rfind("}")
            if s >= 0 and e > s:
                d = json.loads(txt[s:e + 1])
                d["_usage"] = {"in": usage.get("prompt_tokens", 0),
                               "out": usage.get("completion_tokens", 0)}
                return d
        except Exception as ex:
            time.sleep(1.5 * (attempt + 1))
            last = str(ex)[:160]
    return {"_error": locals().get("last", "unparseable")}


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
        chosen = jobs[:]
        rng.shuffle(chosen)
        chosen = chosen[:n]
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
    done = {f"{r['method']}|{r['query_id']}|{r['option']}" for r in existing}
    jobs = collect_jobs(logs, plans_by_qid, done)
    jobs = sample_jobs(jobs, args.n, args.oversample_pattern, args.pattern_frac, args.seed)
    print(f"  methods={[m for m,_ in logs]}  to judge: {len(jobs)}  "
          f"(already done {len(done)})  provider={args.provider} model={model}")

    results = existing
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
                                "target": target_text(job["attrs"]),
                                "judge": d})
                n += 1
                if n % 25 == 0:
                    print(f"  [{n}/{len(jobs)}] last={d.get('match','?')}/"
                          f"{d.get('pattern_full','?')}")
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
    by = defaultdict(lambda: {"n": 0, "match_sum": 0, "match_n": 0,
                              "pf_yes": 0, "pf_n": 0, "err": 0})
    def bucket(key, r):
        b = by[key]; b["n"] += 1
        j = r["judge"]
        if not isinstance(j, dict) or "_error" in j or "match" not in j:
            b["err"] += 1; return
        try:
            b["match_sum"] += int(j["match"]); b["match_n"] += 1
        except Exception:
            b["err"] += 1
        if r["pattern_nonsolid"] and str(j.get("pattern_full")).lower() in ("yes", "no"):
            b["pf_n"] += 1
            if str(j.get("pattern_full")).lower() == "yes":
                b["pf_yes"] += 1

    for r in results:
        bucket(f"method:{r['method']}", r)
        bucket(f"axis:{r['active_axis']}", r)
        bucket("ALL", r)

    def line(key):
        b = by[key]
        ma = b["match_sum"] / b["match_n"] if b["match_n"] else float("nan")
        pf = b["pf_yes"] / b["pf_n"] if b["pf_n"] else float("nan")
        return (f"  {key:22s} n={b['n']:4d}  mean_match={ma:4.2f}/3  "
                f"pattern_full_acc={'  n/a' if pf!=pf else f'{pf*100:5.1f}%'} "
                f"(pf_n={b['pf_n']})  err={b['err']}")

    print("\n" + "=" * 78)
    print("  VLM-AS-JUDGE COLLECTION QUALITY")
    print("=" * 78)
    for key in [k for k in by if k.startswith("method:")] + \
               [k for k in by if k.startswith("axis:")] + ["ALL"]:
        print(line(key))

    # rough cost (gpt-5-nano: $0.05/M in, $0.40/M out)
    cost = None
    if provider == "gpt5_nano" and (tok_in or tok_out):
        cost = tok_in / 1e6 * 0.05 + tok_out / 1e6 * 0.40
        print(f"\n  tokens in={tok_in:,} out={tok_out:,}  est_cost=${cost:.3f} (gpt-5-nano)")

    rep = {"by": {k: dict(v) for k, v in by.items()},
           "tokens": {"in": tok_in, "out": tok_out},
           "est_cost_usd": cost}
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  -> {report_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="method:path,... collection logs")
    ap.add_argument("--plans", type=Path, default=Path("data/options/option_plans.jsonl"))
    ap.add_argument("--provider", choices=["gpt5_nano", "vllm"], default="gpt5_nano")
    ap.add_argument("--vlm-url", default="http://127.0.0.1:8002/v1", help="for --provider vllm")
    ap.add_argument("--vlm-model", default=None,
                    help="override model id (default gpt-5-nano / Qwen3-VL-30B)")
    ap.add_argument("--n", type=int, default=500, help="images to judge (0=all)")
    ap.add_argument("--oversample-pattern", action="store_true")
    ap.add_argument("--pattern-frac", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("data/eval/judge.jsonl"))
    ap.add_argument("--report", type=Path, default=Path("data/eval/judge_report.json"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.force and Path(args.out).exists():
        Path(args.out).unlink()
    run(args)


if __name__ == "__main__":
    main()
