#!/usr/bin/env python3
"""
grade_options.py — score each option image independently against its target
----------------------------------------------------------------------------
Dead-simple grader. ONE collection at a time, no method comparison.

Layout assumed:
  <image-root>/
    U001_..._vG/   (folder name contains the plan_id)
        A.jpg  B.jpg  C.jpg  D.jpg   (filename contains the option letter)
    U001_..._vC/
        ...

plan_id — NOT query_id — is the primary key: a dress-code query with two
violation axes emits two plans (`__vG`/`__vC`/`__vP`) that differ in their
options, and keying by query_id collapses them into one. Folders named after a
query_id still resolve when that query has exactly one plan (legacy collections);
a two-plan query_id folder is ambiguous and gets skipped with a warning.

For each option image we look up its TARGET attributes (garment/color/pattern)
from option_plans.jsonl and ask the judge a 1-5 score per axis. That's it.
Output: one row per (plan, option). A flat independent table.

  python scripts/grade_options.py --image-root data/images_dpv \
      --plans data/options/option_plans.jsonl \
      --provider gpt5_nano \
      --out data/eval/grade_dpv.jsonl --report data/eval/grade_dpv_report.json
"""
import argparse, base64, json, re, time
from collections import defaultdict
from pathlib import Path
import os, requests
from requests.adapters import HTTPAdapter

_S = requests.Session()
_S.mount("https://", HTTPAdapter(pool_connections=16, pool_maxsize=32, max_retries=0))
_S.mount("http://", HTTPAdapter(pool_connections=16, pool_maxsize=32, max_retries=0))

_SOLID = {"", "solid", "plain", "none"}

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
    3 = the target pattern on only about half the body, OR right family but ambiguous coverage.
    2 = pattern only LOCALIZED (sleeves/collar/trim/one panel/only-upper or only-lower),
        OR a different but related pattern.
    1 = no such pattern on the body, or a clearly wrong pattern.

pattern_full = "na" if the target is solid; otherwise "yes" ONLY if the pattern score is
>= 4, else "no". Be strict: when in doubt, "no".
set_ok = "no" if the model is a child/baby, or the garment is so cropped/draped that you
cannot tell what it is; otherwise "yes".
A repeating heart/star/character/novelty/logo/text motif counts as graphic_print, NOT
polka_dot and NOT checkered."""


def load_jsonl(p):
    out = []
    if not Path(p).exists():
        return out
    for l in open(p):
        l = l.strip()
        if l:
            out.append(json.loads(l))
    return out


def b64(p):
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode()


def target_text(a):
    g = (a.get("garment_category") or "").replace("_", " ")
    c = a.get("color") or ""
    p = (a.get("pattern") or "solid").replace("_", " ")
    return f"garment={g}; color={c}; pattern={p}"


# ---- scan the folder tree: (query_id, option_letter, image_path) ----
_OPT_RE = re.compile(r"(?:^|[^a-zA-Z])([ABCD])(?:[^a-zA-Z]|$)")
_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}

def scan_images(root, plans_by_pid, qid_to_pids):
    """Match each image folder to a plan, pick one image file per option letter.

    Folders are keyed by plan_id. Collections made before the plan_id switch used
    query_id folder names; those still resolve, but only when the query has a
    single plan — a two-plan query cannot be disambiguated from a query_id folder,
    so it is reported as ambiguous rather than silently graded against one plan.
    """
    root = Path(root)
    found = []
    unmatched = []
    ambiguous = []
    used_qid_fallback = False
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        # resolve which plan this folder belongs to: longest plan_id contained in the name
        pid = None
        for p in plans_by_pid:
            if p in d.name:
                if pid is None or len(p) > len(pid):
                    pid = p
        if pid is None:
            # legacy fallback: folder named after the query_id
            qid = None
            for q in qid_to_pids:
                if q in d.name:
                    if qid is None or len(q) > len(qid):
                        qid = q
            if qid is not None and len(qid_to_pids[qid]) == 1:
                pid = qid_to_pids[qid][0]
                if not used_qid_fallback:
                    used_qid_fallback = True
                    print(f"  [WARN] folder '{d.name}' has no plan_id; falling back to "
                          f"query_id -> {pid}. Re-collect images under plan_id folders.")
            elif qid is not None:
                ambiguous.append(d.name)
                continue
        if pid is None:
            unmatched.append(d.name)
            continue
        # collect image files, bucket by option letter
        per_opt = {}
        for f in sorted(d.iterdir()):
            if f.suffix.lower() not in _IMG_EXT:
                continue
            m = _OPT_RE.search(f.stem.upper())
            if not m:
                continue
            letter = m.group(1)
            per_opt.setdefault(letter, f)  # first match wins
        for letter, f in per_opt.items():
            found.append((pid, letter, str(f)))
    return found, unmatched, ambiguous


def call_judge(provider, base, model, img, attrs, key=None, timeout=90, retries=3):
    txt_in = f"TARGET attributes: {target_text(attrs)}\nGrade the three axes with the rubric. Return the JSON."
    content = [{"type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64(img)}"}},
               {"type": "text", "text": txt_in}]
    payload = {"model": model,
               "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                            {"role": "user", "content": content}]}
    if provider == "gpt5_nano":
        payload["max_completion_tokens"] = 512
        payload["reasoning_effort"] = "minimal"
    else:
        payload["max_tokens"] = 200
        payload["temperature"] = 0.0
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    url = f"{base.rstrip('/')}/chat/completions"
    last = "unparseable"
    for att in range(retries):
        try:
            r = _S.post(url, json=payload, headers=headers, timeout=timeout)
            if r.status_code != 200:
                if 400 <= r.status_code < 500 and r.status_code != 429:
                    return {"_error": f"HTTP {r.status_code}: {r.text[:160]}"}
                last = f"HTTP {r.status_code}"; time.sleep(1.5 * (att + 1)); continue
            data = r.json()
            t = data["choices"][0].get("message", {}).get("content") or ""
            usage = data.get("usage", {})
            if "</think>" in t:
                t = t.split("</think>", 1)[1]
            s, e = t.find("{"), t.rfind("}")
            if s >= 0 and e > s:
                d = json.loads(t[s:e + 1])
                d["_usage"] = {"in": usage.get("prompt_tokens", 0),
                               "out": usage.get("completion_tokens", 0)}
                return d
            last = "empty_content"; time.sleep(1.0 * (att + 1))
        except Exception as ex:
            last = str(ex)[:160]; time.sleep(1.5 * (att + 1))
    return {"_error": last}


def run(args):
    plan_rows = load_jsonl(args.plans)
    # plan_id is the primary key: keying by query_id silently dropped one of the two
    # plans for every query with two violation axes (458 queries / 916 plans in v2).
    plans = {(p.get("plan_id") or p["query_id"]): p for p in plan_rows}
    qid_to_pids = defaultdict(list)
    for p in plan_rows:
        qid_to_pids[p["query_id"]].append(p.get("plan_id") or p["query_id"])
    found, unmatched, ambiguous = scan_images(args.image_root, plans, qid_to_pids)
    print(f"  scanned {args.image_root}: {len(found)} option images, "
          f"{len({p for p,_,_ in found})} plans matched, {len(unmatched)} folders unmatched")
    if unmatched[:3]:
        print(f"  (unmatched folder examples: {unmatched[:3]})")
    if ambiguous:
        print(f"  [WARN] {len(ambiguous)} folders name a query_id that has 2 plans and "
              f"cannot be resolved; skipped (examples: {ambiguous[:3]})")

    if args.provider == "gpt5_nano":
        base = "https://api.openai.com/v1"; model = args.vlm_model or "gpt-5-nano"
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit("OPENAI_API_KEY not set")
    else:
        base = args.vlm_url; model = args.vlm_model or "Qwen/Qwen3-VL-30B-A3B-Instruct"; key = None

    out = Path(args.out)
    existing = load_jsonl(out)

    def _key(r):
        # older result files carry only query_id; fall back so a resume of a
        # pre-plan_id run still recognizes what it already graded
        return f"{r.get('plan_id') or r.get('query_id')}|{r['option']}"

    done = {_key(r) for r in existing
            if isinstance(r.get("judge"), dict) and "garment" in r["judge"]}
    results = [r for r in existing if _key(r) in done]

    jobs = []
    for pid, letter, path in found:
        if f"{pid}|{letter}" in done:
            continue
        attrs = ((plans[pid].get("options") or {}).get(letter) or {}).get("attributes", {}) or {}
        axis = plans[pid].get("active_axis")
        qid = plans[pid].get("query_id")
        jobs.append((pid, qid, letter, path, attrs, axis))
    print(f"  to grade: {len(jobs)} (already done {len(done)})  model={model}")

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    lock = threading.Lock(); ti = to = 0

    def work(job):
        pid, qid, letter, path, attrs, axis = job
        return job, call_judge(args.provider, base, model, path, attrs, key=key)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        n = 0
        for fut in as_completed(futs):
            (pid, qid, letter, path, attrs, axis), d = fut.result()
            with lock:
                u = d.pop("_usage", {}) if isinstance(d, dict) else {}
                ti += u.get("in", 0); to += u.get("out", 0)
                results.append({"plan_id": pid, "query_id": qid, "option": letter,
                                "active_axis": axis, "image_path": path,
                                "target": target_text(attrs), "judge": d})
                n += 1
                if n % 50 == 0:
                    print(f"  [{n}/{len(jobs)}]")
                    _save(results, out)
    _save(results, out)
    report(results, ti, to, args.provider, args.report)


def _save(rows, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _i(v):
    try:
        return min(5, max(1, int(round(float(v)))))
    except Exception:
        return None


def report(results, ti, to, provider, rp):
    # independent table: per OPTION letter, per AXIS, and overall
    by = defaultdict(lambda: {"n": 0, "err": 0, "an": 0, "g": 0, "c": 0, "p": 0,
                              "ov": 0.0, "pfy": 0, "pfn": 0, "sb": 0})

    def bucket(k, r):
        b = by[k]; b["n"] += 1
        j = r["judge"]
        if not isinstance(j, dict) or "garment" not in j:
            b["err"] += 1; return
        g, c, p = _i(j.get("garment")), _i(j.get("color")), _i(j.get("pattern"))
        if None in (g, c, p):
            b["err"] += 1; return
        b["an"] += 1; b["g"] += g; b["c"] += c; b["p"] += p; b["ov"] += (g + c + p) / 3
        if str(j.get("set_ok")).lower() == "no":
            b["sb"] += 1
        nonsolid = (r["active_axis"] == "pattern"
                    and r["target"].split("pattern=")[-1].strip() not in _SOLID)
        if nonsolid and str(j.get("pattern_full")).lower() in ("yes", "no"):
            b["pfn"] += 1
            if str(j.get("pattern_full")).lower() == "yes":
                b["pfy"] += 1

    for r in results:
        bucket(f"option:{r['option']}", r)
        bucket(f"axis:{r['active_axis']}", r)
        bucket("ALL", r)

    def avg(s, n):
        return s / n if n else float("nan")

    def line(k):
        b = by[k]; an = b["an"]
        pf = avg(b["pfy"], b["pfn"]); pfs = "  n/a" if pf != pf else f"{pf*100:5.1f}%"
        return (f"  {k:14s} n={b['n']:4d} ok={an:4d} err={b['err']:3d} | "
                f"garment={avg(b['g'],an):4.2f} color={avg(b['c'],an):4.2f} "
                f"pattern={avg(b['p'],an):4.2f} overall={avg(b['ov'],an):4.2f}/5 | "
                f"pattern_full={pfs}(n={b['pfn']}) set_bad={b['sb']}")

    print("\n" + "=" * 96)
    print("  OPTION GRADING (1-5 per axis, independent)")
    print("=" * 96)
    for k in ([f"option:{x}" for x in "ABCD" if f"option:{x}" in by] +
              [k for k in by if k.startswith("axis:")] + ["ALL"]):
        print(line(k))
    cost = None
    if provider == "gpt5_nano" and (ti or to):
        cost = ti / 1e6 * 0.05 + to / 1e6 * 0.40
        print(f"\n  tokens in={ti:,} out={to:,}  est_cost=${cost:.3f}")
    Path(rp).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"by": {k: dict(v) for k, v in by.items()},
               "tokens": {"in": ti, "out": to}, "cost": cost},
              open(rp, "w"), ensure_ascii=False, indent=2, default=str)
    print(f"\n  -> {rp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-root", required=True, help="e.g. data/images_dpv")
    ap.add_argument("--plans", type=Path, default=Path("data/options/option_plans.jsonl"))
    ap.add_argument("--provider", choices=["gpt5_nano", "vllm"], default="gpt5_nano")
    ap.add_argument("--vlm-url", default="http://127.0.0.1:8002/v1")
    ap.add_argument("--vlm-model", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.force and Path(a.out).exists():
        Path(a.out).unlink()
    run(a)


if __name__ == "__main__":
    main()