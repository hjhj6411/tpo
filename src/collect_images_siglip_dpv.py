#!/usr/bin/env python3
"""
collect_images_siglip_dpv.py  —  Decomposed Perceptual Verification collector
-----------------------------------------------------------------------------
TOP-1 FashionSigLIP collector whose verification stage is DECOMPOSED into three
single-purpose VLM gates, composed by DETERMINISTIC instance-metadata routing
(no LLM agent, no ReAct, no query rewrite, no best-of-top-k). Retrieval is frozen
(FashionSigLIP on :1235); we only change what happens AFTER an image is fetched.

Each gate is ONE focused VLM call answering ONE narrow question:

  Gate 1  PROMINENCE  (full image, runs always, cheapest -> first, short-circuits)
          "Is the <garment> worn AND clearly the main, fully-visible subject?"
          Catches draped/tied-over-arm, top/bottom-ambiguous crops, face/accessory-
          dominated frames. A visible face is fine.  -> NO rejects the candidate.

  Gate 2  DEMOGRAPHIC (full image, runs always, LABEL only)
          -> {gender in man/woman/unclear, age in adult/child/unclear}.
          age==child rejects the candidate (adult-only). gender feeds a
          deterministic SET-LEVEL lock (majority vote, or A-anchor) so all four
          options A-D share one gender.

  Gate 3  PATTERN-COVERAGE (TORSO CROP, routed: only when active_axis==pattern AND
          the option's target pattern is non-solid)
          "Does <pattern> cover essentially the ENTIRE region?" on a deterministic
          torso crop (sleeves/collar/head excluded) -> accept only COVERS_WHOLE.
          Skipped for color-axis / solid / non-pattern options.

Composition: per candidate, run prominence -> demographic -> pattern, short-
circuiting to the next rank on any rejection. A 4-option set is collected
gender-agnostically first, then the gender lock reconciles any minority option by
re-walking its ranks with the gender constraint (cached, so no redundant calls).

ALL gates are FAIL-OPEN (VLM error / unparseable -> pass / unclear), so a flaky
VLM degrades to plain top-1, never a skip-storm.

Routing is decided ONLY by instance metadata (active_axis, target pattern,
garment type) — fully deterministic and reproducible.

Servers:  KNN serve_fsiglip_knn.py :1235 ; VLM OpenAI-compat (e.g. :8002..:8005)

Usage:
  python src/collect_images_siglip_dpv.py \
    --client-url http://127.0.0.1:1235/knn-service \
    --vlm-urls "http://127.0.0.1:8002/v1,http://127.0.0.1:8003/v1,http://127.0.0.1:8004/v1,http://127.0.0.1:8005/v1" \
    --vlm-model Qwen/Qwen3-VL-4B-Instruct \
    --gates prominence,demographic,pattern --gender-lock majority \
    --image_root data/images_dpv --output data/images_dpv/collection_log.jsonl \
    --workers 4 --top_k 12 --limit 30 --force

Ablations: --gates "" -> pure top-1; --gates prominence -> framing gate only;
           --pattern-dual -> require BOTH an upper- and a lower-torso crop to be
           COVERS_WHOLE (rejects one-panel / only-upper / only-lower);
           --no-pattern-crop -> ask the pattern question on the full image (weaker).
"""
import argparse
import base64
import io
import json
import time
from collections import Counter
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from PIL import Image

UA = {"User-Agent": "Mozilla/5.0 (compatible; POD-Bench-SigLIP/1.0)"}

_SESSION = requests.Session()
_SESSION.headers.update(UA)
_adapter = HTTPAdapter(pool_connections=32, pool_maxsize=64, max_retries=0)
_SESSION.mount("http://", _adapter)
_SESSION.mount("https://", _adapter)

_SOLID_LIKE = {"", "solid", "plain", "none"}

# deterministic torso-crop geometry (fractions of the image box).
# x: drop outer arm columns; y: drop head/collar at top, keep upper+lower torso.
CROP_CFG = {"x0": 0.18, "x1": 0.82, "y0": 0.15, "ymid": 0.46, "y1": 0.78}


# ── jsonl io ──────────────────────────────────────────────────────────────
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


def save_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def is_solid(p):
    return (p or "").strip().lower() in _SOLID_LIKE


# ── retrieval + fetch (retrieval is FROZEN; query sent verbatim) ───────────
def knn_search(client_url, query, top_k=12, indice_name="pod_fashion"):
    try:
        resp = _SESSION.post(client_url, json={
            "text": query, "modality": "image",
            "num_images": top_k, "num_result_ids": top_k,
            "indice_name": indice_name}, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"    [knn] query failed for {query!r}: {e}")
        return []


def fetch_image(url, timeout=15, min_side=64):
    """Download -> PIL (no disk write). None on failure / too small."""
    try:
        r = _SESSION.get(url, timeout=timeout)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None
    if min(img.size) < min_side:
        return None
    return img


# ── Stage 0: deterministic torso crop (no VLM) ─────────────────────────────
def make_crops(img, dual, no_crop, cfg=CROP_CFG):
    if no_crop:
        return [img]
    w, h = img.size
    x0, x1 = int(w * cfg["x0"]), int(w * cfg["x1"])
    x0, x1 = max(0, x0), min(w, max(x0 + 1, x1))
    if dual:
        ya, ym, yb = (int(h * cfg["y0"]), int(h * cfg["ymid"]), int(h * cfg["y1"]))
        ym = max(ya + 1, min(yb - 1, ym))
        return [img.crop((x0, ya, x1, ym)), img.crop((x0, ym, x1, yb))]
    ya, yb = int(h * cfg["y0"]), int(h * cfg["y1"])
    yb = max(ya + 1, yb)
    return [img.crop((x0, ya, x1, yb))]


# ── one OpenAI-compatible VLM call (PIL in) ────────────────────────────────
def _b64_pil(img):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _vlm(vlm_url, model, system, user, pil_img, max_tokens, timeout=120, retries=2):
    """Return text, or None on failure (caller fail-opens)."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{_b64_pil(pil_img)}"}},
                {"type": "text", "text": user}]},
        ],
        "max_tokens": max_tokens, "temperature": 0.0,
    }
    for attempt in range(retries):
        try:
            r = _SESSION.post(f"{vlm_url.rstrip('/')}/chat/completions",
                              json=payload, timeout=timeout)
            if r.status_code != 200:
                time.sleep(0.4 * (attempt + 1)); continue
            txt = (r.json()["choices"][0]["message"].get("content") or "")
            if "</think>" in txt:
                txt = txt.split("</think>", 1)[1]
            return txt
        except Exception:
            time.sleep(0.4 * (attempt + 1))
    return None


# ── Gate 1: prominence / framing (one yes-no, fail-open PASS) ──────────────
_PROM_SYS = ("You are a fashion product-image inspector. "
             "Answer with ONE word: YES or NO.")


def _prom_user(garment):
    g = garment or "garment"
    return (f"Is the {g} being WORN on a body AND clearly the MAIN, fully-visible "
            f"subject of this image? Answer NO if the {g} is only draped / folded / "
            f"held / tied over an arm or shoulder, if the image is cropped so you "
            f"cannot tell whether it is a top or a bottom, or if it is only a minor "
            f"element behind a dominant face/headshot or an accessory (bag/shoes/hat). "
            f"A visible face is fine as long as the {g} is clearly worn and prominent. "
            f"Answer only YES or NO.")


def gate_prominence(vlm_url, model, pil, garment):
    txt = _vlm(vlm_url, model, _PROM_SYS, _prom_user(garment), pil, max_tokens=4)
    if txt is None:
        return True  # fail-open
    toks = txt.strip().lower().split()
    return not (toks and toks[0].startswith("no"))


# ── Gate 2: demographic label (gender + age; fail-open unclear/unclear) ────
_DEMO_SYS = ("You are a fashion product-image inspector. Reply ONLY with compact JSON.")
_DEMO_USER = ('Report the person modeling the garment as JSON: '
              '{"gender":"man|woman|unclear","age":"adult|child|unclear"}. '
              'gender = the apparent gender of the model; flat-lay / mannequin / '
              'no visible person => unclear. age = "child" for a baby/toddler/'
              'pre-teen kid, "adult" for a clearly grown adult, "unclear" if no '
              'person is visible. Judge by face and body proportions, not garment size.')


def _norm_gender(s):
    t = (s or "").strip().lower()
    if "woman" in t or "female" in t:
        return "woman"
    if "man" in t or "male" in t:
        return "man"
    return "unclear"


def _norm_age(s):
    t = (s or "").strip().lower()
    if "child" in t or "baby" in t or "kid" in t or "toddler" in t:
        return "child"
    if "adult" in t:
        return "adult"
    return "unclear"


def gate_demographic(vlm_url, model, pil):
    txt = _vlm(vlm_url, model, _DEMO_SYS, _DEMO_USER, pil, max_tokens=24)
    if not txt:
        return "unclear", "unclear"
    s, e = txt.find("{"), txt.rfind("}")
    if s >= 0 and e > s:
        try:
            d = json.loads(txt[s:e + 1])
            return _norm_gender(d.get("gender")), _norm_age(d.get("age"))
        except Exception:
            pass
    return _norm_gender(txt), _norm_age(txt)


# ── Gate 3: pattern coverage on torso crop(s) (fail-open PASS) ─────────────
_PAT_SYS = ("You are a fashion product-image inspector looking at a CROPPED region "
            "showing the torso/body of a garment (sleeves, collar, and head are "
            "excluded). Answer with ONE word.")


def _pat_user(pattern):
    p = (pattern or "pattern").replace("_", " ")
    return (f"Does a {p} pattern cover essentially the ENTIRE visible region "
            f"(almost the whole area), only PART of it, or NONE of it? "
            f"Answer exactly one of: COVERS_WHOLE, PARTIAL, NONE.")


def gate_pattern(vlm_url, model, crops, pattern):
    """Require every crop to read COVERS_WHOLE. Any PARTIAL/NONE -> reject.
    VLM failure / ambiguous -> that crop passes (fail-open)."""
    for crop in crops:
        txt = _vlm(vlm_url, model, _PAT_SYS, _pat_user(pattern), crop, max_tokens=6)
        if txt is None:
            continue
        t = txt.strip().upper().replace(" ", "_")
        if "COVERS_WHOLE" in t:
            continue
        if "PARTIAL" in t or "NONE" in t:
            return False
        # ambiguous -> fail-open pass
    return True


# ── set-level gender decision (deterministic) ──────────────────────────────
def decide_target_gender(picks, mode):
    known = [picks[k]["gender"] for k in "ABCD"
             if picks.get(k) and picks[k]["gender"] in ("man", "woman")]
    if not known:
        return None
    a = picks.get("A")
    a_g = a["gender"] if (a and a["gender"] in ("man", "woman")) else None
    if mode == "anchor":
        return a_g or Counter(known).most_common(1)[0][0]
    c = Counter(known).most_common()
    if len(c) >= 2 and c[0][1] == c[1][1] and a_g:   # tie -> A breaks it
        return a_g
    return c[0][0]


# ── per-option resolution (the cascade) ────────────────────────────────────
def resolve_option(k, params, active_axis, client_url, top_k, indice_name,
                   vlm_url, vlm_model, gates, dual, no_crop, target_gender, cache):
    """Walk ranks; first candidate passing every ROUTED gate (and the gender
    constraint, if set) wins. Verdicts memoized per (option, url) so the gender
    reconcile pass costs no extra VLM calls. Returns verdict dict or None."""
    query = params["query"]; garment = params["garment"]
    target_pattern = params["target_pattern"]; do_pattern = params["do_pattern"]

    cands = knn_search(client_url, query, top_k=top_k, indice_name=indice_name)
    for rank, c in enumerate(cands[:top_k]):
        url = c.get("url") or c.get("image_url")
        if not url:
            continue
        key = (k, url)
        v = cache.get(key)
        if v is None:
            pil = fetch_image(url)
            if pil is None:
                cache[key] = {"dead": True}
                continue
            prom = True
            if "prominence" in gates:
                prom = gate_prominence(vlm_url, vlm_model, pil, garment)
            gender, age = "unclear", "unclear"
            if "demographic" in gates:
                gender, age = gate_demographic(vlm_url, vlm_model, pil)
            pat_ok = True
            if do_pattern:
                crops = make_crops(pil, dual, no_crop)
                pat_ok = gate_pattern(vlm_url, vlm_model, crops, target_pattern)
            v = {"dead": False, "prom": prom, "gender": gender, "age": age,
                 "pat_ok": pat_ok, "pil": pil, "rank": rank,
                 "similarity": c.get("similarity"),
                 "caption": (c.get("caption") or "")[:120]}
            cache[key] = v

        if v["dead"] or not v["prom"]:
            continue
        if v["age"] == "child":
            continue
        if do_pattern and not v["pat_ok"]:
            continue
        if (target_gender in ("man", "woman") and v["gender"] in ("man", "woman")
                and v["gender"] != target_gender):
            continue
        return v
    return None


def collect_for_plan(plan, image_root, client_url, top_k, indice_name,
                     vlm_url, vlm_model, gates, gender_lock, dual, no_crop):
    qid = plan["query_id"]
    out_dir = Path(image_root) / qid
    out_dir.mkdir(parents=True, exist_ok=True)
    active_axis = plan.get("active_axis")

    result = {
        "query_id": qid, "user_id": plan.get("user_id"),
        "domain": plan.get("domain", "fashion"),
        "active_axis": active_axis,
        "main_category": plan.get("main_category"),
        "scenario_archetype": plan.get("scenario_archetype"),
        "gates_active": sorted(gates), "gender_lock": gender_lock,
        "options": {}, "all_collected": False, "skipped": False,
        "set_gender": None, "gender_consistent": True,
    }

    # per-option routing params (deterministic, from metadata)
    params = {}
    for k in "ABCD":
        opt = plan["options"][k]
        attrs = opt.get("attributes", {}) or {}
        tp = attrs.get("pattern")
        params[k] = {
            "query": opt["search_query"],
            "garment": (attrs.get("garment_category") or "").replace("_", " "),
            "target_pattern": tp,
            "do_pattern": ("pattern" in gates and active_axis == "pattern"
                           and not is_solid(tp)),
        }

    cache = {}
    picks = {}
    # phase 1: gender-agnostic
    for k in "ABCD":
        picks[k] = resolve_option(k, params[k], active_axis, client_url, top_k,
                                  indice_name, vlm_url, vlm_model, gates,
                                  dual, no_crop, target_gender=None, cache=cache)

    # phase 2: set-level gender lock (reconcile minority options)
    target_gender = decide_target_gender(picks, gender_lock) if "demographic" in gates else None
    if target_gender:
        for k in "ABCD":
            p = picks[k]
            if p and p["gender"] in ("man", "woman") and p["gender"] != target_gender:
                np_ = resolve_option(k, params[k], active_axis, client_url, top_k,
                                     indice_name, vlm_url, vlm_model, gates,
                                     dual, no_crop, target_gender=target_gender, cache=cache)
                if np_:
                    picks[k] = np_
                else:
                    result["gender_consistent"] = False  # could not reconcile

    # save + build options
    n_ok = 0
    for k in "ABCD":
        p = picks[k]
        img_path = out_dir / f"{k}.jpg"
        if p:
            try:
                p["pil"].save(img_path, "JPEG", quality=92)
            except Exception:
                p = None
        if p:
            n_ok += 1
            rec = {"image_path": str(img_path), "source": "fashionsiglip_dpv",
                   "search_query": params[k]["query"], "rank_used": p["rank"],
                   "source_title": p["caption"], "similarity": p["similarity"],
                   "prominence": ("yes" if "prominence" in gates else "n/a"),
                   "model_gender": p["gender"], "model_age": p["age"]}
            if params[k]["do_pattern"]:
                rec["pattern_coverage"] = "covers_whole"
            result["options"][k] = rec
        else:
            result["options"][k] = {"image_path": None, "source": "FAILED",
                                    "search_query": params[k]["query"]}

    result["set_gender"] = target_gender
    result["all_collected"] = (n_ok == 4)
    result["skipped"] = (n_ok < 4)
    return result


def run(plan_path, output_path, image_root, client_url, indice_name,
        limit, top_k, workers, vlm_urls, vlm_model, gates, gender_lock,
        dual, no_crop):
    plans = load_jsonl(plan_path)
    print(f"Loaded {len(plans)} plans  | server={client_url}  top_k={top_k}")
    print(f"  gates={sorted(gates)}  gender_lock={gender_lock}  "
          f"pattern_dual={dual}  no_pattern_crop={no_crop}")
    vlm_urls = vlm_urls or [None]
    print(f"  VLM: {vlm_urls}  model={vlm_model}")
    if vlm_urls == [None] and gates:
        print("  NOTE: no --vlm-urls but gates requested; gates need a VLM. "
              "Pure top-1 fallback (all gates fail-open).")

    if Path(output_path).exists():
        existing = load_jsonl(output_path)
        done = {r["query_id"] for r in existing}
        results = existing
        todo = [p for p in plans if p["query_id"] not in done]
        print(f"  resuming: {len(done)} already done")
    else:
        results, todo = [], plans
    if limit > 0:
        todo = todo[:limit]
    print(f"  to collect: {len(todo)}")

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    lock = threading.Lock()

    def worker(idx_plan):
        idx, plan = idx_plan
        vlm_url = vlm_urls[idx % len(vlm_urls)]
        return collect_for_plan(plan, image_root, client_url, top_k, indice_name,
                                vlm_url, vlm_model, gates, gender_lock, dual, no_crop)

    n_workers = max(1, workers)
    if n_workers == 1:
        for i, plan in enumerate(todo):
            print(f"  [{i+1}/{len(todo)}] {plan['query_id']} ({plan.get('active_axis')})")
            try:
                results.append(worker((i, plan)))
                if (i + 1) % 10 == 0:
                    save_jsonl(results, output_path)
            except Exception as e:
                print(f"    ERROR {plan['query_id']}: {e}")
            time.sleep(0.02)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(worker, (i, p)): p for i, p in enumerate(todo)}
            done_n = 0
            for fut in as_completed(futs):
                p = futs[fut]
                try:
                    r = fut.result()
                    with lock:
                        results.append(r)
                        done_n += 1
                        tag = "OK" if r["all_collected"] else "SKIP"
                        gc = "" if r["gender_consistent"] else " GENDER!"
                        print(f"  [{done_n}/{len(todo)}] {p['query_id']} "
                              f"({p.get('active_axis')}) -> {tag}{gc}")
                        if done_n % 10 == 0:
                            save_jsonl(results, output_path)
                except Exception as e:
                    print(f"    ERROR {p['query_id']}: {e}")

    save_jsonl(results, output_path)
    n_done = sum(1 for r in results if r.get("all_collected"))
    n_skip = sum(1 for r in results if r.get("skipped"))
    n_gincon = sum(1 for r in results if not r.get("gender_consistent", True))
    print(f"\nSaved {len(results)}: complete={n_done} "
          f"({n_done/max(len(results),1):.0%}), skipped={n_skip}, "
          f"gender_inconsistent={n_gincon}")


def selftest(client_url, indice_name, top_k):
    for q in ["beige plaid shirt", "white polka dot t shirt", "navy coat"]:
        cands = knn_search(client_url, q, top_k=top_k, indice_name=indice_name)
        print(f"\nquery: {q!r}  -> {len(cands)} candidates")
        for i, c in enumerate(cands[:5]):
            print(f"  [{i}] {c.get('similarity')} | {c.get('caption','')[:70]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan_path", type=Path,
                    default=Path("data/options/option_plans.jsonl"))
    ap.add_argument("--output", type=Path,
                    default=Path("data/images_dpv/collection_log.jsonl"))
    ap.add_argument("--image_root", type=Path, default=Path("data/images_dpv"))
    ap.add_argument("--client-url", default="http://127.0.0.1:1235/knn-service")
    ap.add_argument("--indice-name", default="pod_fashion")
    ap.add_argument("--vlm-urls", default="",
                    help="comma-separated VLM endpoints; empty = pure top-1 (gates fail-open)")
    ap.add_argument("--vlm-model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--gates", default="prominence,demographic,pattern",
                    help="subset of prominence,demographic,pattern (comma-sep; ''=pure top-1)")
    ap.add_argument("--gender-lock", choices=["majority", "anchor"], default="majority")
    ap.add_argument("--pattern-dual", action="store_true",
                    help="require BOTH upper- and lower-torso crops to be COVERS_WHOLE")
    ap.add_argument("--no-pattern-crop", action="store_true",
                    help="ask the pattern question on the full image (weaker; ablation)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top_k", type=int, default=12)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest(args.client_url, args.indice_name, args.top_k)
        return
    if args.force and Path(args.output).exists():
        Path(args.output).unlink()
    gates = {g.strip() for g in args.gates.split(",") if g.strip()}
    bad = gates - {"prominence", "demographic", "pattern"}
    if bad:
        raise SystemExit(f"unknown gate(s): {bad}")
    vlm_urls = [u.strip() for u in args.vlm_urls.split(",") if u.strip()] or None
    run(args.plan_path, args.output, args.image_root, args.client_url,
        args.indice_name, args.limit, args.top_k, args.workers,
        vlm_urls=vlm_urls, vlm_model=args.vlm_model, gates=gates,
        gender_lock=args.gender_lock, dual=args.pattern_dual,
        no_crop=args.no_pattern_crop)


if __name__ == "__main__":
    main()