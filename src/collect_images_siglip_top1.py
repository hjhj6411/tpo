#!/usr/bin/env python3
"""
collect_images_siglip_top1.py  (+ gender + optional pattern-coverage)
---------------------------------------------------------------------
TOP-1 FashionSigLIP collector. Always keeps the four options A-D
gender-consistent (single-word VLM check, fail-open). On the PATTERN axis only,
it can additionally require the pattern to cover ALMOST THE ENTIRE garment body
(strict all-over), via three switchable strategies:

  --pattern-mode off      pure top-1 (gender check only)               [default]
  --pattern-mode specify  query specification only: append an all-over
                          phrase to the retrieval query so SigLIP ranks
                          full-coverage items higher. No extra VLM cost.
  --pattern-mode verify   VLM verification only: inspect the image and reject
                          partial/localized patterns (sleeves/trim/half body).
  --pattern-mode both     specify + verify.

The strict bar for "full": the pattern must blanket essentially the WHOLE body
(top + bottom of the torso), reading as that pattern at a glance. Sleeves-only,
collar/trim-only, one panel, or only-upper/only-lower => NOT full.

COLOR axis is never touched by any pattern mode. gender + pattern share ONE VLM
call per candidate. All VLM use is fail-open (failure -> pass), so it never
turns into the SKIP storm from before.

Servers:  KNN serve_fsiglip_knn.py :1235 ; VLM OpenAI-compat :8002..:8005

Usage:
# A) specify
python src/collect_images_siglip_top1.py --pattern-mode specify \
  --client-url http://127.0.0.1:1235/knn-service \
  --vlm-urls "http://127.0.0.1:8002/v1,http://127.0.0.1:8003/v1,http://127.0.0.1:8004/v1,http://127.0.0.1:8005/v1" \
  --vlm-model Qwen/Qwen3-VL-30B-A3B-Instruct \
  --image_root data/img_specify --output data/img_specify/log.jsonl \
  --workers 4 --top_k 12 --limit 30 --force

# B) verify  (--image_root data/img_verify, --pattern-mode verify)
# C) both    (--image_root data/img_both,    --pattern-mode both)
"""
import argparse
import base64
import io
import json
import time
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

# Non-pattern targets that should NOT get an all-over phrase appended
_SOLID_LIKE = {"", "solid", "plain", "none"}


# -- jsonl io --
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


# -- retrieval + download --
def knn_search(client_url, query, top_k=10, indice_name="pod_fashion"):
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


def download_image(url, dest_path, min_side=64, timeout=15):
    try:
        resp = _SESSION.get(url, timeout=timeout)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return False
    if min(img.size) < min_side:
        return False
    try:
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(dest_path, "JPEG", quality=92)
        return True
    except Exception:
        return False


# -- query specification (retrieval-side) --
def specify_query(query, target_pattern):
    """Append a strict all-over phrase. PATTERN axis, non-solid target only."""
    tp = (target_pattern or "").strip().lower()
    if tp in _SOLID_LIKE:
        return query
    return (f"{query}, {tp} pattern covering the entire garment body, "
            f"bold all-over print from top to bottom")


# -- single VLM call: gender (+ strict full-coverage when asked) --
_GENDER_ONLY_SYSTEM = (
    "You see one clothing product image. Reply with ONLY one word: "
    "man, woman, or unclear - the gender of the person modeling the garment. "
    "If it is a flat-lay / mannequin / no person, reply unclear."
)

_GENDER_PATTERN_SYSTEM = (
    "You inspect one clothing product image. Judge the MAIN garment.\n"
    "Return ONLY compact JSON: {\"gender\":\"man|woman|unclear\","
    "\"pattern_full\":true|false}.\n"
    "gender: the model's gender; flat-lay/mannequin/no person => unclear.\n"
    "pattern_full = true ONLY IF the TARGET pattern blankets ALMOST THE ENTIRE "
    "garment BODY - both the upper and lower torso - so the garment reads as that "
    "pattern at a glance. Set false if the pattern is ONLY on the sleeves, ONLY "
    "on the collar/cuffs/trim/hem/pocket, on just one panel, on only the upper OR "
    "only the lower half, or is a small/localized motif. Be STRICT: when in doubt, false."
)


def _post_vlm(vlm_url, model, system, user_text, image_path, max_tokens,
              timeout=120, retries=2):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": user_text}]},
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


def _word_to_gender(txt):
    t = (txt or "").lower()
    if "woman" in t or "female" in t:
        return "woman"
    if "man" in t or "male" in t:
        return "man"
    return "unclear"


def vlm_check(vlm_url, model, image_path, need_pattern):
    """Returns (gender, pattern_full).
    gender in man/woman/unclear; pattern_full in True/False/None.
    Fail-open: on any failure -> ('unclear', None)."""
    if not need_pattern:
        txt = _post_vlm(vlm_url, model, _GENDER_ONLY_SYSTEM,
                        "man, woman, or unclear?", image_path, max_tokens=6)
        return _word_to_gender(txt), None
    txt = _post_vlm(vlm_url, model, _GENDER_PATTERN_SYSTEM,
                    "Return the JSON.", image_path, max_tokens=40)
    if not txt:
        return "unclear", None
    s, e = txt.find("{"), txt.rfind("}")
    if s >= 0 and e > s:
        try:
            d = json.loads(txt[s:e + 1])
            return _word_to_gender(d.get("gender")), bool(d.get("pattern_full"))
        except Exception:
            pass
    return _word_to_gender(txt), None  # JSON broke -> gender only, pattern unknown


def resolve_option_top1(query, img_path, client_url, top_k, indice_name,
                        vlm_url, vlm_model, target_gender,
                        active_axis, target_pattern, pattern_mode):
    """Top-1 that downloads AND is gender-consistent, AND (pattern axis +
    verify/both) has a strict all-over pattern. Walks down only on: dead URL,
    gender conflict, or partial pattern. fail-open everywhere.
    Returns (option_result, seen_gender)."""
    pat_axis = (active_axis == "pattern")
    tp = (target_pattern or "").strip().lower()
    nonsolid = tp not in _SOLID_LIKE

    do_specify = pat_axis and nonsolid and pattern_mode in ("specify", "both")
    do_verify  = pat_axis and nonsolid and pattern_mode in ("verify", "both")

    q = specify_query(query, target_pattern) if do_specify else query
    cands = knn_search(client_url, q, top_k=top_k, indice_name=indice_name)

    for rank, c in enumerate(cands):
        url = c.get("url") or c.get("image_url")
        if not url:
            continue
        if not download_image(url, img_path):
            continue

        if not vlm_url:  # no VLM -> pure top-1 (with optional specify only)
            return ({"image_path": str(img_path), "source": "fashionsiglip_top1",
                     "search_query": q, "rank_used": rank,
                     "source_title": c.get("caption", "")[:120],
                     "similarity": c.get("similarity")}, "unclear")

        g, pfull = vlm_check(vlm_url, vlm_model, img_path, need_pattern=do_verify)

        # gender gate (always on)
        if target_gender in ("man", "woman") and g in ("man", "woman") and g != target_gender:
            continue
        # strict full-coverage gate (pattern axis + verify/both); fail-open if None
        if do_verify and pfull is False:
            continue

        rec = {"image_path": str(img_path), "source": "fashionsiglip_top1",
               "search_query": q, "rank_used": rank,
               "source_title": c.get("caption", "")[:120],
               "similarity": c.get("similarity"), "model_gender": g}
        if do_verify:
            rec["pattern_full"] = pfull
        return (rec, g)

    return ({"image_path": None, "source": "FAILED", "search_query": q}, None)


def collect_for_plan(plan, image_root, client_url, top_k, indice_name,
                     vlm_url, vlm_model, pattern_mode):
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
        "options": {}, "all_collected": False, "skipped": False,
        "set_gender": None,
    }

    n_ok = 0
    target_gender = None
    for k in "ABCD":
        opt = plan["options"][k]
        query = opt["search_query"]
        target_pattern = (opt.get("attributes", {}) or {}).get("pattern")
        img_path = out_dir / f"{k}.jpg"
        res, seen = resolve_option_top1(
            query, img_path, client_url, top_k, indice_name,
            vlm_url, vlm_model, target_gender,
            active_axis, target_pattern, pattern_mode)
        result["options"][k] = res
        if res["image_path"]:
            n_ok += 1
            if target_gender is None and seen in ("man", "woman"):
                target_gender = seen

    result["set_gender"] = target_gender
    result["all_collected"] = (n_ok == 4)
    result["skipped"] = (n_ok < 4)
    return result


def run(plan_path, output_path, image_root, client_url, indice_name,
        limit, top_k, workers, vlm_urls, vlm_model, pattern_mode):
    plans = load_jsonl(plan_path)
    print(f"Loaded {len(plans)} plans  | server={client_url}  top_k={top_k}  "
          f"pattern_mode={pattern_mode}")
    vlm_urls = vlm_urls or [None]
    print(f"  VLM: {vlm_urls}  model={vlm_model}")

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
                                vlm_url, vlm_model, pattern_mode)

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
                        print(f"  [{done_n}/{len(todo)}] {p['query_id']} "
                              f"({p.get('active_axis')}) -> {tag}")
                        if done_n % 10 == 0:
                            save_jsonl(results, output_path)
                except Exception as e:
                    print(f"    ERROR {p['query_id']}: {e}")

    save_jsonl(results, output_path)
    n_done = sum(1 for r in results if r.get("all_collected"))
    n_skip = sum(1 for r in results if r.get("skipped"))
    print(f"\nSaved {len(results)}: complete={n_done} "
          f"({n_done/max(len(results),1):.0%}), skipped={n_skip}")


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
                    default=Path("data/images_siglip_top1/collection_log.jsonl"))
    ap.add_argument("--image_root", type=Path,
                    default=Path("data/images_siglip_top1"))
    ap.add_argument("--client-url", default="http://127.0.0.1:1235/knn-service")
    ap.add_argument("--indice-name", default="pod_fashion")
    ap.add_argument("--vlm-urls", default="",
                    help="comma-separated VLM endpoints; empty = pure top-1 (no gender, no pattern)")
    ap.add_argument("--vlm-model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--pattern-mode", choices=["off", "specify", "verify", "both"],
                    default="off",
                    help="pattern-axis full-coverage strategy (color axis untouched)")
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
    vlm_urls = [u.strip() for u in args.vlm_urls.split(",") if u.strip()] or None
    run(args.plan_path, args.output, args.image_root,
        args.client_url, args.indice_name, args.limit, args.top_k, args.workers,
        vlm_urls=vlm_urls, vlm_model=args.vlm_model, pattern_mode=args.pattern_mode)


if __name__ == "__main__":
    main()