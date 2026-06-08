#!/usr/bin/env python3
"""
collect_images_siglip_rerank_v4.py  —  RERANK + gender + intra-set URL dedup
---------------------------------------------------------------------------
FashionSigLIP collector with a 2-STAGE retrieve-then-rerank pipeline. SigLIP does
the initial top-k recall (frozen); then a dedicated multimodal reranker
(Qwen3-VL-Reranker-8B, served by vLLM with the /rerank endpoint) scores each
eligible candidate image against THAT option's spec sentence, and we take the
argmax. This replaces the v3 best-of-k VLM *generation* choice with a
deterministic reranker *score* - no answer parsing, no ReAct.

It does NOT look at scenario/TPO - matching the occasion is the benchmark subject
model's job; here we only rank how well each image matches the garment spec.

Constraints retained from v2:

  1. GENDER consistency across the four options A-D (single-word VLM check,
     fail-open). The first option that yields a confident man/woman locks the
     set gender; later options that conflict walk down to the next candidate.

  2. INTRA-SET URL DEDUP: the four options of ONE query must be four DIFFERENT
     images. A candidate URL already used by an earlier option in the same query
     is skipped (judged on the URL BEFORE downloading, so no wasted bandwidth /
     VLM call). Dedup is per-query only; different queries may reuse a URL.

Plus ONE retrieval-side touch (no VLM): on the PATTERN axis, for any option whose
target pattern is non-solid, an all-over phrase is appended to the search query so
the retriever ranks WHOLE-BODY patterns above localized (sleeve/trim-only) ones.
This is query specification only - it changes the text sent to the KNN server, not
the candidate filtering. color axis and solid options are never touched.

No VLM pattern check, no best-of-k. Pure top-1 walk that short-circuits only on:
dead URL, duplicate URL (within the query), or gender conflict. All VLM use is
fail-open.

Servers:  KNN serve_fsiglip_knn.py :1235 ; VLM OpenAI-compat :8002..:8005

Usage:
python src/collect_images_siglip_rerank_v4.py \
  --plan_path data/options/option_plans.jsonl \
  --client-url http://127.0.0.1:1235/knn-service \
  --vlm-urls "http://127.0.0.1:8002/v1" --vlm-model Qwen/Qwen3-VL-4B-Instruct \
  --rerank-url http://127.0.0.1:8010/v1 --rerank-model Qwen/Qwen3-VL-Reranker-8B \
  --image_root data/images_siglip_rerank \
  --output data/images_siglip_rerank/collection_log.jsonl \
  --top_k 5 --k-pool 5 --workers 4 --limit 100 --force

  # pure top-1 + dedup, NO gender (omit --vlm-urls):
  python src/collect_images_siglip_top1_v2.py --client-url ... --top_k 12
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

UA = {"User-Agent": "Mozilla/5.0 (compatible; POD-Bench-SigLIP/2.0)"}

_SESSION = requests.Session()
_SESSION.headers.update(UA)
_adapter = HTTPAdapter(pool_connections=32, pool_maxsize=64, max_retries=0)
_SESSION.mount("http://", _adapter)
_SESSION.mount("https://", _adapter)

SOURCE_TAG = "fashionsiglip_top1"

# pattern values that are NOT a real (non-solid) pattern
_SOLID_LIKE = {"", "solid", "plain", "none"}


def _clean(s):
    return (s or "").strip().lower().replace("_", " ")


def specify_query(garment, target_color=None, target_pattern=None,
                  axis=None, fallback_query=None):
    """Build ONE natural-language CLIP prompt (not a comma list).

    CLIP/SigLIP text encoders were trained on caption-like sentences and are
    robust to short natural phrasing ("a photo of a {label}"), not to long
    keyword pile-ups that blow past the ~77-token limit. So we compose a single
    sentence template:

        "a full-body product photo of a {color} {garment}[ with an all-over
         {pattern} print covering the whole garment], the entire garment
         clearly visible and not cropped"

    - color adjective: included whenever a target color is known.
    - all-over pattern clause: ONLY on the pattern axis with a non-solid target
      (color axis / solid -> omitted, so the sentence stays clean).
    - framing ("full-body product photo ... not cropped") is baked into the
      template itself, not tacked on as keywords.

    If no garment label is available, fall back to the original search query
    wrapped minimally as "a full-body product photo of {query}, not cropped"."""
    g = _clean(garment)
    c = _clean(target_color)
    tp = _clean(target_pattern)

    if not g:
        base = _clean(fallback_query) or "a clothing item"
        return f"a full-body product photo of {base}, the whole item visible and not cropped"

    head = "a full-body product photo of a"
    color_adj = f" {c}" if c and c not in ("none", "unknown") else ""
    pattern_clause = ""
    if axis == "pattern" and tp not in _SOLID_LIKE:
        pattern_clause = f" with an all-over {tp} print covering the whole garment"

    return (f"{head}{color_adj} {g}{pattern_clause}, "
            f"the entire garment clearly visible and not cropped")


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


# -- single VLM call: gender only --
_GENDER_SYSTEM = (
    "You see one clothing product image. Reply with ONLY one word: "
    "man, woman, or unclear - the gender of the person modeling the garment. "
    "If it is a flat-lay / mannequin / no person, reply unclear."
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


def vlm_gender(vlm_url, model, image_path):
    """Returns gender in man/woman/unclear. Fail-open -> 'unclear'."""
    txt = _post_vlm(vlm_url, model, _GENDER_SYSTEM,
                    "man, woman, or unclear?", image_path, max_tokens=6)
    return _word_to_gender(txt)


# -- reranker: score each candidate image against the option spec, take argmax --
def _spec_text(attrs):
    g = (attrs.get("garment_category") or "").replace("_", " ")
    c = attrs.get("color") or "any color"
    p = (attrs.get("pattern") or "solid").replace("_", " ")
    return (f"a {c} {g} with {p} pattern, the whole garment clearly visible "
            f"and not cropped")


def reranker_best(rerank_url, rerank_model, image_paths, attrs,
                  timeout=120, retries=2):
    """POST the option spec (query) + candidate images (documents) to a vLLM
    /rerank endpoint; return the index of the highest-scoring candidate, plus the
    list of scores. fail-open -> (None, None) so caller falls back to first."""
    query = _spec_text(attrs)
    documents = []
    for p in image_paths:
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        documents.append({"content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]})
    payload = {"model": rerank_model, "query": query, "documents": documents}
    url = f"{rerank_url.rstrip('/')}/rerank"
    for attempt in range(retries):
        try:
            r = _SESSION.post(url, json=payload, timeout=timeout)
            if r.status_code != 200:
                time.sleep(0.5 * (attempt + 1)); continue
            data = r.json()
            results = data.get("results") or data.get("data") or []
            if not results:
                return None, None
            # results carry original index + relevance_score; map back to order
            scores = [None] * len(image_paths)
            for item in results:
                idx = item.get("index")
                sc = item.get("relevance_score", item.get("score"))
                if idx is not None and 0 <= idx < len(scores):
                    scores[idx] = sc
            # argmax over present scores
            best_i, best_s = None, None
            for i, s in enumerate(scores):
                if s is None:
                    continue
                if best_s is None or s > best_s:
                    best_s, best_i = s, i
            return best_i, scores
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    return None, None


def _cand_url(c):
    return c.get("url") or c.get("image_url")


def resolve_option_bestofk(query, img_path, client_url, top_k, indice_name,
                           vlm_url, vlm_model, target_gender, used_urls,
                           attrs, k_pool, rerank_url, rerank_model):
    """Gather up to k_pool eligible candidates (downloadable, NEW url within the
    query, gender-consistent), then RERANK them against `attrs` and take argmax.
    Falls back to the first eligible candidate if the reranker call fails.
    Returns (option_result, seen_gender, chosen_url).

    The downloaded candidate images are kept in a temp pool; the chosen one is
    saved to img_path. fail-open everywhere."""
    cands = knn_search(client_url, query, top_k=top_k, indice_name=indice_name)

    pool = []  # list of dicts: {url, rank, caption, similarity, gender, tmp_path}
    pool_dir = img_path.parent / ".pool"
    pool_dir.mkdir(parents=True, exist_ok=True)

    for rank, c in enumerate(cands):
        if len(pool) >= k_pool:
            break
        url = _cand_url(c)
        if not url or url in used_urls:
            continue
        tmp = pool_dir / f"{img_path.stem}_{rank}.jpg"
        if not download_image(url, tmp):
            continue
        g = "unclear"
        if vlm_url:
            g = vlm_gender(vlm_url, vlm_model, tmp)
            if target_gender in ("man", "woman") and g in ("man", "woman") and g != target_gender:
                try: tmp.unlink()
                except Exception: pass
                continue                 # gender conflict -> not eligible
        pool.append({"url": url, "rank": rank, "caption": c.get("caption", "")[:120],
                     "similarity": c.get("similarity"), "gender": g, "tmp": tmp})

    if not pool:
        return ({"image_path": None, "source": "FAILED", "search_query": query}, None, None)

    # choose: reranker argmax over the pool
    chosen_i = 0
    picked_by = "fallback_first"
    rr_score = None
    if rerank_url and len(pool) > 1:
        bi, scores = reranker_best(rerank_url, rerank_model,
                                   [str(p["tmp"]) for p in pool], attrs)
        if bi is not None:
            chosen_i = bi
            picked_by = "reranker"
            rr_score = scores[bi] if scores else None
    chosen = pool[chosen_i]

    # promote chosen tmp -> img_path; clean the rest
    try:
        if img_path.exists():
            img_path.unlink()
        Path(chosen["tmp"]).replace(img_path)
    except Exception:
        # last-resort: re-download chosen
        download_image(chosen["url"], img_path)
    for p in pool:
        if p is not chosen:
            try: Path(p["tmp"]).unlink()
            except Exception: pass
    try:
        pool_dir.rmdir()
    except Exception:
        pass

    rec = {"image_path": str(img_path), "source": SOURCE_TAG,
           "search_query": query, "rank_used": chosen["rank"], "url": chosen["url"],
           "source_title": chosen["caption"], "similarity": chosen["similarity"],
           "picked_by": picked_by, "pool_size": len(pool), "rerank_score": rr_score}
    if vlm_url:
        rec["model_gender"] = chosen["gender"]
    return (rec, chosen["gender"], chosen["url"])


def collect_for_plan(plan, image_root, client_url, top_k, indice_name,
                     vlm_url, vlm_model, k_pool, rerank_url, rerank_model):
    qid = plan["query_id"]
    out_dir = Path(image_root) / qid
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "query_id": qid, "user_id": plan.get("user_id"),
        "domain": plan.get("domain", "fashion"),
        "active_axis": plan.get("active_axis"),
        "main_category": plan.get("main_category"),
        "scenario_archetype": plan.get("scenario_archetype"),
        "options": {}, "all_collected": False, "skipped": False,
        "set_gender": None, "gender_consistent": True,
    }

    active_axis = plan.get("active_axis")
    n_ok = 0
    target_gender = None
    used_urls = set()                  # per-query URL dedup
    genders_seen = []
    for k in "ABCD":
        opt = plan["options"][k]
        attrs = opt.get("attributes", {}) or {}
        # build ONE natural-language CLIP prompt from the option's attributes:
        #   - garment label + color adjective always
        #   - all-over pattern clause only on pattern axis + non-solid
        #   - full-body / not-cropped framing baked into the template
        target_color = attrs.get("color")
        target_pattern = attrs.get("pattern")
        query = specify_query(attrs.get("garment_category"),
                              target_color=target_color,
                              target_pattern=target_pattern,
                              axis=active_axis,
                              fallback_query=opt.get("search_query"))
        img_path = out_dir / f"{k}.jpg"
        res, seen, url = resolve_option_bestofk(
            query, img_path, client_url, top_k, indice_name,
            vlm_url, vlm_model, target_gender, used_urls,
            attrs, k_pool, rerank_url, rerank_model)
        result["options"][k] = res
        if res["image_path"]:
            n_ok += 1
            if url:
                used_urls.add(url)
            if seen in ("man", "woman"):
                genders_seen.append(seen)
                if target_gender is None:
                    target_gender = seen

    result["set_gender"] = target_gender
    # consistency = all confident genders agree with the locked target
    if target_gender and genders_seen:
        result["gender_consistent"] = all(g == target_gender for g in genders_seen)
    result["all_collected"] = (n_ok == 4)
    result["skipped"] = (n_ok < 4)
    return result


def run(plan_path, output_path, image_root, client_url, indice_name,
        limit, top_k, workers, vlm_urls, vlm_model, k_pool, rerank_url, rerank_model):
    plans = load_jsonl(plan_path)
    print(f"Loaded {len(plans)} plans | server={client_url} top_k={top_k} "
          f"(gender={'on' if vlm_urls and vlm_urls[0] else 'off'}, dedup=on, "
          f"rerank={'on' if rerank_url else 'off'})")
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
                                vlm_url, vlm_model, k_pool, rerank_url, rerank_model)

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
    n_gincons = sum(1 for r in results if r.get("gender_consistent") is False)
    print(f"\nSaved {len(results)}: complete={n_done} "
          f"({n_done/max(len(results),1):.0%}), skipped={n_skip}, "
          f"gender_inconsistent={n_gincons}")


def selftest(client_url, indice_name, top_k):
    for q in ["beige plaid shirt", "white polka dot t shirt", "navy coat"]:
        cands = knn_search(client_url, q, top_k=top_k, indice_name=indice_name)
        print(f"\nquery: {q!r}  -> {len(cands)} candidates")
        for i, c in enumerate(cands[:5]):
            print(f"  [{i}] {c.get('similarity')} | {_cand_url(c)} | {c.get('caption','')[:60]}")


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
                    help="comma-separated VLM endpoints; empty = top-1 + dedup, no gender")
    ap.add_argument("--vlm-model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top_k", type=int, default=12,
                    help="candidates fetched from KNN per option")
    ap.add_argument("--k-pool", type=int, default=5,
                    help="max eligible candidates passed to the reranker")
    ap.add_argument("--rerank-url", default="http://127.0.0.1:8010/v1",
                    help="vLLM /rerank endpoint base (Qwen3-VL-Reranker). empty = no rerank (first eligible)")
    ap.add_argument("--rerank-model", default="Qwen/Qwen3-VL-Reranker-8B")
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
        vlm_urls=vlm_urls, vlm_model=args.vlm_model, k_pool=args.k_pool,
        rerank_url=args.rerank_url, rerank_model=args.rerank_model)


if __name__ == "__main__":
    main()
