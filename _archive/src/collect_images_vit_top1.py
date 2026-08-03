#!/usr/bin/env python3
"""
collect_images_vit_top1_v2.py  —  TOP-1 + gender + intra-set URL dedup
---------------------------------------------------------------------
TOP-1 ViT-L/14 CLIP collector (identical logic to the FashionSigLIP v2;
only the KNN backbone/port differs) with exactly TWO quality constraints, nothing else:

  1. GENDER consistency across the four options A-D (single-word VLM check,
     fail-open). The first option that yields a confident man/woman locks the
     set gender; later options that conflict walk down to the next candidate.

  2. INTRA-SET URL DEDUP: the four options of ONE query must be four DIFFERENT
     images. A candidate URL already used by an earlier option in the same query
     is skipped (judged on the URL BEFORE downloading, so no wasted bandwidth /
     VLM call). Dedup is per-query only; different queries may reuse a URL.

No pattern-coverage logic, no query rewriting, no best-of-k. Pure top-1 walk that
short-circuits only on: dead URL, duplicate URL (within the query), or gender
conflict. All VLM use is fail-open.

Servers:  KNN clip-retrieval back (ViT-L/14) :1234 ; VLM OpenAI-compat :8002..:8005

Usage:
  python src/collect_images_vit_top1.py \
    --client-url http://127.0.0.1:1234/knn-service \
    --vlm-urls "http://127.0.0.1:8002/v1,http://127.0.0.1:8003/v1" \
    --vlm-model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --image_root data/images_vit_top1 \
    --output data/images_vit_top1/collection_log.jsonl \
    --workers 4 --top_k 12 --force

  # pure top-1 + dedup, NO gender (omit --vlm-urls):
  python src/collect_images_vit_top1_v2.py --client-url ... --top_k 12
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

UA = {"User-Agent": "Mozilla/5.0 (compatible; POD-Bench-ViT/2.0)"}

_SESSION = requests.Session()
_SESSION.headers.update(UA)
_adapter = HTTPAdapter(pool_connections=32, pool_maxsize=64, max_retries=0)
_SESSION.mount("http://", _adapter)
_SESSION.mount("https://", _adapter)

SOURCE_TAG = "vit_top1"


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


def _cand_url(c):
    return c.get("url") or c.get("image_url")


def resolve_option_top1(query, img_path, client_url, top_k, indice_name,
                        vlm_url, vlm_model, target_gender, used_urls):
    """Top-1 that (a) downloads, (b) is a NEW url within this query, and
    (c) is gender-consistent. Walks down only on: dead/dup url or gender
    conflict. fail-open on VLM. Returns (option_result, seen_gender, chosen_url)."""
    cands = knn_search(client_url, query, top_k=top_k, indice_name=indice_name)

    for rank, c in enumerate(cands):
        url = _cand_url(c)
        if not url:
            continue
        if url in used_urls:           # intra-set duplicate -> skip BEFORE download
            continue
        if not download_image(url, img_path):
            continue

        if not vlm_url:                # pure top-1 + dedup (no gender)
            return ({"image_path": str(img_path), "source": SOURCE_TAG,
                     "search_query": query, "rank_used": rank, "url": url,
                     "source_title": c.get("caption", "")[:120],
                     "similarity": c.get("similarity")}, "unclear", url)

        g = vlm_gender(vlm_url, vlm_model, img_path)
        if target_gender in ("man", "woman") and g in ("man", "woman") and g != target_gender:
            continue                   # gender conflict -> next candidate

        return ({"image_path": str(img_path), "source": SOURCE_TAG,
                 "search_query": query, "rank_used": rank, "url": url,
                 "source_title": c.get("caption", "")[:120],
                 "similarity": c.get("similarity"), "model_gender": g}, g, url)

    return ({"image_path": None, "source": "FAILED", "search_query": query}, None, None)


def collect_for_plan(plan, image_root, client_url, top_k, indice_name,
                     vlm_url, vlm_model):
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

    n_ok = 0
    target_gender = None
    used_urls = set()                  # per-query URL dedup
    genders_seen = []
    for k in "ABCD":
        opt = plan["options"][k]
        query = opt["search_query"]
        img_path = out_dir / f"{k}.jpg"
        res, seen, url = resolve_option_top1(
            query, img_path, client_url, top_k, indice_name,
            vlm_url, vlm_model, target_gender, used_urls)
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
        limit, top_k, workers, vlm_urls, vlm_model):
    plans = load_jsonl(plan_path)
    print(f"Loaded {len(plans)} plans | server={client_url} top_k={top_k} "
          f"(gender={'on' if vlm_urls and vlm_urls[0] else 'off'}, intra-set dedup=on)")
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
                                vlm_url, vlm_model)

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
                    default=Path("data/images_vit_top1/collection_log.jsonl"))
    ap.add_argument("--image_root", type=Path,
                    default=Path("data/images_vit_top1"))
    ap.add_argument("--client-url", default="http://127.0.0.1:1234/knn-service")
    ap.add_argument("--indice-name", default="pod_fashion")
    ap.add_argument("--vlm-urls", default="",
                    help="comma-separated VLM endpoints; empty = top-1 + dedup, no gender")
    ap.add_argument("--vlm-model", default="Qwen/Qwen3-VL-4B-Instruct")
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
        vlm_urls=vlm_urls, vlm_model=args.vlm_model)


if __name__ == "__main__":
    main()