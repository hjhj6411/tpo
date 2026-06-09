#!/usr/bin/env python3
"""
collect_images_siglip_coverage_v6.py — coverage-first TOP-1 + gender + dedup
----------------------------------------------------------------------------
Builds on v2 (top-1 + gender consistency + intra-set URL dedup), but fixes the
localized-vs-all-over pattern problem with a PATCH-COVERAGE re-rank.

Per option (PATTERN axis, non-solid target only):
  1. retrieve top-k from FashionSigLIP using the C1_allover query.
  2. score each candidate's patch coverage via the server /patch-coverage
     endpoint (NxN grid, each tile classified "{pattern} fabric" vs "plain",
     coverage = patterned_tiles / non-background_tiles), against the option's
     OWN target pattern/color/garment.
  3. RE-RANK candidates by (coverage desc, retrieval sim desc).
  4. walk that order applying the SAME gender + dedup gates as v2; take the
     first survivor.

Coverage cleanly separates whole-garment patterns (cov ~1.0) from localized /
faint / mis-collected ones (cov low), which the global SigLIP similarity cannot
do (FashionVLP, CVPR'22: a single average-pooled embedding loses location
salience). Verified on striped / plaid / checkered.

color-axis and solid options skip coverage entirely and use retrieval order, as
coverage is meaningless without a real pattern.

Servers:  KNN+coverage serve_fsiglip_knn.py :1235 ; VLM OpenAI-compat :8002..
Defaults: --top_k 12 --grid 5

Usage:
python src/collect_images_siglip_coverage_v6.py \
  --plan_path data/options/option_plans.jsonl \
  --client-url   http://127.0.0.1:1235/knn-service \
  --coverage-url http://127.0.0.1:1235/patch-coverage \
  --vlm-urls "http://127.0.0.1:8002/v1,http://127.0.0.1:8003/v1,http://127.0.0.1:8004/v1,http://127.0.0.1:8005/v1" --vlm-model Qwen/Qwen3-VL-30B-A3B-Instruct \
  --image_root data/images_siglip_cov --top_k 12 --grid 5 \
  --limit 20 --workers 4 --verbose --force
  # coverage + dedup, NO gender (omit --vlm-urls)
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

UA = {"User-Agent": "Mozilla/5.0 (compatible; POD-Bench-SigLIP-cov/6.0)"}

_SESSION = requests.Session()
_SESSION.headers.update(UA)
_adapter = HTTPAdapter(pool_connections=32, pool_maxsize=64, max_retries=0)
_SESSION.mount("http://", _adapter)
_SESSION.mount("https://", _adapter)


SOURCE_TAG = "fashionsiglip_cov"

# pattern values that are NOT a real (non-solid) pattern
_SOLID_LIKE = {"", "solid", "plain", "none"}


def _clean(s):
    return (s or "").strip().lower().replace("_", " ")


def specify_query(garment, target_color=None, target_pattern=None,
                  axis=None, fallback_query=None):
    """C1_allover query (validated by the coverage diagnostics): the proven
    all-over phrase on the T4 'studio product shot' skeleton.

        "a {color} {garment} with an all-over {pattern} pattern,
         studio product shot, not cropped"

    - color adjective when known.
    - all-over clause ONLY on the pattern axis with a non-solid target.
    - no garment label -> minimal wrapper on the raw query."""
    g = _clean(garment)
    c = _clean(target_color)
    tp = _clean(target_pattern)

    if not g:
        base = _clean(fallback_query) or "a clothing item"
        return f"a {base}, studio product shot, not cropped"

    color_adj = f"{c} " if c and c not in ("none", "unknown") else ""
    pattern_clause = ""
    if axis == "pattern" and tp not in _SOLID_LIKE:
        pattern_clause = f" with an all-over {tp} pattern"
    return (f"a {color_adj}{g}{pattern_clause}, "
            f"studio product shot, not cropped").replace("  ", " ")


def coverage_score_urls(coverage_url, urls, pattern, color, garment, grid):
    """Score patch coverage for a list of candidate URLs via the server's
    /patch-coverage endpoint. Returns list aligned to `urls` of coverage floats
    (None if the server couldn't score that url). Fail-open: on any error,
    returns all-None so the caller falls back to retrieval order."""
    if not urls:
        return []
    try:
        resp = _SESSION.post(coverage_url, json={
            "images": list(urls), "pattern": pattern, "color": color,
            "garment": garment, "grid": grid, "drop_white": True}, timeout=300)
        resp.raise_for_status()
        res = resp.json()
        out = []
        for r in res:
            if isinstance(r, dict) and "coverage" in r:
                out.append(r.get("coverage"))
            else:
                out.append(None)
        # pad/truncate to match urls length defensively
        if len(out) < len(urls):
            out += [None] * (len(urls) - len(out))
        return out[:len(urls)]
    except Exception as e:
        print(f"    [coverage] failed ({len(urls)} urls): {e}")
        return [None] * len(urls)


def coverage_rerank(cands, coverages):
    """Return candidate indices ordered by (coverage desc, similarity desc).
    `coverages` is aligned to `cands`. Candidates with no coverage (None) sort
    LAST but keep similarity order among themselves. A coverage of None is
    treated as -1 so any scored candidate outranks an unscored one."""
    idx = list(range(len(cands)))

    def sim_of(i):
        s = cands[i].get("similarity")
        return s if isinstance(s, (int, float)) else float("-inf")

    def cov_of(i):
        c = coverages[i] if i < len(coverages) else None
        return c if isinstance(c, (int, float)) else -1.0

    idx.sort(key=lambda i: (-cov_of(i), -sim_of(i)))
    return idx


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


def _print_rank_table(opt_label, query, cands, coverages, order, chosen_rank,
                      skipped):
    """Print the coverage-first ranking table for one option (verbose mode).
    `skipped` maps rank -> reason ('dup'/'dead'/'gender') for non-chosen rows."""
    print(f"    [{opt_label}] query: {query}")
    print(f"      {'new':>3} {'retr':>4} {'cov':>5} {'sim':>8}  pick  caption")
    for nr, rank in enumerate(order):
        c = cands[rank]
        cov = coverages[rank] if rank < len(coverages) else None
        covs = "  -- " if not isinstance(cov, (int, float)) else f"{cov:.2f}"
        sim = c.get("similarity")
        sims = f"{sim:.4f}" if isinstance(sim, (int, float)) else "  --  "
        if rank == chosen_rank:
            mark = "  <=PICK"
        elif rank in skipped:
            mark = f"  x{skipped[rank]}"
        else:
            mark = ""
        cap = (c.get("caption") or "")[:42]
        print(f"      {nr:>3} {rank:>4} {covs:>5} {sims:>8}{mark:>9}  {cap}")


def resolve_option_top1(query, img_path, client_url, top_k, indice_name,
                        vlm_url, vlm_model, target_gender, used_urls,
                        coverage_url=None, cov_pattern=None, cov_color=None,
                        cov_garment=None, grid=5, verbose=False, opt_label=""):
    """Retrieve top-k, optionally RE-RANK by patch coverage (coverage desc, then
    retrieval sim desc), then walk that order applying dedup + gender gates.
    Returns (option_result, seen_gender, chosen_url).

    Coverage is applied only when coverage_url AND a non-solid cov_pattern are
    given (caller passes these for pattern-axis non-solid options only). On any
    coverage error it falls back to plain retrieval order (fail-open).

    verbose -> print the query and the coverage-first ranking table, marking the
    chosen candidate and why others were skipped."""
    cands = knn_search(client_url, query, top_k=top_k, indice_name=indice_name)
    if not cands:
        if verbose:
            print(f"    [{opt_label}] query: {query}\n      (no candidates)")
        return ({"image_path": None, "source": "FAILED", "search_query": query}, None, None)

    # coverage re-rank (pattern-axis non-solid only)
    coverages = [None] * len(cands)
    use_cov = bool(coverage_url and cov_pattern and cov_pattern not in _SOLID_LIKE)
    if use_cov:
        urls = [_cand_url(c) or "" for c in cands]
        coverages = coverage_score_urls(coverage_url, urls, cov_pattern,
                                        cov_color or "", cov_garment or "garment", grid)
        order = coverage_rerank(cands, coverages)
    else:
        order = list(range(len(cands)))

    skipped = {}                       # rank -> short reason, for verbose table
    chosen = None
    result = ({"image_path": None, "source": "FAILED", "search_query": query}, None, None)
    for rank in order:
        c = cands[rank]
        url = _cand_url(c)
        if not url:
            skipped[rank] = "dead"; continue
        if url in used_urls:           # intra-set duplicate -> skip BEFORE download
            skipped[rank] = "dup"; continue
        if not download_image(url, img_path):
            skipped[rank] = "dead"; continue

        base = {"image_path": str(img_path), "source": SOURCE_TAG,
                "search_query": query, "rank_used": rank, "url": url,
                "source_title": c.get("caption", "")[:120],
                "similarity": c.get("similarity"),
                "coverage": coverages[rank] if rank < len(coverages) else None,
                "cov_reranked": use_cov}

        if not vlm_url:                # coverage/top-1 + dedup (no gender)
            chosen = rank; result = (base, "unclear", url); break

        g = vlm_gender(vlm_url, vlm_model, img_path)
        if target_gender in ("man", "woman") and g in ("man", "woman") and g != target_gender:
            skipped[rank] = "gender"; continue   # gender conflict -> next candidate
        base["model_gender"] = g
        chosen = rank; result = (base, g, url); break

    if verbose:
        _print_rank_table(opt_label, query, cands, coverages, order, chosen, skipped)
    return result


def collect_for_plan(plan, image_root, client_url, top_k, indice_name,
                     vlm_url, vlm_model, coverage_url=None, grid=5, verbose=False):
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
        target_color = attrs.get("color")
        target_pattern = attrs.get("pattern")
        target_garment = attrs.get("garment_category")
        # C1_allover query (all-over clause only on pattern axis + non-solid)
        query = specify_query(target_garment,
                              target_color=target_color,
                              target_pattern=target_pattern,
                              axis=active_axis,
                              fallback_query=opt.get("search_query"))
        # coverage re-rank only for pattern-axis non-solid options
        cov_pat = _clean(target_pattern) if active_axis == "pattern" else None
        img_path = out_dir / f"{k}.jpg"
        res, seen, url = resolve_option_top1(
            query, img_path, client_url, top_k, indice_name,
            vlm_url, vlm_model, target_gender, used_urls,
            coverage_url=coverage_url, cov_pattern=cov_pat,
            cov_color=_clean(target_color), cov_garment=_clean(target_garment),
            grid=grid, verbose=verbose, opt_label=f"{qid} {k}")
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
        limit, top_k, workers, vlm_urls, vlm_model,
        coverage_url=None, grid=5, verbose=False):
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
        vlm_url = vlm_urls[idx % len(vlm_urls)] if vlm_urls else None
        return collect_for_plan(plan, image_root, client_url, top_k, indice_name,
                                vlm_url, vlm_model,
                                coverage_url=coverage_url, grid=grid, verbose=verbose)

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
                    default=Path("data/images_siglip_cov/collection_log.jsonl"))
    ap.add_argument("--image_root", type=Path,
                    default=Path("data/images_siglip_cov"))
    ap.add_argument("--client-url", default="http://127.0.0.1:1235/knn-service")
    ap.add_argument("--coverage-url", default="http://127.0.0.1:1235/patch-coverage",
                    help="patch-coverage endpoint; set empty to disable coverage re-rank")
    ap.add_argument("--indice-name", default="pod_fashion")
    ap.add_argument("--vlm-urls", default="",
                    help="comma-separated VLM endpoints; empty = coverage + dedup, no gender")
    ap.add_argument("--vlm-model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top_k", type=int, default=12)
    ap.add_argument("--grid", type=int, default=5, help="NxN patch grid for coverage")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verbose", action="store_true",
                    help="print per-option query + coverage-first ranking table "
                         "(what was retrieved, scored, and picked). Use with "
                         "--workers 1 for readable output.")
    args = ap.parse_args()

    if args.selftest:
        selftest(args.client_url, args.indice_name, args.top_k)
        return
    if args.force and Path(args.output).exists():
        Path(args.output).unlink()
    vlm_urls = [u.strip() for u in args.vlm_urls.split(",") if u.strip()] or None
    coverage_url = args.coverage_url.strip() or None
    run(args.plan_path, args.output, args.image_root,
        args.client_url, args.indice_name, args.limit, args.top_k, args.workers,
        vlm_urls=vlm_urls, vlm_model=args.vlm_model,
        coverage_url=coverage_url, grid=args.grid, verbose=args.verbose)


if __name__ == "__main__":
    main()