#!/usr/bin/env python3
"""
collect_images_vit_coverage_v7.py — coverage-first TOP-1 + COLOR coverage
--------------------------------------------------------------------------
v6 (ViT retrieval :1234 + FSigLIP patch-coverage :1235 + gender + dedup) PLUS a
COLOR patch-coverage re-rank that targets the known ViT-L/14 color weakness
(Navy<->Blue, Beige -> Ivory/Brown). Pattern scoring is UNCHANGED from v6.

WHY COLOR COVERAGE NEEDS A 13-WAY ARGMAX
  The pattern gate works as a binary "{pattern} fabric vs plain fabric" argmax.
  Color cannot: a blue tile is closer to "navy fabric" than to "plain fabric",
  so a binary anchor passes it. The server's /patch-color-coverage instead
  argmaxes each tile over ALL vocabulary colors (pattern held constant in the
  anchor text), so a blue tile loses to the "blue" anchor and counts AGAINST a
  navy target. coverage = target-color tiles / non-background tiles.

ROUTING (--color-cov, default "color" — one knob at a time):
  off   : exact v6 behavior. color cov never scored.
  color : COLOR-axis options re-ranked by (color_cov desc, sim desc).
          PATTERN-axis options keep the proven v6 order (pattern_cov, sim).
  all   : color cov everywhere —
            color axis            : (color_cov, sim)
            pattern axis nonsolid : combined score = pcov * ccov desc, then sim
            pattern axis solid    : (color_cov, sim)
          NOTE: "all" scores every candidate on BOTH endpoints; the server's
          URL->PIL cache avoids downloading each image twice.

PATTERN × COLOR COMBINED SCORE (pattern axis non-solid, color-cov != off):
  score = pcov * ccov
  Rationale: "black camouflage" requires BOTH camo pattern AND black color.
  Lexicographic (pcov, ccov) fails because pcov 1.0 (army-green camo) always
  beats pcov 0.85 (black camo), and ccov never gets a chance as tie-breaker.
  The product score makes pcov=1.0/ccov=0.09=0.09 lose to pcov=0.85/ccov=0.95=0.81.
  color axis and pattern-solid still use the single relevant coverage score.

The option's PATTERN CONTEXT is always sent to the color endpoint (on the color
axis this is the fixed pattern; on the pattern axis it is the option's own
target pattern), so the server can hold the pattern constant in its anchors —
this is how the base-color ambiguity of checked/striped garments is handled:
anchors differ on color only, and re-ranking needs only RELATIVE order.

All scoring is fail-open: any coverage error falls back to retrieval order.

Servers:  ViT clip-retrieval back :1234 ; SigLIP coverage :1235 ; VLM :8002..
Defaults: --top_k 12 --grid 5 --color-cov color

Usage:
python vit/collect_images_vit_coverage_v7.py \
--plan_path data/options/option_plans.jsonl \
--client-url       http://127.0.0.1:1234/knn-service \
--coverage-url     http://127.0.0.1:1235/patch-coverage \
--color-coverage-url http://127.0.0.1:1235/patch-color-coverage \
--vlm-urls "http://127.0.0.1:8002/v1,http://127.0.0.1:8003/v1" \
--vlm-model Qwen/Qwen3-VL-30B-A3B-Instruct \
--image_root data/images_vit_cov_c \
--output data/images_vit_cov_c/collection_log.jsonl \
--top_k 12 --grid 5 --workers 1 --verbose --color-cov all --force --limit 10

  # color cov + dedup, NO gender (omit --vlm-urls)
  # exact v6 reproduction: --color-cov off
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

UA = {"User-Agent": "Mozilla/5.0 (compatible; POD-Bench-ViT-cov/7.0)"}

_SESSION = requests.Session()
_SESSION.headers.update(UA)
_adapter = HTTPAdapter(pool_connections=32, pool_maxsize=64, max_retries=0)
_SESSION.mount("http://", _adapter)
_SESSION.mount("https://", _adapter)


SOURCE_TAG = "vit_covc"

# pattern values that are NOT a real (non-solid) pattern
_SOLID_LIKE = {"", "solid", "plain", "none"}


def _clean(s):
    return (s or "").strip().lower().replace("_", " ")


def specify_query(garment, target_color=None, target_pattern=None,
                  axis=None, fallback_query=None):
    g = _clean(garment)
    c = _clean(target_color)
    tp = _clean(target_pattern)

    if not g:
        base = _clean(fallback_query) or "clothing"
        return (
            f"a fashion catalog photo of a person wearing {base}, "
            f"entire garment visible, plain background"
        ).replace("  ", " ")

    color_adj = f"{c} " if c and c not in ("none", "unknown") else ""
    garment_core = f"{color_adj}{g}".replace("  ", " ").strip()

    if axis == "pattern" and tp not in _SOLID_LIKE:
        return (
            f"a fashion catalog photo of a person wearing "
            f"a {garment_core} with an all-over {tp} pattern, "
            f"front view, entire garment visible, plain background"
        ).replace("  ", " ")

    return (
        f"a fashion catalog photo of a person wearing "
        f"a {garment_core}, front view, entire garment visible, plain background"
    ).replace("  ", " ")


def coverage_score_urls(coverage_url, urls, pattern, color, garment, grid):
    """Score PATTERN patch coverage for candidate URLs via /patch-coverage.
    Returns list aligned to `urls` of coverage floats (None if the server
    couldn't score that url). Fail-open: on any error, returns all-None so the
    caller falls back to retrieval order."""
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


def color_coverage_score_urls(color_coverage_url, urls, color, pattern, garment, grid):
    """Score COLOR patch coverage for candidate URLs via /patch-color-coverage.
    `pattern` is the option's pattern context (held constant in the server's
    anchors so the per-tile argmax varies on color only). Same alignment and
    fail-open semantics as coverage_score_urls."""
    if not urls:
        return []
    try:
        resp = _SESSION.post(color_coverage_url, json={
            "images": list(urls), "color": color, "pattern": pattern,
            "garment": garment, "grid": grid, "drop_white": True}, timeout=300)
        resp.raise_for_status()
        res = resp.json()
        out = []
        for r in res:
            if isinstance(r, dict) and "coverage" in r:
                out.append(r.get("coverage"))
            else:
                out.append(None)
        if len(out) < len(urls):
            out += [None] * (len(urls) - len(out))
        return out[:len(urls)]
    except Exception as e:
        print(f"    [color-coverage] failed ({len(urls)} urls): {e}")
        return [None] * len(urls)


def _combined_score(pcov_val, ccov_val):
    """Product of pcov * ccov for pattern-axis non-solid candidates.
    A None score is treated as -1 so unscored candidates sort to the bottom.
    Returns a float in [-1, 1]."""
    p = pcov_val if isinstance(pcov_val, (int, float)) else -1.0
    c = ccov_val if isinstance(ccov_val, (int, float)) else -1.0
    if p < 0 or c < 0:
        # at least one missing: use whichever is available, or -1
        if p >= 0:
            return p
        if c >= 0:
            return c
        return -1.0
    return p * c


def coverage_rerank(cands, primary, secondary=None, combined=False):
    """Return candidate indices ordered by coverage scores.

    Modes:
      combined=True  -> sort by pcov*ccov product (primary=pcovs, secondary=ccovs)
      secondary only -> (primary desc, secondary desc, sim desc)
      primary only   -> (primary desc, sim desc)
      neither        -> plain retrieval order

    A missing/None coverage is treated as -1 so any scored candidate
    outranks an unscored one, and unscored candidates keep similarity order
    among themselves."""
    idx = list(range(len(cands)))

    def sim_of(i):
        s = cands[i].get("similarity")
        return s if isinstance(s, (int, float)) else float("-inf")

    def cov_of(arr, i):
        c = arr[i] if (arr is not None and i < len(arr)) else None
        return c if isinstance(c, (int, float)) else -1.0

    if combined:
        # pcov * ccov product score
        idx.sort(key=lambda i: (
            -_combined_score(cov_of(primary, i), cov_of(secondary, i)),
            -sim_of(i)
        ))
    elif secondary is None:
        idx.sort(key=lambda i: (-cov_of(primary, i), -sim_of(i)))
    else:
        idx.sort(key=lambda i: (-cov_of(primary, i), -cov_of(secondary, i),
                                -sim_of(i)))
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


def _fmt_cov(arr, rank):
    v = arr[rank] if (arr is not None and rank < len(arr)) else None
    return f"{v:.2f}" if isinstance(v, (int, float)) else "  -- "


def _fmt_combined(pcovs, ccovs, rank):
    p = pcovs[rank] if (pcovs is not None and rank < len(pcovs)) else None
    c = ccovs[rank] if (ccovs is not None and rank < len(ccovs)) else None
    if isinstance(p, (int, float)) and isinstance(c, (int, float)):
        return f"{p * c:.2f}"
    return "  -- "


def _print_rank_table(opt_label, query, cands, pcovs, ccovs, order, chosen_rank,
                      skipped, use_combined=False):
    """Print the coverage-first ranking table for one option (verbose mode).
    pcov = pattern coverage, ccov = color coverage ('--' if not scored).
    combined = pcov*ccov column shown when use_combined=True.
    `skipped` maps rank -> reason ('dup'/'dead'/'gender') for non-chosen rows."""
    print(f"    [{opt_label}] query: {query}")
    if use_combined:
        print(f"      {'new':>3} {'retr':>4} {'pcov':>5} {'ccov':>5} {'comb':>6} {'sim':>8}  pick  caption")
    else:
        print(f"      {'new':>3} {'retr':>4} {'pcov':>5} {'ccov':>5} {'sim':>8}  pick  caption")
    for nr, rank in enumerate(order):
        c = cands[rank]
        sim = c.get("similarity")
        sims = f"{sim:.4f}" if isinstance(sim, (int, float)) else "  --  "
        if rank == chosen_rank:
            mark = "  <=PICK"
        elif rank in skipped:
            mark = f"  x{skipped[rank]}"
        else:
            mark = ""
        cap = (c.get("caption") or "")[:42]
        if use_combined:
            comb = _fmt_combined(pcovs, ccovs, rank)
            print(f"      {nr:>3} {rank:>4} {_fmt_cov(pcovs, rank):>5} "
                  f"{_fmt_cov(ccovs, rank):>5} {comb:>6} "
                  f"{sims:>8}{mark:>9}  {cap}")
        else:
            print(f"      {nr:>3} {rank:>4} {_fmt_cov(pcovs, rank):>5} "
                  f"{_fmt_cov(ccovs, rank):>5} "
                  f"{sims:>8}{mark:>9}  {cap}")


def resolve_option_top1(query, img_path, client_url, top_k, indice_name,
                        vlm_url, vlm_model, target_gender, used_urls,
                        coverage_url=None, cov_pattern=None, cov_color=None,
                        cov_garment=None, grid=5,
                        color_coverage_url=None, ccov_color=None,
                        ccov_pattern=None,
                        verbose=False, opt_label=""):
    """Retrieve top-k, optionally RE-RANK by patch coverage, then walk that
    order applying dedup + gender gates. Returns (option_result, seen_gender,
    chosen_url).

    Coverage routing (the caller decides WHICH apply by passing/omitting them):
      pattern axis non-solid + color cov available
                     : sort by pcov*ccov PRODUCT (combined score), then sim
      pattern only   : sort (pattern_cov desc, sim desc)  [v6 unchanged]
      color only     : sort (color_cov desc, sim desc)
      neither        : plain retrieval order
    On any coverage error it falls back to retrieval order (fail-open).

    verbose -> print the query and the coverage-first ranking table, marking the
    chosen candidate and why others were skipped."""
    cands = knn_search(client_url, query, top_k=top_k, indice_name=indice_name)
    if not cands:
        if verbose:
            print(f"    [{opt_label}] query: {query}\n      (no candidates)")
        return ({"image_path": None, "source": "FAILED", "search_query": query}, None, None)

    pcovs = [None] * len(cands)
    ccovs = [None] * len(cands)
    use_pcov = bool(coverage_url and cov_pattern and cov_pattern not in _SOLID_LIKE)
    use_ccov = bool(color_coverage_url and ccov_color)
    urls = [_cand_url(c) or "" for c in cands]
    if use_pcov:
        pcovs = coverage_score_urls(coverage_url, urls, cov_pattern,
                                    cov_color or "", cov_garment or "garment", grid)
    if use_ccov:
        ccovs = color_coverage_score_urls(color_coverage_url, urls, ccov_color,
                                          ccov_pattern or "",
                                          cov_garment or "garment", grid)

    # Determine sort mode
    # pattern non-solid + color cov both available -> pcov*ccov product
    use_combined = use_pcov and use_ccov

    if use_combined:
        order = coverage_rerank(cands, pcovs, ccovs, combined=True)
    elif use_pcov:
        order = coverage_rerank(cands, pcovs)          # v6 pattern-only (unchanged)
    elif use_ccov:
        order = coverage_rerank(cands, ccovs)          # color axis
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
                "coverage": pcovs[rank] if rank < len(pcovs) else None,
                "cov_reranked": use_pcov,
                "color_coverage": ccovs[rank] if rank < len(ccovs) else None,
                "ccov_reranked": use_ccov,
                "combined_score": (
                    _combined_score(pcovs[rank] if rank < len(pcovs) else None,
                                    ccovs[rank] if rank < len(ccovs) else None)
                    if use_combined else None
                )}

        if not vlm_url:                # coverage/top-1 + dedup (no gender)
            chosen = rank; result = (base, "unclear", url); break

        g = vlm_gender(vlm_url, vlm_model, img_path)
        if target_gender in ("man", "woman") and g in ("man", "woman") and g != target_gender:
            skipped[rank] = "gender"; continue   # gender conflict -> next candidate
        base["model_gender"] = g
        chosen = rank; result = (base, g, url); break

    if verbose:
        _print_rank_table(opt_label, query, cands, pcovs, ccovs, order, chosen,
                          skipped, use_combined=use_combined)
    return result


def collect_for_plan(plan, image_root, client_url, top_k, indice_name,
                     vlm_url, vlm_model, coverage_url=None, grid=5,
                     color_coverage_url=None, color_cov_mode="color",
                     verbose=False):
    qid = plan["query_id"]
    out_dir = Path(image_root) / qid
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "query_id": qid, "user_id": plan.get("user_id"),
        "domain": plan.get("domain", "fashion"),
        "active_axis": plan.get("active_axis"),
        "main_category": plan.get("main_category"),
        "scenario_archetype": plan.get("scenario_archetype"),
        "color_cov_mode": color_cov_mode,
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
        # PATTERN coverage: pattern-axis non-solid options only (v6, unchanged)
        cov_pat = _clean(target_pattern) if active_axis == "pattern" else None
        # COLOR coverage routing by --color-cov mode
        #   off   -> never
        #   color -> color-axis options only (pattern axis stays pure v6)
        #   all   -> every option with a target color
        if color_cov_mode == "off" or not color_coverage_url:
            ccov_color = None
        elif color_cov_mode == "color":
            ccov_color = _clean(target_color) if active_axis == "color" else None
        else:  # "all"
            ccov_color = _clean(target_color)
        if ccov_color in ("", "none", "unknown"):
            ccov_color = None

        img_path = out_dir / f"{k}.jpg"
        res, seen, url = resolve_option_top1(
            query, img_path, client_url, top_k, indice_name,
            vlm_url, vlm_model, target_gender, used_urls,
            coverage_url=coverage_url, cov_pattern=cov_pat,
            cov_color=_clean(target_color), cov_garment=_clean(target_garment),
            grid=grid,
            color_coverage_url=color_coverage_url, ccov_color=ccov_color,
            ccov_pattern=_clean(target_pattern),
            verbose=verbose, opt_label=f"{qid} {k}")
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
        coverage_url=None, grid=5,
        color_coverage_url=None, color_cov_mode="color", verbose=False):
    plans = load_jsonl(plan_path)
    print(f"Loaded {len(plans)} plans | server={client_url} top_k={top_k} "
          f"(gender={'on' if vlm_urls and vlm_urls[0] else 'off'}, intra-set dedup=on, "
          f"color_cov={color_cov_mode})")
    vlm_urls = vlm_urls or [None]
    print(f"  VLM: {vlm_urls}  model={vlm_model}")
    print(f"  pattern cov: {coverage_url or 'OFF'} | color cov: "
          f"{(color_coverage_url if color_cov_mode != 'off' else None) or 'OFF'}")

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
                                coverage_url=coverage_url, grid=grid,
                                color_coverage_url=color_coverage_url,
                                color_cov_mode=color_cov_mode, verbose=verbose)

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
    n_ccov = sum(1 for r in results for k in "ABCD"
                 if (r.get("options", {}).get(k) or {}).get("ccov_reranked"))
    n_combined = sum(1 for r in results for k in "ABCD"
                     if (r.get("options", {}).get(k) or {}).get("combined_score") is not None)
    print(f"\nSaved {len(results)}: complete={n_done} "
          f"({n_done/max(len(results),1):.0%}), skipped={n_skip}, "
          f"gender_inconsistent={n_gincons}, color_cov_options={n_ccov}, "
          f"combined_sort_options={n_combined}")


def selftest(client_url, indice_name, top_k, color_coverage_url=None):
    for q in ["beige plaid shirt", "white polka dot t shirt", "navy coat"]:
        cands = knn_search(client_url, q, top_k=top_k, indice_name=indice_name)
        print(f"\nquery: {q!r}  -> {len(cands)} candidates")
        for i, c in enumerate(cands[:5]):
            print(f"  [{i}] {c.get('similarity')} | {_cand_url(c)} | {c.get('caption','')[:60]}")
    if color_coverage_url:
        # one color-coverage smoke call on the first navy-coat candidate
        cands = knn_search(client_url, "navy coat", top_k=3, indice_name=indice_name)
        urls = [_cand_url(c) for c in cands if _cand_url(c)][:3]
        covs = color_coverage_score_urls(color_coverage_url, urls,
                                         "navy", "solid", "coat", 5)
        print(f"\ncolor-coverage smoke (navy coat): {covs}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan_path", type=Path,
                    default=Path("data/options/option_plans.jsonl"))
    ap.add_argument("--output", type=Path,
                    default=Path("data/images_vit_cov_c/collection_log.jsonl"))
    ap.add_argument("--image_root", type=Path,
                    default=Path("data/images_vit_cov_c"))
    ap.add_argument("--client-url", default="http://127.0.0.1:1234/knn-service")
    ap.add_argument("--coverage-url", default="http://127.0.0.1:1235/patch-coverage",
                    help="PATTERN patch-coverage endpoint; empty to disable")
    ap.add_argument("--color-coverage-url",
                    default="http://127.0.0.1:1235/patch-color-coverage",
                    help="COLOR patch-coverage endpoint; empty to disable")
    ap.add_argument("--color-cov", choices=["color", "all", "off"], default="color",
                    help="color-coverage routing: 'color' = color-axis options only "
                         "(default; pattern axis stays exactly v6), 'all' = every "
                         "option incl. pattern non-solid with pcov*ccov sort, "
                         "'off' = exact v6 behavior")
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
                         "(pcov, ccov, comb=pcov*ccov columns for pattern non-solid). "
                         "Use with --workers 1 for readable output.")
    args = ap.parse_args()

    color_coverage_url = args.color_coverage_url.strip() or None
    if args.selftest:
        selftest(args.client_url, args.indice_name, args.top_k,
                 color_coverage_url=color_coverage_url)
        return
    if args.force and Path(args.output).exists():
        Path(args.output).unlink()
    vlm_urls = [u.strip() for u in args.vlm_urls.split(",") if u.strip()] or None
    coverage_url = args.coverage_url.strip() or None
    run(args.plan_path, args.output, args.image_root,
        args.client_url, args.indice_name, args.limit, args.top_k, args.workers,
        vlm_urls=vlm_urls, vlm_model=args.vlm_model,
        coverage_url=coverage_url, grid=args.grid,
        color_coverage_url=color_coverage_url, color_cov_mode=args.color_cov,
        verbose=args.verbose)


if __name__ == "__main__":
    main()
