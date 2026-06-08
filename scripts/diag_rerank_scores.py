#!/usr/bin/env python3
"""
diag_rerank_scores.py — inspect SigLIP recall vs reranker scores, side by side
------------------------------------------------------------------------------
For a few option specs, fetch the SigLIP top-k candidates and print, per
candidate: rank, SigLIP similarity, reranker score, and caption/url. This shows
exactly what the reranker is doing to the ordering and whether it is pulling
WORSE images to the top.

It also (a) probes which rerank endpoint/format the server actually speaks, and
(b) lets you eyeball the spec text being sent.

  python diag_rerank_scores.py \
    --plans data/options/option_plans_nochk.jsonl \
    --client-url http://127.0.0.1:1235/knn-service \
    --rerank-url http://127.0.0.1:8010/v1 \
    --rerank-model Qwen/Qwen3-VL-Reranker-8B \
    --top_k 12 --n-options 5 --save-images data/eval/rerank_probe
"""
import argparse, base64, io, json, time
from pathlib import Path
import requests
from PIL import Image

S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0 (POD-Bench-diag)"})


def load_jsonl(p):
    out = []
    for l in open(p):
        l = l.strip()
        if l:
            out.append(json.loads(l))
    return out


def knn_search(client_url, query, top_k, indice="pod_fashion"):
    r = S.post(client_url, json={"text": query, "modality": "image",
               "num_images": top_k, "num_result_ids": top_k,
               "indice_name": indice}, timeout=60)
    r.raise_for_status()
    return r.json()


def fetch_img_b64(url, save_path=None, timeout=15):
    try:
        r = S.get(url, timeout=timeout); r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            img.save(save_path, "JPEG", quality=90)
        buf = io.BytesIO(); img.save(buf, "JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        return None


_SOLID_LIKE = {"", "solid", "plain", "none"}


def spec_text(attrs):
    """C1_allover spec (the phrasing validated by diag_coverage_queries): the
    proven all-over extent phrase on the T4 'studio product shot' skeleton.

        "a {color} {garment} with an all-over {pattern} pattern,
         studio product shot, not cropped"

    The all-over clause is included only for a non-solid pattern; for solid (or
    missing) pattern it is dropped so the spec stays clean. Used for BOTH the
    SigLIP recall query and the reranker query, so the two stages share intent."""
    g = (attrs.get("garment_category") or "").replace("_", " ").strip().lower()
    c = (attrs.get("color") or "").replace("_", " ").strip().lower()
    p = (attrs.get("pattern") or "").replace("_", " ").strip().lower()
    color_adj = f"{c} " if c and c not in ("none", "unknown", "any color") else ""
    pat_clause = f" with an all-over {p} pattern" if p not in _SOLID_LIKE else ""
    g = g or "garment"
    return f"a {color_adj}{g}{pat_clause}, studio product shot, not cropped".replace("  ", " ").strip()


def probe_endpoints(rerank_url, model, query, doc_b64):
    """Try /rerank and /v1/score; report which returns 200 and the shape."""
    doc = {"content": [{"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{doc_b64}"}}]}
    out = {}
    # /rerank (Jina/Cohere-style)
    try:
        r = S.post(f"{rerank_url.rstrip('/')}/rerank",
                   json={"model": model, "query": query, "documents": [doc]}, timeout=60)
        out["/rerank"] = (r.status_code, r.text[:200])
    except Exception as e:
        out["/rerank"] = ("ERR", str(e)[:120])
    # /score (text_1 / text_2)
    try:
        r = S.post(f"{rerank_url.rstrip('/')}/score",
                   json={"model": model, "text_1": query,
                         "text_2": {"content": doc["content"]}}, timeout=60)
        out["/score"] = (r.status_code, r.text[:200])
    except Exception as e:
        out["/score"] = ("ERR", str(e)[:120])
    return out


def rerank_scores(rerank_url, model, query, docs_b64):
    """Call /rerank with all docs; return list of scores aligned to docs order."""
    documents = [{"content": [{"type": "image_url",
                  "image_url": {"url": f"data:image/jpeg;base64,{b}"}}]} for b in docs_b64]
    r = S.post(f"{rerank_url.rstrip('/')}/rerank",
               json={"model": model, "query": query, "documents": documents}, timeout=120)
    r.raise_for_status()
    data = r.json()
    results = data.get("results") or data.get("data") or []
    scores = [None] * len(docs_b64)
    for it in results:
        idx = it.get("index")
        sc = it.get("relevance_score", it.get("score"))
        if idx is not None and 0 <= idx < len(scores):
            scores[idx] = sc
    return scores, data


def _minmax_norm(vals):
    """Min-max normalize a list (None-safe). Returns list of 0-1 (None stays None).
    If all equal, returns 0.5 for each present value."""
    present = [v for v in vals if v is not None]
    if not present:
        return [None] * len(vals)
    lo, hi = min(present), max(present)
    if hi - lo < 1e-9:
        return [None if v is None else 0.5 for v in vals]
    return [None if v is None else (v - lo) / (hi - lo) for v in vals]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", required=True)
    ap.add_argument("--client-url", default="http://127.0.0.1:1235/knn-service")
    ap.add_argument("--indice-name", default="pod_fashion")
    ap.add_argument("--rerank-url", default="http://127.0.0.1:8010/v1")
    ap.add_argument("--rerank-model", default="Qwen/Qwen3-VL-Reranker-8B")
    ap.add_argument("--top_k", type=int, default=12)
    ap.add_argument("--n-options", type=int, default=5, help="how many option specs to probe")
    ap.add_argument("--axis", default=None, help="only probe this active_axis (e.g. pattern)")
    ap.add_argument("--save-images", default=None, help="dir to save candidate jpgs for eyeballing")
    ap.add_argument("--siglip-weight", type=float, default=0.7,
                    help="weight on SigLIP in the combined score (0..1); rerank gets 1-w. "
                         "0.7 = trust SigLIP more, let rerank only nudge.")
    ap.add_argument("--raw-query", action="store_true",
                    help="use the plan's original search_query (NO spec-sentence "
                         "augmentation) for BOTH the SigLIP recall and the reranker query")
    args = ap.parse_args()
    w = max(0.0, min(1.0, args.siglip_weight))

    def query_for(attrs, sq):
        """spec-sentence (default) or the raw original search_query (--raw-query)."""
        if args.raw_query:
            return sq or spec_text(attrs)
        return spec_text(attrs)

    plans = load_jsonl(args.plans)
    if args.axis:
        plans = [p for p in plans if p.get("active_axis") == args.axis]

    # collect a handful of (qid, option, attrs)
    specs = []
    for p in plans:
        for k in "ABCD":
            o = (p.get("options") or {}).get(k) or {}
            attrs = o.get("attributes", {}) or {}
            specs.append((p["query_id"], k, attrs, o.get("search_query")))
            if len(specs) >= args.n_options:
                break
        if len(specs) >= args.n_options:
            break

    print(f"probing {len(specs)} option specs, top_k={args.top_k}, "
          f"combined = {w:.2f}*siglip_norm + {1-w:.2f}*rerank_norm"
          f"{'  [RAW search_query, no augmentation]' if args.raw_query else '  [spec-sentence augmented]'}\n")

    # endpoint probe once
    qid0, k0, attrs0, sq0 = specs[0]
    q0 = query_for(attrs0, sq0)
    cands0 = knn_search(args.client_url, q0, 1, args.indice_name)
    if cands0:
        url0 = cands0[0].get("url") or cands0[0].get("image_url")
        b0 = fetch_img_b64(url0)
        if b0:
            print("=== ENDPOINT PROBE ===")
            for ep, (code, body) in probe_endpoints(args.rerank_url, args.rerank_model,
                                                     q0, b0).items():
                print(f"  {ep:8s} -> {code}  {body}")
            print()

    for (qid, k, attrs, sq) in specs:
        spec = query_for(attrs, sq)
        print("=" * 96)
        print(f"{qid} opt {k}  query: {spec}")
        print(f"   (spec sentence: {spec_text(attrs)})")
        print(f"   (orig search_query: {sq})")
        cands = knn_search(args.client_url, spec, args.top_k, args.indice_name)
        rows = []
        b64s = []
        for rank, c in enumerate(cands):
            url = c.get("url") or c.get("image_url")
            sim = c.get("similarity")
            cap = (c.get("caption") or "")[:55]
            sp = f"{args.save_images}/{qid}_{k}_r{rank}.jpg" if args.save_images else None
            b = fetch_img_b64(url, sp) if url else None
            b64s.append(b)
            rows.append({"rank": rank, "sim": sim, "cap": cap, "url": url,
                         "img": sp, "has_img": b is not None})

        # reranker over downloaded candidates
        valid_idx = [i for i, b in enumerate(b64s) if b]
        rr = [None] * len(b64s)
        try:
            sc, raw = rerank_scores(args.rerank_url, args.rerank_model,
                                    spec, [b64s[i] for i in valid_idx])
            for j, i in enumerate(valid_idx):
                rr[i] = sc[j]
        except Exception as e:
            print(f"   [rerank failed: {e}]")

        # normalize each score within this candidate pool, then weighted-combine
        sims = [r["sim"] for r in rows]
        sim_n = _minmax_norm(sims)
        rr_n = _minmax_norm(rr)
        for i, r in enumerate(rows):
            sn, rn = sim_n[i], rr_n[i]
            r["sim_n"], r["rr"], r["rr_n"] = sn, rr[i], rn
            if sn is None and rn is None:
                r["combined"] = None
            elif rn is None:           # reranker missing -> fall back to siglip only
                r["combined"] = sn
            elif sn is None:
                r["combined"] = rn
            else:
                r["combined"] = w * sn + (1 - w) * rn

        # sort by combined score, descending
        ordered = sorted([r for r in rows if r["combined"] is not None],
                         key=lambda r: r["combined"], reverse=True)

        print(f"   sorted by COMBINED (w_siglip={w:.2f}):")
        print(f"   {'newrank':>7} {'origrank':>8} {'combined':>8} "
              f"{'sigN':>6} {'rrN':>6} {'siglip':>8} {'rerank':>8}  caption")
        for new_rank, r in enumerate(ordered):
            snd = "  --  " if r["sim_n"] is None else f"{r['sim_n']:.3f}"
            rnd = "  --  " if r["rr_n"] is None else f"{r['rr_n']:.3f}"
            rrd = "   --   " if r["rr"] is None else f"{r['rr']:8.4f}"
            mark = "  <- SigLIP top1" if r["rank"] == 0 else ""
            print(f"   {new_rank:>7} {r['rank']:>8} {r['combined']:>8.3f} "
                  f"{snd:>6} {rnd:>6} {str(r['sim'])[:8]:>8} {rrd:>8}  {r['cap']}{mark}")
        if ordered:
            top = ordered[0]
            print(f"   -> COMBINED top-1 = orig rank {top['rank']}"
                  f"  ({'= SigLIP top1' if top['rank'] == 0 else 'differs from SigLIP top1'})")
            if top.get("img"):
                print(f"      image: {top['img']}")
        print()


if __name__ == "__main__":
    main()
