#!/usr/bin/env python3
"""
diag_four_methods.py — coverage-first ranking with 4 tie-break strategies
-------------------------------------------------------------------------
All four methods rank candidates by patch COVERAGE first (descending). They
differ ONLY in how COVERAGE TIES are broken:

  M1  retrieval=ViT      tie-break = 0.5*ViT_sim_norm   + 0.5*rerank_norm
  M2  retrieval=FSigLIP  tie-break = 0.5*FSigLIP_sim_norm+ 0.5*rerank_norm
  M3  retrieval=FSigLIP  tie-break = rerank score (alone)
  M4  retrieval=ViT      tie-break = rerank score (alone)

For each option we run BOTH retrieval backbones (so M1/M4 share the ViT candidate
pool, M2/M3 share the FSigLIP pool), score coverage for every candidate via the
SigLIP /patch-coverage server, score the reranker for every candidate, then build
each method's ordering and show the top-1 side by side.

Servers: ViT :1234 (knn) | FSigLIP :1235 (knn + /patch-coverage) | reranker :8010

  python diag_four_methods.py \
    --plans data/options/option_plans.jsonl \
    --vit-url     http://127.0.0.1:1234/knn-service \
    --siglip-url  http://127.0.0.1:1235/knn-service \
    --coverage-url http://127.0.0.1:1235/patch-coverage \
    --rerank-url  http://127.0.0.1:8010/v1 \
    --rerank-model Qwen/Qwen3-VL-Reranker-8B \
    --axis pattern --top_k 12 --grid 5 --n-options 6 \
    --save-images data/eval/four_methods
"""
import argparse, io, json, os, base64
import requests
from PIL import Image

S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0 (pod-bench-4m)"})
_SOLID_LIKE = {"", "solid", "plain", "none"}


def _clean(s):
    return (s or "").strip().lower().replace("_", " ")


def c1_allover(color, garment, pattern):
    c = f"{color} " if color and color not in ("none", "unknown") else ""
    g = garment or "garment"
    if pattern in _SOLID_LIKE:
        return f"a {c}{g}, studio product shot, not cropped".replace("  ", " ")
    return (f"a {c}{g} with an all-over {pattern} pattern, "
            f"studio product shot, not cropped").replace("  ", " ")


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def _url(c):
    return c.get("url") or c.get("image_url")


def knn(url, text, k, indice="pod_fashion"):
    try:
        r = S.post(url, json={"text": text, "modality": "image",
                   "num_images": k, "num_result_ids": k, "indice_name": indice}, timeout=60)
        r.raise_for_status(); return r.json()
    except Exception as e:
        print(f"    [knn {url}] failed: {e}"); return []


def coverage_for(cov_url, urls, pattern, color, garment, grid):
    if not urls:
        return {}
    try:
        r = S.post(cov_url, json={"images": list(urls), "pattern": pattern,
                   "color": color, "garment": garment, "grid": grid,
                   "drop_white": True}, timeout=300)
        r.raise_for_status()
        res = r.json()
        return {u: (row.get("coverage") if isinstance(row, dict) else None)
                for u, row in zip(urls, res)}
    except Exception as e:
        print(f"    [coverage] failed: {e}"); return {u: None for u in urls}


def fetch_b64(url):
    try:
        r = S.get(url, timeout=15); r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        buf = io.BytesIO(); img.save(buf, "JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def rerank_for(rerank_url, model, query, urls):
    """Return {url: rerank_score}. Fetches each image, calls /rerank once."""
    b64s, keep = [], []
    for u in urls:
        b = fetch_b64(u)
        if b:
            b64s.append(b); keep.append(u)
    if not b64s:
        return {u: None for u in urls}
    docs = [{"content": [{"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b}"}}]} for b in b64s]
    try:
        r = S.post(f"{rerank_url.rstrip('/')}/rerank",
                   json={"model": model, "query": query, "documents": docs}, timeout=120)
        r.raise_for_status()
        data = r.json()
        results = data.get("results") or data.get("data") or []
        out = {u: None for u in urls}
        for it in results:
            idx = it.get("index")
            sc = it.get("relevance_score", it.get("score"))
            if idx is not None and 0 <= idx < len(keep):
                out[keep[idx]] = sc
        return out
    except Exception as e:
        print(f"    [rerank] failed: {e}"); return {u: None for u in urls}


def _minmax(d):
    """min-max normalize a {key: value} (None-safe) -> {key: 0..1 or None}."""
    vals = [v for v in d.values() if isinstance(v, (int, float))]
    if not vals:
        return {k: None for k in d}
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return {k: (0.5 if isinstance(v, (int, float)) else None) for k, v in d.items()}
    return {k: ((v - lo) / (hi - lo) if isinstance(v, (int, float)) else None)
            for k, v in d.items()}


def order_by(cov, tiebreak, urls):
    """Order urls by (coverage desc, tiebreak desc). None coverage -> last;
    None tiebreak -> 0."""
    def cov_of(u):
        c = cov.get(u)
        return c if isinstance(c, (int, float)) else -1.0

    def tb_of(u):
        t = tiebreak.get(u)
        return t if isinstance(t, (int, float)) else 0.0

    return sorted(urls, key=lambda u: (-cov_of(u), -tb_of(u)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", required=True)
    ap.add_argument("--vit-url", default="http://127.0.0.1:1234/knn-service")
    ap.add_argument("--siglip-url", default="http://127.0.0.1:1235/knn-service")
    ap.add_argument("--coverage-url", default="http://127.0.0.1:1235/patch-coverage")
    ap.add_argument("--rerank-url", default="http://127.0.0.1:8010/v1")
    ap.add_argument("--rerank-model", default="Qwen/Qwen3-VL-Reranker-8B")
    ap.add_argument("--indice-name", default="pod_fashion")
    ap.add_argument("--axis", default="pattern")
    ap.add_argument("--top_k", type=int, default=12)
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--n-options", type=int, default=6)
    ap.add_argument("--save-images", default=None)
    args = ap.parse_args()

    plans = load_jsonl(args.plans)
    if args.axis:
        plans = [p for p in plans if p.get("active_axis") == args.axis]

    specs = []
    for p in plans:
        for k in "ABCD":
            o = (p.get("options") or {}).get(k) or {}
            a = o.get("attributes", {}) or {}
            if _clean(a.get("pattern")) in _SOLID_LIKE:
                continue
            specs.append((p["query_id"], k, a)); break
        if len(specs) >= args.n_options:
            break

    for (qid, k, attrs) in specs:
        pat, col, gar = _clean(attrs.get("pattern")), _clean(attrs.get("color")), _clean(attrs.get("garment_category"))
        query = c1_allover(col, gar, pat)
        print("=" * 100)
        print(f"{qid}  opt {k}  target pattern={pat} color={col} garment={gar}")
        print(f"   query: {query}")

        # candidate pools
        vit = knn(args.vit_url, query, args.top_k, args.indice_name)
        sig = knn(args.siglip_url, query, args.top_k, args.indice_name)
        vit_sim = {_url(c): c.get("similarity") for c in vit if _url(c)}
        sig_sim = {_url(c): c.get("similarity") for c in sig if _url(c)}
        all_urls = list(dict.fromkeys(list(vit_sim) + list(sig_sim)))  # dedup, keep order

        # coverage for every url (scored once, on the SigLIP coverage server)
        cov = coverage_for(args.coverage_url, all_urls, pat, col, gar, args.grid)
        # reranker for every url (one call over the union)
        rr = rerank_for(args.rerank_url, args.rerank_model, query, all_urls)

        # normalized components for tie-breaks
        vit_n = _minmax(vit_sim)
        sig_n = _minmax(sig_sim)
        rr_n = _minmax(rr)

        def combo(a_norm, w_a=0.5, w_r=0.5):
            out = {}
            for u in all_urls:
                an, rn = a_norm.get(u), rr_n.get(u)
                if an is None and rn is None:
                    out[u] = None
                elif an is None:
                    out[u] = rn
                elif rn is None:
                    out[u] = an
                else:
                    out[u] = w_a * an + w_r * rn
            return out

        tb_m1 = combo(vit_n)       # 0.5 ViT + 0.5 rerank
        tb_m2 = combo(sig_n)       # 0.5 SigLIP + 0.5 rerank
        tb_m3 = rr                 # rerank alone (FSigLIP pool)
        tb_m4 = rr                 # rerank alone (ViT pool)

        methods = [
            ("M1 ViT+RR", list(vit_sim), tb_m1),
            ("M2 SiG+RR", list(sig_sim), tb_m2),
            ("M3 SiG cov>RR", list(sig_sim), tb_m3),
            ("M4 ViT cov>RR", list(vit_sim), tb_m4),
        ]

        print(f"   {'method':>15}  {'top1_cov':>8} {'top1_rr':>8}  caption / url-hash")
        tops = {}
        for name, pool, tb in methods:
            if not pool:
                print(f"   {name:>15}  (no candidates)"); continue
            ordered = order_by(cov, tb, pool)
            top = ordered[0]
            tops[name] = top
            cv = cov.get(top); cvs = f"{cv:.2f}" if isinstance(cv, (int, float)) else " -- "
            rv = rr.get(top); rvs = f"{rv:.3f}" if isinstance(rv, (int, float)) else "  -- "
            cap = ""
            for c in (vit + sig):
                if _url(c) == top:
                    cap = (c.get("caption") or "")[:40]; break
            import hashlib
            h = hashlib.md5(top.encode()).hexdigest()[:10]
            print(f"   {name:>15}  {cvs:>8} {rvs:>8}  {cap}  [{h}]")
            if args.save_images:
                try:
                    os.makedirs(args.save_images, exist_ok=True)
                    b = fetch_b64(top)
                    if b:
                        Image.open(io.BytesIO(base64.b64decode(b))).save(
                            f"{args.save_images}/{qid}_{k}_{name.split()[0]}.jpg")
                except Exception:
                    pass

        # agreement summary
        uniq = len(set(tops.values()))
        print(f"   -> distinct top-1 across 4 methods: {uniq}")
        if uniq > 1:
            for name, t in tops.items():
                import hashlib
                print(f"        {name:>15}: {hashlib.md5(t.encode()).hexdigest()[:10]}")
        print()


if __name__ == "__main__":
    main()
