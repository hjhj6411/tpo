#!/usr/bin/env python3
"""
diag_coverage_rank.py — re-rank retrieval top-k by patch COVERAGE, then sim
---------------------------------------------------------------------------
For each option: retrieve top-k from the FashionSigLIP server (C1_allover query),
score each candidate's patch coverage via the server's /patch-coverage endpoint,
then ORDER the candidates by:
    1) coverage  (descending)   -- whole-garment pattern wins
    2) retrieval sim (descending) on ties
so you can see what a coverage-first collector would pick as top-1, vs the plain
retrieval top-1.

Coverage is computed against the option's OWN target pattern (color/garment too).
plaid & checkered are INCLUDED here: a real plaid/checkered garment should score
HIGH coverage; a near-solid mislabeled one should score low. That is the test.

  python diag_coverage_rank.py \
    --plans data/options/option_plans.jsonl \
    --client-url http://127.0.0.1:1235/knn-service \
    --coverage-url http://127.0.0.1:1235/patch-coverage \
    --axis pattern --top_k 12 --n-options 6 --grid 3 \
    --save-images data/eval/cov_rank
"""
import argparse, io, json, os
import requests
from PIL import Image

S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0 (pod-bench-covrank)"})

_SOLID_LIKE = {"", "solid", "plain", "none"}


def _clean(s):
    return (s or "").strip().lower().replace("_", " ")


def c1_allover(color, garment, pattern):
    c = f"{color} " if color and color not in ("none", "unknown") else ""
    g = garment or "garment"
    if pattern in _SOLID_LIKE:
        return f"a {c}{g}, not cropped".replace("  ", " ")
    return (f"a {c}{g} with an all-over {pattern} pattern, "
            f"not cropped").replace("  ", " ")


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def _url(c):
    return c.get("url") or c.get("image_url")


def knn(client_url, text, k, indice="pod_fashion"):
    r = S.post(client_url, json={"text": text, "modality": "image",
               "num_images": k, "num_result_ids": k, "indice_name": indice}, timeout=60)
    r.raise_for_status()
    return r.json()


def coverage_batch(cov_url, srcs, pattern, color, garment, grid):
    r = S.post(cov_url, json={"images": srcs, "pattern": pattern, "color": color,
               "garment": garment, "grid": grid, "drop_white": True}, timeout=300)
    r.raise_for_status()
    return r.json()


def save_img(url, path):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        r = S.get(url, timeout=15); r.raise_for_status()
        Image.open(io.BytesIO(r.content)).convert("RGB").save(path, "JPEG", quality=90)
        return True
    except Exception as e:
        print(f"   [save failed] {path}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", required=True)
    ap.add_argument("--client-url", default="http://127.0.0.1:1235/knn-service")
    ap.add_argument("--coverage-url", default="http://127.0.0.1:1235/patch-coverage")
    ap.add_argument("--indice-name", default="pod_fashion")
    ap.add_argument("--axis", default="pattern", help="only probe this active_axis")
    ap.add_argument("--top_k", type=int, default=12)
    ap.add_argument("--n-options", type=int, default=6)
    ap.add_argument("--grid", type=int, default=3)
    ap.add_argument("--include-solid", action="store_true",
                    help="also probe solid-pattern options (coverage is ~meaningless there)")
    ap.add_argument("--save-images", default=None)
    args = ap.parse_args()

    plans = load_jsonl(args.plans)
    if args.axis:
        plans = [p for p in plans if p.get("active_axis") == args.axis]

    # gather non-solid pattern-axis options (plaid/checkered INCLUDED)
    specs = []
    for p in plans:
        for k in "ABCD":
            o = (p.get("options") or {}).get(k) or {}
            a = o.get("attributes", {}) or {}
            pat = _clean(a.get("pattern"))
            if pat in _SOLID_LIKE and not args.include_solid:
                continue
            specs.append((p["query_id"], k, a))
            break
        if len(specs) >= args.n_options:
            break

    if not specs:
        print("no pattern options found"); return

    for (qid, k, attrs) in specs:
        pat = _clean(attrs.get("pattern"))
        col = _clean(attrs.get("color"))
        gar = _clean(attrs.get("garment_category"))
        query = c1_allover(col, gar, pat)

        print("=" * 104)
        print(f"{qid}  opt {k}  target: pattern={pat} color={col} garment={gar}")
        print(f"   query: {query}")

        cands = knn(args.client_url, query, args.top_k, args.indice_name)
        urls = [_url(c) for c in cands]
        sims = [c.get("similarity") for c in cands]

        # download to a temp dir the SERVER can also read? No -- the coverage
        # endpoint fetches URLs itself. Pass the URLs straight through.
        valid = [(i, u) for i, u in enumerate(urls) if u]
        cov_res = coverage_batch(args.coverage_url, [u for _, u in valid],
                                 pat, col, gar, args.grid)
        cov_by_i = {}
        for (i, _), r in zip(valid, cov_res):
            cov_by_i[i] = r

        rows = []
        for i, c in enumerate(cands):
            r = cov_by_i.get(i, {})
            rows.append({
                "retr_rank": i,
                "sim": sims[i] if isinstance(sims[i], (int, float)) else float("nan"),
                "coverage": r.get("coverage"),
                "npat": r.get("n_pattern"), "nval": r.get("n_valid"),
                "url": urls[i], "cap": (c.get("caption") or "")[:46],
            })

        # ORDER: coverage desc, then sim desc. Missing coverage sorts last.
        def sort_key(rr):
            cov = rr["coverage"] if isinstance(rr["coverage"], (int, float)) else -1.0
            return (-cov, -(rr["sim"] if rr["sim"] == rr["sim"] else -1e9))
        ordered = sorted(rows, key=sort_key)

        print(f"   {'newrank':>7} {'retr':>4} {'cov':>5} {'pat/val':>8} {'sim':>8}  caption")
        for nr, rr in enumerate(ordered):
            cov = "  -- " if rr["coverage"] is None else f"{rr['coverage']:.2f}"
            pv = "" if rr["npat"] is None else f"{rr['npat']}/{rr['nval']}"
            sim = f"{rr['sim']:.4f}" if rr["sim"] == rr["sim"] else "  --  "
            mark = "  <- retr top1" if rr["retr_rank"] == 0 else ""
            print(f"   {nr:>7} {rr['retr_rank']:>4} {cov:>5} {pv:>8} {sim:>8}  {rr['cap']}{mark}")
            if args.save_images:
                save_img(rr["url"], f"{args.save_images}/{qid}_{k}_new{nr}_retr{rr['retr_rank']}.jpg")

        top = ordered[0]
        same = "SAME as retrieval top-1" if top["retr_rank"] == 0 else \
               f"DIFFERS (retrieval top-1 was retr_rank 0, cov-rank picks retr_rank {top['retr_rank']})"
        print(f"   -> coverage-first top-1: retr_rank {top['retr_rank']}, "
              f"cov {top['coverage']}  [{same}]")
        print()


if __name__ == "__main__":
    main()
