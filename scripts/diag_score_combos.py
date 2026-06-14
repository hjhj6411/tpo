#!/usr/bin/env python3
"""
diag_score_combos.py — eyeball the right way to combine pcov, ccov, sim
------------------------------------------------------------------------
For each option (A/B/C/D) of each plan, retrieve top-k candidates, score every
candidate on THREE metrics, and compare TWO combination rules side by side:

    PRODUCT : pcov * ccov * sim
    SUM     : pcov + ccov + sim          (each metric min-max normalized within
                                          the candidate set first, so sim's tiny
                                          ~0.28 band doesn't get drowned out)

It does NOT pick / dedup / gender-gate — this is a LOOKING tool. It:
  - saves ALL top-k candidate images (rank 0..k-1) for BOTH rules into separate
    folders, filenames prefixed with the new rank + raw scores, so you can open
    a folder and read the chosen order off the filenames;
  - prints, per option, every candidate's pcov / ccov / sim and the resulting
    PRODUCT-rank and SUM-rank, plus the two top-1 picks.

pcov uses the (now solid-aware) /patch-coverage; ccov uses /patch-color-coverage.
Both pattern coverage on solid targets and color coverage are scored for EVERY
option, so you can also judge whether scoring pcov on solid is trustworthy.

Backbone is selectable so you can compare ViT vs FSigLIP retrieval:
  --client-url http://127.0.0.1:1234/knn-service   (ViT)
  --client-url http://127.0.0.1:1235/knn-service   (FSigLIP)
Coverage scoring ALWAYS goes to the SigLIP coverage server (:1235), which is
model-agnostic (it only classifies tiles).

Layout written under --out-dir:
  <out>/<backbone>/product/<qid>/<OPT>/r{newrank}_k{origrank}_p{pcov}_c{ccov}_s{sim}.jpg
  <out>/<backbone>/sum/<qid>/<OPT>/r{newrank}_k{origrank}_p{pcov}_c{ccov}_s{sim}.jpg
  <out>/<backbone>/scores.jsonl      (one row per option, all candidate metrics)

Usage (ViT backbone):
  python scripts/diag_score_combos.py \
    --plans data/options/option_plans.jsonl \
    --client-url http://127.0.0.1:1234/knn-service \
    --coverage-url       http://127.0.0.1:1235/patch-coverage \
    --color-coverage-url http://127.0.0.1:1235/patch-color-coverage \
    --backbone vit --top_k 12 --grid 5 --n-options 8 \
    --out-dir data/eval/score_combos

  # FSigLIP backbone: --client-url http://127.0.0.1:1235/knn-service --backbone fsiglip
"""
import argparse
import io
import json
import re
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from PIL import Image

UA = {"User-Agent": "Mozilla/5.0 (compatible; POD-Bench-diag-combos/1.0)"}
_SESSION = requests.Session()
_SESSION.headers.update(UA)
_adapter = HTTPAdapter(pool_connections=32, pool_maxsize=64, max_retries=0)
_SESSION.mount("http://", _adapter)
_SESSION.mount("https://", _adapter)

_SOLID_LIKE = {"", "solid", "plain", "none"}


def _clean(s):
    return (s or "").strip().lower().replace("_", " ")


# ── shared query builder (matches the v6/v7 collectors) ──────────────────
def specify_query(garment, target_color=None, target_pattern=None,
                  axis=None, fallback_query=None):
    g = _clean(garment); c = _clean(target_color); tp = _clean(target_pattern)
    if not g:
        base = _clean(fallback_query) or "clothing"
        return (f"a fashion catalog photo of studio product shot {base}, "
                f"entire garment visible, plain background, studio product shot").replace("  ", " ")
    color_adj = f"{c} " if c and c not in ("none", "unknown") else ""
    core = f"{color_adj}{g}".replace("  ", " ").strip()
    if axis == "pattern" and tp not in _SOLID_LIKE:
        return (f"a fashion catalog photo of studio product shot {core} with an "
                f"all-over {tp} pattern, front view, entire garment visible, "
                f"plain background").replace("  ", " ")
    return (f"a fashion catalog photo of studio product shot a {core}, front view, "
            f"entire garment visible, plain background, studio product shot").replace("  ", " ")


# ── io ───────────────────────────────────────────────────────────────────
def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _cand_url(c):
    return c.get("url") or c.get("image_url")


def knn_search(client_url, query, top_k, indice_name="pod_fashion"):
    try:
        r = _SESSION.post(client_url, json={
            "text": query, "modality": "image", "num_images": top_k,
            "num_result_ids": top_k, "indice_name": indice_name}, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"    [knn] {query!r}: {e}")
        return []


def pcov_urls(url, urls, pattern, color, garment, grid):
    if not urls:
        return []
    try:
        r = _SESSION.post(url, json={"images": list(urls), "pattern": pattern,
                                     "color": color, "garment": garment,
                                     "grid": grid, "drop_white": True}, timeout=300)
        r.raise_for_status()
        res = r.json()
        out = [x.get("coverage") if isinstance(x, dict) and "coverage" in x else None
               for x in res]
    except Exception as e:
        print(f"    [pcov] {e}")
        out = [None] * len(urls)
    return (out + [None] * len(urls))[:len(urls)]


def ccov_urls(url, urls, color, pattern, garment, grid):
    if not urls:
        return []
    try:
        r = _SESSION.post(url, json={"images": list(urls), "color": color,
                                     "pattern": pattern, "garment": garment,
                                     "grid": grid, "drop_white": True}, timeout=300)
        r.raise_for_status()
        res = r.json()
        out = [x.get("coverage") if isinstance(x, dict) and "coverage" in x else None
               for x in res]
    except Exception as e:
        print(f"    [ccov] {e}")
        out = [None] * len(urls)
    return (out + [None] * len(urls))[:len(urls)]


def download(url, dest, min_side=64, timeout=15):
    try:
        r = _SESSION.get(url, timeout=timeout); r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return False
    if min(img.size) < min_side:
        return False
    try:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        img.save(dest, "JPEG", quality=92)
        return True
    except Exception:
        return False


# ── scoring helpers ──────────────────────────────────────────────────────
def _f(x):
    return x if isinstance(x, (int, float)) else None


def _minmax(vals):
    """min-max normalize a list (None -> 0 after norm). If all equal, -> all 1.0
    so a metric with no spread doesn't kill the SUM."""
    nums = [v for v in vals if isinstance(v, (int, float))]
    if not nums:
        return [0.0] * len(vals)
    lo, hi = min(nums), max(nums)
    if hi - lo < 1e-9:
        return [1.0 if isinstance(v, (int, float)) else 0.0 for v in vals]
    return [((v - lo) / (hi - lo)) if isinstance(v, (int, float)) else 0.0
            for v in vals]


# AFTER
def product_scores(pcovs, ccovs, sims):
    """normalized pcov * normalized ccov * normalized sim.
    각 메트릭을 candidate set 내 min-max 정규화한 뒤 곱함.
    None → 0 처리 (missing metric → product 0)."""
    np_, nc, ns = _minmax(pcovs), _minmax(ccovs), _minmax(sims)
    return [a * b * d for a, b, d in zip(np_, nc, ns)]


def sum_scores(pcovs, ccovs, sims):
    """normalized pcov + normalized ccov + normalized sim. Each metric is
    min-max normalized WITHIN the candidate set first (sim's ~0.28 band would
    otherwise be negligible against 0..1 coverages)."""
    np_, nc, ns = _minmax(pcovs), _minmax(ccovs), _minmax(sims)
    return [a + b + d for a, b, d in zip(np_, nc, ns)]


def rank_order(scores):
    """indices sorted by score desc (stable)."""
    return sorted(range(len(scores)), key=lambda i: -scores[i])


def _fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else " -- "


def _safe(x):
    return re.sub(r"[^0-9p.]", "", f"{x:.2f}") if isinstance(x, (int, float)) else "na"


# ── per-option diagnosis ─────────────────────────────────────────────────
def diagnose_option(qid, k, opt, axis, client_url, coverage_url,
                    color_coverage_url, top_k, grid, indice_name,
                    out_root, backbone, save_images=True):
    attrs = opt.get("attributes", {}) or {}
    color = _clean(attrs.get("color"))
    pattern = _clean(attrs.get("pattern"))
    garment = _clean(attrs.get("garment_category"))
    query = specify_query(attrs.get("garment_category"),
                          target_color=attrs.get("color"),
                          target_pattern=attrs.get("pattern"),
                          axis=axis, fallback_query=opt.get("search_query"))

    cands = knn_search(client_url, query, top_k, indice_name)
    if not cands:
        print(f"    [{qid} {k}] no candidates")
        return None
    urls = [_cand_url(c) or "" for c in cands]
    sims = [_f(c.get("similarity")) for c in cands]

    # pcov scored for EVERY option now (solid-aware server); ccov always.
    pcovs = pcov_urls(coverage_url, urls, pattern or "solid", color, garment, grid) \
        if coverage_url else [None] * len(cands)
    ccovs = ccov_urls(color_coverage_url, urls, color, pattern, garment, grid) \
        if (color_coverage_url and color) else [None] * len(cands)

    prod = product_scores(pcovs, ccovs, sims)
    summ = sum_scores(pcovs, ccovs, sims)
    prod_order = rank_order(prod)
    sum_order = rank_order(summ)
    prod_rank = {orig: nr for nr, orig in enumerate(prod_order)}
    sum_rank = {orig: nr for nr, orig in enumerate(sum_order)}

    # ── print table ──
    print(f"\n  [{qid} {k}] axis={axis} target: color={color} pattern={pattern} garment={garment}")
    print(f"    query: {query}")
    print(f"    {'k':>2} {'pcov':>6} {'ccov':>6} {'sim':>7} {'PROD':>7} {'SUM':>6} "
          f"{'Prk':>3} {'Srk':>3}  caption")
    for i, c in enumerate(cands):
        print(f"    {i:>2} {_fmt(pcovs[i]):>6} {_fmt(ccovs[i]):>6} {_fmt(sims[i]):>7} "
              f"{prod[i]:>7.4f} {summ[i]:>6.3f} {prod_rank[i]:>3} {sum_rank[i]:>3}  "
              f"{(c.get('caption') or '')[:40]}")
    p1p, p1s = prod_order[0], sum_order[0]
    same = "  (SAME pick)" if p1p == p1s else "  (DIFFERENT pick)"
    print(f"    => PRODUCT top-1 = k{p1p} | SUM top-1 = k{p1s}{same}")

    # ── save all candidates into product/ and sum/ folders ──
    if save_images:
        for method, order in (("product", prod_order), ("sum", sum_order)):
            d = Path(out_root) / backbone / method / qid / k
            d.mkdir(parents=True, exist_ok=True)
            for nr, orig in enumerate(order):
                fn = (f"r{nr:02d}_k{orig:02d}_p{_safe(pcovs[orig])}"
                      f"_c{_safe(ccovs[orig])}_s{_safe(sims[orig])}.jpg")
                if not download(urls[orig], d / fn):
                    # leave a marker so a dead URL is visible in the folder
                    (d / (fn + ".DEAD.txt")).write_text(urls[orig])

    return {
        "query_id": qid, "option": k, "active_axis": axis, "backbone": backbone,
        "target": {"color": color, "pattern": pattern, "garment": garment},
        "query": query,
        "candidates": [
            {"k": i, "url": urls[i], "caption": (cands[i].get("caption") or "")[:120],
             "pcov": pcovs[i], "ccov": ccovs[i], "sim": sims[i],
             "product": prod[i], "sum": summ[i],
             "product_rank": prod_rank[i], "sum_rank": sum_rank[i]}
            for i in range(len(cands))],
        "product_top1_k": p1p, "sum_top1_k": p1s, "picks_agree": p1p == p1s,
    }


def run(args):
    plans = load_jsonl(args.plans)
    if args.limit > 0:
        plans = plans[:args.limit]
    out_root = Path(args.out_dir)
    scores_path = out_root / args.backbone / "scores.jsonl"
    scores_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"backbone={args.backbone}  retrieval={args.client_url}")
    print(f"pcov={args.coverage_url}  ccov={args.color_coverage_url}")
    print(f"plans={len(plans)} n_options={args.n_options} top_k={args.top_k} grid={args.grid}")

    rows = []
    n_opt = 0
    agree = 0
    for plan in plans:
        qid = plan["query_id"]
        axis = plan.get("active_axis")
        for k in "ABCD":
            if args.n_options and n_opt >= args.n_options:
                break
            opt = plan["options"][k]
            row = diagnose_option(
                qid, k, opt, axis, args.client_url, args.coverage_url,
                args.color_coverage_url, args.top_k, args.grid, args.indice_name,
                out_root, args.backbone, save_images=not args.no_images)
            if row:
                rows.append(row); n_opt += 1
                agree += int(row["picks_agree"])
        if args.n_options and n_opt >= args.n_options:
            break

    with open(scores_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  scored {n_opt} options | PRODUCT and SUM agreed on top-1 in "
          f"{agree}/{n_opt} ({agree/max(n_opt,1):.0%})")
    print(f"  scores -> {scores_path}")
    if not args.no_images:
        print(f"  images -> {out_root/args.backbone}/{{product,sum}}/<qid>/<OPT>/  "
              f"(filenames carry new rank + metrics)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", type=Path,
                    default=Path("data/options/option_plans.jsonl"))
    ap.add_argument("--client-url", default="http://127.0.0.1:1234/knn-service",
                    help="retrieval backbone: ViT :1234 or FSigLIP :1235")
    ap.add_argument("--coverage-url", default="http://127.0.0.1:1235/patch-coverage")
    ap.add_argument("--color-coverage-url",
                    default="http://127.0.0.1:1235/patch-color-coverage")
    ap.add_argument("--backbone", choices=["vit", "fsiglip"], default="vit",
                    help="label for the retrieval backbone (folder name only)")
    ap.add_argument("--indice-name", default="pod_fashion")
    ap.add_argument("--top_k", type=int, default=12)
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--n-options", type=int, default=8,
                    help="how many options (A/B/C/D across plans) to diagnose")
    ap.add_argument("--limit", type=int, default=0, help="cap plans scanned")
    ap.add_argument("--out-dir", type=Path, default=Path("data/eval/score_combos"))
    ap.add_argument("--no-images", action="store_true",
                    help="only print + write scores.jsonl, do not save images")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()