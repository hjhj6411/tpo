"""
e4_retrieval_ablation/run.py — image-collection ablation grid (list C, 1-5).

Objective : ONE grid answers all five questions:
  C-1 backbone        : ViT (:1234) vs FashionSigLIP (:1235)
  C-2 query           : person(v1) vs product/title (person-free, augmented)
  C-3 pcov impact     : rules sim_only -> pcov_only -> cov_product
  C-4 ccov impact     : rules sim_only -> ccov_only -> cov_product
  C-5 best combo      : sim_only | sum_norm(v1) | prod_sim(v1) | cov_product(beta)
Design    : per (plan option x backbone x query style) the top-k POOL is
            retrieved ONCE and coverage-scored ONCE; every scoring rule is then
            applied OFFLINE to that fixed pool -> rerank comparisons exclude
            retrieval effects (and rule sweeps cost zero extra GPU).
Input     : --plans option_plans.jsonl; live servers :1234/:1235 (knn) and
            :1235 coverage routes (pcovB/pcovT/ccov/garment).
Output    : <results>/e4_retrieval_ablation/<ts>_<tag>/
              pools.jsonl       per (qid,opt,backbone,style): pool + metrics
              cells.jsonl       per (... x rule x beta): ranked top1 url
              label_sheet.csv   unique top-1 urls for human labeling
                                (url,color_ok,pattern_ok,garment_ok columns to fill)
Metrics   : produced by aggregate_labels.py after human labeling:
            per-cell ok-rates, pool hit rate, McNemar between rules.
Run       :
  python -m e4_retrieval_ablation.run --plans data/options/option_plans.jsonl \
      --backends vit=http://127.0.0.1:1234/knn-service \
                 fsiglip=http://127.0.0.1:1235/knn-service \
      --query-styles person product title --rules sim_only sum_norm prod_sim \
      cov_product --betas 1.0 0.5 --top-k 12 --n-options 120 --tag grid_v1
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import add_common_args, load_cfg
from common.data import load_plans
from common.io_utils import make_run_dir, save_manifest, write_jsonl
from common.retrieval import (QUERY_STYLES, RULES, apply_rule, build_query,
                              knn_search, score_pool)

EXP = "e4_retrieval_ablation"


def _kv(pairs):
    out = {}
    for p in pairs:
        k, v = p.split("=", 1)
        out[k] = v
    return out


def iter_options(plans, n_options):
    n = 0
    for plan in plans:
        for k in "ABCD":
            if n_options and n >= n_options:
                return
            if k in plan["options"]:
                yield plan["query_id"], plan.get("active_axis"), k, \
                    plan["options"][k]
                n += 1


def run(cfg, plans):
    backends = _kv(cfg["backends"])
    cov_urls = {"pcovB": cfg["pcovb_url"], "pcovT": cfg["pcovt_url"],
                "ccov": cfg["ccov_url"], "garment": cfg["garment_url"]}
    pools, cells = [], []
    for qid, axis, opt, o in iter_options(plans, cfg["n_options"]):
        attrs = o.get("attributes", {}) or {}
        target = {"color": (attrs.get("color") or "").lower(),
                  "pattern": (attrs.get("pattern") or "").lower(),
                  "garment": (attrs.get("garment_category") or "").lower()}
        for bk, bk_url in backends.items():
            for style in cfg["query_styles"]:
                q = build_query(style, attrs.get("garment_category"),
                                attrs.get("color"), attrs.get("pattern"),
                                axis, o.get("search_query"))
                try:
                    cands = knn_search(bk_url, q, cfg["top_k"])
                except Exception as e:                  # noqa: BLE001
                    print(f"  [knn:{bk}/{style}] {qid} {opt}: {e}")
                    continue
                urls = [c.get("url") or c.get("image_url") or "" for c in cands]
                sims = [c.get("similarity") for c in cands]
                m = score_pool(urls, target, cov_urls, cfg["grid"])
                pools.append({"query_id": qid, "option": opt, "axis": axis,
                              "backbone": bk, "style": style, "query": q,
                              "target": target, "urls": urls, "sims": sims,
                              **m})
                for rule in cfg["rules"]:
                    betas = cfg["betas"] if rule == "cov_product" else [1.0]
                    for beta in betas:
                        rank = apply_rule(rule, sims, m, beta)
                        cells.append({
                            "query_id": qid, "option": opt, "axis": axis,
                            "backbone": bk, "style": style, "rule": rule,
                            "beta": beta, "top1_url": urls[rank[0]],
                            "top1_rank_orig": rank[0],
                            "ranked_idx": rank})
        print(f"  scored {qid} {opt}")
    return pools, cells


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--plans", type=str, default=None)
    ap.add_argument("--backends", nargs="+",
                    default=["vit=http://127.0.0.1:1234/knn-service",
                             "fsiglip=http://127.0.0.1:1235/knn-service"])
    ap.add_argument("--query-styles", nargs="+", default=["person", "product"],
                    choices=list(QUERY_STYLES))
    ap.add_argument("--rules", nargs="+", choices=list(RULES),
                    default=["sim_only", "pcov_only", "ccov_only", "sum_norm",
                             "prod_sim", "cov_product"])
    ap.add_argument("--betas", nargs="+", type=float, default=[1.0, 0.5])
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--n-options", type=int, default=120,
                    help="stratified human-label budget (0 = all)")
    ap.add_argument("--pcovb-url",
                    default="http://127.0.0.1:1235/patch-coverage")
    ap.add_argument("--pcovt-url",
                    default="http://127.0.0.1:1235/patch-pattern-coverage")
    ap.add_argument("--ccov-url",
                    default="http://127.0.0.1:1235/patch-color-coverage")
    ap.add_argument("--garment-url",
                    default="http://127.0.0.1:1235/garment-check")
    cfg = load_cfg(ap.parse_args())
    plans = load_plans(cfg["plans"], cfg["limit"])
    run_dir = make_run_dir(cfg["results_root"], EXP, cfg["tag"])
    save_manifest(run_dir, cfg, {"n_plans": len(plans)})

    pools, cells = run(cfg, plans)
    write_jsonl(run_dir / "pools.jsonl", pools)
    write_jsonl(run_dir / "cells.jsonl", cells)

    seen = sorted({c["top1_url"] for c in cells})
    with open(run_dir / "label_sheet.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["url", "color_ok", "pattern_ok", "garment_ok"])
        for u in seen:
            w.writerow([u, "", "", ""])
    print(f"\n[{EXP}] pools={len(pools)} cells={len(cells)} "
          f"unique top-1 urls to label={len(seen)}")
    print(f"  fill label_sheet.csv (1/0) then run aggregate_labels.py")
    print(f"  -> {run_dir}")


if __name__ == "__main__":
    main()
