"""
e4_retrieval_ablation/aggregate_labels.py — join human labels with the grid.

Objective : turn the filled label_sheet.csv (gold human judgments, keyed by
            image url so one judgment serves every cell that picked that
            image) into the paper tables:
  * per-cell (backbone x style x rule x beta) top-1 ok-rates:
      color_ok / pattern_ok / garment_ok / all_ok            [C-3, C-4, C-5]
  * pool hit rate per (backbone x style): fraction of options whose top-k pool
    contains >=1 labeled all-ok image (the right metric for retrieval quality
    when a reranker follows)                                  [C-1, C-2]
  * exact McNemar between two named rule-cells on all_ok      [significance]
Input     : --run-dir (pools.jsonl + cells.jsonl), --labels label_sheet.csv
            with 1/0 in color_ok/pattern_ok/garment_ok (blank = unlabeled).
Output    : <run-dir>/aggregate.json + printed table.
Run       :
  python -m e4_retrieval_ablation.aggregate_labels \
      --run-dir data/exp_results/e4_retrieval_ablation/<run> \
      --labels data/exp_results/.../label_sheet.csv \
      --compare "fsiglip|product|cov_product|1.0" "fsiglip|product|prod_sim|1.0"
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.io_utils import read_jsonl, write_json
from common.metrics import mcnemar, wilson


def load_labels(path):
    lab = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            vals = {}
            for k in ("color_ok", "pattern_ok", "garment_ok"):
                v = (row.get(k) or "").strip()
                if v in ("0", "1"):
                    vals[k] = bool(int(v))
            if vals:
                vals["all_ok"] = all(vals.get(k, False) for k in
                                     ("color_ok", "pattern_ok", "garment_ok"))
                lab[row["url"]] = vals
    return lab


def cell_key(c):
    return f"{c['backbone']}|{c['style']}|{c['rule']}|{c['beta']}"


def aggregate(cells, pools, labels):
    out = {"n_labeled_urls": len(labels), "cells": {}, "pool_hit": {}}
    by_cell = defaultdict(list)
    for c in cells:
        by_cell[cell_key(c)].append(c)
    for key, cs in sorted(by_cell.items()):
        labeled = [labels[c["top1_url"]] for c in cs if c["top1_url"] in labels]
        n = len(labeled)
        row = {"n_top1_labeled": n, "label_coverage":
               round(n / max(len(cs), 1), 3)}
        for m in ("color_ok", "pattern_ok", "garment_ok", "all_ok"):
            k = sum(int(l.get(m, False)) for l in labeled)
            p, lo, hi = wilson(k, n)
            row[m] = {"rate": p, "ci95": [lo, hi]}
        out["cells"][key] = row
    for p in pools:
        key = f"{p['backbone']}|{p['style']}"
        lab_in_pool = [labels[u]["all_ok"] for u in p["urls"] if u in labels]
        if lab_in_pool:
            out["pool_hit"].setdefault(key, []).append(int(any(lab_in_pool)))
    out["pool_hit"] = {k: {"hit_rate": round(sum(v) / len(v), 4), "n": len(v)}
                       for k, v in sorted(out["pool_hit"].items())}
    return out, by_cell


def compare_cells(by_cell, labels, key1, key2):
    """Paired exact McNemar on all_ok between two cells (paired by qid+opt)."""
    def per_q(cs):
        return {(c["query_id"], c["option"]):
                labels.get(c["top1_url"], {}).get("all_ok")
                for c in cs}
    a, b = per_q(by_cell[key1]), per_q(by_cell[key2])
    joint = [q for q in a if q in b and a[q] is not None and b[q] is not None]
    x = sum(1 for q in joint if a[q] and not b[q])
    y = sum(1 for q in joint if b[q] and not a[q])
    return {"cell1": key1, "cell2": key2, "n_pairs": len(joint),
            **mcnemar(x, y)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--compare", nargs=2, default=None,
                    metavar=("CELL1", "CELL2"),
                    help='cell key "backbone|style|rule|beta"')
    a = ap.parse_args()
    rd = Path(a.run_dir)
    cells = read_jsonl(rd / "cells.jsonl")
    pools = read_jsonl(rd / "pools.jsonl")
    labels = load_labels(a.labels)
    agg, by_cell = aggregate(cells, pools, labels)
    if a.compare:
        agg["mcnemar"] = compare_cells(by_cell, labels, *a.compare)
    write_json(rd / "aggregate.json", agg)

    print(f"[aggregate] labeled urls={agg['n_labeled_urls']}")
    print(f"{'cell':<42} {'all_ok':>7} {'color':>6} {'pattern':>8} "
          f"{'garment':>8} {'n':>4}")
    for key, row in agg["cells"].items():
        print(f"{key:<42} {row['all_ok']['rate']:>7.3f} "
              f"{row['color_ok']['rate']:>6.3f} "
              f"{row['pattern_ok']['rate']:>8.3f} "
              f"{row['garment_ok']['rate']:>8.3f} {row['n_top1_labeled']:>4}")
    print("pool hit:", agg["pool_hit"])
    if "mcnemar" in agg:
        print("mcnemar:", agg["mcnemar"])
    print(f"-> {rd/'aggregate.json'}")


if __name__ == "__main__":
    main()
