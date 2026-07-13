"""Phase 2 — support matrices and availability grading.

Reads the classified candidate catalog and produces:
  corpus_support.json   {garment: {pattern: {"n": distinct, "colors": {color: n}}}}
  pipeline_yield.json   same shape, but only mask-successful candidates
                        (merged from an actual SAM3 collector output dir)

Counts are DISTINCT verified products (URL-deduped), not raw retrieval hits.
Grading: green / yellow / red per (garment, pattern) — see audit_config.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import FASHION_ATTRIBUTE_AXES
from availability_audit.audit_config import (
    AUDIT_DIR, CORPUS_SUPPORT_PATH, PIPELINE_YIELD_PATH,
    GRADE_GREEN_MIN, GRADE_YELLOW_MIN, MASK_SUCCESS_MIN,
)

PATTERNS = FASHION_ATTRIBUTE_AXES["pattern"]


def _load_jsonl(path):
    if not Path(path).exists():
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def build_support(rows):
    """rows -> nested dict garment -> pattern -> {n, colors}."""
    seen = set()
    sup = defaultdict(lambda: defaultdict(lambda: {"n": 0, "colors": defaultdict(int)}))
    for r in rows:
        if not r.get("garment_pass") or "pattern" not in r:
            continue
        key = (r["garment"], r["url"])
        if key in seen:            # distinct products only
            continue
        seen.add(key)
        cell = sup[r["garment"]][r["pattern"]]
        cell["n"] += 1
        if r.get("color"):
            cell["colors"][r["color"]] += 1
    return {g: {p: {"n": c["n"], "colors": dict(c["colors"])}
                for p, c in ps.items()}
            for g, ps in sup.items()}


def grade(corpus_n, mask_rate=None):
    if corpus_n >= GRADE_GREEN_MIN and (mask_rate is None or mask_rate >= MASK_SUCCESS_MIN):
        return "green"
    if corpus_n >= GRADE_YELLOW_MIN:
        return "yellow"
    return "red"


def merge_collector_masks(rows, collector_dir):
    """Attach mask_status from a SAM3 collector output dir (matched by URL).

    The collector writes per-option records containing the candidate url and
    whether SAM3 produced a usable mask; any jsonl under the dir with fields
    {url|image_url, mask_status|mask_ok} is accepted.
    """
    status = {}
    for p in Path(collector_dir).rglob("*.jsonl"):
        for r in _load_jsonl(p):
            url = r.get("url") or r.get("image_url")
            ok = r.get("mask_status", "success" if r.get("mask_ok") else None)
            if url and ok is not None:
                status[url] = (ok == "success" or ok is True)
    n_hit = 0
    for r in rows:
        if r["url"] in status:
            r["mask_pass"] = status[r["url"]]
            n_hit += 1
    print(f"  merged mask status for {n_hit} candidates from {collector_dir}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", type=Path,
                    default=AUDIT_DIR / "availability_catalog.classified.jsonl")
    ap.add_argument("--collector-dir", type=Path, default=None,
                    help="SAM3 collector output dir to derive pipeline_yield")
    args = ap.parse_args()

    rows = _load_jsonl(args.catalog)
    if not rows:
        print(f"no catalog at {args.catalog} — run build_candidate_bank first")
        return

    if args.collector_dir:
        rows = merge_collector_masks(rows, args.collector_dir)

    corpus = build_support(rows)
    CORPUS_SUPPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CORPUS_SUPPORT_PATH.write_text(json.dumps(corpus, indent=2, ensure_ascii=False))

    masked = [r for r in rows if r.get("mask_pass")]
    pyield = build_support(masked)
    PIPELINE_YIELD_PATH.write_text(json.dumps(pyield, indent=2, ensure_ascii=False))

    # grading table
    print(f"\n{'garment':16s} " + " ".join(f"{p:>10s}" for p in PATTERNS))
    for g in sorted(corpus):
        cells = []
        for p in PATTERNS:
            n = corpus[g].get(p, {}).get("n", 0)
            m = pyield.get(g, {}).get(p, {}).get("n", 0)
            rate = (m / n) if (n and masked) else None
            cells.append(f"{n:>4d}/{grade(n, rate)[0].upper()}")
        print(f"{g:16s} " + " ".join(f"{c:>10s}" for c in cells))
    print(f"\nsaved -> {CORPUS_SUPPORT_PATH}\n      -> {PIPELINE_YIELD_PATH}")
    print("cell = distinct_verified/G|Y|R  (mask-rate gating applies only when "
          "--collector-dir was given)")


if __name__ == "__main__":
    main()
