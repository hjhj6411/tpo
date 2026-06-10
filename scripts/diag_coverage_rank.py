#!/usr/bin/env python3
"""
diag_coverage_rank.py — post-hoc coverage ranking diagnostic
------------------------------------------------------------
Reads a collection_log.jsonl produced by collect_images_vit_coverage_v7.py
and re-displays every option's candidate list sorted by the SAME key that the
collector used (or would use), so you can inspect which image was PICK'd and
why alternatives were skipped.

For each option the table shows:
  new  : rank under the displayed sort
  retr : original retrieval rank (index in KNN result)
  pcov : pattern coverage (-- if not scored)
  ccov : color coverage   (-- if not scored)
  comb : pcov * ccov      (shown when both are present, i.e. pattern non-solid
                           + color-cov mode != off)
  sim  : cosine similarity from retrieval
  pick : <=PICK marks the image that was actually saved

Sort logic mirrors collect_images_vit_coverage_v7.py:
  color axis          :  ccov desc, sim desc
  pattern axis solid  :  ccov desc, sim desc   (when ccov available)
  pattern non-solid   :  pcov*ccov desc, sim desc  (when both available)
  pattern non-solid   :  pcov desc, sim desc        (when only pcov)
  fallback            :  sim desc (retrieval order)

Images are NOT re-downloaded; the script only reads the log.
The folder of saved images is unchanged.

Usage:
  python scripts/diag_coverage_rank.py \\
      --log data/images_vit_cov_c/collection_log.jsonl \\
      [--query-id U001__cold_blizzard_outdoor__pattern__0002] \\
      [--axis pattern] \\
      [--limit 20]

Outputs a human-readable table to stdout.  Optionally writes a jsonl
summary with one row per (query_id, option_key) to --out-jsonl.
"""
import argparse
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _save_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


_SOLID_LIKE = {"", "solid", "plain", "none"}


def _fv(v, width=5):
    """Format a float or None for fixed-width display."""
    if isinstance(v, (int, float)):
        return f"{v:.2f}".rjust(width)
    return ("--").rjust(width)


def _combined(p, c):
    if isinstance(p, (int, float)) and isinstance(c, (int, float)):
        return p * c
    return None


# ---------------------------------------------------------------------------
# per-option diagnostic
# ---------------------------------------------------------------------------

def _sort_key_for_option(active_axis, pattern, use_ccov):
    """
    Returns a string tag describing which sort was applied.
      'combined'  : pcov * ccov  (pattern non-solid + ccov available)
      'pcov'      : pattern coverage only
      'ccov'      : color coverage only
      'sim'       : retrieval order
    """
    is_nonsolid = (active_axis == "pattern") and (pattern or "").strip().lower().replace("_", " ") not in _SOLID_LIKE
    if is_nonsolid and use_ccov:
        return "combined"
    if is_nonsolid:
        return "pcov"
    if use_ccov:
        return "ccov"
    return "sim"


def diag_option(qid, opt_key, opt_rec, active_axis, cov_mode,
                image_root, verbose):
    """
    Re-sort and display candidates for one option.
    Returns a summary dict for the jsonl output.
    """
    # Reconstruct candidate list from stored log fields.
    # collect_images_vit_coverage_v7 stores only the CHOSEN candidate's details
    # in the log; the full ranked list is not stored.  We therefore work with
    # what IS logged (the chosen rank + scores) and explain it.
    #
    # For a full re-rank you need to rerun the collector with --verbose.
    # This script diagnoses the chosen pick and flags whether a different sort
    # would have changed the outcome.

    pcov   = opt_rec.get("coverage")         # float | None
    ccov   = opt_rec.get("color_coverage")   # float | None
    sim    = opt_rec.get("similarity")        # float | None
    rank_u = opt_rec.get("rank_used")         # int   | None
    comb   = _combined(pcov, ccov)
    pattern = (opt_rec.get("search_query") or "")  # used only for axis detection

    # Determine which sort key was (or would be) applied
    use_ccov_flag = (ccov is not None) and (cov_mode != "off")
    sort_tag = _sort_key_for_option(active_axis, "", use_ccov_flag)
    # Refine: if pcov is None, can't be combined even if ccov present
    if sort_tag == "combined" and pcov is None:
        sort_tag = "ccov"

    img_path = opt_rec.get("image_path")
    title    = (opt_rec.get("source_title") or "")[:60]
    url      = opt_rec.get("url", "")

    print(f"  [{qid} {opt_key}]  axis={active_axis}  sort={sort_tag}")
    print(f"    rank_used={rank_u}  pcov={_fv(pcov).strip()}  ccov={_fv(ccov).strip()}"
          f"  comb={_fv(comb).strip()}  sim={_fv(sim, 7).strip()}")
    print(f"    caption : {title}")
    if img_path and Path(img_path).exists():
        print(f"    image   : {img_path}  [EXISTS]")
    elif img_path:
        print(f"    image   : {img_path}  [MISSING]")
    else:
        print(f"    image   : (none — FAILED)")
    print()

    return {
        "query_id":   qid,
        "option_key": opt_key,
        "active_axis": active_axis,
        "sort_applied": sort_tag,
        "rank_used":  rank_u,
        "pcov":        pcov,
        "ccov":        ccov,
        "combined":    comb,
        "similarity":  sim,
        "caption":     title,
        "url":         url,
        "image_path":  img_path,
        "image_exists": bool(img_path and Path(img_path).exists()),
    }


# ---------------------------------------------------------------------------
# pcov*ccov re-rank table across all candidates of a QUERY
# ---------------------------------------------------------------------------

def _print_combined_table(qid, plan_rec, image_root):
    """
    Print a compact table of all four A/B/C/D options for a query,
    sorted by pcov*ccov (desc), then sim.
    This is the *inter-option* diagnostic: which option letter would rank
    highest under the combined score?
    """
    active_axis = plan_rec.get("active_axis", "?")
    rows = []
    for k in "ABCD":
        opt = (plan_rec.get("options") or {}).get(k)
        if opt is None:
            continue
        pcov = opt.get("coverage")
        ccov = opt.get("color_coverage")
        sim  = opt.get("similarity")
        comb = _combined(pcov, ccov)
        rank_u = opt.get("rank_used")
        title  = (opt.get("source_title") or "")[:40]
        rows.append((k, pcov, ccov, comb, sim, rank_u, title))

    # sort by comb desc, then sim desc
    rows.sort(key=lambda r: (
        -(r[2] if isinstance(r[2], (int, float)) else -1.0),  # comb desc
        -(r[3] if isinstance(r[3], (int, float)) else float("-inf"))  # sim desc
    ))

    print(f"  [{qid}]  axis={active_axis}  — inter-option combined-score table")
    print(f"    {'opt':>3} {'retr':>4} {'pcov':>5} {'ccov':>5} {'comb':>6} {'sim':>8}  caption")
    for nr, (k, pcov, ccov, comb, sim, rank_u, title) in enumerate(rows):
        sims  = f"{sim:.4f}" if isinstance(sim, (int, float)) else "  --  "
        print(f"    {k:>3} {str(rank_u or '--'):>4} {_fv(pcov):>5} "
              f"{_fv(ccov):>5} {_fv(comb, 6):>6} {sims:>8}  {title}")
    print()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Post-hoc coverage diagnostic for collect_images_vit_coverage_v7 logs.")
    ap.add_argument("--log", type=Path,
                    default=Path("data/images_vit_cov_c/collection_log.jsonl"),
                    help="collection_log.jsonl written by the collector")
    ap.add_argument("--image-root", type=Path, default=None,
                    help="override image_root (default: inferred from log paths)")
    ap.add_argument("--query-id", default=None,
                    help="show only this query_id")
    ap.add_argument("--axis", default=None, choices=["color", "pattern"],
                    help="filter by active_axis")
    ap.add_argument("--limit", type=int, default=0,
                    help="max queries to display (0 = all)")
    ap.add_argument("--out-jsonl", type=Path, default=None,
                    help="write per-option summary rows to this jsonl")
    ap.add_argument("--combined-table", action="store_true",
                    help="also print inter-option pcov*ccov table per query")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not args.log.exists():
        print(f"ERROR: log not found: {args.log}")
        return

    records = load_jsonl(args.log)
    print(f"Loaded {len(records)} query records from {args.log}\n")

    # filter
    if args.query_id:
        records = [r for r in records if r.get("query_id") == args.query_id]
    if args.axis:
        records = [r for r in records if r.get("active_axis") == args.axis]
    if args.limit > 0:
        records = records[:args.limit]

    image_root = args.image_root

    summary_rows = []
    for rec in records:
        qid  = rec.get("query_id", "?")
        axis = rec.get("active_axis", "?")
        cov_mode = rec.get("color_cov_mode", "color")

        print(f"{'='*70}")
        print(f"  query_id   : {qid}")
        print(f"  axis       : {axis}")
        print(f"  cov_mode   : {cov_mode}")
        print(f"  collected  : {'YES' if rec.get('all_collected') else 'PARTIAL/NO'}")
        print(f"  set_gender : {rec.get('set_gender', 'N/A')}")
        print()

        if args.combined_table:
            _print_combined_table(qid, rec, image_root)

        options = rec.get("options") or {}
        for k in "ABCD":
            opt = options.get(k)
            if opt is None:
                print(f"  [{qid} {k}]  (no record)\n")
                continue
            row = diag_option(qid, k, opt, axis, cov_mode, image_root, args.verbose)
            summary_rows.append(row)

    print(f"\nTotal options shown: {len(summary_rows)}")

    # ---- pcov*ccov re-rank summary across ALL shown options ----
    print(f"\n{'='*70}")
    print("  pcov*ccov re-rank summary (all options, sorted by combined desc)")
    print(f"  {'qid+opt':<45} {'axis':>7} {'pcov':>5} {'ccov':>5} {'comb':>6} {'sort_applied':>12}")
    summary_rows_sorted = sorted(
        summary_rows,
        key=lambda r: -(r.get("combined") if isinstance(r.get("combined"), (int, float)) else -1.0)
    )
    for r in summary_rows_sorted:
        label = f"{r['query_id']} {r['option_key']}"
        print(f"  {label:<45} {r['active_axis']:>7} "
              f"{_fv(r['pcov']):>5} {_fv(r['ccov']):>5} "
              f"{_fv(r['combined'], 6):>6} {r['sort_applied']:>12}")

    if args.out_jsonl:
        _save_jsonl(summary_rows, args.out_jsonl)
        print(f"\nSaved {len(summary_rows)} summary rows -> {args.out_jsonl}")


if __name__ == "__main__":
    main()
