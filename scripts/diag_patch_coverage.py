#!/usr/bin/env python3
"""
diag_patch_coverage.py — patch-wise pattern COVERAGE vs global SigLIP score
---------------------------------------------------------------------------
Uses the ALREADY-RUNNING KNN server's /patch-coverage endpoint, so it reuses the
server's SigLIP (no second model load, and scores are on the SAME scale as
retrieval). Motivation (FashionVLP, CVPR'22): a single global image embedding is
average-pooled and "fails to preserve location-specific salience", so frozen
SigLIP can't tell shoulder-only plaid from all-over plaid. Splitting the garment
into an NxN grid and classifying each patch as "{pattern} fabric" vs "plain
fabric" recovers the spatial signal with no training.

  python diag_patch_coverage.py \
    --images data/eval/rerank_c1 \
    --server http://127.0.0.1:1235/patch-coverage \
    --grid 3 --pattern striped

Source: a dir of jpgs (sent as local paths the SERVER can read), or a --manifest
jsonl with {"path":..,"pattern":..,"color":..,"garment":..}. Paths must be
readable by the server process (same machine).
"""
import argparse, json, glob, os, re
try:
    import requests
except ModuleNotFoundError:
    raise SystemExit("diag_patch_coverage needs 'requests' (pip install requests), "
                     "or run it from an env that has it; the SERVER no longer requires it.")

# filename pattern saved by the other diags: {query_id}_{OPT}_r{rank}.jpg
# query_id itself contains underscores, so anchor on the trailing _<A-D>_r<digits>.
_FN_RE = re.compile(r"^(?P<qid>.+)_(?P<opt>[A-D])_r(?P<rank>\d+)\.jpg$")


def parse_fn(fn):
    m = _FN_RE.match(os.path.basename(fn))
    if not m:
        return None, None
    return m.group("qid"), m.group("opt")


def load_plans(path):
    """Return {query_id: plan}. Plans are the option_plans jsonl."""
    by_qid = {}
    for l in open(path):
        l = l.strip()
        if l:
            p = json.loads(l)
            by_qid[p["query_id"]] = p
    return by_qid


def attrs_for(plans, qid, opt):
    """The (pattern, color, garment) the option was actually built for.
    Returns None if the query or option isn't in the plans."""
    p = plans.get(qid)
    if not p:
        return None
    opts = p.get("options") or {}
    if opt not in opts:
        return None
    a = (opts.get(opt) or {}).get("attributes", {}) or {}
    return (a.get("pattern") or "solid", a.get("color") or "",
            a.get("garment_category") or "garment")

S = requests.Session()


def post_batch(server, srcs, pattern, color, garment, grid, drop_white):
    r = S.post(server, json={"images": srcs, "pattern": pattern, "color": color,
               "garment": garment, "grid": grid, "drop_white": drop_white}, timeout=300)
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=None, help="dir of {qid}_{OPT}_r{n}.jpg files")
    ap.add_argument("--manifest", default=None,
                    help="jsonl with path/pattern/color/garment (bypasses plan lookup)")
    ap.add_argument("--plans", default=None,
                    help="option_plans jsonl; used to look up each option's REAL "
                         "target pattern/color/garment from the filename. Strongly "
                         "recommended with --images so each option is scored against "
                         "its own pattern, not one global --pattern.")
    ap.add_argument("--server", default="http://127.0.0.1:1235/patch-coverage")
    ap.add_argument("--grid", type=int, default=3)
    ap.add_argument("--pattern", default="striped",
                    help="fallback pattern when no --plans/--manifest attrs are found")
    ap.add_argument("--color", default="")
    ap.add_argument("--garment", default="garment")
    ap.add_argument("--skip-patterns", default="",
                    help="comma list to skip, e.g. checkered,plaid")
    ap.add_argument("--include-solid", action="store_true",
                    help="also score options whose target pattern is solid "
                         "(by default solid options are skipped — coverage is "
                         "meaningless for them)")
    ap.add_argument("--no-drop-white", action="store_true")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    drop_white = not args.no_drop_white
    skip = {s.strip().lower() for s in args.skip_patterns.split(",") if s.strip()}
    solid_like = {"", "solid", "plain", "none"}

    plans = load_plans(args.plans) if args.plans else {}

    # build (path, pattern, color, garment, opt, qid) with per-option attrs
    raw = []
    if args.manifest:
        for l in open(args.manifest):
            l = l.strip()
            if not l:
                continue
            d = json.loads(l)
            raw.append((d["path"], (d.get("pattern") or args.pattern),
                        d.get("color") or args.color, d.get("garment") or args.garment,
                        None, None))
    elif args.images:
        for p in sorted(glob.glob(os.path.join(args.images, "*.jpg"))):
            path = os.path.abspath(p)
            qid, opt = parse_fn(p)
            pat, col, gar = args.pattern, args.color, args.garment
            if plans and qid:
                a = attrs_for(plans, qid, opt)
                if a:
                    pat, col, gar = a
            raw.append((path, (pat or "solid").replace("_", " ").lower(),
                        (col or "").replace("_", " ").lower(),
                        (gar or "garment").replace("_", " ").lower(), opt, qid))
    else:
        print("need --images or --manifest"); return

    # filter: skip solid (unless --include-solid) and skip-patterns; these aren't
    # pattern-coverage cases and scoring them just produces noise like the 0/9 rows
    items = []
    skipped_solid = skipped_skip = 0
    for rec in raw:
        pat = rec[1]
        if pat in solid_like and not args.include_solid:
            skipped_solid += 1; continue
        if pat in skip:
            skipped_skip += 1; continue
        items.append(rec)
    items = items[:args.limit]

    if not items:
        print(f"no pattern-bearing options to score "
              f"(skipped {skipped_solid} solid, {skipped_skip} in skip-patterns). "
              f"Did you pass --plans? Without it every file uses --pattern={args.pattern!r}.")
        return
    print(f"scoring {len(items)} option-images "
          f"(skipped {skipped_solid} solid, {skipped_skip} skip-pattern)\n")

    print(f"{'file':<46} {'opt':>3} {'target_pat':>11} {'global_sim':>10} {'cov':>6} {'pat/valid':>10}")
    print("-" * 96)
    rows = []
    i = 0
    while i < len(items):
        chunk = items[i:i + args.batch]
        pat, col, gar = chunk[0][1], chunk[0][2], chunk[0][3]
        same = [r for r in chunk if (r[1], r[2], r[3]) == (pat, col, gar)]
        srcs = [r[0] for r in same]
        try:
            res = post_batch(args.server, srcs, pat, col, gar, args.grid, drop_white)
        except Exception as e:
            print(f"  [server error] {e}"); return
        for rec, r in zip(same, res):
            path, p_pat, _, _, opt, _ = rec
            fn = os.path.basename(path)
            if "error" in r:
                print(f"{fn:<46} {str(opt):>3} {p_pat:>11}  [err: {r['error']}]"); continue
            pv = f"{r['n_pattern']}/{r['n_valid']}"
            print(f"{fn:<46} {str(opt):>3} {p_pat:>11} {r['global_sim']:>10.4f} "
                  f"{r['coverage']:>6.2f} {pv:>10}")
            rows.append((fn, r["global_sim"], r["coverage"], p_pat))
        i += len(same)

    if rows:
        print("-" * 96)
        cv = [r[2] for r in rows]
        print(f"coverage range: {min(cv):.2f}..{max(cv):.2f}  (spread {max(cv)-min(cv):.2f})")
        print("\nNow every option is scored against ITS OWN target pattern (via --plans),")
        print("so a low coverage means that pattern-intended item really lacks an")
        print("all-over pattern. Eyeball high vs low coverage files to confirm the")
        print("patch signal tracks visible pattern extent.")


if __name__ == "__main__":
    main()