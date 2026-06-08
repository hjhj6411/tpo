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
import argparse, json, glob, os
import requests

S = requests.Session()


def post_batch(server, srcs, pattern, color, garment, grid, drop_white):
    r = S.post(server, json={"images": srcs, "pattern": pattern, "color": color,
               "garment": garment, "grid": grid, "drop_white": drop_white}, timeout=300)
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--server", default="http://127.0.0.1:1235/patch-coverage")
    ap.add_argument("--grid", type=int, default=3)
    ap.add_argument("--pattern", default="striped")
    ap.add_argument("--color", default="")
    ap.add_argument("--garment", default="garment")
    ap.add_argument("--no-drop-white", action="store_true",
                    help="keep background tiles in the denominator")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    drop_white = not args.no_drop_white

    items = []
    if args.manifest:
        for l in open(args.manifest):
            l = l.strip()
            if l:
                d = json.loads(l)
                items.append((d["path"], d.get("pattern") or args.pattern,
                              d.get("color") or args.color, d.get("garment") or args.garment))
    elif args.images:
        for p in sorted(glob.glob(os.path.join(args.images, "*.jpg")))[:args.limit]:
            items.append((os.path.abspath(p), args.pattern, args.color, args.garment))
    else:
        print("need --images or --manifest"); return
    items = items[:args.limit]

    print(f"{'file':<46} {'global_sim':>10} {'cov':>6} {'pat/valid':>10}")
    print("-" * 78)
    rows = []
    # group by (pattern,color,garment) so a batch shares the text anchors
    i = 0
    while i < len(items):
        chunk = items[i:i + args.batch]
        # batch only items that share attrs (here all same when from --images)
        pat, col, gar = chunk[0][1], chunk[0][2], chunk[0][3]
        same = [(p, a, b, c) for (p, a, b, c) in chunk if (a, b, c) == (pat, col, gar)]
        rest = [x for x in chunk if x not in same]
        srcs = [p for (p, _, _, _) in same]
        try:
            res = post_batch(args.server, srcs, pat, col, gar, args.grid, drop_white)
        except Exception as e:
            print(f"  [server error] {e}"); return
        for (path, _, _, _), r in zip(same, res):
            fn = os.path.basename(path)
            if "error" in r:
                print(f"{fn:<46}  [err: {r['error']}]"); continue
            pv = f"{r['n_pattern']}/{r['n_valid']}"
            print(f"{fn:<46} {r['global_sim']:>10.4f} {r['coverage']:>6.2f} {pv:>10}")
            rows.append((fn, r["global_sim"], r["coverage"]))
        i += len(same) if same else args.batch

    if rows:
        print("-" * 78)
        gs = [r[1] for r in rows]; cv = [r[2] for r in rows]
        print(f"global_sim range: {min(gs):.4f}..{max(gs):.4f}  (spread {max(gs)-min(gs):.4f})")
        print(f"coverage   range: {min(cv):.2f}..{max(cv):.2f}  (spread {max(cv)-min(cv):.2f})")
        print("\nIf global_sim is bunched but coverage spreads, the patch method adds")
        print("the spatial signal SigLIP's global vector lost. Eyeball a high-coverage")
        print("file (truly all-over?) vs a low one (localized/near-solid?).")


if __name__ == "__main__":
    main()