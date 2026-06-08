#!/usr/bin/env python3
"""
diag_coverage_queries.py — compare PATTERN-EXTENT query phrasings, qualitatively
--------------------------------------------------------------------------------
Goal: the retriever keeps picking LOCALIZED-pattern garments (plaid only on the
shoulder, stripe only on a panel) when the option wants an ALL-OVER pattern.
This probes, on the SAME pattern-axis options, whether different ways of saying
"the pattern covers the whole garment" actually pull whole-pattern images.

It is built on the T4 skeleton that survived the earlier method comparison:
    "a {color} {garment} {EXTENT}, studio product shot, not cropped"
Only the EXTENT phrase changes between candidates, so any difference in the
retrieved top-1 is attributable to that phrase alone (one knob at a time).

Candidate extent phrasings (one per family):
  C0_base        : (no extent phrase)            -- control, T4 with plain pattern
  C1_allover     : with an all-over {pat} pattern
  C2_vocab       : throughout / entire-surface vocabulary
  C3_negation    : all-over {pat}, not just on one part      (negation test)
  C4_repeat      : with a repeating {pat} print across the whole fabric

Single-text searches against --client-url (/knn-service).

  python diag_coverage_queries.py \
    --plans data/options/option_plans.jsonl \
    --client-url http://127.0.0.1:1235/knn-service \
    --axis pattern --top_k 10 --n-options 6 \
    --save-images data/eval/coverage_q
"""
import argparse, io, json, hashlib
from pathlib import Path
import requests
from PIL import Image

S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0 (POD-Bench-covq)"})

_SOLID_LIKE = {"", "solid", "plain", "none"}

def _clean(s):
    return (s or "").strip().lower().replace("_", " ")

# Each builder takes (color_adj, garment, pattern) and returns ONE query string.
# color_adj already has trailing space or is "".
def _c0_base(c, g, p):
    # control: T4 skeleton, pattern named inline but NO extent phrase
    pat = f" {p}" if p not in _SOLID_LIKE else ""
    return f"a {c}{g}{pat}, studio product shot, not cropped".replace("  ", " ")

def _c1_allover(c, g, p):
    ext = f" with an all-over {p} pattern" if p not in _SOLID_LIKE else ""
    return f"a {c}{g}{ext}, studio product shot, not cropped".replace("  ", " ")

def _c2_vocab(c, g, p):
    ext = f" with {p} pattern throughout the entire surface" if p not in _SOLID_LIKE else ""
    return f"a {c}{g}{ext}, studio product shot, not cropped".replace("  ", " ")

def _c3_negation(c, g, p):
    ext = f" with an all-over {p} pattern, not just on one part" if p not in _SOLID_LIKE else ""
    return f"a {c}{g}{ext}, studio product shot, not cropped".replace("  ", " ")

def _c4_repeat(c, g, p):
    ext = f" with a repeating {p} print across the whole fabric" if p not in _SOLID_LIKE else ""
    return f"a {c}{g}{ext}, studio product shot, not cropped".replace("  ", " ")

CANDIDATES = [
    ("C0_base", _c0_base),
    ("C1_allover", _c1_allover),
    ("C2_vocab", _c2_vocab),
    ("C3_negation", _c3_negation),
    ("C4_repeat", _c4_repeat),
]

def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]

def _url(c):
    return c.get("url") or c.get("image_url")

def short_hash(u, n=14):
    return hashlib.md5(u.encode()).hexdigest()[:n] if u else "—"

def search_text(url, text, k, indice="pod_fashion"):
    try:
        r = S.post(url, json={"text": text, "modality": "image",
                   "num_images": k, "num_result_ids": k, "indice_name": indice}, timeout=60)
        r.raise_for_status(); return r.json()
    except Exception as e:
        print(f"    [search failed] {text!r}: {e}"); return []

def save_img(url, path):
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
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
    ap.add_argument("--indice-name", default="pod_fashion")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--n-options", type=int, default=6)
    ap.add_argument("--axis", default="pattern",
                    help="only probe this active_axis (default pattern; the extent problem)")
    ap.add_argument("--skip-patterns", default="",
                    help="comma list of pattern words to skip (e.g. plaid,checkered)")
    ap.add_argument("--save-images", default=None)
    args = ap.parse_args()
    skip = {s.strip().lower() for s in args.skip_patterns.split(",") if s.strip()}

    if args.save_images:
        Path(args.save_images).mkdir(parents=True, exist_ok=True)
        print(f"saving top-1 images to: {Path(args.save_images).resolve()}\n")

    plans = load_jsonl(args.plans)
    if args.axis:
        plans = [p for p in plans if p.get("active_axis") == args.axis]

    # collect pattern-axis options that actually have a non-solid pattern target
    specs = []
    for p in plans:
        ax = p.get("active_axis")
        for k in "ABCD":
            o = (p.get("options") or {}).get(k) or {}
            a = o.get("attributes", {}) or {}
            pat = _clean(a.get("pattern"))
            if pat in _SOLID_LIKE or pat in skip:
                continue
            specs.append((p["query_id"], k, a, ax))
            break
        if len(specs) >= args.n_options:
            break

    if not specs:
        print("no non-solid pattern-axis options found (check --axis / --skip-patterns)")
        return

    # per-candidate: count how often it returns an image NO other candidate returned,
    # and track sims, to summarize at the end
    agg = {name: {"top1_sims": [], "unique_top1": 0} for name, _ in CANDIDATES}

    for (qid, k, attrs, axis) in specs:
        g = _clean(attrs.get("garment_category"))
        c = _clean(attrs.get("color"))
        p = _clean(attrs.get("pattern"))
        c_adj = f"{c} " if c and c not in ("none", "unknown") else ""

        print("=" * 100)
        print(f"{qid}  opt {k}  [axis={axis}]  color={c} pattern={p} garment={g}")
        top1 = {}
        for name, fn in CANDIDATES:
            q = fn(c_adj, g, p)
            cands = search_text(args.client_url, q, args.top_k, args.indice_name)
            c0 = cands[0] if cands else None
            h = short_hash(_url(c0)) if c0 else "—"
            sim = c0.get("similarity") if c0 else None
            top1[name] = h
            if isinstance(sim, (int, float)):
                agg[name]["top1_sims"].append(sim)
            sims = f"{sim:.4f}" if isinstance(sim, (int, float)) else "—"
            print(f"   {name:>12}  sim={sims:>8}  {h}  | {q}")
            if c0:
                print(f"   {'':>12}  cap: {(c0.get('caption') or '')[:70]}")
            if args.save_images and c0:
                save_img(_url(c0), f"{args.save_images}/{qid}_{k}_{name}.jpg")

        # overlap: which candidates agree, which is unique
        by_img = {}
        for nm, h in top1.items():
            by_img.setdefault(h, []).append(nm)
        distinct = [h for h in by_img if h != "—"]
        print(f"   -> distinct top-1 across {len(CANDIDATES)} candidates: {len(distinct)}")
        for h, names in sorted(by_img.items(), key=lambda x: -len(x[1])):
            tag = "  <- shared" if len(names) > 1 else "  <- UNIQUE"
            print(f"        {h}: {', '.join(names)}{tag}")
            if len(names) == 1 and h != "—":
                agg[names[0]]["unique_top1"] += 1
        print()

    # summary across all probed options
    print("=" * 100)
    print("SUMMARY across", len(specs), "pattern-axis options")
    print(f"   {'candidate':>12}  {'mean_top1_sim':>13}  {'n_unique_top1':>13}")
    for name, _ in CANDIDATES:
        sims = agg[name]["top1_sims"]
        ms = sum(sims) / len(sims) if sims else float("nan")
        print(f"   {name:>12}  {ms:>13.4f}  {agg[name]['unique_top1']:>13}")
    print("\nNote: higher mean sim is NOT necessarily better (it may just mean the")
    print("phrase drifted toward easy generic matches). Judge by the SAVED IMAGES:")
    print("does the candidate pull WHOLE-garment pattern, or localized patches?")


if __name__ == "__main__":
    main()