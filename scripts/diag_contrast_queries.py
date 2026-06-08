#!/usr/bin/env python3
"""
diag_contrast_queries.py — compare pattern CONTRAST/SALIENCE phrasings
----------------------------------------------------------------------
Earlier (diag_coverage_queries) the all-over EXTENT phrasing was shown to work:
"with an all-over {pat} pattern" reliably pulls whole-garment patterns. The
remaining failure is low-CONTRAST cases (e.g. faint tonal stripes on black, pale
gingham) where the pattern covers the garment but is barely salient, so the
retriever ranks it like a near-solid.

This holds the proven all-over phrase FIXED and varies ONLY a contrast/salience
modifier, so any change in the retrieved top-1 is attributable to that modifier.

Base (fixed): "a {color} {garment} with an all-over {MOD}{pat} pattern, studio product shot, not cropped"

Candidates (MOD slot):
  K0_allover_only : ""                         -- control: extent only, no contrast word
  K1_bold         : "bold "
  K2_highcontrast : "high-contrast "
  K3_distinct     : "distinct, clearly visible "
  K4_strong_color : "{pat} in two strongly contrasting colors"   (restructured clause)

K4 restructures the clause rather than prefixing an adjective, to test whether
naming the contrast mechanism (two colors) beats a vague adjective.

python scripts/diag_contrast_queries.py \
  --plans data/options/option_plans.jsonl \
  --client-url http://127.0.0.1:1235/knn-service \
  --axis pattern --skip-patterns plaid \
  --top_k 10 --n-options 6 \
  --save-images data/eval/contrast_q
"""
import argparse, io, json, hashlib
from pathlib import Path
import requests
from PIL import Image

S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0 (POD-Bench-contrastq)"})

_SOLID_LIKE = {"", "solid", "plain", "none"}

def _clean(s):
    return (s or "").strip().lower().replace("_", " ")

# All builders share the proven base + all-over phrase; only the contrast modifier differs.
def _base(c, g, body):
    return f"a {c}{g}{body}, studio product shot, not cropped".replace("  ", " ").strip()

def _k0_allover_only(c, g, p):
    body = f" with an all-over {p} pattern" if p not in _SOLID_LIKE else ""
    return _base(c, g, body)

def _k1_bold(c, g, p):
    body = f" with an all-over bold {p} pattern" if p not in _SOLID_LIKE else ""
    return _base(c, g, body)

def _k2_highcontrast(c, g, p):
    body = f" with an all-over high-contrast {p} pattern" if p not in _SOLID_LIKE else ""
    return _base(c, g, body)

def _k3_distinct(c, g, p):
    body = f" with an all-over distinct, clearly visible {p} pattern" if p not in _SOLID_LIKE else ""
    return _base(c, g, body)

def _k4_strong_color(c, g, p):
    body = (f" with an all-over {p} pattern in two strongly contrasting colors"
            if p not in _SOLID_LIKE else "")
    return _base(c, g, body)

CANDIDATES = [
    ("K0_allover_only", _k0_allover_only),
    ("K1_bold", _k1_bold),
    ("K2_highcontrast", _k2_highcontrast),
    ("K3_distinct", _k3_distinct),
    ("K4_strong_color", _k4_strong_color),
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
    ap.add_argument("--axis", default="pattern")
    ap.add_argument("--skip-patterns", default="")
    ap.add_argument("--save-images", default=None)
    args = ap.parse_args()
    skip = {s.strip().lower() for s in args.skip_patterns.split(",") if s.strip()}

    if args.save_images:
        Path(args.save_images).mkdir(parents=True, exist_ok=True)
        print(f"saving top-1 images to: {Path(args.save_images).resolve()}\n")

    plans = load_jsonl(args.plans)
    if args.axis:
        plans = [p for p in plans if p.get("active_axis") == args.axis]

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
        print("no non-solid pattern-axis options found")
        return

    agg = {name: {"top1_sims": [], "unique_top1": 0,
                  "differs_from_control": 0} for name, _ in CANDIDATES}

    for (qid, k, attrs, axis) in specs:
        g = _clean(attrs.get("garment_category"))
        c = _clean(attrs.get("color"))
        p = _clean(attrs.get("pattern"))
        c_adj = f"{c} " if c and c not in ("none", "unknown") else ""

        print("=" * 100)
        print(f"{qid}  opt {k}  [axis={axis}]  color={c} pattern={p} garment={g}")
        top1 = {}
        control_h = None
        for name, fn in CANDIDATES:
            q = fn(c_adj, g, p)
            cands = search_text(args.client_url, q, args.top_k, args.indice_name)
            c0 = cands[0] if cands else None
            h = short_hash(_url(c0)) if c0 else "—"
            sim = c0.get("similarity") if c0 else None
            top1[name] = h
            if name == "K0_allover_only":
                control_h = h
            if isinstance(sim, (int, float)):
                agg[name]["top1_sims"].append(sim)
            sims = f"{sim:.4f}" if isinstance(sim, (int, float)) else "—"
            print(f"   {name:>16}  sim={sims:>8}  {h}  | {q}")
            if c0:
                print(f"   {'':>16}  cap: {(c0.get('caption') or '')[:68]}")
            if args.save_images and c0:
                save_img(_url(c0), f"{args.save_images}/{qid}_{k}_{name}.jpg")

        # does each contrast variant actually change the pick vs control (K0)?
        for name, _ in CANDIDATES:
            if name == "K0_allover_only":
                continue
            if control_h is not None and top1.get(name) not in (control_h, "—"):
                agg[name]["differs_from_control"] += 1

        by_img = {}
        for nm, h in top1.items():
            by_img.setdefault(h, []).append(nm)
        distinct = [h for h in by_img if h != "—"]
        print(f"   -> distinct top-1 across {len(CANDIDATES)} candidates: {len(distinct)}")
        for h, names in sorted(by_img.items(), key=lambda x: -len(x[1])):
            tag = "  <- shared" if len(names) > 1 else "  <- UNIQUE"
            note = "  (= control K0)" if h == control_h and len(names) > 1 else ""
            print(f"        {h}: {', '.join(names)}{tag}{note}")
            if len(names) == 1 and h != "—":
                agg[names[0]]["unique_top1"] += 1
        print()

    print("=" * 100)
    print("SUMMARY across", len(specs), "pattern-axis options")
    print(f"   {'candidate':>16}  {'mean_top1_sim':>13}  {'differs_K0':>10}  {'unique':>6}")
    for name, _ in CANDIDATES:
        sims = agg[name]["top1_sims"]
        ms = sum(sims) / len(sims) if sims else float("nan")
        dk = "" if name == "K0_allover_only" else f"{agg[name]['differs_from_control']:>10}"
        print(f"   {name:>16}  {ms:>13.4f}  {dk:>10}  {agg[name]['unique_top1']:>6}")
    print("\nRead via SAVED IMAGES, not sim alone: for the low-contrast cases (faint")
    print("tonal stripe, pale gingham), does the contrast word pull a MORE salient")
    print("pattern than K0? 'differs_K0' = how often the modifier changed the pick at all;")
    print("if a modifier rarely differs from K0, CLIP is ignoring it.")


if __name__ == "__main__":
    main()
