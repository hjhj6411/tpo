#!/usr/bin/env python3
"""
diag_query_methods.py — compare query-augmentation METHODS on the same options
------------------------------------------------------------------------------
For each probed option, runs several query strategies against the SAME corpus
and lays the results side by side so you can see, qualitatively:

  (1) how different each method's top-1 is (overlap vs divergence)
  (2) which INDIVIDUAL template gives the best result
  (3) whether the ENSEMBLE actually beats its component templates

Methods compared per option:
  raw        : the plan's original search_query  ("striped white t shirt")
  sentence   : the v2 single natural sentence    ("a full-body product photo ...")
  T0..Tn     : each individual fashion template  (the ensemble's components)
  ENSEMBLE   : server-side embedding average over T0..Tn  (/knn-service-ensemble)

Single-text methods (raw, sentence, T0..Tn) hit --client-url (/knn-service).
ENSEMBLE hits --ensemble-url (/knn-service-ensemble).

  python scripts/diag_query_methods.py \
  --plans data/options/option_plans.jsonl \
  --client-url   http://127.0.0.1:1235/knn-service \
  --ensemble-url http://127.0.0.1:1235/knn-service-ensemble \
  --top_k 10 --n-options 5 \
  --save-images data/eval/qmethods
"""
import argparse, io, json, hashlib
from pathlib import Path
import requests
from PIL import Image

S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0 (POD-Bench-qdiag)"})

# ---- query builders (mirror collector logic) -------------------------------
_SOLID_LIKE = {"", "solid", "plain", "none"}

def _clean(s):
    return (s or "").strip().lower().replace("_", " ")

def sentence_query(garment, color=None, pattern=None, axis=None, fallback=None):
    g = _clean(garment)
    if not g:
        base = _clean(fallback) or "a clothing item"
        return f"a full-body product photo of {base}, the whole item visible and not cropped"
    c = _clean(color); tp = _clean(pattern)
    color_adj = f" {c}" if c and c not in ("none", "unknown") else ""
    pat = ""
    if axis == "pattern" and tp not in _SOLID_LIKE:
        pat = f" with an all-over {tp} print covering the whole garment"
    return (f"a full-body product photo of a{color_adj} {g}{pat}, "
            f"the entire garment clearly visible and not cropped")

_TEMPLATES_BASE = [
    "a photo of a {c}{g}{pat}",
    "a product photo of a {c}{g}{pat}",
    "a full-body photo of a {c}{g}{pat}, the whole garment visible",
    "front view of a {c}{g}{pat} on a plain background",
    "a {c}{g}{pat}, studio product shot, not cropped",
    "a catalog image of a {c}{g}{pat}",
]
_TEMPLATES_PATTERN = [
    "a {c}{g} with the {patname} pattern covering the whole garment",
    "a {c}{g} printed all over with {patname}, full garment shown",
]

def build_templates(garment, color=None, pattern=None, axis=None, fallback=None):
    g = _clean(garment)
    if not g:
        base = _clean(fallback) or "a clothing item"
        return [f"a photo of {base}",
                f"a product photo of {base}, the whole item visible"]
    c = _clean(color); tp = _clean(pattern)
    c_adj = f"{c} " if c and c not in ("none", "unknown") else ""
    nonsolid = (axis == "pattern" and tp not in _SOLID_LIKE)
    pat_clause = f" with an all-over {tp} pattern" if nonsolid else ""
    prompts = [t.format(c=c_adj, g=g, pat=pat_clause).replace("  ", " ").strip()
               for t in _TEMPLATES_BASE]
    if nonsolid:
        prompts += [t.format(c=c_adj, g=g, patname=tp).replace("  ", " ").strip()
                    for t in _TEMPLATES_PATTERN]
    seen, uniq = set(), []
    for p in prompts:
        if p not in seen:
            seen.add(p); uniq.append(p)
    return uniq

# ---- io / search -----------------------------------------------------------
def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]

def _url(c):
    return c.get("url") or c.get("image_url")

def search_text(url, text, k, indice="pod_fashion"):
    try:
        r = S.post(url, json={"text": text, "modality": "image",
                   "num_images": k, "num_result_ids": k, "indice_name": indice}, timeout=60)
        r.raise_for_status(); return r.json()
    except Exception as e:
        print(f"    [search failed] {text!r}: {e}"); return []

def search_ensemble(url, texts, k, indice="pod_fashion"):
    try:
        r = S.post(url, json={"texts": texts, "modality": "image",
                   "num_images": k, "num_result_ids": k, "indice_name": indice}, timeout=60)
        r.raise_for_status(); return r.json()
    except Exception as e:
        print(f"    [ensemble failed] {len(texts)} texts: {e}"); return []

def short_url(u, n=14):
    if not u: return "—"
    h = hashlib.md5(u.encode()).hexdigest()[:n]
    return h

def save_img(url, path):
    try:
        r = S.get(url, timeout=15); r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        img.save(path, "JPEG", quality=90); return True
    except Exception:
        return False

# ---- main ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", required=True)
    ap.add_argument("--client-url", default="http://127.0.0.1:1235/knn-service")
    ap.add_argument("--ensemble-url", default="http://127.0.0.1:1235/knn-service-ensemble")
    ap.add_argument("--indice-name", default="pod_fashion")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--n-options", type=int, default=5)
    ap.add_argument("--axis", default=None, help="only probe this active_axis")
    ap.add_argument("--save-images", default=None,
                    help="dir to save each method's top-1 jpg for eyeballing")
    args = ap.parse_args()

    plans = load_jsonl(args.plans)
    if args.axis:
        plans = [p for p in plans if p.get("active_axis") == args.axis]

    specs = []
    for p in plans:
        ax = p.get("active_axis")
        for k in "ABCD":
            o = (p.get("options") or {}).get(k) or {}
            specs.append((p["query_id"], k, o.get("attributes", {}) or {},
                          o.get("search_query"), ax))
            if len(specs) >= args.n_options: break
        if len(specs) >= args.n_options: break

    for (qid, k, attrs, sq, axis) in specs:
        garment = attrs.get("garment_category")
        color, pattern = attrs.get("color"), attrs.get("pattern")
        raw = sq or ""
        sent = sentence_query(garment, color, pattern, axis, sq)
        templates = build_templates(garment, color, pattern, axis, sq)

        print("=" * 100)
        print(f"{qid}  opt {k}  [axis={axis}]  target: color={color} pattern={pattern} garment={garment}")
        print(f"   raw      : {raw}")
        print(f"   sentence : {sent}")
        for i, t in enumerate(templates):
            print(f"   T{i}       : {t}")
        print()

        # run every method
        methods = [("raw", raw), ("sentence", sent)]
        methods += [(f"T{i}", t) for i, t in enumerate(templates)]
        results = {}   # name -> list of candidate dicts
        for name, text in methods:
            results[name] = search_text(args.client_url, text, args.top_k, args.indice_name) if text else []
        results["ENSEMBLE"] = search_ensemble(args.ensemble_url, templates, args.top_k, args.indice_name)

        # (1)+(2)+(3): table of each method's top-1 + sim, with a shared short-hash
        #              so identical images are visually obvious across methods
        print(f"   {'method':>9} {'top1_sim':>9}  {'img_hash':>14}  caption")
        top1_hashes = {}
        ens_sim = None
        for name, _ in methods + [("ENSEMBLE", None)]:
            cands = results.get(name, [])
            if not cands:
                print(f"   {name:>9} {'—':>9}  {'—':>14}  (no results)")
                continue
            c0 = cands[0]; u0 = _url(c0); h0 = short_url(u0)
            top1_hashes[name] = h0
            sim0 = c0.get("similarity")
            if name == "ENSEMBLE": ens_sim = sim0
            sims = f"{sim0:.4f}" if isinstance(sim0, (int, float)) else "—"
            print(f"   {name:>9} {sims:>9}  {h0:>14}  {(c0.get('caption') or '')[:50]}")
            if args.save_images:
                save_img(u0, f"{args.save_images}/{qid}_{k}_{name}.jpg")
        print()

        # (1) overlap analysis: how many DISTINCT top-1 images across methods
        distinct = set(top1_hashes.values())
        print(f"   -> distinct top-1 images across {len(top1_hashes)} methods: {len(distinct)}")
        # group methods by which image they returned
        by_img = {}
        for nm, h in top1_hashes.items():
            by_img.setdefault(h, []).append(nm)
        for h, names in sorted(by_img.items(), key=lambda x: -len(x[1])):
            print(f"        {h}: {', '.join(names)}")

        # (2) best individual template by top-1 sim
        tmpl_sims = []
        for i in range(len(templates)):
            cs = results.get(f"T{i}", [])
            if cs and isinstance(cs[0].get("similarity"), (int, float)):
                tmpl_sims.append((f"T{i}", cs[0]["similarity"]))
        if tmpl_sims:
            best = max(tmpl_sims, key=lambda x: x[1])
            worst = min(tmpl_sims, key=lambda x: x[1])
            print(f"   -> best individual template: {best[0]} (sim {best[1]:.4f}); "
                  f"worst: {worst[0]} (sim {worst[1]:.4f})")
            # (3) ensemble vs components
            if ens_sim is not None:
                ens_h = top1_hashes.get("ENSEMBLE")
                matches = [nm for nm, h in top1_hashes.items() if h == ens_h and nm != "ENSEMBLE"]
                verdict = ("matches " + ", ".join(matches)) if matches else "UNIQUE (no single method picked it)"
                cmp_best = ("ABOVE" if ens_sim > best[1] else "below") + " best template sim"
                print(f"   -> ENSEMBLE top-1 sim {ens_sim:.4f} ({cmp_best}); ensemble image {verdict}")
        print()


if __name__ == "__main__":
    main()