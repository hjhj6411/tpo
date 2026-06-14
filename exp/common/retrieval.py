"""
common/retrieval.py — retrieval + coverage helpers + offline scoring rules.

Objective : e4 building blocks. Ports the diag helpers so the ablation grid
            reuses the EXACT query/scoring code paths used in collection.
Query styles (literature-grounded; "a person" removed on purpose — product
metadata is the encoders' training text distribution):
  person  : v1 catalog phrase (kept as before/after baseline)
  product : person-free studio-product phrase            [default]
  title   : bare "{color} {pattern} {garment}" (e-commerce title/keyword)
  photo   : CLIP-default "a photo of a ..."
Scoring rules (applied OFFLINE to a fixed pool -> rerank-only comparison,
retrieval effect excluded):
  sim_only | pcov_only | ccov_only | sum_norm(v1 SUM, pcovB)
  | prod_sim(v1 PRODUCT, pcovB*ccov*sim) | cov_product(pcovT*ccov**beta, sim tie)
Input     : knn endpoints (:1234 ViT / :1235 FSigLIP), coverage routes (:1235).
Output    : candidate dicts with sims + coverage metrics; ranked index lists.
Metrics   : consumed by e4 aggregation (ok-rates, pool hit rate).
Run       : imported by e4_retrieval_ablation/run.py.
"""
import requests
from requests.adapters import HTTPAdapter

_S = requests.Session()
_S.headers.update({"User-Agent": "POD-Bench-exp-retrieval/1.0"})
_ad = HTTPAdapter(pool_connections=16, pool_maxsize=32)
_S.mount("http://", _ad)
_S.mount("https://", _ad)

_SOLID_LIKE = {"", "solid", "plain", "none"}
QUERY_STYLES = ("person", "product", "title", "photo")
RULES = ("sim_only", "pcov_only", "ccov_only", "sum_norm", "prod_sim",
         "cov_product")


def _clean(s):
    return (s or "").strip().lower().replace("_", " ")


def build_query(style, garment, color=None, pattern=None, axis=None,
                fallback=None):
    g, c, tp = _clean(garment), _clean(color), _clean(pattern)
    cadj = f"{c} " if c and c not in ("none", "unknown") else ""
    has_pat = (axis == "pattern" and tp not in _SOLID_LIKE)
    if not g:
        base = _clean(fallback) or "clothing"
        return {"person": f"a fashion catalog photo of a person wearing {base}, "
                          f"entire garment visible, plain background",
                "title": base,
                "photo": f"a photo of {base}"}.get(
            style, f"a studio product photo of {base}, full garment visible, "
                   f"plain background")
    core = f"{cadj}{g}".strip()
    if style == "person":
        pat = f" with an all-over {tp} pattern" if has_pat else ""
        return (f"a fashion catalog photo of a person wearing a {core}{pat}, "
                f"front view, entire garment visible, plain background")
    if style == "title":
        return f"{cadj}{tp} {g}".strip() if has_pat else core
    if style == "photo":
        return (f"a photo of a {cadj}{tp} {g}".strip() if has_pat
                else f"a photo of a {core}")
    pat = f" with an all-over {tp} pattern" if has_pat else ""
    return (f"a studio product photo of a {core}{pat}, full garment visible, "
            f"plain background")


def knn_search(url, query, top_k, indice_name="pod_fashion", timeout=60):
    r = _S.post(url, json={"text": query, "modality": "image",
                           "num_images": top_k, "num_result_ids": top_k,
                           "indice_name": indice_name}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _cov(url, payload, n, field="coverage"):
    try:
        r = _S.post(url, json=payload, timeout=300)
        r.raise_for_status()
        res = r.json()
        vals = [x.get(field) if isinstance(x, dict) else None for x in res]
        oth = [x.get("top_other") if isinstance(x, dict) else None for x in res]
    except Exception as e:                              # noqa: BLE001
        print(f"    [cov] {e}")
        vals, oth = [None] * n, [None] * n
    return (vals + [None] * n)[:n], (oth + [None] * n)[:n]


def score_pool(urls, target, cov_urls, grid=5):
    """One coverage pass per pool: pcovB / pcovT / ccov / garment top-2."""
    n = len(urls)
    pat = target.get("pattern") or "solid"
    col, gar = target.get("color"), target.get("garment")
    out = {"pcovB": [None] * n, "pcovT": [None] * n, "pat_other": [None] * n,
           "ccov": [None] * n, "col_other": [None] * n,
           "g_match": [None] * n, "g_top1": [None] * n}
    if cov_urls.get("pcovB"):
        out["pcovB"], _ = _cov(cov_urls["pcovB"],
                               {"images": urls, "pattern": pat, "color": col,
                                "garment": gar, "grid": grid,
                                "drop_white": True}, n)
    if cov_urls.get("pcovT"):
        out["pcovT"], out["pat_other"] = _cov(
            cov_urls["pcovT"], {"images": urls, "pattern": pat, "color": col,
                                "grid": grid, "drop_white": True}, n)
    if cov_urls.get("ccov") and col:
        out["ccov"], out["col_other"] = _cov(
            cov_urls["ccov"], {"images": urls, "color": col, "pattern": pat,
                               "garment": gar, "grid": grid,
                               "drop_white": True}, n)
    if cov_urls.get("garment") and gar:
        try:
            r = _S.post(cov_urls["garment"],
                        json={"images": urls, "garment": gar, "top_n": 2},
                        timeout=300)
            r.raise_for_status()
            res = r.json()
            out["g_match"] = [x.get("match") if isinstance(x, dict) else None
                              for x in res][:n]
            out["g_top1"] = [(x.get("top") or [None])[0]
                             if isinstance(x, dict) else None for x in res][:n]
        except Exception as e:                          # noqa: BLE001
            print(f"    [g] {e}")
    return out


def _f(x):
    return x if isinstance(x, (int, float)) else None


def _minmax(vals):
    xs = [_f(v) for v in vals]
    known = [x for x in xs if x is not None]
    if not known:
        return [0.0] * len(xs)
    lo, hi = min(known), max(known)
    if hi - lo < 1e-12:
        return [1.0 if x is not None else 0.0 for x in xs]
    return [0.0 if x is None else (x - lo) / (hi - lo) for x in xs]


def apply_rule(rule, sims, m, beta=1.0):
    """Ranked candidate indices under one scoring rule (sim tie-break where
    the rule excludes sim; v1 rules reproduced faithfully for the comparison)."""
    n = len(sims)
    s = [_f(x) or 0.0 for x in sims]
    pB = [_f(x) or 0.0 for x in m["pcovB"]]
    pT = [_f(x) or 0.0 for x in m["pcovT"]]
    cc = [_f(x) or 0.0 for x in m["ccov"]]
    if rule == "sim_only":
        key = [(-s[i], i) for i in range(n)]
    elif rule == "pcov_only":
        key = [(-pT[i], -s[i], i) for i in range(n)]
    elif rule == "ccov_only":
        key = [(-cc[i], -s[i], i) for i in range(n)]
    elif rule == "sum_norm":
        nm = list(zip(_minmax(m["pcovB"]), _minmax(m["ccov"]), _minmax(sims)))
        key = [(-(a + b + c), i) for i, (a, b, c) in enumerate(nm)]
    elif rule == "prod_sim":
        key = [(-(pB[i] * cc[i] * s[i]), i) for i in range(n)]
    elif rule == "cov_product":
        key = [(-(pT[i] * (cc[i] ** beta)), -s[i], i) for i in range(n)]
    else:
        raise ValueError(f"unknown rule {rule}")
    return [i for _, i in sorted(zip(key, range(n)))]
