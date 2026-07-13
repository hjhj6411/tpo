"""Phase 1 — garment-first candidate bank.

Availability must NOT be measured with complex queries (they are the thing
that fails). Instead: retrieve broadly per garment, then push the verifiers
onto the candidates.

Stages (each resumable, all over HTTP — no local GPU needed):
  retrieve        garment-only alias-union KNN -> candidates.jsonl
  verify-garment  VLM closed-vocabulary garment classification
  classify        pattern via /patch-coverage 6-way argmax,
                  then color via /patch-color-coverage 13-way argmax
                  -> availability_catalog.jsonl

`--pairs pilot` restricts to garments appearing in the pilot pair lists.
"""

import argparse
import base64
import json
import sys
from collections import Counter
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import FASHION_ATTRIBUTE_AXES
from availability_audit.audit_config import (
    KNN_URL, VLM_URL, VLM_MODEL, AUDIT_DIR, CANDIDATES_PATH, CATALOG_PATH,
    TOP_K_PER_GARMENT, KNN_BATCH, PATCH_GRID, GARMENT_RETRIEVAL_ALIASES,
    PILOT_RARE_PAIRS, PILOT_COMMON_PAIRS,
)

PATTERNS = FASHION_ATTRIBUTE_AXES["pattern"]
COLORS = FASHION_ATTRIBUTE_AXES["color"]
GARMENT_VOCAB_DISPLAY = [g.replace("_", " ") for g in FASHION_ATTRIBUTE_AXES["garment_category"]]


def _load_jsonl(path):
    if not Path(path).exists():
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def _append_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _garments_for(pairs_mode):
    if pairs_mode == "pilot":
        return sorted({g for _, g in PILOT_RARE_PAIRS + PILOT_COMMON_PAIRS})
    return list(GARMENT_RETRIEVAL_ALIASES)


# ── stage: retrieve ───────────────────────────────────────

def stage_retrieve(garments, top_k):
    done = {r["garment"] for r in _load_jsonl(CANDIDATES_PATH)}
    for g in garments:
        if g in done:
            print(f"  [skip] {g} (already retrieved)")
            continue
        seen, rows = set(), []
        for alias in GARMENT_RETRIEVAL_ALIASES[g]:
            fetched = 0
            while fetched < top_k:
                k = min(KNN_BATCH, top_k - fetched)
                resp = requests.post(f"{KNN_URL}/knn-service",
                                     json={"text": alias, "num_images": fetched + k},
                                     timeout=120).json()
                batch = resp[fetched:] if isinstance(resp, list) else []
                if not batch:
                    break
                for item in batch:
                    url = item.get("url") or item.get("image_url")
                    if not url or url in seen:
                        continue
                    seen.add(url)
                    rows.append({"garment": g, "alias": alias, "url": url,
                                 "caption": item.get("caption", ""),
                                 "similarity": item.get("similarity")})
                fetched += k
        _append_jsonl(CANDIDATES_PATH, rows)
        print(f"  {g}: {len(rows)} unique candidates "
              f"({len(GARMENT_RETRIEVAL_ALIASES[g])} aliases)")


# ── stage: verify-garment ─────────────────────────────────

VLM_GARMENT_PROMPT = (
    "You are a fashion garment classifier. Look only at the single main "
    "clothing item worn or displayed in the image; ignore the background, "
    "the person, and accessories.\n\n"
    "Classify that garment into exactly ONE of these categories:\n{options}\n- other\n\n"
    "Answer with only the category name, exactly as written above."
)


def _vlm_classify_garment(url):
    prompt = VLM_GARMENT_PROMPT.format(
        options="\n".join(f"- {c}" for c in GARMENT_VOCAB_DISPLAY))
    body = {
        "model": VLM_MODEL,
        "temperature": 0,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": prompt},
        ]}],
    }
    r = requests.post(f"{VLM_URL}/chat/completions", json=body, timeout=180)
    ans = r.json()["choices"][0]["message"]["content"].strip().lower()
    for c in GARMENT_VOCAB_DISPLAY:
        if ans == c or ans.startswith(c):
            return c.replace(" ", "_")
    return "other"


def stage_verify_garment(garments):
    cands = [r for r in _load_jsonl(CANDIDATES_PATH) if r["garment"] in garments]
    done = {r["url"] for r in _load_jsonl(CATALOG_PATH)}
    todo = [r for r in cands if r["url"] not in done]
    print(f"  verifying {len(todo)} candidates (of {len(cands)})")
    buf = []
    for i, r in enumerate(todo, 1):
        try:
            pred = _vlm_classify_garment(r["url"])
        except Exception as e:
            pred = f"error:{type(e).__name__}"
        buf.append({**r, "garment_pred": pred,
                    "garment_pass": pred == r["garment"]})
        if len(buf) >= 50 or i == len(todo):
            _append_jsonl(CATALOG_PATH, buf)
            buf = []
            print(f"  [{i}/{len(todo)}]")


# ── stage: classify (pattern -> color) ────────────────────

def _coverage(endpoint, urls, **params):
    """Batched call to /patch-coverage or /patch-color-coverage."""
    r = requests.post(f"{KNN_URL}/{endpoint}",
                      json={"images": urls, "grid": PATCH_GRID,
                            "drop_white": True, **params},
                      timeout=600)
    return r.json()


def stage_classify(garments, batch=32):
    rows = [r for r in _load_jsonl(CATALOG_PATH)
            if r.get("garment_pass") and r["garment"] in garments
            and "pattern" not in r]
    print(f"  classifying {len(rows)} garment-pass candidates")
    out_path = AUDIT_DIR / "availability_catalog.classified.jsonl"
    for s in range(0, len(rows), batch):
        chunk = rows[s:s + batch]
        urls = [r["url"] for r in chunk]
        g_disp = chunk[0]["garment"].replace("_", " ")
        # 6-way pattern argmax: one batched endpoint call per pattern target
        cov = {}
        for p in PATTERNS:
            res = _coverage("patch-coverage", urls, pattern=p, garment=g_disp)
            cov[p] = [x.get("coverage", 0.0) for x in res]
        for i, r in enumerate(chunk):
            p_best = max(PATTERNS, key=lambda p: cov[p][i])
            r["pattern"] = p_best
            r["pattern_score"] = cov[p_best][i]
        # 13-way color argmax conditioned on the assigned pattern
        ccov = {}
        for c in COLORS:
            res = _coverage("patch-color-coverage", urls, color=c,
                            pattern=chunk[0].get("pattern"), garment=g_disp)
            ccov[c] = [x.get("coverage", 0.0) for x in res]
        for i, r in enumerate(chunk):
            c_best = max(COLORS, key=lambda c: ccov[c][i])
            r["color"] = c_best
            r["color_score"] = ccov[c_best][i]
        _append_jsonl(out_path, chunk)
        print(f"  [{min(s + batch, len(rows))}/{len(rows)}]")
    print(f"  -> {out_path}")
    print("  NOTE: mask_status (SAM3) is measured by the actual collector run; "
          "merge its output via support_matrix.py --collector-dir.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["retrieve", "verify-garment", "classify"])
    ap.add_argument("--pairs", default="all", choices=["all", "pilot"])
    ap.add_argument("--top_k", type=int, default=TOP_K_PER_GARMENT)
    args = ap.parse_args()

    garments = _garments_for(args.pairs)
    print(f"stage={args.stage}  garments={garments}")
    if args.stage == "retrieve":
        stage_retrieve(garments, args.top_k)
    elif args.stage == "verify-garment":
        stage_verify_garment(garments)
    else:
        stage_classify(garments)


if __name__ == "__main__":
    main()
