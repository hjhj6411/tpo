#!/usr/bin/env python3
"""Screen which (garment, pattern) combinations can actually be found as images.

WHY
---
The option planner may specify combinations like `polka_dot puffer_jacket` that
are fine as text but barely exist as products. Since every option must be
realized as an IMAGE, a combination nobody manufactures cannot be retrieved.
This script measures, for all 20 x 6 = 120 combinations, how many of the top-k
retrieved candidates score non-zero on both the garment and the pattern axis.

This is MEASUREMENT ONLY. It does not filter anything, does not touch
construction/, and does not regenerate data. The resulting table is the evidence
for a later decision about which combinations to exclude and at what threshold.

WHAT "SCORE > 0" MEANS  (needed to cite these numbers in the paper)
-------------------------------------------------------------------
For each retrieved candidate image we compute two scores with the SAME scorers
`retrieval/collect_topk_scored_gallery.py` uses — this module imports them, it
does not reimplement them:

  garment_score : one global Qwen-VL call, `garment_prompt(garment)`, parsed as
                  JSON {"score": 0..1}. A failed/unparseable call scores 0.0.
  pattern_score : for pattern == "solid", one global call with `solid_prompt()`.
                  Otherwise `score_pattern()` patch coverage — the image is cut
                  into a `--tile-grid` grid, up to `--pattern-max-tiles`
                  non-background tiles are scored with `pattern_tile_prompt()`,
                  and the score is
                      (# tiles with score >= --pattern-threshold) / (# tiles scored).
                  So a non-zero pattern score means AT LEAST ONE TILE passed the
                  threshold.

  cell score    = sqrt(garment_score * pattern_score)

`nonzero` counts candidates whose cell score > 0, which (being a geometric mean)
happens exactly when BOTH garment_score > 0 AND pattern_score > 0. Colour is
deliberately excluded: this grid is garment x pattern only.

THIS NUMBER IS AN UPPER BOUND
-----------------------------
A non-zero score does NOT certify that the image really shows that garment in
that pattern. One passing tile is enough, the VLM is not calibrated here, and no
human has checked the result. Use it to rule OUT combinations that are clearly
unobtainable (near-zero cells), never to claim a combination is well covered.
A high cell means "worth collecting", not "verified".

Captions are never used — the corpus captions are shop marketing copy and are
not valid labels. Scoring is image-only.

RUNNING IT ON 4 GPUs
--------------------
This script never touches CUDA — the vLLM servers own the GPUs, so "use 4 GPUs"
means "serve four endpoints and keep all of them busy". Start one server per
card, each pinned with CUDA_VISIBLE_DEVICES:

    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$i vllm serve Qwen/Qwen3-VL-4B-Instruct \
        --port 800$((i+1)) --max-model-len 8192 &
    done

(One model replica per card beats one tensor-parallel-4 replica here: the work
is thousands of tiny independent image calls, which is throughput-bound, not
latency-bound, and TP-4 would add cross-GPU sync to every one of them.)

Then:

    python availability_audit/screen_garment_pattern.py \
        --knn-url http://127.0.0.1:1235/knn-service \
        --vlm-urls http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1,\
http://127.0.0.1:8003/v1,http://127.0.0.1:8004/v1 \
        --vlm-model Qwen/Qwen3-VL-4B-Instruct \
        --top-k 100 --per-gpu-workers 8

Candidates are round-robined across the endpoints and scored by a thread pool
of `--per-gpu-workers x n_endpoints` (default 32). Raise --per-gpu-workers
until GPU utilisation stops improving; lower it if the servers start returning
timeouts.

Three things keep the GPUs from being wasted:
  * images are keyed by URL and downloaded once, shared across all 120 cells;
  * (image, garment) and (image, pattern) scores are memoised, so the 6 pattern
    cells of one garment reuse each other's garment calls (see ScoreCache);
  * a candidate whose garment score is 0 skips the tile pass entirely, since
    the geometric mean is already 0.

    # resume after an interruption: just re-run, finished cells and cached
    # scores are skipped
    # debug on a few cells:
    ... --limit-cells 3
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import FASHION_ATTRIBUTE_AXES, OPTIONS_DIR
# Reuse the production retrieval + scoring path; do NOT write new scorers.
from retrieval.collect_topk_scored_gallery import (
    query_knn, download_image, call_vlm_score, garment_prompt, score_pattern,
    norm_attr,
)

DEFAULT_KNN_URL = os.environ.get("POD_KNN_URL", "http://127.0.0.1:1235/knn-service")
# One vLLM per GPU. Override with POD_VLM_URLS or --vlm-urls; the count of
# endpoints is what makes this use all four cards, nothing here talks to CUDA
# directly (the servers own the GPUs).
DEFAULT_VLM_URLS = os.environ.get(
    "POD_VLM_URLS",
    "http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1,"
    "http://127.0.0.1:8003/v1,http://127.0.0.1:8004/v1")
DEFAULT_VLM_MODEL = os.environ.get("POD_VLM_MODEL", "Qwen/Qwen3-VL-30B-A3B-Instruct")
DEFAULT_INDEX = os.environ.get("POD_KNN_INDEX", "fsiglip")


# ── preflight ─────────────────────────────────────────────────────────────
def preflight(knn_url: str, vlm_urls: list[str], vlm_model: str, index_name: str,
              timeout: float = 30.0) -> None:
    """Fail fast if any endpoint is wrong.

    A previous audit run (`build_candidate_bank.py`) recorded 8,265 rows of
    `error:KeyError` because it pointed at a host that was not a vLLM at all.
    Silent total failure is worse than no run, so both services are proven
    reachable AND serving the expected model before any cell is measured.

    With one vLLM per GPU, EVERY endpoint is checked: a dead 4th server would
    otherwise zero out a quarter of the candidates and quietly depress the whole
    table, which looks like "these combinations do not exist".
    """
    print("── preflight ──────────────────────────────────────────────")

    # 1. VLM: each /v1/models must list the model we are about to call
    for vlm_url in vlm_urls:
        models_url = vlm_url.rstrip("/") + "/models"
        try:
            r = requests.get(models_url, timeout=timeout)
        except Exception as e:
            raise SystemExit(f"  [abort] VLM unreachable at {models_url}: "
                             f"{type(e).__name__}: {str(e)[:200]}")
        if r.status_code != 200:
            raise SystemExit(f"  [abort] {models_url} returned HTTP {r.status_code}. "
                             f"Body[:200]: {r.text[:200]!r}\n"
                             f"          That host is not an OpenAI-compatible vLLM.")
        try:
            served = [m["id"] for m in r.json()["data"]]
        except Exception as e:
            raise SystemExit(f"  [abort] {models_url} is not an OpenAI /v1/models "
                             f"response ({type(e).__name__}). Body[:200]: {r.text[:200]!r}")
        if vlm_model not in served:
            raise SystemExit(f"  [abort] model mismatch at {models_url}. Requested "
                             f"{vlm_model!r}, server serves {served}. "
                             f"Pass --vlm-model with one of those.")
        print(f"  VLM  OK  {vlm_url}  serving {vlm_model}")

    # 2. KNN: a trivial query must return a list
    try:
        probe = query_knn(requests.Session(), knn_url, "black t shirt", 3,
                          index_name, timeout, 1)
    except Exception as e:
        raise SystemExit(f"  [abort] KNN service failed at {knn_url}: "
                         f"{type(e).__name__}: {str(e)[:200]}")
    if not probe:
        raise SystemExit(f"  [abort] KNN service at {knn_url} returned 0 results "
                         f"for a trivial probe query; index {index_name!r} may be wrong.")
    print(f"  KNN  OK  {knn_url}  index={index_name}  probe returned {len(probe)}")
    print()


# ── row order: benchmark usage, computed not hardcoded ────────────────────
def garment_usage_order(garments: list[str], plans_path: Path) -> tuple[list[str], dict[str, int]]:
    """Order garments by how often they appear in a NON-SOLID-pattern option.

    That is the population this screening is about: a solid garment is trivially
    retrievable, so the interesting question is how often we ask for a garment
    together with a real pattern. Garments never used that way sort last
    (alphabetically among themselves) rather than being dropped.
    """
    usage = {g: 0 for g in garments}
    if not plans_path.exists():
        print(f"  [warn] {plans_path} not found; falling back to config order")
        return list(garments), usage
    with open(plans_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for opt in (json.loads(line).get("options") or {}).values():
                a = opt.get("attributes") or {}
                g, p = a.get("garment_category"), a.get("pattern")
                if g in usage and p and p != "solid":
                    usage[g] += 1
    order = sorted(garments, key=lambda g: (-usage[g], g))
    return order, usage


# ── one cell ──────────────────────────────────────────────────────────────
def search_text(garment: str, pattern: str) -> str:
    """Human-readable query, same rendering the collectors use."""
    return f"{norm_attr(pattern)} {norm_attr(garment)}".strip()


def url_key(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:20]


# ── per-candidate raw score log ───────────────────────────────────────────
# Shared with the 3-D screening, which imports these rather than copying them.
#
# `nonzero` alone answers only "did ANY candidate clear the bar", which is a
# very loose test — one lucky tile is enough. Pre-counting a few stricter
# thresholds into every cell makes it possible to ask whether the ranking of
# cells survives a stricter bar without re-reading the raw log.
# `nonzero_at["0"]` reproduces `nonzero` exactly (score strictly > 0); every
# other threshold t counts score >= t.
NONZERO_THRESHOLDS = (0.0, 0.3, 0.5)
CANDIDATES_JSONL = "screen_gp_candidates.jsonl"


def nonzero_at(scores: list[float],
               thresholds: tuple[float, ...] = NONZERO_THRESHOLDS) -> dict[str, int]:
    return {("%g" % t): sum(1 for s in scores if (s > 0.0 if t <= 0.0 else s >= t))
            for t in thresholds}


def append_candidates(path: Path, recs: list[dict[str, Any]]) -> None:
    """Append one JSON line per candidate, once the whole cell is complete.

    Append-only, and flushed per cell rather than per candidate, so an
    interrupted run leaves whole cells behind and never a half-written one.
    Kept OUT of the summary json because that file is rewritten in full after
    every cell — putting ~100x the rows there would make the rewrite the
    dominant cost of the run.
    """
    if not recs:
        return
    with open(path, "a", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class ScoreCache:
    """Process-wide memo for (image, garment) and (image, pattern) scores.

    The 6 pattern cells of one garment retrieve heavily overlapping images, so
    the same (image, garment) pair recurs across cells; likewise the same image
    can come back for several patterns. Every hit is one VLM call saved, which
    is the dominant cost. Persisted so a resumed run keeps the savings.
    """

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.g: dict[str, float] = {}
        self.p: dict[str, float] = {}
        self.hits = self.misses = 0
        if path.exists():
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
                self.g, self.p = d.get("garment", {}), d.get("pattern", {})
            except Exception as e:
                print(f"  [warn] could not read score cache: {type(e).__name__}: {e}")

    def get(self, kind: str, key: str):
        with self.lock:
            v = (self.g if kind == "g" else self.p).get(key)
            if v is None:
                self.misses += 1
            else:
                self.hits += 1
            return v

    def put(self, kind: str, key: str, value: float) -> None:
        with self.lock:
            (self.g if kind == "g" else self.p)[key] = value

    def save(self) -> None:
        with self.lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"garment": self.g, "pattern": self.p}),
                           encoding="utf-8")
            tmp.replace(self.path)


def _score_one(i: int, item: dict, garment: str, pattern: str,
               args: argparse.Namespace, session: requests.Session,
               imgdir: Path,
               cache: ScoreCache) -> tuple[float, list[str], dict[str, Any] | None]:
    """Download + score ONE candidate. Runs on a worker thread.

    The VLM endpoint is picked round-robin by candidate index, so with four
    single-GPU vLLM servers the load spreads evenly without any coordination.

    The third return value is the raw per-axis record for the JSONL log: the
    garment and pattern scores kept SEPARATE, plus which axis stopped the
    evaluation. An axis skipped by the abort optimisation is None, not 0.0 —
    "never measured" and "measured as zero" must stay distinguishable.
    """
    errs: list[str] = []
    url = str(item.get("image_url") or item.get("url") or "")
    if not url:
        return 0.0, ["missing_url"], None
    vlm_url = args.vlm_urls[i % len(args.vlm_urls)]
    uk = url_key(url)
    rec: dict[str, Any] = {"rank": i, "url_key": uk, "g": None, "p": None,
                           "score": 0.0, "aborted_axis": None}

    # images are shared across cells: keyed by URL, downloaded at most once
    try:
        path, err = download_image(session, url, imgdir / uk, args.download_timeout,
                                   args.retries, args.max_image_bytes, False)
    except Exception as e:
        path, err = None, f"{type(e).__name__}: {str(e)[:200]}"
    if path is None:
        rec["aborted_axis"] = "download"
        return 0.0, [f"download[{i}]: {str(err)[:200]}"], rec

    gkey = f"{uk}|{garment}"
    g = cache.get("g", gkey)
    if g is None:
        g_score, _parsed, g_err = call_vlm_score(
            image_path=path, prompt=garment_prompt(norm_attr(garment)),
            vlm_url=vlm_url, model=args.vlm_model,
            timeout=args.score_timeout, retries=args.retries,
            max_tokens=args.vlm_max_tokens)
        if g_err:
            errs.append(f"garment[{i}] @{vlm_url}: {str(g_err)[:200]}")
        g = float(g_score or 0.0)
        cache.put("g", gkey, g)

    rec["g"] = round(g, 4)
    if g <= 0.0:
        # geometric mean is 0 regardless of the pattern score; skip the
        # (much more expensive) tile pass
        rec["aborted_axis"] = "g"
        return 0.0, errs, rec

    pkey = f"{uk}|{pattern}"
    p = cache.get("p", pkey)
    if p is None:
        pat = score_pattern(image_path=path, pattern=pattern,
                            rec_id=f"{uk}__{pattern}", args=args, vlm_url=vlm_url)
        if pat.get("error"):
            errs.append(f"pattern[{i}] @{vlm_url}: {str(pat['error'])[:200]}")
        p = float(pat.get("score") or 0.0)
        cache.put("p", pkey, p)

    rec["p"] = round(p, 4)
    score = math.sqrt(max(g, 0.0) * max(p, 0.0))
    rec["score"] = round(score, 4)
    if p <= 0.0:
        rec["aborted_axis"] = "p"
    return score, errs, rec


def measure_cell(garment: str, pattern: str, args: argparse.Namespace,
                 session: requests.Session, imgdir: Path,
                 cache: ScoreCache,
                 jsonl_path: Path | None = None) -> dict[str, Any]:
    text = search_text(garment, pattern)
    t0 = time.time()
    errors: list[str] = []

    try:
        results = query_knn(session, args.knn_url, text, args.top_k,
                            args.index_name, args.timeout, args.retries)
    except Exception as e:
        return {"nonzero": 0, "top1": 0.0, "median": 0.0, "n_candidates": 0,
                "nonzero_at": nonzero_at([]),
                "query": text, "seconds": round(time.time() - t0, 1),
                "errors": [f"knn: {type(e).__name__}: {str(e)[:200]}"]}

    scores = [0.0] * len(results)
    recs: list[dict[str, Any] | None] = [None] * len(results)
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_score_one, i, item, garment, pattern, args,
                          session, imgdir, cache): i
                for i, item in enumerate(results)}
        for fut in cf.as_completed(futs):
            i = futs[fut]
            try:
                s, errs, rec = fut.result()
            except Exception as e:
                s, errs, rec = (0.0,
                                [f"worker[{i}]: {type(e).__name__}: {str(e)[:200]}"],
                                None)
            scores[i] = s
            recs[i] = rec
            for e_ in errs:
                if len(errors) < 20:
                    errors.append(e_)

    if jsonl_path is not None:
        append_candidates(jsonl_path,
                          [{"garment": garment, "pattern": pattern, **r}
                           for r in recs if r is not None])

    ordered = sorted(scores, reverse=True)
    return {
        "nonzero": sum(1 for s in ordered if s > 0.0),
        "nonzero_at": nonzero_at(ordered),
        "top1": round(ordered[0], 4) if ordered else 0.0,
        "median": round(statistics.median(ordered), 4) if ordered else 0.0,
        "n_candidates": len(results),
        "query": text,
        "seconds": round(time.time() - t0, 1),
        "errors": errors,
    }


# ── outputs ───────────────────────────────────────────────────────────────
def print_table(table: dict, rows: list[str], cols: list[str],
                usage: dict[str, int]) -> None:
    w = max(len(g) for g in rows) + 1
    head = f"{'garment':<{w}}{'uses':>6}  " + "".join(f"{c[:9]:>10}" for c in cols)
    print("\n" + "=" * len(head))
    print("  (garment, pattern) availability — non-zero candidates out of top-k")
    print("=" * len(head))
    print(head)
    print("-" * len(head))
    for g in rows:
        line = f"{g:<{w}}{usage.get(g, 0):>6}  "
        for p in cols:
            cell = table.get(g, {}).get(p)
            line += f"{'-':>10}" if cell is None else f"{cell['nonzero']:>10}"
        print(line)
    print("=" * len(head))


def write_heatmap(table: dict, rows: list[str], cols: list[str], out_png: Path,
                  top_k: int) -> str:
    """Matplotlib if available, else a self-contained PIL renderer."""
    grid = [[(table.get(g, {}).get(p) or {}).get("nonzero", 0) for p in cols]
            for g in rows]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots(figsize=(1.05 * len(cols) + 3.2, 0.42 * len(rows) + 2.0))
        im = ax.imshow(np.array(grid, dtype=float), cmap="coolwarm",
                       vmin=0, vmax=top_k, aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([c.replace("_", " ") for c in cols], rotation=30, ha="right")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r.replace("_", " ") for r in rows])
        ax.set_xlabel("pattern")
        ax.set_ylabel("garment  (most-used at top)")
        ax.set_title(f"(garment, pattern) image availability\n"
                     f"non-zero candidates out of top-{top_k}")
        for i in range(len(rows)):
            for j in range(len(cols)):
                v = grid[i][j]
                ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                        color="white" if (v > 0.66 * top_k or v < 0.12 * top_k) else "black")
        cb = fig.colorbar(im, ax=ax)
        cb.set_label(f"# candidates with score > 0  (0 – {top_k})")
        fig.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        return "matplotlib"
    except ImportError:
        pass

    from PIL import Image, ImageDraw
    cw, ch, lpad, tpad, bar = 62, 26, 190, 92, 46
    W, H = lpad + cw * len(cols) + bar + 70, tpad + ch * len(rows) + 46
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    def color(v):
        t = 0.0 if top_k <= 0 else min(max(v / top_k, 0.0), 1.0)
        return (int(59 + t * (196 - 59)), int(76 + t * (60 - 76)), int(192 + t * (54 - 192)))

    d.text((lpad, 14), "(garment, pattern) image availability", fill="black")
    d.text((lpad, 30), f"non-zero candidates out of top-{top_k}", fill="black")
    for j, p in enumerate(cols):
        d.text((lpad + j * cw + 4, tpad - 18), p.replace("_", " ")[:9], fill="black")
    for i, g in enumerate(rows):
        y = tpad + i * ch
        d.text((6, y + 7), g.replace("_", " ")[:26], fill="black")
        for j in range(len(cols)):
            v = grid[i][j]
            x = lpad + j * cw
            d.rectangle([x, y, x + cw - 1, y + ch - 1], fill=color(v), outline="white")
            d.text((x + cw // 2 - 8, y + 7), str(v), fill="white" if v > top_k * 0.55 else "black")
    d.text((lpad, tpad + ch * len(rows) + 12), "pattern ->", fill="black")
    bx = lpad + cw * len(cols) + 24
    for k in range(200):
        v = top_k * (1 - k / 199)
        yy = tpad + int(k * (ch * len(rows)) / 200)
        d.rectangle([bx, yy, bx + bar - 12, yy + 3], fill=color(v))
    d.text((bx, tpad - 18), f"{top_k}", fill="black")
    d.text((bx, tpad + ch * len(rows) + 4), "0", fill="black")
    d.text((bx - 6, tpad + ch * len(rows) + 20), "score > 0", fill="black")
    img.save(out_png)
    return "PIL"


# ── main ──────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--limit-cells", type=int, default=0, help="debug: only N cells")
    p.add_argument("--knn-url", default=DEFAULT_KNN_URL)
    p.add_argument("--vlm-urls", default=DEFAULT_VLM_URLS,
                   help="comma-separated vLLM endpoints, ONE PER GPU. Candidates "
                        "are round-robined across them, e.g. "
                        "http://127.0.0.1:8001/v1,...:8002/v1,...:8003/v1,...:8004/v1")
    p.add_argument("--vlm-url", default=None,
                   help="deprecated single-endpoint alias for --vlm-urls")
    p.add_argument("--workers", type=int, default=0,
                   help="concurrent candidates in flight; 0 = auto "
                        "(--per-gpu-workers x number of endpoints)")
    p.add_argument("--per-gpu-workers", type=int, default=8,
                   help="in-flight requests per vLLM endpoint (auto workers)")
    p.add_argument("--vlm-model", default=DEFAULT_VLM_MODEL)
    p.add_argument("--index-name", default=DEFAULT_INDEX)
    p.add_argument("--plans", type=Path,
                   default=OPTIONS_DIR / "option_plans.jsonl",
                   help="used only to order rows by benchmark usage")
    p.add_argument("--force", action="store_true", help="ignore saved partial results")
    p.add_argument("--skip-preflight", action="store_true",
                   help="NOT recommended; the preflight exists to stop silent total failure")
    # scorer knobs — same names/defaults as collect_topk_scored_gallery.py
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--download-timeout", type=float, default=30.0)
    p.add_argument("--score-timeout", type=float, default=120.0)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--max-image-bytes", type=int, default=20_000_000)
    p.add_argument("--vlm-max-tokens", type=int, default=64)
    p.add_argument("--tile-grid", type=int, default=5)
    p.add_argument("--pattern-max-tiles", type=int, default=12)
    p.add_argument("--pattern-threshold", type=float, default=0.5)
    p.add_argument("--tile-white-thresh", type=float, default=242.0)
    p.add_argument("--tile-min-std", type=float, default=8.0)
    p.add_argument("--save-tiles", action="store_true")
    return p.parse_args()


# ── DEPRECATED: WRONG SCORER — DO NOT RUN ────────────────────────────────
# This script scores with retrieval/collect_topk_scored_gallery.py: no SAM3, a
# fixed grid over the whole image, colour from one global VLM call, and each
# axis asked in isolation against an absolute threshold. availability_audit is
# specified (README.md, Step 3) on the SAM3 patch scorer in
# fsiglip/collect_img_sam3.py, whose closed-vocabulary argmax is a different
# rule, not a tuning of this one. Every table this file produced is measured
# against the wrong scorer.
#
# It is kept only so its structure (usage ordering, resume, the raw candidate
# log) can be read by the replacement, and it refuses to run rather than burn
# GPU on numbers that cannot be used. Use:
#
#     python -m availability_audit.screen_sam3
#
# If you really need the old behaviour for a one-off comparison, pass
# --i-know-this-is-the-wrong-scorer.
DEPRECATED_MSG = """
  [STOP] {name} uses the pre-SAM3 scorer and its output is not valid
         for this audit. Use:  python -m availability_audit.screen_sam3
         (override: --i-know-this-is-the-wrong-scorer)
"""


def _refuse_unless_overridden(name: str) -> None:
    import sys as _sys
    flag = "--i-know-this-is-the-wrong-scorer"
    if flag in _sys.argv:
        _sys.argv.remove(flag)
        print(f"  [warn] running {name} with the WRONG scorer, as requested")
        return
    raise SystemExit(DEPRECATED_MSG.format(name=name))


def main() -> int:
    _refuse_unless_overridden("screen_garment_pattern.py")
    args = parse_args()

    # ── endpoint fan-out: one vLLM per GPU ────────────────────────────────
    if args.vlm_url:      # deprecated alias wins only if --vlm-urls untouched
        if args.vlm_urls == DEFAULT_VLM_URLS:
            args.vlm_urls = args.vlm_url
        else:
            raise SystemExit("  [error] pass either --vlm-urls or --vlm-url, not both")
    args.vlm_urls = [u.strip() for u in str(args.vlm_urls).split(",") if u.strip()]
    if not args.vlm_urls:
        raise SystemExit("  [error] --vlm-urls is empty")
    if args.workers <= 0:
        args.workers = max(1, args.per_gpu_workers * len(args.vlm_urls))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "screen_gp.json"
    out_png = args.out_dir / "screen_gp.png"
    jsonl_path = args.out_dir / CANDIDATES_JSONL
    # score_pattern writes tiles under output_root; keep them out of the repo
    args.output_root = Path(tempfile.gettempdir()) / "pod_screen_gp"
    args.output_root.mkdir(parents=True, exist_ok=True)
    # images are keyed by URL and shared across all cells, so an image retrieved
    # for several (garment, pattern) queries is downloaded once
    imgdir = args.output_root / "img"
    imgdir.mkdir(parents=True, exist_ok=True)

    # vocabulary straight from config — never hardcoded, so the grid follows it
    garments = list(FASHION_ATTRIBUTE_AXES["garment_category"])
    patterns = list(FASHION_ATTRIBUTE_AXES["pattern"])
    rows, usage = garment_usage_order(garments, args.plans)

    print("=" * 70)
    print(f"  garment x pattern availability screening "
          f"({len(rows)} x {len(patterns)} = {len(rows) * len(patterns)} cells)")
    print(f"  top-k per cell: {args.top_k}   model: {args.vlm_model}")
    print(f"  vLLM endpoints: {len(args.vlm_urls)} "
          f"({', '.join(args.vlm_urls)})")
    print(f"  workers: {args.workers} in flight "
          f"(~{args.workers / len(args.vlm_urls):.1f} per endpoint)")
    print("=" * 70)

    if not args.skip_preflight:
        preflight(args.knn_url, args.vlm_urls, args.vlm_model, args.index_name,
                  args.timeout)

    table: dict[str, dict[str, Any]] = {}
    if out_json.exists() and not args.force:
        table = json.loads(out_json.read_text(encoding="utf-8"))
        done = sum(len(v) for v in table.values())
        if done:
            print(f"  resuming: {done} cells already measured in {out_json}\n")
        # cells saved before the raw log existed have no JSONL lines and, being
        # "done", would never get any — say so rather than leave a silent hole
        stale = sum(1 for gg in table.values() for cell in gg.values()
                    if "nonzero_at" not in cell)
        if stale:
            print(f"  [warn] {stale} saved cells predate the per-candidate log; "
                  f"they have no lines in {jsonl_path.name}.\n"
                  f"         Re-run with --force to log them.\n")
    elif args.force and jsonl_path.exists():
        jsonl_path.unlink()   # --force re-measures everything; don't double-count

    cells = [(g, p) for g in rows for p in patterns]
    todo = [(g, p) for g, p in cells if p not in table.get(g, {})]
    if args.limit_cells > 0:
        todo = todo[:args.limit_cells]

    session = requests.Session()
    # one pooled connection per in-flight request, else the workers serialize on
    # the default 10-connection pool and the GPUs sit idle
    adapter = requests.adapters.HTTPAdapter(pool_connections=args.workers * 2,
                                            pool_maxsize=args.workers * 2,
                                            max_retries=0)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    cache = ScoreCache(args.out_dir / "screen_gp.scorecache.json")
    t_start = time.time()
    for n, (g, p) in enumerate(todo, 1):
        print(f"  [{n:3d}/{len(todo)}] {g:16s} x {p:10s} …", end="", flush=True)
        cell = measure_cell(g, p, args, session, imgdir, cache, jsonl_path)
        table.setdefault(g, {})[p] = cell
        print(f" nonzero={cell['nonzero']:3d}/{cell['n_candidates']:<3d} "
              f"top1={cell['top1']:.2f} ({cell['seconds']}s)"
              + (f"  [{len(cell['errors'])} errors]" if cell["errors"] else ""))
        # save after EVERY cell so an interrupted run resumes here
        out_json.write_text(json.dumps(table, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        cache.save()

    elapsed = time.time() - t_start
    total_lookups = cache.hits + cache.misses
    print(f"\n  measured {len(todo)} cells in {elapsed:.0f}s"
          f" ({elapsed / max(len(todo), 1):.1f}s/cell)")
    print(f"  score cache: {cache.hits} hits / {total_lookups} lookups "
          f"({100 * cache.hits / max(total_lookups, 1):.0f}% VLM calls avoided)")
    print(f"  -> {out_json}")
    if jsonl_path.exists():
        n_lines = sum(1 for _ in open(jsonl_path, encoding="utf-8"))
        print(f"  -> {jsonl_path} ({n_lines} candidate rows, "
              f"{jsonl_path.stat().st_size / 1e6:.1f} MB)")

    print_table(table, rows, patterns, usage)
    backend = write_heatmap(table, rows, patterns, out_png, args.top_k)
    print(f"\n  heatmap ({backend}) -> {out_png}")

    flat = [(g, p, c["nonzero"]) for g in table for p, c in table[g].items()]
    if flat:
        flat.sort(key=lambda x: x[2])
        print("\n  lowest 10 cells:")
        for g, p, v in flat[:10]:
            print(f"    {v:3d}  {p} {g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
