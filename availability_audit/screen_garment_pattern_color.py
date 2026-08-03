#!/usr/bin/env python3
"""Same screening as screen_garment_pattern.py, once per COLOUR.

WHAT THIS ADDS
--------------
`screen_garment_pattern.py` measures a 20 x 6 (garment x pattern) grid with the
colour axis left out. This script runs that same grid once for every colour in
`configs.config.FASHION_ATTRIBUTE_AXES["color"]`, i.e. a
13 x 20 x 6 = 1,560-cell sweep, and emits one table + one heatmap per colour
from a single run:

    screen_gpc.json                      every cell (color -> garment -> pattern)
    screen_gpc_candidates.jsonl          one line per CANDIDATE: the g/c/p axis
                                         scores kept separate, appended per cell
                                         (see availability_audit/analyze_screen.py)
    screen_gpc_black.png                 one heatmap per colour
    screen_gpc_yellow.png                ... 13 of them
    screen_gpc_summary.png               colour x garment, averaged over patterns

    images/black/leopard formal shirt/   the --save-top (default 5) best-scoring
        1__score_0.62__<urlkey>.jpg      images for that cell, rank and score in
        2__score_0.55__<urlkey>.jpg      the filename
    images/yellow/floral puffer jacket/  ...

Only candidates scoring > 0 are saved, so a sparse folder is itself the finding:
it means the retriever could not produce that combination. The images are the
qualitative check on a number that is only an upper bound — look at them before
trusting any cell.

Structure, CLI, resume behaviour, preflight and the 4-GPU fan-out are inherited
from screen_garment_pattern.py — this module imports them rather than copying,
so the two screenings cannot drift apart.

WHAT "SCORE > 0" MEANS HERE  (differs from the 2-D version — read this)
----------------------------------------------------------------------
Three scorers now run per candidate, all imported from
`retrieval/collect_topk_scored_gallery.py`, unchanged:

  garment_score : global Qwen-VL call, `garment_prompt(garment)`  -> {"score":0..1}
  color_score   : global Qwen-VL call, `color_prompt(color)`      -> {"score":0..1}
  pattern_score : `solid_prompt()` when pattern == "solid", else `score_pattern()`
                  patch coverage =
                  (# tiles scoring >= --pattern-threshold) / (# tiles scored)

  cell score    = (garment_score * color_score * pattern_score) ** (1/3)

Being a geometric mean, `nonzero` counts candidates where ALL THREE are > 0.
This is strictly stricter than the 2-D screening, so cells here are <= the
corresponding cell in screen_gp.json. Do not compare the two tables directly.

Retrieval text is `"{pattern} {color} {garment}"` (e.g. `checkered navy puffer
jacket`), the same rendering order `option_to_text()` uses in the collectors.
`solid` is dropped from the query string exactly as the collectors drop it,
because "solid" is not a useful retrieval term — but it is still SCORED, via
`solid_prompt()`.

STILL AN UPPER BOUND
--------------------
A non-zero score does not certify the image really is that garment, in that
colour, with that pattern: one passing tile is enough, the VLM is uncalibrated,
and nobody has checked the output by hand. Use near-zero cells to rule
combinations OUT; never read a high cell as "verified". Captions are never used
— scoring is image-only.

RUNNING IT ON 4 GPUs
--------------------
Identical to the 2-D script: one vLLM per card, endpoints round-robined.

    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$i vllm serve Qwen/Qwen3-VL-4B-Instruct \\
        --port 800$((i+1)) --max-model-len 8192 &
    done

    python availability_audit/screen_garment_pattern_color.py \
        --knn-url http://127.0.0.1:1235/knn-service \
        --vlm-urls http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1,\
http://127.0.0.1:8003/v1,http://127.0.0.1:8004/v1 \
        --vlm-model Qwen/Qwen3-VL-4B-Instruct \
        --top-k 100 --per-gpu-workers 8

COST WARNING — read before launching
------------------------------------
This is 13x the work of the 2-D screening. Budget accordingly, and use the
cheap-out flags rather than discovering the runtime the hard way:

    --colors black,navy,red     only these colours (default: all 13)
    --top-k 40                  fewer candidates per cell
    --skip-solid                drop the solid column (it is the easy case)

Three savings are built in and need no flags:
  * one image per URL, downloaded once and shared across every colour/cell;
  * (image, garment) / (image, colour) / (image, pattern) scores are memoised,
    so the 13 colours of one (garment, pattern) reuse each other's garment and
    pattern calls — expect a high cache hit rate;
  * a candidate is abandoned as soon as any axis scores 0, cheapest axis first
    (garment -> colour -> pattern tiles), because the geometric mean is then 0.

Resume: every cell is written to screen_gpc.json as it finishes, and the score
cache persists, so re-running continues where it stopped.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import sys
import tempfile
import time
import concurrent.futures as cf
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import FASHION_ATTRIBUTE_AXES, OPTIONS_DIR
# production retrieval + scoring path; NO new scorers are defined here
from retrieval.collect_topk_scored_gallery import (
    query_knn, download_image, call_vlm_score, garment_prompt, color_prompt,
    score_pattern, norm_attr,
)
# structure shared with the 2-D screening; imported, never copied
from availability_audit.screen_garment_pattern import (
    preflight, garment_usage_order, ScoreCache, url_key,
    nonzero_at, append_candidates, NONZERO_THRESHOLDS,
    DEFAULT_KNN_URL, DEFAULT_VLM_URLS, DEFAULT_VLM_MODEL, DEFAULT_INDEX,
)


# ── cache with a third axis ───────────────────────────────────────────────
class ColorScoreCache(ScoreCache):
    """ScoreCache plus a separate colour dict.

    Subclassed rather than editing the 2-D script: its cache knows only the
    garment and pattern kinds, and routing colour scores into the pattern dict
    would work only by the accident that no colour name equals a pattern name.
    The on-disk file stays compatible — the 2-D script reads the "garment" and
    "pattern" sections and simply ignores "color".
    """

    def __init__(self, path: Path):
        super().__init__(path)
        self.c: dict[str, float] = {}
        if path.exists():
            try:
                self.c = json.loads(path.read_text(encoding="utf-8")).get("color", {})
            except Exception:
                pass

    def _bucket(self, kind: str) -> dict[str, float]:
        return {"g": self.g, "p": self.p, "c": self.c}[kind]

    def get(self, kind: str, key: str):
        with self.lock:
            v = self._bucket(kind).get(key)
            if v is None:
                self.misses += 1
            else:
                self.hits += 1
            return v

    def put(self, kind: str, key: str, value: float) -> None:
        with self.lock:
            self._bucket(kind)[key] = value

    def save(self) -> None:
        with self.lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"garment": self.g, "pattern": self.p,
                                       "color": self.c}), encoding="utf-8")
            tmp.replace(self.path)


# ── ordering ──────────────────────────────────────────────────────────────
def color_usage_order(colors: list[str], plans_path: Path) -> tuple[list[str], dict[str, int]]:
    """Order colours by how often the benchmark actually asks for them.

    Same idea as garment_usage_order: the colours the dataset leans on are read
    first. Note this counts colours as they appear in OPTIONS, which is not the
    same as profile preferences — the RESERVE colours (orange/yellow/pink/white)
    never appear in any profile's likes/dislikes, yet they are used heavily in
    options precisely because being preference-neutral for every user makes them
    ideal fixed/violation values. So they do NOT sort last here.
    """
    usage = {c: 0 for c in colors}
    if not plans_path.exists():
        print(f"  [warn] {plans_path} not found; falling back to config order")
        return list(colors), usage
    with open(plans_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for opt in (json.loads(line).get("options") or {}).values():
                c = (opt.get("attributes") or {}).get("color")
                if c in usage:
                    usage[c] += 1
    return sorted(colors, key=lambda c: (-usage[c], c)), usage


# the 3-D log gets its own file; the row format and the threshold counting
# are the 2-D module's, imported above so the two logs cannot drift apart
CANDIDATES_JSONL = "screen_gpc_candidates.jsonl"


def search_text_c(garment: str, pattern: str, color: str) -> str:
    """`"{pattern} {color} {garment}"`, dropping `solid` like the collectors do."""
    parts = []
    if pattern and pattern != "solid":
        parts.append(norm_attr(pattern))
    if color:
        parts.append(norm_attr(color))
    parts.append(norm_attr(garment))
    return " ".join(p for p in parts if p).strip()


# ── saving the best images per cell ───────────────────────────────────────
def save_top_images(garment: str, pattern: str, color: str,
                    scores: list[float], paths: list, results: list[dict],
                    args: argparse.Namespace) -> list[dict[str, Any]]:
    """Copy the best-scoring candidates into <out-dir>/images/<color>/<combo>/.

    Layout, as requested:

        images/black/leopard formal shirt/1__score_0.62__<urlkey>.jpg
        images/black/leopard formal shirt/2__score_0.55__<urlkey>.jpg

    The rank and the score are in the FILENAME so a folder can be eyeballed
    without opening the json — these images are the qualitative check on a
    number that is only an upper bound, so it must be obvious which image
    earned which score.

    Only candidates that scored > 0 are saved: a zero-scoring image is not
    evidence of the combination existing, and saving it would suggest it is.
    So a near-empty folder is a real signal, not a bug. Files are COPIED from
    the shared URL-keyed download cache, which stays intact for other cells.
    """
    if args.save_top <= 0:
        return []
    ranked = sorted(((s, p, i) for i, (s, p) in enumerate(zip(scores, paths))
                     if s > 0.0 and p is not None),
                    key=lambda t: -t[0])[:args.save_top]
    if not ranked:
        return []

    combo = f"{norm_attr(pattern)} {norm_attr(garment)}".strip()
    dest = args.out_dir / "images" / norm_attr(color) / combo
    dest.mkdir(parents=True, exist_ok=True)
    saved = []
    for rank, (s, src, i) in enumerate(ranked, 1):
        src = Path(src)
        url = str(results[i].get("image_url") or results[i].get("url") or "")
        name = f"{rank}__score_{s:.3f}__{url_key(url)}{src.suffix or '.jpg'}"
        out = dest / name
        try:
            shutil.copyfile(src, out)
            saved.append({"rank": rank, "score": round(s, 4),
                          "file": str(out.relative_to(args.out_dir)), "url": url})
        except Exception as e:
            print(f"    [warn] could not save {out.name}: {type(e).__name__}: {e}")
    return saved


# ── one candidate ─────────────────────────────────────────────────────────
def _score_one_c(i: int, item: dict, garment: str, pattern: str, color: str,
                 args: argparse.Namespace, session: requests.Session,
                 imgdir: Path, cache: ScoreCache) -> tuple[float, list[str], Path | None,
                                                           dict[str, Any] | None]:
    """Download + score ONE candidate on three axes. Runs on a worker thread.

    Axes are evaluated cheapest-first and abandoned at the first zero: garment
    (1 call), then colour (1 call), then pattern (up to --pattern-max-tiles
    calls). The geometric mean is 0 as soon as any axis is 0, so the remaining
    calls cannot change the result.

    Also returns the raw per-axis scores for the JSONL log. The three axes are
    kept SEPARATE — a geometric mean of 0 says nothing about which axis failed,
    which is exactly the question the log exists to answer. Axes skipped by the
    abort optimisation carry the value None, distinct from a measured 0.0;
    conflating the two would make any per-axis threshold analysis wrong,
    because "never looked" would be counted as evidence of absence.
    """
    errs: list[str] = []
    url = str(item.get("image_url") or item.get("url") or "")
    if not url:
        return 0.0, ["missing_url"], None, None
    vlm_url = args.vlm_urls[i % len(args.vlm_urls)]
    uk = url_key(url)
    rec: dict[str, Any] = {"rank": i, "url_key": uk,
                           "g": None, "c": None, "p": None,
                           "score": 0.0, "aborted_axis": None}

    try:
        path, err = download_image(session, url, imgdir / uk, args.download_timeout,
                                   args.retries, args.max_image_bytes, False)
    except Exception as e:
        path, err = None, f"{type(e).__name__}: {str(e)[:200]}"
    if path is None:
        rec["aborted_axis"] = "download"
        return 0.0, [f"download[{i}]: {str(err)[:200]}"], None, rec

    def cached_global(kind: str, key: str, prompt: str, label: str):
        v = cache.get(kind, key)
        if v is not None:
            return v
        score, _parsed, e_ = call_vlm_score(
            image_path=path, prompt=prompt, vlm_url=vlm_url, model=args.vlm_model,
            timeout=args.score_timeout, retries=args.retries,
            max_tokens=args.vlm_max_tokens)
        if e_:
            errs.append(f"{label}[{i}] @{vlm_url}: {str(e_)[:200]}")
        v = float(score or 0.0)
        cache.put(kind, key, v)
        return v

    g = cached_global("g", f"{uk}|{garment}", garment_prompt(norm_attr(garment)), "garment")
    rec["g"] = round(g, 4)
    if g <= 0.0:
        rec["aborted_axis"] = "g"
        return 0.0, errs, path, rec

    c = cached_global("c", f"{uk}|{color}", color_prompt(norm_attr(color)), "color")
    rec["c"] = round(c, 4)
    if c <= 0.0:
        rec["aborted_axis"] = "c"
        return 0.0, errs, path, rec

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

    score = (max(g, 0.0) * max(c, 0.0) * max(p, 0.0)) ** (1.0 / 3.0)
    rec["score"] = round(score, 4)
    if p <= 0.0:
        rec["aborted_axis"] = "p"
    return score, errs, path, rec


def measure_cell_c(garment: str, pattern: str, color: str,
                   args: argparse.Namespace, session: requests.Session,
                   imgdir: Path, cache: ScoreCache,
                   jsonl_path: Path | None = None) -> dict[str, Any]:
    text = search_text_c(garment, pattern, color)
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
    paths: list[Path | None] = [None] * len(results)
    recs: list[dict[str, Any] | None] = [None] * len(results)
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_score_one_c, i, item, garment, pattern, color, args,
                          session, imgdir, cache): i
                for i, item in enumerate(results)}
        for fut in cf.as_completed(futs):
            i = futs[fut]
            try:
                s, errs, path, rec = fut.result()
            except Exception as e:
                s, errs, path, rec = (0.0,
                                      [f"worker[{i}]: {type(e).__name__}: {str(e)[:200]}"],
                                      None, None)
            scores[i] = s
            paths[i] = path
            recs[i] = rec
            for e_ in errs:
                if len(errors) < 20:
                    errors.append(e_)

    if jsonl_path is not None:
        append_candidates(jsonl_path,
                          [{"color": color, "garment": garment, "pattern": pattern, **r}
                           for r in recs if r is not None])

    saved = save_top_images(garment, pattern, color, scores, paths, results, args)
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
        "saved_images": saved,
    }


# ── reporting ─────────────────────────────────────────────────────────────
def print_color_table(color: str, sub: dict, rows: list[str], cols: list[str],
                      usage: dict[str, int], top_k: int) -> None:
    w = max(len(g) for g in rows) + 1
    head = f"{'garment':<{w}}{'uses':>6}  " + "".join(f"{c[:9]:>10}" for c in cols)
    print("\n" + "=" * len(head))
    print(f"  COLOR = {color.upper()}   (non-zero candidates out of top-{top_k})")
    print("=" * len(head))
    print(head)
    print("-" * len(head))
    for g in rows:
        line = f"{g:<{w}}{usage.get(g, 0):>6}  "
        for p in cols:
            cell = sub.get(g, {}).get(p)
            line += f"{'-':>10}" if cell is None else f"{cell['nonzero']:>10}"
        print(line)
    print("=" * len(head))


def print_color_summary(table: dict, colors: list[str], rows: list[str],
                        cols: list[str], cusage: dict[str, int], top_k: int) -> None:
    """One line per colour: mean cell value, and how many cells are effectively 0."""
    head = (f"{'color':<10}{'uses':>7}{'cells':>7}{'mean':>8}{'median':>8}"
            f"{'zero':>7}{'<3':>6}{'best cell':>28}")
    print("\n" + "=" * len(head))
    print(f"  SUMMARY ACROSS COLORS  (cell = non-zero candidates of top-{top_k})")
    print("=" * len(head))
    print(head)
    print("-" * len(head))
    for c in colors:
        vals, best = [], (None, None, -1)
        for g in rows:
            for p in cols:
                cell = (table.get(c, {}).get(g, {}) or {}).get(p)
                if cell is None:
                    continue
                v = cell["nonzero"]
                vals.append(v)
                if v > best[2]:
                    best = (g, p, v)
        if not vals:
            print(f"{c:<10}{cusage.get(c, 0):>7}{0:>7}{'-':>8}{'-':>8}{'-':>7}{'-':>6}")
            continue
        z = sum(1 for v in vals if v == 0)
        lo = sum(1 for v in vals if v < 3)
        label = f"{best[1]} {best[0]} ({best[2]})" if best[0] else "-"
        print(f"{c:<10}{cusage.get(c, 0):>7}{len(vals):>7}"
              f"{statistics.mean(vals):>8.1f}{statistics.median(vals):>8.1f}"
              f"{z:>7}{lo:>6}{label:>28}")
    print("=" * len(head))


def write_labeled_heatmap(grid: list[list[int]], rows: list[str], cols: list[str],
                          out_png: Path, top_k: int, title: str,
                          xlabel: str, ylabel: str) -> str:
    """Heatmap with a caller-supplied title and axis labels.

    `write_heatmap` in the 2-D script hardcodes its title and an x-axis of
    "pattern", which would mislabel all 13 per-colour figures (no colour name)
    and the summary figure (x axis is colour, not pattern). These are appendix
    figures, so a wrong axis label is not acceptable; this renderer takes the
    labels as arguments. Colour mapping and the matplotlib/PIL fallback follow
    the same scheme so the two scripts' figures look alike.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots(figsize=(1.05 * len(cols) + 3.2, 0.42 * len(rows) + 2.2))
        im = ax.imshow(np.array(grid, dtype=float), cmap="coolwarm",
                       vmin=0, vmax=top_k, aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([c.replace("_", " ") for c in cols], rotation=30, ha="right")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r.replace("_", " ") for r in rows])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
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
    cw, ch, lpad, tpad, bar = 62, 26, 190, 96, 46
    W, H = lpad + cw * len(cols) + bar + 70, tpad + ch * len(rows) + 56
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    def color_of(v):
        t = 0.0 if top_k <= 0 else min(max(v / top_k, 0.0), 1.0)
        return (int(59 + t * (196 - 59)), int(76 + t * (60 - 76)), int(192 + t * (54 - 192)))

    d.text((lpad, 14), title, fill="black")
    d.text((lpad, 30), f"non-zero candidates out of top-{top_k}", fill="black")
    d.text((6, 52), ylabel, fill="black")
    for j, c in enumerate(cols):
        d.text((lpad + j * cw + 4, tpad - 18), c.replace("_", " ")[:9], fill="black")
    for i, r in enumerate(rows):
        y = tpad + i * ch
        d.text((6, y + 7), r.replace("_", " ")[:26], fill="black")
        for j in range(len(cols)):
            v = grid[i][j]
            x = lpad + j * cw
            d.rectangle([x, y, x + cw - 1, y + ch - 1], fill=color_of(v), outline="white")
            d.text((x + cw // 2 - 8, y + 7), str(v),
                   fill="white" if v > top_k * 0.55 else "black")
    d.text((lpad, tpad + ch * len(rows) + 14), f"{xlabel} ->", fill="black")
    bx = lpad + cw * len(cols) + 24
    for k in range(200):
        yy = tpad + int(k * (ch * len(rows)) / 200)
        d.rectangle([bx, yy, bx + bar - 12, yy + 3], fill=color_of(top_k * (1 - k / 199)))
    d.text((bx, tpad - 18), f"{top_k}", fill="black")
    d.text((bx, tpad + ch * len(rows) + 4), "0", fill="black")
    d.text((bx - 6, tpad + ch * len(rows) + 20), "score > 0", fill="black")
    img.save(out_png)
    return "PIL"


def write_color_png(color: str, sub: dict, rows: list[str], cols: list[str],
                    out_png: Path, top_k: int) -> str:
    grid = [[(sub.get(g, {}).get(p) or {}).get("nonzero", 0) for p in cols]
            for g in rows]
    return write_labeled_heatmap(
        grid, rows, cols, out_png, top_k,
        title=f"COLOR = {color}  |  (garment, pattern) image availability",
        xlabel="pattern", ylabel="garment  (most-used at top)")


def write_color_summary_png(table: dict, colors: list[str], rows: list[str],
                            cols: list[str], out_png: Path, top_k: int) -> str:
    """colour x garment heatmap, averaged over patterns — the one-page overview."""
    grid = []
    for g in rows:
        line = []
        for c in colors:
            vals = [(table.get(c, {}).get(g, {}) or {}).get(p, {}).get("nonzero")
                    for p in cols]
            vals = [v for v in vals if v is not None]
            line.append(round(statistics.mean(vals)) if vals else 0)
        grid.append(line)
    return write_labeled_heatmap(
        grid, rows, colors, out_png, top_k,
        title="Availability by colour and garment (mean over patterns)",
        xlabel="color", ylabel="garment  (most-used at top)")


# ── main ──────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--colors", default="",
                   help="comma-separated subset, e.g. black,navy,red (default: all)")
    p.add_argument("--save-top", type=int, default=5,
                   help="save this many best-scoring images per cell under "
                        "<out-dir>/images/<color>/<pattern> <garment>/ (0 = none)")
    p.add_argument("--skip-solid", action="store_true",
                   help="drop the solid pattern column (13x20 fewer cells)")
    p.add_argument("--limit-cells", type=int, default=0, help="debug: only N cells")
    p.add_argument("--knn-url", default=DEFAULT_KNN_URL)
    p.add_argument("--vlm-urls", default=DEFAULT_VLM_URLS,
                   help="comma-separated vLLM endpoints, ONE PER GPU")
    p.add_argument("--vlm-url", default=None,
                   help="deprecated single-endpoint alias for --vlm-urls")
    p.add_argument("--workers", type=int, default=0,
                   help="concurrent candidates in flight; 0 = auto")
    p.add_argument("--per-gpu-workers", type=int, default=8)
    p.add_argument("--vlm-model", default=DEFAULT_VLM_MODEL)
    p.add_argument("--index-name", default=DEFAULT_INDEX)
    p.add_argument("--plans", type=Path, default=OPTIONS_DIR / "option_plans.jsonl",
                   help="used only to order rows/colours by benchmark usage")
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
    _refuse_unless_overridden("screen_garment_pattern_color.py")
    args = parse_args()

    if args.vlm_url:
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
    out_json = args.out_dir / "screen_gpc.json"
    jsonl_path = args.out_dir / CANDIDATES_JSONL
    # tiles + images live outside the repo; images are shared across all colours
    args.output_root = Path(tempfile.gettempdir()) / "pod_screen_gp"
    imgdir = args.output_root / "img"
    imgdir.mkdir(parents=True, exist_ok=True)

    # vocabulary from config — the sweep follows the canonical axes, not a list
    garments = list(FASHION_ATTRIBUTE_AXES["garment_category"])
    patterns = list(FASHION_ATTRIBUTE_AXES["pattern"])
    all_colors = list(FASHION_ATTRIBUTE_AXES["color"])
    if args.skip_solid:
        patterns = [p for p in patterns if p != "solid"]
    rows, gusage = garment_usage_order(garments, args.plans)
    colors, cusage = color_usage_order(all_colors, args.plans)
    if args.colors:
        want = [c.strip() for c in args.colors.split(",") if c.strip()]
        bad = [c for c in want if c not in all_colors]
        if bad:
            raise SystemExit(f"  [error] not in the colour vocabulary: {bad}\n"
                             f"          available: {all_colors}")
        colors = [c for c in colors if c in want]

    n_cells = len(colors) * len(rows) * len(patterns)
    print("=" * 74)
    print(f"  garment x pattern x COLOR availability screening")
    print(f"  {len(colors)} colors x {len(rows)} garments x {len(patterns)} patterns "
          f"= {n_cells} cells")
    print(f"  top-k per cell: {args.top_k}   model: {args.vlm_model}")
    print(f"  vLLM endpoints: {len(args.vlm_urls)} ({', '.join(args.vlm_urls)})")
    print(f"  workers: {args.workers} in flight "
          f"(~{args.workers / len(args.vlm_urls):.1f} per endpoint)")
    print("=" * 74)

    if not args.skip_preflight:
        preflight(args.knn_url, args.vlm_urls, args.vlm_model, args.index_name,
                  args.timeout)

    table: dict[str, dict[str, dict[str, Any]]] = {}
    if out_json.exists() and not args.force:
        table = json.loads(out_json.read_text(encoding="utf-8"))
        done = sum(len(pp) for cc in table.values() for pp in cc.values())
        if done:
            print(f"  resuming: {done} cells already measured in {out_json}\n")
        # Cells measured before the raw log existed have no line in the JSONL
        # and, being "done", would never get one. Say so loudly: a silently
        # partial JSONL would make the offline analysis wrong rather than empty.
        stale = sum(1 for cc in table.values() for gg in cc.values()
                    for cell in gg.values() if "nonzero_at" not in cell)
        if stale:
            print(f"  [warn] {stale} saved cells predate the per-candidate log; "
                  f"they have no lines in {jsonl_path.name} and will be skipped "
                  f"as already done.\n         Re-run with --force to log them.\n")
    elif args.force and jsonl_path.exists():
        # --force re-measures every cell; keeping the old lines would double-count
        jsonl_path.unlink()

    todo = [(c, g, p) for c in colors for g in rows for p in patterns
            if p not in table.get(c, {}).get(g, {})]
    if args.limit_cells > 0:
        todo = todo[:args.limit_cells]

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=args.workers * 2,
                                            pool_maxsize=args.workers * 2,
                                            max_retries=0)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # shared with the 2-D screening on purpose: identical (image, garment) and
    # (image, pattern) keys, so whichever script runs first warms the other
    cache = ColorScoreCache(args.out_dir / "screen_gp.scorecache.json")
    t_start = time.time()
    for n, (c, g, p) in enumerate(todo, 1):
        print(f"  [{n:4d}/{len(todo)}] {c:8s} | {g:16s} x {p:10s} …",
              end="", flush=True)
        cell = measure_cell_c(g, p, c, args, session, imgdir, cache, jsonl_path)
        table.setdefault(c, {}).setdefault(g, {})[p] = cell
        print(f" nonzero={cell['nonzero']:3d}/{cell['n_candidates']:<3d} "
              f"top1={cell['top1']:.2f} ({cell['seconds']}s)"
              + (f"  [{len(cell['errors'])} errors]" if cell["errors"] else ""))
        out_json.write_text(json.dumps(table, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        cache.save()

    elapsed = time.time() - t_start
    looks = cache.hits + cache.misses
    print(f"\n  measured {len(todo)} cells in {elapsed:.0f}s "
          f"({elapsed / max(len(todo), 1):.1f}s/cell)")
    print(f"  score cache: {cache.hits} hits / {looks} lookups "
          f"({100 * cache.hits / max(looks, 1):.0f}% VLM calls avoided)")
    print(f"  -> {out_json}")
    if jsonl_path.exists():
        n_lines = sum(1 for _ in open(jsonl_path, encoding="utf-8"))
        print(f"  -> {jsonl_path} ({n_lines} candidate rows, "
              f"{jsonl_path.stat().st_size / 1e6:.1f} MB)")

    # ── per-colour table + heatmap, then the cross-colour summary ─────────
    written = []
    for c in colors:
        sub = table.get(c, {})
        if not sub:
            continue
        print_color_table(c, sub, rows, patterns, gusage, args.top_k)
        png = args.out_dir / f"screen_gpc_{c}.png"
        backend = write_color_png(c, sub, rows, patterns, png, args.top_k)
        written.append(png)
        print(f"  heatmap ({backend}) -> {png}")

    print_color_summary(table, colors, rows, patterns, cusage, args.top_k)
    summary_png = args.out_dir / "screen_gpc_summary.png"
    backend = write_color_summary_png(table, colors, rows, patterns,
                                      summary_png, args.top_k)
    print(f"\n  summary heatmap, colour x garment averaged over patterns "
          f"({backend}) -> {summary_png}")
    print(f"  {len(written)} per-colour heatmaps written to {args.out_dir}")
    if args.save_top > 0:
        imgroot = args.out_dir / "images"
        n_files = sum(1 for _ in imgroot.rglob("*")) if imgroot.exists() else 0
        n_dirs = sum(1 for d in imgroot.rglob("*") if d.is_dir()) if imgroot.exists() else 0
        print(f"  top-{args.save_top} images: {n_files - n_dirs} files in "
              f"{n_dirs} folders under {imgroot}")

    flat = [(c, g, p, cell["nonzero"])
            for c in table for g in table[c] for p, cell in table[c][g].items()]
    if flat:
        flat.sort(key=lambda x: x[3])
        print("\n  lowest 15 cells overall:")
        for c, g, p, v in flat[:15]:
            print(f"    {v:3d}  {p} {c} {g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
