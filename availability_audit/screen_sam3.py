#!/usr/bin/env python3
"""garment x pattern x colour availability screening on the SAM3 patch scorer.

ONE COMMAND, FOUR GPUs, LIVE OUTPUT

    conda activate sam3
    python -m availability_audit.screen_sam3

No launcher script and no log-file chase: this process owns four worker
processes, one per card, and prints every finished cell on this terminal as it
lands — the same running table the older screening printed, only with four
cards filling it in.

WHAT IT MEASURES
----------------
For each (colour, garment, pattern) cell it retrieves the top-k images for
"{pattern} {colour} {garment}" and asks how many survive scoring. It is
MEASUREMENT ONLY: nothing is filtered, construction/ is untouched, no data is
regenerated. The table is evidence for a later human decision.

SCORING IS THE PIPELINE'S OWN, IMPORTED NOT REIMPLEMENTED
---------------------------------------------------------
Every rule comes from `fsiglip/collect_img_sam3.py`. Per candidate image, for
the cell's garment:

    SAM3 text-prompted mask   best mask, box expanded by --box-expand
    crop + mask               crop_image_and_mask
    pattern patches           make_patch_files(short_side_tiles=5)   large
    colour patches            make_patch_files(short_side_tiles=15)  dense
    pattern score             score_pattern_patches   6-way argmax + margin
    colour score              score_color_patches    13-way argmax + margin
    garment score             score_garment_vlm      PASS 1.0 / FAIL 0.0
    cell score                sqrt(pattern * colour * garment), then the
                              --attr-min-score floor

The two patch densities are separate because patterns need a window wide enough
to show their repeat while colours are better judged from many small local
patches. The argmax rule is what separates this from any "is this striped?"
threshold scorer: a patch counts only if the target anchor BEATS the others, so
a seam has to out-score "plain solid-colored" to register as a stripe.

The pattern vocabulary is derived from the benchmark config and currently has
six anchors: solid, striped, checkered, floral, polka dot, and leopard. Every
target is scored against this same anchor set. `leopard` is an explicit anchor;
`animal print` is not a separate competing anchor, and there are currently no
pattern synonym-equivalence groups. Therefore the current score uses exact
anchor matching. `--pattern-strict` is retained for compatibility with future
equivalence groups but does not change scores with the current vocabulary.

MASK FAILURE IS NOT ABSENCE
---------------------------
README.md Step 3 is explicit that a segmentation failure must not be counted as
"this garment does not exist" — that would shrink the benchmark to fit SAM3's
blind spots. Every cell reports `mask_status`, and `nonzero` is over SCORED
candidates only, never over candidates SAM3 could not segment.

HOW THE WORK IS SPLIT
---------------------
The sweep runs in colour waves: every garment x pattern cell for one colour
finishes before the next colour is queued. Within a wave, one task is one
garment (up to 6 patterns), so four cards still share the 20 garments on demand.
Crops are cached on (image, garment) in garment-named files and survive between
colour waves, even when a different worker picks up that garment next time.

SAM3 and the vLLM engine share a card. vLLM's default
--gpu-memory-utilization 0.9 leaves under 3 GiB and SAM3 will not fit; serve the
engines at 0.72:

    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$i vllm serve Qwen/Qwen3-VL-4B-Instruct \\
        --port 800$((i+1)) --gpu-memory-utilization 0.72 &
    done

OUTPUTS
-------
    screen_sam3.json                  every cell
    screen_sam3_candidates.jsonl      one row per candidate: the three axis
                                      scores kept separate, plus mask_status
    screen_sam3_<colour>.png          one heatmap per colour
    screen_sam3_summary.png           colour x garment, averaged over patterns
    images_sam3/<colour>/<pattern> <garment>/1__score_0.62__<key>.jpg
                                      the best CROPS — what was actually scored,
                                      not the original frame
    cache_sam3/<garment>.json         crop scores, so a re-run resumes cheaply

Resume is automatic: finished cells are skipped and cached crops are reused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import queue as queuelib
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import FASHION_ATTRIBUTE_AXES, OPTIONS_DIR

# torch, SAM3 and the collector's scorers are imported INSIDE the worker
# processes, after CUDA_VISIBLE_DEVICES is set. Importing them here would give
# the parent a CUDA context on card 0, and would make the env var arrive too
# late to pin a worker to its own card.

OUT_JSON = "screen_sam3.json"
CANDIDATES_JSONL = "screen_sam3_candidates.jsonl"
RUN_CONFIG_JSON = "screen_sam3_config.json"
CACHE_DIR = "cache_sam3"
IMAGE_DIR = "images_sam3"

# Pre-counted per cell so the table can be redrawn at a stricter bar without
# re-reading the row log. "0" reproduces `nonzero` exactly (score > 0); every
# other threshold t counts score >= t.
NONZERO_THRESHOLDS = (0.0, 0.3, 0.5)


# ══ helpers shared by parent and worker ═══════════════════════════════════
def url_key(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:20]


def nonzero_at(scores: list[float]) -> dict[str, int]:
    return {("%g" % t): sum(1 for s in scores if (s > 0.0 if t <= 0.0 else s >= t))
            for t in NONZERO_THRESHOLDS}


def write_json_atomic(path: Path, obj: Any) -> None:
    """Replace one JSON file only after the complete new payload is on disk."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def measurement_config(args, garments: list[str], patterns: list[str],
                       colors: list[str]) -> dict[str, Any]:
    """Settings that can change a measured cell or its reusable crop cache."""
    return {
        "schema_version": 1,
        "experiment": "screen_sam3",
        "top_k": args.top_k,
        "index_name": args.index_name,
        "garment_vocabulary": list(garments),
        # The list the VLM is actually shown, in the form it is shown in. This
        # is NOT redundant with `garment_vocabulary` above: that one is the
        # benchmark axis, this one is the classifier's closed option set, and
        # for the whole v4 sweep the two silently disagreed — the axis said
        # `long_coat`, the classifier was offered `wool coat`. Recording it
        # makes the same-conditions guard refuse to mix the two regimes.
        "vlm_garment_vocabulary": [g.replace("_", " ") for g in garments],
        "patterns_scored": list(patterns),
        "colors_scored": list(colors),
        "pattern_short_side_tiles": args.pattern_short_side_tiles,
        "color_short_side_tiles": args.color_short_side_tiles,
        "min_coverage": args.min_coverage,
        "patch_mask_fill": args.patch_mask_fill,
        "pattern_margin": args.pattern_margin,
        "color_margin": args.color_margin,
        "attr_min_score": args.attr_min_score,
        "min_mask_area_ratio": args.min_mask_area_ratio,
        "max_mask_area_ratio": args.max_mask_area_ratio,
        "min_box_area_ratio": args.min_box_area_ratio,
        "max_box_area_ratio": args.max_box_area_ratio,
        "box_expand": args.box_expand,
        "vlm_model": args.vlm_model,
        "vlm_max_tokens": args.vlm_max_tokens,
        "vlm_temperature": args.vlm_temperature,
        "vlm_error_policy": args.vlm_error_policy,
        "pattern_strict": bool(args.pattern_strict),
        # Pinned False, not read from args: the alias layer no longer exists.
        # The key stays in the manifest so result directories written before
        # docs/PROCESS.md §8 remain resumable under the same-conditions guard.
        "allow_garment_equivalence": False,
        "garment_fail_skip": bool(args.garment_fail_skip),
    }


def prepare_run_config(path: Path, out_json: Path, current: dict[str, Any],
                       redo: bool) -> None:
    """Prevent future resumes from silently mixing experiment conditions.

    Existing result files created before this manifest was introduced remain
    resumable, but are called out as unverified rather than being silently
    assigned settings they may not have used.
    """
    if not out_json.exists():
        write_json_atomic(path, current)
        return
    if not path.exists():
        if redo:
            raise SystemExit(
                f"  [error] cannot safely --redo legacy {out_json.name}: "
                f"{path.name} is missing.\n"
                "          use a new --out-dir so old caches and results remain "
                "traceable")
        print(f"  [warn] {out_json.name} predates {path.name}; resume settings "
              "cannot be verified. Continuing this legacy resume unchanged; "
              "use a new --out-dir for a condition-consistent sweep.")
        return
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"  [error] cannot read {path}: {e}")
    changed = [k for k in sorted(set(saved) | set(current))
               if saved.get(k) != current.get(k)]
    if changed:
        details = "\n".join(
            f"          {k}: saved={saved.get(k)!r}, current={current.get(k)!r}"
            for k in changed[:12])
        raise SystemExit(
            f"  [error] refusing to mix screen conditions in {out_json}:\n"
            f"{details}\n"
            "          use the saved settings or a new --out-dir")


def _plan_usage(plans_path: Path, axis: str, keys: list[str]) -> dict[str, int]:
    """How often each value of one attribute axis appears in the option plans."""
    usage = {k: 0 for k in keys}
    if not plans_path.exists():
        return usage
    with open(plans_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                opts = (json.loads(line).get("options") or {}).values()
            except Exception:
                continue
            for opt in opts:
                v = (opt.get("attributes") or {}).get(axis)
                if v in usage:
                    usage[v] += 1
    return usage


def garment_usage_order(garments: list[str], plans_path: Path) -> tuple[list[str], dict[str, int]]:
    """Most-used garments first, so an interrupted sweep covers what matters."""
    if not plans_path.exists():
        print(f"  [warn] {plans_path} not found; using config order")
        return list(garments), {g: 0 for g in garments}
    usage = _plan_usage(plans_path, "garment_category", garments)
    return sorted(garments, key=lambda g: (-usage[g], g)), usage


def color_task_waves(table: dict, colors: list[str], garments: list[str],
                     patterns: list[str], limit_cells: int = 0
                     ) -> tuple[list[tuple[str, list[dict], int]], int]:
    """Pending work grouped so no later colour is queued before this one ends."""
    waves = []
    n_cells = 0
    for color in colors:
        tasks = []
        wave_cells = 0
        for garment in garments:
            cells = [(color, pattern) for pattern in patterns
                     if pattern not in table.get(color, {}).get(garment, {})]
            if limit_cells > 0:
                cells = cells[:max(0, limit_cells - n_cells)]
            if cells:
                tasks.append({"garment": garment, "cells": cells})
                wave_cells += len(cells)
                n_cells += len(cells)
            if limit_cells > 0 and n_cells >= limit_cells:
                break
        if tasks:
            waves.append((color, tasks, wave_cells))
        if limit_cells > 0 and n_cells >= limit_cells:
            break
    return waves, n_cells


# ══ worker: one GPU ═══════════════════════════════════════════════════════
def _worker(worker_id: int, gpu: str, vlm_url: str, args: argparse.Namespace,
            task_q, res_q) -> None:
    """Pull one colour x garment block off the queue; stream cells back.

    The card is chosen by the environment, never by a `cuda:N` string: SAM3's
    builder moves the model only when device == "cuda" exactly
    (sam3/model_builder.py), so an index is ignored and the model silently
    stays on the CPU. Setting CUDA_VISIBLE_DEVICES before the first torch
    import is what pins this process.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    args.vlm_url = vlm_url

    def say(kind: str, **kw):
        res_q.put({"kind": kind, "worker": worker_id, "gpu": gpu, **kw})

    try:
        import concurrent.futures as cf
        import shutil
        import torch
        from PIL import Image
        from fsiglip.collect_img_sam3 import (
            query_knn, download_image, load_sam3, run_sam3_all, expand_box,
            crop_image_and_mask, make_patch_files, score_pattern_patches,
            score_color_patches, score_garment_vlm, norm, save_mask,
            masked_crop_preview, TEXT_QUERY_GARMENT_VOCAB, GARMENT_VOCAB,
            PATTERN_VOCAB,
            pattern_patch_rows, color_patch_rows,
        )
    except Exception as e:
        say("fatal", error=f"import failed: {type(e).__name__}: {e}")
        return

    try:
        free = torch.cuda.mem_get_info(0)[0] / 1024 ** 3
        if free < args.min_free_gib:
            say("fatal", error=(
                f"only {free:.1f} GiB free; SAM3 needs about {args.min_free_gib:g}. "
                f"It shares the card with the vLLM engine — re-serve vLLM with "
                f"--gpu-memory-utilization 0.72, or pass --min-free-gib 0."))
            return
        say("info", text=f"{free:.1f} GiB free, {len(PATTERN_VOCAB)} pattern "
                         f"anchors, loading SAM3")
        processor = load_sam3("cuda")
    except Exception as e:
        say("fatal", error=f"SAM3 load failed: {type(e).__name__}: {e}")
        return

    session = requests.Session()
    pool = max(8, args.download_workers * 2)
    adapter = requests.adapters.HTTPAdapter(pool_connections=pool, pool_maxsize=pool,
                                            max_retries=0)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    # Report the vocabulary the VLM will ACTUALLY be shown. The parent records
    # its own config-derived expectation in the manifest and aborts on any
    # difference: the `wool coat` contamination survived a full sweep precisely
    # because the manifest described the intended vocabulary and nothing ever
    # compared it with the list the model saw (docs/PROCESS.md §8-2).
    say("ready", garment_vocab=list(GARMENT_VOCAB))

    def score_image(image_path: Path, garment: str, work: Path) -> dict[str, Any]:
        """SAM3 + patches + EVERY pattern and colour for this crop.

        All targets on both axes are scored here, not just the current cell's:
        the expensive part (SAM3, the crop, the patch files) is already paid
        for, and the other 77 cells of this garment will ask for them.

        Each target gets its OWN call to score_*_patches. Reading a second
        target's hits off one shared pass would use the wrong margin —
        `tile_scores` records `margin` relative to the single target the pass
        ran for — and reconstructing it here would mean writing the scoring
        rule down a second time, which is how a screening drifts away from the
        pipeline it is meant to mirror.
        """
        out: dict[str, Any] = {"mask_status": "ok", "garment": 0.0,
                               "pattern": {}, "color": {}}
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            out["mask_status"] = "image_unreadable"
            out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            return out

        try:
            cands = run_sam3_all(processor, image,
                                 [norm(garment) or "clothing item"], args)
        except Exception as e:
            out["mask_status"] = "sam3_error"
            out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            return out
        if not cands:
            out["mask_status"] = "no_sam3_mask"
            return out

        best = cands[0]
        box = expand_box(best["box"], image.size, args.box_expand)
        crop, crop_mask, xyxy = crop_image_and_mask(image, best["mask"], box)

        work.mkdir(parents=True, exist_ok=True)
        crop_path = work / "crop.jpg"
        crop.save(crop_path, quality=95)
        if args.save_masks:
            save_mask(crop_mask, work / "mask.png")
            masked_crop_preview(crop, crop_mask, args.patch_mask_fill).save(
                work / "masked_crop.jpg", quality=95)

        pat = make_patch_files(crop=crop, crop_mask=crop_mask, args=args,
                               patch_dir=work / "patches",
                               short_side_tiles=args.pattern_short_side_tiles,
                               axis="pattern")
        col = make_patch_files(crop=crop, crop_mask=crop_mask, args=args,
                               patch_dir=work / "color_patches",
                               short_side_tiles=args.color_short_side_tiles,
                               axis="color")
        if not pat or not col:
            # a mask too small for even one patch at --min-coverage is a mask
            # failure, not a statement about the garment
            out["mask_status"] = "no_valid_patches"
            return out
        out["n_pattern_patches"], out["n_color_patches"] = len(pat), len(col)

        # ONE encode per axis, reused by every target on it. The scoring
        # service re-encodes the images on every call, so asking per target
        # would re-send the same ~300 colour patches 13 times — 12x the work,
        # and it is the dominant cost of the sweep. The hit rule still lives in
        # score_*_patches; only the encoding is hoisted.
        prows = pattern_patch_rows(session, args, pat)
        for p in args.all_patterns:
            info = score_pattern_patches(session, args, p, pat, rows=prows) or {}
            out["pattern"][p] = float(info.get("score") or 0.0)
        crows = color_patch_rows(session, args, col)
        for c in args.all_colors:
            info = score_color_patches(session=session, args=args, color=c,
                                       patches=col, rows=crows) or {}
            out["color"][c] = float(info.get("score") or 0.0)

        g_score, g_info = score_garment_vlm(session=session, args=args,
                                            crop_path=crop_path, garment=garment,
                                            vocab=tuple(TEXT_QUERY_GARMENT_VOCAB))
        out["garment"] = float(g_score)
        out["garment_mode"] = g_info.get("mode")
        out["crop"] = str(crop_path)
        # keeps "was this really a crop, or the whole picture?" answerable after
        # the node-local work dir is gone
        out["crop_area_frac"] = round(
            ((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))
            / float(image.size[0] * image.size[1]), 4)
        out["mask_area_frac"] = round(float(best["mask_area_ratio"]), 4)
        return out

    def save_top_crops(color, garment, pattern, rows, cache) -> list[dict]:
        """Save the best-scoring CROPS for this cell.

        The crop is what the score was computed on, so it is what gets saved;
        the retrieved frame is mostly background, face and other garments and
        tells you nothing about the number. Only candidates scoring > 0 are
        saved, so a sparse folder is itself the finding.
        """
        if args.save_top <= 0:
            return []
        best = sorted((r for r in rows
                       if r["score"] > 0.0 and r["mask_status"] == "ok"),
                      key=lambda r: -r["score"])[:args.save_top]
        if not best:
            return []
        dest = (args.out_dir / IMAGE_DIR / norm(color) /
                f"{norm(pattern)} {norm(garment)}".strip())
        dest.mkdir(parents=True, exist_ok=True)
        saved = []
        for rank, r in enumerate(best, 1):
            src = (cache.get(r["url_key"]) or {}).get("crop")
            if not src or not Path(src).exists():
                continue
            out = dest / f"{rank}__score_{r['score']:.3f}__{r['url_key']}.jpg"
            try:
                shutil.copyfile(src, out)
                saved.append({"rank": rank, "score": r["score"],
                              "file": str(out.relative_to(args.out_dir))})
            except Exception:
                pass
        return saved

    # counted per run, not stored in the cache file: a persisted counter would
    # keep accumulating across resumes and the reported hit rate would be a
    # figure for every run ever rather than for this one
    tally = {"hits": 0, "misses": 0}

    def measure_cell(color: str, garment: str, pattern: str,
                     cache: dict, imgdir: Path) -> tuple[dict, list]:
        parts = ([norm(pattern)] if pattern and pattern != "solid" else []) \
            + [norm(color), norm(garment)]
        text = " ".join(x for x in parts if x).strip()
        t0 = time.time()
        errors: list[str] = []

        try:
            results = query_knn(session, args, text)[:args.top_k]
        except Exception as e:
            return ({"nonzero": 0, "top1": 0.0, "median": 0.0, "n_candidates": 0,
                     "n_scored": 0, "nonzero_at": nonzero_at([]), "query": text,
                     "mask_status": {}, "seconds": round(time.time() - t0, 1),
                     "errors": [f"knn: {type(e).__name__}: {str(e)[:200]}"],
                     "saved_crops": []}, [])

        urls = [str(r.get("image_url") or r.get("url") or "") for r in results]
        paths: list[Path | None] = [None] * len(urls)

        def fetch(iu):
            i, url = iu
            if not url:
                return i, None, "missing_url"
            try:
                p, err = download_image(session, url, imgdir / url_key(url), args)
                return i, p, err
            except Exception as e:
                return i, None, f"{type(e).__name__}: {str(e)[:200]}"

        with cf.ThreadPoolExecutor(max_workers=args.download_workers) as ex:
            for i, path, err in ex.map(fetch, list(enumerate(urls))):
                paths[i] = path
                if path is None and len(errors) < 20:
                    errors.append(f"download[{i}]: {str(err)[:200]}")

        scores, rows, status = [], [], {}
        for i, (url, path) in enumerate(zip(urls, paths)):
            row = {"color": color, "garment": garment, "pattern": pattern,
                   "rank": i, "url_key": url_key(url) if url else None,
                   "pattern_score": None, "color_score": None,
                   "garment_score": None, "score": 0.0,
                   "mask_status": None, "skip_reason": None,
                   "crop_area_frac": None, "mask_area_frac": None}
            if path is None:
                row["mask_status"] = "download_failed"
                status["download_failed"] = status.get("download_failed", 0) + 1
                rows.append(row)
                continue

            if args.heartbeat > 0 and i and i % args.heartbeat == 0:
                say("info", text=f"{garment} x {pattern} {color}: "
                                 f"{i}/{len(urls)} candidates")
            uk = url_key(url)
            info = cache.get(uk)
            if info is None:
                info = score_image(path, garment,
                                   args.work_root / "crops" /
                                   f"{uk}__{norm(garment).replace(' ', '_')}")
                cache[uk] = info
                tally["misses"] += 1
            else:
                tally["hits"] += 1

            ms = info.get("mask_status", "ok")
            row["mask_status"] = ms
            status[ms] = status.get(ms, 0) + 1
            if ms != "ok":
                # axis scores stay None: this candidate was never judged, and a
                # 0.0 here would read back as an absent garment
                rows.append(row)
                continue

            p_s = float(info["pattern"].get(pattern, 0.0))
            c_s = float(info["color"].get(color, 0.0))
            g_s = float(info.get("garment", 0.0))
            score = max(0.0, p_s * c_s * g_s) ** 0.5
            skip = None
            below = [a for a, v in (("pattern", p_s), ("color", c_s))
                     if args.attr_min_score > 0.0 and v <= args.attr_min_score]
            if below:
                score, skip = 0.0, "_".join(below) + "_below_min"
            elif g_s <= 0.0:
                skip = "garment_fail"
            row.update({"pattern_score": round(p_s, 4), "color_score": round(c_s, 4),
                        "garment_score": g_s, "score": round(score, 4),
                        "skip_reason": skip,
                        "crop_area_frac": info.get("crop_area_frac"),
                        "mask_area_frac": info.get("mask_area_frac")})
            scores.append(score)
            rows.append(row)

        saved = save_top_crops(color, garment, pattern, rows, cache)
        ordered = sorted(scores, reverse=True)
        return ({"nonzero": sum(1 for s in ordered if s > 0.0),
                 "nonzero_at": nonzero_at(ordered),
                 "top1": round(ordered[0], 4) if ordered else 0.0,
                 "median": round(statistics.median(ordered), 4) if ordered else 0.0,
                 "n_candidates": len(results), "n_scored": len(ordered),
                 "mask_status": status, "query": text,
                 "seconds": round(time.time() - t0, 1), "errors": errors,
                 "saved_crops": saved}, rows)

    imgdir = args.work_root / "img"
    imgdir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out_dir / CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            task = task_q.get(timeout=2)
        except queuelib.Empty:
            continue
        if task is None:
            break
        garment, cells = task["garment"], task["cells"]

        # The cache is keyed by GARMENT, not by worker: which worker picks up a
        # garment differs between runs, so a worker-keyed cache would miss on
        # every resume.
        cache_path = cache_dir / f"{norm(garment).replace(' ', '_')}.json"
        cache: dict[str, Any] = {}
        if cache_path.exists() and not args.redo:
            try:
                cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception as e:
                say("info", text=f"cache {cache_path.name} unreadable ({e}); starting fresh")

        for color, pattern in cells:
            try:
                cell, rows = measure_cell(color, garment, pattern, cache, imgdir)
            except Exception as e:
                cell, rows = ({"nonzero": 0, "top1": 0.0, "median": 0.0,
                               "n_candidates": 0, "n_scored": 0,
                               "nonzero_at": nonzero_at([]), "query": "",
                               "mask_status": {}, "seconds": 0.0,
                               "saved_crops": [],
                               "errors": [f"cell: {type(e).__name__}: {str(e)[:200]}"]},
                              [])
            say("cell", color=color, garment=garment, pattern=pattern,
                cell=cell, rows=rows)
            try:
                tmp = cache_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(cache), encoding="utf-8")
                tmp.replace(cache_path)
            except Exception:
                pass

        say("garment_done", garment=garment,
            hits=tally["hits"], misses=tally["misses"])
        tally["hits"] = tally["misses"] = 0

    say("bye")


# ══ parent: tables and figures ════════════════════════════════════════════
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
        label = f"{best[1]} {best[0]} ({best[2]})" if best[0] else "-"
        print(f"{c:<10}{cusage.get(c, 0):>7}{len(vals):>7}"
              f"{statistics.mean(vals):>8.1f}{statistics.median(vals):>8.1f}"
              f"{sum(1 for v in vals if v == 0):>7}"
              f"{sum(1 for v in vals if v < 3):>6}{label:>28}")
    print("=" * len(head))


def write_labeled_heatmap(grid, rows, cols, out_png: Path, top_k: int,
                          title: str, xlabel: str, ylabel: str) -> str:
    """Heatmap with caller-supplied title and axis labels.

    Falls back to PIL where matplotlib is missing, so a headless environment
    still produces the figure rather than silently skipping it.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots(figsize=(1.05 * len(cols) + 3.2,
                                        0.42 * len(rows) + 2.2))
        values = np.array([[np.nan if v is None else v for v in line]
                           for line in grid], dtype=float)
        cmap = plt.get_cmap("coolwarm").copy()
        cmap.set_bad("#bdbdbd")
        im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap,
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
                label = "—" if v is None else str(v)
                text_color = ("black" if v is None else
                              "white" if (v > 0.66 * top_k or
                                          v < 0.12 * top_k) else "black")
                ax.text(j, i, label, ha="center", va="center", fontsize=8,
                        color=text_color)
        cb = fig.colorbar(im, ax=ax)
        cb.set_label(f"# candidates with score > 0  (0 - {top_k})")
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
        if v is None:
            return (189, 189, 189)
        t = 0.0 if top_k <= 0 else min(max(v / top_k, 0.0), 1.0)
        return (int(59 + t * 137), int(76 - t * 16), int(192 - t * 138))

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
            d.rectangle([x, y, x + cw - 1, y + ch - 1], fill=color_of(v),
                        outline="white")
            label = "-" if v is None else str(v)
            d.text((x + cw // 2 - 8, y + 7), label,
                   fill="white" if v is not None and v > top_k * 0.55
                   else "black")
    d.text((lpad, tpad + ch * len(rows) + 14), f"{xlabel} ->", fill="black")
    img.save(out_png)
    return "PIL"


def write_reports(table: dict, colors: list[str], rows: list[str],
                  patterns: list[str], gusage: dict, cusage: dict, args) -> None:
    written = []
    for c in colors:
        sub = table.get(c, {})
        if not sub:
            continue
        print_color_table(c, sub, rows, patterns, gusage, args.top_k)
        png = args.out_dir / f"screen_sam3_{c}.png"
        grid = [[(sub.get(g, {}).get(p) or {}).get("nonzero") for p in patterns]
                for g in rows]
        measured = sum(v is not None for line in grid for v in line)
        total = len(rows) * len(patterns)
        progress = "" if measured == total else f" | PARTIAL {measured}/{total}"
        backend = write_labeled_heatmap(
            grid, rows, patterns, png, args.top_k,
            title=(f"COLOR = {c}  |  (garment, pattern) image availability"
                   f"{progress}"),
            xlabel="pattern", ylabel="garment  (most-used at top)")
        written.append(png)
        print(f"  heatmap ({backend}) -> {png}")

    print_color_summary(table, colors, rows, patterns, cusage, args.top_k)

    grid = []
    for g in rows:
        line = []
        for c in colors:
            vals = [(table.get(c, {}).get(g, {}) or {}).get(p, {}).get("nonzero")
                    for p in patterns]
            line.append(round(statistics.mean(vals))
                        if vals and all(v is not None for v in vals) else None)
        grid.append(line)
    measured = sum(
        1 for c in colors for g in rows for p in patterns
        if (table.get(c, {}).get(g, {}) or {}).get(p) is not None)
    total = len(colors) * len(rows) * len(patterns)
    progress = "" if measured == total else f" | PARTIAL {measured}/{total}"
    summary_png = args.out_dir / "screen_sam3_summary.png"
    backend = write_labeled_heatmap(
        grid, rows, colors, summary_png, args.top_k,
        title=("Availability by colour and garment (mean over patterns)"
               f"{progress}"),
        xlabel="color", ylabel="garment  (most-used at top)")
    print(f"\n  summary heatmap, colour x garment averaged over patterns "
          f"({backend}) -> {summary_png}")
    print(f"  {len(written)} per-colour heatmaps written to {args.out_dir}")

    agg: dict[str, int] = {}
    for cc in table.values():
        for gg in cc.values():
            for cell in gg.values():
                for k, v in (cell.get("mask_status") or {}).items():
                    agg[k] = agg.get(k, 0) + v
    total = sum(agg.values()) or 1
    print("\n  candidate outcomes across all cells "
          "(a mask failure is NOT evidence of absence):")
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<20}{v:>8}  ({100 * v / total:.1f}%)")

    flat = [(c, g, p, cell["nonzero"])
            for c in table for g in table[c] for p, cell in table[c][g].items()]
    if flat:
        flat.sort(key=lambda x: x[3])
        print("\n  lowest 15 cells overall:")
        for c, g, p, v in flat[:15]:
            print(f"    {v:3d}  {p} {c} {g}")


# ══ parent: preflight ═════════════════════════════════════════════════════
def preflight(args, vlm_urls: list[str]) -> None:
    """Prove every dependency before a single GPU is touched.

    Loading four SAM3 models takes a while, so a wrong endpoint found after
    that wastes the load and produces four identical tracebacks. The `sam3`
    import is checked first: the wrong conda environment is both the most
    common failure and the cheapest to detect.
    """
    print("-- preflight " + "-" * 58)
    import importlib.util
    if importlib.util.find_spec("sam3") is None:
        raise SystemExit(
            f"  [abort] `sam3` is not importable in this environment "
            f"({sys.prefix}).\n          conda activate sam3, then re-run.")
    print(f"  ok  sam3      {sys.prefix}")

    s = requests.Session()
    try:
        r = s.post(args.client_url,
                   json={"text": "black formal shirt", "modality": "image",
                         "num_images": 1, "indice_name": args.index_name},
                   timeout=args.timeout)
        r.raise_for_status()
        if not isinstance(r.json(), list):
            raise RuntimeError(f"expected a list, got {r.text[:120]!r}")
        print(f"  ok  knn       {args.client_url}")
    except Exception as e:
        raise SystemExit(f"  [abort] knn {args.client_url}: "
                         f"{type(e).__name__}: {str(e)[:200]}")

    # The probe must carry a real image: this route rejects an empty
    # image_paths with 400, and a per-image failure comes back INSIDE a 200 as
    # {"error": ...}, so checking the HTTP status alone proves nothing.
    probe = args.work_root / "preflight_probe.jpg"
    try:
        probe.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        Image.new("RGB", (64, 64), (32, 32, 32)).save(probe, quality=90)
        r = s.post(args.score_url,
                   json={"texts": {"probe": "a close-up of black clothing fabric"},
                         "image_paths": [str(probe)]},
                   timeout=args.score_timeout)
        r.raise_for_status()
        rowz = (r.json() or {}).get("scores")
        if not isinstance(rowz, list) or not rowz:
            raise RuntimeError(f"no rows: {r.text[:160]!r}")
        if "error" in rowz[0]:
            raise RuntimeError(f"probe not scorable: {rowz[0]['error']}")
        if "probe" not in rowz[0]:
            raise RuntimeError(f"row is missing the anchor: {rowz[0]}")
        print(f"  ok  score     {args.score_url}")
    except Exception as e:
        raise SystemExit(f"  [abort] score {args.score_url}: "
                         f"{type(e).__name__}: {str(e)[:200]}")

    for u in vlm_urls:
        models = u.split("/chat/completions")[0].rstrip("/") + "/models"
        try:
            r = s.get(models, timeout=args.vlm_timeout)
            r.raise_for_status()
            served = [m["id"] for m in r.json()["data"]]
        except Exception as e:
            raise SystemExit(f"  [abort] vlm {models}: "
                             f"{type(e).__name__}: {str(e)[:200]}")
        if args.vlm_model not in served:
            raise SystemExit(f"  [abort] {models} serves {served}, "
                             f"not {args.vlm_model!r}")
    print(f"  ok  vlm       {len(vlm_urls)} endpoints serving {args.vlm_model}")
    print("-" * 71)


# ══ parent: adopt what earlier layouts produced ═══════════════════════════
def adopt_legacy(args) -> None:
    """Fold in results and crop caches written by the earlier shard layout.

    Those runs cost real GPU time and their per-image SAM3 records are still
    valid — the scoring rules did not change, only how the work is divided.
    Old files are RENAMED, not deleted, so the migration can be undone.
    """
    out_dir = args.out_dir

    old_caches = sorted(out_dir.glob("screen_sam3.scorecache*.json"))
    if old_caches:
        per_garment: dict[str, dict] = {}
        n = 0
        for f in old_caches:
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            for key, rec in d.items():
                if "|" not in key or not isinstance(rec, dict):
                    continue
                uk, garment = key.split("|", 1)
                per_garment.setdefault(garment.strip().replace(" ", "_"), {})[uk] = rec
                n += 1
        if n:
            cache_dir = out_dir / CACHE_DIR
            cache_dir.mkdir(parents=True, exist_ok=True)
            for garment, recs in per_garment.items():
                path = cache_dir / f"{garment}.json"
                cur = {}
                if path.exists():
                    try:
                        cur = json.loads(path.read_text(encoding="utf-8"))
                    except Exception:
                        cur = {}
                cur.update(recs)
                path.write_text(json.dumps(cur), encoding="utf-8")
            print(f"  adopted {n} cached crops from {len(old_caches)} legacy "
                  f"cache file(s)")
        for f in old_caches:
            f.rename(f.with_name(f.name + ".migrated"))

    shard_jsons = sorted(out_dir.glob("screen_sam3.shard*.json"))
    if shard_jsons:
        p = out_dir / OUT_JSON
        base = {}
        if p.exists():
            try:
                base = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                base = {}
        n = 0
        for f in shard_jsons:
            try:
                part = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            for c, gg in part.items():
                for g, pp in gg.items():
                    for pat, cell in pp.items():
                        base.setdefault(c, {}).setdefault(g, {})[pat] = cell
                        n += 1
            f.rename(f.with_name(f.name + ".migrated"))
        write_json_atomic(p, base)
        print(f"  adopted {n} cells from {len(shard_jsons)} legacy shard file(s)")

    shard_rows = sorted(out_dir.glob("screen_sam3_candidates.shard*.jsonl"))
    if shard_rows:
        with open(out_dir / CANDIDATES_JSONL, "a", encoding="utf-8") as out:
            for f in shard_rows:
                out.write(f.read_text(encoding="utf-8"))
                f.rename(f.with_name(f.name + ".migrated"))
        print(f"  adopted rows from {len(shard_rows)} legacy row log(s)")


# ══ parent: main ══════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--out-dir", type=Path, default=here)
    p.add_argument("--colors", default="", help="comma-separated subset")
    p.add_argument("--garments", default="", help="comma-separated subset")
    p.add_argument("--skip-solid", action="store_true")
    p.add_argument("--limit-cells", type=int, default=0, help="debug: only N cells")
    p.add_argument("--redo", action="store_true",
                   help="ignore saved cells and measure everything again")
    p.add_argument("--plans", type=Path, default=OPTIONS_DIR / "option_plans.jsonl")
    p.add_argument("--save-top", type=int, default=20,
                   help="best-scoring CROPS kept per cell (0 = none)")
    p.add_argument("--save-masks", action="store_true",
                   help="also write mask.png / masked_crop.jpg beside each crop")

    p.add_argument("--gpus", default="0,1,2,3", help="one worker per entry")
    p.add_argument("--vlm-ports", default="8001,8002,8003,8004",
                   help="vLLM port per worker, positional with --gpus")
    p.add_argument("--download-workers", type=int, default=16)
    p.add_argument("--heartbeat", type=int, default=25,
                   help="print progress every N candidates within a cell "
                        "(0 = only per-cell lines)")
    p.add_argument("--min-free-gib", type=float, default=4.0,
                   help="a worker aborts if its card has less free VRAM (0 = off)")
    p.add_argument("--skip-preflight", action="store_true")

    p.add_argument("--client-url", default="http://127.0.0.1:1235/knn-service")
    p.add_argument("--score-url", default="http://127.0.0.1:1235/score-image-files")
    p.add_argument("--index-name", default="fsiglip")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--score-timeout", type=float, default=120.0)
    p.add_argument("--score-batch", type=int, default=128)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--download-timeout", type=float, default=30.0)
    p.add_argument("--max-image-bytes", type=int, default=20_000_000)

    p.add_argument("--min-mask-area-ratio", type=float, default=0.000001)
    p.add_argument("--max-mask-area-ratio", type=float, default=1.0)
    p.add_argument("--min-box-area-ratio", type=float, default=0.000001)
    p.add_argument("--max-box-area-ratio", type=float, default=1.0)
    p.add_argument("--box-expand", type=float, default=0.02)
    # Patch DENSITY, not size: a patch side is ceil(min(crop_w, crop_h) / N), so
    # a LARGER number means SMALLER patches. The defaults are the values
    # collect_img_sam3.py is actually INVOKED with (5 / 15), not its argparse
    # defaults of 8/8 — this screening measures what the real pipeline yields,
    # so the real densities have to be what you get without flags.
    p.add_argument("--short-side-tiles", type=int, default=None,
                   help="set both axes at once")
    p.add_argument("--pattern-short-side-tiles", type=int, default=None,
                   help="pattern patch density (default 5 = large patches)")
    p.add_argument("--color-short-side-tiles", type=int, default=None,
                   help="colour patch density (default 15 = dense patches)")
    p.add_argument("--min-coverage", type=float, default=0.20)
    p.add_argument("--patch-mask-fill",
                   choices=["local_mean", "gray", "white", "black"],
                   default="local_mean")
    p.add_argument("--pattern-margin", type=float, default=0.0)
    p.add_argument("--color-margin", type=float, default=0.005)
    p.add_argument("--attr-min-score", type=float, default=0.2)

    p.add_argument("--vlm-model", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--vlm-max-tokens", type=int, default=16)
    p.add_argument("--vlm-temperature", type=float, default=0.0)
    p.add_argument("--vlm-timeout", type=float, default=120.0)
    p.add_argument("--vlm-retries", type=int, default=4)
    p.add_argument("--vlm-error-policy", choices=["skip", "uncertain"], default="skip")
    p.add_argument("--pattern-strict", action="store_true",
                   help="require an exact pattern anchor match; the current "
                        "vocabulary is already exact-only because it has no "
                        "pattern equivalence groups")
    # --allow-garment-equivalence was removed together with the garment alias
    # table in fsiglip/ (see docs/PROCESS.md §8). The garment verdict is now an
    # exact match against the canonical 23 and nothing can loosen it, so an
    # opt-in flag here would be a false affordance.
    p.add_argument("--garment-fail-skip", action="store_true", default=True)

    a = p.parse_args()
    if a.pattern_short_side_tiles is None:
        a.pattern_short_side_tiles = a.short_side_tiles or 5
    if a.color_short_side_tiles is None:
        a.color_short_side_tiles = a.short_side_tiles or 15
    if a.pattern_short_side_tiles <= 0 or a.color_short_side_tiles <= 0:
        raise SystemExit("  [error] patch densities must be positive")
    # names the imported collector helpers read straight off `args`
    a.retrieval_k = a.top_k
    a.device = "cuda"     # the card is chosen with CUDA_VISIBLE_DEVICES
    a.force = False       # download_image reads this: never refetch
    a.work_root = Path(tempfile.gettempdir()) / "pod_screen_sam3"
    return a


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / OUT_JSON
    jsonl_path = args.out_dir / CANDIDATES_JSONL

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    ports = [p.strip() for p in args.vlm_ports.split(",") if p.strip()]
    if not gpus:
        raise SystemExit("  [error] --gpus is empty")
    if len(ports) < len(gpus):
        raise SystemExit(f"  [error] {len(gpus)} gpus but only {len(ports)} "
                         f"--vlm-ports; give one port per gpu")
    vlm_urls = [f"http://127.0.0.1:{p}/v1/chat/completions" for p in ports[:len(gpus)]]

    garments = list(FASHION_ATTRIBUTE_AXES["garment_category"])
    patterns = list(FASHION_ATTRIBUTE_AXES["pattern"])
    colors = list(FASHION_ATTRIBUTE_AXES["color"])
    if args.skip_solid:
        patterns = [p for p in patterns if p != "solid"]
    rows, gusage = garment_usage_order(garments, args.plans)
    cusage = _plan_usage(args.plans, "color", colors)
    if args.colors:
        want = {c.strip() for c in args.colors.split(",") if c.strip()}
        if want - set(colors):
            raise SystemExit(f"  [error] not colours: {sorted(want - set(colors))}")
        colors = [c for c in colors if c in want]
    if args.garments:
        want = {g.strip() for g in args.garments.split(",") if g.strip()}
        if want - set(garments):
            raise SystemExit(f"  [error] not garments: {sorted(want - set(garments))}")
        rows = [g for g in rows if g in want]
    args.all_patterns, args.all_colors = patterns, colors

    print("=" * 74)
    print("  garment x pattern x COLOR availability screening — SAM3 patch scorer")
    print(f"  {len(colors)} colors x {len(rows)} garments x {len(patterns)} patterns "
          f"= {len(colors) * len(rows) * len(patterns)} cells")
    print(f"  top-k per cell: {args.top_k}   scorer: fsiglip/collect_img_sam3.py")
    print(f"  workers: {len(gpus)} (gpu {', '.join(gpus)} -> port "
          f"{', '.join(ports[:len(gpus)])})")
    print(f"  patches: pattern short-side {args.pattern_short_side_tiles}, "
          f"colour short-side {args.color_short_side_tiles}, "
          f"fill {args.patch_mask_fill}, min coverage {args.min_coverage}")
    print(f"  score:   sqrt(pattern*colour*garment), attr floor "
          f"{args.attr_min_score}, margins p={args.pattern_margin} "
          f"c={args.color_margin}")
    print("=" * 74)

    if args.redo and (args.garments or args.colors):
        raise SystemExit(
            "  [error] --redo deletes screen_sam3.json and "
            "screen_sam3_candidates.jsonl for the WHOLE directory, so pairing "
            "it with\n"
            "          --garments/--colors would throw away the subsets you "
            "did not ask to redo.\n"
            "          Re-measure a subset into a fresh --out-dir instead; the "
            "annotator merges snapshots\n"
            "          (annotation/serve_annotator.py --screen-dir), and the "
            "original results stay intact.")

    if not args.skip_preflight:
        preflight(args, vlm_urls)
    adopt_legacy(args)
    prepare_run_config(
        args.out_dir / RUN_CONFIG_JSON,
        out_json,
        measurement_config(args, garments, patterns, colors),
        args.redo,
    )
    if args.redo:
        # `--redo` means re-score, not merely rebuild the table from old crop
        # scores. Images remain reusable, but result rows and score caches do not.
        if out_json.exists():
            out_json.unlink()
        if jsonl_path.exists():
            jsonl_path.unlink()

    table: dict[str, dict[str, dict[str, Any]]] = {}
    if out_json.exists() and not args.redo:
        try:
            table = json.loads(out_json.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  [warn] could not read {out_json.name}: {e}")
        done = sum(len(pp) for cc in table.values() for pp in cc.values())
        if done:
            print(f"  resuming: {done} cells already measured")

    waves, n_cells = color_task_waves(
        table, colors, rows, patterns, args.limit_cells)
    if not n_cells:
        print("  nothing left to measure")
        write_reports(table, colors, rows, patterns, gusage, cusage, args)
        return 0

    n_tasks = sum(len(tasks) for _, tasks, _ in waves)
    print(f"  {n_cells} cells to measure in {len(waves)} color wave(s), "
          f"{n_tasks} color x garment tasks")
    # the anchor count comes from a worker (PATTERN_VOCAB lives behind the
    # worker-only import; pulling it in here would drag torch into the parent)
    print("  patterns: one shared anchor set for every target"
          + ("; strict anchors (--pattern-strict)" if args.pattern_strict else
             "; leopard also accepts \"animal print\""))
    print()

    ctx = mp.get_context("spawn")   # never fork: workers set CUDA_VISIBLE_DEVICES
    task_q, res_q = ctx.Queue(), ctx.Queue()

    def queue_wave(index: int) -> tuple[str, int]:
        color, tasks, n_wave_cells = waves[index]
        print(f"  color wave {index + 1}/{len(waves)}: {color} "
              f"({n_wave_cells} cells, {len(tasks)} garment tasks)")
        for task in tasks:
            task_q.put(task)
        if index == len(waves) - 1:
            for _ in gpus:
                task_q.put(None)
        return color, n_wave_cells

    wave_index = 0
    active_color, wave_remaining = queue_wave(wave_index)

    procs = []
    for i, gpu in enumerate(gpus):
        p = ctx.Process(target=_worker,
                        args=(i, gpu, vlm_urls[i], args, task_q, res_q),
                        daemon=True)
        p.start()
        procs.append(p)

    done = ready = alive = 0
    hits = misses = 0
    # What the classifier must be offered, in the form it is offered in. A
    # worker that reports anything else is scoring a different experiment.
    expected_vlm_vocab = [g.replace("_", " ") for g in garments]
    vocab_mismatch: tuple[int, list[str]] | None = None
    t0 = time.time()
    jsonl = open(jsonl_path, "a", encoding="utf-8")
    try:
        while done < n_cells:
            try:
                msg = res_q.get(timeout=5)
            except queuelib.Empty:
                if not any(p.is_alive() for p in procs):
                    print("  [warn] every worker exited before the sweep finished")
                    break
                continue

            kind = msg["kind"]
            if kind == "ready":
                ready += 1
                alive += 1
                print(f"  worker {msg['worker']} ready on gpu {msg['gpu']} "
                      f"({ready}/{len(procs)})")
                got = list(msg.get("garment_vocab") or [])
                if got != expected_vlm_vocab:
                    vocab_mismatch = (msg["worker"], got)
                    break
            elif kind == "info":
                print(f"  worker {msg['worker']} (gpu {msg['gpu']}): {msg['text']}")
            elif kind == "fatal":
                print(f"  [abort] worker {msg['worker']} (gpu {msg['gpu']}): "
                      f"{msg['error']}")
            elif kind == "garment_done":
                hits += msg["hits"]
                misses += msg["misses"]
            elif kind == "bye":
                alive -= 1
                if alive <= 0 and done < n_cells:
                    print("  [warn] all workers finished early")
                    break
            elif kind == "cell":
                done += 1
                c, g, p, cell = msg["color"], msg["garment"], msg["pattern"], msg["cell"]
                table.setdefault(c, {}).setdefault(g, {})[p] = cell
                for r in msg["rows"]:
                    jsonl.write(json.dumps(r, ensure_ascii=False) + "\n")
                jsonl.flush()
                nomask = sum(v for k, v in cell["mask_status"].items() if k != "ok")
                print(f"  [{done:4d}/{n_cells}] {c:8s} | {g:16s} x {p:10s} "
                      f"nonzero={cell['nonzero']:3d}/{cell['n_scored']:<3d} "
                      f"top1={cell['top1']:.2f} nomask={nomask:<3d} "
                      f"({cell['seconds']}s)"
                      + (f"  [{len(cell['errors'])} err]" if cell["errors"] else ""))
                write_json_atomic(out_json, table)
                if c == active_color:
                    wave_remaining -= 1
                    if wave_remaining == 0 and wave_index + 1 < len(waves):
                        wave_index += 1
                        active_color, wave_remaining = queue_wave(wave_index)
    except KeyboardInterrupt:
        print("\n  [interrupt] stopping workers; finished cells are saved")
    finally:
        # A worker reports its cache tally AFTER the last cell of its garment,
        # so those messages are still in flight when the cell counter hits the
        # target and the loop exits. Drain them, or the hit rate always reads 0.
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                tail = res_q.get(timeout=0.2)
            except queuelib.Empty:
                if not any(p.is_alive() for p in procs):
                    break
                continue
            if tail["kind"] == "garment_done":
                hits += tail["hits"]
                misses += tail["misses"]
        jsonl.close()
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=20)

    if vocab_mismatch is not None:
        worker_id, got = vocab_mismatch
        missing = [g for g in expected_vlm_vocab if g not in got]
        extra = [g for g in got if g not in expected_vlm_vocab]
        raise SystemExit(
            f"  [error] worker {worker_id} would show the VLM a garment "
            f"vocabulary that is not the benchmark axis.\n"
            f"          missing: {missing}\n"
            f"          extra:   {extra}\n"
            "          fsiglip/collect_img_sam3.py must derive GARMENT_VOCAB "
            "from configs.config; an extra term that is a HYPERNYM of a real\n"
            "          category (e.g. 'wool coat' over 'long coat' and 'pea "
            "coat') silently invalidates every cell of both. No cell was "
            "measured.")

    elapsed = time.time() - t0
    looks = hits + misses
    print(f"\n  measured {done} cells in {elapsed:.0f}s "
          f"({elapsed / max(done, 1):.1f}s/cell across {len(gpus)} gpus)")
    print(f"  crop cache: {hits} hits / {looks} lookups "
          f"({100 * hits / max(looks, 1):.0f}% of SAM3+scoring avoided)")
    print(f"  -> {out_json}")
    if jsonl_path.exists():
        n_rows = sum(1 for _ in open(jsonl_path, encoding="utf-8"))
        print(f"  -> {jsonl_path} ({n_rows} rows, "
              f"{jsonl_path.stat().st_size / 1e6:.1f} MB)")

    write_reports(table, colors, rows, patterns, gusage, cusage, args)
    if args.save_top > 0:
        root = args.out_dir / IMAGE_DIR
        files = sum(1 for f in root.rglob("*") if f.is_file()) if root.exists() else 0
        dirs = sum(1 for f in root.rglob("*") if f.is_dir()) if root.exists() else 0
        print(f"\n  top-{args.save_top} crops: {files} files in {dirs} folders "
              f"under {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
