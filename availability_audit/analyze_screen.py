#!/usr/bin/env python3
"""Offline analysis of the screening's per-candidate log. No server, no GPU.

Reads only `screen_sam3_candidates.jsonl` — the raw per-candidate rows written
by `screen_sam3.py` — and answers three questions the summary table cannot.

Rows whose `mask_status` is not "ok" were never scored on any axis; they are
EXCLUDED from every count here rather than treated as zeros, because a SAM3
segmentation failure is not evidence that the garment is absent from the corpus
(README.md is explicit about keeping those two apart). The mask-failure rate is
reported separately so it cannot be forgotten.

1. WHAT DOES THE TABLE LOOK LIKE AT A STRICTER BAR?
   The published heatmaps count candidates with score > 0, which is still loose:
   a single patch winning the argmax on both attribute axes is enough, once the
   --attr-min-score floor is cleared. This redraws the same
   heatmaps at each --thresholds value, so `screen_sam3_black_t0.3.png` sits next
   to `screen_sam3_black_t0.png` and the difference is visible.

2. DOES THE CELL RANKING SURVIVE THAT BAR?
   Spearman correlation between the cell ranking at threshold 0 and at each
   stricter threshold, overall and per colour, plus the cells that move most.
   High correlation means the loose table is still usable as a filter. Low
   correlation means `nonzero > 0` was mostly noise and the ordering it implies
   should not be trusted.

3. IS `striped` STILL PICKING UP CONSTRUCTION LINES?
   For every (colour, garment): striped_nonzero / max(polka_dot, leopard). Seams,
   quilting channels and knit ribs read as parallel lines. The argmax scorer is
   supposed to suppress this — a seam patch has to beat "plain solid-colored" to
   count as a stripe — so on SAM3 data this is a REGRESSION CHECK, not a hunt:
   outerwear towering over the other patterns would mean the argmax is not doing
   its job. On the old pre-SAM3 logs the same ratio flagged 61 of 260 pairs,
   every leather_jacket among them; that is the baseline to beat.

    python -m availability_audit.analyze_screen --thresholds 0,0.3,0.5

This script only measures and reports. It does not filter anything, write a
blocklist, or touch construction/ — reading the tables and deciding what to
exclude is a human's job.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from availability_audit.screen_sam3 import write_labeled_heatmap  # renderer only

# the artifact hunt compares striped against the other two non-solid,
# non-directional patterns; solid has no lines to confuse and the remaining
# patterns are not in the "parallel lines" family
ARTIFACT_TARGET = "striped"
ARTIFACT_CONTROLS = ("polka_dot", "leopard")


# ── loading ───────────────────────────────────────────────────────────────
def load_scores(path: Path) -> dict[tuple[str, str, str], list[float]]:
    """(color, garment, pattern) -> every candidate score for that cell.

    Rows are appended per cell, so a cell measured twice (a resumed run that was
    forced, say) would appear twice. Keep the LAST block per cell: re-measuring
    is how a cell gets corrected, and averaging a stale block into a fresh one
    would silently blend two different runs.
    """
    if not path.exists():
        raise SystemExit(
            f"  [error] {path} not found.\n"
            f"          It is written by screen_sam3.py. Note that\n"
            f"          screen_gpc_candidates.jsonl is NOT a substitute: those rows\n"
            f"          come from the pre-SAM3 scorer and are not comparable.")

    cells: dict[tuple[str, str, str], list[float]] = {}
    order: list[tuple[str, str, str]] = []
    prev_key = None
    bad = 0
    skipped: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                key = (r["color"], r["garment"], r["pattern"])
            except Exception:
                bad += 1
                continue
            if r.get("mask_status") not in (None, "ok"):
                # never judged on its axes — counting it as 0 would blame the
                # corpus for a segmentation failure
                skipped[str(r.get("mask_status"))] = skipped.get(str(r.get("mask_status")), 0) + 1
                continue
            if key != prev_key:
                # a new contiguous block for this cell supersedes any earlier one
                cells[key] = []
                if key not in order:
                    order.append(key)
                prev_key = key
            cells[key].append(float(r.get("score") or 0.0))
    if bad:
        print(f"  [warn] skipped {bad} unparsable lines in {path.name}")
    if skipped:
        tot = sum(skipped.values())
        print(f"  [note] {tot} candidates excluded (never scored): "
              + ", ".join(f"{k}={v}" for k, v in sorted(skipped.items(), key=lambda kv: -kv[1])))
    return cells


def count_at(scores: list[float], t: float) -> int:
    """Same rule the screening uses for `nonzero_at`: > 0 at t=0, >= t above."""
    return sum(1 for s in scores if (s > 0.0 if t <= 0.0 else s >= t))


# ── spearman without scipy ────────────────────────────────────────────────
def _ranks(vals: list[float]) -> list[float]:
    """Average ranks, so the many tied cells (lots of zeros) are handled right."""
    idx = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0.0] * len(vals)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and vals[idx[j + 1]] == vals[idx[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[idx[k]] = avg
        i = j + 1
    return out


def spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3:
        return None
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    if da == 0 or db == 0:
        return None      # every cell tied at this threshold; rank is undefined
    return num / (da * db)


# ── report 1: heatmaps per threshold ──────────────────────────────────────
def redraw_heatmaps(cells: dict, colors: list[str], garments: list[str],
                    patterns: list[str], thresholds: list[float],
                    out_dir: Path, top_k: int) -> list[Path]:
    written = []
    for t in thresholds:
        for c in colors:
            grid = [[count_at(cells.get((c, g, p), []), t) for p in patterns]
                    for g in garments]
            png = out_dir / f"screen_sam3_{c}_t{('%g' % t)}.png"
            write_labeled_heatmap(
                grid, garments, patterns, png, top_k,
                title=f"COLOR = {c}  |  candidates with score >= {t:g}",
                xlabel="pattern", ylabel="garment")
            written.append(png)
    return written


# ── report 2: rank stability ──────────────────────────────────────────────
def rank_stability(cells: dict, colors: list[str], thresholds: list[float],
                   base: float, top_n: int = 10) -> None:
    keys = sorted(cells)
    base_vals = [float(count_at(cells[k], base)) for k in keys]

    head = f"{'threshold':>10}{'spearman vs t=%g' % base:>22}{'cells':>8}{'nonzero cells':>16}"
    print("\n" + "=" * len(head))
    print("  RANK STABILITY — does the cell ordering survive a stricter bar?")
    print("=" * len(head))
    print(head)
    print("-" * len(head))
    for t in thresholds:
        vals = [float(count_at(cells[k], t)) for k in keys]
        rho = spearman(base_vals, vals)
        print(f"{('%g' % t):>10}{('-' if rho is None else '%.3f' % rho):>22}"
              f"{len(keys):>8}{sum(1 for v in vals if v > 0):>16}")
    print("=" * len(head))

    print("\n  per colour:")
    print(f"    {'color':<10}" + "".join(f"{('t=%g' % t):>10}" for t in thresholds))
    for c in colors:
        ck = [k for k in keys if k[0] == c]
        if len(ck) < 3:
            continue
        bv = [float(count_at(cells[k], base)) for k in ck]
        line = f"    {c:<10}"
        for t in thresholds:
            rho = spearman(bv, [float(count_at(cells[k], t)) for k in ck])
            line += f"{('-' if rho is None else '%.3f' % rho):>10}"
        print(line)

    # which cells were propped up by the loose bar
    for t in thresholds:
        if t == base:
            continue
        vals = [float(count_at(cells[k], t)) for k in keys]
        rb, rt = _ranks(base_vals), _ranks(vals)
        moved = sorted(range(len(keys)), key=lambda i: -(rb[i] - rt[i]))[:top_n]
        print(f"\n  biggest rank DROPS from t={base:g} to t={t:g} "
              f"(high only at the loose bar):")
        print(f"    {'cell':<44}{'n@%g' % base:>8}{'n@%g' % t:>8}{'rank drop':>11}")
        for i in moved:
            c, g, p = keys[i]
            if rb[i] - rt[i] <= 0:
                continue
            print(f"    {p + ' ' + c + ' ' + g:<44}{int(base_vals[i]):>8}"
                  f"{int(vals[i]):>8}{rb[i] - rt[i]:>11.0f}")


# ── report 3: striped artifact ────────────────────────────────────────────
def striped_artifact(cells: dict, colors: list[str], garments: list[str],
                     t: float, flag_ratio: float, agg: str = "max") -> list[tuple]:
    """striped / agg(controls) per (colour, garment). High = suspected artifact.

    The default aggregate is max, not mean or min: the claim being tested is
    that striped is anomalously high, so dividing by the LARGER control is the
    conservative choice — it shrinks the ratio and flags fewer cells. This
    matters, because the two controls can disagree sharply. black fleece_jacket
    is striped 56 / polka_dot 6 / leopard 60: read against polka_dot alone that
    is a 9x artifact, read against leopard it is unremarkable. `--control-agg
    min` reproduces the optimistic reading; both raw counts are always printed
    so the disagreement is visible in the table either way.
    """
    fn = {"max": max, "min": min, "mean": lambda v: sum(v) / len(v)}[agg]
    rows = []
    for c in colors:
        for g in garments:
            s = count_at(cells.get((c, g, ARTIFACT_TARGET), []), t)
            ctrl = [count_at(cells.get((c, g, p), []), t) for p in ARTIFACT_CONTROLS]
            denom = fn(ctrl)
            if s == 0 and denom == 0:
                continue                      # nothing retrieved at all; not a signal
            ratio = float("inf") if denom == 0 else s / denom
            rows.append((ratio, c, g, s, ctrl))
    rows.sort(key=lambda r: -r[0])

    flagged = [r for r in rows if r[0] >= flag_ratio]
    head = (f"{'ratio':>8}  {'color':<9}{'garment':<18}"
            f"{'striped':>9}{'polka_dot':>11}{'leopard':>9}")
    print("\n" + "=" * len(head))
    print(f"  STRIPED ARTIFACT SCREEN  (t={t:g}, flag at ratio >= {flag_ratio:g})")
    print(f"  ratio = striped / {agg}(polka_dot, leopard); a high ratio suggests")
    print("  seams, quilting or ribs being read as stripes, not availability.")
    print("=" * len(head))
    print(head)
    print("-" * len(head))
    for ratio, c, g, s, ctrl in rows:
        mark = " <<" if ratio >= flag_ratio else ""
        rs = "inf" if ratio == float("inf") else f"{ratio:.2f}"
        print(f"{rs:>8}  {c:<9}{g:<18}{s:>9}{ctrl[0]:>11}{ctrl[1]:>9}{mark}")
    print("=" * len(head))
    print(f"  {len(flagged)} of {len(rows)} (colour, garment) pairs flagged")

    if flagged:
        by_g = defaultdict(int)
        for _, _, g, _, _ in flagged:
            by_g[g] += 1
        print("\n  flagged garments (how many colours each):")
        for g, n in sorted(by_g.items(), key=lambda kv: -kv[1]):
            print(f"    {g:<20}{n:>3} / {len(colors)} colours")
    return flagged


# ── main ──────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jsonl", type=Path,
                   default=here / "screen_sam3_candidates.jsonl")
    p.add_argument("--out-dir", type=Path, default=here)
    p.add_argument("--thresholds", default="0,0.3,0.5",
                   help="comma-separated score thresholds; the FIRST is the "
                        "baseline the others are compared against")
    p.add_argument("--flag-ratio", type=float, default=3.0,
                   help="striped/control ratio at which a cell is flagged")
    p.add_argument("--control-agg", default="max", choices=("max", "min", "mean"),
                   help="how the two control patterns are combined into the "
                        "denominator; max (default) is the conservative reading")
    p.add_argument("--artifact-threshold", type=float, default=0.0,
                   help="threshold used for the striped ratio (default 0, i.e. "
                        "the same counts as the published heatmaps)")
    p.add_argument("--top-k", type=int, default=100,
                   help="heatmap colour scale maximum; must match the run")
    p.add_argument("--no-heatmaps", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    if not thresholds:
        raise SystemExit("  [error] --thresholds is empty")

    cells = load_scores(args.jsonl)
    if not cells:
        raise SystemExit(f"  [error] no usable rows in {args.jsonl}")

    # axis order comes from the data, not from config: the log may cover only a
    # subset of colours (--colors) and re-deriving it here would invent columns
    colors = sorted({k[0] for k in cells})
    garments = sorted({k[1] for k in cells})
    patterns = sorted({k[2] for k in cells})
    n_scores = sum(len(v) for v in cells.values())

    print("=" * 74)
    print(f"  offline analysis of {args.jsonl.name}")
    print(f"  {len(cells)} cells, {n_scores} candidate rows "
          f"({len(colors)} colors x {len(garments)} garments x {len(patterns)} patterns)")
    print(f"  thresholds: {', '.join('%g' % t for t in thresholds)} "
          f"(baseline t={thresholds[0]:g})")
    print("=" * 74)

    if not args.no_heatmaps:
        written = redraw_heatmaps(cells, colors, garments, patterns, thresholds,
                                  args.out_dir, args.top_k)
        print(f"\n  {len(written)} heatmaps written to {args.out_dir}")
        for p in written[:3]:
            print(f"    {p.name}")
        if len(written) > 3:
            print(f"    ... and {len(written) - 3} more")

    rank_stability(cells, colors, thresholds, thresholds[0])

    if ARTIFACT_TARGET in patterns and any(p in patterns for p in ARTIFACT_CONTROLS):
        striped_artifact(cells, colors, garments, args.artifact_threshold,
                         args.flag_ratio, args.control_agg)
    else:
        print(f"\n  [skip] striped artifact screen needs {ARTIFACT_TARGET} and one "
              f"of {list(ARTIFACT_CONTROLS)} in the log; found {patterns}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
