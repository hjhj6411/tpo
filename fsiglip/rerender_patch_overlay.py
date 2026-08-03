#!/usr/bin/env python3
"""Re-draw patch overlays for collector outputs that already exist on disk.

The collectors used to draw the patch grid on the RAW crop with a red tint over
the mask, while the tiles they actually scored are cut from the MASKED crop
(background replaced by `patch_mask_fill`). The overlay therefore showed colours
no scorer ever saw -- a blue tank top reads purple under the tint -- which is
useless for auditing the colour axis.

The collectors now draw on the masked crop. This script applies the same fix to
runs already collected, without re-running SAM3 or the VLM: every rank dir keeps
`crop.jpg`, `mask.png` and the per-tile geometry inside `*_patch_scores.json`,
which is everything the overlay needs.

Usage:
  python fsiglip/rerender_patch_overlay.py data/retrieval/sam3vlmfix
  python fsiglip/rerender_patch_overlay.py <root> --label pred
  python fsiglip/rerender_patch_overlay.py <root> --suffix _masked   # keep originals
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# (tile dir, scores filename, overlay filename) per axis, matching the collectors' layout.
AXES = [
    ("patches", "patch_scores.json", "patch_overlay.jpg"),
    ("color_patches", "color_patch_scores.json", "color_patch_overlay.jpg"),
]

# tile_0012__x204_y204_s102_cov1.000.jpg -- the collectors encode the full tile
# geometry in the filename, which is the ONLY source that survives an axis whose
# scoring was skipped.
TILE_RE = re.compile(r"^tile_(\d+)__x(\d+)_y(\d+)_s(\d+)_cov([0-9.]+)\.jpg$")


def font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def fixed_patch_fill(fill_mode: str) -> tuple[int, int, int]:
    if fill_mode == "white":
        return (255, 255, 255)
    if fill_mode == "black":
        return (0, 0, 0)
    return (127, 127, 127)


def masked_crop_preview(crop: Image.Image, crop_mask: np.ndarray, fill_mode: str) -> Image.Image:
    arr = np.asarray(crop.convert("RGB"), dtype=np.uint8)
    mask = crop_mask.astype(bool)

    if fill_mode == "local_mean" and mask.any():
        fill = tuple(int(round(v)) for v in arr[mask].mean(axis=0))
    else:
        fill = fixed_patch_fill(fill_mode)

    out = np.full(arr.shape, fill, dtype=np.uint8)
    out[mask] = arr[mask]
    return Image.fromarray(out, mode="RGB")


def mask_contour(crop_mask: np.ndarray) -> np.ndarray:
    m = crop_mask.astype(bool)
    edge = np.zeros_like(m)
    vert = m[:-1, :] != m[1:, :]
    horz = m[:, :-1] != m[:, 1:]
    edge[:-1, :] |= vert
    edge[1:, :] |= vert
    edge[:, :-1] |= horz
    edge[:, 1:] |= horz
    return edge & m


def draw_patch_overlay(
    crop: Image.Image,
    crop_mask: np.ndarray,
    patches: list[dict[str, Any]],
    out_path: Path,
    fill_mode: str,
    label: str = "coverage",
) -> None:
    out = masked_crop_preview(crop, crop_mask, fill_mode).convert("RGBA")
    arr = np.array(out)
    arr[mask_contour(crop_mask)] = np.array([255, 0, 0, 255], dtype=np.uint8)
    out = Image.fromarray(arr, mode="RGBA")

    draw = ImageDraw.Draw(out)
    f = font(12)
    for p in patches:
        x = int(p["x"])
        y = int(p["y"])
        s = int(p["size"])

        if label == "pred" and p.get("pred"):
            colour = (0, 255, 0, 255) if p.get("hit") else (255, 64, 0, 255)
            text = str(p["pred"])
        else:
            colour = (255, 255, 0, 255)
            text = f"{float(p['mask_coverage']):.2f}"

        draw.rectangle([x, y, x + s, y + s], outline=colour, width=2)
        draw.text((x + 3, y + 3), text, fill=colour, font=f)

    out.convert("RGB").save(out_path, quality=95)


def patches_from_dir(patch_dir: Path) -> list[dict[str, Any]]:
    """Recover tile geometry from the saved tile filenames.

    Deliberately NOT read from `tile_scores`: an axis whose target is absent
    scores as `missing_pattern_target` with an empty tile_scores, even though
    make_patch_files() wrote a full grid of tiles. Keying the overlay off the
    scores drew zero boxes for exactly those ranks.
    """
    rows = []
    for p in sorted(patch_dir.glob("tile_*.jpg")):
        m = TILE_RE.match(p.name)
        if not m:
            continue
        idx, x, y, s, cov = m.groups()
        rows.append({
            "index": int(idx),
            "x": int(x),
            "y": int(y),
            "size": int(s),
            "mask_coverage": float(cov),
        })
    return rows


def rank_dirs(root: Path):
    for crop_path in sorted(root.rglob("crop.jpg")):
        yield crop_path.parent


def rerender(rank_dir: Path, label: str, suffix: str) -> int:
    crop_path = rank_dir / "crop.jpg"
    mask_path = rank_dir / "mask.png"
    if not mask_path.exists():
        return 0

    crop = Image.open(crop_path).convert("RGB")
    crop_mask = np.asarray(Image.open(mask_path).convert("L")) > 127
    if crop_mask.shape != (crop.height, crop.width):
        print(f"  [skip] {rank_dir}: mask {crop_mask.shape} != crop {(crop.height, crop.width)}")
        return 0

    # Every tile row carries the fill it was cut with; the score.json copy is the
    # fallback for runs whose axes produced no tiles at all.
    fill_mode = ""
    score_path = rank_dir / "score.json"
    if score_path.exists():
        fill_mode = json.loads(score_path.read_text()).get("patch_mask_fill") or ""
    if not fill_mode:
        # Guessing wrong here silently repaints the background, so say so.
        print(f"  [warn] {rank_dir}: no patch_mask_fill recorded, assuming the "
              f"collector default 'local_mean'")
        fill_mode = "local_mean"

    written = 0
    for dir_name, scores_name, overlay_name in AXES:
        patch_dir = rank_dir / dir_name
        if not patch_dir.is_dir():
            continue

        patches = patches_from_dir(patch_dir)
        if not patches:
            continue

        # Scores are a label source only, joined by tile index, and are absent
        # whenever the axis was skipped. Geometry never depends on them.
        scores_path = rank_dir / scores_name
        tiles = []
        if scores_path.exists():
            tiles = json.loads(scores_path.read_text()).get("tile_scores") or []
        by_index = {int(t["index"]): t for t in tiles if "index" in t}
        for p in patches:
            t = by_index.get(p["index"])
            if t:
                p["pred"] = t.get("pred")
                p["hit"] = t.get("hit")
                # Full-precision coverage; the filename's 3dp value would round
                # a second time on the way to the 2dp label and drift by 0.01.
                if t.get("mask_coverage") is not None:
                    p["mask_coverage"] = float(t["mask_coverage"])

        axis_fill = tiles[0].get("mask_fill") if tiles else ""
        out_path = rank_dir / (Path(overlay_name).stem + suffix + ".jpg")
        draw_patch_overlay(crop, crop_mask, patches, out_path, axis_fill or fill_mode, label)
        written += 1

    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="collector output root, or a single rank dir")
    ap.add_argument("--label", choices=["coverage", "pred"], default="coverage",
                    help="box label: mask coverage (collector parity), or the tile's "
                         "predicted class coloured by hit/miss")
    ap.add_argument("--suffix", default="",
                    help="append to the overlay filename instead of overwriting "
                         "(e.g. --suffix _masked)")
    args = ap.parse_args()

    dirs = [args.root] if (args.root / "crop.jpg").exists() else list(rank_dirs(args.root))
    if not dirs:
        raise SystemExit(f"no rank dirs (crop.jpg) found under {args.root}")

    total = 0
    for i, d in enumerate(dirs, 1):
        total += rerender(d, args.label, args.suffix)
        if i % 100 == 0:
            print(f"  {i}/{len(dirs)} rank dirs, {total} overlays")

    print(f"done: {total} overlays over {len(dirs)} rank dirs")


if __name__ == "__main__":
    main()
