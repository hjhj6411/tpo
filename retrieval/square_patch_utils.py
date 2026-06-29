#!/usr/bin/env python3
"""Adaptive square patch tiling utilities for retrieval diagnostics.

The previous grid splitter divided width and height by the same grid count, which
creates rectangular patches for portrait product images. This module keeps each
patch square while letting the number of patches adapt to the image aspect ratio.

Interpretation of `grid`:
  grid = number of square patches along the shorter image side.

Example:
  image 800x1200, grid=5 -> patch_size≈160 -> about 5 columns x 8 rows.
  image 1200x800, grid=5 -> patch_size≈160 -> about 8 columns x 5 rows.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageStat


def likely_non_background_tile(tile: Image.Image, white_thresh: float, min_std: float) -> bool:
    rgb = tile.convert("RGB")
    stat = ImageStat.Stat(rgb)
    mean = sum(stat.mean) / 3.0
    std = sum(stat.stddev) / 3.0
    if mean >= white_thresh and std < 35.0:
        return False
    if std < min_std:
        return False
    return True


def _positions(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    positions = list(range(0, max(1, length - patch_size + 1), stride))
    last = length - patch_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def make_adaptive_square_pattern_tiles(
    image_path: Path,
    grid: int,
    max_tiles: int,
    tile_dir: Path,
    save_tiles: bool,
    white_thresh: float,
    min_std: float,
    overlap: float = 0.0,
) -> list[Path]:
    """Create square patch tiles whose count adapts to the image aspect ratio.

    Args match the old `make_pattern_tiles` signature so this can be monkey-patched
    into existing collectors. `grid` now means the target number of square patches
    along the shorter side, not rows=cols.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    grid = max(1, int(grid))
    max_tiles = max(1, int(max_tiles))

    short_side = max(1, min(w, h))
    patch_size = max(1, int(math.ceil(short_side / grid)))
    patch_size = min(patch_size, w, h)
    overlap = max(0.0, min(float(overlap), 0.9))
    stride = max(1, int(round(patch_size * (1.0 - overlap))))

    xs = _positions(w, patch_size, stride)
    ys = _positions(h, patch_size, stride)

    candidates: list[tuple[float, int, int, Image.Image]] = []
    serial = 0
    for y in ys:
        for x in xs:
            tile = img.crop((x, y, x + patch_size, y + patch_size))
            if not likely_non_background_tile(tile, white_thresh=white_thresh, min_std=min_std):
                serial += 1
                continue
            stat = ImageStat.Stat(tile)
            variance_score = float(sum(stat.stddev) / 3.0)
            # Prefer non-background / textured patches, while keeping stable order as tie-breaker.
            candidates.append((variance_score, serial, x, y, tile))
            serial += 1

    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected = candidates[:max_tiles]

    tile_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for rank, (_score, serial_idx, x, y, tile) in enumerate(selected):
        out = tile_dir / f"tile_{rank:03d}__xy_{x}_{y}__s_{patch_size}.jpg"
        if save_tiles or not out.exists():
            tile.save(out, quality=92)
        paths.append(out)
    return paths
