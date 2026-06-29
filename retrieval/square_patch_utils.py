#!/usr/bin/env python3
"""Adaptive square patch tiling utilities for retrieval diagnostics.

This module keeps each patch square while letting the number of patches adapt to
image aspect ratio.

Interpretation of `grid`:
  grid = number of square patches along the shorter image side.

Example:
  image 800x1200, grid=5 -> patch_size≈160 -> about 5 columns x 8 rows.
  image 1200x800, grid=5 -> patch_size≈160 -> about 8 columns x 5 rows.

Important design choice:
  This utility does NOT do background removal or heuristic tile filtering by
  default. For local pattern coverage, dropping tiles before scoring can hide
  exactly the small patterned region we are trying to recover. The QwenEmb score
  should see all square patches unless the caller explicitly sets max_tiles > 0.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image


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
    """Create square patch tiles whose count adapts to image aspect ratio.

    Args match the old `make_pattern_tiles` signature so this can be monkey-patched
    into existing collectors.

    `grid` means the target number of square patches along the shorter side.
    `max_tiles <= 0` means score all generated square patches.

    `white_thresh` and `min_std` are accepted only for backward-compatible CLI
    signatures. They are intentionally ignored here.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    grid = max(1, int(grid))

    short_side = max(1, min(w, h))
    patch_size = max(1, int(math.ceil(short_side / grid)))
    patch_size = min(patch_size, w, h)
    overlap = max(0.0, min(float(overlap), 0.9))
    stride = max(1, int(round(patch_size * (1.0 - overlap))))

    xs = _positions(w, patch_size, stride)
    ys = _positions(h, patch_size, stride)

    tile_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    serial = 0
    for y in ys:
        for x in xs:
            if max_tiles and max_tiles > 0 and len(paths) >= int(max_tiles):
                return paths
            tile = img.crop((x, y, x + patch_size, y + patch_size))
            out = tile_dir / f"tile_{serial:03d}__xy_{x}_{y}__s_{patch_size}.jpg"
            if save_tiles or not out.exists():
                tile.save(out, quality=92)
            paths.append(out)
            serial += 1
    return paths
