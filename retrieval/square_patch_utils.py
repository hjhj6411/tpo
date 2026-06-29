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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class SquarePatchRecord:
    """Metadata for one generated square patch."""

    path: Path
    x: int
    y: int
    size: int
    index: int
    image_width: int
    image_height: int

    def to_json(self) -> dict[str, Any]:
        obj = asdict(self)
        obj["path"] = str(self.path)
        return obj


def _positions(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    positions = list(range(0, max(1, length - patch_size + 1), stride))
    last = length - patch_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def _resolve_patch_size(width: int, height: int, grid: int, patch_size: int | None) -> int:
    if patch_size is not None and int(patch_size) > 0:
        return max(1, min(int(patch_size), width, height))
    grid = max(1, int(grid))
    short_side = max(1, min(width, height))
    return max(1, min(int(math.ceil(short_side / grid)), width, height))


def make_adaptive_square_patch_records(
    image_path: Path,
    grid: int,
    max_tiles: int,
    tile_dir: Path,
    save_tiles: bool,
    white_thresh: float = 242.0,
    min_std: float = 8.0,
    overlap: float = 0.0,
    patch_size: int | None = None,
) -> list[SquarePatchRecord]:
    """Create square patch tiles and return paths plus coordinates.

    `grid` means the target number of square patches along the shorter side.
    `patch_size`, when given, overrides `grid` and uses a fixed square side.
    `max_tiles <= 0` means score all generated square patches.

    `white_thresh` and `min_std` are accepted only for backward-compatible CLI
    signatures. They are intentionally ignored here; foreground/background
    decisions should be done by a separate scoring or mask stage.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    side = _resolve_patch_size(w, h, grid=grid, patch_size=patch_size)
    overlap = max(0.0, min(float(overlap), 0.9))
    stride = max(1, int(round(side * (1.0 - overlap))))

    xs = _positions(w, side, stride)
    ys = _positions(h, side, stride)

    tile_dir.mkdir(parents=True, exist_ok=True)
    records: list[SquarePatchRecord] = []
    serial = 0
    for y in ys:
        for x in xs:
            if max_tiles and max_tiles > 0 and len(records) >= int(max_tiles):
                return records
            tile = img.crop((x, y, x + side, y + side))
            out = tile_dir / f"tile_{serial:03d}__xy_{x}_{y}__s_{side}.jpg"
            if save_tiles or not out.exists():
                tile.save(out, quality=92)
            records.append(SquarePatchRecord(
                path=out,
                x=int(x),
                y=int(y),
                size=int(side),
                index=int(serial),
                image_width=int(w),
                image_height=int(h),
            ))
            serial += 1
    return records


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
    records = make_adaptive_square_patch_records(
        image_path=image_path,
        grid=grid,
        max_tiles=max_tiles,
        tile_dir=tile_dir,
        save_tiles=save_tiles,
        white_thresh=white_thresh,
        min_std=min_std,
        overlap=overlap,
    )
    return [r.path for r in records]
