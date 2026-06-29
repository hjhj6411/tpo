#!/usr/bin/env python3
"""Square-patch entrypoint for QwenEmb-only collection.

This wrapper keeps the main collector untouched while forcing adaptive square
patches. It also makes `--pattern-max-tiles 0` mean all generated patches.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval.square_patch_utils import make_adaptive_square_pattern_tiles
import retrieval.collect_topk_qwenemb_only_gallery as collector


def main() -> None:
    collector.make_pattern_tiles = make_adaptive_square_pattern_tiles
    collector.main()


if __name__ == "__main__":
    main()
