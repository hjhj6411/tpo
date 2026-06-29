#!/usr/bin/env python3
"""Backward-compatible entrypoint for square-patch QwenEmb collection.

`collect_topk_qwenemb_only_gallery.py` now uses adaptive square patches by default.
This wrapper is kept so older commands that call the explicit square collector keep
working.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import retrieval.collect_topk_qwenemb_only_gallery as collector


if __name__ == "__main__":
    collector.main()
