#!/usr/bin/env python3
"""Serve the cell-level SAM3 crop annotator on 127.0.0.1.

The tool reads a snapshot of ``screen_sam3`` outputs and option plans.  Its
only image mutation is copying a selected crop into ``images_final``.  Human
decisions are durably appended to JSONL one record at a time, and the derived
attribute library is replaced atomically after every decision.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mimetypes
import os
import random
import re
import shutil
import sys
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse


REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_HTML = Path(__file__).resolve().parent / "static" / "annotator.html"
ANNOTATION_FILE = "cell_annotations.jsonl"
LIBRARY_FILE = "attribute_library.json"
FINAL_IMAGE_DIR = "images_final"
EXCLUSION_REASONS = (
    "no_such_garment",
    "pattern_absent",
    "color_mismatch",
    "structural_texture",
    "mask_failure",
    "multi_person_or_unclear",
    "other",
)
URL_KEY_RE = re.compile(r"__([0-9a-f]{20})(?:\.[^.]+)$")
PART_RE = re.compile(r"screen_sam3_v\d+_part\d+")


def cell_key(color: str, garment: str, pattern: str) -> str:
    return f"{color}|{garment}|{pattern}"


def stable_seed(base_seed: int, variant: str, key: str, pass_tag: str) -> int:
    raw = f"{base_seed}|{variant}|{key}|{pass_tag}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & 0x7FFFFFFF


def sha1_url(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:20]


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_separator = False
    if path.is_file() and path.stat().st_size:
        with path.open("rb") as existing:
            existing.seek(-1, os.SEEK_END)
            needs_separator = existing.read(1) != b"\n"
    with path.open("a", encoding="utf-8") as handle:
        if needs_separator:
            # Preserve a crash-truncated final line as history, but do not let
            # it consume the next valid JSON object.
            handle.write("\n")
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_annotation_history(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    if not path.exists():
        return records, warnings
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.append(
                    f"{path}: ignored malformed line {line_no}: {exc.msg}"
                )
                continue
            if isinstance(row, dict):
                records.append(row)
            else:
                warnings.append(f"{path}: ignored non-object line {line_no}")
    return records, warnings


def plan_usage(plan_path: Path) -> tuple[
    Counter[str], set[str], list[str], list[str], list[str], dict[str, int]
]:
    usage: Counter[str] = Counter()
    needed: set[str] = set()
    colors: list[str] = []
    garments: list[str] = []
    patterns: list[str] = []
    missing_pattern_pairs: set[tuple[str, str]] = set()
    missing_pattern_options = 0

    def remember(value: str | None, values: list[str]) -> None:
        if value and value not in values:
            values.append(value)

    with plan_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                options = (json.loads(line).get("options") or {}).values()
            except (json.JSONDecodeError, AttributeError) as exc:
                raise SystemExit(f"[error] invalid plan line {line_no}: {exc}") from exc
            for option in options:
                attrs = option.get("attributes") or {}
                color = attrs.get("color")
                garment = attrs.get("garment_category")
                pattern = attrs.get("pattern")
                remember(color, colors)
                remember(garment, garments)
                remember(pattern, patterns)
                if color and garment and pattern:
                    key = cell_key(color, garment, pattern)
                    usage[key] += 1
                    needed.add(key)
                elif color and garment and not pattern:
                    missing_pattern_pairs.add((color, garment))
                    missing_pattern_options += 1
    stats = {
        "needed_cells": len(needed),
        "one_use_cells": sum(count == 1 for count in usage.values()),
        "missing_pattern_pairs": len(missing_pattern_pairs),
        "missing_pattern_options": missing_pattern_options,
    }
    return usage, needed, colors, garments, patterns, stats


def config_signature(config: dict[str, Any]) -> str:
    ignored = {"colors_scored"}
    stable = {key: value for key, value in config.items() if key not in ignored}
    return json.dumps(stable, sort_keys=True, separators=(",", ":"))


def discover_screen_runs(screen_dir: Path) -> list[dict[str, Any]]:
    """Find direct, merged, and split screen snapshots under ``screen_dir``."""
    if not screen_dir.exists():
        raise SystemExit(f"[error] --screen-dir does not exist: {screen_dir}")
    candidates = [screen_dir / "screen_sam3.json"]
    candidates.extend(screen_dir.glob("*/screen_sam3.json"))
    candidates.extend(screen_dir.glob("*/*/screen_sam3.json"))
    runs = []
    seen: set[Path] = set()
    for json_path in candidates:
        json_path = json_path.resolve()
        if json_path in seen or not json_path.is_file():
            continue
        seen.add(json_path)
        root = json_path.parent
        config_path = root / "screen_sam3_config.json"
        candidates_path = root / "screen_sam3_candidates.jsonl"
        try:
            table = read_json(json_path)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[warn] ignoring unreadable snapshot {json_path}: {exc}",
                  file=sys.stderr)
            continue
        config = {}
        if config_path.is_file():
            try:
                config = read_json(config_path)
            except (OSError, json.JSONDecodeError) as exc:
                print(f"[warn] ignoring bad config {config_path}: {exc}",
                      file=sys.stderr)
        flat: dict[str, dict[str, Any]] = {}
        for color, by_garment in table.items():
            if not isinstance(by_garment, dict):
                continue
            for garment, by_pattern in by_garment.items():
                if not isinstance(by_pattern, dict):
                    continue
                for pattern, summary in by_pattern.items():
                    if isinstance(summary, dict):
                        flat[cell_key(color, garment, pattern)] = {
                            "color": color,
                            "garment": garment,
                            "pattern": pattern,
                            "summary": summary,
                        }
        runs.append({
            "root": root,
            "json_path": json_path,
            "candidates_path": candidates_path,
            "config": config,
            "signature": config_signature(config),
            "cells": flat,
            "n_cells": len(flat),
        })
    if not runs:
        raise SystemExit(
            f"[error] no readable screen_sam3.json found under {screen_dir}"
        )
    return runs


def choose_screen_snapshot(
    runs: list[dict[str, Any]],
    plan_colors: list[str],
    plan_garments: list[str],
    plan_patterns: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Choose the compatible run group with the greatest distinct-cell coverage.

    A complete merged run wins over its component shards.  When there is no
    merged run, compatible part directories are combined cell-by-cell.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[run["signature"]].append(run)

    choices = []
    for signature, group in groups.items():
        # Prefer the most complete physical run for duplicate cells so its
        # relative crop path resolves within that same directory.
        ordered = sorted(group, key=lambda item: item["n_cells"], reverse=True)
        merged: dict[str, dict[str, Any]] = {}
        for run in ordered:
            for key, cell in run["cells"].items():
                if key not in merged:
                    merged[key] = dict(cell, screen_root=run["root"],
                                       candidates_path=run["candidates_path"])
        choices.append((len(merged), signature, merged, ordered))

    _, _, cells, selected_group = max(choices, key=lambda item: item[0])
    used_roots = sorted(
        {str(cell["screen_root"]) for cell in cells.values()}
    )
    richest_config = max(
        (run["config"] for run in selected_group),
        key=lambda cfg: (
            len(cfg.get("colors_scored") or []),
            len(cfg.get("garment_vocabulary") or []),
            len(cfg.get("patterns_scored") or []),
        ),
        default={},
    )
    colors = list(richest_config.get("colors_scored") or plan_colors)
    garments = list(richest_config.get("garment_vocabulary") or plan_garments)
    patterns = list(richest_config.get("patterns_scored") or plan_patterns)
    # Plans describe the intended benchmark universe even if a live screen
    # snapshot currently contains only one color wave.
    for value in plan_colors:
        if value not in colors:
            colors.append(value)
    for value in plan_garments:
        if value not in garments:
            garments.append(value)
    for value in plan_patterns:
        if value not in patterns:
            patterns.append(value)
    expected = len(colors) * len(garments) * len(patterns)
    meta = {
        "screen_sources": used_roots,
        "screen_cells_loaded": len(cells),
        "screen_cells_expected": expected,
        "partial": len(cells) < expected,
        "colors": colors,
        "garments": garments,
        "patterns": patterns,
        "config": richest_config,
    }
    return cells, meta


def saved_url_key(saved: dict[str, Any]) -> str | None:
    if saved.get("url_key"):
        return str(saved["url_key"])
    match = URL_KEY_RE.search(str(saved.get("file") or ""))
    return match.group(1) if match else None


def load_candidate_metadata(cells: dict[str, dict[str, Any]]) -> dict[
    tuple[str, str], dict[str, Any]
]:
    """Read only rows corresponding to crops that can be shown."""
    wanted_by_file: dict[Path, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for key, cell in cells.items():
        path = Path(cell["candidates_path"])
        for saved in cell["summary"].get("saved_crops") or []:
            url_key = saved_url_key(saved)
            if url_key:
                wanted_by_file[path][key].add(url_key)

    metadata: dict[tuple[str, str], dict[str, Any]] = {}
    for path, wanted in wanted_by_file.items():
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = cell_key(
                    str(row.get("color") or ""),
                    str(row.get("garment") or ""),
                    str(row.get("pattern") or ""),
                )
                url_key = row.get("url_key")
                if url_key and url_key in wanted.get(key, set()):
                    metadata[(key, str(url_key))] = row
    return metadata


def resolve_source_urls(
    wanted_keys: set[str],
    candidate_meta: dict[tuple[str, str], dict[str, Any]],
) -> tuple[dict[str, str], str]:
    """Resolve SHA-1 URL keys through rows or the FashionSigLIP corpus."""
    found: dict[str, str] = {}
    for row in candidate_meta.values():
        url_key = row.get("url_key")
        url = row.get("source_url") or row.get("image_url") or row.get("url")
        if url_key and url:
            found[str(url_key)] = str(url)
    remaining = wanted_keys - set(found)
    if not remaining:
        return found, "candidate_jsonl"

    parquet = REPO_ROOT / "data" / "fashionsiglip_corpus" / "ids.parquet"
    if parquet.is_file():
        try:
            import pandas as pd  # existing corpus dependency; optional here

            frame = pd.read_parquet(parquet, columns=["url"])
            for raw in frame["url"]:
                if not isinstance(raw, str) or not raw:
                    continue
                key = sha1_url(raw)
                if key in remaining:
                    found[key] = raw
                    remaining.remove(key)
                    if not remaining:
                        break
            return found, f"{parquet} (SHA-1 lookup)"
        except Exception as exc:
            print(f"[warn] could not read {parquet}: {exc}; trying CSV",
                  file=sys.stderr)

    corpus_csv = REPO_ROOT / "corpus" / "amazon_fashion_urls.csv"
    if corpus_csv.is_file() and remaining:
        with corpus_csv.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                url = row.get("url")
                if not url:
                    continue
                key = sha1_url(url)
                if key in remaining:
                    found[key] = url
                    remaining.remove(key)
                    if not remaining:
                        break
        return found, f"{corpus_csv} (SHA-1 lookup)"
    return found, "unavailable"


def prepare_cells(
    raw_cells: dict[str, dict[str, Any]],
    usage: Counter[str],
    candidate_meta: dict[tuple[str, str], dict[str, Any]],
    source_urls: dict[str, str],
) -> dict[str, dict[str, Any]]:
    cells: dict[str, dict[str, Any]] = {}
    for key, raw in raw_cells.items():
        root = Path(raw["screen_root"]).resolve()
        summary = raw["summary"]
        candidates = []
        for saved in summary.get("saved_crops") or []:
            url_key = saved_url_key(saved)
            if not url_key:
                continue
            relative = Path(str(saved.get("file") or ""))
            crop_path = (root / relative).resolve()
            try:
                crop_path.relative_to(root)
                safe = True
            except ValueError:
                safe = False
            row = candidate_meta.get((key, url_key), {})
            candidates.append({
                "true_rank": int(saved.get("rank") or len(candidates) + 1),
                "score": float(saved.get("score") or row.get("score") or 0.0),
                "url_key": url_key,
                "crop_file": relative.as_posix(),
                "crop_path": crop_path if safe and crop_path.is_file() else None,
                "source_url": source_urls.get(url_key),
                "pattern_score": row.get("pattern_score"),
                "color_score": row.get("color_score"),
                "garment_score": row.get("garment_score"),
            })
        candidates.sort(key=lambda item: item["true_rank"])
        mask_status = summary.get("mask_status") or {}
        ok = int(mask_status.get("ok") or 0)
        n_candidates = int(summary.get("n_candidates") or 0)
        mask_failures = sum(
            int(count or 0)
            for status, count in mask_status.items()
            if status != "ok"
        )
        # Some early outputs did not preserve every status count.
        mask_failures = max(mask_failures, max(0, n_candidates - ok))
        cells[key] = {
            "key": key,
            "color": raw["color"],
            "garment": raw["garment"],
            "pattern": raw["pattern"],
            "usage": int(usage.get(key, 0)),
            "nonzero": int(summary.get("nonzero") or 0),
            "nonzero_at": summary.get("nonzero_at") or {
                "0": int(summary.get("nonzero") or 0)
            },
            "n_candidates": n_candidates,
            "n_scored": int(summary.get("n_scored") or 0),
            "mask_status": mask_status,
            "mask_failures": mask_failures,
            "n_saved_crops": len(candidates),
            "screen_root": root,
            "candidates": candidates,
        }
    return cells


def latest_records(
    records: list[dict[str, Any]], decisions: set[str] | None = None
) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        key = cell_key(
            str(record.get("color") or ""),
            str(record.get("garment") or ""),
            str(record.get("pattern") or ""),
        )
        if decisions is None or record.get("decision") in decisions:
            latest[key] = record
    return latest


# A `reset` record retracts whatever verdict came before it, sending the cell
# back to the queue without deleting anything: cell_annotations.jsonl stays
# append-only, so the superseded decision is still readable. Used when the
# measurement behind a verdict is invalidated — the `wool coat` garment
# vocabulary contamination is the case it was written for (docs/PROCESS.md §8).
VERDICT_DECISIONS = {"selected", "excluded"}
SETTLING_DECISIONS = VERDICT_DECISIONS | {"reset"}


def settled_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Cells whose most recent verdict-or-reset is an actual verdict.

    A cell whose latest such record is a `reset` is deliberately absent, which
    is what puts it back into the annotation queue.
    """
    return {key: record
            for key, record in latest_records(records, SETTLING_DECISIONS).items()
            if record.get("decision") in VERDICT_DECISIONS}


def public_candidate(candidate: dict[str, Any], position: int,
                     show_scores: bool) -> dict[str, Any]:
    row = {
        "display_position": position,
        "url_key": candidate["url_key"],
        "crop_available": candidate["crop_path"] is not None,
        "original_available": bool(candidate["source_url"]),
    }
    if show_scores:
        row.update({
            "true_rank": candidate["true_rank"],
            "score": candidate["score"],
            "pattern_score": candidate["pattern_score"],
            "color_score": candidate["color_score"],
            "garment_score": candidate["garment_score"],
        })
    return row


def record_candidate(candidate: dict[str, Any], position: int) -> dict[str, Any]:
    return {
        "true_rank": candidate["true_rank"],
        "display_position": position,
        "url_key": candidate["url_key"],
        "crop_file": candidate["crop_file"],
        "source_url": candidate["source_url"],
        "score": candidate["score"],
        "pattern_score": candidate["pattern_score"],
        "color_score": candidate["color_score"],
        "garment_score": candidate["garment_score"],
    }


def annotation_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class AppState:
    args: argparse.Namespace
    cells: dict[str, dict[str, Any]]
    all_target_keys: list[str]
    queue: list[str]
    screen_meta: dict[str, Any]
    plan_stats: dict[str, int]
    source_url_source: str
    records: list[dict[str, Any]]
    output_dir: Path
    annotation_path: Path
    library_path: Path
    started: float = field(default_factory=time.monotonic)
    writes: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    display_cache: dict[str, tuple[int, list[dict[str, Any]]]] = field(
        default_factory=dict
    )
    recheck_bases: dict[str, dict[str, Any]] = field(default_factory=dict)

    def pass_tag(self, key: str) -> str:
        if key not in self.recheck_bases:
            return "primary"
        base = self.recheck_bases[key]
        return f"recheck|{base.get('ts', '')}|{base.get('record_id', '')}"

    def displayed(self, key: str) -> tuple[int, list[dict[str, Any]]]:
        cached = self.display_cache.get(key)
        if cached is not None:
            return cached
        seed = stable_seed(
            self.args.seed, self.args.variant, key, self.pass_tag(key)
        )
        candidates = list(self.cells[key]["candidates"][:self.args.show_n])
        random.Random(seed).shuffle(candidates)
        self.display_cache[key] = (seed, candidates)
        return seed, candidates

    def completed_keys(self) -> set[str]:
        if self.args.mode == "judge-all":
            return set(latest_records(self.records, {"judge_all"}))
        return set(settled_records(self.records))

    def progress(self) -> dict[str, int]:
        completed = self.completed_keys() & set(self.all_target_keys)
        return {
            "completed": len(completed),
            "total": len(self.all_target_keys),
            "annotatable": len(self.cells.keys() & set(self.all_target_keys)),
            "session_total": len(self.queue),
            "session_written": self.writes,
        }


def next_final_number(directory: Path) -> int:
    largest = 0
    if directory.is_dir():
        for path in directory.glob("*.jpg"):
            try:
                largest = max(largest, int(path.stem))
            except ValueError:
                continue
    return largest + 1


def copy_selected_crops(
    state: AppState,
    cell: dict[str, Any],
    selected: list[dict[str, Any]],
    old_library: dict[str, Any],
) -> list[dict[str, Any]]:
    destination = (
        state.output_dir
        / FINAL_IMAGE_DIR
        / cell["color"]
        / f"{cell['pattern']}_{cell['garment']}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    existing_by_key = {
        image.get("url_key"): image.get("path")
        for image in (
            old_library.get("cells", {}).get(cell["key"], {}).get("images") or []
        )
        if image.get("url_key") and image.get("path")
    }
    number = next_final_number(destination)
    copied = []
    for candidate in selected:
        source = candidate["crop_path"]
        if source is None or not Path(source).is_file():
            raise ValueError(
                f"selected crop is missing for {candidate['url_key']}"
            )
        relative = existing_by_key.get(candidate["url_key"])
        if relative and (state.output_dir / relative).is_file():
            final_path = state.output_dir / relative
        else:
            final_path = destination / f"{number}.jpg"
            number += 1
            shutil.copy2(source, final_path)
            relative = final_path.relative_to(state.output_dir).as_posix()
        copied.append({
            "path": str(relative),
            "true_rank": candidate["true_rank"],
            "url_key": candidate["url_key"],
        })
    return copied


def build_library(state: AppState) -> dict[str, Any]:
    latest = settled_records(state.records)
    retracted = {key: record
                 for key, record in latest_records(state.records,
                                                   SETTLING_DECISIONS).items()
                 if record.get("decision") == "reset"}
    old = {}
    if state.library_path.is_file():
        try:
            old = read_json(state.library_path)
        except (OSError, json.JSONDecodeError):
            old = {}
    cells_out: dict[str, Any] = {}
    available = excluded = 0
    for key in state.all_target_keys:
        record = latest.get(key)
        if record and record.get("decision") == "selected":
            images = []
            for picked in record.get("picked") or []:
                library_path = picked.get("library_path")
                if library_path:
                    images.append({
                        "path": library_path,
                        "true_rank": picked.get("true_rank"),
                        "url_key": picked.get("url_key"),
                    })
            cells_out[key] = {"status": "available", "images": images}
            available += 1
        elif record and record.get("decision") == "excluded":
            cells_out[key] = {
                "status": "excluded",
                "reason": record.get("reason"),
                "images": [],
            }
            excluded += 1
        else:
            previous = old.get("cells", {}).get(key, {})
            cells_out[key] = {
                "status": "pending",
                "images": [],
            }
            if previous.get("judge_all"):
                cells_out[key]["judge_all"] = previous["judge_all"]
            reset = retracted.get(key)
            if reset is not None:
                # Keep the retracted verdict visible in the library, not only in
                # the append-only log, so a before/after comparison after
                # re-collection does not need to replay cell_annotations.jsonl.
                cells_out[key]["reset"] = {
                    "reason": reset.get("reason"),
                    "ts": reset.get("ts"),
                    "superseded": reset.get("superseded"),
                }

    judge_latest = latest_records(state.records, {"judge_all"})
    for key, record in judge_latest.items():
        if key in cells_out:
            judgments = record.get("shown") or []
            cells_out[key]["judge_all"] = {
                "valid": sum(item.get("valid") is True for item in judgments),
                "invalid": sum(item.get("valid") is False for item in judgments),
                "ts": record.get("ts"),
            }
    total = len(state.all_target_keys)
    return {
        "variant": state.args.variant,
        "schema": "color|garment|pattern -> images[]",
        "screen_sources": state.screen_meta["screen_sources"],
        "screen_partial": state.screen_meta["partial"],
        "cells": cells_out,
        "summary": {
            "total": total,
            "available": available,
            "excluded": excluded,
            "pending": total - available - excluded,
        },
    }


def refresh_library(state: AppState) -> dict[str, Any]:
    library = build_library(state)
    write_json_atomic(state.library_path, library)
    return library


def annotation_base(
    state: AppState,
    cell: dict[str, Any],
    seed: int,
    shown: list[dict[str, Any]],
    seconds: float,
) -> dict[str, Any]:
    ts = annotation_timestamp()
    record_id = hashlib.sha256(
        f"{cell['key']}|{ts}|{time.time_ns()}".encode("utf-8")
    ).hexdigest()[:16]
    return {
        "record_id": record_id,
        "color": cell["color"],
        "garment": cell["garment"],
        "pattern": cell["pattern"],
        "annotation_mode": state.args.mode,
        "shuffle_seed": seed,
        "scores_hidden": not state.args.show_scores,
        "n_candidates_shown": len(shown),
        "n_saved_crops": cell["n_saved_crops"],
        "mask_failures": cell["mask_failures"],
        "mask_status": cell["mask_status"],
        "usage_in_plans": cell["usage"],
        "screen_nonzero": cell["nonzero"],
        "screen_nonzero_at": cell["nonzero_at"],
        "annotator": state.args.annotator,
        "seconds": round(max(0.0, seconds), 2),
        "ts": ts,
        "is_recheck": cell["key"] in state.recheck_bases,
    }


def save_decision(state: AppState, payload: dict[str, Any]) -> dict[str, Any]:
    key = str(payload.get("cell_key") or "")
    if key not in state.queue or key not in state.cells:
        raise ValueError("cell is not in this annotation session")
    cell = state.cells[key]
    seed, displayed = state.displayed(key)
    shown = [
        record_candidate(candidate, position)
        for position, candidate in enumerate(displayed, 1)
    ]
    seconds = float(payload.get("seconds") or 0.0)
    record = annotation_base(state, cell, seed, shown, seconds)
    old_library = read_json(state.library_path) if state.library_path.is_file() else {}

    if state.args.mode == "judge-all":
        raw_judgments = payload.get("judgments")
        if not isinstance(raw_judgments, list):
            raise ValueError("judge-all requires judgments[]")
        values: dict[int, bool] = {}
        for item in raw_judgments:
            position = int(item.get("display_position") or 0)
            valid = item.get("valid")
            if position < 1 or position > len(displayed) or not isinstance(valid, bool):
                raise ValueError("invalid judge-all candidate judgment")
            values[position] = valid
        if len(values) != len(displayed):
            raise ValueError("every shown candidate must be marked valid or invalid")
        for item in shown:
            item["valid"] = values[item["display_position"]]
        record.update({
            "decision": "judge_all",
            "picked": [],
            "shown": shown,
            "cell_available": any(values.values()),
        })
    else:
        decision = payload.get("decision")
        if decision == "excluded":
            reason = payload.get("reason")
            if reason not in EXCLUSION_REASONS:
                raise ValueError("invalid exclusion reason")
            record.update({
                "decision": "excluded",
                "reason": reason,
                "picked": [],
                "shown": shown,
            })
        elif decision == "selected":
            raw_positions = payload.get("positions")
            if not isinstance(raw_positions, list):
                raise ValueError("selected decision requires positions[]")
            positions = []
            for raw in raw_positions:
                position = int(raw)
                if position not in positions:
                    positions.append(position)
            if not positions or len(positions) > state.args.pick_n:
                raise ValueError(f"select between 1 and {state.args.pick_n} candidates")
            if any(position < 1 or position > len(displayed) for position in positions):
                raise ValueError("selected display position is out of range")
            selected = [displayed[position - 1] for position in positions]
            library_images = copy_selected_crops(
                state, cell, selected, old_library
            )
            library_by_key = {
                item["url_key"]: item["path"] for item in library_images
            }
            picked = []
            for position, candidate in zip(positions, selected):
                item = record_candidate(candidate, position)
                item["library_path"] = library_by_key[candidate["url_key"]]
                picked.append(item)
            record.update({
                "decision": "selected",
                "picked": picked,
                "shown": shown,
            })
        else:
            raise ValueError("decision must be selected or excluded")

    if key in state.recheck_bases:
        record["recheck_of"] = state.recheck_bases[key].get("record_id")
    if payload.get("auto_accepted") is True:
        record["auto_accepted"] = True

    with state.lock:
        append_jsonl(state.annotation_path, record)
        state.records.append(record)
        state.writes += 1
        library = refresh_library(state)
    return {
        "ok": True,
        "record_id": record["record_id"],
        "progress": state.progress(),
        "library_summary": library["summary"],
    }


def queue_cells(
    args: argparse.Namespace,
    cells: dict[str, dict[str, Any]],
    target_keys: list[str],
    records: list[dict[str, Any]],
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    target_set = set(target_keys)
    available_keys = target_set & set(cells)
    recheck_bases: dict[str, dict[str, Any]] = {}
    if args.recheck:
        if args.mode != "select":
            raise SystemExit("[error] --recheck is supported in select mode")
        bases = settled_records(records)
        eligible = sorted(available_keys & set(bases))
        random.Random(args.seed).shuffle(eligible)
        queue = eligible[:args.recheck]
        recheck_bases = {key: bases[key] for key in queue}
    else:
        if args.mode == "judge-all":
            completed = set(latest_records(records, {"judge_all"}))
        else:
            completed = set(settled_records(records))
        queue = list(available_keys - completed)
        if args.order == "usage":
            queue.sort(key=lambda key: (
                -cells[key]["usage"],
                cells[key]["color"],
                cells[key]["garment"],
                cells[key]["pattern"],
            ))
        elif args.order == "risk":
            queue.sort(key=lambda key: (
                cells[key]["nonzero"],
                -cells[key]["usage"],
                cells[key]["color"],
                cells[key]["garment"],
                cells[key]["pattern"],
            ))
        else:
            order_map = {key: index for index, key in enumerate(target_keys)}
            queue.sort(key=lambda key: order_map[key])
    if args.limit:
        queue = queue[:args.limit]
        recheck_bases = {
            key: record for key, record in recheck_bases.items() if key in queue
        }
    return queue, recheck_bases


def grid_target_keys(
    only_needed: bool,
    needed: set[str],
    colors: list[str],
    garments: list[str],
    patterns: list[str],
) -> list[str]:
    if only_needed:
        needed_set = set(needed)
        return [
            cell_key(color, garment, pattern)
            for color in colors
            for garment in garments
            for pattern in patterns
            if cell_key(color, garment, pattern) in needed_set
        ]
    return [
        cell_key(color, garment, pattern)
        for color in colors
        for garment in garments
        for pattern in patterns
    ]


def cell_payload(state: AppState, index: int) -> dict[str, Any]:
    if not state.queue:
        return {"done": True, "progress": state.progress()}
    if index < 0 or index >= len(state.queue):
        raise ValueError("cell index out of range")
    key = state.queue[index]
    cell = state.cells[key]
    seed, displayed = state.displayed(key)
    payload = {
        "done": False,
        "index": index,
        "session_total": len(state.queue),
        "key": key,
        "color": cell["color"],
        "garment": cell["garment"],
        "pattern": cell["pattern"],
        "usage": cell["usage"],
        "nonzero": cell["nonzero"],
        "nonzero_at": cell["nonzero_at"],
        "n_candidates": cell["n_candidates"],
        "n_scored": cell["n_scored"],
        "n_saved_crops": cell["n_saved_crops"],
        "mask_failures": cell["mask_failures"],
        "mask_status": cell["mask_status"],
        "shuffle_seed": seed,
        "candidates": [
            public_candidate(candidate, position, state.args.show_scores)
            for position, candidate in enumerate(displayed, 1)
        ],
        "progress": state.progress(),
        "already_submitted": key in state.completed_keys(),
    }
    # A recheck must not reveal the earlier decision.
    if key in state.recheck_bases:
        payload["recheck"] = True
    return payload


def auto_accept(state: AppState) -> int:
    threshold = state.args.auto_accept_above
    if threshold is None:
        return 0
    if state.args.mode != "select" or state.args.recheck:
        raise SystemExit(
            "[error] --auto-accept-above requires primary select mode"
        )
    accepted = 0
    for key in list(state.queue):
        cell = state.cells[key]
        seed, displayed = state.displayed(key)
        if not displayed:
            continue
        top = min(displayed, key=lambda item: item["true_rank"])
        if top["score"] < threshold:
            continue
        position = displayed.index(top) + 1
        save_decision(state, {
            "cell_key": key,
            "decision": "selected",
            "positions": [position],
            "seconds": 0,
            "auto_accepted": True,
        })
        accepted += 1
    if accepted:
        completed = state.completed_keys()
        state.queue = [key for key in state.queue if key not in completed]
    return accepted


def json_response(handler: BaseHTTPRequestHandler, payload: Any,
                  status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def file_response(
    handler: BaseHTTPRequestHandler,
    path: Path,
    content_type: str | None = None,
    cache_control: str = "private, max-age=3600",
) -> None:
    if not path.is_file():
        handler.send_error(HTTPStatus.NOT_FOUND)
        return
    content_type = content_type or mimetypes.guess_type(path.name)[0] \
        or "application/octet-stream"
    size = path.stat().st_size
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(size))
    handler.send_header("Cache-Control", cache_control)
    handler.end_headers()
    with path.open("rb") as handle:
        shutil.copyfileobj(handle, handler.wfile)


def make_handler(state: AppState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "PODCellAnnotator/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[http] {self.address_string()} {fmt % args}")

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path in {"/", "/static/annotator.html"}:
                    file_response(
                        self,
                        STATIC_HTML,
                        "text/html; charset=utf-8",
                        cache_control="no-store",
                    )
                    return
                if parsed.path == "/api/bootstrap":
                    json_response(self, {
                        "variant": state.args.variant,
                        "mode": state.args.mode,
                        "pick_n": state.args.pick_n,
                        "show_n": state.args.show_n,
                        "scores_hidden": not state.args.show_scores,
                        "order": state.args.order,
                        "only_needed": state.args.only_needed,
                        "recheck": state.args.recheck,
                        "partial": state.screen_meta["partial"],
                        "screen_cells_loaded": state.screen_meta[
                            "screen_cells_loaded"
                        ],
                        "screen_cells_expected": state.screen_meta[
                            "screen_cells_expected"
                        ],
                        "screen_sources": state.screen_meta["screen_sources"],
                        "source_url_source": state.source_url_source,
                        "plan_stats": state.plan_stats,
                        "exclusion_reasons": list(EXCLUSION_REASONS),
                        "progress": state.progress(),
                    })
                    return
                if parsed.path == "/api/cell":
                    index = int((query.get("index") or ["0"])[0])
                    json_response(self, cell_payload(state, index))
                    return
                if parsed.path == "/api/summary":
                    library = read_json(state.library_path)
                    json_response(self, {
                        "progress": state.progress(),
                        "library": library["summary"],
                    })
                    return
                if parsed.path in {"/media/crop", "/media/original"}:
                    key = (query.get("cell") or [""])[0]
                    url_key = (query.get("key") or [""])[0]
                    if key not in state.cells:
                        self.send_error(HTTPStatus.NOT_FOUND)
                        return
                    candidate = next(
                        (
                            item for item in state.cells[key]["candidates"]
                            if item["url_key"] == url_key
                        ),
                        None,
                    )
                    if candidate is None:
                        self.send_error(HTTPStatus.NOT_FOUND)
                        return
                    if parsed.path == "/media/crop":
                        path = candidate["crop_path"]
                        if path is None:
                            self.send_error(HTTPStatus.NOT_FOUND)
                        else:
                            file_response(self, Path(path), "image/jpeg")
                        return
                    source_url = candidate["source_url"]
                    if not source_url:
                        self.send_error(
                            HTTPStatus.NOT_FOUND, "original URL unavailable"
                        )
                        return
                    self.send_response(HTTPStatus.FOUND)
                    self.send_header("Location", source_url)
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    return
                self.send_error(HTTPStatus.NOT_FOUND)
            except (ValueError, KeyError) as exc:
                json_response(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except BrokenPipeError:
                pass

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            parsed = urlparse(self.path)
            if parsed.path != "/api/annotate":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if self.headers.get_content_type() != "application/json":
                json_response(
                    self,
                    {"error": "Content-Type must be application/json"},
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                )
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if length <= 0 or length > 1_000_000:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length))
                result = save_decision(state, payload)
                json_response(self, result)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                json_response(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except OSError as exc:
                json_response(
                    self,
                    {"error": f"could not persist annotation: {exc}"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local cell-level annotator for screen_sam3 crops."
    )
    parser.add_argument("--variant", default="wacv_scenario_v4")
    parser.add_argument("--screen-dir", type=Path,
                        default=Path("availability_audit/images"))
    parser.add_argument("--plan-path", type=Path)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--annotator", default=os.environ.get("USER", "unknown"))
    parser.add_argument("--show-n", type=int, default=10)
    parser.add_argument("--show-scores", action="store_true")
    parser.add_argument("--mode", choices=("select", "judge-all"),
                        default="select")
    parser.add_argument("--pick-n", type=int, default=1)
    parser.add_argument("--order", choices=("usage", "risk", "grid"),
                        default="usage")
    parser.add_argument(
        "--only-needed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="restrict to fully specified cells used by option plans (default)",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--recheck", type=int, default=0,
                        help="re-annotate N completed cells without showing prior labels")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--auto-accept-above", type=float,
                        help="explicit opt-in: select scorer rank 1 at/above score")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate inputs and print the session summary only")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.show_n <= 0:
        parser.error("--show-n must be positive")
    if args.pick_n <= 0:
        parser.error("--pick-n must be positive")
    if args.limit < 0 or args.recheck < 0:
        parser.error("--limit/--recheck cannot be negative")
    if args.auto_accept_above is not None and args.auto_accept_above < 0:
        parser.error("--auto-accept-above cannot be negative")
    return args


def build_state(args: argparse.Namespace) -> AppState:
    screen_dir = args.screen_dir.resolve()
    output_dir = args.output_dir.resolve()
    plan_path = args.plan_path or (
        REPO_ROOT / f"data_{args.variant}" / "options" / "option_plans.jsonl"
    )
    plan_path = plan_path.resolve()
    if not plan_path.is_file():
        raise SystemExit(f"[error] option plans not found: {plan_path}")
    usage, needed, plan_colors, plan_garments, plan_patterns, plan_stats = \
        plan_usage(plan_path)
    runs = discover_screen_runs(screen_dir)
    raw_cells, screen_meta = choose_screen_snapshot(
        runs, plan_colors, plan_garments, plan_patterns
    )
    candidate_meta = load_candidate_metadata(raw_cells)
    wanted_url_keys = {
        saved_url_key(saved)
        for cell in raw_cells.values()
        for saved in (cell["summary"].get("saved_crops") or [])
    }
    wanted_url_keys.discard(None)
    source_urls, source_url_source = resolve_source_urls(
        set(wanted_url_keys), candidate_meta
    )
    cells = prepare_cells(raw_cells, usage, candidate_meta, source_urls)
    target_keys = grid_target_keys(
        args.only_needed,
        needed,
        screen_meta["colors"],
        screen_meta["garments"],
        screen_meta["patterns"],
    )
    annotation_path = output_dir / ANNOTATION_FILE
    library_path = output_dir / LIBRARY_FILE
    records, warnings = load_annotation_history(annotation_path)
    for warning in warnings:
        print(f"[warn] {warning}", file=sys.stderr)
    queue, recheck_bases = queue_cells(args, cells, target_keys, records)
    state = AppState(
        args=args,
        cells=cells,
        all_target_keys=target_keys,
        queue=queue,
        screen_meta=screen_meta,
        plan_stats=plan_stats,
        source_url_source=source_url_source,
        records=records,
        output_dir=output_dir,
        annotation_path=annotation_path,
        library_path=library_path,
        recheck_bases=recheck_bases,
    )
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        refresh_library(state)
    return state


def print_state_summary(state: AppState) -> None:
    progress = state.progress()
    print("=" * 76)
    print("  POD-Bench SAM3 cell annotator")
    print(f"  variant:       {state.args.variant}")
    print(f"  plans:         {state.args.plan_path or REPO_ROOT / f'data_{state.args.variant}' / 'options' / 'option_plans.jsonl'}")
    print(f"  screen source: {', '.join(state.screen_meta['screen_sources'])}")
    print(f"  screen cells:  {state.screen_meta['screen_cells_loaded']} / "
          f"{state.screen_meta['screen_cells_expected']}"
          f"{'  PARTIAL' if state.screen_meta['partial'] else ''}")
    print(f"  plan cells:    {state.plan_stats['needed_cells']} "
          f"(one-use {state.plan_stats['one_use_cells']})")
    print(f"  target:        {progress['total']} "
          f"({'only-needed' if state.args.only_needed else 'full grid'})")
    print(f"  completed:     {progress['completed']} / {progress['total']}")
    print(f"  this session:  {len(state.queue)} cells "
          f"(mode={state.args.mode}, order={state.args.order})")
    print(f"  source URLs:   {len({c['source_url'] for cell in state.cells.values() for c in cell['candidates'] if c['source_url']})} resolved via {state.source_url_source}")
    print(f"  annotations:   {state.annotation_path}")
    print(f"  library:       {state.library_path}")
    print("=" * 76)


def main() -> int:
    args = parse_args()
    state = build_state(args)
    auto_count = 0 if args.dry_run else auto_accept(state)
    if auto_count:
        print(f"[warn] explicitly auto-accepted {auto_count} cells at score >= "
              f"{args.auto_accept_above}")
    print_state_summary(state)
    if args.dry_run:
        print("  dry-run: no HTTP server started")
        return 0
    if not STATIC_HTML.is_file():
        raise SystemExit(f"[error] static UI not found: {STATIC_HTML}")
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), make_handler(state)
    )
    print(f"  open: http://127.0.0.1:{args.port}")
    print("  bound to 127.0.0.1 only; Ctrl+C stops the server")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopping annotator")
    finally:
        server.server_close()
        elapsed = time.monotonic() - state.started
        progress = state.progress()
        print(f"  session wrote {state.writes} decisions in {elapsed:.1f}s")
        print(f"  overall progress {progress['completed']} / {progress['total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
