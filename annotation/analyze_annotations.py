#!/usr/bin/env python3
"""Offline metrics for cell annotations produced by ``serve_annotator``.

Only JSON/JSONL files are read.  No image, model, GPU, or running annotation
server is required.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent


def key_of(record: dict[str, Any]) -> str:
    return "|".join(str(record.get(name) or "")
                    for name in ("color", "garment", "pattern"))


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    warnings = []
    if not path.is_file():
        raise SystemExit(f"[error] annotation JSONL not found: {path}")
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.append(f"ignored malformed line {line_no}: {exc.msg}")
                continue
            if isinstance(row, dict):
                rows.append(row)
            else:
                warnings.append(f"ignored non-object line {line_no}")
    return rows, warnings


def latest_by_cell(rows: list[dict[str, Any]],
                   decisions: set[str]) -> dict[str, dict[str, Any]]:
    latest = {}
    for row in rows:
        if row.get("decision") in decisions:
            latest[key_of(row)] = row
    return latest


def primary_rows(rows: list[dict[str, Any]], include_auto: bool) \
        -> list[dict[str, Any]]:
    return [
        row for row in rows
        if not row.get("is_recheck")
        and (include_auto or not row.get("auto_accepted"))
    ]


def best_picked_rank(record: dict[str, Any]) -> int | None:
    ranks = [
        int(item["true_rank"])
        for item in (record.get("picked") or [])
        if item.get("true_rank") is not None
    ]
    return min(ranks) if ranks else None


def selection_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latest = latest_by_cell(rows, {"selected", "excluded"})
    selected = [
        record for record in latest.values()
        if record.get("decision") == "selected"
    ]
    ranks = [rank for rank in map(best_picked_rank, selected) if rank is not None]
    positions = Counter()
    for record in selected:
        for picked in record.get("picked") or []:
            if picked.get("display_position") is not None:
                positions[int(picked["display_position"])] += 1
    rank_counts = Counter(ranks)
    n = len(ranks)
    return {
        "selected_cells": len(selected),
        "rank_observations": n,
        "true_rank_distribution": {
            str(rank): rank_counts[rank] for rank in sorted(rank_counts)
        },
        "top1_agreement": (sum(rank == 1 for rank in ranks) / n) if n else None,
        "rank_le_3": (sum(rank <= 3 for rank in ranks) / n) if n else None,
        "rank_le_5": (sum(rank <= 5 for rank in ranks) / n) if n else None,
        "mrr": (sum(1.0 / rank for rank in ranks) / n) if n else None,
        "display_position_distribution": {
            str(position): positions[position] for position in sorted(positions)
        },
    }


def binary_auc(scores: list[float], labels: list[bool]) -> float | None:
    """Mann–Whitney ROC AUC with average ranks for tied scores."""
    if len(scores) != len(labels) or not scores:
        return None
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum_positive = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        rank_sum_positive += average_rank * sum(
            bool(label) for _, label in ordered[index:end]
        )
        index = end
    return (
        rank_sum_positive - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def judge_all_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latest = latest_by_cell(rows, {"judge_all"})
    records = list(latest.values())
    precision = {}
    for k in (1, 5, 10):
        labels = [
            bool(item["valid"])
            for record in records
            for item in (record.get("shown") or [])
            if item.get("valid") is not None
            and int(item.get("true_rank") or 10**9) <= k
        ]
        precision[f"precision_at_{k}"] = (
            sum(labels) / len(labels) if labels else None
        )
        precision[f"n_at_{k}"] = len(labels)

    score_rows = [
        item
        for record in records
        for item in (record.get("shown") or [])
        if isinstance(item.get("valid"), bool)
    ]
    axis_auc = {}
    for field in ("score", "pattern_score", "color_score", "garment_score"):
        pairs = [
            (float(item[field]), bool(item["valid"]))
            for item in score_rows
            if item.get(field) is not None
        ]
        axis_auc[field] = {
            "auc": binary_auc(
                [score for score, _ in pairs],
                [label for _, label in pairs],
            ),
            "n": len(pairs),
        }
    valid = sum(bool(item["valid"]) for item in score_rows)
    return {
        "cells": len(records),
        "candidate_judgments": len(score_rows),
        "valid": valid,
        "invalid": len(score_rows) - valid,
        **precision,
        "auc": axis_auc,
    }


def matrix_counts(pairs: list[tuple[bool, bool]]) -> dict[str, int]:
    result = {
        "predicted_available__human_available": 0,
        "predicted_available__human_unavailable": 0,
        "predicted_unavailable__human_available": 0,
        "predicted_unavailable__human_unavailable": 0,
    }
    for predicted, human in pairs:
        left = "predicted_available" if predicted else "predicted_unavailable"
        right = "human_available" if human else "human_unavailable"
        result[f"{left}__{right}"] += 1
    return result


def cell_confusion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    select_latest = latest_by_cell(rows, {"selected", "excluded"})
    judge_latest = latest_by_cell(rows, {"judge_all"})
    human: dict[str, tuple[bool, dict[str, Any]]] = {}
    for key, record in judge_latest.items():
        human[key] = (bool(record.get("cell_available")), record)
    for key, record in select_latest.items():
        human[key] = (record.get("decision") == "selected", record)

    score_thresholds = sorted({
        str(threshold)
        for _, record in human.values()
        for threshold in (record.get("screen_nonzero_at") or {}).keys()
    }, key=lambda value: float(value))
    curves = {}
    for score_threshold in score_thresholds:
        by_count = {}
        for minimum_count in (1, 3, 5, 10):
            pairs = []
            for available, record in human.values():
                counts = record.get("screen_nonzero_at") or {}
                count = counts.get(score_threshold)
                if count is None:
                    continue
                pairs.append((int(count) >= minimum_count, available))
            by_count[str(minimum_count)] = {
                "n": len(pairs),
                **matrix_counts(pairs),
            }
        curves[score_threshold] = by_count
    return {
        "human_labeled_cells": len(human),
        "score_threshold_to_minimum_nonzero_count": curves,
    }


def recheck_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {
        row.get("record_id"): row
        for row in rows
        if row.get("record_id")
    }
    pairs = []
    for row in rows:
        if not row.get("is_recheck"):
            continue
        base = by_id.get(row.get("recheck_of"))
        if base is None:
            continue
        pairs.append((base, row))
    decision_matches = 0
    exact_matches = 0
    for base, recheck in pairs:
        same_decision = base.get("decision") == recheck.get("decision")
        decision_matches += same_decision
        exact = same_decision
        if exact and base.get("decision") == "selected":
            original = {item.get("url_key") for item in base.get("picked") or []}
            repeated = {item.get("url_key") for item in recheck.get("picked") or []}
            exact = original == repeated
        elif exact and base.get("decision") == "excluded":
            exact = base.get("reason") == recheck.get("reason")
        elif exact and base.get("decision") == "judge_all":
            original = {
                item.get("url_key"): item.get("valid")
                for item in base.get("shown") or []
            }
            repeated = {
                item.get("url_key"): item.get("valid")
                for item in recheck.get("shown") or []
            }
            exact = original == repeated
        exact_matches += exact
    n = len(pairs)
    return {
        "pairs": n,
        "decision_agreement": decision_matches / n if n else None,
        "exact_agreement": exact_matches / n if n else None,
    }


def timing_metrics(rows: list[dict[str, Any]], target_total: int) -> dict[str, Any]:
    latest = latest_by_cell(rows, {"selected", "excluded", "judge_all"})
    seconds = [
        float(row["seconds"])
        for row in latest.values()
        if row.get("seconds") is not None
        and not row.get("auto_accepted")
        and float(row["seconds"]) >= 0
    ]
    mean = statistics.mean(seconds) if seconds else None
    median = statistics.median(seconds) if seconds else None
    return {
        "human_cells_timed": len(seconds),
        "mean_seconds_per_cell": mean,
        "median_seconds_per_cell": median,
        "estimated_hours_for_target": (
            mean * target_total / 3600.0 if mean is not None else None
        ),
        "target_cells": target_total,
    }


def finite_round(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 6)
    if isinstance(value, dict):
        return {key: finite_round(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_round(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print offline metrics from cell_annotations.jsonl."
    )
    parser.add_argument("--annotations", type=Path,
                        default=HERE / "cell_annotations.jsonl")
    parser.add_argument("--library", type=Path,
                        default=HERE / "attribute_library.json")
    parser.add_argument("--target-total", type=int,
                        help="override time-estimate target cell count")
    parser.add_argument("--include-auto", action="store_true",
                        help="include explicitly auto-accepted records in human metrics")
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, warnings = read_jsonl(args.annotations)
    target_total = args.target_total
    library_summary = None
    if args.library.is_file():
        library = json.loads(args.library.read_text(encoding="utf-8"))
        library_summary = library.get("summary")
        if target_total is None:
            target_total = int((library_summary or {}).get("total") or 0)
    if target_total is None:
        target_total = len({
            key_of(row) for row in rows
            if row.get("decision") in {"selected", "excluded", "judge_all"}
        })

    primary = primary_rows(rows, args.include_auto)
    latest_decisions = latest_by_cell(primary, {"selected", "excluded"})
    exclusion_reasons = Counter(
        row.get("reason")
        for row in latest_decisions.values()
        if row.get("decision") == "excluded"
    )
    report = {
        "annotations": str(args.annotations),
        "records": len(rows),
        "warnings": warnings,
        "auto_accepted_records": sum(bool(row.get("auto_accepted")) for row in rows),
        "library_summary": library_summary,
        "selection": selection_metrics(primary),
        "exclusion_reason_distribution": {
            str(reason): count
            for reason, count in sorted(exclusion_reasons.items())
        },
        "judge_all": judge_all_metrics(primary),
        "cell_confusion": cell_confusion(primary),
        "recheck": recheck_metrics(rows),
        "timing": timing_metrics(primary, target_total),
    }
    report = finite_round(report)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.json_out.with_name(args.json_out.name + ".tmp")
        tmp.write_text(text + "\n", encoding="utf-8")
        tmp.replace(args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
