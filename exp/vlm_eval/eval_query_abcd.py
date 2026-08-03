#!/usr/bin/env python3
"""Evaluate Query + optional profile + four (image, text) options.

The three reported experiment accuracies are:
- Profile Accuracy: profile-only, original A or C
- Scenario Accuracy: query-only, original A or B
- TPO Accuracy: query + profile, original A

Run examples (the VLM server must already be running):

    # Query only
    python -m exp.vlm_eval.eval_query_abcd \
      --model Qwen/Qwen3-VL-4B-Instruct \
      --base-urls http://127.0.0.1:8002/v1 \
      --concurrency 8

    # Query + narrative profile
    python -m exp.vlm_eval.eval_query_abcd \
      --model Qwen/Qwen3-VL-4B-Instruct \
      --base-urls http://127.0.0.1:8002/v1 \
      --profile-format narrative \
      --concurrency 8

    # Query + key-value profile
    python -m exp.vlm_eval.eval_query_abcd \
      --model Qwen/Qwen3-VL-4B-Instruct \
      --base-urls http://127.0.0.1:8002/v1 \
      --profile-format key-value \
      --concurrency 8

    # Narrative profile only
    python -m exp.vlm_eval.eval_query_abcd \
      --model Qwen/Qwen3-VL-4B-Instruct \
      --base-urls http://127.0.0.1:8002/v1 \
      --profile-format narrative \
      --profile-only \
      --concurrency 8

    # Key-value profile only
    python -m exp.vlm_eval.eval_query_abcd \
      --model Qwen/Qwen3-VL-4B-Instruct \
      --base-urls http://127.0.0.1:8002/v1 \
      --profile-format key-value \
      --profile-only \
      --concurrency 8
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import hashlib
import io
import json
import mimetypes
import os
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from configs.scenarios import (  # noqa: E402
    EVAL_FRAME_CLAUSE,
    EVAL_PRIORITY_CLAUSE_PREFERENCE,
    EVAL_PRIORITY_CLAUSE_SITUATION,
    PROMPT_VERSION,
)
from scripts.plan_index import PlanIndex, select_by_manifest  # noqa: E402
from exp.llm_eval.text_eval import (  # noqa: E402
    option_to_text,
    parse_answer,
    profile_to_all_kv_text,
    profile_to_narrative,
)


DEFAULT_PLANS = (
    REPO_ROOT / "data_wacv_scenario_v4" / "options" / "option_plans.jsonl"
)
DEFAULT_QUERIES = (
    REPO_ROOT / "data_wacv_scenario_v4" / "queries" / "queries.jsonl"
)
DEFAULT_PROFILES = (
    REPO_ROOT / "data_wacv_scenario_v4" / "profiles" / "profiles.jsonl"
)
DEFAULT_LIBRARY = REPO_ROOT / "annotation" / "attribute_library.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "exp" / "vlm_eval" / "results"
DISPLAY_LABELS = ("A", "B", "C", "D")
TPO_CORRECT_ORIGINALS = {"A", "B"}
PROFILE_CORRECT_ORIGINALS = {"A", "C"}


SYSTEM_PROMPT_QUERY_ONLY = f"""You are a fashion advisor.

You will be given:
1. A fashion query describing a situation or occasion
2. Four clothing options (A, B, C, D), each shown with an image and text

Your task:
Select the single BEST option that best fits the query.
{EVAL_FRAME_CLAUSE}
{EVAL_PRIORITY_CLAUSE_SITUATION}

OUTPUT FORMAT — CRITICAL:
You MUST output EXACTLY one character: A, B, C, or D.
Do NOT output any explanation, reasoning, punctuation, or whitespace.
Do NOT write sentences. Do NOT write words.
Your ENTIRE response must be one of: A  B  C  D
"""

SYSTEM_PROMPT_WITH_PROFILE = f"""You are a fashion advisor.

You will be given:
1. A fashion query describing a situation or occasion
2. User profile information describing fashion likes and dislikes
3. Four clothing options (A, B, C, D), each shown with an image and text

Your task:
Select the single BEST option that best fits both the query and the user's preferences.
{EVAL_FRAME_CLAUSE}
{EVAL_PRIORITY_CLAUSE_SITUATION}
{EVAL_PRIORITY_CLAUSE_PREFERENCE}

OUTPUT FORMAT — CRITICAL:
You MUST output EXACTLY one character: A, B, C, or D.
Do NOT output any explanation, reasoning, punctuation, or whitespace.
Do NOT write sentences. Do NOT write words.
Your ENTIRE response must be one of: A  B  C  D
"""

SYSTEM_PROMPT_PROFILE_ONLY = """You are a fashion advisor.

You will be given:
1. User profile information describing fashion likes and dislikes
2. Four clothing options (A, B, C, D), each shown with an image and text

Your task:
Select the single BEST option that best fits the user's preferences.

OUTPUT FORMAT — CRITICAL:
You MUST output EXACTLY one character: A, B, C, or D.
Do NOT output any explanation, reasoning, punctuation, or whitespace.
Do NOT write sentences. Do NOT write words.
Your ENTIRE response must be one of: A  B  C  D
"""

FINAL_INSTRUCTION_QUERY_ONLY = """=== INSTRUCTION ===
Select the single BEST option for the query.
Respond with ONE letter only: A, B, C, or D.
Do NOT write any explanation or reasoning.
Do NOT write anything before or after the letter.
Your complete response must be a single character.

Answer:"""

FINAL_INSTRUCTION_WITH_PROFILE = """=== INSTRUCTION ===
Select the single BEST option for both the query and the user's preferences.
First eliminate options that are inappropriate for the stated situation.
Among the remaining options, choose the one that best matches the user's preferences.
Respond with ONE letter only: A, B, C, or D.
Do NOT write any explanation or reasoning.
Do NOT write anything before or after the letter.
Your complete response must be a single character.

Answer:"""

FINAL_INSTRUCTION_PROFILE_ONLY = """=== INSTRUCTION ===
Select the single BEST option for the user's preferences.
Respond with ONE letter only: A, B, C, or D.
Do NOT write any explanation or reasoning.
Do NOT write anything before or after the letter.
Your complete response must be a single character.

Answer:"""


def uses_profile(profile_format: str) -> bool:
    return profile_format != "none"


def system_prompt_for(profile_format: str, profile_only: bool = False) -> str:
    if profile_only:
        return SYSTEM_PROMPT_PROFILE_ONLY
    if uses_profile(profile_format):
        return SYSTEM_PROMPT_WITH_PROFILE
    return SYSTEM_PROMPT_QUERY_ONLY


def final_instruction_for(profile_format: str, profile_only: bool = False) -> str:
    if profile_only:
        return FINAL_INSTRUCTION_PROFILE_ONLY
    if uses_profile(profile_format):
        return FINAL_INSTRUCTION_WITH_PROFILE
    return FINAL_INSTRUCTION_QUERY_ONLY


def render_profile(profile: dict[str, Any], profile_format: str) -> str | None:
    if profile_format == "narrative":
        return profile_to_narrative(profile)
    if profile_format == "key-value":
        return profile_to_all_kv_text(profile)
    if profile_format == "none":
        return None
    raise ValueError(f"unknown profile format: {profile_format}")


def default_experiment_name(
    profile_format: str,
    seed: int,
    profile_only: bool = False,
) -> str:
    if profile_format == "none":
        return f"query_abcd_image_text_seed{seed}"
    format_tag = profile_format.replace("-", "_")
    if profile_only:
        return f"profile_{format_tag}_abcd_image_text_seed{seed}"
    return f"query_profile_{format_tag}_abcd_image_text_seed{seed}"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"[error] {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"[error] {path}:{line_no}: expected JSON object")
            rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def sanitize(value: str) -> str:
    value = value.strip().replace("/", "__")
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("._-")
    return value or "model"


def portable_path(path: Path) -> str:
    """Prefer a repo-relative record while allowing configured external inputs."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def stable_plan_seed(seed: int, plan_id: str) -> int:
    raw = f"{seed}|{plan_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def image_cell_key(attributes: dict[str, Any]) -> str | None:
    color = attributes.get("color")
    garment = attributes.get("garment_category")
    pattern = attributes.get("pattern")
    if not color or not garment or not pattern:
        return None
    return f"{color}|{garment}|{pattern}"


def resolve_plan_images(
    plan: dict[str, Any],
    cells: dict[str, Any],
    library_dir: Path,
) -> tuple[dict[str, Path] | None, str]:
    """Resolve exact library paths for A--D or return one plan-level reason."""
    keys = {}
    for option_label in DISPLAY_LABELS:
        option = (plan.get("options") or {}).get(option_label) or {}
        key = image_cell_key(option.get("attributes") or {})
        if key is None:
            return None, "config_incomplete"
        keys[option_label] = key

    entries = {label: cells.get(key) for label, key in keys.items()}
    if any(entry is None for entry in entries.values()):
        return None, "library_cell_missing"
    if any(entry.get("status") == "excluded" for entry in entries.values()):
        return None, "has_excluded"
    if any(entry.get("status") != "available" for entry in entries.values()):
        return None, "not_available"

    paths = {}
    for label, entry in entries.items():
        images = entry.get("images") or []
        relative = images[0].get("path") if images else None
        if not relative:
            return None, "available_without_image"
        path = (library_dir / relative).resolve()
        if not path.is_file():
            return None, "image_file_missing"
        paths[label] = path
    return paths, "survives"


def select_surviving_plans(
    plans: list[dict[str, Any]],
    library: dict[str, Any],
    library_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Path]], Counter[str]]:
    cells = library.get("cells") or {}
    selected = []
    paths_by_plan = {}
    counts: Counter[str] = Counter()
    for plan in plans:
        plan_id = plan.get("plan_id") or plan.get("query_id")
        paths, reason = resolve_plan_images(plan, cells, library_path.parent)
        counts[reason] += 1
        if paths is not None:
            selected.append(plan)
            paths_by_plan[str(plan_id)] = paths
    return selected, paths_by_plan, counts


def query_text(plan: dict[str, Any], queries: dict[str, dict[str, Any]]) -> str:
    query = queries.get(str(plan.get("query_id"))) or {}
    text = query.get("query_text") or plan.get("query_text") or ""
    return str(text).strip()


def make_job(
    plan: dict[str, Any],
    queries: dict[str, dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
    image_paths: dict[str, Path],
    seed: int,
    profile_format: str,
    profile_only: bool,
) -> dict[str, Any]:
    plan_id = str(plan.get("plan_id") or plan.get("query_id"))
    user_id = str(plan.get("user_id") or "")
    items = list((plan.get("options") or {}).items())
    random.Random(stable_plan_seed(seed, plan_id)).shuffle(items)
    displayed = []
    display_to_original = {}
    for display_label, (original_label, option) in zip(DISPLAY_LABELS, items):
        display_to_original[display_label] = original_label
        displayed.append({
            "display_label": display_label,
            "original_label": original_label,
            "text": option_to_text(option),
            "image_path": image_paths[original_label],
        })
    return {
        "plan": plan,
        "plan_id": plan_id,
        "query_text": query_text(plan, queries),
        "profile_format": profile_format,
        "profile_only": profile_only,
        "profile_text": render_profile(profiles.get(user_id, {}), profile_format),
        "displayed": displayed,
        "display_to_original": display_to_original,
        "tpo_correct_display": [
            display for display, original in display_to_original.items()
            if original in TPO_CORRECT_ORIGINALS
        ],
        "strict_correct_display": next(
            display for display, original in display_to_original.items()
            if original == "A"
        ),
        "profile_correct_display": [
            display for display, original in display_to_original.items()
            if original in PROFILE_CORRECT_ORIGINALS
        ],
    }


_IMAGE_SIZE = 768
_IMAGE_LOCK = threading.Lock()


@lru_cache(maxsize=8192)
def encode_image(path_string: str) -> tuple[str, str]:
    path = Path(path_string)
    try:
        from PIL import Image
    except ImportError:
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        return mime, base64.b64encode(path.read_bytes()).decode("ascii")

    with _IMAGE_LOCK:
        with Image.open(path) as image:
            image = image.convert("RGB")
            if _IMAGE_SIZE and max(image.size) > _IMAGE_SIZE:
                image.thumbnail((_IMAGE_SIZE, _IMAGE_SIZE))
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=85)
    return "image/jpeg", base64.b64encode(output.getvalue()).decode("ascii")


def verify_image(path: Path) -> None:
    try:
        from PIL import Image
    except ImportError:
        if path.stat().st_size < 128:
            raise ValueError("image file is too small")
        return
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image.convert("RGB").load()


def preflight_jobs(jobs: list[dict[str, Any]]) -> None:
    failures = []
    checked = set()
    for job in jobs:
        for option in job["displayed"]:
            path = option["image_path"]
            if path in checked:
                continue
            checked.add(path)
            try:
                verify_image(path)
            except Exception as exc:
                failures.append(f"{path}: {type(exc).__name__}: {exc}")
    if failures:
        examples = "\n".join(f"  {failure}" for failure in failures[:10])
        raise SystemExit(
            f"[error] {len(failures)} selected images failed preflight:\n{examples}"
        )
    print(f"  Image preflight OK: {len(checked)} unique files for {len(jobs)} plans")


def build_messages(job: dict[str, Any]) -> list[dict[str, Any]]:
    sections = []
    if not job["profile_only"]:
        sections.append(f"=== QUERY ===\n{job['query_text']}")
    if job["profile_text"] is not None:
        sections.append(f"=== USER PROFILE ===\n{job['profile_text']}")
    sections.append(
        "=== OPTIONS ===\n"
        "Each option below includes both its text description and image."
    )
    content: list[dict[str, Any]] = [{
        "type": "text",
        "text": "\n\n".join(sections),
    }]
    for option in job["displayed"]:
        content.append({
            "type": "text",
            "text": f"Option {option['display_label']}: {option['text']}",
        })
        mime, encoded = encode_image(str(option["image_path"]))
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{encoded}"},
        })
    content.append({
        "type": "text",
        "text": final_instruction_for(
            job["profile_format"], job["profile_only"]
        ),
    })
    return [
        {
            "role": "system",
            "content": system_prompt_for(
                job["profile_format"], job["profile_only"]
            ),
        },
        {"role": "user", "content": content},
    ]


def call_vlm(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    guided: bool,
) -> str:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    extra_body: dict[str, Any] = {
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if guided:
        extra_body["guided_choice"] = list(DISPLAY_LABELS)
    kwargs["extra_body"] = extra_body
    response = client.chat.completions.create(**kwargs)
    message = response.choices[0].message
    return str(message.content or getattr(message, "reasoning", "") or "")


def evaluate_job(
    job_index: int,
    job: dict[str, Any],
    clients: list[Any],
    model: str,
    max_tokens: int,
    guided: bool,
) -> dict[str, Any]:
    started = time.monotonic()
    raw_response = None
    predicted = None
    error = None
    try:
        messages = build_messages(job)
        raw_response = call_vlm(
            clients[job_index % len(clients)], model, messages, max_tokens, guided
        )
        predicted = parse_answer(raw_response)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    predicted_original = job["display_to_original"].get(predicted)
    tpo_correct = predicted_original in TPO_CORRECT_ORIGINALS
    profile_correct = predicted_original in PROFILE_CORRECT_ORIGINALS
    strict_a_correct = predicted_original == "A"
    if job["profile_only"]:
        primary_correct = profile_correct
    elif uses_profile(job["profile_format"]):
        primary_correct = strict_a_correct
    else:
        primary_correct = tpo_correct
    plan = job["plan"]
    return {
        "schema_version": 2,
        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
        "plan_id": job["plan_id"],
        "query_id": plan.get("query_id"),
        "user_id": plan.get("user_id"),
        "track": plan.get("track"),
        "active_axis": plan.get("active_axis"),
        "scenario_id": plan.get("scenario_id"),
        "query_type": plan.get("query_type"),
        "profile_format": job["profile_format"],
        "profile_only": job["profile_only"],
        "display_to_original": job["display_to_original"],
        "scenario_correct_display": job["tpo_correct_display"],
        "joint_tpo_correct_display": job["strict_correct_display"],
        # Legacy name retained in append-only rows: this means A/B.
        "tpo_correct_display": job["tpo_correct_display"],
        "profile_correct_display": job["profile_correct_display"],
        "strict_correct_display": job["strict_correct_display"],
        "option_text": {
            option["display_label"]: option["text"] for option in job["displayed"]
        },
        "image_path": {
            option["display_label"]: portable_path(option["image_path"])
            for option in job["displayed"]
        },
        "predicted_display": predicted,
        "predicted_original": predicted_original,
        "accuracy": int(primary_correct),
        "primary_correct": bool(primary_correct),
        "scenario_correct": bool(tpo_correct),
        "joint_tpo_correct": bool(strict_a_correct),
        # Legacy axis name retained in append-only rows: this means A/B.
        "tpo_correct": bool(tpo_correct),
        "profile_correct": bool(profile_correct),
        "strict_a_correct": bool(strict_a_correct),
        "raw_response": raw_response,
        "error": error,
        "latency_seconds": round(time.monotonic() - started, 4),
    }


def load_latest_results(path: Path) -> dict[str, dict[str, Any]]:
    latest = {}
    if not path.is_file():
        return latest
    for row in load_jsonl(path):
        plan_id = row.get("plan_id")
        if plan_id:
            latest[str(plan_id)] = row
    return latest


def metric_cell(
    rows: list[dict[str, Any]],
    profile_format: str,
    profile_only: bool = False,
) -> dict[str, Any]:
    total = len(rows)
    scenario = sum(
        bool(row.get("scenario_correct", row.get("tpo_correct"))) for row in rows
    )
    joint_tpo = sum(
        bool(row.get("joint_tpo_correct", row.get("strict_a_correct")))
        for row in rows
    )
    profile = sum(
        bool(row.get("profile_correct"))
        if "profile_correct" in row
        else row.get("predicted_original") in PROFILE_CORRECT_ORIGINALS
        for row in rows
    )
    if profile_only:
        primary = profile
    elif uses_profile(profile_format):
        primary = joint_tpo
    else:
        primary = scenario
    parsed = sum(row.get("predicted_display") in DISPLAY_LABELS for row in rows)
    return {
        "n": total,
        "parsed": parsed,
        "invalid_response": total - parsed,
        "accuracy": primary / total if total else None,
        "primary_correct": primary,
        "profile_accuracy": profile / total if total else None,
        "profile_correct": profile,
        "scenario_accuracy": scenario / total if total else None,
        "scenario_correct": scenario,
        "tpo_accuracy": joint_tpo / total if total else None,
        "tpo_correct": joint_tpo,
        "strict_a_accuracy": joint_tpo / total if total else None,
        "strict_a_correct": joint_tpo,
        "legacy_tpo_axis_accuracy": scenario / total if total else None,
        "legacy_tpo_axis_correct": scenario,
    }


def summarize_track(
    track: str,
    expected_plan_ids: set[str],
    latest: dict[str, dict[str, Any]],
    profile_format: str,
    profile_only: bool,
) -> dict[str, Any]:
    relevant = {plan_id: latest[plan_id] for plan_id in expected_plan_ids if plan_id in latest}
    api_errors = [row for row in relevant.values() if row.get("error")]
    completed = [row for row in relevant.values() if not row.get("error")]
    axes = {}
    for axis in ("color", "garment_category", "pattern"):
        axes[axis] = metric_cell([
            row for row in completed if row.get("active_axis") == axis
        ], profile_format, profile_only)
    if profile_only:
        primary_metric = "Profile Accuracy = original A or C"
    elif uses_profile(profile_format):
        primary_metric = "TPO Accuracy = original A (query + profile)"
    else:
        primary_metric = "Scenario Accuracy = original A or B"
    return {
        "schema_version": 2,
        "track": track,
        "profile_format": profile_format,
        "profile_only": profile_only,
        "primary_metric": primary_metric,
        "n_expected": len(expected_plan_ids),
        "n_completed": len(completed),
        "n_api_errors": len(api_errors),
        "overall": metric_cell(completed, profile_format, profile_only),
        "by_active_axis": axes,
        "predicted_display_distribution": dict(sorted(Counter(
            row.get("predicted_display") or "invalid" for row in completed
        ).items())),
        "predicted_original_distribution": dict(sorted(Counter(
            row.get("predicted_original") or "invalid" for row in completed
        ).items())),
    }


def print_track_summary(summary: dict[str, Any]) -> None:
    overall = summary["overall"]
    profile_pct = 100.0 * overall["profile_accuracy"] if overall["profile_accuracy"] is not None else 0.0
    scenario_pct = 100.0 * overall["scenario_accuracy"] if overall["scenario_accuracy"] is not None else 0.0
    tpo_pct = 100.0 * overall["tpo_accuracy"] if overall["tpo_accuracy"] is not None else 0.0
    print("\n" + "=" * 72)
    print(f"  TRACK: {summary['track']}")
    primary_name = (
        "PROFILE" if summary["profile_only"]
        else "TPO" if uses_profile(summary["profile_format"])
        else "SCENARIO"
    )
    print(f"  PRIMARY: {primary_name} ACCURACY")
    print(f"  Profile Accuracy:  {overall['profile_correct']}/{overall['n']} = {profile_pct:.2f}%")
    print(f"  Scenario Accuracy: {overall['scenario_correct']}/{overall['n']} = {scenario_pct:.2f}%")
    print(f"  TPO Accuracy:      {overall['tpo_correct']}/{overall['n']} = {tpo_pct:.2f}%")
    print(f"  parsed: {overall['parsed']}/{overall['n']}  api_errors: {summary['n_api_errors']}")
    print("  By active axis (Profile / Scenario / TPO):")
    for axis, cell in summary["by_active_axis"].items():
        axis_label = "garment" if axis == "garment_category" else axis
        values = [
            100.0 * cell[key] if cell[key] is not None else 0.0
            for key in ("profile_accuracy", "scenario_accuracy", "tpo_accuracy")
        ]
        print(
            f"    {axis_label:<8} n={cell['n']:<4} "
            f"{values[0]:6.2f}% / {values[1]:6.2f}% / {values[2]:6.2f}%"
        )
    print("=" * 72)


def preview_job(job: dict[str, Any]) -> None:
    print("=" * 80)
    print(f"plan_id: {job['plan_id']}")
    print(f"track: {job['plan'].get('track')}  active_axis: {job['plan'].get('active_axis')}")
    if not job["profile_only"]:
        print(f"query: {job['query_text']}")
    if job["profile_text"] is not None:
        print(f"profile_format: {job['profile_format']}")
        print(f"profile:\n{job['profile_text']}")
    print(f"display_to_original: {job['display_to_original']}")
    print(f"Scenario-correct displayed labels: {job['tpo_correct_display']}")
    if job["profile_text"] is not None:
        print(f"profile-correct displayed labels: {job['profile_correct_display']}")
        if not job["profile_only"]:
            print(f"TPO-correct displayed label: {job['strict_correct_display']}")
    for option in job["displayed"]:
        print(
            f"  {option['display_label']}. {option['text']} -> "
            f"{portable_path(option['image_path'])}"
        )
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query + A/B/C/D (image+text) VLM evaluation."
    )
    parser.add_argument("--plans", type=Path, default=DEFAULT_PLANS)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--profile-format",
        choices=("none", "narrative", "key-value"),
        default="none",
        help="profile representation; none preserves the query-only condition",
    )
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="omit the query; requires narrative or key-value profile format",
    )
    parser.add_argument(
        "--experiment-name",
        help="default is derived from profile format and seed",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--base-urls",
        default="http://127.0.0.1:8002/v1",
        help="comma-separated OpenAI-compatible base URLs",
    )
    parser.add_argument(
        "--api-key", default=os.environ.get("OPENAI_API_KEY", "token-abc123")
    )
    parser.add_argument("--track", choices=("physical", "dress_code"))
    parser.add_argument("--plan-id-manifest", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates all survivors")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--guided", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="select and preflight plans, print one prompt mapping, then exit without writes or API calls",
    )
    args = parser.parse_args()
    if args.limit < 0 or args.concurrency <= 0 or args.image_size < 0:
        parser.error("--limit must be nonnegative; concurrency positive; image-size nonnegative")
    if args.max_tokens <= 0 or args.request_timeout <= 0 or args.log_every <= 0:
        parser.error("--max-tokens, --request-timeout, and --log-every must be positive")
    if args.profile_only and not uses_profile(args.profile_format):
        parser.error("--profile-only requires --profile-format narrative or key-value")
    if args.experiment_name is None:
        args.experiment_name = default_experiment_name(
            args.profile_format, args.seed, args.profile_only
        )
    return args


def main() -> int:
    args = parse_args()
    required_paths = [
        (args.plans, "plans"),
        (args.library, "attribute library"),
    ]
    if not args.profile_only:
        required_paths.append((args.queries, "queries"))
    for path, label in required_paths:
        if not path.is_file():
            raise SystemExit(f"[error] {label} not found: {path}")
    if uses_profile(args.profile_format) and not args.profiles.is_file():
        raise SystemExit(f"[error] profiles not found: {args.profiles}")

    plans = load_jsonl(args.plans)
    queries_list = load_jsonl(args.queries) if not args.profile_only else []
    queries = {str(row["query_id"]): row for row in queries_list}
    profiles_list = load_jsonl(args.profiles) if uses_profile(args.profile_format) else []
    profiles = {str(row["user_id"]): row for row in profiles_list}
    library = json.loads(args.library.read_text(encoding="utf-8"))
    plan_index = PlanIndex(plans)
    plan_index.assert_unique()

    manifest_meta = {}
    if args.plan_id_manifest:
        plans, manifest_meta = select_by_manifest(
            plans, args.plan_id_manifest, index=plan_index, label="plans"
        )

    survivors, paths_by_plan, selection_counts = select_surviving_plans(
        plans, library, args.library
    )
    if args.track:
        survivors = [plan for plan in survivors if plan.get("track") == args.track]
    if args.limit:
        survivors = survivors[:args.limit]
    if not survivors:
        raise SystemExit("[error] no strict surviving plans selected")

    jobs = [
        make_job(
            plan,
            queries,
            profiles,
            paths_by_plan[str(plan.get("plan_id") or plan.get("query_id"))],
            args.seed,
            args.profile_format,
            args.profile_only,
        )
        for plan in survivors
    ]
    missing_queries = [
        job["plan_id"] for job in jobs
        if not args.profile_only and not job["query_text"]
    ]
    if missing_queries:
        raise SystemExit(
            f"[error] {len(missing_queries)} plans have empty query text "
            f"(e.g. {missing_queries[:3]})"
        )
    missing_profiles = [
        job["plan_id"] for job in jobs
        if uses_profile(args.profile_format) and not job["profile_text"]
    ]
    if missing_profiles:
        raise SystemExit(
            f"[error] {len(missing_profiles)} plans have missing/empty profiles "
            f"(e.g. {missing_profiles[:3]})"
        )
    preflight_jobs(jobs)

    track_counts = Counter(str(job["plan"].get("track")) for job in jobs)
    print(f"  Plans loaded: {len(plans)}")
    print(f"  Strict selection: {dict(selection_counts)}")
    print(f"  Selected for this run: {len(jobs)} {dict(track_counts)}")
    print(f"  Profile format: {args.profile_format}")
    if args.profile_only:
        print("  Input condition: profile only (query omitted)")
        print("  Primary metric: Profile Accuracy (original A or C); random baseline 50%")
        print("  Also reported: Scenario Accuracy (A/B), TPO Accuracy (A)")
    elif uses_profile(args.profile_format):
        print("  Primary metric: TPO Accuracy (original A); random baseline 25%")
        print("  Also reported: Profile Accuracy (A/C), Scenario Accuracy (A/B)")
    else:
        print("  Primary metric: Scenario Accuracy (original A or B); random baseline 50%")
        print("  Also reported: Profile Accuracy (A/C), TPO Accuracy (A)")
    preview_job(jobs[0])
    if args.dry_run:
        print("  dry-run: no output files written and no VLM calls made")
        return 0

    try:
        import openai
    except ImportError as exc:
        raise SystemExit(
            "[error] openai package is required in the VLM evaluation environment"
        ) from exc

    base_urls = [url.strip() for url in args.base_urls.split(",") if url.strip()]
    if not base_urls:
        raise SystemExit("[error] --base-urls is empty")
    clients = [
        openai.OpenAI(
            base_url=url,
            api_key=args.api_key,
            timeout=args.request_timeout,
            max_retries=2,
        )
        for url in base_urls
    ]

    global _IMAGE_SIZE
    _IMAGE_SIZE = args.image_size or 0
    model_tag = sanitize(args.model)
    run_dir = args.output_dir / sanitize(args.experiment_name) / model_tag
    profile_enabled = uses_profile(args.profile_format)
    if args.profile_only:
        condition = (
            f"{args.profile_format} profile + A/B/C/D option image + option text; "
            "no query"
        )
        primary_metric = "profile accuracy: predicted original option is A or C"
    elif profile_enabled:
        condition = (
            f"query + {args.profile_format} profile + "
            "A/B/C/D option image + option text"
        )
        primary_metric = "joint query+profile accuracy: predicted original option is A"
    else:
        condition = "query + A/B/C/D option image + option text; no user profile"
        primary_metric = "TPO accuracy: predicted original option is A or B"
    semantic_config = {
        "schema_version": 1,
        "experiment_name": args.experiment_name,
        "condition": condition,
        "primary_metric": primary_metric,
        "strict_a_metric": (
            "primary" if profile_enabled and not args.profile_only else "diagnostic only"
        ),
        "model": args.model,
        "base_urls": base_urls,
        "seed": args.seed,
        "track_filter": args.track,
        "limit": args.limit,
        "image_size": args.image_size,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "guided_choice": bool(args.guided),
        "prompt_version": PROMPT_VERSION,
        "system_prompt": system_prompt_for(args.profile_format, args.profile_only),
        "final_instruction": final_instruction_for(
            args.profile_format, args.profile_only
        ),
        "plans": str(args.plans.resolve()),
        "plans_sha256": sha256_file(args.plans),
        "library": str(args.library.resolve()),
        "library_sha256": sha256_file(args.library),
        "plan_selection": dict(selection_counts),
        "n_selected_for_run": len(jobs),
        "track_counts": dict(track_counts),
        **manifest_meta,
    }
    if profile_enabled:
        semantic_config.update({
            "profile_format": args.profile_format,
            "profiles": str(args.profiles.resolve()),
            "profiles_sha256": sha256_file(args.profiles),
        })
    if not args.profile_only:
        semantic_config.update({
            "queries": str(args.queries.resolve()),
            "queries_sha256": sha256_file(args.queries),
        })
    config_path = run_dir / "run_config.json"
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != semantic_config:
            raise SystemExit(
                f"[error] refusing to mix a changed condition into {run_dir}; "
                "use a new --experiment-name"
            )
        print(f"  Resuming matching run: {run_dir}")
    else:
        write_json_atomic(config_path, semantic_config)

    jobs_by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        track = job["plan"].get("track")
        if track not in {"physical", "dress_code"}:
            raise SystemExit(f"[error] unknown/missing track in {job['plan_id']}: {track}")
        jobs_by_track[str(track)].append(job)

    summaries = {}
    incomplete_run = False
    for track in ("physical", "dress_code"):
        track_jobs = jobs_by_track.get(track) or []
        if not track_jobs:
            continue
        result_path = run_dir / track / "results.jsonl"
        latest = load_latest_results(result_path)
        completed_ids = {
            plan_id for plan_id, row in latest.items() if not row.get("error")
        }
        todo = [job for job in track_jobs if job["plan_id"] not in completed_ids]
        print(
            f"\n  track={track}: expected={len(track_jobs)} "
            f"completed={len(completed_ids)} todo={len(todo)}"
        )

        if todo:
            with cf.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures = {
                    executor.submit(
                        evaluate_job,
                        index,
                        job,
                        clients,
                        args.model,
                        args.max_tokens,
                        args.guided,
                    ): job
                    for index, job in enumerate(todo)
                }
                done = 0
                for future in cf.as_completed(futures):
                    record = future.result()
                    append_jsonl(result_path, record)
                    latest[record["plan_id"]] = record
                    done += 1
                    if done % args.log_every == 0 or done == len(todo):
                        mark = "error" if record.get("error") else (
                            "correct" if record.get("primary_correct") else "wrong"
                        )
                        print(
                            f"  [{done}/{len(todo)}] {mark} "
                            f"{record['plan_id']} pred={record['predicted_display']}"
                        )

        expected_ids = {job["plan_id"] for job in track_jobs}
        summary = summarize_track(
            track,
            expected_ids,
            latest,
            args.profile_format,
            args.profile_only,
        )
        write_json_atomic(run_dir / track / "summary.json", summary)
        summaries[track] = summary
        print_track_summary(summary)
        if summary["n_completed"] != summary["n_expected"]:
            incomplete_run = True

    combined_summary = {
        "schema_version": 2,
        "experiment_name": args.experiment_name,
        "model": args.model,
        "profile_format": args.profile_format,
        "profile_only": args.profile_only,
        "primary_metric": next(iter(summaries.values()))["primary_metric"],
        "note": "Tracks are intentionally reported separately; no pooled accuracy.",
        "tracks": summaries,
    }
    write_json_atomic(run_dir / "summary.json", combined_summary)
    print(f"\n  Results: {run_dir}")
    if incomplete_run:
        print("  [error] API failures remain; rerun the same command to resume them")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
