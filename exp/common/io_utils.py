"""
common/io_utils.py — structured result directories + jsonl IO + manifests.

Objective : every experiment writes into results/<exp_name>/<timestamp>_<tag>/
            with a manifest.json capturing config + seed for reproducibility.
Input     : python dicts / jsonl paths.
Output    : run dir Path, results.jsonl, analysis.json, manifest.json.
Metrics   : n/a (infrastructure).
Run       : not executable; imported by every runner.
"""
import hashlib
import json
import subprocess
import time
from pathlib import Path


def read_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def sha_short(s, n=10):
    return hashlib.sha256(str(s).encode()).hexdigest()[:n]


def _git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def make_run_dir(results_root, exp_name, tag=""):
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = f"{ts}_{tag}" if tag else ts
    d = Path(results_root) / exp_name / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_manifest(run_dir, cfg_dict, extra=None):
    m = {"config": cfg_dict, "git": _git_hash(),
         "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    if extra:
        m.update(extra)
    write_json(Path(run_dir) / "manifest.json", m)
    return m
