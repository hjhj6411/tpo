"""
common/config.py — yaml config + CLI merge + model registry.

Objective : single place for experiment configuration. Precedence:
            CLI args > --config yaml > configs/default.yaml built-ins.
Input     : configs/default.yaml, configs/models.yaml, argparse namespace.
Output    : flat dict cfg; resolve_model() -> {base_url, model_id, api_key, guided}.
Metrics   : n/a (infrastructure).
Run       : imported by every runner. Example CLI block each runner gets:
              --items data/bench/items.jsonl --model qwen3-vl-30b \
              --models-config exp/configs/models.yaml --seed 17 --limit 0
"""
import argparse
import os
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent.parent

BUILTIN_DEFAULTS = {
    "items": "data/bench/items.jsonl",
    "plans": "data/options/option_plans.jsonl",
    "results_root": "data/exp_results",
    "seed": 17,
    "limit": 0,                 # 0 = all
    "model": "qwen3-vl-30b",
    "models_config": str(_HERE / "configs" / "models.yaml"),
    "max_tokens": 16,
    "temperature": 0.0,
    "image_max_side": 768,      # uniform pre-resize across ALL providers
    "guided": None,             # None -> take from model registry
    "cache": True,
    "condition": "both",        # both | scenario_only | profile_only | none
    "profile_mode": "text",     # text | image | both
    "layout": "text_first",     # text_first | img_first
}


def add_common_args(p: argparse.ArgumentParser):
    p.add_argument("--config", type=str, default=None,
                   help="optional yaml overriding built-in defaults")
    p.add_argument("--items", type=str, default=None)
    p.add_argument("--results-root", type=str, default=None)
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="cap on items (0 = all)")
    p.add_argument("--model", type=str, default=None,
                   help="key in models.yaml")
    p.add_argument("--models-config", type=str, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--image-max-side", type=int, default=None)
    p.add_argument("--guided", dest="guided", action="store_true", default=None,
                   help="force vLLM guided_choice decoding")
    p.add_argument("--no-guided", dest="guided", action="store_false")
    p.add_argument("--no-cache", dest="cache", action="store_false", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="build + save prompts, no API calls")
    return p


def load_cfg(args: argparse.Namespace) -> dict:
    cfg = dict(BUILTIN_DEFAULTS)
    if getattr(args, "config", None):
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f) or {})
    for k, v in vars(args).items():
        kk = k.replace("-", "_")
        if v is not None and kk != "config":
            cfg[kk] = v
    return cfg


def resolve_model(cfg) -> dict:
    """models.yaml entry -> connection spec. api_key read from env var so
    keys never land in manifests."""
    with open(cfg["models_config"]) as f:
        reg = (yaml.safe_load(f) or {}).get("models", {})
    name = cfg["model"]
    if name not in reg:
        raise SystemExit(f"model '{name}' not in {cfg['models_config']} "
                         f"(have: {sorted(reg)})")
    m = dict(reg[name])
    env = m.get("api_key_env")
    m["api_key"] = os.environ.get(env) if env else None
    if cfg.get("guided") is not None:          # CLI override
        m["guided"] = bool(cfg["guided"])
    m.setdefault("guided", False)
    return m
