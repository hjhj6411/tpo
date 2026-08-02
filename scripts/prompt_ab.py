#!/usr/bin/env python3
"""Compare garment-classifier PROMPTS against the human-approved labels.

WHY
---
Changing VLM_GARMENT_PROMPT changes the garment verdict for every cell, and a
full re-screen costs hours of GPU plus a six-hour annotation pass. This scores
candidate prompts in minutes instead, on images whose correct answer is already
known: the crops a human selected in annotation/attribute_library.json.

WHAT IT MEASURES
----------------
For every sampled crop, the target garment is the one the human approved. A
prompt is right when the model returns exactly that category. Accuracy is
reported per garment, so a prompt that helps t-shirts while starving pea coats
is visible rather than averaged away.

This is NOT a benchmark of the model. It is a regression test on the prompt:
the images all passed the OLD prompt and a human, so any prompt that scores
much lower on them is rejecting garments it should accept.

USAGE
    conda activate pod        # no GPU, no SAM3, no torch
    python -m scripts.prompt_ab --per-garment 20

    # smoke test first
    python -m scripts.prompt_ab --per-garment 2

Runtime: (23 garments x N) x (number of prompts) requests, spread over the
vLLM ports. At N=20 and three prompts that is ~1,400 calls; with four engines
it lands in a few minutes.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from configs.config import FASHION_ATTRIBUTE_AXES

GARMENT_VOCAB = [g.replace("_", " ")
                 for g in FASHION_ATTRIBUTE_AXES["garment_category"]]

HEAD = (
    "You are a fashion garment classifier. Look only at the single main clothing "
    "item worn on the body in the image, and ignore the background, the person, "
    "the pose, and any accessories.\n\n"
    "Classify that garment into exactly ONE of these categories:\n"
    "{options}\n\n"
)
TAIL = (
    "- Choose from only the listed query garment categories; do not introduce "
    "another category.\n"
    "Answer with only the category name, exactly as written above, and nothing else."
)
BARE_TAIL = (
    "Answer with only the category name, exactly as written above, and nothing else."
)

# A — what is in the tree right now (the change that cost 12 of 15 pea coats)
PROMPT_A = HEAD + (
    "Rules:\n"
    "- Outerwear, keep these distinct. Judge coats by LENGTH and CLOSURE, never by "
    "fabric (both coats below are usually wool): a pea coat is hip-length and "
    "double-breasted with wide lapels; a long coat reaches the knee or below and is "
    "usually single-breasted; a fleece jacket is a soft brushed "
    "insulating jacket; a puffer jacket is "
    "a quilted insulated jacket; a windbreaker is a thin lightweight nylon "
    "shell; a leather jacket is a leather outerwear jacket; a blazer is a "
    "structured tailored jacket worn on its own.\n"
    "- A suit vest is a sleeveless tailored waistcoat, not a fleece or puffer vest.\n"
) + TAIL

# B — the original v4 wording, with only the two changes that were actually
#     required: the self-contradiction removed and the hypernym renamed.
PROMPT_B = HEAD + (
    "Rules:\n"
    "- Outerwear, keep these distinct: a pea coat is short and double-breasted; "
    "a long coat reaches roughly the knee or below; a fleece jacket is a soft brushed "
    "insulating jacket; a puffer jacket is "
    "a quilted insulated jacket; a windbreaker is a thin lightweight nylon "
    "shell; a leather jacket is a leather outerwear jacket; a blazer is a "
    "structured tailored jacket worn on its own.\n"
    "- A suit vest is a sleeveless tailored waistcoat, not a fleece or puffer vest.\n"
) + TAIL

# C — no definitions at all: the closed option list is the whole instruction
PROMPT_C = HEAD + BARE_TAIL

# D — C plus the one rule that is a misclassification guard rather than a
#     definition, so C-vs-D isolates whether that guard is doing any work
PROMPT_D = HEAD + (
    "Rules:\n"
    "- A suit vest is a sleeveless tailored waistcoat, not a fleece or puffer vest.\n"
) + BARE_TAIL

PROMPTS = {"A_current": PROMPT_A, "B_minimal": PROMPT_B,
           "C_bare": PROMPT_C, "D_bare_plus_vest": PROMPT_D}


def norm(s: str) -> str:
    return " ".join(str(s or "").replace("_", " ").strip().lower().split())


def parse_pred(text: str, vocab: list[str]) -> str | None:
    """Longest match wins, so 'tank top' beats 'top' — same rule the pipeline uses."""
    hay = f" {norm(text)} "
    hits = [v for v in vocab if f" {norm(v)} " in hay]
    if not hits:
        return None
    return max(hits, key=lambda v: (len(norm(v).split()), len(v)))


def data_uri(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".") or "jpeg"
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64," + base64.b64encode(path.read_bytes()).decode()


def resolve(rel: str, bases: list[Path]) -> Path | None:
    for b in bases:
        p = (b / rel)
        if p.is_file():
            return p
    return None


def collect(library: Path, bases: list[Path], per_garment: int, seed: int):
    lib = json.loads(library.read_text())
    by_garment = defaultdict(list)
    missing = 0
    for key, cell in lib.get("cells", {}).items():
        if cell.get("status") != "available":
            continue
        color, garment, pattern = key.split("|")
        for img in cell.get("images") or []:
            p = resolve(str(img.get("path") or ""), bases)
            if p is None:
                missing += 1
                continue
            by_garment[garment].append((p, garment, color, pattern))
            break                      # one crop per cell is enough
    rng = random.Random(seed)
    out = []
    for g in sorted(by_garment):
        items = by_garment[g]
        rng.shuffle(items)
        out.extend(items[:per_garment])
    return out, missing, {g: len(v) for g, v in by_garment.items()}


def classify(session, url, model, prompt, uri, timeout, retries=3):
    payload = {"model": model,
               "messages": [{"role": "user", "content": [
                   {"type": "text", "text": prompt},
                   {"type": "image_url", "image_url": {"url": uri}}]}],
               "max_tokens": 16, "temperature": 0.0}
    last = None
    for _ in range(retries):
        try:
            r = session.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            last = e
    raise RuntimeError(last)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", type=Path,
                    default=REPO / "annotation" / "attribute_library.json")
    ap.add_argument("--image-base", action="append", default=[],
                    help="where the library's relative paths live; repeatable")
    ap.add_argument("--per-garment", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vlm-ports", default="8001,8002,8003,8004")
    ap.add_argument("--vlm-model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--prompts", default="A_current,B_minimal,C_bare,D_bare_plus_vest")
    ap.add_argument("--out", type=Path, default=REPO / "_local" / "prompt_ab_results.json")
    args = ap.parse_args()

    bases = [Path(b) for b in args.image_base] or [
        REPO / "availability_audit" / "images",
        REPO / "annotation",
        REPO,
    ]
    urls = [f"http://127.0.0.1:{p.strip()}/v1/chat/completions"
            for p in args.vlm_ports.split(",") if p.strip()]
    names = [n.strip() for n in args.prompts.split(",") if n.strip()]
    for n in names:
        if n not in PROMPTS:
            raise SystemExit(f"[error] unknown prompt {n!r}; have {list(PROMPTS)}")

    items, missing, counts = collect(args.library, bases, args.per_garment, args.seed)
    if not items:
        raise SystemExit(
            "[error] no crops resolved. Pass --image-base pointing at the "
            f"directory that contains 'images_final/'. Tried: {bases}")
    print(f"  crops: {len(items)} sampled from {len(counts)} garments"
          f"   (unresolved paths: {missing})")
    thin = {g: n for g, n in counts.items() if n < args.per_garment}
    if thin:
        print(f"  fewer than {args.per_garment} available: {thin}")
    print(f"  prompts: {', '.join(names)}   endpoints: {len(urls)}\n")

    uris = {}
    for p, *_ in items:
        if p not in uris:
            uris[p] = data_uri(p)

    session = requests.Session()
    session.mount("http://", requests.adapters.HTTPAdapter(
        pool_connections=args.workers * 2, pool_maxsize=args.workers * 2))

    results = {}
    for name in names:
        prompt = PROMPTS[name].format(
            options="\n".join(f"- {c}" for c in GARMENT_VOCAB))

        def one(i_item):
            i, (path, garment, color, pattern) = i_item
            url = urls[i % len(urls)]
            try:
                raw = classify(session, url, args.vlm_model, prompt,
                               uris[path], args.timeout)
            except Exception as e:
                return {"garment": garment, "pred": None,
                        "error": f"{type(e).__name__}: {str(e)[:120]}",
                        "path": str(path), "color": color, "pattern": pattern}
            pred = parse_pred(raw, GARMENT_VOCAB)
            return {"garment": garment, "pred": pred, "raw": str(raw)[:80],
                    "path": str(path), "color": color, "pattern": pattern}

        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            rows = list(ex.map(one, enumerate(items)))
        results[name] = rows
        acc = sum(1 for r in rows if r["pred"] == norm(r["garment"])) / len(rows)
        err = sum(1 for r in rows if r.get("error"))
        print(f"  {name:18s} overall {acc:6.1%}"
              + (f"   [{err} request errors]" if err else ""))

    # ── per-garment table ────────────────────────────────────────────────
    garments = sorted({r["garment"] for r in results[names[0]]})
    w = max(len(g) for g in garments) + 1
    print("\n" + "=" * (w + 6 + 11 * len(names)))
    print(f"{'garment':<{w}}{'n':>5}" + "".join(f"{n[:10]:>11}" for n in names))
    print("-" * (w + 6 + 11 * len(names)))
    for g in garments:
        n = sum(1 for r in results[names[0]] if r["garment"] == g)
        line = f"{g:<{w}}{n:>5}"
        for name in names:
            sub = [r for r in results[name] if r["garment"] == g]
            hit = sum(1 for r in sub if r["pred"] == norm(g))
            line += f"{hit/max(len(sub),1):>10.0%} "
        print(line)
    print("-" * (w + 6 + 11 * len(names)))
    line = f"{'ALL':<{w}}{len(items):>5}"
    for name in names:
        rows = results[name]
        line += f"{sum(1 for r in rows if r['pred']==norm(r['garment']))/len(rows):>10.0%} "
    print(line)
    print("=" * (w + 6 + 11 * len(names)))

    # ── where the misses go ──────────────────────────────────────────────
    print("\n  most common confusions (true -> predicted), per prompt:")
    for name in names:
        conf = Counter((r["garment"], r["pred"]) for r in results[name]
                       if r["pred"] != norm(r["garment"]))
        top = ", ".join(f"{t}->{p}={c}" for (t, p), c in conf.most_common(5))
        print(f"    {name:18s} {top or '(none)'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"vocab": GARMENT_VOCAB, "per_garment": args.per_garment,
         "seed": args.seed, "results": results}, indent=1))
    print(f"\n  raw rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
