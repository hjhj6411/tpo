"""
common/data.py — benchmark item / option-plan loading + schema validation.

Objective : ONE place that knows the data schema. If the repo's real field
            names differ, adapt them HERE and every experiment follows.
Input     : items jsonl (benchmark questions), option_plans.jsonl (collection).
Output    : list[dict] with validated fields.

ITEM SCHEMA (one MCQ question per line):
{
  "query_id":   "U001__cold_blizzard_outdoor__pattern__0002",
  "active_axis":"pattern",                      # pattern | color
  "scenario_text": "...",                       # TPO scenario description
  "profile_text":  "...",                       # text-attribute profile (opt)
  "profile_images": ["url|path", ...],          # image-instance profile (opt)
  "profile_entries": [                          # dialogue source (e6, opt)
      {"explicit": "This yellow shirt's color is my favorite.",
       "implicit": "This shirt's color is my favorite.",
       "image": "url|path"}
  ],
  "options": {"A": {"image": "..."}, "B": {...}, "C": {...}, "D": {...}},
  "gold": "A"                                   # default "A" (= tpo+pref)
}
Canonical option semantics (fixed by benchmark construction):
  A = tpo+pref (gold) | B = tpo_only | C = pref_only | D = neither

Metrics   : n/a (infrastructure).
Run       : imported; `python -m common.data --items x.jsonl` validates a file.
"""
import argparse

from .io_utils import read_jsonl

CANON = ["A", "B", "C", "D"]
SEMANTICS = {"A": "tpo+pref", "B": "tpo_only", "C": "pref_only", "D": "neither"}


def _err(i, qid, msg):
    raise SystemExit(f"[data] item #{i} ({qid}): {msg}")


def load_items(path, limit=0, need_entries=False, need_options=CANON):
    rows = read_jsonl(path)
    if limit:
        rows = rows[:limit]
    for i, it in enumerate(rows):
        qid = it.get("query_id", "?")
        if "query_id" not in it:
            _err(i, qid, "missing query_id")
        if "scenario_text" not in it:
            _err(i, qid, "missing scenario_text")
        opts = it.get("options") or {}
        for k in need_options:
            if k not in opts or "image" not in opts[k]:
                _err(i, qid, f"options.{k}.image missing")
        it.setdefault("gold", "A")
        it.setdefault("active_axis", "unknown")
        if need_entries:
            ents = it.get("profile_entries") or []
            if not ents:
                _err(i, qid, "profile_entries required for dialogue experiment")
            for j, e in enumerate(ents):
                for f in ("explicit", "implicit", "image"):
                    if f not in e:
                        _err(i, qid, f"profile_entries[{j}].{f} missing")
    return rows


def load_plans(path, limit=0):
    rows = read_jsonl(path)
    if limit:
        rows = rows[:limit]
    for i, p in enumerate(rows):
        if "query_id" not in p or "options" not in p:
            raise SystemExit(f"[data] plan #{i}: needs query_id + options")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--items")
    ap.add_argument("--plans")
    a = ap.parse_args()
    if a.items:
        print(f"items ok: {len(load_items(a.items))}")
    if a.plans:
        print(f"plans ok: {len(load_plans(a.plans))}")
