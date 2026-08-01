#!/usr/bin/env python3
"""Retract annotation verdicts for a set of cells so they can be re-annotated.

Nothing is deleted. A `reset` record is APPENDED to ``cell_annotations.jsonl``
carrying the verdict it supersedes; ``serve_annotator.py`` treats a cell whose
most recent verdict-or-reset is a reset as pending, so the cell returns to the
queue and the old decision stays readable in the log.

Written for the ``wool coat`` garment-vocabulary contamination
(``docs/PROCESS.md`` §8): the ``long_coat`` and ``pea_coat`` cells were screened
under a closed VLM vocabulary that offered a hypernym of both categories, so
their candidate pools — and therefore the verdicts made on them — are not
trustworthy. Re-screen those cells first, then reset, then re-annotate.

    # see what would change (default: writes nothing)
    python -m annotation.reset_cells --garments long_coat,pea_coat

    # actually append the reset records
    python -m annotation.reset_cells --garments long_coat,pea_coat --apply \
        --reason "rescreened under corrected garment vocabulary (PROCESS.md 8)"

The library is rebuilt from the log the next time the annotator starts, so this
script never writes ``attribute_library.json`` itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import FASHION_ATTRIBUTE_AXES

HERE = Path(__file__).resolve().parent
ANNOTATION_FILE = HERE / "cell_annotations.jsonl"
LIBRARY_FILE = HERE / "attribute_library.json"
VERDICTS = {"selected", "excluded"}
SETTLING = VERDICTS | {"reset"}


def load_records(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"[error] no annotation log at {path}")
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"[error] {path}:{n} is not valid JSON: {exc}")
    return out


def cell_key(color: str, garment: str, pattern: str) -> str:
    return f"{color}|{garment}|{pattern}"


def latest_settling(records: list[dict]) -> dict[str, dict]:
    """Most recent selected/excluded/reset record per cell, in file order."""
    latest: dict[str, dict] = {}
    for record in records:
        if record.get("decision") not in SETTLING:
            continue
        latest[cell_key(str(record.get("color") or ""),
                        str(record.get("garment") or ""),
                        str(record.get("pattern") or ""))] = record
    return latest


def superseded_view(record: dict) -> dict:
    """The part of the retracted verdict worth keeping next to the reset."""
    view = {
        "record_id": record.get("record_id"),
        "decision": record.get("decision"),
        "ts": record.get("ts"),
        "annotator": record.get("annotator"),
    }
    if record.get("decision") == "excluded":
        view["reason"] = record.get("reason")
    else:
        view["images"] = [p.get("library_path") for p in record.get("picked") or []]
    return view


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--garments", default="",
                    help="comma-separated canonical garments to reset, "
                         "e.g. long_coat,pea_coat")
    ap.add_argument("--cells", default="",
                    help="comma-separated color|garment|pattern keys; combines "
                         "with --garments")
    ap.add_argument("--reason", default="",
                    help="why the verdicts are being retracted (recorded)")
    ap.add_argument("--annotator", default="reset_cells",
                    help="recorded as the author of the reset records")
    ap.add_argument("--annotations", type=Path, default=ANNOTATION_FILE)
    ap.add_argument("--library", type=Path, default=LIBRARY_FILE,
                    help="read-only; used to restrict resets to annotation "
                         "targets and to report the current status")
    ap.add_argument("--apply", action="store_true",
                    help="append the records; without it nothing is written")
    args = ap.parse_args()

    garments = {g.strip() for g in args.garments.split(",") if g.strip()}
    unknown = garments - set(FASHION_ATTRIBUTE_AXES["garment_category"])
    if unknown:
        raise SystemExit(f"[error] not canonical garments: {sorted(unknown)}")
    explicit = {c.strip() for c in args.cells.split(",") if c.strip()}
    if not garments and not explicit:
        raise SystemExit("[error] give --garments and/or --cells")

    records = load_records(args.annotations)
    latest = latest_settling(records)

    library = {}
    if args.library.is_file():
        library = json.loads(args.library.read_text(encoding="utf-8"))
    target_keys = set(library.get("cells") or {})
    if not target_keys:
        raise SystemExit(f"[error] no cells found in {args.library}")

    bad = explicit - target_keys
    if bad:
        raise SystemExit(f"[error] --cells not in the library: {sorted(bad)}")

    wanted = set(explicit)
    for key in target_keys:
        if key.split("|")[1] in garments:
            wanted.add(key)

    to_reset, already_pending = [], []
    for key in sorted(wanted):
        record = latest.get(key)
        if record is None or record.get("decision") == "reset":
            already_pending.append(key)
        else:
            to_reset.append((key, record))

    status = Counter((library.get("cells", {}).get(k) or {}).get("status")
                     for k in sorted(wanted))
    print(f"  target cells        : {len(wanted)}")
    print(f"  current library status: "
          + ", ".join(f"{k}={v}" for k, v in sorted(status.items(), key=str)))
    print(f"  verdicts to retract : {len(to_reset)}")
    print(f"  already pending     : {len(already_pending)}")
    by_decision = Counter(r.get("decision") for _, r in to_reset)
    if by_decision:
        print("  retracting          : "
              + ", ".join(f"{k}={v}" for k, v in sorted(by_decision.items(), key=str)))

    if not to_reset:
        print("  nothing to do")
        return 0

    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    new_records = []
    for key, record in to_reset:
        color, garment, pattern = key.split("|")
        rid = hashlib.sha256(
            f"reset|{key}|{ts}|{time.time_ns()}".encode("utf-8")
        ).hexdigest()[:16]
        new_records.append({
            "record_id": rid,
            "color": color,
            "garment": garment,
            "pattern": pattern,
            "decision": "reset",
            "reason": args.reason,
            "superseded": superseded_view(record),
            "annotator": args.annotator,
            "ts": ts,
        })

    if not args.apply:
        print("\n  dry run — nothing written. Re-run with --apply.")
        print("  sample record:")
        print("   " + json.dumps(new_records[0], ensure_ascii=False))
        return 0

    with args.annotations.open("a", encoding="utf-8") as fh:
        for record in new_records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\n  appended {len(new_records)} reset records -> {args.annotations}")
    print("  the library is rebuilt from the log when serve_annotator.py next "
          "starts (without --dry-run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
