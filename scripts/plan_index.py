"""Shared plan_id indexing for every downstream consumer.

`plan_id` is the item key of the benchmark. 458 dress-code queries constrain two
axes and therefore emit two plans (`__vG`/`__vC`/`__vP`) that share a query_id
but differ in their options, so anything keyed on query_id silently collapses
916 plans (27%) into 458.

Image collections made before the plan_id switch used query_id folder names.
Those are only interpretable when the query produced exactly ONE plan; for a
two-plan query the folder is genuinely ambiguous and must not be guessed at.
`resolve_folder_key` implements exactly that rule, and every consumer shares it
so the eval path and the grading path can never disagree about which plan an
image belongs to.
"""
import hashlib
import json
from collections import defaultdict


class PlanIndex:
    """plan_id -> plan, plus the query_id -> [plan_id] map used for fallback."""

    def __init__(self, plans):
        self.plans = plans
        self.by_plan_id = {}
        self.qid_to_pids = defaultdict(list)
        for p in plans:
            pid = p.get("plan_id") or p["query_id"]
            self.by_plan_id[pid] = p
            self.qid_to_pids[p["query_id"]].append(pid)

    def __contains__(self, pid):
        return pid in self.by_plan_id

    def __len__(self):
        return len(self.by_plan_id)

    def assert_unique(self):
        """Fail loudly if plan_ids are not unique — every key below assumes it."""
        pids = [p.get("plan_id") or p["query_id"] for p in self.plans]
        dupes = [k for k, n in _counts(pids).items() if n > 1]
        if dupes:
            raise SystemExit(
                f"  [error] plan_id is not unique: {len(dupes)} duplicated "
                f"(e.g. {dupes[:3]}). The item key is broken; refusing to run.")
        return len(pids)

    def resolve_folder_key(self, name):
        """Map a collection folder name to a plan_id.

        Returns (plan_id, how) where `how` is "plan_id", "query_id" (legacy
        single-plan fallback), "ambiguous" (query_id of a two-plan query — the
        caller must skip it), or "unmatched".
        """
        pid = _longest_contained(name, self.by_plan_id)
        if pid is not None:
            return pid, "plan_id"
        qid = _longest_contained(name, self.qid_to_pids)
        if qid is None:
            return None, "unmatched"
        pids = self.qid_to_pids[qid]
        if len(pids) == 1:
            return pids[0], "query_id"
        return None, "ambiguous"


def _counts(items):
    out = defaultdict(int)
    for i in items:
        out[i] += 1
    return out


def _longest_contained(name, keys):
    best = None
    for k in keys:
        if k in name and (best is None or len(k) > len(best)):
            best = k
    return best


def load_plan_id_manifest(path, index):
    """Load a counterbalanced-subset manifest and validate it against the plans.

    Expects the object written by scripts/validate_options.py:
        {"id_kind": "plan_id", "n": <int>, "ids": [...]}
    A bare list is rejected: that is the pre-2026-07-27 query_id format, whose
    ids do not address a single plan.

    Returns (ids, sha256).
    """
    raw = open(path, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    doc = json.loads(raw.decode("utf-8"))

    if isinstance(doc, list):
        raise SystemExit(
            f"  [error] {path} is a bare list — that is the legacy query_id "
            f"manifest. Re-run scripts.validate_options to emit plan_ids.")
    if not isinstance(doc, dict) or doc.get("id_kind") != "plan_id":
        raise SystemExit(
            f"  [error] {path}: id_kind must be 'plan_id', got "
            f"{doc.get('id_kind') if isinstance(doc, dict) else type(doc).__name__!r}.")

    ids = doc.get("ids")
    if not isinstance(ids, list) or not ids:
        raise SystemExit(f"  [error] {path}: 'ids' is missing or empty.")

    dupes = [k for k, n in _counts(ids).items() if n > 1]
    if dupes:
        raise SystemExit(f"  [error] {path}: {len(dupes)} duplicated ids "
                         f"(e.g. {dupes[:3]}).")

    unknown = [i for i in ids if i not in index]
    if unknown:
        raise SystemExit(f"  [error] {path}: {len(unknown)} ids are not in the "
                         f"plans file (e.g. {unknown[:3]}).")

    declared = doc.get("n")
    if declared is not None and declared != len(ids):
        raise SystemExit(f"  [error] {path}: declared n={declared} but "
                         f"{len(ids)} ids present.")
    return ids, sha


def select_by_manifest(plans, manifest_path, index=None, label="plans"):
    """Filter `plans` down to a validated manifest. Returns (plans, meta)."""
    index = index or PlanIndex(plans)
    ids, sha = load_plan_id_manifest(manifest_path, index)
    keep = set(ids)
    selected = [p for p in plans
                if (p.get("plan_id") or p["query_id"]) in keep]
    print(f"  Manifest: {manifest_path}")
    print(f"    sha256={sha}")
    print(f"    {len(ids)} plan_ids -> selected {len(selected)}/{len(plans)} {label}")
    return selected, {"plan_id_manifest": str(manifest_path),
                      "plan_id_manifest_sha256": sha,
                      "plan_id_manifest_n": len(ids)}
