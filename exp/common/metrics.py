"""
common/metrics.py — benchmark semantics + statistics.

Objective : decomposed accuracy (strict / TPO / preference) + position-bias
            statistics + paired tests, all in one importable module.
Semantics (fixed by construction):
  canonical A = tpo+pref -> strict correct
  TPO satisfied  iff canonical choice in {A, B}
  pref satisfied iff canonical choice in {A, C}
Statistics:
  wilson(k,n)            : 95% binomial CI (Wilson)
  chi2_uniform(counts)   : position-preference vs uniform (stat, dof, p)
  mcnemar(b, c)          : exact two-sided paired test
  cochran_q(matrix)      : k paired binary conditions (accuracy by gold pos)
  robust_accuracy        : fraction of qids correct under ALL permutation runs
Input     : records from common/clients.execute_tasks (+ 'order' meta).
Output    : plain dicts (json-serializable).
Run       : imported; covered by tests/test_exp_suite.py.
"""
import math
from collections import Counter, defaultdict

try:
    from scipy.stats import chi2 as _chi2
    def _chi2_sf(x, df):
        return float(_chi2.sf(x, df))
except Exception:                                  # pragma: no cover
    def _chi2_sf(x, df):
        return None

from .data import CANON

_TPO_OK = {"A", "B"}
_PREF_OK = {"A", "C"}


def canon_choice(display_letter, order, letters):
    """Display letter -> canonical option letter via this task's order."""
    if display_letter is None:
        return None
    pos = list(letters).index(display_letter)
    return CANON[order[pos]]


def score(canon_letter):
    if canon_letter is None:
        return {"strict": False, "tpo": False, "pref": False, "parsed": False}
    return {"strict": canon_letter == "A",
            "tpo": canon_letter in _TPO_OK,
            "pref": canon_letter in _PREF_OK,
            "parsed": True}


def attach_scores(records):
    for r in records:
        c = canon_choice(r.get("choice_display"), r["order"], r["letters"])
        r["choice_canon"] = c
        r.update(score(c))
    return records


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(p, 4), round(c - h, 4), round(c + h, 4))


def acc_summary(records):
    n = len(records)
    out = {"n": n}
    for m in ("strict", "tpo", "pref"):
        k = sum(int(r[m]) for r in records)
        p, lo, hi = wilson(k, n)
        out[m] = {"acc": p, "ci95": [lo, hi], "k": k}
    out["parse_fail_rate"] = round(
        sum(1 for r in records if not r["parsed"]) / max(n, 1), 4)
    return out


def position_matrix(records, letters=("A", "B", "C", "D")):
    cnt = Counter(r.get("choice_display") for r in records)
    counts = [cnt.get(l, 0) for l in letters]
    return {"letters": list(letters), "counts": counts,
            **chi2_uniform(counts)}


def chi2_uniform(counts):
    n = sum(counts)
    if n == 0:
        return {"chi2": 0.0, "dof": len(counts) - 1, "p": None}
    e = n / len(counts)
    stat = sum((o - e) ** 2 / e for o in counts)
    dof = len(counts) - 1
    return {"chi2": round(stat, 3), "dof": dof, "p": _chi2_sf(stat, dof)}


def mcnemar(b, c):
    """b = cond1-only successes, c = cond2-only successes; exact two-sided."""
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p": 1.0}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return {"b": b, "c": c, "p": round(min(1.0, 2 * tail), 6)}


def paired_mcnemar(rec1, rec2, metric="strict", key="query_id"):
    m1 = {r[key]: r[metric] for r in rec1}
    m2 = {r[key]: r[metric] for r in rec2}
    common = sorted(set(m1) & set(m2))
    b = sum(1 for q in common if m1[q] and not m2[q])
    c = sum(1 for q in common if m2[q] and not m1[q])
    return {**mcnemar(b, c), "n_pairs": len(common)}


def cochran_q(matrix):
    """matrix: rows = subjects (qids), cols = conditions, binary."""
    k = len(matrix[0])
    G = [sum(row[j] for row in matrix) for j in range(k)]
    L = [sum(row) for row in matrix]
    num = (k - 1) * (k * sum(g * g for g in G) - sum(G) ** 2)
    den = k * sum(L) - sum(l * l for l in L)
    if den == 0:
        return {"Q": 0.0, "dof": k - 1, "p": None}
    Q = num / den
    return {"Q": round(Q, 3), "dof": k - 1, "p": _chi2_sf(Q, k - 1)}


def acc_by_gold_position(records, letters=("A", "B", "C", "D")):
    """Accuracy grouped by WHERE the gold sat. Cochran's Q over qids that
    appear at every gold position (cyclic / fixed_gold give exactly that)."""
    by_pos = defaultdict(list)
    by_qid = defaultdict(dict)
    for r in records:
        gp = r["order"].index(0)            # display position of canonical A
        by_pos[gp].append(int(r["strict"]))
        by_qid[r["query_id"]][gp] = int(r["strict"])
    out = {}
    for pos in sorted(by_pos):
        k, n = sum(by_pos[pos]), len(by_pos[pos])
        p, lo, hi = wilson(k, n)
        out[letters[pos]] = {"acc": p, "ci95": [lo, hi], "n": n}
    full = [[row[p] for p in range(len(letters))]
            for row in by_qid.values() if len(row) == len(letters)]
    q = cochran_q(full) if full else None
    accs = [v["acc"] for v in out.values()]
    return {"by_gold_pos": out, "spread": round(max(accs) - min(accs), 4)
            if accs else None, "cochran_q": q}


def robust_accuracy(records):
    """Fraction of qids strict-correct under ALL their permutation runs —
    the 'no position luck' accuracy. Gap vs mean accuracy = luck share."""
    by_qid = defaultdict(list)
    for r in records:
        by_qid[r["query_id"]].append(bool(r["strict"]))
    n = len(by_qid)
    robust = sum(1 for v in by_qid.values() if all(v)) / max(n, 1)
    mean = sum(int(r["strict"]) for r in records) / max(len(records), 1)
    return {"n_items": n, "mean_acc": round(mean, 4),
            "robust_acc": round(robust, 4),
            "luck_gap": round(mean - robust, 4)}
