"""Phase 3 — feasibility interface for the option planner.

The planner imports exactly these functions:

    from availability_audit.feasibility import (
        load_support, option_set_is_retrievable, set_support, rarity_cost,
        pick_fixed_color,
    )

Hard constraint : all four (garment, pattern) edges of the 2x2 rectangle must
                  meet their usage-aware minimum support.
Soft penalty    : rarity_cost = 1 / (min(s_A, s_B, s_C, s_D) + eps), added to
                  the existing balance + confusability + repetition objective.

Support source: corpus_support.json (default) — pipeline_yield.json can be
passed instead to plan strictly within what the SAM3 pipeline yields today,
but that couples the benchmark to segmentation bias; prefer corpus support
and fix masking where corpus > yield (see README).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from availability_audit.audit_config import CORPUS_SUPPORT_PATH, n_min

_SUPPORT = None


def load_support(path=CORPUS_SUPPORT_PATH):
    global _SUPPORT
    _SUPPORT = json.loads(Path(path).read_text())
    return _SUPPORT


def support(garment, pattern, color=None):
    """Distinct verified candidates for (garment, pattern[, color])."""
    if _SUPPORT is None:
        raise RuntimeError("call load_support() first")
    cell = _SUPPORT.get(garment, {}).get(pattern)
    if cell is None:
        return 0
    if color is None:
        return cell.get("n", 0)
    return cell.get("colors", {}).get(color, 0)


def minimum_support(planned_uses=1, max_reuse=2, buffer=2):
    return n_min(planned_uses, max_reuse, buffer)


def _edge_support(garment, value, axis, fixed_attr):
    """Support of one 2x2 edge.

    axis="pattern": value is a pattern, fixed_attr is the fixed color (or None).
    axis="color"  : value is a color, fixed_attr is the fixed pattern; when the
                    pattern is unfixed, fall back to pattern-marginal support.
    """
    if axis == "pattern":
        return support(garment, value, fixed_attr)
    if fixed_attr:
        return support(garment, fixed_attr, value)
    return sum(support(garment, p, value) for p in _SUPPORT.get(garment, {}))


def option_set_is_retrievable(a_value, b_value, compat_garment, incompat_garment,
                              axis="pattern", fixed_attr=None,
                              planned_uses=1, max_reuse=2, buffer=2):
    """All four 2x2 edges must exist with at least the usage-aware minimum."""
    need = minimum_support(planned_uses, max_reuse, buffer)
    return all(_edge_support(g, v, axis, fixed_attr) >= need
               for g in (compat_garment, incompat_garment)
               for v in (a_value, b_value))


def set_support(a_value, b_value, compat_garment, incompat_garment,
                axis="pattern", fixed_attr=None):
    """min over the four edges — the bottleneck of the 2x2 rectangle."""
    return min(_edge_support(g, v, axis, fixed_attr)
               for g in (compat_garment, incompat_garment)
               for v in (a_value, b_value))


def rarity_cost(a_value, b_value, compat_garment, incompat_garment,
                axis="pattern", fixed_attr=None, eps=1.0):
    return 1.0 / (set_support(a_value, b_value, compat_garment,
                              incompat_garment, axis, fixed_attr) + eps)


def pick_fixed_color(neutral_colors, p_like, p_dislike, compat_garment,
                     incompat_garment, need=1):
    """c* = argmax_c min over the four (garment, pattern, c) edges.

    Returns (color, bottleneck_support) or (None, 0) when no neutral color
    supports the full rectangle — in that case the tuple itself should be
    rejected, not the color forced.
    """
    best, best_s = None, 0
    for c in neutral_colors:
        s = min(support(g, p, c)
                for g in (compat_garment, incompat_garment)
                for p in (p_like, p_dislike))
        if s > best_s:
            best, best_s = c, s
    if best_s < need:
        return None, best_s
    return best, best_s


if __name__ == "__main__":
    # self-test on a synthetic support matrix (module-level rebind)
    _SUPPORT = {
        "pea_coat":  {"striped": {"n": 12, "colors": {"navy": 7, "beige": 4}},
                         "leopard": {"n": 1, "colors": {"beige": 1}},
                         "solid":   {"n": 140, "colors": {"beige": 60, "navy": 50}}},
        "puffer_jacket": {"striped": {"n": 4, "colors": {"navy": 3}},
                          "leopard": {"n": 0, "colors": {}},
                          "solid":   {"n": 180, "colors": {"black": 90, "navy": 60}}},
    }
    assert support("pea_coat", "striped") == 12
    assert support("puffer_jacket", "striped", "navy") == 3
    # leopard puffer edge missing -> rectangle infeasible
    assert not option_set_is_retrievable("striped", "leopard",
                                         "pea_coat", "puffer_jacket")
    # striped/solid rectangle feasible
    assert option_set_is_retrievable("solid", "striped",
                                     "pea_coat", "puffer_jacket")
    assert set_support("solid", "striped", "pea_coat", "puffer_jacket") == 4
    c, s = pick_fixed_color(["navy", "beige", "black"], "solid", "striped",
                            "pea_coat", "puffer_jacket")
    assert c == "navy" and s == 3, (c, s)
    print("feasibility self-test OK")
