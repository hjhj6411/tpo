"""
common/permute.py — answer-position designs: latin-square (cyclic) + gold-fixing.

Objective : decouple option CONTENT from display POSITION.
  * cyclic: per item, a seeded base shuffle then 4 cyclic rotations -> every
    canonical option visits every display position exactly once (latin square).
    ONE protocol yields three analyses: shuffled preference (list-A1), accuracy
    by gold position (A2), position-bias existence per model (A4).
  * fixed_gold: gold pinned to one display position, distractor relative order
    held constant across the 4 positions (only gold moves -> clean A2 reading).
Input     : global seed + query_id.
Output    : `order` = list of canonical indices by display position, e.g.
            order=[2,0,3,1] means display A shows canonical C, B shows A, ...
Metrics   : verify_latin() asserts the latin property (used in tests).
Run       : imported. Quick check:
              python -c "from common.permute import *; print(cyclic_orders(base_perm(item_rng(17,'q1'))))"
"""
import hashlib
import random

N_OPT = 4


def item_rng(global_seed, query_id):
    h = hashlib.sha256(f"{global_seed}:{query_id}".encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def base_perm(rng, n=N_OPT):
    p = list(range(n))
    rng.shuffle(p)
    return p


def rotate(seq, r):
    r %= len(seq)
    return list(seq[r:]) + list(seq[:r])


def cyclic_orders(base, n_runs=N_OPT):
    """n_runs rotations of the seeded base perm. For n_runs == len(base) this
    is a latin square: every canonical index hits every display position once."""
    return [rotate(base, r) for r in range(n_runs)]


def fixed_gold_order(base, gold_idx, display_pos):
    """Gold at display_pos; distractors keep their base relative order, so
    across the 4 sub-runs ONLY the gold position moves."""
    distract = [i for i in base if i != gold_idx]
    order = distract[:display_pos] + [gold_idx] + distract[display_pos:]
    return order


def verify_latin(orders):
    n = len(orders[0])
    for pos in range(n):
        seen = {o[pos] for o in orders}
        if len(seen) != len(orders):
            return False
    for o in orders:
        if sorted(o) != list(range(n)):
            return False
    return True
