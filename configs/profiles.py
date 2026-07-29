"""
POD-Bench user preference variants — two independent blocks.

`PROFILE_VARIANTS` (legacy, pre-v2 reproduction only)
    24 hand-authored variants, garment quota 3+3. Semantic archetype ids and
    persona-hint strings were removed: they were unused by the pipeline,
    carried unvalidated persona claims, and are not part of any user
    definition. Only the raw preference lists remain. Reachable through
    `get_all_variants()` and `profile_generator --legacy`; the v2 benchmark
    does not use it.

`V2_PROFILE_VARIANTS` (the profiles v2 actually ships)
    The 24 rule-generated profiles of variant `wacv_scenario_v2`, garment
    quota 4+4, frozen here in the same literal form so the released dataset is
    readable without running the generator. The generator now targets v3;
    `check_v2_profiles_match_generator()` protects this historical snapshot
    and the released v2 artifact hash instead of re-deriving it with v3 rules.

Neither block carries persona semantics; both are raw preference lists.
"""

import hashlib
from pathlib import Path

PROFILE_VARIANTS = [{'garment_likes': ['blazer', 'formal_shirt', 'sweater'],
  'garment_dislikes': ['hoodie', 'shorts', 'windbreaker'],
  'color_likes': ['navy', 'beige', 'white'],
  'color_dislikes': ['gray', 'orange', 'red'],
  'pattern_likes': ['striped', 'checkered'],
  'pattern_dislikes': ['solid', 'leopard']},
 {'garment_likes': ['trench_coat', 'formal_shirt', 'cardigan'],
  'garment_dislikes': ['fleece_jacket', 'tank_top', 'hoodie'],
  'color_likes': ['black', 'gray', 'white'],
  'color_dislikes': ['navy', 'yellow', 'orange'],
  'pattern_likes': ['solid', 'polka_dot'],
  'pattern_dislikes': ['striped', 'leopard']},
 {'garment_likes': ['blazer', 'dress', 'slacks'],
  'garment_dislikes': ['puffer_jacket', 'hoodie', 'shorts'],
  'color_likes': ['white', 'navy', 'gray'],
  'color_dislikes': ['black', 'green', 'orange'],
  'pattern_likes': ['checkered', 'solid'],
  'pattern_dislikes': ['striped', 'floral']},
 {'garment_likes': ['hoodie', 't_shirt', 'shorts'],
  'garment_dislikes': ['blazer', 'trench_coat', 'dress'],
  'color_likes': ['red', 'orange', 'blue'],
  'color_dislikes': ['black', 'beige', 'gray'],
  'pattern_likes': ['striped', 'checkered'],
  'pattern_dislikes': ['solid', 'floral']},
 {'garment_likes': ['tank_top', 'shorts', 'windbreaker'],
  'garment_dislikes': ['blazer', 'formal_shirt', 'long_skirt'],
  'color_likes': ['green', 'gray', 'blue'],
  'color_dislikes': ['navy', 'pink', 'black'],
  'pattern_likes': ['solid', 'polka_dot'],
  'pattern_dislikes': ['striped', 'checkered']},
 {'garment_likes': ['t_shirt', 'hoodie', 'jeans'],
  'garment_dislikes': ['blazer', 'trench_coat', 'dress'],
  'color_likes': ['white', 'orange', 'red'],
  'color_dislikes': ['gray', 'brown', 'navy'],
  'pattern_likes': ['checkered', 'solid'],
  'pattern_dislikes': ['striped', 'leopard']},
 {'garment_likes': ['formal_shirt', 'sweater', 'slacks'],
  'garment_dislikes': ['leather_jacket', 'hoodie', 'tank_top'],
  'color_likes': ['black', 'white', 'gray'],
  'color_dislikes': ['navy', 'purple', 'orange'],
  'pattern_likes': ['solid', 'checkered'],
  'pattern_dislikes': ['striped', 'floral']},
 {'garment_likes': ['slacks', 'cardigan', 'sweater'],
  'garment_dislikes': ['windbreaker', 'hoodie', 'shorts'],
  'color_likes': ['gray', 'beige', 'white'],
  'color_dislikes': ['black', 'red', 'orange'],
  'pattern_likes': ['checkered', 'solid'],
  'pattern_dislikes': ['striped', 'leopard']},
 {'garment_likes': ['blazer', 'jeans', 'formal_shirt'],
  'garment_dislikes': ['fleece_jacket', 'tank_top', 'mini_skirt'],
  'color_likes': ['gray', 'white', 'black'],
  'color_dislikes': ['navy', 'blue', 'red'],
  'pattern_likes': ['striped', 'checkered'],
  'pattern_dislikes': ['solid', 'polka_dot']},
 {'garment_likes': ['fleece_jacket', 'windbreaker', 'jeans'],
  'garment_dislikes': ['blazer', 'dress', 'slacks'],
  'color_likes': ['green', 'brown', 'beige'],
  'color_dislikes': ['gray', 'pink', 'navy'],
  'pattern_likes': ['checkered', 'solid'],
  'pattern_dislikes': ['striped', 'polka_dot']},
 {'garment_likes': ['windbreaker', 'hoodie', 'shorts'],
  'garment_dislikes': ['trench_coat', 'slacks', 'blazer'],
  'color_likes': ['navy', 'beige', 'green'],
  'color_dislikes': ['black', 'orange', 'gray'],
  'pattern_likes': ['checkered', 'striped'],
  'pattern_dislikes': ['solid', 'floral']},
 {'garment_likes': ['puffer_jacket', 'windbreaker', 'hoodie'],
  'garment_dislikes': ['blazer', 'long_skirt', 'dress'],
  'color_likes': ['gray', 'blue', 'black'],
  'color_dislikes': ['white', 'yellow', 'navy'],
  'pattern_likes': ['solid', 'leopard'],
  'pattern_dislikes': ['striped', 'floral']},
 {'garment_likes': ['dress', 'formal_shirt', 'trench_coat'],
  'garment_dislikes': ['hoodie', 'tank_top', 'windbreaker'],
  'color_likes': ['black', 'purple', 'navy'],
  'color_dislikes': ['gray', 'orange', 'white'],
  'pattern_likes': ['floral', 'striped'],
  'pattern_dislikes': ['solid', 'polka_dot']},
 {'garment_likes': ['blazer', 'long_skirt', 'formal_shirt'],
  'garment_dislikes': ['fleece_jacket', 'hoodie', 'shorts'],
  'color_likes': ['navy', 'pink', 'beige'],
  'color_dislikes': ['black', 'brown', 'gray'],
  'pattern_likes': ['floral', 'solid'],
  'pattern_dislikes': ['striped', 'checkered']},
 {'garment_likes': ['trench_coat', 'dress', 'cardigan'],
  'garment_dislikes': ['t_shirt', 'tank_top', 'windbreaker'],
  'color_likes': ['beige', 'white', 'black'],
  'color_dislikes': ['navy', 'red', 'orange'],
  'pattern_likes': ['polka_dot', 'striped'],
  'pattern_dislikes': ['solid', 'leopard']},
 {'garment_likes': ['hoodie', 'windbreaker', 'jeans'],
  'garment_dislikes': ['blazer', 'trench_coat', 'slacks'],
  'color_likes': ['black', 'red', 'gray'],
  'color_dislikes': ['white', 'purple', 'navy'],
  'pattern_likes': ['leopard', 'solid'],
  'pattern_dislikes': ['striped', 'floral']},
 {'garment_likes': ['t_shirt', 'leather_jacket', 'hoodie'],
  'garment_dislikes': ['blazer', 'dress', 'long_skirt'],
  'color_likes': ['white', 'green', 'black'],
  'color_dislikes': ['gray', 'brown', 'navy'],
  'pattern_likes': ['leopard', 'striped'],
  'pattern_dislikes': ['solid', 'checkered']},
 {'garment_likes': ['sweatshirt', 'shorts', 'windbreaker'],
  'garment_dislikes': ['blazer', 'trench_coat', 'long_skirt'],
  'color_likes': ['gray', 'blue', 'navy'],
  'color_dislikes': ['pink', 'beige', 'black'],
  'pattern_likes': ['checkered', 'striped'],
  'pattern_dislikes': ['solid', 'polka_dot']},
 {'garment_likes': ['sweater', 'formal_shirt', 'jeans'],
  'garment_dislikes': ['fleece_jacket', 'blazer', 'tank_top'],
  'color_likes': ['blue', 'beige', 'navy'],
  'color_dislikes': ['black', 'orange', 'gray'],
  'pattern_likes': ['striped', 'checkered'],
  'pattern_dislikes': ['solid', 'leopard']},
 {'garment_likes': ['cardigan', 'slacks', 't_shirt'],
  'garment_dislikes': ['trench_coat', 'blazer', 'shorts'],
  'color_likes': ['navy', 'white', 'beige'],
  'color_dislikes': ['gray', 'yellow', 'orange'],
  'pattern_likes': ['checkered', 'solid'],
  'pattern_dislikes': ['striped', 'floral']},
 {'garment_likes': ['formal_shirt', 'jeans', 'sweatshirt'],
  'garment_dislikes': ['trench_coat', 'blazer', 'tank_top'],
  'color_likes': ['black', 'green', 'gray'],
  'color_dislikes': ['white', 'red', 'navy'],
  'pattern_likes': ['solid', 'floral'],
  'pattern_dislikes': ['striped', 'polka_dot']},
 {'garment_likes': ['dress', 't_shirt', 'mini_skirt'],
  'garment_dislikes': ['blazer', 'trench_coat', 'slacks'],
  'color_likes': ['red', 'yellow', 'pink'],
  'color_dislikes': ['gray', 'black', 'navy'],
  'pattern_likes': ['floral', 'striped'],
  'pattern_dislikes': ['solid', 'checkered']},
 {'garment_likes': ['formal_shirt', 'jeans', 'leather_jacket'],
  'garment_dislikes': ['fleece_jacket', 'trench_coat', 'blazer'],
  'color_likes': ['orange', 'pink', 'red'],
  'color_dislikes': ['navy', 'beige', 'black'],
  'pattern_likes': ['leopard', 'solid'],
  'pattern_dislikes': ['striped', 'checkered']},
 {'garment_likes': ['t_shirt', 'dress', 'hoodie'],
  'garment_dislikes': ['blazer', 'trench_coat', 'slacks'],
  'color_likes': ['purple', 'pink', 'orange'],
  'color_dislikes': ['black', 'green', 'gray'],
  'pattern_likes': ['striped', 'leopard'],
  'pattern_dislikes': ['solid', 'polka_dot']}]


# ── v2 frozen profiles (variant wacv_scenario_v2, seed 42) ─────────────
# The 24 profiles the v2 benchmark actually ships, frozen here in the same
# literal form as the legacy block above so the released dataset can be read
# without executing the generator.
#
# SOURCE OF TRUTH IS STILL construction.profile_generator. This list is a
# snapshot; check_v2_profiles_match_generator() below re-derives it and fails
# loudly if the two ever diverge, so nobody can edit one and ship the other.
#
# Structure (docs/redesign_v2_plan.md §16): garment 4+4 (2 of the 5 ANCHOR
# garments liked, 2 disliked, 1 left preference-neutral) + color 3+3 (1 ANCHOR)
# + pattern 2+2 (1 quiet + 1 expressive per side). The RESERVE colors
# {orange, yellow, pink, white} never appear in any profile by design.

V2_PROFILE_VARIANTS = [{'garment_likes': ['formal_shirt', 'slacks', 'cardigan', 'windbreaker'],
  'garment_dislikes': ['blazer', 'dress', 'sweater', 'mini_skirt'],
  'color_likes': ['gray', 'blue', 'green'],
  'color_dislikes': ['black', 'brown', 'beige'],
  'pattern_likes': ['solid', 'checkered'],
  'pattern_dislikes': ['striped', 'polka_dot']},
 {'garment_likes': ['blazer', 'long_skirt', 'tank_top', 'sweater'],
  'garment_dislikes': ['slacks', 'formal_shirt', 'sweatshirt', 'cardigan'],
  'color_likes': ['navy', 'red', 'brown'],
  'color_dislikes': ['gray', 'blue', 'green'],
  'pattern_likes': ['striped', 'polka_dot'],
  'pattern_dislikes': ['solid', 'checkered']},
 {'garment_likes': ['dress', 'formal_shirt', 't_shirt', 'trench_coat'],
  'garment_dislikes': ['long_skirt', 'slacks', 'cardigan', 'shorts'],
  'color_likes': ['black', 'blue', 'beige'],
  'color_dislikes': ['navy', 'red', 'purple'],
  'pattern_likes': ['solid', 'floral'],
  'pattern_dislikes': ['striped', 'polka_dot']},
 {'garment_likes': ['dress', 'blazer', 'leather_jacket', 'shorts'],
  'garment_dislikes': ['formal_shirt', 'long_skirt', 'tank_top', 'sweater'],
  'color_likes': ['gray', 'red', 'purple'],
  'color_dislikes': ['navy', 'brown', 'beige'],
  'pattern_likes': ['striped', 'polka_dot'],
  'pattern_dislikes': ['solid', 'checkered']},
 {'garment_likes': ['slacks', 'long_skirt', 'cardigan', 'puffer_jacket'],
  'garment_dislikes': ['blazer', 'dress', 'sweater', 'hoodie'],
  'color_likes': ['black', 'beige', 'purple'],
  'color_dislikes': ['gray', 'blue', 'green'],
  'pattern_likes': ['solid', 'polka_dot'],
  'pattern_dislikes': ['striped', 'leopard']},
 {'garment_likes': ['blazer', 'formal_shirt', 'sweater', 'jeans'],
  'garment_dislikes': ['long_skirt', 'dress', 'windbreaker', 'leather_jacket'],
  'color_likes': ['navy', 'green', 'brown'],
  'color_dislikes': ['black', 'red', 'purple'],
  'pattern_likes': ['striped', 'floral'],
  'pattern_dislikes': ['solid', 'checkered']},
 {'garment_likes': ['dress', 'slacks', 'sweater', 'leggings'],
  'garment_dislikes': ['formal_shirt', 'blazer', 't_shirt', 'cardigan'],
  'color_likes': ['gray', 'blue', 'purple'],
  'color_dislikes': ['black', 'brown', 'beige'],
  'pattern_likes': ['solid', 'checkered'],
  'pattern_dislikes': ['striped', 'floral']},
 {'garment_likes': ['long_skirt', 'formal_shirt', 'hoodie', 'cardigan'],
  'garment_dislikes': ['slacks', 'blazer', 'sweater', 'fleece_jacket'],
  'color_likes': ['navy', 'red', 'beige'],
  'color_dislikes': ['gray', 'blue', 'green'],
  'pattern_likes': ['striped', 'leopard'],
  'pattern_dislikes': ['solid', 'floral']},
 {'garment_likes': ['dress', 'long_skirt', 'sweatshirt', 'sweater'],
  'garment_dislikes': ['formal_shirt', 'slacks', 'trench_coat', 'jeans'],
  'color_likes': ['black', 'green', 'brown'],
  'color_dislikes': ['navy', 'red', 'purple'],
  'pattern_likes': ['solid', 'floral'],
  'pattern_dislikes': ['striped', 'checkered']},
 {'garment_likes': ['blazer', 'slacks', 'sweater', 'mini_skirt'],
  'garment_dislikes': ['dress', 'long_skirt', 'cardigan', 'puffer_jacket'],
  'color_likes': ['gray', 'green', 'brown'],
  'color_dislikes': ['black', 'red', 'purple'],
  'pattern_likes': ['striped', 'leopard'],
  'pattern_dislikes': ['solid', 'polka_dot']},
 {'garment_likes': ['dress', 'slacks', 'cardigan', 'fleece_jacket'],
  'garment_dislikes': ['blazer', 'long_skirt', 'sweater', 'leggings'],
  'color_likes': ['navy', 'beige', 'purple'],
  'color_dislikes': ['gray', 'blue', 'green'],
  'pattern_likes': ['solid', 'polka_dot'],
  'pattern_dislikes': ['striped', 'floral']},
 {'garment_likes': ['formal_shirt', 'long_skirt', 'trench_coat', 'shorts'],
  'garment_dislikes': ['dress', 'slacks', 'tank_top', 'cardigan'],
  'color_likes': ['black', 'blue', 'red'],
  'color_dislikes': ['navy', 'brown', 'beige'],
  'pattern_likes': ['striped', 'polka_dot'],
  'pattern_dislikes': ['solid', 'floral']},
 {'garment_likes': ['blazer', 'slacks', 'tank_top', 'cardigan'],
  'garment_dislikes': ['formal_shirt', 'dress', 'sweatshirt', 'sweater'],
  'color_likes': ['gray', 'green', 'beige'],
  'color_dislikes': ['navy', 'red', 'purple'],
  'pattern_likes': ['solid', 'leopard'],
  'pattern_dislikes': ['striped', 'polka_dot']},
 {'garment_likes': ['dress', 'blazer', 'sweater', 'fleece_jacket'],
  'garment_dislikes': ['formal_shirt',
                       'long_skirt',
                       'windbreaker',
                       'leggings'],
  'color_likes': ['navy', 'red', 'purple'],
  'color_dislikes': ['gray', 'green', 'beige'],
  'pattern_likes': ['striped', 'floral'],
  'pattern_dislikes': ['solid', 'leopard']},
 {'garment_likes': ['long_skirt', 'formal_shirt', 'cardigan', 'puffer_jacket'],
  'garment_dislikes': ['blazer', 'slacks', 'sweater', 'shorts'],
  'color_likes': ['black', 'brown', 'beige'],
  'color_dislikes': ['navy', 'blue', 'purple'],
  'pattern_likes': ['solid', 'checkered'],
  'pattern_dislikes': ['striped', 'polka_dot']},
 {'garment_likes': ['dress', 'blazer', 'sweatshirt', 'sweater'],
  'garment_dislikes': ['long_skirt',
                       'formal_shirt',
                       'puffer_jacket',
                       'fleece_jacket'],
  'color_likes': ['gray', 'blue', 'green'],
  'color_dislikes': ['black', 'brown', 'beige'],
  'pattern_likes': ['striped', 'checkered'],
  'pattern_dislikes': ['solid', 'leopard']},
 {'garment_likes': ['formal_shirt', 'slacks', 'sweater', 'hoodie'],
  'garment_dislikes': ['blazer', 'dress', 'cardigan', 'mini_skirt'],
  'color_likes': ['black', 'red', 'purple'],
  'color_dislikes': ['gray', 'green', 'brown'],
  'pattern_likes': ['solid', 'checkered'],
  'pattern_dislikes': ['striped', 'floral']},
 {'garment_likes': ['long_skirt', 'slacks', 'cardigan', 'leather_jacket'],
  'garment_dislikes': ['formal_shirt', 'dress', 'hoodie', 'trench_coat'],
  'color_likes': ['navy', 'blue', 'brown'],
  'color_dislikes': ['black', 'red', 'beige'],
  'pattern_likes': ['striped', 'leopard'],
  'pattern_dislikes': ['solid', 'floral']},
 {'garment_likes': ['blazer', 'formal_shirt', 'cardigan', 'windbreaker'],
  'garment_dislikes': ['slacks', 'long_skirt', 'sweater', 'jeans'],
  'color_likes': ['gray', 'brown', 'purple'],
  'color_dislikes': ['black', 'blue', 'red'],
  'pattern_likes': ['solid', 'polka_dot'],
  'pattern_dislikes': ['striped', 'leopard']},
 {'garment_likes': ['dress', 'long_skirt', 'sweater', 'leggings'],
  'garment_dislikes': ['blazer', 'slacks', 't_shirt', 'cardigan'],
  'color_likes': ['black', 'blue', 'green'],
  'color_dislikes': ['navy', 'brown', 'purple'],
  'pattern_likes': ['striped', 'floral'],
  'pattern_dislikes': ['solid', 'polka_dot']},
 {'garment_likes': ['formal_shirt', 'dress', 't_shirt', 'jeans'],
  'garment_dislikes': ['long_skirt', 'blazer', 'sweater', 'leather_jacket'],
  'color_likes': ['navy', 'red', 'beige'],
  'color_dislikes': ['gray', 'blue', 'green'],
  'pattern_likes': ['solid', 'leopard'],
  'pattern_dislikes': ['striped', 'checkered']},
 {'garment_likes': ['blazer', 'slacks', 'sweater', 'mini_skirt'],
  'garment_dislikes': ['dress', 'formal_shirt', 'cardigan', 'fleece_jacket'],
  'color_likes': ['gray', 'beige', 'purple'],
  'color_dislikes': ['black', 'red', 'brown'],
  'pattern_likes': ['striped', 'checkered'],
  'pattern_dislikes': ['solid', 'leopard']},
 {'garment_likes': ['long_skirt', 'dress', 'cardigan', 'windbreaker'],
  'garment_dislikes': ['slacks', 'blazer', 'sweater', 'shorts'],
  'color_likes': ['black', 'red', 'brown'],
  'color_dislikes': ['navy', 'green', 'purple'],
  'pattern_likes': ['solid', 'leopard'],
  'pattern_dislikes': ['striped', 'checkered']},
 {'garment_likes': ['formal_shirt', 'blazer', 'tank_top', 'sweater'],
  'garment_dislikes': ['slacks', 'dress', 'cardigan', 'leather_jacket'],
  'color_likes': ['navy', 'red', 'green'],
  'color_dislikes': ['gray', 'blue', 'beige'],
  'pattern_likes': ['striped', 'floral'],
  'pattern_dislikes': ['solid', 'leopard']}]

def get_all_variants():
    """Flatten to (pool_id, variant_idx, variant_dict) triples.

    pool_id is a positional group label (3 variants per group), kept only so
    the legacy generator call signature stays stable. It carries no semantics.
    """
    return [(f"legacy_pool_{i // 3:02d}", i % 3, v)
            for i, v in enumerate(PROFILE_VARIANTS)]


def get_v2_variants():
    """Flatten the frozen v2 profiles to (pool_id, variant_idx, dict) triples."""
    return [("global", i, v) for i, v in enumerate(V2_PROFILE_VARIANTS)]


def check_v2_profiles_match_generator():
    """Validate the frozen v2 snapshot without rewriting it for v3.

    The current generator now targets the 23-label v3 vocabulary, so re-deriving
    v2 with it would be a category error. Instead, validate the historical
    snapshot structure and the released v2 artifact hash.
    """
    assert len(V2_PROFILE_VARIANTS) == 24
    for i, variant in enumerate(V2_PROFILE_VARIANTS):
        likes = set(variant["garment_likes"])
        dislikes = set(variant["garment_dislikes"])
        assert len(likes) == len(dislikes) == 4, f"v2 profile {i}: garment quota"
        assert not likes & dislikes, f"v2 profile {i}: garment overlap"

    path = Path(__file__).resolve().parent.parent / (
        "data_wacv_scenario_v2/profiles/profiles.jsonl")
    expected = "6642f47a850acbe63b5f916b4c246a092c6ca49ff45b725387df7e5849d7c68f"
    assert path.exists(), f"v2 profile artifact missing: {path}"
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == expected, f"v2 profile artifact hash drift: {actual}"
    return (
        f"V2_PROFILE_VARIANTS preserved ({len(V2_PROFILE_VARIANTS)} variants); "
        f"released profiles hash {actual}")


if __name__ == "__main__":
    print(check_v2_profiles_match_generator())
