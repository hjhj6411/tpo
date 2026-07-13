"""
POD-Bench clean user preference profiles — leopard edition.

8 preference archetypes x 3 variants = 24 users.
All values are drawn from configs.config.FASHION_ATTRIBUTE_AXES.

Canonical cleanup:
- pattern removed: camouflage, argyle, plaid
- pattern added: leopard
- garment removed: jacket, suit_jacket, blouse, pants, coat, cotton_pants,
  jogger_pants, polo_shirt, denim_jacket, mini_dress, midi_dress, maxi_dress
- dress is kept as a single canonical category: dress
- skirt is split only into mini_skirt / long_skirt

Leopard policy:
- Use the short canonical value `leopard`, not `leopard_print`.
- It is assigned as a like to bold_expressive/streetwear profiles (plus one
  adventurous_outdoor variant) where CLIP retrieval is likely stable.
- It is assigned as a dislike in conservative/minimal profiles so the active pattern axis has
  enough contrast without treating leopard as universally inappropriate.

Strict 2x2 revision:
- B/D must use true profile dislikes, never neutral fallbacks.
- Every variant now has a liked and disliked value within the common
  dress-code-safe pattern set {solid, striped}, so pattern can remain a strict
  like/dislike active axis even in formal, judicial, mourning, religious, and
  wedding scenarios.
- Color likes/dislikes are redistributed so common dress-code-safe colors
  {black, navy, gray, white, beige} appear on both sides across the user pool,
  avoiding one-way values such as "white only liked" or "blue only liked".
- Color likes/dislikes use three values per side so one user is not locked into
  the same two colors across many garments. At least one safe neutral color is
  still left outside each user's likes/dislikes for non-active-axis fixing.
- Garment likes/dislikes are kept persona-consistent but avoid consuming whole
  scenario garment pools, so the TPO garment axis can stay preference-neutral.
"""

PREFERENCE_ARCHETYPES = [{'archetype_id': 'classic_formal',
  'persona_hint': 'Prefers timeless tailored pieces; avoids sporty or exposed casual items.',
  'variants': [{'garment_likes': ['blazer', 'formal_shirt', 'sweater'],
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
                'pattern_dislikes': ['striped', 'floral']}]},
 {'archetype_id': 'casual_sporty',
  'persona_hint': 'Comfortable athletic-leaning pieces; dislikes rigid formal wear.',
  'variants': [{'garment_likes': ['hoodie', 't_shirt', 'shorts'],
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
                'pattern_dislikes': ['striped', 'leopard']}]},
 {'archetype_id': 'minimalist',
  'persona_hint': 'Clean, understated pieces; avoids loud patterns and bulky silhouettes.',
  'variants': [{'garment_likes': ['formal_shirt', 'sweater', 'slacks'],
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
                'pattern_dislikes': ['solid', 'polka_dot']}]},
 {'archetype_id': 'adventurous_outdoor',
  'persona_hint': 'Rugged outdoor gear; dislikes formal or delicate items.',
  'variants': [{'garment_likes': ['fleece_jacket', 'windbreaker', 'jeans'],
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
                'pattern_dislikes': ['striped', 'floral']}]},
 {'archetype_id': 'elegant',
  'persona_hint': 'Graceful, polished pieces; avoids sporty or overly casual items.',
  'variants': [{'garment_likes': ['dress', 'formal_shirt', 'trench_coat'],
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
                'pattern_dislikes': ['solid', 'leopard']}]},
 {'archetype_id': 'streetwear',
  'persona_hint': 'Urban street style; prefers hoodies, relaxed layers, and bold casual pieces.',
  'variants': [{'garment_likes': ['hoodie', 'windbreaker', 'jeans'],
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
                'pattern_dislikes': ['solid', 'polka_dot']}]},
 {'archetype_id': 'relaxed_neutral',
  'persona_hint': 'Easy-going, middle-of-the-road taste; avoids extremes in either direction.',
  'variants': [{'garment_likes': ['sweater', 'formal_shirt', 'jeans'],
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
                'pattern_dislikes': ['striped', 'polka_dot']}]},
 {'archetype_id': 'bold_expressive',
  'persona_hint': 'Loves vivid colors and visible patterns; avoids plain, muted looks.',
  'variants': [{'garment_likes': ['dress', 't_shirt', 'mini_skirt'],
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
                'pattern_dislikes': ['solid', 'polka_dot']}]}]



def get_all_variants():
    """Flatten all (archetype_id, variant_idx, variant_dict) triples."""
    out = []
    for arch in PREFERENCE_ARCHETYPES:
        for i, var in enumerate(arch["variants"]):
            out.append((arch["archetype_id"], i, {
                **var,
                "persona_hint": arch["persona_hint"],
            }))
    return out


if __name__ == "__main__":
    from collections import Counter
    variants = get_all_variants()
    print(f"Users: {len(variants)}")
    pc, pd, cc, gc = Counter(), Counter(), Counter(), Counter()
    overlap_errors = []
    for aid, vi, v in variants:
        for ax, like_k, dis_k in [("pattern", "pattern_likes", "pattern_dislikes"),
                                  ("color", "color_likes", "color_dislikes"),
                                  ("garment_category", "garment_likes", "garment_dislikes")]:
            ov = set(v[like_k]) & set(v[dis_k])
            if ov:
                overlap_errors.append((aid, vi, ax, sorted(ov)))
        pc.update(v["pattern_likes"])
        pd.update(v["pattern_dislikes"])
        cc.update(v["color_likes"])
        gc.update(v["garment_likes"])
    print("pattern_likes tally:", dict(pc.most_common()))
    print("pattern_dislikes tally:", dict(pd.most_common()))
    print("color_likes tally:", dict(cc.most_common()))
    print("garment_likes tally:", dict(gc.most_common()))
    print("like/dislike overlaps (should be empty):", overlap_errors)
