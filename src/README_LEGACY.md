# `src/` — legacy, NOT the current pipeline

The v2 benchmark is built by `construction/` (profile → query → option plan) and
evaluated by `scripts/`. Nothing in this directory constructs or evaluates the
shipped dataset.

**Do not read this code as documentation of the benchmark.** It predates the v2
vocabulary revision and still contains retired attribute values — `plaid`,
`camouflage`, `graphic_print`, `argyle` and the pre-split garment labels
(`shirt`, `suit_jacket`, `jacket`, `coat`, `skirt`) — that no longer exist in
`configs/config.py`. Numbers, axis definitions, or vocabularies taken from here
will not match the dataset.

## What is still live

Three modules are still imported by current code and must stay:

| module | used by |
|---|---|
| `utils.py` | `scripts/text_only_eval.py`, `scripts/label_verifier.py`, `text_exp/*`, `vit/collect_images_clip_retrieval.py` |
| `option_planner.py` | legacy collector path |
| `image_collector.py` | legacy collector path |

`label_verifier.py` was **promoted out of this directory** to
`scripts/label_verifier.py` on 2026-07-27. It is the only implementation of
Stage 5, so leaving it here made a live stage look retired; it is now keyed on
plan_id like the rest of the downstream tooling.

The superseded generators (`profile_generator.py`, `query_generator.py`,
`compatibility.py`) were removed on 2026-07-27 — use `construction/` instead.
