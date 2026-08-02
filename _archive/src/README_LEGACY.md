# `src/` — legacy, NOT the current pipeline

> **Archived.** This directory now lives at `_archive/src/`. Nothing in it runs
> and nothing outside it imports from it. `utils.py` — the one module that was
> still live — was **promoted to `scripts/utils.py`**; import it as
> `from scripts.utils import ...`. The remaining files are retired collectors,
> kept only as provenance.

The v2 benchmark is built by `construction/` (profile → query → option plan) and
evaluated by `scripts/`. Nothing in this directory constructs or evaluates the
shipped dataset.

**Do not read this code as documentation of the benchmark.** It predates the v2
vocabulary revision and still contains retired attribute values — `plaid`,
`camouflage`, `graphic_print`, `argyle` and the pre-split garment labels
(`shirt`, `suit_jacket`, `jacket`, `coat`, `skirt`) — that no longer exist in
`configs/config.py`. Numbers, axis definitions, or vocabularies taken from here
will not match the dataset.

## What was promoted out of here

Two modules outgrew this directory. Both were moved rather than copied, so
there is no second implementation to drift:

| module | promoted to | when |
|---|---|---|
| `label_verifier.py` | `scripts/label_verifier.py` | 2026-07-27 |
| `utils.py` | `scripts/utils.py` | 2026-08-02 |

`label_verifier.py` is the only implementation of Stage 5, so leaving it here
made a live stage look retired; it is now keyed on plan_id like the rest of the
downstream tooling.

`utils.py` (`call_llm`, `call_vlm`, `load_jsonl`, `save_jsonl`, `log_step`,
`parse_json_response`) was imported by `scripts/label_verifier.py`,
`scripts/text_only_eval.py` and all of `text_exp/` — live code on a legacy
path, which is what blocked archiving this directory. It depends only on
`configs.config`, so promoting it was a straight move. All consumers now say
`from scripts.utils import ...`.

## What is left here

Retired collectors only: `option_planner.py`, `image_collector.py`, and the
`collect_images_*.py` family. Their `from src.utils import ...` lines are stale
and were deliberately not rewritten — this code is not meant to run. The
superseded generators (`profile_generator.py`, `query_generator.py`,
`compatibility.py`) were removed on 2026-07-27; use `construction/` instead.
