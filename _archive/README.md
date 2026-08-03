# `_archive/` — retired pipelines

Nothing in this directory runs, and nothing outside it imports from it. It is
kept for provenance: these paths are cited by `docs/` and by the revision
history, so deleting them would break the record of how the benchmark got to
its current shape.

The live pipeline is `configs/` → `construction/` → `fsiglip/` → `scripts/`.

| directory | what it was | why it is here |
|---|---|---|
| `src/` | the pre-`construction/` pipeline: legacy collectors, `option_planner.py`, `image_collector.py` | superseded by `construction/` + `fsiglip/`. Its one live module, `utils.py`, was promoted to `scripts/utils.py` before the move — see `_archive/src/README_LEGACY.md`. **Do not read it as documentation of the benchmark**; it predates the v2 vocabulary revision |
| `before_configs/` | snapshot of `configs/` (profiles, scenarios) before the v2 vocabulary revision | physical evidence for the revision history quoted in `docs/redesign_v2_plan.md` |
| `vit/` | ViT-L/14 collection path | superseded by `fsiglip/collector_sam3.py`. Its garment vocabulary still carries the retired `wool coat` hypernym and the `GARMENT_EQUIV_GROUPS` alias table — **do not copy vocabulary out of here** (see `docs/FIX_GARMENT_VOCAB.md`) |
| `QwenEmb/` | Qwen3-VL-Embedding retrieval backend trial | not adopted; Stage 4 is frozen on Marqo-FashionSigLIP. Setup record in `docs/SETUP_QWENEMB.md` |
| `fsiglip2/` | zooclaw-FashionSigLIP2 backend trial | not adopted. Its corpus *was* fully built (132 shards) and still sits at `data/fashionsiglip2_corpus/`; the embedding space is incompatible with v1's, so the two corpora must never be mixed |

Paths inside these files (`vit/...`, `QwenEmb/...`) are written relative to the
old repository root. Prefix them with `_archive/` to locate the file today.
