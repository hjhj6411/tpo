#!/usr/bin/env bash
# verify_release.sh — reproduce the shipped dataset and prove it is unchanged.
#
#   scripts/verify_release.sh [VARIANT]      (default: wacv_scenario_v5)
#
# Pins the GENERATION INPUTS, regenerates profiles/queries/option_plans into a
# scratch variant, compares the three SHA256s against the released files, runs
# the construction validator, and runs the validator's own mutation test. Any
# mismatch exits non-zero.
#
# The regeneration writes to data_<VARIANT>_verify/ — the released data is only
# ever read, never overwritten.
#
# Reproducibility is a property of HEAD, not of every past release. v5 changed
# the pattern-preference rule in profile_generator, so v3/v4 can no longer be
# rebuilt bit-identically from this tree; their released directories stay as
# historical artifacts and their reports quote the hashes they shipped with.
# Check out the tag/commit that produced a variant to reverify it.
set -euo pipefail

VARIANT="${1:-wacv_scenario_v5}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RELEASED="data_${VARIANT}"
VERIFY_VARIANT="${VARIANT}_verify"
VERIFY_DIR="data_${VERIFY_VARIANT}"

# ── per-variant Stage 3 flags ────────────────────────────────────────────
# v5 takes the human-audited image cell library as a generation INPUT: the
# option planner only places options on cells a human marked "available", and
# holds the pattern axis at its baseline when pattern is the nuisance axis.
# Older variants predate both flags and must be rebuilt without them.
#
# v5b adds two more generation inputs: the MATERIALIZED-cell manifest (an option
# may only land on a cell that exists as a jpg, not merely on one a human
# approved) and the dialogue-cell reservation. Its case must come first — the
# v5 pattern would otherwise swallow it and rebuild v5b with v5's flags.
PLANNER_FLAGS=()
CELL_LIBRARY=""
CELL_LIBRARY_SHA=""
IMAGE_MANIFEST=""
IMAGE_MANIFEST_SHA=""
case "$VARIANT" in
  wacv_scenario_v5b*)
    CELL_LIBRARY="annotation/attribute_library.json"
    CELL_LIBRARY_SHA="ab9f99aa1c29b063a9c067f53809e06af4a97a05931139ac26a06cd2dc4f7293"
    IMAGE_MANIFEST="data_wacv_scenario_v5b/images_manifest.json"
    IMAGE_MANIFEST_SHA="21d5e05fd49a60aec490ec6747b98e6c582b5ef7dfd915f27653a6e2c55bf1ea"
    PLANNER_FLAGS=(--cell-library "$CELL_LIBRARY" --solid-baseline
                   --reserve-dialog-cells 3 --image-manifest "$IMAGE_MANIFEST")
    ;;
  wacv_scenario_v5*)
    CELL_LIBRARY="annotation/attribute_library.json"
    CELL_LIBRARY_SHA="72cc8665d6f92d143f850b751bba1767c342a2e9b857292e6738666bea86baae"
    PLANNER_FLAGS=(--cell-library "$CELL_LIBRARY" --solid-baseline)
    ;;
esac

if [[ ! -d "$RELEASED" ]]; then
  echo "  [error] released data dir not found: $RELEASED" >&2
  exit 1
fi

echo "══ 1/5  pinning generation inputs ══"
if [[ -n "$CELL_LIBRARY" ]]; then
  have="$(sha256sum "$CELL_LIBRARY" | cut -d' ' -f1)"
  if [[ "$have" == "$CELL_LIBRARY_SHA" ]]; then
    echo "  OK    ${CELL_LIBRARY}  ${have:0:12}…"
  else
    echo "  DIFF  ${CELL_LIBRARY}" >&2
    echo "        pinned:  $CELL_LIBRARY_SHA" >&2
    echo "        on disk: $have" >&2
    echo "  [FAIL] the cell library is a generation input; a changed library" >&2
    echo "         changes the released plans." >&2
    exit 1
  fi
else
  echo "  (none — ${VARIANT} predates the cell-library input)"
fi
if [[ -n "$IMAGE_MANIFEST" ]]; then
  have="$(sha256sum "$IMAGE_MANIFEST" | cut -d' ' -f1)"
  if [[ "$have" == "$IMAGE_MANIFEST_SHA" ]]; then
    echo "  OK    ${IMAGE_MANIFEST}  ${have:0:12}…"
  else
    echo "  DIFF  ${IMAGE_MANIFEST}" >&2
    echo "        pinned:  $IMAGE_MANIFEST_SHA" >&2
    echo "        on disk: $have" >&2
    echo "  [FAIL] the image manifest is a generation input; materializing a" >&2
    echo "         different cell set changes the released plans." >&2
    exit 1
  fi
fi

echo
echo "══ 2/5  regenerating into ${VERIFY_DIR} (seed 42, deterministic) ══"
rm -rf "$VERIFY_DIR"
export POD_VARIANT="$VERIFY_VARIANT"
python -m construction.profile_generator --n_users 24 --force
python -m construction.query_generator   --force
python -m construction.option_planner    --force ${PLANNER_FLAGS[@]+"${PLANNER_FLAGS[@]}"}

echo
echo "══ 3/5  comparing SHA256 against ${RELEASED} ══"
status=0
for f in profiles/profiles.jsonl queries/queries.jsonl options/option_plans.jsonl; do
  a="$(sha256sum "${RELEASED}/${f}"   | cut -d' ' -f1)"
  b="$(sha256sum "${VERIFY_DIR}/${f}" | cut -d' ' -f1)"
  if [[ "$a" == "$b" ]]; then
    echo "  OK    ${f}  ${a:0:12}…"
  else
    echo "  DIFF  ${f}"
    echo "        released: $a"
    echo "        rebuilt:  $b"
    status=1
  fi
done
if [[ $status -ne 0 ]]; then
  echo "  [FAIL] regeneration is not bit-identical — the release is not reproducible." >&2
  exit 1
fi

echo
echo "══ 4/5  construction validator on the released data ══"
POD_VARIANT="$VARIANT" python -m scripts.validate_options

echo
echo "══ 5/5  validator mutation test (must print 48/48) ══"
POD_VARIANT="$VARIANT" python tests/test_option_validator_mutations.py

echo
echo "══ ALL CHECKS PASSED — ${VARIANT} is reproducible and validated ══"
echo "   (scratch copy left at ${VERIFY_DIR}; remove it when done)"
