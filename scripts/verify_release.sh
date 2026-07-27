#!/usr/bin/env bash
# verify_release.sh — reproduce the shipped dataset and prove it is unchanged.
#
#   scripts/verify_release.sh [VARIANT]      (default: wacv_scenario_v2)
#
# Regenerates profiles/queries/option_plans into a scratch variant, compares the
# three SHA256s against the released files, runs the construction validator, and
# runs the validator's own mutation test. Any mismatch exits non-zero.
#
# The regeneration writes to data_<VARIANT>_verify/ — the released data is only
# ever read, never overwritten.
set -euo pipefail

VARIANT="${1:-wacv_scenario_v2}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RELEASED="data_${VARIANT}"
VERIFY_VARIANT="${VARIANT}_verify"
VERIFY_DIR="data_${VERIFY_VARIANT}"

if [[ ! -d "$RELEASED" ]]; then
  echo "  [error] released data dir not found: $RELEASED" >&2
  exit 1
fi

echo "══ 1/4  regenerating into ${VERIFY_DIR} (seed 42, deterministic) ══"
rm -rf "$VERIFY_DIR"
export POD_VARIANT="$VERIFY_VARIANT"
python -m construction.profile_generator --n_users 24 --force
python -m construction.query_generator   --force
python -m construction.option_planner    --force

echo
echo "══ 2/4  comparing SHA256 against ${RELEASED} ══"
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
echo "══ 3/4  construction validator on the released data ══"
POD_VARIANT="$VARIANT" python -m scripts.validate_options

echo
echo "══ 4/4  validator mutation test (must print 48/48) ══"
POD_VARIANT="$VARIANT" python tests/test_option_validator_mutations.py

echo
echo "══ ALL CHECKS PASSED — ${VARIANT} is reproducible and validated ══"
echo "   (scratch copy left at ${VERIFY_DIR}; remove it when done)"
