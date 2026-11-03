#!/bin/bash
# Score the sampled layouts of one test with the certified scorer, compute their per-cell errors at matched size and, once per
# test (or subset), the within-type-shuffle ruler.
#
# usage: DATA_DIR=... OUT_DIR=... scripts/score.sh CONFIG
#
# CONFIG is a shell file (sourced) that sets:
#   TEST            test card name: orbis/metrics/cards/$TEST.json
#   TEST_DIR        test prebuilt directory, relative to DATA_DIR
#   LAYOUTS         layout npz files (sampler outputs), relative to OUT_DIR, space separated
#   STORE_DIR       whole brains only: the test brain's cell store, relative to DATA_DIR
#   STORE_ROWS      whole brains whose card says store_rows "file": prebuilt -> store row map (.npy), relative to DATA_DIR
#   CELL_METADATA   whole brains only: atlas cell metadata CSV(s) in the order of the card's label rule, relative to DATA_DIR
#   KEEP            optional: sorted rows (.npy) of a subset of the test cells, relative to OUT_DIR
#   RULERS          optional: corruptions of the truth to score as well (e.g. "@truth @shrink0.5 @axis0x0.75 @noise200 @shuffle @collapse")
#   SCORE_DIR       optional: output directory relative to OUT_DIR (default scores/$TEST, or scores/$TEST/<KEEP stem>)
# Optional environment: PYTHON (default python), ORBIS_WORKERS (nearest-neighbour threads).
set -euo pipefail
DATA_DIR=${DATA_DIR:?set DATA_DIR}
OUT_DIR=${OUT_DIR:?set OUT_DIR}
CONFIG=${1:?usage: scripts/score.sh CONFIG}
source "$CONFIG"
TEST=${TEST:?CONFIG must set TEST}
TEST_DIR=${TEST_DIR:?CONFIG must set TEST_DIR}
LAYOUTS=${LAYOUTS:-}
RULERS=${RULERS:-}
KEEP=${KEEP:-}
PYTHON=${PYTHON:-python}
if [ -n "$KEEP" ]; then
  SCORE_DIR=${SCORE_DIR:-scores/$TEST/$(basename "$KEEP" .npy)}
else
  SCORE_DIR=${SCORE_DIR:-scores/$TEST}
fi
cd "$(dirname "$0")/.."

ARGS=(--card "orbis/metrics/cards/$TEST.json" --test_dir "$DATA_DIR/$TEST_DIR" --cache "$OUT_DIR/cache/$TEST")
[ -n "${STORE_DIR:-}" ] && ARGS+=(--store_dir "$DATA_DIR/$STORE_DIR")
[ -n "${STORE_ROWS:-}" ] && ARGS+=(--store_rows "$DATA_DIR/$STORE_ROWS")
if [ -n "${CELL_METADATA:-}" ]; then
  ARGS+=(--cell_metadata)
  for f in $CELL_METADATA; do ARGS+=("$DATA_DIR/$f"); done
fi
[ -n "$KEEP" ] && ARGS+=(--keep "$OUT_DIR/$KEEP")

LAYOUT_PATHS=()
for f in $LAYOUTS; do LAYOUT_PATHS+=("$OUT_DIR/$f"); done
RULER_ARGS=()
[ -n "$RULERS" ] && RULER_ARGS=(--rulers $RULERS)

if [ ${#LAYOUT_PATHS[@]} -gt 0 ] || [ -n "$RULERS" ]; then
  "$PYTHON" -m orbis.metrics.score "${ARGS[@]}" --layouts "${LAYOUT_PATHS[@]}" "${RULER_ARGS[@]}" --out_dir "$OUT_DIR/$SCORE_DIR"
fi
if [ ${#LAYOUT_PATHS[@]} -gt 0 ]; then
  "$PYTHON" -m orbis.metrics.matched "${ARGS[@]}" --layouts "${LAYOUT_PATHS[@]}" --out_dir "$OUT_DIR/$SCORE_DIR/matched"
fi
if [ ! -f "$OUT_DIR/$SCORE_DIR/ruler.json" ]; then
  "$PYTHON" -m orbis.metrics.ruler "${ARGS[@]}" --out "$OUT_DIR/$SCORE_DIR/ruler.json"
fi
