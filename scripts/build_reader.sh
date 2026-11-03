#!/bin/bash
# Fit the niche reader of one test set (or of several test sets sharing the training specimens) on CPU.
# usage: bash scripts/build_reader.sh CONFIG
# CONFIG is a shell file that sets:
#   DATA_DIR, OUT_DIR       input and output roots
#   READER_NAME             name of the reader (outputs reader_<name>_<test>.npz, k32_<name>.npz, reader_<name>.json)
#   STORES=( ... )          training store directories (feat.npy, pos.npy[, sec.npy]); the order matters
#   POS_DIRS=( ... )        optional: directories with pos.npy / sec.npy when they differ from STORES (one per store)
#   MM_PER_UNIT or UM_PER_UNIT   store positions in um = pos x MM_PER_UNIT x 1000, or pos x UM_PER_UNIT (exact values)
#   SECTIONS                1 = serial sections (in-section neighbours, needs sec.npy), 0 = continuous 3-D
#   PANEL                   first | largest | none (specimen of the excluded Moran gene panel)
#   SPLIT                   sections | tiles | specimen (held-out rows); VAL_SPECIMEN / INNER_SPECIMEN for specimen
#   TESTS=( NAME=DIR ... )  test sets (DIR holds feat.npy with the stores' feature columns)
#   CHUNKED_STANDARDISER    1 = standardise in chunks (C-ordered output), 0 = in one piece (default); set per reader as fitted
# optional: THREADS (default 8), PY (python)
# Outputs: $OUT_DIR/readers/
set -euo pipefail
CONFIG=${1:?usage: bash scripts/build_reader.sh CONFIG}
source "$CONFIG"
DATA_DIR=${DATA_DIR:?set DATA_DIR}; OUT_DIR=${OUT_DIR:?set OUT_DIR}
: "${READER_NAME:?set READER_NAME}" "${PANEL:?set PANEL}" "${SPLIT:?set SPLIT}"
THREADS=${THREADS:-8}; PY=${PY:-python}; SECTIONS=${SECTIONS:-0}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ARGS=(--name "$READER_NAME" --stores "${STORES[@]}" --panel "$PANEL" --split "$SPLIT" --tests "${TESTS[@]}" --out_dir "$OUT_DIR/readers"
      --threads "$THREADS")
[ -n "${POS_DIRS+x}" ] && ARGS+=(--pos_dirs "${POS_DIRS[@]}")
[ -n "${MM_PER_UNIT:-}" ] && ARGS+=(--mm_per_unit "$MM_PER_UNIT")
[ -n "${UM_PER_UNIT:-}" ] && ARGS+=(--um_per_unit "$UM_PER_UNIT")
[ "$SECTIONS" = 1 ] && ARGS+=(--sections)
[ "${CHUNKED_STANDARDISER:-0}" = 1 ] && ARGS+=(--chunked_standardiser)
[ -n "${VAL_SPECIMEN:-}" ] && ARGS+=(--val_specimen "$VAL_SPECIMEN" --inner_specimen "${INNER_SPECIMEN:?set INNER_SPECIMEN}")
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS NUMBA_NUM_THREADS=$THREADS NUMPY_MADVISE_HUGEPAGE=0
export CUDA_VISIBLE_DEVICES=""
mkdir -p "$OUT_DIR/readers"
cd "$ROOT"
"$PY" -m orbis.reseat.build_reader "${ARGS[@]}" 2>&1 | tee "$OUT_DIR/readers/reader_$READER_NAME.log"
