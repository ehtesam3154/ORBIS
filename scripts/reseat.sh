#!/bin/bash
# Reseat sampled layouts with the niche engine (one GPU per call).
# usage: bash scripts/reseat.sh CONFIG
# CONFIG is a shell file that sets:
#   DATA_DIR, OUT_DIR       input and output roots
#   SAMPLES=( ... )         finished sample layouts (npz with P, G, cc, iso)
#   GUESSES=( ... )         one-call best guess of the same checkpoint and sampler seed, one per sample (same order)
#   READER, READER_MD5      reader npz of the test set (scripts/build_reader.sh) and its md5 (READER_MD5 may be empty)
#   TEST_DIR                prebuilt test directory (pos.npy, cell_class.npy), relative to DATA_DIR
#   MM_PER_UNIT             millimetres per model unit of the test set (write the exact value, e.g. 13.45773983001709)
# optional: THREADS (default 2), QUANTILE (0.9), GPU (0), NVCC (default $CUDA_HOME/bin/nvcc), CCBIN (g++-11), PY (python)
# Outputs: $OUT_DIR/reseated/<sample file name> and a run json beside each (same name, .json).
set -euo pipefail
CONFIG=${1:?usage: bash scripts/reseat.sh CONFIG}
source "$CONFIG"
DATA_DIR=${DATA_DIR:?set DATA_DIR}; OUT_DIR=${OUT_DIR:?set OUT_DIR}
: "${READER:?set READER}" "${TEST_DIR:?set TEST_DIR}" "${MM_PER_UNIT:?set MM_PER_UNIT}"
READER_MD5=${READER_MD5:-}; THREADS=${THREADS:-2}; QUANTILE=${QUANTILE:-0.9}; GPU=${GPU:-0}
NVCC=${NVCC:-}; CCBIN=${CCBIN:-g++-11}; PY=${PY:-python}
[ ${#SAMPLES[@]} -eq ${#GUESSES[@]} ] || { echo "SAMPLES and GUESSES differ in length"; exit 1; }
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=$OUT_DIR/reseated; mkdir -p "$OUT"
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS NUMBA_NUM_THREADS=$THREADS NUMPY_MADVISE_HUGEPAGE=0
export CUDA_VISIBLE_DEVICES=$GPU
cd "$ROOT"
for i in "${!SAMPLES[@]}"; do
  S=${SAMPLES[$i]}; G=${GUESSES[$i]}; O=$OUT/$(basename "$S")
  if [ -e "$O" ]; then echo "exists: $O"; continue; fi
  "$PY" -m orbis.reseat.run --sample "$S" --guess "$G" --reader "$READER" --reader_md5 "$READER_MD5" --test_dir "$DATA_DIR/$TEST_DIR" \
    --mm_per_unit "$MM_PER_UNIT" --out "$O" --threads "$THREADS" --quantile "$QUANTILE" ${NVCC:+--nvcc "$NVCC"} --ccbin "$CCBIN" \
    --cuda_build_dir "$OUT/cuda_build" 2>&1 | tee "${O%.npz}.log"
done
