#!/bin/bash
# Sample layouts of one test set with one checkpoint (one GPU per call).
# usage: bash scripts/sample.sh CONFIG
# CONFIG is a shell file that sets:
#   DATA_DIR, OUT_DIR       input and output roots
#   CKPT                    checkpoint file (orbis_epoch<E>.pt)
#   SAMPLE_ARGS=( ... )     arguments of orbis.sample.run, at least --test_dir DIR --tag TAG --steps K
#                           (e.g. --steps 50 --schedule edm --calibrate 0 --global_state 1 --sample_seed 1000)
#   UNFOLD_STORES           optional: comma-separated training store directories of an embryo model trained on folded
#                           views; every written layout is then unfolded (orbis.sample.unfold, side seed UNFOLD_SEED, default 0)
# optional: GPU (default 0), THREADS (default 4), PY (python)
# Outputs: $OUT_DIR/samples/<tag>_steps<K>.npz, <tag>.json [, <tag>_unfold<seed>k<n>_steps<K>.npz]
set -euo pipefail
CONFIG=${1:?usage: bash scripts/sample.sh CONFIG}
source "$CONFIG"
DATA_DIR=${DATA_DIR:?set DATA_DIR}; OUT_DIR=${OUT_DIR:?set OUT_DIR}; CKPT=${CKPT:?set CKPT}
GPU=${GPU:-0}; THREADS=${THREADS:-4}; PY=${PY:-python}; UNFOLD_SEED=${UNFOLD_SEED:-0}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TAG=""; KS=(); opt=""
for x in "${SAMPLE_ARGS[@]}"; do
  if [[ $x == --* ]]; then opt=$x; continue; fi
  case $opt in --tag) TAG=$x ;; --steps) KS+=("$x") ;; esac
done
: "${TAG:?SAMPLE_ARGS must hold --tag}"
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS NUMPY_MADVISE_HUGEPAGE=0
mkdir -p "$OUT_DIR/samples"
cd "$ROOT"
"$PY" -m orbis.sample.run --ckpt "$CKPT" --out_dir "$OUT_DIR/samples" --gpu 0 "${SAMPLE_ARGS[@]}" 2>&1 | tee "$OUT_DIR/samples/$TAG.log"
if [ -n "${UNFOLD_STORES:-}" ]; then
  for K in "${KS[@]}"; do
    "$PY" -m orbis.sample.unfold "$OUT_DIR/samples/${TAG}_steps$K.npz" "$UNFOLD_SEED" "$UNFOLD_STORES" 2>&1 | tee -a "$OUT_DIR/samples/$TAG.log"
  done
fi
