#!/bin/bash
# Train one model.  usage: bash scripts/train.sh CONFIG
# CONFIG is a bash file that sets DATA_DIR (inputs), OUT_DIR (outputs) and the ARGS array of training options
# (paths inside ARGS are built from DATA_DIR / OUT_DIR). NPROC (default 1) = GPUs, one process each.
set -euo pipefail
CONFIG=${1:?usage: bash scripts/train.sh CONFIG}
source "$CONFIG"
DATA_DIR=${DATA_DIR:?set DATA_DIR}
OUT_DIR=${OUT_DIR:?set OUT_DIR}
NPROC=${NPROC:-1}
cd "$(dirname "$0")/.."
mkdir -p "$OUT_DIR"
PORT=$(python -c 'import socket; s = socket.socket(); s.bind(("localhost", 0)); print(s.getsockname()[1])')
torchrun --nnodes=1 --nproc_per_node="$NPROC" --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" \
    -m orbis.train.train "${ARGS[@]}"
