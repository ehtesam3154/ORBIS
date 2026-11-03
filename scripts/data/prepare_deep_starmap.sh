#!/bin/bash
# Deep-STARmap blocks, both folds. Downloads in $DATA_DIR/raw/deep_starmap/; outputs in $DATA_DIR/blocks/.
# Usage: DATA_DIR=<data folder> bash scripts/data/prepare_deep_starmap.sh
set -euo pipefail
DATA_DIR=${DATA_DIR:?set DATA_DIR}
${PY:-python} scripts/data/prepare_deep_starmap.py --raw $DATA_DIR/raw/deep_starmap --out $DATA_DIR/blocks
