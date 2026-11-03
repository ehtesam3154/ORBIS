#!/bin/bash
# Flysta3D-v2 embryos. Downloads in $DATA_DIR/raw/flysta3d/; outputs in $DATA_DIR/embryos/ (all shared genes: the model's
# training and test data) and $DATA_DIR/embryos_panel/ (the 2,000 split genes: read by the reseat step and the baselines).
# Usage: DATA_DIR=<data folder> bash scripts/data/prepare_flysta3d.sh
set -euo pipefail
DATA_DIR=${DATA_DIR:?set DATA_DIR}
${PY:-python} scripts/data/prepare_flysta3d.py --features all --raw $DATA_DIR/raw/flysta3d --out $DATA_DIR/embryos
${PY:-python} scripts/data/prepare_flysta3d.py --features panel --raw $DATA_DIR/raw/flysta3d --out $DATA_DIR/embryos_panel
