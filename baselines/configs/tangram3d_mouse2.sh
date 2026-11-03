#!/bin/bash
# Tangram-3D on mouse 2 (mice 1, 3, 4 -> mouse 2).
# Tangram-3D, test mouse 2 (mice 1, 3, 4 -> mouse 2): mode 'cells', density prior 'uniform', lr 0.1; reference = seeded uniform subsample of
# 15,000 training cells; query in 10,000-cell chunks; 500 epochs (the value in Tangram's own tests); reference and epochs set by the compute budget.
# Position = the 3-D coordinate of the most probable reference cell.
# usage: DATA_DIR=... OUT_DIR=... [STAGES="train collect"] bash baselines/configs/tangram3d_mouse2.sh
set -e
source "$(dirname "$0")/common.sh"
official Tangram https://github.com/broadinstitute/Tangram.git 4c68995a418f41dc8caef567598c4d9b47781a13
export PYTHONPATH=$TANGRAM_SRC${PYTHONPATH:+:$PYTHONPATH} CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
if has train; then
  $PY_TANGRAM -W ignore -u $BL/tangram3d/tangram_3d.py --ref_dirs $STORES/abca1,$STORES/abca3,$STORES/abca4 --query_dir $STORES/abca2 \
    --n_ref 15000 --chunk 10000 --epochs 500 --out $OUT_DIR/tangram3d/tangram3d_mouse2
fi
if has collect; then
  $PY_TOOLS $BL/remap/to_layout.py $OUT_DIR/tangram3d/tangram3d_mouse2/final_coords.npz $LAYOUTS/tangram3d_mouse2.npz $(layout_args mouse2)
fi
