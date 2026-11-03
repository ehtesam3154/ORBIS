#!/bin/bash
# Tangram-3D on embryo c (11 training embryos -> 17.34h_E14-16h_c).
# Tangram-3D, test embryo c (11 training embryos -> 17.34h_E14-16h_c): mode 'cells', density prior 'uniform', lr 0.1; reference = seeded uniform subsample of
# 48,000 training cells; query in 10,000-cell chunks; 1000 epochs (Tangram's default).
# Position = the 3-D coordinate of the most probable reference cell.
# usage: DATA_DIR=... OUT_DIR=... [STAGES="train collect"] bash baselines/configs/tangram3d_embryo_c.sh
set -e
source "$(dirname "$0")/common.sh"
official Tangram https://github.com/broadinstitute/Tangram.git 4c68995a418f41dc8caef567598c4d9b47781a13
export PYTHONPATH=$TANGRAM_SRC${PYTHONPATH:+:$PYTHONPATH} CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
if has train; then
  $PY_TANGRAM -W ignore -u $BL/tangram3d/tangram_3d.py --ref_dirs $FLY --query_dir $EMB/prebuilt_test_17.34h_E14-16h_c \
    --n_ref 48000 --chunk 10000 --epochs 1000 --out $OUT_DIR/tangram3d/tangram3d_embryo_c
fi
if has collect; then
  $PY_TOOLS $BL/remap/to_layout.py $OUT_DIR/tangram3d/tangram3d_embryo_c/final_coords.npz $LAYOUTS/tangram3d_embryo_c.npz $(layout_args embryo_c)
fi
