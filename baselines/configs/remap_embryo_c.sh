#!/bin/bash
# REMAP on embryo c (11 training embryos -> 17.34h_E14-16h_c).
# REMAP, test embryo c: as embryo d; the one zero-count test cell gets the mean predicted position (--query_zero fill_mean).
# usage: DATA_DIR=... OUT_DIR=... [STAGES="train collect"] bash baselines/configs/remap_embryo_c.sh
set -e
source "$(dirname "$0")/common.sh"
official REMAP https://github.com/ShunzhouJiang/REMAP.git b663e2ba21d93a8451e088336fd3fdc32633ae5b
export PYTHONPATH=$REMAP_SRC${PYTHONPATH:+:$PYTHONPATH} CUDA_VISIBLE_DEVICES=${GPU:-0} TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8
if has train; then
  $PY_REMAP -u $BL/remap/remap_3d_stores.py --ref_dirs $FLY --query_dir $EMB/prebuilt_test_17.34h_E14-16h_c --drop_zero_ref 1 --query_zero fill_mean --n_ref 0 --harmony 0 --fit lowmem \
    --out $OUT_DIR/remap/remap_embryo_c
fi
if has collect; then
  $PY_TOOLS $BL/remap/to_layout.py $OUT_DIR/remap/remap_embryo_c/final_coords.npz $LAYOUTS/remap_embryo_c.npz $(layout_args embryo_c)
fi
