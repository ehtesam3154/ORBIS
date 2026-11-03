#!/bin/bash
# REMAP on embryo d (11 training embryos -> 16.88h_E14-16h_d).
# REMAP, test embryo d: reference = EVERY cell of the 11 training embryos minus 34 zero-count cells, 2,000 HVGs, full body;
# harmony=False; official settings; --fit lowmem; 8 threads.
# usage: DATA_DIR=... OUT_DIR=... [STAGES="train collect"] bash baselines/configs/remap_embryo_d.sh
set -e
source "$(dirname "$0")/common.sh"
official REMAP https://github.com/ShunzhouJiang/REMAP.git b663e2ba21d93a8451e088336fd3fdc32633ae5b
export PYTHONPATH=$REMAP_SRC${PYTHONPATH:+:$PYTHONPATH} CUDA_VISIBLE_DEVICES=${GPU:-0} TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8
if has train; then
  $PY_REMAP -u $BL/remap/remap_3d_stores.py --ref_dirs $FLY --query_dir $EMB/prebuilt_test_16.88h_E14-16h_d --drop_zero_ref 1 --n_ref 0 --harmony 0 --fit lowmem \
    --out $OUT_DIR/remap/remap_embryo_d
fi
if has collect; then
  $PY_TOOLS $BL/remap/to_layout.py $OUT_DIR/remap/remap_embryo_d/final_coords.npz $LAYOUTS/remap_embryo_d.npz $(layout_args embryo_d)
fi
