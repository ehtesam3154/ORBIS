#!/bin/bash
# REMAP on Deep-STARmap block A (main block -> replicate block).
# REMAP, test block A: reference = EVERY cell of the main block, query = every replicate-block cell; 1,017 genes; harmony=False;
# official settings; memory-only path (--fit lowmem); 8 threads.
# usage: DATA_DIR=... OUT_DIR=... [STAGES="train collect"] bash baselines/configs/remap_blockA.sh
set -e
source "$(dirname "$0")/common.sh"
official REMAP https://github.com/ShunzhouJiang/REMAP.git b663e2ba21d93a8451e088336fd3fdc32633ae5b
export PYTHONPATH=$REMAP_SRC${PYTHONPATH:+:$PYTHONPATH} CUDA_VISIBLE_DEVICES=${GPU:-0} TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8
if has train; then
  $PY_REMAP -u $BL/remap/remap_3d_stores.py --ref_dirs $BLK/foldA/stores/STARmap_main --query_dir $BLK/foldA/prebuilt_test_STARmap_replicate --n_ref 0 --harmony 0 --fit lowmem \
    --out $OUT_DIR/remap/remap_blockA
fi
if has collect; then
  $PY_TOOLS $BL/remap/to_layout.py $OUT_DIR/remap/remap_blockA/final_coords.npz $LAYOUTS/remap_blockA.npz $(layout_args blockA)
fi
