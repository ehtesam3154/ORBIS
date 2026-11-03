#!/bin/bash
# REMAP on mouse 1 (mice 2, 3, 4 -> mouse 1).
# REMAP, test mouse 1: reference = EVERY cell of mice 2, 3, 4 in mouse 2's ruler (2,975,768), query = every mouse-1 cell;
# harmony=False; official settings; memory-only path remap_lowmem.py (--lowmem_all 1); 8 threads.
# usage: DATA_DIR=... OUT_DIR=... [STAGES="train collect"] bash baselines/configs/remap_mouse1.sh
set -e
source "$(dirname "$0")/common.sh"
official REMAP https://github.com/ShunzhouJiang/REMAP.git b663e2ba21d93a8451e088336fd3fdc32633ae5b
export PYTHONPATH=$REMAP_SRC${PYTHONPATH:+:$PYTHONPATH} CUDA_VISIBLE_DEVICES=${GPU:-0} TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8
if has train; then
  $PY_REMAP -u $BL/remap/remap_3d.py --ref abca2,abca3_ref2,abca4_ref2 --query abca1 --n_ref 0 --harmony 0 --lowmem_all 1 --out_tag f2_q1_all --root $STORES --out_root $OUT_DIR/remap
fi
if has collect; then
  mouse1_map; $PY_TOOLS $BL/remap/to_layout.py $OUT_DIR/remap/bl_refreg_f2_q1_all/final_coords.npz $LAYOUTS/remap_mouse1.npz $(layout_args mouse1)
fi
