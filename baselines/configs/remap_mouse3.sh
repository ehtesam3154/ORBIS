#!/bin/bash
# REMAP on mouse 3 (mice 1, 2 -> mouse 3).
# REMAP, test mouse 3: reference = EVERY cell of mice 1 + 2 (4,074,316), query = every mouse-3 cell; harmony=False;
# official settings (k 100, 100 covariance genes, niche model 10,000 steps, pve 0.98, n_iter 3, hidden 400 / 200 / 100, batch 256);
# memory-only path remap_lowmem.py (--lowmem_all 1). Threads: 8 (the thread count changes only float rounding).
# usage: DATA_DIR=... OUT_DIR=... [STAGES="train collect"] bash baselines/configs/remap_mouse3.sh
set -e
source "$(dirname "$0")/common.sh"
official REMAP https://github.com/ShunzhouJiang/REMAP.git b663e2ba21d93a8451e088336fd3fdc32633ae5b
export PYTHONPATH=$REMAP_SRC${PYTHONPATH:+:$PYTHONPATH} CUDA_VISIBLE_DEVICES=${GPU:-0} TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8
if has train; then
  $PY_REMAP -u $BL/remap/remap_3d.py --ref abca1,abca2_ref1 --query abca3 --n_ref 0 --harmony 0 --lowmem_all 1 --out_tag m12_q3_all --root $STORES --out_root $OUT_DIR/remap
fi
if has collect; then
  $PY_TOOLS $BL/remap/to_layout.py $OUT_DIR/remap/bl_refreg_m12_q3_all/final_coords.npz $LAYOUTS/remap_mouse3.npz $(layout_args mouse3)
fi
