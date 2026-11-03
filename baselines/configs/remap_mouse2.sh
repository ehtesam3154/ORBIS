#!/bin/bash
# REMAP on mouse 2 (mice 1, 3, 4 -> mouse 2).
# REMAP, test mouse 2: reference = a seeded uniform 400,000-cell subsample of mice 1, 3, 4 (proportional per animal; memory),
# query = every mouse-2 cell; Harmony on (the authors' advice for cross-subject use); official code path otherwise.
# Flags as the run's log reports (reference composition, Harmony); the run's command line itself was not logged.
# usage: DATA_DIR=... OUT_DIR=... [STAGES="train collect"] bash baselines/configs/remap_mouse2.sh
set -e
source "$(dirname "$0")/common.sh"
official REMAP https://github.com/ShunzhouJiang/REMAP.git b663e2ba21d93a8451e088336fd3fdc32633ae5b
export PYTHONPATH=$REMAP_SRC${PYTHONPATH:+:$PYTHONPATH} CUDA_VISIBLE_DEVICES=${GPU:-0} TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8
if has train; then
  $PY_REMAP -u $BL/remap/remap_3d.py --ref abca1,abca3,abca4 --query abca2 --n_ref 400000 --harmony 1 --root $STORES --out_root $OUT_DIR/remap
fi
if has collect; then
  $PY_TOOLS $BL/remap/to_layout.py $OUT_DIR/remap/bl_refreg_m134/final_coords.npz $LAYOUTS/remap_mouse2.npz $(layout_args mouse2)
fi
