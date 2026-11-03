#!/bin/bash
# novoSpaRc-3D on Deep-STARmap block B (replicate block -> main block).
# novoSpaRc-3D, test Deep-STARmap block B (replicate block -> main block): atlas-assisted, the authors' Drosophila tutorial settings (alpha_linear 0.8, epsilon 5e-3,
# k 5 / 5, all shared genes as markers); target = seeded uniform subsample of 10,000 training cells; query in seeded 10,000-cell
# chunks (2 part(s)); position = transport-plan-weighted mean of the target positions; CPU, 4 BLAS threads per part.
# Target and chunk size set by the compute budget.
# usage: DATA_DIR=... OUT_DIR=... [STAGES="train collect"] bash baselines/configs/novosparc3d_blockB.sh
set -e
source "$(dirname "$0")/common.sh"
official novosparc https://github.com/rajewsky-lab/novosparc.git a3c9f3d76e59c7a601cfcd0e211630e2458d79a3
export PYTHONPATH=$NOVOSPARC_SRC${PYTHONPATH:+:$PYTHONPATH} CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=${THREADS:-4} MKL_NUM_THREADS=${THREADS:-4} OPENBLAS_NUM_THREADS=${THREADS:-4} NUMBA_NUM_THREADS=${THREADS:-4}
D=$OUT_DIR/novosparc3d/novosparc3d_blockB; NP=2   # the parts are independent shares of the same seeded chunks; run them in parallel if wanted
if has train; then
  for p in $(seq 0 $((NP - 1))); do
    [ -f $D/part_$p.npz ] || $PY_NOVOSPARC -W ignore -u $BL/novosparc3d/novosparc_3d.py --ref_dirs $BLK/foldB/stores/STARmap_replicate --query_dir $BLK/foldB/prebuilt_test_STARmap_main \
      --n_ref 10000 --chunk 10000 --part $p/$NP --out $D
  done
  $PY_NOVOSPARC -W ignore $BL/novosparc3d/novosparc_3d.py --ref_dirs $BLK/foldB/stores/STARmap_replicate --query_dir $BLK/foldB/prebuilt_test_STARmap_main --n_ref 10000 --chunk 10000 --merge $NP --out $D
fi
if has collect; then
  $PY_TOOLS $BL/remap/to_layout.py $D/final_coords.npz $LAYOUTS/novosparc3d_blockB.npz $(layout_args blockB)
fi
