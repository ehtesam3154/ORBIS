#!/bin/bash
# novoSpaRc-3D on mouse 1 (mice 2, 3, 4 -> mouse 1).
# novoSpaRc-3D, test mouse 1 (mice 2, 3, 4 -> mouse 1): atlas-assisted, the authors' Drosophila tutorial settings (alpha_linear 0.8, epsilon 5e-3,
# k 5 / 5, all shared genes as markers); target = seeded uniform subsample of 5,000 training cells; query in seeded 5,000-cell
# chunks (57 part(s)); position = transport-plan-weighted mean of the target positions; CPU, 4 BLAS threads per part.
# Target and chunk size set by the compute budget.
# usage: DATA_DIR=... OUT_DIR=... [STAGES="train collect"] bash baselines/configs/novosparc3d_mouse1.sh
set -e
source "$(dirname "$0")/common.sh"
official novosparc https://github.com/rajewsky-lab/novosparc.git a3c9f3d76e59c7a601cfcd0e211630e2458d79a3
export PYTHONPATH=$NOVOSPARC_SRC${PYTHONPATH:+:$PYTHONPATH} CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=${THREADS:-4} MKL_NUM_THREADS=${THREADS:-4} OPENBLAS_NUM_THREADS=${THREADS:-4} NUMBA_NUM_THREADS=${THREADS:-4}
D=$OUT_DIR/novosparc3d/novosparc3d_mouse1; NP=57   # the parts are independent shares of the same seeded chunks; run them in parallel if wanted
if has train; then
  for p in $(seq 0 $((NP - 1))); do
    [ -f $D/part_$p.npz ] || $PY_NOVOSPARC -W ignore -u $BL/novosparc3d/novosparc_3d.py --ref_dirs $STORES/abca2,$STORES/abca3_ref2,$STORES/abca4_ref2 --query_dir $STORES/abca1 \
      --n_ref 5000 --chunk 5000 --part $p/$NP --out $D
  done
  $PY_NOVOSPARC -W ignore $BL/novosparc3d/novosparc_3d.py --ref_dirs $STORES/abca2,$STORES/abca3_ref2,$STORES/abca4_ref2 --query_dir $STORES/abca1 --n_ref 5000 --chunk 5000 --merge $NP --out $D
fi
if has collect; then
  mouse1_map; $PY_TOOLS $BL/remap/to_layout.py $D/final_coords.npz $LAYOUTS/novosparc3d_mouse1.npz $(layout_args mouse1)
fi
