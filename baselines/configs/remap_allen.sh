#!/bin/bash
# REMAP on Allen 638850 (Zhuang mice 1-4 -> Allen brain).
# REMAP, test Allen: reference = EVERY cell of Zhuang mice 1-4 minus 83 zero-count cells (5,822,593), query = every Allen cell;
# 477 measured genes; the SAVED Harmony input (genes reconstructed from the 200 corrected PCs) fed as the tutorials feed external
# correction (harmony=False + obsm['corrected']); official settings; memory-only fit (--fit lowmem2); 8 threads.
# usage: DATA_DIR=... OUT_DIR=... [STAGES="train collect"] bash baselines/configs/remap_allen.sh
set -e
source "$(dirname "$0")/common.sh"
official REMAP https://github.com/ShunzhouJiang/REMAP.git b663e2ba21d93a8451e088336fd3fdc32633ae5b
export PYTHONPATH=$REMAP_SRC${PYTHONPATH:+:$PYTHONPATH} CUDA_VISIBLE_DEVICES=${GPU:-0} TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8
if has train; then
  $PY_REMAP -u $BL/remap/remap_3d_stores.py --ref_dirs $STORES/abca1,$STORES/abca2_ref1,$STORES/abca3,$STORES/abca4 --query_dir $STORES/allen638850 \
    --genes_meta $STORES/allen638850/meta.json --genes_key measured --drop_zero_ref 1 --harmony 0 \
    --corr_dirs $HARM/stores/abca1,$HARM/stores/abca2,$HARM/stores/abca3,$HARM/stores/abca4 --corr_query $HARM/allen/feat.npy \
    --corr_query_map $PRE_ALLEN/cell_ID.npy --corr_pca $HARM/pca.npz --n_ref 0 --fit lowmem2 \
    --out $OUT_DIR/remap/remap_allen
fi
if has collect; then
  $PY_TOOLS $BL/remap/to_layout.py $OUT_DIR/remap/remap_allen/final_coords.npz $LAYOUTS/remap_allen.npz $(layout_args allen)
fi
