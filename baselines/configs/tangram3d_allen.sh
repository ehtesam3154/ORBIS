#!/bin/bash
# Tangram-3D on Allen 638850 (Zhuang mice 1-4 -> Allen brain).
# Tangram-3D, test Allen 638850 (Zhuang mice 1-4 -> Allen brain): mode 'cells', density prior 'uniform', lr 0.1; reference = seeded uniform subsample of
# 15,000 training cells; query in 10,000-cell chunks; 500 epochs (the value in Tangram's own tests); reference and epochs set by the compute budget.
# Position = the 3-D coordinate of the most probable reference cell. Input: the saved Harmony input as reconstructed genes.
# usage: DATA_DIR=... OUT_DIR=... [STAGES="train collect"] bash baselines/configs/tangram3d_allen.sh
set -e
source "$(dirname "$0")/common.sh"
official Tangram https://github.com/broadinstitute/Tangram.git 4c68995a418f41dc8caef567598c4d9b47781a13
export PYTHONPATH=$TANGRAM_SRC${PYTHONPATH:+:$PYTHONPATH} CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
if has train; then
  $PY_TANGRAM -W ignore -u $BL/tangram3d/tangram_3d.py --ref_dirs $STORES/abca1,$STORES/abca2_ref1,$STORES/abca3,$STORES/abca4 --query_dir $STORES/allen638850 \
    --genes_meta $STORES/allen638850/meta.json --genes_key measured --corr_dirs $HARM/stores/abca1,$HARM/stores/abca2,$HARM/stores/abca3,$HARM/stores/abca4 \
    --corr_query $HARM/allen/feat.npy --corr_query_map $PRE_ALLEN/cell_ID.npy --corr_pca $HARM/pca.npz \
    --n_ref 15000 --chunk 10000 --epochs 500 --out $OUT_DIR/tangram3d/tangram3d_allen
fi
if has collect; then
  $PY_TOOLS $BL/remap/to_layout.py $OUT_DIR/tangram3d/tangram3d_allen/final_coords.npz $LAYOUTS/tangram3d_allen.npz $(layout_args allen)
fi
