#!/bin/bash
# CeLEry-3D on Allen 638850 (Zhuang mice 1-4 -> Allen brain).
# CeLEry-3D, test Allen 638850 (Zhuang mice 1-4 -> Allen brain): CeLEry's coordinate regressor (3 outputs) trained on EVERY reference cell; CPU, 1 BLAS thread.
# Defaults as released: hidden [30, 25, 15], batch 4, lr 1e-3, Adam wd 1e-5, LR halving rule. Budget rule in the driver:
# epochs = max(1, ceil(728,000 / steps per epoch)) -> 1 epoch (1,455,669 steps) in the final run.
# Features: the 200 saved Harmony PCs.
# usage: DATA_DIR=... OUT_DIR=... [STAGES="train collect"] bash baselines/configs/celery3d_allen.sh
set -e
source "$(dirname "$0")/common.sh"
official CeLEry https://github.com/QihuangZhang/CeLEry.git b0b73764bb517b0b7b360d4b18bed5d6b77a4615 $BL/celery3d/celery_3d.patch
export PYTHONPATH=$CELERY_SRC/CeLEry_package${PYTHONPATH:+:$PYTHONPATH} CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
if has train; then
  $PY_CELERY -W ignore -u $BL/celery3d/celery_3d.py --ref_dirs $STORES/abca1,$STORES/abca2_ref1,$STORES/abca3,$STORES/abca4 --query_dir $STORES/allen638850 \
    --pcs --pc_ref_dirs $HARM/stores/abca1,$HARM/stores/abca2,$HARM/stores/abca3,$HARM/stores/abca4 \
    --pc_query $HARM/allen/feat.npy --pc_query_map $PRE_ALLEN/cell_ID.npy --out $OUT_DIR/celery3d/celery3d_allen
fi
if has collect; then
  $PY_TOOLS $BL/remap/to_layout.py $OUT_DIR/celery3d/celery3d_allen/final_coords.npz $LAYOUTS/celery3d_allen.npz $(layout_args allen)
fi
