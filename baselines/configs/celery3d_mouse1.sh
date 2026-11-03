#!/bin/bash
# CeLEry-3D on mouse 1 (mice 2, 3, 4 -> mouse 1).
# CeLEry-3D, test mouse 1 (mice 2, 3, 4 -> mouse 1): CeLEry's coordinate regressor (3 outputs) trained on EVERY reference cell; CPU, 1 BLAS thread.
# Defaults as released: hidden [30, 25, 15], batch 4, lr 1e-3, Adam wd 1e-5, LR halving rule. Budget rule in the driver:
# epochs = max(1, ceil(728,000 / steps per epoch)) -> 1 epoch (743,942 steps) in the final run.
# Features: counts -> library size 1e4 -> log1p -> CeLEry's z-score.
# usage: DATA_DIR=... OUT_DIR=... [STAGES="train collect"] bash baselines/configs/celery3d_mouse1.sh
set -e
source "$(dirname "$0")/common.sh"
official CeLEry https://github.com/QihuangZhang/CeLEry.git b0b73764bb517b0b7b360d4b18bed5d6b77a4615 $BL/celery3d/celery_3d.patch
export PYTHONPATH=$CELERY_SRC/CeLEry_package${PYTHONPATH:+:$PYTHONPATH} CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
if has train; then
  $PY_CELERY -W ignore -u $BL/celery3d/celery_3d.py --ref_dirs $STORES/abca2,$STORES/abca3_ref2,$STORES/abca4_ref2 --query_dir $STORES/abca1 --out $OUT_DIR/celery3d/celery3d_mouse1
fi
if has collect; then
  mouse1_map; $PY_TOOLS $BL/remap/to_layout.py $OUT_DIR/celery3d/celery3d_mouse1/final_coords.npz $LAYOUTS/celery3d_mouse1.npz $(layout_args mouse1)
fi
