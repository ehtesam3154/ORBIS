#!/bin/bash
# LUNA-3D on mouse 1 (mice 2, 3, 4 -> mouse 1).
# LUNA-3D, test mouse 1 (mice 2, 3, 4 -> mouse 1): trains on EVERY cell of abca2, abca3, abca4, samples every test cell.
# Inputs: random ~5,000-cell graphs (seed 0); the training table is binary and memory-mapped; 1122 features.
# Recipe as released (LR 5e-4, seed 0, 1,000-step sampler) except 300 epochs, checkpoint every 100 -> epoch=299.ckpt; global batch 8
# (final run: 1 GPU x 8, 8 DataLoader workers per rank). Test: LUNA's own sampler on all 12 test parts.
# Collect: protocol R (training-graph unit, ONE rotation / reflection, no scale).
# usage: DATA_DIR=... OUT_DIR=... [STAGES="prep train test collect"] bash baselines/configs/luna3d_mouse1.sh
set -e
source "$(dirname "$0")/common.sh"
W=$OUT_DIR/luna3d/inputs; RUNS=$OUT_DIR/luna3d/runs; NAME=luna3d_fold2_allcells
official LUNA https://github.com/mlbio-epfl/LUNA.git 64f10eb55a4cdc862c0535051a1fe042ab2ceadc $BL/luna3d/luna_3d.patch
if has prep; then
  $PY_LUNA $BL/luna3d/make_luna_csv.py --fold 2 --store_root $STORES --out $W
  $PY_LUNA $BL/luna3d/make_luna_bin.py --fold 2 --store_root $STORES --out $W/fold2_allcells_bin
  mkdir -p $W/fold2_allcells; ln -sfn $W/fold2/train_infos.csv $W/fold2/test_part*.csv $W/fold2_allcells/
  head -n 2001 $W/fold2/test_part00.csv > $W/fold2_allcells/test_tiny.csv
fi
if has train; then
  LUNA_SRC=$LUNA_SRC PYTHON=$PY_LUNA N_EPOCHS=300 SAVE_EVERY=100 GLOBAL_BATCH=8 WORKERS=8 NGENES=1122 \
    bash $BL/luna3d/run_luna3d_train_ddp.sh $NAME ${GPUS:-0} $W/fold2_allcells_bin $W/fold2_allcells/test_tiny.csv $RUNS
fi
if has test; then
  LUNA_SRC=$LUNA_SRC PYTHON=$PY_LUNA CKPT=epoch=299.ckpt NGENES=1122 bash $BL/luna3d/run_luna3d_test.sh 2_allcells ${GPU:-0} $W $RUNS 00 01 02 03 04 05 06 07 08 09 10 11
fi
if has collect; then
  mouse1_map; $PY_TOOLS $BL/luna3d/luna_unstretched.py --pred_root $RUNS/$NAME --train_table $W/fold2_allcells_bin \
    --truth_dir $PRE_MOUSE1 --row_map $MAP_MOUSE1 --truth_pos $STORES/abca1/pos.npy --truth_scale $ISO \
    --store_ruler $S1 --train_ruler $S2 --id_prefix 1 --out_mm_per_unit $ISO --out $LAYOUTS/luna3d_mouse1.npz
fi
