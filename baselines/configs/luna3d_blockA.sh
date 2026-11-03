#!/bin/bash
# LUNA-3D on Deep-STARmap block A (main block -> replicate block).
# LUNA-3D, test Deep-STARmap block A (main block -> replicate block): trains on EVERY cell of the training block (1,017 genes), samples every test cell.
# Recipe as released (LR 5e-4, seed 0) except 400 epochs, checkpoint every 100 -> epoch=399.ckpt; global batch 8 (final run: 1 GPU x 8,
# 8 DataLoader workers). Test: LUNA's own sampler on the one test part. Collect: protocol R (one physical ruler, no fold).
# usage: DATA_DIR=... OUT_DIR=... [STAGES="prep train test collect"] bash baselines/configs/luna3d_blockA.sh
set -e
source "$(dirname "$0")/common.sh"
W=$OUT_DIR/luna3d/inputs/blocks; RUNS=$OUT_DIR/luna3d/runs; NAME=luna3d_folddstarA
official LUNA https://github.com/mlbio-epfl/LUNA.git 64f10eb55a4cdc862c0535051a1fe042ab2ceadc $BL/luna3d/luna_3d.patch
if has prep; then
  $PY_LUNA $BL/luna3d/make_luna_flysta.py --root $BLK/foldA --stores stores/STARmap_main --test STARmap_replicate:20 --fold dstarA --n_genes 1017 --out $W
fi
if has train; then
  LUNA_SRC=$LUNA_SRC PYTHON=$PY_LUNA N_EPOCHS=400 SAVE_EVERY=100 GLOBAL_BATCH=8 WORKERS=8 NGENES=1017 \
    bash $BL/luna3d/run_luna3d_train_ddp.sh $NAME ${GPUS:-0} $W/folddstarA_allcells_bin $W/folddstarA/test_tiny.csv $RUNS
fi
if has test; then
  LUNA_SRC=$LUNA_SRC PYTHON=$PY_LUNA CKPT=epoch=399.ckpt NGENES=1017 bash $BL/luna3d/run_luna3d_test.sh dstarA ${GPU:-0} $W $RUNS 00
fi
if has collect; then
  P=$BLK/foldA/prebuilt_test_STARmap_replicate
  $PY_TOOLS $BL/luna3d/luna_unstretched.py --pred_root $RUNS/$NAME --train_table $W/folddstarA_allcells_bin --truth_dir $P \
    --row_map $P/cell_ID.npy --truth_pos $P/pos_canonical.npy --truth_scale $MM_BLOCK_A --id_prefix 20 --out_mm_per_unit $MM_BLOCK_A \
    --save_rotation $LAYOUTS/luna3d_blockA_rotation.npy --out $LAYOUTS/luna3d_blockA.npz
fi
