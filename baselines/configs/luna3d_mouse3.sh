#!/bin/bash
# LUNA-3D on mouse 3 (mice 1, 2 -> mouse 3).
# LUNA-3D, test mouse 3 (mice 1, 2 -> mouse 3): trains on EVERY cell of abca1, abca2_ref1, samples every test cell.
# Inputs: random ~5,000-cell graphs (seed 0); the training table is binary and memory-mapped; 1122 features.
# Recipe as released (LR 5e-4, seed 0, 1,000-step sampler) except 300 epochs, checkpoint every 100 -> epoch=299.ckpt; global batch 8
# (final run: 4 GPUs x 2, 6 DataLoader workers per rank). Test: LUNA's own sampler on all 7 test parts.
# Collect: protocol R (training-graph unit, ONE rotation / reflection, no scale).
# usage: DATA_DIR=... OUT_DIR=... [STAGES="prep train test collect"] bash baselines/configs/luna3d_mouse3.sh
set -e
source "$(dirname "$0")/common.sh"
W=$OUT_DIR/luna3d/inputs; RUNS=$OUT_DIR/luna3d/runs; NAME=luna3d_foldm12_allcells
official LUNA https://github.com/mlbio-epfl/LUNA.git 64f10eb55a4cdc862c0535051a1fe042ab2ceadc $BL/luna3d/luna_3d.patch
if has prep; then
  $PY_LUNA $BL/luna3d/make_luna_csv.py --fold m12 --store_root $STORES --out $W
  $PY_LUNA $BL/luna3d/make_luna_bin.py --fold m12 --store_root $STORES --out $W/foldm12_allcells_bin
  mkdir -p $W/foldm12_allcells; ln -sfn $W/foldm12/train_infos.csv $W/foldm12/test_part*.csv $W/foldm12_allcells/
  (head -1 $W/foldm12/test_part00.csv; tail -n +2 $W/foldm12/test_part00.csv | grep ",abca3_g0027,") > $W/foldm12_allcells/test_tiny.csv
fi
if has train; then
  LUNA_SRC=$LUNA_SRC PYTHON=$PY_LUNA N_EPOCHS=300 SAVE_EVERY=100 GLOBAL_BATCH=8 WORKERS=6 NGENES=1122 \
    bash $BL/luna3d/run_luna3d_train_ddp.sh $NAME ${GPUS:-0,1,2,3} $W/foldm12_allcells_bin $W/foldm12_allcells/test_tiny.csv $RUNS
fi
if has test; then
  LUNA_SRC=$LUNA_SRC PYTHON=$PY_LUNA CKPT=epoch=299.ckpt NGENES=1122 bash $BL/luna3d/run_luna3d_test.sh m12_allcells ${GPU:-0} $W $RUNS 00 01 02 03 04 05 06
fi
if has collect; then
  $PY_TOOLS $BL/luna3d/luna_unstretched.py --pred_root $RUNS/$NAME --train_table $W/foldm12_allcells_bin \
    --truth_dir $PRE_MOUSE3 --row_map $MAP_MOUSE3 --truth_pos $STORES/abca3/pos.npy --truth_scale $ISO \
    --store_ruler $S1 --train_ruler $S1 --id_prefix 3 --out_mm_per_unit $ISO --out $LAYOUTS/luna3d_mouse3.npz
fi
