#!/bin/bash
# LUNA-3D on Allen 638850 (Zhuang mice 1-4 -> Allen brain).
# LUNA-3D, test Allen 638850 (Zhuang mice 1-4 -> Allen brain): trains on EVERY cell of abca1, abca2_ref1, abca3, abca4 (Harmony PC store views), samples every test cell.
# Inputs: random ~5,000-cell graphs (seed 0); the training table is binary and memory-mapped; the 200 saved Harmony PCs as features.
# Recipe as released (LR 5e-4, seed 0, 1,000-step sampler) except 300 epochs, checkpoint every 100 -> epoch=299.ckpt; global batch 8
# (final run: 4 GPUs x 2, 6 DataLoader workers per rank). Test: LUNA's own sampler on all 15 test parts.
# Collect: protocol R (training-graph unit, ONE rotation / reflection, no scale); truth folded at the midline for the fit.
# usage: DATA_DIR=... OUT_DIR=... [STAGES="prep train test collect"] bash baselines/configs/luna3d_allen.sh
set -e
source "$(dirname "$0")/common.sh"
W=$OUT_DIR/luna3d/inputs/allen; RUNS=$OUT_DIR/luna3d/runs; NAME=luna3d_foldallen_allcells
official LUNA https://github.com/mlbio-epfl/LUNA.git 64f10eb55a4cdc862c0535051a1fe042ab2ceadc $BL/luna3d/luna_3d.patch
if has prep; then
  $PY_LUNA $BL/luna3d/make_allen_store_views.py $W/stores $HARM/abca2_ref1_meta.json $PRE_ALLEN $HARM $STORES
  $PY_LUNA $BL/luna3d/make_luna_csv.py --fold allen --store_root $W/stores --out $W
  $PY_LUNA $BL/luna3d/make_luna_bin.py --fold allen --store_root $W/stores --out $W/foldallen_allcells_bin
  mkdir -p $W/foldallen_allcells; ln -sfn $W/foldallen/train_infos.csv $W/foldallen/test_part*.csv $W/foldallen_allcells/
  (head -1 $W/foldallen/test_part00.csv; tail -n +2 $W/foldallen/test_part00.csv | grep ",allen638850_g0016,") > $W/foldallen_allcells/test_tiny.csv
fi
if has train; then
  LUNA_SRC=$LUNA_SRC PYTHON=$PY_LUNA N_EPOCHS=300 SAVE_EVERY=100 GLOBAL_BATCH=8 WORKERS=6 NGENES=200 \
    bash $BL/luna3d/run_luna3d_train_ddp.sh $NAME ${GPUS:-0,1,2,3} $W/foldallen_allcells_bin $W/foldallen_allcells/test_tiny.csv $RUNS
fi
if has test; then
  LUNA_SRC=$LUNA_SRC PYTHON=$PY_LUNA CKPT=epoch=299.ckpt NGENES=200 bash $BL/luna3d/run_luna3d_test.sh allen_allcells ${GPU:-0} $W $RUNS 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14
fi
if has collect; then
  $PY_TOOLS $BL/luna3d/luna_unstretched.py --pred_root $RUNS/$NAME --train_table $W/foldallen_allcells_bin \
    --truth_dir $PRE_ALLEN --row_map $PRE_ALLEN/cell_ID.npy --truth_pos $STORES/allen638850/pos.npy --truth_scale $ISO \
    --store_ruler $S1 --train_ruler $S1 --fold_meta $STORES/allen638850/meta.json --id_prefix 9 --out_mm_per_unit $ISO --out $LAYOUTS/luna3d_allen.npz
fi
