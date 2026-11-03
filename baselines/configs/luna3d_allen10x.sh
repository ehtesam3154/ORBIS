#!/bin/bash
# LUNA-3D in the dissociated-cell (10x) application.
# Training: EVERY cell of the five spatial brains (Zhuang mice 1-4 and the Allen hemisphere: stores of $DATA_DIR/allen10x/stores) in the
# 200-feature reference space, random ~5,000-cell graphs (seed 0); recipe as released (LR 5e-4, seed 0) except 300 epochs, checkpoint every
# 100 -> epoch=299.ckpt; global batch 8 (final run: 1 GPU x 8, 8 DataLoader workers).
# Sampling: every cohort of the application (primary and strat, cohorts 1-5, seeds 1000 / 1001 / 1002), expression only, 20 random graphs of
# 5,000 cells per cohort (partition seed 25000 + cohort), LUNA's 1,000-step sampler; outputs $APP/raw/baseline_<kind><c>_s<seed>.npz (+ .json
# with train_unit_EXT), read by scripts/allen10x/application.py. The cohorts come from `scripts/allen10x/run_application.sh cohorts`.
# usage: DATA_DIR=... OUT_DIR=... [APP_DIR=$OUT_DIR/application STAGES="prep train sample" GPU=0] bash baselines/configs/luna3d_allen10x.sh
set -e
STAGES=${STAGES:-"prep train sample"}
source "$(dirname "$0")/common.sh"
A10=$DATA_DIR/allen10x; W=$OUT_DIR/luna3d/inputs/allen10x; RUNS=$OUT_DIR/luna3d/runs; NAME=luna3d_a10x_allcells
APP=${APP_DIR:-$OUT_DIR/application}
official LUNA https://github.com/mlbio-epfl/LUNA.git 64f10eb55a4cdc862c0535051a1fe042ab2ceadc $BL/luna3d/luna_3d.patch
if has prep; then
  $PY_LUNA $BL/luna3d/make_luna_bin_a10x.py --fold a10x --store_root $A10/stores --out $W/foldA10X_allcells_bin
  $PY_LUNA $BL/luna3d/make_a10x_tiny_csvs.py --store_root $A10/stores --out $W/foldA10X_allcells
fi
if has train; then
  LUNA_SRC=$LUNA_SRC PYTHON=$PY_LUNA N_EPOCHS=300 SAVE_EVERY=100 GLOBAL_BATCH=8 WORKERS=8 NGENES=200 \
    bash $BL/luna3d/run_luna3d_train_ddp.sh $NAME ${GPUS:-0} $W/foldA10X_allcells_bin $W/foldA10X_allcells/test_tiny.csv $RUNS
fi
if has sample; then
  mkdir -p $APP/raw
  for kind in primary strat; do for c in 1 2 3 4 5; do for s in 1000 1001 1002; do
    d=$APP/cohorts/$kind$c
    PYTHONPATH=$LUNA_SRC${PYTHONPATH:+:$PYTHONPATH} TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
      $PY_LUNA $BL/luna3d/luna_sample_features.py --ckpt $RUNS/$NAME/checkpoints/epoch=299.ckpt --run_dir $RUNS/$NAME \
      --train_dir $W/foldA10X_allcells_bin --features $d/features.npy --ids $d/ids.npy --feature_names_json $d/names.json \
      --partition_seed $((25000 + c)) --seed_offset $s --init_seed $s --steps 1000 --gpu ${GPU:-0} --out $APP/raw/baseline_${kind}${c}_s${s}.npz
  done; done; done
fi
