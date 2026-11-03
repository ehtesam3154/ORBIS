#!/bin/bash
# LUNA-3D on embryo d (11 training embryos -> 16.88h_E14-16h_d).
# LUNA-3D, test embryo d (11 training embryos -> 16.88h_E14-16h_d): ONE run trains on EVERY cell of the 11 training embryos (2,000 HVGs, full body) and is
# tested on both held-out embryos (part 00 = embryo d, part 01 = embryo c).
# Recipe as released (LR 5e-4, seed 0) except 300 epochs, checkpoint every 100 -> epoch=299.ckpt; global batch 8 (final run: 1 GPU x 8,
# 8 DataLoader workers). Collect: protocol R per embryo (one physical ruler, not folded).
# usage: DATA_DIR=... OUT_DIR=... [STAGES="prep train test collect"] bash baselines/configs/luna3d_embryo_d.sh
set -e
source "$(dirname "$0")/common.sh"
W=$OUT_DIR/luna3d/inputs/embryos; RUNS=$OUT_DIR/luna3d/runs; NAME=luna3d_foldflysta
official LUNA https://github.com/mlbio-epfl/LUNA.git 64f10eb55a4cdc862c0535051a1fe042ab2ceadc $BL/luna3d/luna_3d.patch
if has prep && [ ! -f $W/foldflysta/manifest.json ]; then
  $PY_LUNA $BL/luna3d/make_luna_flysta.py --root $EMB --stores "$(echo $FLY_STORES | tr ' ' ,)" \
    --test 16.88h_E14-16h_d:20,17.34h_E14-16h_c:21 --fold flysta --n_genes 2000 --out $W
fi
if has train && [ ! -f $RUNS/$NAME/checkpoints/epoch=299.ckpt ]; then   # one run serves embryo d and embryo c
  LUNA_SRC=$LUNA_SRC PYTHON=$PY_LUNA N_EPOCHS=300 SAVE_EVERY=100 GLOBAL_BATCH=8 WORKERS=8 NGENES=2000 \
    bash $BL/luna3d/run_luna3d_train_ddp.sh $NAME ${GPUS:-0} $W/foldflysta_allcells_bin $W/foldflysta/test_tiny.csv $RUNS
fi
if has test; then
  LUNA_SRC=$LUNA_SRC PYTHON=$PY_LUNA CKPT=epoch=299.ckpt NGENES=2000 bash $BL/luna3d/run_luna3d_test.sh flysta ${GPU:-0} $W $RUNS 00
fi
if has collect; then
  P=$EMB/prebuilt_test_16.88h_E14-16h_d
  $PY_TOOLS $BL/luna3d/luna_unstretched.py --pred_root $RUNS/$NAME --train_table $W/foldflysta_allcells_bin --truth_dir $P \
    --row_map $P/cell_ID.npy --truth_pos $P/pos_canonical.npy --truth_scale $MM_EMB --id_prefix 20 --mixed_parts --out_mm_per_unit $MM_EMB \
    --save_rotation $LAYOUTS/luna3d_embryo_d_rotation.npy --out $LAYOUTS/luna3d_embryo_d.npz
fi
