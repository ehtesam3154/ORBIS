#!/bin/bash
# Allen MERFISH brain 638850: cell store, test prebuilt and Harmony-corrected features. Run after prepare_abc_mice.sh.
# Downloads in $DATA_DIR/raw/allen_merfish/ (see prepare_allen_merfish.py); outputs in $DATA_DIR/brains/.
# Usage: DATA_DIR=<data folder> bash scripts/data/prepare_allen_merfish.sh
set -euo pipefail
DATA_DIR=${DATA_DIR:?set DATA_DIR}
PY=${PY:-python}
P="$PY scripts/data/prepare_allen_merfish.py"
B=$DATA_DIR/brains
S=$B/stores

$P store --raw_dir $DATA_DIR/raw/allen_merfish --zhuang_raw_dir $DATA_DIR/raw/abc_mice --ref_store $S/abca1 --class_list $S/classes.json --out_dir $S/allen638850
$P test --store $S/allen638850 --raw_dir $DATA_DIR/raw/allen_merfish --infos_from $B/prebuilt_test_mouse2 --out_dir $B/prebuilt_test_allen638850
$P harmony --stores_dir $S --ref1_store $S/abca2_ref1 --allen_test $B/prebuilt_test_allen638850 \
  --train_prebuilt $B/prebuilt_train --k 200 --out $B/allen_harmony
