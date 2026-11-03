#!/bin/bash
# Zhuang ABC-atlas mice 1-4: every prebuilt, cell store and row map the mouse experiments read, built in dependency order.
# Downloads in $DATA_DIR/raw/abc_mice/ (see prepare_abc_mice.py); outputs in $DATA_DIR/brains/.
# Usage (from the repository root): DATA_DIR=<data folder> bash scripts/data/prepare_abc_mice.sh
set -euo pipefail
DATA_DIR=${DATA_DIR:?set DATA_DIR}
PY=${PY:-python}
P="$PY scripts/data/prepare_abc_mice.py"
RAW=$DATA_DIR/raw/abc_mice
B=$DATA_DIR/brains
A=$B/annotation
S=$B/stores
STATS=$B/coord_stats_mouse1.json
mkdir -p $S $B/base

$P annotate --raw_dir $RAW --out_dir $A
$P classes --raw_dir $RAW --annot_dir $A --out $S/classes.json

# per-section prebuilts (mouse 1 also fixes the coordinate statistics every mouse prebuilt is normalised with)
$P prebuilt --animal 1 --raw_dir $RAW --chunk_max_cells 20000 --write_coord_stats_json $STATS --out_dir $B/base/prebuilt_mouse1
$P prebuilt --animal 2 --raw_dir $RAW --chunk_max_cells 40000 --coord_stats_json $STATS --out_dir $B/base/prebuilt_mouse2
$P prebuilt --animal 3 --raw_dir $RAW --annot_dir $A --chunk_max_cells 20000 --coord_stats_json $A/iso_stats_abca3.json --out_dir $B/prebuilt_test_mouse3

# training prebuilt (the trainer reads its infos.pkl and its number of graphs per epoch) and the test prebuilts of mice 1 / 2
$P mixed --src $B/base/prebuilt_mouse1 --out $B/base/prebuilt_train_mixed --cells 20000 --dense_stride 2 \
  --block_k 64 --block_stride 4 --block_reps 2 --block_k2 32 --block2_stride 4 --block2_reps 2 --volume_reps 36 --seed 1
$P iso --src $B/base/prebuilt_train_mixed --out $B/prebuilt_train --coord_stats_json $STATS
$P iso --src $B/base/prebuilt_mouse1 --out $B/prebuilt_test_mouse1 --coord_stats_json $STATS
$P iso --src $B/base/prebuilt_mouse2 --out $B/prebuilt_test_mouse2 --coord_stats_json $STATS

# training cell stores, each on the ruler of the training reference mouse
for n in 1 2 3 4; do
  $P store --animal $n --raw_dir $RAW --annot_dir $A --class_list $S/classes.json --out_dir $S/abca$n
done
$P rescale --store $S/abca3 --ref_store $S/abca1 --raw_dir $RAW
$P rescale --store $S/abca4 --ref_store $S/abca1 --raw_dir $RAW
$P rescale --store $S/abca2 --ref_store $S/abca1 --raw_dir $RAW --out $S/abca2_ref1
$P rereference --store $S/abca3 --ref_store $S/abca2 --raw_dir $RAW --out $S/abca3_ref2
$P rereference --store $S/abca4 --ref_store $S/abca2 --raw_dir $RAW --out $S/abca4_ref2

# test prebuilt row -> cell-store row maps read by the scorer (mouse 2: cell-label join; mouse 3: geometric match)
$P rowmap --animal 2 --match labels --raw_dir $RAW --prebuilt $B/prebuilt_test_mouse2 --store $S/abca2 --out $B/maps/mouse2_prebuilt_to_store_row.npy
$P rowmap --animal 3 --match geometry --prebuilt $B/prebuilt_test_mouse3 --store $S/abca3 --out $B/maps/mouse3_prebuilt_to_store_row.npy
