#!/bin/bash
# Train LUNA-3D with LUNA's own main.py (patched clone in LUNA_SRC) on a binary training table (memory-mapped features).
# GLOBAL_BATCH is split over the listed GPUs (LUNA's train.batch_size is per rank); LR, model and loss as released; seed 0.
# The run trains only (test.checkpoints_name_list=[]); TEST_TINY is one test graph that the data module loads.
# usage: LUNA_SRC=... [PYTHON=python N_EPOCHS=300 SAVE_EVERY=100 GLOBAL_BATCH=8 WORKERS=4 NGENES=1122] \
#        bash run_luna3d_train_ddp.sh NAME GPUS BIN_DIR TEST_TINY RUN_ROOT
set -e
NAME=$1; GPUS=$2; BIN=$3; TT=$4; RUN=$5/$NAME; NE=${N_EPOCHS:-300}; SE=${SAVE_EVERY:-100}; GB=${GLOBAL_BATCH:-8}; NW=${WORKERS:-4}; NGN=${NGENES:-1122}
LUNA_SRC=${LUNA_SRC:?set LUNA_SRC}; V=${PYTHON:-python}
NG=$(echo $GPUS | tr ',' '\n' | wc -l); BS=$((GB / NG))
[ $((NE % SE)) -eq 0 ] || { echo "N_EPOCHS must be a multiple of SAVE_EVERY"; exit 1; }
[ $((BS * NG)) -eq $GB ] || { echo "GLOBAL_BATCH $GB not divisible by $NG GPUs"; exit 1; }
export NUMPY_MADVISE_HUGEPAGE=0
cd $LUNA_SRC
echo "START $(date) GPUs $GPUS run $RUN epochs $NE save every $SE | global batch $GB = $NG GPU(s) x $BS per GPU, workers $NW per rank, table $BIN, feature columns $NGN"
CUDA_VISIBLE_DEVICES=$GPUS OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $V main.py general.name=$NAME general.wandb=disabled \
  general.enable_progress_bar=False dataset.gene_columns_start=0 dataset.gene_columns_end=$NGN \
  dataset.train_data_path=$BIN/meta.json +dataset.train_memmap_dir=$BIN dataset.test_data_path=$TT \
  distribute.gpus_per_node=$NG train.batch_size=$BS +train.num_workers=$NW \
  train.n_epochs=$NE validation.save_model_every_n_epochs=$SE \
  'test.checkpoints_name_list=[]' test.save_dir=$RUN/test_tiny hydra.run.dir=$RUN
echo "TRAIN_DONE $(date)"
