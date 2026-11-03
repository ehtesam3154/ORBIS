#!/bin/bash
# Sample every test part with LUNA's own test_only mode and a trained checkpoint (luna_test_nometrics.py: LUNA's per-graph metric
# step replaced by a stub; predictions unchanged). DATA_ROOT/fold<FOLD>/ holds train_infos.csv (read only for the feature count)
# and test_part<PART>.csv; the run is RUN_ROOT/luna3d_fold<FOLD>. Writes test_part<PART>/**/metadata_{true,pred}.csv.
# usage: LUNA_SRC=... [PYTHON=python CKPT=epoch=299.ckpt NGENES=1122] bash run_luna3d_test.sh FOLD GPU DATA_ROOT RUN_ROOT PART [PART ...]
set -e
F=$1; GPU=$2; D=$3/fold$F; RUN=$4/luna3d_fold$F; shift 4; CK=${CKPT:-epoch=299.ckpt}; NGN=${NGENES:-1122}
export LUNA_SRC=${LUNA_SRC:?set LUNA_SRC}; export PYTHONPATH=$LUNA_SRC${PYTHONPATH:+:$PYTHONPATH}; V=${PYTHON:-python}
HERE=$(cd "$(dirname "$0")" && pwd)
export NUMPY_MADVISE_HUGEPAGE=0
cd $LUNA_SRC
for P in "$@"; do
  echo "PART $P START $(date) GPU $GPU checkpoint $CK feature columns $NGN"
  CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $V $HERE/luna_test_nometrics.py general.name=luna3d_fold$F general.mode=test_only \
    general.wandb=disabled general.enable_progress_bar=False dataset.gene_columns_start=0 dataset.gene_columns_end=$NGN \
    dataset.train_data_path=$D/train_infos.csv dataset.test_data_path=$D/test_part$P.csv \
    test.checkpoints_parent_dir=$RUN/checkpoints "test.checkpoints_name_list=[\"$CK\"]" test.save_dir=$RUN/test_part$P \
    hydra.run.dir=$RUN/test_part$P/hydra
  echo "PART $P DONE $(date)"
done
