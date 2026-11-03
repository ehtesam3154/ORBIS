# Final model of the dissociated-cell application: all five MERFISH brains, 200 Harmony-integrated features; training seed 0.
# Training: scripts/train.sh. Application (mapping of the 10x cells, sampling, readouts): scripts/allen10x/run_application.sh with
# configs/allen10x/application.json.
DATA_DIR=${DATA_DIR:?set DATA_DIR}
OUT_DIR=${OUT_DIR:?set OUT_DIR}
NAME=allen10x
RUN_DIR=$OUT_DIR/models/$NAME
CKPT=${CKPT:-$RUN_DIR/orbis_epoch359.pt}
ARGS=(
  --prebuilt_dir "${DATA_DIR}/brains/allen_harmony/train_prebuilt"
  --views_stores "${DATA_DIR}/allen10x/stores/abca1,${DATA_DIR}/allen10x/stores/abca2_ref1,${DATA_DIR}/allen10x/stores/abca3,${DATA_DIR}/allen10x/stores/abca4,${DATA_DIR}/allen10x/stores/allen638850_hemi0"
  --out "${RUN_DIR}"
  --gene_columns_start 0
  --gene_columns_end 200
  --epochs 360
  --lr 0.0005
  --eps 0.001
  --t_mu 0.5
  --t_sigma 1.2
  --t_mix_p 0.5
  --t_mix_mu 2.32
  --seed 0
  --save_every 30
  --ema 0.999
  --ema_rampup 0.05
  --grad_clip 0.0
  --lr_decay_epochs 60
  --accum 2
  --global_batch 2
  --views_whole 0,1,2,4
  --views_mix 0.5,0.25,0.25
  --views_n 10000,20000
  --views_thin 1.0
  --views_workers 6
  --views_pin 1
  --views_fold_axis -1
  --views_per_epoch 0
  --view_label 1
  --loss_mse 1
  --plane_balanced 0.25
  --dit_hidden 256
  --dit_layers 8
  --dit_heads 8
  --xt_precond 0
  --edm_weight 0
  --data_rms 0.163
  --noise_scale 1.0
)
