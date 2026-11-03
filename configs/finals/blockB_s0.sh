# Final model: Deep-STARmap, test = block B; training seed 0.
# Used by scripts/train.sh, scripts/sample.sh, scripts/reseat.sh and scripts/score.sh (see README.md).
DATA_DIR=${DATA_DIR:?set DATA_DIR}
OUT_DIR=${OUT_DIR:?set OUT_DIR}
NAME=blockB_s0
RUN_DIR=$OUT_DIR/models/$NAME
CKPT=${CKPT:-$RUN_DIR/orbis_epoch359.pt}

# ---- training (scripts/train.sh)
ARGS=(
  --prebuilt_dir "${DATA_DIR}/blocks/foldB/prebuilt_infos_train"
  --views_stores "${DATA_DIR}/blocks/foldB/stores/STARmap_replicate"
  --out "${RUN_DIR}"
  --gene_columns_start 0
  --gene_columns_end 1017
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
  --grad_clip 0.05
  --lr_decay_epochs 60
  --qk_norm 1
  --log_attn 1
  --log_gnorm 0
  --accum 2
  --global_batch 2
  --views_whole 0
  --views_mix 1,0,0
  --views_n 10000,20000
  --views_thin 0.55
  --views_workers 6
  --views_pin 1
  --views_fold_axis -1
  --views_per_epoch 246
  --view_label 1
  --loss_mse 1
  --plane_balanced 0.25
  --dit_hidden 256
  --dit_layers 8
  --dit_heads 8
  --xt_precond 1
  --edm_weight 1
  --data_rms 0.163
  --noise_scale 1.0
)

# ---- test set
TESTKEY=blockB
case $TESTKEY in
  blockB)
    TEST=dstarB; SAMPLE_TEST_DIR=blocks/foldB/prebuilt_test_STARmap_main; TEST_DIR=blocks/foldB/prebuilt_test_STARmap_main; MM_PER_UNIT=5.637126154749018
    READER=${READER:-$OUT_DIR/readers/reader_blockB_dstarB.npz}; READER_MD5=${READER_MD5:-}   # md5 of the published reader: aaf41248a3cbc1d20172cc38c247dede
    GUESS_SCHEDULE="--schedule edm" ;;
esac

# ---- sampling (scripts/sample.sh): MODE=draw (50 steps) or MODE=guess (one call, the reseat anchor); SEED = sampler seed (1000-1003)
SEED=${SEED:-1000}
MODE=${MODE:-draw}
TAG=${NAME}_${TESTKEY}_seed${SEED}
if [ "$MODE" = guess ]; then
  TAG=${NAME}_${TESTKEY}_guess_seed${SEED}
  SAMPLE_ARGS=(--test_dir "${DATA_DIR}/${SAMPLE_TEST_DIR}" --mm "$MM_PER_UNIT" --tag "$TAG" --calibrate 0 --global_state 1
               --steps 1 --global_k1 1 $GUESS_SCHEDULE --sample_seed "$SEED")
else
  SAMPLE_ARGS=(--test_dir "${DATA_DIR}/${SAMPLE_TEST_DIR}" --mm "$MM_PER_UNIT" --tag "$TAG" --calibrate 0 --global_state 1
               --steps 50 --schedule edm --sample_seed "$SEED")
fi
# ---- reseating (scripts/reseat.sh): the four draws and their seed-matched one-call guesses
SFX=""
SAMPLES=(); GUESSES=()
for S in 1000 1001 1002 1003; do
  SAMPLES+=("$OUT_DIR/samples/${NAME}_${TESTKEY}_seed${S}${SFX}_steps50.npz")
  GUESSES+=("$OUT_DIR/samples/${NAME}_${TESTKEY}_guess_seed${S}${SFX}_steps1.npz")
done
# ---- scoring (scripts/score.sh): raw and reseated draws (paths relative to OUT_DIR)
LAYOUTS=""
for S in 1000 1001 1002 1003; do
  LAYOUTS="$LAYOUTS samples/${NAME}_${TESTKEY}_seed${S}${SFX}_steps50.npz reseated/${NAME}_${TESTKEY}_seed${S}${SFX}_steps50.npz"
done
