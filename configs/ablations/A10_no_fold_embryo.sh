# Training ablation: no left-right folding of the training views (no unfold after sampling). Base: the seed-0 final of Flysta3D embryos, tests = embryos c and d; changed flags: --views_fold_axis -1.
# Used by scripts/train.sh, scripts/sample.sh, scripts/reseat.sh and scripts/score.sh (see README.md).
DATA_DIR=${DATA_DIR:?set DATA_DIR}
OUT_DIR=${OUT_DIR:?set OUT_DIR}
NAME=A10_no_fold_embryo
RUN_DIR=$OUT_DIR/models/$NAME
CKPT=${CKPT:-$RUN_DIR/orbis_epoch299.pt}

# ---- training (scripts/train.sh)
ARGS=(
  --prebuilt_dir "${DATA_DIR}/embryos/prebuilt_infos_train"
  --views_stores "${DATA_DIR}/embryos/stores/16.32h_E12-14h_c,${DATA_DIR}/embryos/stores/16.53h_E16-18h_b,${DATA_DIR}/embryos/stores/16.57h_E16-18h_a,${DATA_DIR}/embryos/stores/16.63h_E16-18h_e,${DATA_DIR}/embryos/stores/17.61h_E14-16h_f,${DATA_DIR}/embryos/stores_more/13.79h_E10-12h_d,${DATA_DIR}/embryos/stores_more/15.75h_E12-14h_a,${DATA_DIR}/embryos/stores_more/15.77h_E12-14h_b,${DATA_DIR}/embryos/stores_more/18.39h_E2.5-3h_d,${DATA_DIR}/embryos/stores_more/18.97h_E2.5-3h_c,${DATA_DIR}/embryos/stores_more/19.84h_E18-20h_d"
  --out "${RUN_DIR}"
  --gene_columns_start 0
  --gene_columns_end 6936
  --epochs 300
  --lr 0.0005
  --eps 0.001
  --t_mu 0.5
  --t_sigma 1.2
  --t_mix_p 0.5
  --t_mix_mu 2.32
  --seed 0
  --save_every 50
  --ema 0.999
  --ema_rampup 0.05
  --grad_clip 0.05
  --lr_decay_epochs 50
  --qk_norm 1
  --log_attn 1
  --log_gnorm 0
  --accum 2
  --global_batch 0
  --views_whole 0,1,2,3,4,5,6,7,8,9,10
  --views_mix 0.5,0.25,0.25
  --views_n 10000,20000
  --views_thin 0.55
  --views_workers 5
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
EMBRYO=${EMBRYO:-c}
case $EMBRYO in c|d) ;; *) echo "EMBRYO must be c or d"; exit 1 ;; esac
TESTKEY=embryo_$EMBRYO
case $TESTKEY in
  embryo_c)
    TEST=flysta2_c; SAMPLE_TEST_DIR=embryos/prebuilt_test_17.34h_E14-16h_c; TEST_DIR=embryos_panel/prebuilt_test_17.34h_E14-16h_c; MM_PER_UNIT=0.45217850389081243
    READER=${READER:-$OUT_DIR/readers/reader_embryos_flysta2_c.npz}; READER_MD5=${READER_MD5:-}   # md5 of the published reader: 90aa7adace7398d07090559253af6408
    GUESS_SCHEDULE="--schedule edm" ;;
  embryo_d)
    TEST=flysta2_d; SAMPLE_TEST_DIR=embryos/prebuilt_test_16.88h_E14-16h_d; TEST_DIR=embryos_panel/prebuilt_test_16.88h_E14-16h_d; MM_PER_UNIT=0.45217850389081243
    READER=${READER:-$OUT_DIR/readers/reader_embryos_flysta2_d.npz}; READER_MD5=${READER_MD5:-}   # md5 of the published reader: e0b66952690b8a5cbfd66a34435d4543
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
