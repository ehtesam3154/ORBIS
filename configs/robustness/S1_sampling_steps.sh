# Inference ablation / robustness run: number of sampling steps K (set STEPS: 2, 5, 10, 25 or 100).
# Runs on the seed-0 final of MODEL (mouse1, mouse2, mouse3, allen, blockA, blockB or embryo; for embryo set EMBRYO=c|d); the final's
# checkpoint is reused, nothing is retrained. Used by scripts/sample.sh and scripts/score.sh.
MODEL=${MODEL:?set MODEL}
source "$(dirname "${BASH_SOURCE[0]}")/../finals/${MODEL}_s0.sh"
NAME=S1_sampling_steps_steps${STEPS}_${MODEL}
SEED=${SEED:-1000}
TAG=${NAME}_${TESTKEY}_seed${SEED}
SAMPLE_ARGS=(--test_dir "${DATA_DIR}/${SAMPLE_TEST_DIR}" --mm "$MM_PER_UNIT" --tag "$TAG" --calibrate 0 --global_state 1 --sample_seed "$SEED" --steps "${STEPS:?set STEPS}" --schedule edm)
# ---- scoring (scripts/score.sh) of this run's four draws; these runs are not reseated
unset SAMPLES GUESSES
LAYOUTS=""
for S in 1000 1001 1002 1003; do LAYOUTS="$LAYOUTS samples/${NAME}_${TESTKEY}_seed${S}${SFX}_steps${STEPS}.npz"; done
