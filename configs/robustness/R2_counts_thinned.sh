# Inference ablation / robustness run: test counts thinned to a share (set LEVEL: 0.5 or 0.25); first run scripts/ablations/perturb.py.
# Runs on the seed-0 final of MODEL (mouse1, mouse2, mouse3, allen, blockA, blockB or embryo; for embryo set EMBRYO=c|d); the final's
# checkpoint is reused, nothing is retrained. Used by scripts/sample.sh and scripts/score.sh.
MODEL=${MODEL:?set MODEL}
source "$(dirname "${BASH_SOURCE[0]}")/../finals/${MODEL}_s0.sh"
PERTURBED_DIR=$OUT_DIR/perturbed/${TESTKEY}_thin${LEVEL:?set LEVEL}
# made by: python scripts/ablations/perturb.py "$DATA_DIR/$SAMPLE_TEST_DIR" "$PERTURBED_DIR" thin "$LEVEL"
NAME=R2_counts_thinned_thin${LEVEL}_${MODEL}
SEED=${SEED:-1000}
TAG=${NAME}_${TESTKEY}_seed${SEED}
SAMPLE_ARGS=(--test_dir "$PERTURBED_DIR" --mm "$MM_PER_UNIT" --tag "$TAG" --calibrate 0 --global_state 1 --sample_seed "$SEED" --steps 50 --schedule edm)
# ---- scoring (scripts/score.sh) of this run's four draws; these runs are not reseated
unset SAMPLES GUESSES
LAYOUTS=""
for S in 1000 1001 1002 1003; do LAYOUTS="$LAYOUTS samples/${NAME}_${TESTKEY}_seed${S}${SFX}_steps50.npz"; done
