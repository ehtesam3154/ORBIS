# Inference ablation / robustness run: a random subset of the test cells (set PERCENT: 50, 25 or 10); first run scripts/ablations/make_subsets.py.
# Runs on the seed-0 final of MODEL (mouse1, mouse2, mouse3, allen, blockA, blockB or embryo; for embryo set EMBRYO=c|d); the final's
# checkpoint is reused, nothing is retrained. Used by scripts/sample.sh and scripts/score.sh.
MODEL=${MODEL:?set MODEL}
source "$(dirname "${BASH_SOURCE[0]}")/../finals/${MODEL}_s0.sh"
SUBSET_DIR=$OUT_DIR/subsets/${TESTKEY}_sub${PERCENT:?set PERCENT}
# made by: python scripts/ablations/make_subsets.py "$DATA_DIR/$SAMPLE_TEST_DIR" "$OUT_DIR/subsets" "$TESTKEY" (writes the 50, 25 and 10 % subsets)
# scoring: the subset draws, and the seed-0 final draws restricted to the same cells (scripts/ablations/restrict_layout.py), with
# KEEP = the kept rows of the full test set
KEEP=subsets/${TESTKEY}_sub${PERCENT}/keep_idx.npy
NAME=R3_fewer_cells_sub${PERCENT}_${MODEL}
SEED=${SEED:-1000}
TAG=${NAME}_${TESTKEY}_seed${SEED}
SAMPLE_ARGS=(--test_dir "$SUBSET_DIR" --mm "$MM_PER_UNIT" --tag "$TAG" --calibrate 0 --global_state 1 --sample_seed "$SEED" --steps 50 --schedule edm)
# ---- scoring (scripts/score.sh) of this run's four draws; these runs are not reseated
unset SAMPLES GUESSES
LAYOUTS=""
for S in 1000 1001 1002 1003; do LAYOUTS="$LAYOUTS samples/${NAME}_${TESTKEY}_seed${S}${SFX}_steps50.npz"; done
