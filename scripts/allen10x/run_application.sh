#!/bin/bash
# Dissociated-cell application: frozen inputs, cohorts, sampling, label-free pose, evaluation, summaries, official-annotation scorer, gene maps.
# Inputs below DATA_DIR (see paths.py), outputs below OUT_DIR. Run from this directory. CPU steps; the sampling step needs one GPU.
set -euo pipefail
DATA_DIR=${DATA_DIR:?set DATA_DIR}
OUT_DIR=${OUT_DIR:?set OUT_DIR}
export DATA_DIR OUT_DIR NUMPY_MADVISE_HUGEPAGE=0
ORBIS_ROOT=${ORBIS_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
CKPT=${CKPT:-$DATA_DIR/checkpoints/allen10x/orbis_epoch359.pt}
APP=${APP_DIR:-$OUT_DIR/application}
STEP=${1:-all}

frozen() {
  python units.py
  python support.py
  python quotas.py
  python landmarks.py "$APP/landmarks"
  for c in chance a b c; do python controls.py $c --out "$DATA_DIR/allen10x/frozen"; done
}

cohorts() {
  python application.py pool_unit
  python application.py cohorts --kind primary
  python application.py cohorts --kind strat
  python application.py overlap
}

sample() {
  mkdir -p "$APP/raw"
  for kind in primary strat; do for c in 1 2 3 4 5; do for s in 1000 1001 1002; do
    d=$APP/cohorts/$kind$c
    (cd "$ORBIS_ROOT" && python -m scripts.allen10x.sample_cohort --ckpt "$CKPT" --features "$d/features.npy" --ids "$d/ids.npy" \
       --feature_names_json "$d/names.json" --train_meta "$DATA_DIR/allen10x/stores/abca1/meta.json" \
       --sample_seed $s --steps 50 --schedule edm --group 20000 --gpu "${GPU:-0}" --out "$APP/raw/ours_${kind}${c}_s${s}.npz")
  done; done; done
}

evaluate() {
  for kind in primary strat; do
    python application.py layout --kind $kind
    python application.py freeze --kind $kind
    python application.py evaluate --kind $kind --freeze_md5 "$(md5sum "$APP/freeze_manifest_$kind.json" | cut -d' ' -f1)"
    python summarize.py --kind $kind
  done
  python contrasts.py
  python region_table.py
}

ccf() {
  python ccf_support.py
  python ccf_annotation.py check
  python ccf_translate.py validate
  python ccf_translate.py layouts
  python ccf_translate.py manifest
  for kind in primary strat; do for step in layouts refs summary; do python ccf_membership.py $step --kind $kind; done; done
}

genemaps() {
  python genemaps_genes.py
  python genemaps_fields.py
  python genemaps_score.py --workers 24
}

cd "$(dirname "$0")"
mkdir -p "$OUT_DIR/results" "$OUT_DIR/ccf_validation" "$OUT_DIR/genemaps"
case $STEP in
  all) frozen; cohorts; sample; evaluate; ccf; genemaps ;;
  frozen|cohorts|sample|evaluate|ccf|genemaps) $STEP ;;
  *) echo "usage: $0 [all|frozen|cohorts|sample|evaluate|ccf|genemaps]"; exit 2 ;;
esac
