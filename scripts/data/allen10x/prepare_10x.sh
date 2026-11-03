#!/bin/bash
# Inputs of the dissociated-cell application:
#   1. download the 13 WMB-10Xv3 raw files of the ABC release (size and md5 checked against the manifest), extract the 1,122 panel genes,
#      merge one v3 pool, per-region pool counts, the official CCFv3 annotation volume;
#   2. Symphony reference on the five MERFISH training brains, mapping of the whole pool in one call;
#   3. the five training stores of the 10x model with the reference embedding as features.
# DATA_DIR (layout in scripts/allen10x/paths.py) holds raw/allen_10x/metadata (cell_metadata.csv, gene.csv, WMB-taxonomy/), brains/stores,
# brains/allen_harmony and allen10x/merfish_allen638850_g477;
# OUT_DIR receives the file list. Everything derived is written below DATA_DIR/allen10x; the fixed inputs of
# scripts/allen10x/inputs/ are copied to DATA_DIR/allen10x/frozen/.
set -euo pipefail
DATA_DIR=${DATA_DIR:?set DATA_DIR}
OUT_DIR=${OUT_DIR:?set OUT_DIR}
MANIFEST=${MANIFEST:?set MANIFEST (ABC release manifest, releases/20231215/manifest.json of the allen-brain-cell-atlas bucket)}
export DATA_DIR OUT_DIR NUMPY_MADVISE_HUGEPAGE=0
R=$DATA_DIR/raw/allen_10x
A=$DATA_DIR/allen10x
cd "$(dirname "$0")"
mkdir -p "$R/expression" "$A/state" "$A/frozen" "$OUT_DIR"

# fixed inputs shipped with the release, and the public tables of the Allen CCF
cp ../../allen10x/inputs/roi_ccf_crosswalk.csv ../../allen10x/inputs/zhuang_g1122_gene_overlap.csv \
   ../../allen10x/inputs/allen_g477_gene_overlap.csv ../../allen10x/inputs/allen_merfish_638850_measured.txt "$A/frozen/"
curl -sS -f -L -o "$A/frozen/ccf_structure_graph_1.json" https://api.brain-map.org/api/v2/structure_graph_download/1.json
[ "$(md5sum "$A/frozen/ccf_structure_graph_1.json" | cut -d' ' -f1)" = 2edb440f2c6afd463f6261712ba5bb6f ] || echo "note: structure graph differs from the one the paper used (md5 2edb440f...)"
mkdir -p "$DATA_DIR/raw/abc_mice"
[ -e "$DATA_DIR/raw/abc_mice/ccf2020_parcellation_to_parcellation_term_membership_acronym.csv" ] || curl -sS -f -L \
  -o "$DATA_DIR/raw/abc_mice/ccf2020_parcellation_to_parcellation_term_membership_acronym.csv" \
  https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/metadata/Allen-CCF-2020/20230630/views/parcellation_to_parcellation_term_membership_acronym.csv

S3=https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com
mkdir -p "$R/metadata/WMB-taxonomy/views"
for f in cell_metadata.csv gene.csv; do [ -e "$R/metadata/$f" ] || curl -sS -f -L -o "$R/metadata/$f" "$S3/metadata/WMB-10X/20231215/$f"; done
f=WMB-taxonomy/views/cluster_to_cluster_annotation_membership_pivoted.csv
[ -e "$R/metadata/$f" ] || curl -sS -f -L -o "$R/metadata/$f" "$S3/metadata/WMB-taxonomy/20231215/views/cluster_to_cluster_annotation_membership_pivoted.csv"

python make_file_list.py "$MANIFEST"
while IFS=$'\t' read -r name url size md5; do
  f=$R/expression/${name}-raw.h5ad
  [ -e "$f" ] || curl -sS -f -L --retry 20 -C - -o "$f.part" "$url" && { [ -e "$f" ] || mv "$f.part" "$f"; }
  [ "$(stat -c %s "$f")" -eq "$size" ] || { echo "size mismatch $name"; exit 1; }
  [ "$(md5sum "$f" | cut -d' ' -f1)" = "$md5" ] || { echo "md5 mismatch $name"; exit 1; }
  printf '%s\t%s\t%s\t%s\n' "$name" "$size" "$md5" "$(date '+%F %T')" > "$A/state/dl_$name.done"
done < "$OUT_DIR/v3_raw_file_list.tsv"
python prep_metadata.py
cut -f1 "$OUT_DIR/v3_raw_file_list.tsv" | while read -r name; do python extract_file.py "$name" 2000; done
python merge_pool.py
python verify_pool.py
python -c "
import os, pandas as pd
m = pd.read_csv(os.path.join(os.environ['DATA_DIR'], 'allen10x', 'v3_pool', 'EVAL_ONLY_cell_metadata.csv.gz'), usecols=['region_of_interest_acronym'])
m.region_of_interest_acronym.value_counts().rename_axis('region_of_interest_acronym').rename('n_cells').reset_index().to_csv(
    os.path.join(os.environ['DATA_DIR'], 'allen10x', 'frozen', 'pool_by_roi.csv'), index=False)"
curl -sS -f -L -o "$A/frozen/annotation_25.nrrd" \
  https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_25.nrrd
[ "$(md5sum "$A/frozen/annotation_25.nrrd" | cut -d' ' -f1)" = 5012b7fa0c0fd23dc455462e9b6298f4 ] || { echo "annotation volume md5 differs"; exit 1; }

python build_symphony_reference.py stage-a --out "$A/reference" --workers 12 \
  --stores_dir "$DATA_DIR/brains/stores" --allen_dir "$A/merfish_allen638850_g477" \
  --panel "$A/frozen/allen_merfish_638850_measured.txt" --sign_pca "$DATA_DIR/brains/allen_harmony/pca.npz"
python build_symphony_reference.py harmony --ref "$A/reference" --out "$A/reference" --seed 0 --path symphonypy
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python map_query.py --route sparse --save_f64 \
  --ref "$A/reference" --pool "$A/v3_pool" --out "$A/mapped_v3"

python build_stores.py --cellstore_root "$DATA_DIR/brains/stores" --template_root "$DATA_DIR/brains/allen_harmony" --out_root "$A/stores"
python fill_features.py --stores "$A/stores" --template_root "$DATA_DIR/brains/allen_harmony" --ref_dir "$A/reference"
