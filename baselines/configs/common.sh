# Shared settings of the baseline configs (sourced by baselines/configs/<baseline>_<test>.sh).
# Inputs come from DATA_DIR (prepared cell stores and test prebuilts), outputs go to OUT_DIR.
DATA_DIR=${DATA_DIR:?set DATA_DIR}; OUT_DIR=${OUT_DIR:?set OUT_DIR}
BL=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STAGES=${STAGES:-"prep train test collect"}
has() { [[ " $STAGES " == *" $1 "* ]]; }
LAYOUTS=${LAYOUTS:-$OUT_DIR/layouts}; mkdir -p "$LAYOUTS"

# Official code at the pinned commits (fetched by `official`; LUNA and CeLEry get the patch from baselines/<name>/).
SRC=${SRC:-$OUT_DIR/official}
LUNA_SRC=${LUNA_SRC:-$SRC/LUNA}; REMAP_SRC=${REMAP_SRC:-$SRC/REMAP}; CELERY_SRC=${CELERY_SRC:-$SRC/CeLEry}
TANGRAM_SRC=${TANGRAM_SRC:-$SRC/Tangram}; NOVOSPARC_SRC=${NOVOSPARC_SRC:-$SRC/novosparc}
official() {   # official NAME URL COMMIT [PATCH]: clone into <NAME>_SRC (upper case) unless a checkout is already there
  local v; v="$(echo "$1" | tr '[:lower:]' '[:upper:]')_SRC"; local d=${!v}
  [ -d "$d/.git" ] && return 0
  mkdir -p "$(dirname "$d")"; git clone -q "$2" "$d" && git -C "$d" checkout -q "$3" || return 1
  [ -n "${4:-}" ] && git -C "$d" apply "$4"
  return 0
}

# Python interpreters (one environment per baseline; versions in baselines/<name>/README.md).
PY_LUNA=${PY_LUNA:-python}; PY_REMAP=${PY_REMAP:-python}; PY_CELERY=${PY_CELERY:-python}
PY_TANGRAM=${PY_TANGRAM:-python}; PY_NOVOSPARC=${PY_NOVOSPARC:-python}; PY_TOOLS=${PY_TOOLS:-python}
export NUMPY_MADVISE_HUGEPAGE=0

# Prepared data.
BRAINS=$DATA_DIR/brains                                   # Zhuang ABCA mice 1-4 and Allen 638850
STORES=$BRAINS/stores                                     # abca1 abca2 abca2_ref1 abca3 abca3_ref2 abca4 abca4_ref2 allen638850
PRE_MOUSE1=$BRAINS/prebuilt_test_mouse1; PRE_MOUSE2=$BRAINS/prebuilt_test_mouse2; PRE_MOUSE3=$BRAINS/prebuilt_test_mouse3
PRE_ALLEN=$BRAINS/prebuilt_test_allen638850
MAP_MOUSE2=$BRAINS/maps/mouse2_prebuilt_to_store_row.npy; MAP_MOUSE3=$BRAINS/maps/mouse3_prebuilt_to_store_row.npy
MAP_MOUSE1=${MAP_MOUSE1:-$OUT_DIR/maps/mouse1_prebuilt_to_store_row.npy}   # written by baselines/luna3d/make_row_map.py
HARM=$BRAINS/allen_harmony                                # saved Harmony input: stores/abca1-4, allen/feat.npy, allen/cell_ID.npy, pca.npz
EMB=$DATA_DIR/embryos_panel                                  # Flysta3D-v2: stores/, stores_more/, prebuilt_test_<embryo>, build_summary*.json
BLK=$DATA_DIR/blocks                                      # Deep-STARmap: foldA (train main block), foldB (train replicate block)
FLY_STORES="stores/16.32h_E12-14h_c stores/16.53h_E16-18h_b stores/16.57h_E16-18h_a stores/16.63h_E16-18h_e stores/17.61h_E14-16h_f
  stores_more/13.79h_E10-12h_d stores_more/15.75h_E12-14h_a stores_more/15.77h_E12-14h_b stores_more/18.39h_E2.5-3h_d
  stores_more/18.97h_E2.5-3h_c stores_more/19.84h_E18-20h_d"
FLY=$(for s in $FLY_STORES; do printf '%s,' "$EMB/$s"; done); FLY=${FLY%,}   # the 11 training embryos, in this order

# Units of the prepared data.
ISO=13.45773983001709                                     # brain stores: mm per unit
S1=1.0185775453536927,1.2931099417291327,0.9995762396395721   # per-axis calibration, mouse-1 ruler
S2=1.0293507964344841,1.244982133512552,1.0015486559068691    # per-axis calibration, mouse-2 ruler
MM_BLOCK_A=5.660694041161909; MM_BLOCK_B=5.637126154749018; MM_EMB=0.45217850389081243

# Test -> converter arguments (baselines/remap/to_layout.py) for the baselines whose output is in store / prebuilt row order.
layout_args() {
  case $1 in
    mouse1) echo "--truth_dir $PRE_MOUSE1 --row_map $MAP_MOUSE1 --store_pos $STORES/abca1/pos.npy --mm_per_unit $ISO";;
    mouse2) echo "--truth_dir $PRE_MOUSE2 --row_map $MAP_MOUSE2 --store_pos $STORES/abca2/pos.npy --mm_per_unit $ISO";;
    mouse3) echo "--truth_dir $PRE_MOUSE3 --row_map $MAP_MOUSE3 --store_pos $STORES/abca3/pos.npy --mm_per_unit $ISO";;
    allen) echo "--truth_dir $PRE_ALLEN --row_map $PRE_ALLEN/cell_ID.npy --store_pos $STORES/allen638850/pos.npy --mm_per_unit $ISO";;
    blockA) echo "--truth_dir $BLK/foldA/prebuilt_test_STARmap_replicate --mm_per_unit $MM_BLOCK_A";;
    blockB) echo "--truth_dir $BLK/foldB/prebuilt_test_STARmap_main --mm_per_unit $MM_BLOCK_B";;
    embryo_d) echo "--truth_dir $EMB/prebuilt_test_16.88h_E14-16h_d --mm_per_unit $MM_EMB";;
    embryo_c) echo "--truth_dir $EMB/prebuilt_test_17.34h_E14-16h_c --mm_per_unit $MM_EMB";;
  esac
}
mouse1_map() {   # the mouse-1 truth -> store row map (exact position match), written once
  [ -f "$MAP_MOUSE1" ] || { mkdir -p "$(dirname "$MAP_MOUSE1")"; $PY_TOOLS $BL/luna3d/make_row_map.py --prebuilt $PRE_MOUSE1 --store $STORES/abca1 --out $MAP_MOUSE1; }
}
