# Niche reader of the allen test set(s) (scripts/build_reader.sh). THREADS and CHUNKED_STANDARDISER = the settings the published reader
# was fitted with. The k-means initialisation sums thread-partial results in arrival order, so a refit can differ in the last digits
# from the published reader; use the published reader file to reproduce the paper's reseated layouts exactly.
DATA_DIR=${DATA_DIR:?set DATA_DIR}
OUT_DIR=${OUT_DIR:?set OUT_DIR}
READER_NAME=allen
STORES=("${DATA_DIR}/brains/allen_harmony/stores/abca1" "${DATA_DIR}/brains/allen_harmony/stores/abca2" "${DATA_DIR}/brains/allen_harmony/stores/abca3" "${DATA_DIR}/brains/allen_harmony/stores/abca4")
POS_DIRS=("${DATA_DIR}/brains/stores/abca1" "${DATA_DIR}/brains/stores/abca2_ref1" "${DATA_DIR}/brains/stores/abca3" "${DATA_DIR}/brains/stores/abca4")
MM_PER_UNIT=13.45773983001709
SECTIONS=1
PANEL=none
SPLIT=sections
CHUNKED_STANDARDISER=1
TESTS=("allen638850=${DATA_DIR}/brains/allen_harmony/allen")
THREADS=4
