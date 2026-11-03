# Niche reader of the mouse1 test set(s) (scripts/build_reader.sh). THREADS and CHUNKED_STANDARDISER = the settings the published reader
# was fitted with. The k-means initialisation sums thread-partial results in arrival order, so a refit can differ in the last digits
# from the published reader; use the published reader file to reproduce the paper's reseated layouts exactly.
DATA_DIR=${DATA_DIR:?set DATA_DIR}
OUT_DIR=${OUT_DIR:?set OUT_DIR}
READER_NAME=mouse1
STORES=("${DATA_DIR}/brains/stores/abca2" "${DATA_DIR}/brains/stores/abca3_ref2" "${DATA_DIR}/brains/stores/abca4_ref2")
MM_PER_UNIT=13.45773983001709
SECTIONS=1
PANEL=first
SPLIT=sections
CHUNKED_STANDARDISER=0
TESTS=("abca1=${DATA_DIR}/brains/prebuilt_test_mouse1")
THREADS=8
