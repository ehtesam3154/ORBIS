# Niche reader of the blockA test set(s) (scripts/build_reader.sh). THREADS and CHUNKED_STANDARDISER = the settings the published reader
# was fitted with. The k-means initialisation sums thread-partial results in arrival order, so a refit can differ in the last digits
# from the published reader; use the published reader file to reproduce the paper's reseated layouts exactly.
DATA_DIR=${DATA_DIR:?set DATA_DIR}
OUT_DIR=${OUT_DIR:?set OUT_DIR}
READER_NAME=blockA
STORES=("${DATA_DIR}/blocks/foldA/stores/STARmap_main")
MM_PER_UNIT=5.660694041161909
SECTIONS=0
PANEL=first
SPLIT=tiles
CHUNKED_STANDARDISER=0
TESTS=("dstarA=${DATA_DIR}/blocks/foldA/prebuilt_test_STARmap_replicate")
THREADS=8   # the published fit used 6 or 8 threads (not recorded)
