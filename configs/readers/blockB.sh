# Niche reader of the blockB test set(s) (scripts/build_reader.sh). THREADS and CHUNKED_STANDARDISER = the settings the published reader
# was fitted with. The k-means initialisation sums thread-partial results in arrival order, so a refit can differ in the last digits
# from the published reader; use the published reader file to reproduce the paper's reseated layouts exactly.
DATA_DIR=${DATA_DIR:?set DATA_DIR}
OUT_DIR=${OUT_DIR:?set OUT_DIR}
READER_NAME=blockB
STORES=("${DATA_DIR}/blocks/foldB/stores/STARmap_replicate")
MM_PER_UNIT=5.637126154749018
SECTIONS=0
PANEL=first
SPLIT=tiles
CHUNKED_STANDARDISER=1
TESTS=("dstarB=${DATA_DIR}/blocks/foldB/prebuilt_test_STARmap_main")
THREADS=6
