# Niche reader of the embryos test set(s) (scripts/build_reader.sh). THREADS and CHUNKED_STANDARDISER = the settings the published reader
# was fitted with. The k-means initialisation sums thread-partial results in arrival order, so a refit can differ in the last digits
# from the published reader; use the published reader file to reproduce the paper's reseated layouts exactly.
DATA_DIR=${DATA_DIR:?set DATA_DIR}
OUT_DIR=${OUT_DIR:?set OUT_DIR}
READER_NAME=embryos
STORES=("${DATA_DIR}/embryos_panel/stores/16.32h_E12-14h_c" "${DATA_DIR}/embryos_panel/stores/16.53h_E16-18h_b" "${DATA_DIR}/embryos_panel/stores/16.57h_E16-18h_a" "${DATA_DIR}/embryos_panel/stores/16.63h_E16-18h_e" "${DATA_DIR}/embryos_panel/stores/17.61h_E14-16h_f" "${DATA_DIR}/embryos_panel/stores_more/13.79h_E10-12h_d" "${DATA_DIR}/embryos_panel/stores_more/15.75h_E12-14h_a" "${DATA_DIR}/embryos_panel/stores_more/15.77h_E12-14h_b" "${DATA_DIR}/embryos_panel/stores_more/18.39h_E2.5-3h_d" "${DATA_DIR}/embryos_panel/stores_more/18.97h_E2.5-3h_c" "${DATA_DIR}/embryos_panel/stores_more/19.84h_E18-20h_d")
UM_PER_UNIT=452.1785038908124
SECTIONS=0
PANEL=largest
SPLIT=specimen
CHUNKED_STANDARDISER=1
VAL_SPECIMEN=16.53h_E16-18h_b
INNER_SPECIMEN=16.63h_E16-18h_e
TESTS=("flysta2_c=${DATA_DIR}/embryos_panel/prebuilt_test_17.34h_E14-16h_c" "flysta2_d=${DATA_DIR}/embryos_panel/prebuilt_test_16.88h_E14-16h_d")
THREADS=4
