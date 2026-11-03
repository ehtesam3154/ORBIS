"""Directory layout of the dissociated-cell application (inputs below the environment variable DATA_DIR, outputs below OUT_DIR).

  DATA_DIR/raw/allen_10x/{expression,metadata}/      downloaded 10x v3 expression files, release metadata and taxonomy
  DATA_DIR/raw/abc_mice/                             Zhuang MERFISH raw h5ad, CCF coordinates, cell metadata, CCF parcellation table
  DATA_DIR/raw/allen_merfish/                        Allen MERFISH CCF coordinates and cell metadata
  DATA_DIR/brains/stores/, brains/allen_harmony/     MERFISH cell stores and the previous training-store tree (template, train prebuilt)
  DATA_DIR/allen10x/state/, allen10x/v3_pool/        extraction state and the merged 10x v3 pool
  DATA_DIR/allen10x/merfish_allen638850_g477/        Allen MERFISH cells with the 477 panel genes (Symphony reference input)
  DATA_DIR/allen10x/reference/, allen10x/mapped_v3/  Symphony reference and the mapped pool
  DATA_DIR/allen10x/stores/                          the five training stores of the 10x model
  DATA_DIR/allen10x/frozen/                          fixed tables (region crosswalk, structure graph, gene overlaps, unit tables, quotas,
                                                     controls, per-region pool counts, CCFv3 annotation volume)
  OUT_DIR/application/      support map, landmarks, cohorts, sampler outputs, layouts, evaluation
  OUT_DIR/ccf_validation/   official-annotation scorer outputs
  OUT_DIR/genemaps/         gene-map caches and fields
  OUT_DIR/results/          summary tables
"""
import os

DATA_DIR = os.environ.get("DATA_DIR", "")
OUT_DIR = os.environ.get("OUT_DIR", "")

POOL = os.path.join(DATA_DIR, "allen10x", "v3_pool")
TAXONOMY = os.path.join(DATA_DIR, "raw", "allen_10x", "metadata", "WMB-taxonomy", "views", "cluster_to_cluster_annotation_membership_pivoted.csv")
MAPPED = os.path.join(DATA_DIR, "allen10x", "mapped_v3")
STORES = os.path.join(DATA_DIR, "allen10x", "stores")
MERFISH_RAW = os.path.join(DATA_DIR, "raw", "abc_mice")
ALLEN_MERFISH_RAW = os.path.join(DATA_DIR, "raw", "allen_merfish")
FROZEN = os.path.join(DATA_DIR, "allen10x", "frozen")

APP = os.environ.get("APP_DIR", os.path.join(OUT_DIR, "application"))
SUPPORT = os.path.join(APP, "support")
LANDMARKS = os.path.join(APP, "landmarks", "landmarks_k64.npz")
CCF_VALIDATION = os.path.join(OUT_DIR, "ccf_validation")
GENEMAPS = os.path.join(OUT_DIR, "genemaps")
RESULTS = os.path.join(OUT_DIR, "results")
