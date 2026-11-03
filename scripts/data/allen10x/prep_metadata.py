#!/usr/bin/env python
"""Cache the 10x v3 rows of the release cell_metadata.csv (the columns the extraction needs) as DATA_DIR/allen_10x/state/v3_meta.parquet."""
import numpy as np, pandas as pd, json, hashlib, time
import os
M = os.path.join(os.environ["DATA_DIR"], "raw", "allen_10x", "metadata", "cell_metadata.csv")
OUT = os.path.join(os.environ["DATA_DIR"], "allen10x", "state", "v3_meta.parquet")
cols = ["cell_label", "barcoded_cell_sample_label", "library_label", "feature_matrix_label", "library_method", "region_of_interest_acronym",
        "donor_label", "donor_genotype", "donor_sex", "cluster_alias"]
t0 = time.time(); parts = []; n_all = 0; lm_counts = {}; fm_v3_nonv3method = 0; method_v3_nonv3fm = 0
for ch in pd.read_csv(M, usecols=cols, dtype=str, chunksize=500_000, keep_default_na=False):
    n_all += len(ch)
    for k, v in ch.library_method.value_counts().items():
        lm_counts[k] = lm_counts.get(k, 0) + int(v)
    is_m = ch.library_method == "10Xv3"; is_f = ch.feature_matrix_label.str.startswith("WMB-10Xv3-")
    method_v3_nonv3fm += int((is_m & ~is_f).sum()); fm_v3_nonv3method += int((~is_m & is_f).sum())
    parts.append(ch[is_m].copy())
v3 = pd.concat(parts, ignore_index=True); del parts
print(f"metadata rows {n_all:,}  library_method counts {lm_counts}  v3 rows {len(v3):,}  [library_method==10Xv3 but feature_matrix not v3: {method_v3_nonv3fm}; feature_matrix v3 but method not 10Xv3: {fm_v3_nonv3method}]  ({time.time() - t0:.0f} s)")
assert v3.cell_label.is_unique, "duplicate cell_label among v3 rows"
assert method_v3_nonv3fm == 0 and fm_v3_nonv3method == 0
assert (v3.cluster_alias != "").all()
v3["cluster_alias"] = v3.cluster_alias.astype(np.int32)
for c in ("barcoded_cell_sample_label", "library_label", "feature_matrix_label", "library_method", "region_of_interest_acronym", "donor_label", "donor_genotype", "donor_sex"):
    v3[c] = v3[c].astype("category")
v3.to_parquet(OUT, index=False)
print("per feature_matrix_label:", v3.feature_matrix_label.value_counts().sort_index().to_dict())
print("n donors", v3.donor_label.nunique(), "n libraries", v3.library_label.nunique(), "n ROI", v3.region_of_interest_acronym.nunique(), "n clusters", v3.cluster_alias.nunique())
print("wrote", OUT, "md5 of source csv", hashlib.md5(open(M, "rb").read()).hexdigest())
