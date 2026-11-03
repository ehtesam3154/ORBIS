#!/usr/bin/env python
"""Independent check of the merged pool against the per-file extractions and the raw files.  usage: verify_pool.py [POOL_DIR] [n_per_file]"""
import sys, json, time
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad
import os
POOL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ["DATA_DIR"], "allen10x", "v3_pool")
NPF = int(sys.argv[2]) if len(sys.argv) > 2 else 24
RAW = os.path.join(os.environ["DATA_DIR"], "raw", "allen_10x")
BASE = os.path.join(os.environ["DATA_DIR"], "allen10x")
t0 = time.time()
P = sp.load_npz(f"{POOL}/counts_g1122.npz").tocsr(); L = np.load(f"{POOL}/cell_labels.npy"); idx477 = np.load(f"{POOL}/idx_g477.npy")
tall = np.load(f"{POOL}/total_umi_all.npy"); t477 = np.load(f"{POOL}/total_umi_g477.npy")
G = pd.read_csv(f"{POOL}/genes_g1122.csv"); M = pd.read_parquet(f"{POOL}/EVAL_ONLY_cell_metadata.parquet", columns=["cell_label", "src_file"])
assert (M.cell_label.values == L).all()
rng = np.random.default_rng(25000); res = dict(pool=POOL, seed=25000, per_file=NPF, cells_checked=0, mismatching_cells=0, details=[])
for fn, rows in M.groupby("src_file", observed=True).indices.items():
    pick = np.sort(rng.choice(rows, size=min(NPF, len(rows)), replace=False))
    A = ad.read_h5ad(f"{RAW}/expression/{fn}-raw.h5ad", backed="r")
    cols = A.var_names.get_indexer(G.ensembl_id); assert (cols >= 0).all()
    sub = A[L[pick].tolist(), :].X; sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
    for k, r in enumerate(pick):
        row = P[r].toarray().ravel(); ref = sub[k]
        ok = (ref[cols] == row).all() and (ref[cols][idx477] == row[idx477]).all() and int(ref.sum()) == int(tall[r]) and int(ref[cols][idx477].sum()) == int(t477[r])
        res["cells_checked"] += 1; res["mismatching_cells"] += int(not ok)
        if not ok: res["details"].append(dict(cell=str(L[r]), file=str(fn)))
    A.file.close()
    print(f"{fn}: {len(pick)} cells checked", flush=True)
res["wall_seconds"] = round(time.time() - t0, 1); res["PASS"] = res["mismatching_cells"] == 0
json.dump(res, open(f"{POOL}/verify_report.json", "w"), indent=1)
print(json.dumps({k: res[k] for k in ("cells_checked", "mismatching_cells", "PASS", "wall_seconds")}))
assert res["PASS"]
