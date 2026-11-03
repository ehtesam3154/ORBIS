#!/usr/bin/env python
"""Stream one WMB-10Xv3 raw h5ad and extract the 1,122 Zhuang-panel genes as a sparse integer CSR matrix in the fixed gene order
(exact Ensembl ids), with per-cell library totals over all genes, the 1,122 and the 477 panel genes.  Keeps the cells that are v3
rows of the release metadata.  usage: extract_file.py WMB-10Xv3-<NAME> [chunk_rows]  ->  DATA_DIR/allen_10x/v3_pool/per_file/<NAME>/"""
import sys, os, json, time, resource, csv
import numpy as np, pandas as pd, h5py, scipy.sparse as sp

NAME = sys.argv[1]
CHUNK = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
RAW = os.path.join(os.environ["DATA_DIR"], "raw", "allen_10x")
BASE = os.path.join(os.environ["DATA_DIR"], "allen10x")
DOCS = os.path.join(os.environ["DATA_DIR"], "allen10x", "frozen")
H5 = f"{RAW}/expression/{NAME}-raw.h5ad"
OUT = f"{BASE}/v3_pool/per_file/{NAME}"
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
rep = dict(name=NAME, h5ad=H5, chunk_rows=CHUNK, h5ad_bytes=os.path.getsize(H5))
dl = f"{BASE}/state/dl_{NAME}.done"
assert os.path.exists(dl), f"{NAME}: no md5-verified download marker"
rep["download_marker"] = open(dl).read().strip()


def log(*a):
    print(f"[{time.time() - t0:7.1f}s rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:5.2f} GB]", *a, flush=True)


gene_csv = pd.read_csv(f"{RAW}/metadata/gene.csv")
gid = gene_csv.gene_identifier.values.astype(str)
assert len(gid) == 32285 and len(set(gid)) == 32285
G1122 = pd.read_csv(f"{DOCS}/zhuang_g1122_gene_overlap.csv"); G477 = pd.read_csv(f"{DOCS}/allen_g477_gene_overlap.csv")
assert len(G1122) == 1122 and len(G477) == 477 and (G1122.panel_index.values == np.arange(1122)).all() and (G477.panel_index.values == np.arange(477)).all()
assert (gid[G1122["10x_gene_index"].values] == G1122.ensembl_id.values).all(), "1122 csv 10x_gene_index does not point at its Ensembl ID in gene.csv"
assert (gid[G477["10x_gene_index"].values] == G477.ensembl_id.values).all(), "477 csv 10x_gene_index does not point at its Ensembl ID in gene.csv"
assert G1122.ensembl_id.is_unique and G477.ensembl_id.is_unique and set(G477.ensembl_id) <= set(G1122.ensembl_id)
colmap = np.full(32285, -1, dtype=np.int16); colmap[G1122["10x_gene_index"].values] = np.arange(1122, dtype=np.int16)
in477 = np.zeros(32285, dtype=bool); in477[G477["10x_gene_index"].values] = True
assert (colmap >= 0).sum() == 1122 and in477.sum() == 477 and (colmap[in477] >= 0).all()

v3 = pd.read_parquet(f"{BASE}/state/v3_meta.parquet", columns=["cell_label", "feature_matrix_label", "library_label"])
meta_f = v3[v3.feature_matrix_label.astype(str) == NAME]
rep["metadata_cells_this_file"] = int(len(meta_f))
all_v3_labels = pd.Index(v3.cell_label.values)
meta_labels = pd.Index(meta_f.cell_label.values)
log(f"metadata: {len(v3):,} v3 rows, {len(meta_f):,} with feature_matrix_label {NAME}")

with h5py.File(H5, "r") as f:
    X = f["X"]
    assert X.attrs["encoding-type"] == "csr_matrix", X.attrs["encoding-type"]
    n_rows, n_cols = [int(s) for s in X.attrs["shape"]]
    assert n_cols == 32285
    rep.update(n_obs=n_rows, n_vars=n_cols, x_encoding="csr_matrix", data_dtype=str(X["data"].dtype), indices_dtype=str(X["indices"].dtype),
               indptr_dtype=str(X["indptr"].dtype), data_chunks=X["data"].chunks, data_compression=X["data"].compression, nnz_file=int(X["data"].shape[0]))
    assert f["obs"].attrs["_index"] == "cell_label" and f["var"].attrs["_index"] == "gene_identifier"
    obs = np.array([x.decode() for x in f["obs/cell_label"][:]])
    var = np.array([x.decode() for x in f["var/gene_identifier"][:]])
    assert len(obs) == n_rows and len(set(obs)) == n_rows
    assert len(var) == 32285 and (var == gid).all(), "h5ad var order differs from metadata gene.csv order"
    rep["var_order_equals_gene_csv"] = True
    rep["genes_found_g1122"] = int(np.isin(G1122.ensembl_id.values, var).sum()); rep["genes_found_g477"] = int(np.isin(G477.ensembl_id.values, var).sum())
    assert rep["genes_found_g1122"] == 1122 and rep["genes_found_g477"] == 477
    indptr = X["indptr"][:].astype(np.int64)
    assert indptr[0] == 0 and indptr[-1] == X["data"].shape[0] and (np.diff(indptr) >= 0).all()
    rep["row_nnz_min_mean_max"] = [int(np.diff(indptr).min()), float(np.diff(indptr).mean()), int(np.diff(indptr).max())]

    obs_ix = pd.Index(obs)
    keep = obs_ix.isin(meta_labels)
    in_other = obs_ix.isin(all_v3_labels) & ~keep
    rep["cells_in_file"] = int(n_rows); rep["cells_kept"] = int(keep.sum())
    rep["file_not_in_metadata"] = int((~keep).sum())
    rep["file_cells_that_are_metadata_cells_of_another_v3_file"] = int(in_other.sum())
    rep["metadata_not_in_file"] = int((~meta_labels.isin(obs_ix)).sum())
    log(f"file {n_rows:,} cells; kept {rep['cells_kept']:,}; file-not-in-metadata {rep['file_not_in_metadata']:,} (of which in another v3 file: {rep['file_cells_that_are_metadata_cells_of_another_v3_file']}); metadata-not-in-file {rep['metadata_not_in_file']:,}")

    dX, iX = X["data"], X["indices"]
    parts = []; tot_all = np.zeros(n_rows, np.int64); tot1122 = np.zeros(n_rows, np.int64); tot477 = np.zeros(n_rows, np.int64)
    n_nonint = n_neg = n_nonfinite = n_explicit0 = 0; dmax = 0.0; dmin = np.inf; panel_nnz_all = 0; idx_unsorted_rows = 0
    for a in range(0, n_rows, CHUNK):
        b = min(a + CHUNK, n_rows); nr = b - a
        lo, hi = int(indptr[a]), int(indptr[b])
        d = dX[lo:hi]; ix = iX[lo:hi]; rowlen = np.diff(indptr[a:b + 1])
        if hi > lo:
            fin = np.isfinite(d); n_nonfinite += int((~fin).sum())
            n_nonint += int((np.rint(d) != d).sum()); n_neg += int((d < 0).sum()); n_explicit0 += int((d == 0).sum())
            dmax = max(dmax, float(d.max())); dmin = min(dmin, float(d.min()))
            assert ix.min() >= 0 and ix.max() < 32285
        rowid = np.repeat(np.arange(nr, dtype=np.int32), rowlen)
        tot_all[a:b] = np.rint(np.bincount(rowid, weights=d, minlength=nr)).astype(np.int64)
        m = colmap[ix]
        sel = (m >= 0) & (d != 0)
        sel477 = in477[ix]
        rs = rowid[sel]; ds = d[sel]; ms = m[sel]
        tot1122[a:b] = np.rint(np.bincount(rs, weights=ds, minlength=nr)).astype(np.int64)
        tot477[a:b] = np.rint(np.bincount(rowid[sel477], weights=d[sel477], minlength=nr)).astype(np.int64)
        cnt = np.bincount(rs, minlength=nr)
        ptr = np.zeros(nr + 1, np.int64); np.cumsum(cnt, out=ptr[1:])
        csr = sp.csr_matrix((np.rint(ds).astype(np.int32), ms.astype(np.int32), ptr), shape=(nr, 1122))
        csr.sort_indices()
        assert csr.has_canonical_format, "duplicate (cell, gene) entries in the file"
        sel_rows = np.flatnonzero(keep[a:b])
        parts.append(csr if len(sel_rows) == nr else csr[sel_rows])
        panel_nnz_all += csr.nnz
        if (a // CHUNK) % 5 == 0 or b == n_rows:
            log(f"rows {b:,}/{n_rows:,}")
    rep.update(integer_check=dict(values_checked=int(indptr[-1]), n_non_integer=n_nonint, n_negative=n_neg, n_non_finite=n_nonfinite, n_explicit_zero=n_explicit0, min=dmin, max=dmax))
    assert n_nonint == 0 and n_neg == 0 and n_nonfinite == 0, f"{NAME}: X is not raw integer counts {rep['integer_check']}"
    log(f"integer check over all {int(indptr[-1]):,} stored values: non-integer {n_nonint}, negative {n_neg}, non-finite {n_nonfinite}, explicit zeros {n_explicit0}, min {dmin}, max {dmax}")

    Xf = sp.vstack(parts, format="csr"); del parts
    Xf.sort_indices(); assert Xf.has_canonical_format and Xf.dtype == np.int32 and Xf.shape == (rep["cells_kept"], 1122)
    kept_idx = np.flatnonzero(keep)
    labels = obs[kept_idx]
    t_all, t_1122, t_477 = tot_all[kept_idx], tot1122[kept_idx], tot477[kept_idx]
    assert (np.asarray(Xf.sum(1)).ravel() == t_1122).all(), "panel-1122 total != row sum of the extracted CSR"
    assert (t_477 <= t_1122).all() and (t_1122 <= t_all).all()

    rng = np.random.default_rng(25000); chk = rng.choice(len(kept_idx), size=min(200, len(kept_idx)), replace=False)
    g_pos = G1122["10x_gene_index"].values
    for j in chk:
        r = int(kept_idx[j]); sl = slice(int(indptr[r]), int(indptr[r + 1]))
        dense = np.zeros(32285, np.float64); dense[iX[sl]] = dX[sl]
        assert (Xf[j].toarray().ravel() == dense[g_pos]).all() and int(dense.sum()) == t_all[j] and int(dense[G477["10x_gene_index"].values].sum()) == t_477[j]
    rep["rowwise_spotcheck_cells"] = int(len(chk))
    log(f"row-wise spot check passed on {len(chk)} random kept cells")

dropped = np.flatnonzero(~keep)
with open(f"{OUT}/file_not_in_metadata.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["cell_label", "total_umi_all"])
    for r in dropped:
        w.writerow([obs[r], int(tot_all[r])])

rep.update(
    panel_nnz_kept=int(Xf.nnz), panel_nnz_per_cell_mean=float(Xf.nnz / max(1, Xf.shape[0])), panel_umi_fraction_of_all_median=float(np.median(t_1122 / np.maximum(t_all, 1))),
    zero_cells_g477=int((t_477 == 0).sum()), zero_cells_g1122=int((t_1122 == 0).sum()), zero_cells_all_genes=int((t_all == 0).sum()),
    g477_total_percentiles={str(q): float(np.percentile(t_477, q)) for q in (0, 1, 5, 25, 50, 75, 95, 99, 100)},
    all_total_percentiles={str(q): float(np.percentile(t_all, q)) for q in (0, 1, 5, 25, 50, 75, 95, 99, 100)},
)
np.save(f"{OUT}/cell_labels.npy", labels); np.save(f"{OUT}/total_umi_all.npy", t_all); np.save(f"{OUT}/total_umi_g1122.npy", t_1122); np.save(f"{OUT}/total_umi_g477.npy", t_477)
sp.save_npz(f"{OUT}/counts_g1122.npz", Xf, compressed=False)
rep["wall_seconds"] = round(time.time() - t0, 1); rep["python_ru_maxrss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 3)
json.dump(rep, open(f"{OUT}/report.json", "w"), indent=1)
open(f"{BASE}/state/ex_{NAME}.done", "w").write(f"{NAME}\t{rep['cells_kept']}\t{time.strftime('%F %T')}\n")
log("DONE", json.dumps({k: rep[k] for k in ("cells_in_file", "cells_kept", "file_not_in_metadata", "metadata_not_in_file", "zero_cells_g477", "zero_cells_g1122", "panel_nnz_kept", "wall_seconds", "python_ru_maxrss_gb")}))
