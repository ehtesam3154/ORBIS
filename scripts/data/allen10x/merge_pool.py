#!/usr/bin/env python
"""Merge the per-file extractions into one v3 pool with a fixed row order (np.sort of the cell labels): counts_g1122.npz, cell_labels.npy,
genes_g1122.csv, idx_g477.npy, library totals and the evaluation-only metadata.  usage: merge_pool.py [--names n1,n2] [--out DIR]"""
import sys, os, json, time, resource, argparse
import numpy as np, pandas as pd, scipy.sparse as sp

ap = argparse.ArgumentParser(); ap.add_argument("--names", default=""); ap.add_argument("--out", default=""); a = ap.parse_args()
RAW = os.path.join(os.environ["DATA_DIR"], "raw", "allen_10x")
BASE = os.path.join(os.environ["DATA_DIR"], "allen10x"); DOCS = os.path.join(os.environ["DATA_DIR"], "allen10x", "frozen")
PF = f"{BASE}/v3_pool/per_file"; OUT = a.out or f"{BASE}/v3_pool"
LIST = [l.split("\t")[0] for l in open(os.path.join(os.environ["OUT_DIR"], "v3_raw_file_list.tsv"))]
names = [n for n in a.names.split(",") if n] if a.names else LIST
assert set(names) <= set(LIST)
os.makedirs(OUT, exist_ok=True)
t0 = time.time()


def log(*x):
    print(f"[{time.time() - t0:7.1f}s rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:5.2f} GB]", *x, flush=True)


rep = dict(files=names, out=OUT, full_set=(names == LIST))
G1122 = pd.read_csv(f"{DOCS}/zhuang_g1122_gene_overlap.csv"); G477 = pd.read_csv(f"{DOCS}/allen_g477_gene_overlap.csv")
p1122 = {e: i for i, e in enumerate(G1122.ensembl_id)}
idx477 = np.array([p1122[e] for e in G477.ensembl_id], dtype=np.int64)
assert len(idx477) == 477 and len(set(idx477.tolist())) == 477 and (G1122.ensembl_id.values[idx477] == G477.ensembl_id.values).all()
old = np.load(os.path.join(os.environ["DATA_DIR"], "brains", "allen_harmony", "pca.npz"), allow_pickle=True)["genes"].astype(str)
rep["g477_order_equals_sign_reference_pca_genes"] = bool((old == G477.ensembl_id.values).all())
assert rep["g477_order_equals_sign_reference_pca_genes"], "STOP: the 477-gene order differs from brains/allen_harmony/pca.npz"
rep["g1122_order_equals_zhuang_cellstore_gene_names"] = bool(json.load(open(os.path.join(os.environ["DATA_DIR"], "brains", "stores", "abca1", "meta.json")))["gene_names"] == G1122.ensembl_id.tolist())
assert rep["g1122_order_equals_zhuang_cellstore_gene_names"]

lab, tall, t1122, t477, ips, src = [], [], [], [], [], []
fr = {}
for fi, n in enumerate(names):
    d = f"{PF}/{n}"
    assert os.path.exists(f"{BASE}/state/ex_{n}.done"), f"{n} not extracted"
    fr[n] = json.load(open(f"{d}/report.json"))
    L = np.load(f"{d}/cell_labels.npy"); lab.append(L); src.append(np.full(len(L), fi, np.int16))
    tall.append(np.load(f"{d}/total_umi_all.npy")); t1122.append(np.load(f"{d}/total_umi_g1122.npy")); t477.append(np.load(f"{d}/total_umi_g477.npy"))
    z = np.load(f"{d}/counts_g1122.npz"); ips.append(z["indptr"].astype(np.int64)); assert z["shape"].tolist() == [len(L), 1122] and len(ips[-1]) == len(L) + 1
    assert fr[n]["cells_kept"] == len(L) and fr[n]["panel_nnz_kept"] == int(ips[-1][-1])
rep["per_file_cells"] = {n: int(len(l)) for n, l in zip(names, lab)}
cat = np.concatenate(lab); N = len(cat)
tall, t1122, t477, src = map(np.concatenate, (tall, t1122, t477, src))
rowlen = np.concatenate([np.diff(i) for i in ips]); total_nnz = int(rowlen.sum())
off = np.concatenate([[0], np.cumsum([len(l) for l in lab])]).astype(np.int64)
idx_dtype = np.int32 if total_nnz < 2**31 - 1 else np.int64
rep.update(pool_cells=int(N), pool_nnz=total_nnz, index_dtype=str(np.dtype(idx_dtype)), int32_limit=2**31 - 1, label_dtype=str(cat.dtype))
log(f"pool: {N:,} cells, nnz {total_nnz:,} (int32 limit {2**31 - 1:,}) -> index dtype {np.dtype(idx_dtype)}; label dtype {cat.dtype}")

order = np.argsort(cat, kind="stable")
pool_labels = cat[order]
assert (pool_labels[1:] != pool_labels[:-1]).all(), "duplicate cell_label across v3 files"
assert (pool_labels == np.sort(cat)).all()
pos = np.empty(N, np.int64); pos[order] = np.arange(N)
rl_pool = rowlen[order]
indptr_pool = np.zeros(N + 1, np.int64); np.cumsum(rl_pool, out=indptr_pool[1:])
assert indptr_pool[-1] == total_nnz

data_pool = np.empty(total_nnz, np.int32); ind_pool = np.empty(total_nnz, idx_dtype)
SUB = 50_000; chk_sum = 0
for fi, n in enumerate(names):
    z = np.load(f"{PF}/{n}/counts_g1122.npz"); data = z["data"]; ind = z["indices"]; ip = ips[fi]; nf = len(ip) - 1
    assert data.dtype == np.int32 and len(data) == ip[-1]
    chk_sum += int(data.sum(dtype=np.int64))
    for a0 in range(0, nf, SUB):
        b0 = min(a0 + SUB, nf)
        prow = pos[off[fi] + a0: off[fi] + b0]
        dst0 = indptr_pool[prow]; src0 = ip[a0:b0]; ln = ip[a0 + 1:b0 + 1] - src0
        lo, hi = int(ip[a0]), int(ip[b0])
        dest = np.arange(lo, hi, dtype=np.int64) + np.repeat(dst0 - src0, ln)
        data_pool[dest] = data[lo:hi]; ind_pool[dest] = ind[lo:hi]
    log(f"scattered {n} ({nf:,} rows, {int(ip[-1]):,} nnz)")
    del z, data, ind
P = sp.csr_matrix((data_pool, ind_pool, indptr_pool.astype(idx_dtype)), shape=(N, 1122), copy=False)
assert P.indices.dtype == np.dtype(idx_dtype) and P.indptr.dtype == np.dtype(idx_dtype) and P.nnz == total_nnz
assert P.data is data_pool or np.shares_memory(P.data, data_pool)
if not P.has_sorted_indices:
    P.sort_indices()
assert P.has_canonical_format
BLK = 200_000
rs = np.concatenate([np.asarray(P[a0:a0 + BLK].sum(1)).ravel().astype(np.int64) for a0 in range(0, N, BLK)])
assert (rs == t1122[order]).all(), "pool row sums != streamed panel-1122 totals"
assert int(P.data.sum(dtype=np.int64)) == chk_sum
rep["checks"] = dict(row_sums_equal_streamed_g1122_totals=True, value_checksum_equals_sum_of_files=True, canonical_csr=True)
c477 = np.concatenate([np.asarray(P[a0:a0 + BLK][:, idx477].sum(1)).ravel().astype(np.int64) for a0 in range(0, N, BLK)])
assert (c477 == t477[order]).all(), "pool 477-column sums != streamed panel-477 totals (idx_g477 or the streamed 477 mask is wrong)"
rep["checks"]["g477_columns_sum_equals_streamed_g477_totals"] = True
log("consistency checks passed")

sp.save_npz(f"{OUT}/counts_g1122.npz", P, compressed=False)
np.save(f"{OUT}/cell_labels.npy", pool_labels); np.save(f"{OUT}/idx_g477.npy", idx477)
np.save(f"{OUT}/total_umi_all.npy", tall[order]); np.save(f"{OUT}/total_umi_g1122.npy", t1122[order]); np.save(f"{OUT}/total_umi_g477.npy", t477[order])
g = G1122[["panel_index", "ensembl_id", "gene_symbol", "10x_gene_index"]].copy()
g["in_g477"] = False; g["g477_index"] = -1
g.loc[idx477, "in_g477"] = True; g.loc[idx477, "g477_index"] = np.arange(477)
g.to_csv(f"{OUT}/genes_g1122.csv", index=False)
log("wrote matrix + arrays")

meta = pd.read_parquet(f"{BASE}/state/v3_meta.parquet")
mi = pd.Index(meta.cell_label.values)
ix = mi.get_indexer(pool_labels); assert (ix >= 0).all()
m = meta.iloc[ix].reset_index(drop=True)
assert (m.cell_label.values == pool_labels).all()
piv = pd.read_csv(f"{RAW}/metadata/WMB-taxonomy/views/cluster_to_cluster_annotation_membership_pivoted.csv")
assert piv.cluster_alias.is_unique
m = m.merge(piv, on="cluster_alias", how="left", validate="many_to_one")
rep["cells_without_taxonomy_row"] = int(m["subclass"].isna().sum())
assert (m.cell_label.values == pool_labels).all()
for c in ("neurotransmitter", "class", "subclass", "supertype", "cluster"):
    m[c] = m[c].astype("category")
m["src_file"] = np.array(names, dtype=object)[src[order]]
m["src_file"] = m["src_file"].astype("category")
m["total_umi_all"] = tall[order]; m["total_umi_g477"] = t477[order]; m["total_umi_g1122"] = t1122[order]
m.to_parquet(f"{OUT}/EVAL_ONLY_cell_metadata.parquet", index=False)
rep["src_file_equals_metadata_feature_matrix_label"] = bool((m.src_file.astype(str).values == m.feature_matrix_label.astype(str).values).all())
assert rep["src_file_equals_metadata_feature_matrix_label"]
rep.update(wall_seconds=round(time.time() - t0, 1), python_ru_maxrss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 3))
json.dump(rep, open(f"{OUT}/merge_report.json", "w"), indent=1)
log("MERGE_DONE", json.dumps({k: rep[k] for k in ("pool_cells", "pool_nnz", "index_dtype", "cells_without_taxonomy_row", "wall_seconds", "python_ru_maxrss_gb")}))
