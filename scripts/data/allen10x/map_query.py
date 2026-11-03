#!/usr/bin/env python
"""Map the whole 10x v3 pool into the Symphony reference with ONE pooled symphonypy.tl.map_embedding call and ONE batch label.

Query = counts of the 477 panel genes -> library size over the 477 genes to 1e4 (floor max(sum, 1)) -> log1p -> map_embedding (reference
mean / SD scaling, clip 10, reference loadings, soft assignment, Symphony ridge correction).  Symphony's correction depends on the query
collection, so the pool is always mapped in one call.  No label file is read.
--route sparse keeps the query as a float64 CSR (symphonypy densifies it itself; identical to --route dense)."""
import argparse
import datetime
import json
import os
import sys
import time

import numpy as np

import symphony_common as C

POOL = None
BATCH_LABEL = "tenx_v3"


def log(msg):
    cur, hwm = C.rss_gb()
    print(f"[{datetime.datetime.now():%H:%M:%S}] rss {cur:5.1f}/{hwm:5.1f} GB  {msg}", flush=True)


def lognorm_csr(c):
    import scipy.sparse as sps
    s = np.asarray(c.sum(axis=1)).ravel()
    den = np.maximum(s, 1.0)
    rows = np.repeat(np.arange(c.shape[0]), np.diff(c.indptr))
    data = np.log1p(c.data / den[rows] * C.LIB)
    return sps.csr_matrix((data, c.indices.copy(), c.indptr.copy()), shape=c.shape), s


def main(a):
    """Map the pool (or a seeded fixture of it) and save the corrected / uncorrected embeddings and the soft assignments."""
    global POOL
    POOL = a.pool
    import anndata as ad
    import pandas as pd
    import scanpy as sc
    import scipy.sparse as sps
    import symphonypy as sp
    from importlib.metadata import version
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True) if a.out else None
    adata_ref = sc.read_h5ad(f"{a.ref}/symphony_ref.h5ad")
    pca = np.load(f"{a.ref}/pca.npz", allow_pickle=True)
    pca_genes = np.array([str(x) for x in pca["genes"]])
    g = pd.read_csv(f"{POOL}/genes_g1122.csv")
    ids = g["ensembl_id"].to_numpy().astype(str)
    idx = np.load(f"{POOL}/idx_g477.npy")
    assert len(idx) == C.NGENES and len(ids) == 1122
    assert (ids[idx] == pca_genes).all(), "pool idx_g477 genes != pca.npz genes"
    assert list(adata_ref.var_names) == list(pca_genes), "reference var_names != pca.npz genes"
    sub = g[g["in_g477"]].sort_values("g477_index")
    assert (sub["ensembl_id"].to_numpy().astype(str) == pca_genes).all() and (np.sort(sub.index.to_numpy()) == np.sort(idx)).all()
    hs = adata_ref.uns["harmony"]
    log(f"reference loaded: var {adata_ref.shape}, K={int(hs['K'])}, converged={hs['converged']}, sigma uniq={np.unique(np.asarray(hs['sigma']))}; "
        f"gene order asserted three-way (pool genes_g1122[idx_g477] == pca.npz genes == symphony_ref var_names)")
    labels = np.load(f"{POOL}/cell_labels.npy")
    counts = sps.load_npz(f"{POOL}/counts_g1122.npz")
    assert counts.shape == (len(labels), 1122) and counts.format == "csr", (counts.shape, counts.format)
    log(f"pool loaded: {counts.shape} {counts.format} {counts.dtype} nnz {counts.nnz:,}; cell_labels {labels.shape} {labels.dtype}")
    rows = None
    if a.n_cells:
        rows = np.sort(np.random.default_rng(a.seed).choice(counts.shape[0], a.n_cells, replace=False))
        counts = counts[rows]
        labels = labels[rows]
        log(f"FIXTURE: {a.n_cells:,} rows drawn with default_rng({a.seed}) (code/memory test; corrected values are NOT those of the whole-pool mapping)")
    X477 = counts[:, idx].tocsr()
    del counts
    X477 = X477.astype(np.float64)
    N = X477.shape[0]
    log(f"477-gene counts: {X477.shape} float64 CSR nnz {X477.nnz:,}; max count {X477.data.max():.0f}; min nonzero {X477.data.min():.0f}")
    t1 = time.time()
    if a.route == "sparse":
        Xq, lib = lognorm_csr(X477)
        if a.check_lognorm:
            bad = 0
            for lo in range(0, N, 50000):
                hi = min(N, lo + 50000)
                xd, ld = C.lognorm(X477[lo:hi].toarray())
                bad += int((xd != Xq[lo:hi].toarray()).sum()) + int((ld != lib[lo:hi]).sum())
            log(f"lognorm_csr vs symphony_common.lognorm (dense chunks of 50k): {bad} differing entries")
            assert bad == 0
    else:
        Xq, lib = C.lognorm(X477.toarray())
    del X477
    floor_zero, floor_small = int((lib == 0).sum()), int(((lib > 0) & (lib < 1)).sum())
    log(f"lognorm done in {time.time() - t1:.1f} s; route={a.route}; query container {type(Xq).__name__} {Xq.dtype}; library over 477 genes: min {lib.min():.0f} "
        f"median {np.median(lib):.0f} max {lib.max():.0f}; floor hits: all-zero {floor_zero}, 0<sum<1 {floor_small}")
    obs = pd.DataFrame({"platform": pd.Categorical([BATCH_LABEL] * N)}, index=pd.Index(labels.astype(str)))
    aq = ad.AnnData(X=Xq, obs=obs, var=pd.DataFrame(index=pd.Index(list(adata_ref.var_names))))
    aq.uns["log1p"] = {}
    del Xq
    log(f"AnnData query {aq.shape}; platform levels {obs['platform'].value_counts().to_dict()} (ONE batch label)")
    try:
        open("/proc/self/clear_refs", "w").write("5")
        hwm_reset = True
    except Exception:
        hwm_reset = False
    rss_before_map = C.rss_gb()[0]
    t2 = time.time()
    sp.tl.map_embedding(adata_query=aq, adata_ref=adata_ref, key="platform", use_genes_column=None)
    t_map = time.time() - t2
    map_peak = C.rss_gb()[1]
    log(f"map_embedding peak RSS since reset ({hwm_reset}): {map_peak:.2f} GB (RSS before the call {rss_before_map:.2f} GB)")
    Xp = aq.obsm["X_pca_reference"]
    Xh = aq.obsm["X_pca_harmony"]
    R = aq.obsm["X_pca_harmony_symphony_R"]
    log(f"map_embedding done in {t_map:.1f} s; X_pca_reference {Xp.shape} {Xp.dtype} C={Xp.flags['C_CONTIGUOUS']} F={Xp.flags['F_CONTIGUOUS']}; "
        f"X_pca_harmony {Xh.shape} {Xh.dtype} C={Xh.flags['C_CONTIGUOUS']} F={Xh.flags['F_CONTIGUOUS']}; R {R.shape} {R.dtype}")
    assert Xp.shape == (N, C.NPC) and Xh.shape == (N, C.NPC) and R.shape == (N, int(hs["K"]))
    assert np.isfinite(Xh).all() and np.isfinite(Xp).all() and np.isfinite(R).all()
    Rsum = R.sum(1)
    dnorm = np.linalg.norm(Xh - Xp, axis=1)
    stats = dict(R_rowsum_min=float(Rsum.min()), R_rowsum_max=float(Rsum.max()), R_max_per_cell_median=float(np.median(R.max(1))),
                 correction_norm_median=float(np.median(dnorm)), correction_norm_p05_p95=[float(np.percentile(dnorm, 5)), float(np.percentile(dnorm, 95))],
                 X_pca_reference_norm_median=float(np.median(np.linalg.norm(Xp, axis=1))), X_pca_harmony_norm_median=float(np.median(np.linalg.norm(Xh, axis=1))),
                 column_means_corrected_first5=Xh.mean(0)[:5].tolist(), column_means_uncorrected_first5=Xp.mean(0)[:5].tolist())
    log(f"sanity: R row sums [{stats['R_rowsum_min']:.12f}, {stats['R_rowsum_max']:.12f}]; median correction norm {stats['correction_norm_median']:.4f}; "
        f"median |X_pca_reference| {stats['X_pca_reference_norm_median']:.3f}, median |X_pca_harmony| {stats['X_pca_harmony_norm_median']:.3f}")
    saved = {}
    if a.out:
        t3 = time.time()
        np.save(f"{a.out}/X_pca_harmony_f32.npy", np.ascontiguousarray(Xh, dtype=np.float32)); saved["X_pca_harmony_f32.npy"] = (N, C.NPC)
        if a.save_f64:
            np.save(f"{a.out}/X_pca_harmony_f64.npy", np.ascontiguousarray(Xh)); saved["X_pca_harmony_f64.npy"] = (N, C.NPC)
            np.save(f"{a.out}/X_pca_reference_f64.npy", np.ascontiguousarray(Xp)); saved["X_pca_reference_f64.npy"] = (N, C.NPC)
            np.save(f"{a.out}/X_pca_harmony_symphony_R_f64.npy", np.ascontiguousarray(R)); saved["X_pca_harmony_symphony_R_f64.npy"] = tuple(R.shape)
        np.save(f"{a.out}/cell_labels.npy", labels); saved["cell_labels.npy"] = tuple(labels.shape)
        if rows is not None:
            np.save(f"{a.out}/fixture_rows.npy", rows)
        log(f"saved {list(saved)} in {time.time() - t3:.1f} s")
    cur, hwm = C.rss_gb()
    man = dict(
        what="10x v3 pool mapped into the Symphony reference, ONE pooled symphonypy.tl.map_embedding call, ONE batch label",
        mode="FIXTURE (code/memory test)" if a.n_cells else "WHOLE POOL", route=a.route, n_cells=int(N), fixture_seed=a.seed if a.n_cells else None,
        batch_label=BATCH_LABEL, batch_key="platform", batch_levels=1, lamb="symphonypy default: 1 per batch level (intercept 0)", pooled_calls=1,
        use_genes_column=None, query_dtype="float64", query_container=a.route, reference=a.ref, pool=POOL,
        reference_K=int(hs["K"]), reference_sigma=float(np.unique(np.asarray(hs["sigma"]))[0]), reference_converged=bool(hs["converged"]),
        library_floor_hits_all_zero=floor_zero, library_floor_hits_0_lt_sum_lt_1=floor_small,
        library_477_min=float(lib.min()), library_477_median=float(np.median(lib)), library_477_max=float(lib.max()),
        outputs=dict(X_pca_reference=dict(shape=list(Xp.shape), dtype=str(Xp.dtype)), X_pca_harmony=dict(shape=list(Xh.shape), dtype=str(Xh.dtype)),
                     X_pca_harmony_symphony_R=dict(shape=list(R.shape), dtype=str(R.dtype), note="stored (cells, K) as symphonypy writes it")),
        saved=saved, sanity=stats, seconds=dict(total=round(time.time() - t0, 1), map_embedding=round(t_map, 1)), rss_gb_current_peak=[cur, hwm],
        map_embedding_peak_rss_gb=map_peak, rss_before_map_gb=rss_before_map, hwm_reset_worked=hwm_reset,
        versions={k: version(k) for k in ("numpy", "scanpy", "anndata", "harmonypy", "symphonypy", "scipy", "pandas")}, python=sys.version.split()[0],
        threads={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMPY_MADVISE_HUGEPAGE")},
        cmdline=sys.argv, finished=f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    if a.out:
        json.dump(man, open(f"{a.out}/mapping_manifest{'_fixture' if a.n_cells else ''}.json", "w"), indent=1)
    print("MANIFEST " + json.dumps({k: man[k] for k in ("mode", "route", "n_cells", "seconds", "rss_gb_current_peak", "library_floor_hits_all_zero")}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="Symphony reference directory (symphony_ref.h5ad, pca.npz)")
    ap.add_argument("--pool", required=True, help="merged 10x v3 pool (counts_g1122.npz, genes_g1122.csv, idx_g477.npy, cell_labels.npy)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--route", choices=["dense", "sparse"], default="sparse")
    ap.add_argument("--n_cells", type=int, default=0, help="0 = whole pool; else a seeded fixture of that many rows (code/memory test only)")
    ap.add_argument("--seed", type=int, default=25002)
    ap.add_argument("--save_f64", action="store_true")
    ap.add_argument("--check_lognorm", action="store_true")
    main(ap.parse_args())
