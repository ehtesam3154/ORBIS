#!/usr/bin/env python
"""Gene-map field code, the MERFISH reference maps and the normalised 10x expression of the uniform cohorts.
Per cell, within each assay: counts / total over the 1,122 shared genes x 1e4, log1p.  Map of a gene = Gaussian-smoothed sum of
expression / Gaussian-weighted cell count (sigma 200 um on the 200-um grid); a voxel with weighted count < 5 is not covered.
Reference = measured cells of the four Zhuang brains (mask: smoothed count >= 20); ceiling = mouse 1 against mice 2-4.
`python genemaps_fields.py` writes GENEMAPS/genemaps_reference.npz and GENEMAPS/genemaps_10x_norm_primary.npz."""
import hashlib, os, time
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
import pandas as pd
import h5py
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata

import endpoints as EV
import paths as PA

APP = PA.APP
SUP = PA.SUPPORT
POOL = PA.POOL
ZR = PA.MERFISH_RAW
STORES = PA.STORES
CACHE = PA.GENEMAPS
GENES_DOC = PA.FROZEN
FROZEN = f"{CACHE}/genemaps_genes.csv"
ZB = [("abca1", "Zhuang-ABCA-1"), ("abca2_ref1", "Zhuang-ABCA-2"), ("abca3", "Zhuang-ABCA-3"), ("abca4", "Zhuang-ABCA-4")]
VOX, PAD, SIGMA = 0.2, 4, 1.0
KPEAK = (2 * np.pi) ** 1.5
MASK_MIN, FILL_MIN = 20.0, 5.0
GATE_CEIL, GATE_RAND = 0.2, 0.05


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


class Grid:
    def __init__(self, lo, shape):
        self.lo = np.asarray(lo); self.shape = tuple(int(x) for x in shape); self.nvox = int(np.prod(self.shape))

    def flat(self, pos):
        ijk = np.floor(pos / VOX).astype(int) - self.lo
        ok = np.all((ijk >= 0) & (ijk < np.array(self.shape)), axis=1)
        f = np.full(len(pos), -1, np.int64)
        f[ok] = np.ravel_multi_index(ijk[ok].T, self.shape)
        return f


def smooth(a, grid):
    return gaussian_filter(a.reshape(grid.shape), SIGMA, mode="constant", truncate=4.0).ravel()


def sums_at_mask(grid, mask, pos, X, w):
    f = grid.flat(pos); ins = f >= 0
    V = sp.csr_matrix((w[ins], (f[ins], np.flatnonzero(ins))), shape=(grid.nvox, len(w)))
    cnt = KPEAK * smooth(np.asarray(V.sum(1)).ravel(), grid)[mask]
    F = (V @ X).toarray().astype(np.float32)
    num = np.empty((len(mask), X.shape[1]), np.float64)
    for g in range(X.shape[1]):
        num[:, g] = KPEAK * smooth(F[:, g], grid)[mask]
    gmean = np.asarray(X.T @ w).ravel() / w.sum()
    return cnt, num, gmean


def field(cnt, num, gmean):
    fill = cnt < FILL_MIN
    fld = num / np.where(fill, 1.0, cnt)[:, None]
    fld[fill] = gmean[None, :]
    return fld, float(fill.mean())


def corr_cols(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    den = np.sqrt((A * A).sum(0) * (B * B).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (A * B).sum(0) / den
    r[den <= 0] = np.nan
    return r


def score(fld, ref, ref_rank):
    return corr_cols(rankdata(fld, axis=0), ref_rank), corr_cols(fld, ref)


def read_zhuang():
    fz = pd.read_csv(FROZEN); fz = fz[fz.eligible].reset_index(drop=True)
    assert len(fz) == 645, len(fz)
    z1122 = pd.read_csv(f"{GENES_DOC}/zhuang_g1122_gene_overlap.csv")
    assert len(z1122) == 1122
    brain = np.load(f"{SUP}/support_brain.npy"); row = np.load(f"{SUP}/support_row.npy"); pcal = np.load(f"{SUP}/support_pos_cal_mm.npy").astype(np.float64)
    r3 = np.load(f"{CACHE}/genemaps_grid.npz")
    grid = Grid(r3["lo"], r3["shape"])
    assert list(r3["ensembl"]) == list(fz.ensembl_id)
    Xs, Ps, ndrop = [], [], {}
    for bi, (sname, zname) in enumerate(ZB):
        t0 = time.time()
        labels = np.array(open(f"{STORES}/{sname}/cell_labels.txt").read().split(), dtype=object)
        sel = np.flatnonzero(brain == bi)
        assert np.array_equal(row[sel], np.arange(len(labels))), sname
        with h5py.File(f"{ZR}/{zname}-raw.h5ad", "r") as f:
            gid = np.array([x.decode() for x in f["var"]["gene_identifier"][:]])
            g1122 = pd.Index(gid).get_indexer(z1122.ensembl_id); assert (g1122 >= 0).all()
            g645 = pd.Index(gid).get_indexer(fz.ensembl_id); assert (g645 >= 0).all()
            obs = np.array([x.decode() for x in f["obs"]["cell_label"][:]])
            rix = pd.Index(obs).get_indexer(pd.Index(labels.astype(str))); assert (rix >= 0).all()
            X = f["X"]; N = X.shape[0]; CH = (X.chunks[0] if X.chunks else 1682) * 20
            need = np.zeros(N, bool); need[rix] = True
            blocks = []
            for a in range(0, N, CH):
                b = min(N, a + CH)
                if not need[a:b].any():
                    continue
                keep = np.flatnonzero(need[a:b])
                blk = X[a:b][keep]
                assert np.all(blk == np.round(blk)), f"{zname}: non-integer counts"
                tot = blk[:, g1122].sum(1, dtype=np.float64)
                e = blk[:, g645].astype(np.float64)
                with np.errstate(invalid="ignore", divide="ignore"):
                    e = np.log1p(e / tot[:, None] * 1e4)
                e[tot <= 0] = 0.0
                blocks.append((a + keep, sp.csr_matrix(e.astype(np.float32)), tot > 0))
        rows_h5 = np.concatenate([k for k, _, _ in blocks]); M = sp.vstack([m for _, m, _ in blocks]).tocsr()
        okc = np.concatenate([o for _, _, o in blocks])
        pih = pd.Index(rows_h5).get_indexer(rix)
        M, okc = M[pih], okc[pih]
        ndrop[sname] = int((~okc).sum())
        Xs.append(M[okc]); Ps.append(pcal[sel][okc])
        log(f"  {sname}: {len(labels):,} cells, dropped total-0 {ndrop[sname]:,}, nnz {M.nnz:,}, {time.time() - t0:.0f} s")
    return grid, fz, Xs, Ps, ndrop


def stage_a():
    os.makedirs(CACHE, exist_ok=True)
    out = f"{CACHE}/genemaps_reference.npz"
    if os.path.exists(out):
        log("stage A: cached", out, md5(out))
        return out
    grid, fz, Xs, Ps, ndrop = read_zhuang()
    cnt_b = []
    for P in Ps:
        f = grid.flat(P); cnt_b.append(KPEAK * smooth(np.bincount(f[f >= 0], minlength=grid.nvox).astype(np.float64), grid))
    cnt_all = sum(cnt_b)
    mask = np.flatnonzero(cnt_all >= MASK_MIN)
    log(f"mask voxels {len(mask):,} (smoothed Zhuang count >= {MASK_MIN})")
    parts = [sums_at_mask(grid, mask, P, Xb, np.ones(Xb.shape[0])) for P, Xb in zip(Ps, Xs)]
    nb = [Xb.shape[0] for Xb in Xs]

    def combine(ix):
        c = sum(parts[i][0] for i in ix); n = sum(parts[i][1] for i in ix)
        gm = sum(parts[i][2] * nb[i] for i in ix) / sum(nb[i] for i in ix)
        return c, n, gm
    c, n, gm = combine([0, 1, 2, 3])
    ref, fill_ref = field(c, n, gm)
    assert fill_ref == 0.0
    ref_rank = rankdata(ref, axis=0)
    fa, fill_a = field(*combine([0])); fb, fill_b = field(*combine([1, 2, 3]))
    ceil_sp, ceil_pe = corr_cols(rankdata(fa, axis=0), rankdata(fb, axis=0)), corr_cols(fa, fb)
    log(f"CEILING mouse 1 vs mice 2-4: macro Spearman {np.nanmean(ceil_sp):.4f}, Pearson {np.nanmean(ceil_pe):.4f}, fill {fill_a:.3f} / {fill_b:.3f}, NaN genes {int(np.isnan(ceil_sp).sum())}")
    np.savez_compressed(out, mask=mask, ref=ref.astype(np.float32), ref_rank=ref_rank.astype(np.float32), lo=grid.lo, shape=np.array(grid.shape),
                        ensembl=fz.ensembl_id.to_numpy().astype(str), symbol=fz.gene_symbol.to_numpy().astype(str),
                        ceil_sp=ceil_sp, ceil_pe=ceil_pe, ceil_fill=np.array([fill_a, fill_b]), zh_dropped=np.array([ndrop[s] for s, _ in ZB]))
    return out


G_ = {}


def init_worker(ref, xq_path):
    z = np.load(ref)
    G_["mask"] = z["mask"]; G_["ref"] = z["ref"].astype(np.float64); G_["ref_rank"] = z["ref_rank"].astype(np.float64)
    G_["grid"] = Grid(z["lo"], z["shape"])
    q = np.load(xq_path, allow_pickle=True)
    G_["Xq"] = sp.csr_matrix((q["data"], q["indices"], q["indptr"]), shape=tuple(q["shape"]))
    G_["rowpos"] = pd.Index(q["pool_rows"])
    G_["ok"] = q["ok"]


def cache_10x(ref):
    """Normalised 10x expression of the frozen genes for every cell of the five uniform cohorts -> GENEMAPS/genemaps_10x_norm_primary.npz."""
    z = np.load(ref); ens = z["ensembl"]
    cohort_rows = {c: np.load(f"{APP}/cohorts/primary{c}/rows.npy") for c in [1, 2, 3, 4, 5]}
    xq = f"{CACHE}/genemaps_10x_norm_primary.npz"
    if not os.path.exists(xq):
        pg = pd.read_csv(f"{POOL}/genes_g1122.csv"); assert len(pg) == 1122
        col = pd.Index(pg.ensembl_id).get_indexer(ens); assert (col >= 0).all()
        allrows = np.unique(np.concatenate(list(cohort_rows.values())))
        M = sp.load_npz(f"{POOL}/counts_g1122.npz").tocsr()[allrows]
        tot = np.asarray(M.sum(1)).ravel().astype(np.float64)
        E = M[:, col].astype(np.float64).tocsr()
        sc = np.where(tot > 0, 1e4 / np.where(tot > 0, tot, 1), 0.0)
        E = sp.diags(sc) @ E; E.data = np.log1p(E.data); E = E.astype(np.float32).tocsr(); E.eliminate_zeros()
        np.savez_compressed(xq, data=E.data, indices=E.indices, indptr=E.indptr, shape=np.array(E.shape), pool_rows=allrows, ensembl=ens, ok=tot > 0)
        log(f"10x normalised cached: {E.shape} nnz {E.nnz:,}, total-0 cells {int((tot <= 0).sum())} -> {xq}")
    return xq


def main():
    """Reference maps of the MERFISH brains, then the normalised 10x expression."""
    cache_10x(stage_a())


if __name__ == "__main__":
    main()
