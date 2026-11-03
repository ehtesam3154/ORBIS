#!/usr/bin/env python
"""Genes outside the model input for the gene-map readout, fixed from the MERFISH training brains alone (no 10x count is read):
the 645 genes of the 1,122-gene Zhuang panel that are not among the 477 input genes, kept when detected (count > 0) in >= 200
store cells of >= 2 of the four Zhuang brains and when their smoothed reference map is not constant; plus the 200-um grid of the
five-brain training support (4 voxels padding).  Writes GENEMAPS/genemaps_genes.csv and GENEMAPS/genemaps_grid.npz."""
import hashlib, os, time
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
import pandas as pd
import h5py
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter

import paths as PA

SUP = PA.SUPPORT
ZR = PA.MERFISH_RAW
STORES = PA.STORES
CACHE = PA.GENEMAPS
GENES_DOC = PA.FROZEN
ZB = [("abca1", "Zhuang-ABCA-1"), ("abca2_ref1", "Zhuang-ABCA-2"), ("abca3", "Zhuang-ABCA-3"), ("abca4", "Zhuang-ABCA-4")]
VOX, PAD, SIGMA = 0.2, 4, 1.0


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


class Grid:
    def __init__(self, pos_all):
        self.lo = np.floor(pos_all.min(0) / VOX).astype(int) - PAD
        hi = np.floor(pos_all.max(0) / VOX).astype(int) + PAD
        self.shape = tuple((hi - self.lo + 1).tolist())
        self.nvox = int(np.prod(self.shape))

    def flat(self, pos):
        ijk = np.floor(pos / VOX).astype(int) - self.lo
        ok = np.all((ijk >= 0) & (ijk < np.array(self.shape)), axis=1)
        f = np.full(len(pos), -1, np.int64)
        f[ok] = np.ravel_multi_index(ijk[ok].T, self.shape)
        return f


def smoothed_fields(grid, pos, X, w=None):
    n = X.shape[0]
    w = np.ones(n) if w is None else w
    f = grid.flat(pos)
    ins = f >= 0
    V = sp.csr_matrix((w[ins], (f[ins], np.flatnonzero(ins))), shape=(grid.nvox, n))
    F = (V @ X).toarray().astype(np.float64)
    tot = np.asarray(X.T @ w).ravel().astype(np.float64)
    return F, tot


def stage_a():
    os.makedirs(CACHE, exist_ok=True)
    out = f"{CACHE}/genemaps_grid.npz"
    if os.path.exists(out):
        log("stage A: cached", out, md5(out))
        return out
    z1122 = pd.read_csv(f"{GENES_DOC}/zhuang_g1122_gene_overlap.csv")
    z477 = pd.read_csv(f"{GENES_DOC}/allen_g477_gene_overlap.csv")
    cand = z1122[~z1122.ensembl_id.isin(set(z477.ensembl_id))].reset_index(drop=True)
    log(f"stage A: candidates {len(cand)} (1,122 - 477)")
    assert len(cand) == 645, len(cand)
    brain = np.load(f"{SUP}/support_brain.npy"); row = np.load(f"{SUP}/support_row.npy"); pcal = np.load(f"{SUP}/support_pos_cal_mm.npy").astype(np.float64)
    grid = Grid(pcal)
    log(f"grid shape {grid.shape} ({grid.nvox:,} voxels), lo {grid.lo.tolist()} (voxel units of {VOX} mm), support cells {len(pcal):,}")
    Xs, Ps, det = [], [], {}
    for bi, (sname, zname) in enumerate(ZB):
        t0 = time.time()
        labels = np.array(open(f"{STORES}/{sname}/cell_labels.txt").read().split(), dtype=object)
        sel = np.flatnonzero(brain == bi)
        assert np.array_equal(row[sel], np.arange(len(labels))), sname
        with h5py.File(f"{ZR}/{zname}-raw.h5ad", "r") as f:
            gid = np.array([x.decode() for x in f["var"]["gene_identifier"][:]])
            gix = pd.Index(gid).get_indexer(cand.ensembl_id)
            assert (gix >= 0).all(), f"{zname}: genes missing"
            obs = np.array([x.decode() for x in f["obs"]["cell_label"][:]])
            rix = pd.Index(obs).get_indexer(pd.Index(labels.astype(str)))
            assert (rix >= 0).all(), f"{zname}: store cells missing in h5ad"
            X = f["X"]; N = X.shape[0]; CH = X.chunks[0] if X.chunks else 20000
            need = np.zeros(N, bool); need[rix] = True
            blocks = []
            for a in range(0, N, CH):
                b = min(N, a + CH)
                if not need[a:b].any():
                    continue
                blk = X[a:b][:, gix]
                assert np.all(blk == np.round(blk)), f"{zname}: non-integer counts"
                keep = np.flatnonzero(need[a:b])
                blocks.append((a + keep, sp.csr_matrix(blk[keep].astype(np.float32))))
        rows_h5 = np.concatenate([k for k, _ in blocks]); M = sp.vstack([m for _, m in blocks]).tocsr()
        pos_in_h5 = pd.Index(rows_h5).get_indexer(rix)
        M = M[pos_in_h5]
        det[sname] = np.asarray((M > 0).sum(0)).ravel()
        Xs.append(M); Ps.append(pcal[sel])
        log(f"  {sname}: {len(labels):,} cells, nnz {M.nnz:,}, {time.time() - t0:.0f} s")
    Xz = sp.vstack(Xs).tocsr(); Pz = np.concatenate(Ps)
    detm = np.stack([det[s] for s, _ in ZB])
    ok_det = (detm >= 200).sum(0) >= 2
    F, tot = smoothed_fields(grid, Pz, Xz)
    Q = np.zeros((grid.nvox + 1, len(cand)), np.float32); nonconst = np.zeros(len(cand), bool)
    for g in range(len(cand)):
        if tot[g] <= 0:
            continue
        s = np.clip(gaussian_filter(F[:, g].reshape(grid.shape), SIGMA, mode="constant", truncate=4.0).ravel(), 0, None)
        over = max(tot[g] - s.sum(), 0.0)
        Q[:-1, g] = s / tot[g]; Q[-1, g] = over / tot[g]
        nonconst[g] = s.max() > s.min()
    elig = ok_det & nonconst
    log(f"stage A: detection rule pass {ok_det.sum()}, nonconstant {nonconst.sum()}, ELIGIBLE {elig.sum()}")
    cand = cand.assign(**{f"det_{s}": det[s] for s, _ in ZB}, eligible=elig, zhuang_total_counts=tot)
    gl = f"{CACHE}/genemaps_genes.csv"
    cand.to_csv(gl, index=False)
    e = np.flatnonzero(elig)
    np.savez_compressed(out, sqrtQ=np.sqrt(Q[:, e]).astype(np.float32), ensembl=cand.ensembl_id.to_numpy()[e].astype(str),
                        symbol=cand.gene_symbol.to_numpy()[e].astype(str), lo=grid.lo, shape=np.array(grid.shape))
    return out


def main():
    """Fix the gene list and the grid."""
    stage_a()


if __name__ == "__main__":
    main()
