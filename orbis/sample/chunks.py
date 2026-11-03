"""Chunk sampler: independent samples of random chunks that share 2,000 anchor cells, stitched on the anchors."""
import numpy as np
import torch

from orbis.sample.sampler import STEP_EPS, rf_sample_xt


def proc(P, Q):
    """Procrustes fit of P onto Q (rotation, reflection and one scale); returns (R, scale, mean P, mean Q)."""
    a, b = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(b.T @ a); Rm = U @ Vt
    o = a @ Rm.T
    sc = np.sqrt((b ** 2).sum()) / (np.sqrt((o ** 2).sum()) + 1e-12)
    return Rm, sc, P.mean(0), Q.mean(0)


def chunk_partition(N):
    """2,000 anchor cells and the remaining cells split into parts of at most 18,000 (fixed seed 0)."""
    perm = np.random.default_rng(0).permutation(N)
    anchors, rest = np.sort(perm[:2000]), perm[2000:]
    parts = np.array_split(rest, int(np.ceil(len(rest) / 18000)))
    return anchors, parts


@torch.no_grad()
def run_chunk(net, ds, rows, K, seed, dev, noise_scale, data_rms, schedule, align):
    """Sample one chunk (rows of the prebuilt) as one graph; the noise comes from torch.manual_seed(seed)."""
    rows = np.sort(rows)
    n = len(rows)
    X = torch.from_numpy(np.ascontiguousarray(ds.feat[rows])).float().unsqueeze(0).to(dev)
    cc = torch.from_numpy(np.ascontiguousarray(ds.cc[rows])).view(1, n, 1).to(dev)
    cid = torch.from_numpy(np.ascontiguousarray(ds.cid[rows])).view(1, n, 1).to(dev)
    mask = torch.ones(1, n, dtype=torch.bool, device=dev)
    torch.manual_seed(seed)
    Y = rf_sample_xt(net, X, mask, cc, cid, K, STEP_EPS, noise_scale=noise_scale, align=align, pos_dim=3,
                     schedule=schedule, data_rms=data_rms)
    return rows, Y[0][mask[0]].cpu().numpy().astype(float)


def sample_chunks(net, ds, N, K, dev, noise_scale, data_rms, schedule, align):
    """All N cells: chunk k (its part plus the anchors) is sampled with seed 1000 + k; every later chunk is mapped
    onto the first chunk's anchor positions by proc and its non-anchor cells are written. Returns (N, 3)."""
    anchors, parts = chunk_partition(N)
    P_all = np.zeros((N, 3)); ref = None
    for k, part in enumerate(parts):
        r, P = run_chunk(net, ds, np.concatenate([part, anchors]), K, 1000 + k, dev, noise_scale, data_rms, schedule, align)
        is_a = np.isin(r, anchors)
        if ref is None:
            ref = P[is_a]; Pk = P; P_all[r[is_a]] = P[is_a]
        else:
            Rm, sc, mP, mQ = proc(P[is_a], ref); Pk = (P - mP) @ Rm.T * sc + mQ
        P_all[r[~is_a]] = Pk[~is_a]
    return P_all
