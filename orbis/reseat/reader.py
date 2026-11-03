"""Niche reader of the reseating step.

From expression alone, the reader predicts for every test cell the soft composition of its neighbourhood at two scales (32 and 128
nearest other cells). Pieces: a gene panel (the 50 genes with the highest Moran's I on one training specimen, excluded from all inputs);
K32 (standardiser -> PCA-30 -> 32-component diagonal Gaussian mixture; p = posterior); targets (mean p over each training cell's
nearest other cells, within its section for serial-section data, in 3-D otherwise); ridge and MLP readers picked per scale on held-out
rows; sigma = held-out RMS residual of a reader fitted without those rows. Outputs are clipped and renormalised to the simplex."""
import json
import os
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

from orbis.reseat.inputs import md5

WORKERS = max(1, len(os.sched_getaffinity(0)))
K_COMP, N_PC, N_FIT_K32, N_READER, PANEL_N, MORAN_K = 32, 30, 200_000, 300_000, 50, 15
RS = (32, 128)
T0 = time.time()


def P(*a):
    """Print with the elapsed time."""
    print(f"[{time.time() - T0:7.0f} s]", *a, flush=True)


def knn_excluding_self(Z, k, tree=None, query=None, chunk=400_000):
    """(len(query), k) indices of the k nearest other points (self removed wherever it appears, else the last of k + 1 dropped)."""
    tree = cKDTree(Z) if tree is None else tree
    q = np.arange(len(Z)) if query is None else np.asarray(query)
    out = np.empty((len(q), k), np.int64)
    for s in range(0, len(q), chunk):
        qq = q[s:s + chunk]
        _, nb = tree.query(Z[qq], k=k + 1, workers=WORKERS)
        o = nb[:, 1:].copy()
        for r in np.flatnonzero(nb[:, 0] != qq):
            o[r] = nb[r][nb[r] != qq[r]][:k]
        out[s:s + len(qq)] = o
    return out


def section_frame_fit(X, sec):
    """Rotation taking X to a frame where every section is a plane of constant last coordinate (normal = smallest-eigenvalue axis of the pooled within-section scatter)."""
    D = X.shape[1]
    W = np.zeros((D, D))
    for s in np.unique(sec):
        Z = X[sec == s]
        Z = Z - Z.mean(0)
        W += Z.T @ Z
    ev, V = np.linalg.eigh(W)
    n = V[:, 0]
    a = int(np.argmax(np.abs(n)))
    n = n * np.sign(n[a])
    axes = []
    for b in [i for i in range(D) if i != a]:
        e = np.zeros(D)
        e[b] = 1.0
        for v in axes + [n]:
            e = e - (e @ v) * v
        axes.append(e / np.linalg.norm(e))
    R = np.stack(axes + [n], axis=1)
    return R, np.zeros(D)


class Gene:
    """Standardiser fitted on training rows (float64 sums over chunks; sd floor 1e-3). The call returns float32 values; with
    Gene.chunked = True it works in chunks of 100,000 rows into a C-ordered output (same values, no full-size float64 temporaries; the
    memory order of the output can differ, which changes later float32 reductions such as the PCA mean)."""
    chunked = False

    def __init__(self, X):
        s = np.zeros(X.shape[1]); s2 = np.zeros(X.shape[1])
        for i in range(0, len(X), 100000):
            c = np.asarray(X[i:i + 100000], dtype=np.float64); s += c.sum(0); s2 += (c * c).sum(0)
        self.mu = s / len(X); self.sd = np.sqrt(np.maximum(s2 / len(X) - self.mu ** 2, 0)) + 1e-3

    def __call__(self, X):
        if self.chunked:
            X = np.asarray(X)
            out = np.empty(X.shape, np.float32)
            for i in range(0, len(X), 100000):
                out[i:i + 100000] = ((np.asarray(X[i:i + 100000]) - self.mu) / self.sd).astype(np.float32)
            return out
        return ((np.asarray(X) - self.mu) / self.sd).astype(np.float32)


def moments(X, Y, rows):
    """Gram and cross-moment sums of X and Y over the given rows (float64, chunked)."""
    d, t = X.shape[1], Y.shape[1]; G = np.zeros((d, d)); sx = np.zeros(d); BY = np.zeros((d, t)); sy = np.zeros(t)
    for i in range(0, len(rows), 50000):
        r = rows[i:i + 50000]; c = X[r].astype(np.float64); y = Y[r].astype(np.float64)
        G += c.T @ c; sx += c.sum(0); BY += c.T @ y; sy += y.sum(0)
    return dict(G=G, sx=sx, BY=BY, sy=sy, n=len(rows))


ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]


def _centred(m):
    """Centred moments, means and count."""
    n = m["n"]; mx, my = m["sx"] / n, m["sy"] / n
    return m["G"] - n * np.outer(mx, mx), m["BY"] - n * np.outer(mx, my), mx, my, n


def ridge(X, Y, va, tests, blocks):
    """Ridge reader: per target column, the alpha with the lowest MSE on the held-out rows, then a refit on all rows with that alpha."""
    tr = np.flatnonzero(~va); vr = np.flatnonzero(va); T = Y.shape[1]
    Gc, Bc, mx, my, n = _centred(moments(X, Y, tr)); mv = moments(X, Y, vr)
    Xv = X[vr].astype(np.float64); Yv = Y[vr].astype(np.float64); errs, pvs = [], []
    for al in ALPHAS:
        Wm = np.linalg.solve(Gc + al * n * np.eye(len(mx)), Bc); pv = (Xv - mx) @ Wm + my
        errs.append(((pv - Yv) ** 2).mean(0)); pvs.append(pv.astype(np.float32))
    bi = np.argmin(np.array(errs), 0); pv = np.stack([pvs[bi[j]][:, j] for j in range(T)], 1); del pvs
    mt = moments(X, Y, tr); ma = {k: mt[k] + mv[k] for k in ("G", "sx", "BY", "sy", "n")}
    Gc2, Bc2, mx2, my2, n2 = _centred(ma)
    out = {k: np.zeros((len(v), T), np.float32) for k, v in tests.items()}
    for i in np.unique(bi):
        js = np.flatnonzero(bi == i); Wm = np.linalg.solve(Gc2 + ALPHAS[i] * n2 * np.eye(len(mx2)), Bc2[:, js])
        for k, Xt in tests.items():
            for s in range(0, len(Xt), 50000):
                out[k][s:s + 50000, js] = (Xt[s:s + 50000].astype(np.float64) - mx2) @ Wm + my2[js]
    return out, dict(alpha_counts={str(ALPHAS[i]): int((bi == i).sum()) for i in np.unique(bi)}, val=block_scores(Yv, pv, blocks),
                     n_fit=len(tr), n_val=len(vr))


def mlp(X, Y, va, tests, blocks, max_ep=60, patience=6, bs=2048, seed=0):
    """MLP reader: early stopping per target block on the held-out rows, then a refit from scratch on all rows; each block is read at its best epoch."""
    d, T = X.shape[1], Y.shape[1]; hid = (256, 256) if d <= 200 else (512, 256)
    Xt = torch.from_numpy(X); Tt = {k: torch.from_numpy(v) for k, v in tests.items()}
    tr = np.flatnonzero(~va); vr = np.flatnonzero(va); allr = np.arange(len(X))

    def build():
        torch.manual_seed(seed)
        return torch.nn.Sequential(torch.nn.Linear(d, hid[0]), torch.nn.GELU(), torch.nn.Linear(hid[0], hid[1]), torch.nn.GELU(),
                                   torch.nn.Linear(hid[1], T))

    def predict(net, Xq):
        net.eval(); o = []
        with torch.no_grad():
            for i in range(0, len(Xq), 65536):
                o.append(net(Xq[i:i + 65536]))
        return torch.cat(o).numpy().astype(np.float64)

    def train(rows, epochs, after_epoch):
        ym, ys = Y[rows].mean(0).astype(np.float64), Y[rows].std(0).astype(np.float64); ys = np.where(ys > 0, ys, 1.0)
        Yn = torch.from_numpy(((Y - ym) / ys).astype(np.float32)); net = build()
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4); rt = torch.from_numpy(rows)
        g = torch.Generator().manual_seed(seed)
        for ep in range(epochs):
            net.train(); perm = rt[torch.randperm(len(rt), generator=g)]; tl = 0.0; nb = 0
            for i in range(0, len(perm), bs):
                idx = perm[i:i + bs]; loss = ((net(Xt[idx]) - Yn[idx]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step(); tl += float(loss); nb += 1
            if after_epoch(ep, net, ym, ys, tl / nb):
                return ep + 1
        return epochs

    nB = len(blocks); best = np.full(nB, np.inf); best_ep = np.zeros(nB, int); since = [0]; Xv = Xt[vr]; Yv = Y[vr].astype(np.float64)
    hist = []; vpred = np.zeros((len(vr), T), np.float32)

    def ev1(ep, net, ym, ys, tl):
        p = predict(net, Xv) * ys + ym; msec = ((p - Yv) ** 2).mean(0) / ys ** 2; mse = np.array([msec[b].mean() for b in blocks])
        imp = mse < best - 1e-5; hist.append((ep, round(tl, 4), np.round(mse, 4).tolist()))
        for j in np.flatnonzero(imp):
            best[j] = mse[j]; best_ep[j] = ep; vpred[:, blocks[j]] = p[:, blocks[j]]
        since[0] = 0 if imp.any() else since[0] + 1
        P(f"    MLP epoch {ep}: train loss {tl:.4f} | val block MSE (std) {np.round(mse, 3).tolist()}")
        return since[0] >= patience
    n1 = train(tr, max_ep, ev1)
    E = int(best_ep.max()) + 1; out = {k: np.zeros((len(v), T), np.float32) for k, v in tests.items()}

    def ev2(ep, net, ym, ys, tl):
        js = [j for j in range(nB) if best_ep[j] == ep]
        if js:
            cols = np.concatenate([blocks[j] for j in js])
            for k in tests:
                out[k][:, cols] = (predict(net, Tt[k]) * ys + ym)[:, cols]
        return False
    train(allr, E, ev2)
    return out, dict(best_epoch=best_ep.tolist(), epochs_phase1=n1, epochs_refit=E, val_mse_std=best.tolist(), n_fit=len(tr),
                     n_val=len(vr), hidden=hid, val_curve=hist, val=block_scores(Yv, vpred, blocks))


def comp_scores(Yt, Yp):
    """Pooled R2 and mean per-component Pearson rho of predicted against true compositions."""
    Yt = np.asarray(Yt, np.float64); Yp = np.asarray(Yp, np.float64)
    mt = Yt.mean(0); sst = float(((Yt - mt) ** 2).sum()); sse = float(((Yt - Yp) ** 2).sum())
    st, sp = Yt.std(0), Yp.std(0); ok = st > 1e-12
    cov = ((Yt - mt) * (Yp - Yp.mean(0))).mean(0)
    r = np.where(sp > 1e-12, cov / np.maximum(st * sp, 1e-300), 0.0)[ok]
    return dict(R2=1.0 - sse / sst if sst > 0 else float("nan"), rho=float(r.mean()) if len(r) else float("nan"),
                rho2=float((r ** 2).mean()) if len(r) else float("nan"), n_comp=int(ok.sum()))


def block_scores(Yt, Yp, blocks):
    """comp_scores per target block."""
    return [comp_scores(Yt[:, b], Yp[:, b]) for b in blocks]


class Geo:
    """Neighbourhood geometry in um: one 3-D group for continuous data, or in-plane coordinates grouped by section."""
    def __init__(self, Pum, groups=None, R=None):
        Pum = np.asarray(Pum, np.float64)
        if groups is None:
            self.C = np.ascontiguousarray(Pum); self.g = np.zeros(len(Pum), np.int64)
        else:
            Q = Pum @ (np.eye(Pum.shape[1]) if R is None else np.asarray(R))
            self.C = np.ascontiguousarray(Q[:, :2]); self.g = np.asarray(groups, np.int64)
        self.N = len(Pum); self.order = np.argsort(self.g, kind="stable"); gs = self.g[self.order]
        self.ug, st = np.unique(gs, return_index=True); self.bounds = np.r_[st, len(gs)]

    def groups(self):
        for i in range(len(self.ug)):
            yield self.ug[i], self.order[self.bounds[i]:self.bounds[i + 1]]


def knn_sets(geo, k, q):
    """(len(q), k) ids of each query's k nearest other cells within its group, sorted by distance; -1 pads small groups."""
    out = np.full((len(q), k), -1, np.int64); pos = np.full(geo.N, -1, np.int64); pos[q] = np.arange(len(q))
    for _, idx in geo.groups():
        ql = np.flatnonzero(pos[idx] >= 0)
        if len(ql) == 0 or len(idx) < 2:
            continue
        kk = min(k, len(idx) - 1)
        nb = knn_excluding_self(geo.C[idx], kk, cKDTree(geo.C[idx]), query=ql)
        out[pos[idx[ql]], :kk] = idx[nb]
    return out


def fit_readers(F, Y, va, tests, blocks):
    """Fit the ridge and MLP readers and pick, per target block, the one with the higher held-out R2."""
    t0 = time.time(); g = Gene(F); Xs = g(F); Ts = {k: g(v) for k, v in tests.items()}
    P(f"  readers: {len(Xs):,} training cells ({int(va.sum()):,} held out for alpha / early stopping), {Xs.shape[1]} genes, {Y.shape[1]} "
      f"target columns in {len(blocks)} blocks")
    rr, ri = ridge(Xs, Y, va, Ts, blocks); P(f"  RIDGE alphas {ri['alpha_counts']} | val R2 per block {[round(v['R2'], 3) for v in ri['val']]} ({time.time() - t0:.0f} s)")
    t1 = time.time(); mr, mi = mlp(Xs, Y, va, Ts, blocks)
    P(f"  MLP {mi['hidden']} best epochs {mi['best_epoch']} (phase 1 ran {mi['epochs_phase1']}, refit {mi['epochs_refit']}) | val R2 per "
      f"block {[round(v['R2'], 3) for v in mi['val']]} ({time.time() - t1:.0f} s)")
    pick = ["mlp" if mi["val"][j]["R2"] >= ri["val"][j]["R2"] else "ridge" for j in range(len(blocks))]
    mi.pop("val_curve", None)
    return {"ridge": rr, "mlp": mr}, dict(ridge=ri, mlp=mi, pick_by_val=pick)


def section_geo(pos_um, sec):
    """In-plane geometry within sections, in the fitted section frame."""
    R, _ = section_frame_fit(pos_um, sec)
    return Geo(pos_um, sec, R), R


def moran_all(F, nbr, chunk_genes=64, log=P):
    """Moran's I of every column of F on a row-normalised k-NN graph (nbr: neighbour ids, -1 = missing)."""
    N, G = F.shape
    ok = nbr >= 0
    cnt = ok.sum(1)
    has = cnt > 0
    nb = np.where(ok, nbr, 0)
    out = np.zeros(G)
    for g0 in range(0, G, chunk_genes):
        Z = np.asarray(F[:, g0:g0 + chunk_genes], dtype=np.float64)
        Z = Z - Z.mean(0)
        lag = np.zeros_like(Z)
        for c in range(nbr.shape[1]):
            lag += np.where(ok[:, c:c + 1], Z[nb[:, c]], 0.0)
        lag[has] /= cnt[has, None]
        num = (Z[has] * lag[has]).sum(0)
        den = (Z[has] ** 2).sum(0)
        out[g0:g0 + chunk_genes] = np.where(den > 0, num / np.maximum(den, 1e-300), 0.0)
        if log and g0 % (chunk_genes * 4) == 0:
            log(f"    Moran genes {g0}-{min(G, g0 + chunk_genes)} of {G}")
    return out


class K32:
    """Soft molecular representation: standardiser -> PCA-30 -> diagonal Gaussian mixture with 32 components; p = posterior."""
    def __init__(self, keep_cols):
        self.keep = np.asarray(keep_cols, np.int64)

    def fit(self, F_fit, seed=0):
        """Fit the standardiser, PCA and mixture on the given training rows."""
        from sklearn.decomposition import PCA
        from sklearn.mixture import GaussianMixture
        Fk = np.asarray(F_fit[:, self.keep], dtype=np.float32)
        self.gene = Gene(Fk)
        Zs = self.gene(Fk)
        self.pca = PCA(N_PC, svd_solver="randomized", random_state=seed).fit(Zs)
        Y = self.pca.transform(Zs)
        self.gmm = GaussianMixture(K_COMP, covariance_type="diag", init_params="kmeans", n_init=1, max_iter=200, tol=1e-3, reg_covar=1e-6,
                                   random_state=seed).fit(Y)
        self.info = dict(n_fit=int(len(Fk)), n_genes=int(len(self.keep)), pca_var_explained=float(self.pca.explained_variance_ratio_.sum()),
                         gmm_converged=bool(self.gmm.converged_), gmm_n_iter=int(self.gmm.n_iter_), gmm_lower_bound=float(self.gmm.lower_bound_),
                         weights=self.gmm.weights_.tolist())
        return self

    def transform(self, F, chunk=200_000):
        """Posterior of every row (chunked)."""
        out = np.empty((F.shape[0], K_COMP), np.float32)
        for s in range(0, F.shape[0], chunk):
            Fk = np.asarray(F[s:s + chunk][:, self.keep], dtype=np.float32)
            out[s:s + chunk] = self.gmm.predict_proba(self.pca.transform(self.gene(Fk))).astype(np.float32)
        return out

    def save(self, path):
        """Save the fitted arrays; returns the file md5."""
        np.savez(path, keep=self.keep, gene_mu=self.gene.mu, gene_sd=self.gene.sd, pca_components=self.pca.components_, pca_mean=self.pca.mean_,
                 pca_explained_variance=self.pca.explained_variance_, gmm_weights=self.gmm.weights_, gmm_means=self.gmm.means_,
                 gmm_covariances=self.gmm.covariances_, gmm_precisions_cholesky=self.gmm.precisions_cholesky_, info=json.dumps(self.info))
        return md5(path)


def simplex(Q):
    """Clip at 0 and renormalise rows; an all-zero row becomes uniform."""
    Q = np.maximum(np.asarray(Q, np.float64), 0.0)
    s = Q.sum(1, keepdims=True)
    return np.where(s > 0, Q / np.where(s > 0, s, 1.0), 1.0 / Q.shape[1])


def env_targets(geo, p, q_rows, rs=RS):
    """Mean posterior over each query cell's r nearest other cells of its group, for every r in rs; also the realised radii and short rows."""
    nb = knn_sets(geo, max(rs), q_rows)
    out, rad, short = [], {}, {}
    for r in rs:
        sub = nb[:, :r]
        ok = sub >= 0
        cnt = ok.sum(1)
        acc = np.zeros((len(q_rows), p.shape[1]), np.float64)
        for c in range(r):
            m = ok[:, c]
            acc[m] += p[sub[m, c]]
        out.append((acc / np.maximum(cnt, 1)[:, None]).astype(np.float32))
        last = sub[np.arange(len(sub)), np.maximum(cnt - 1, 0)]
        d = np.linalg.norm(geo.C[last] - geo.C[q_rows], axis=1)
        rad[r] = dict(median_um=float(np.median(d[cnt > 0])), p90_um=float(np.percentile(d[cnt > 0], 90)))
        short[r] = int((cnt < r).sum())
    return out, rad, short
