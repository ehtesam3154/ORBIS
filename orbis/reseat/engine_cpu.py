"""CPU niche engine of the reseating step.

The sites (sampled positions) stay fixed; only which cell sits at which site changes, through exact two-cell swaps with Metropolis
acceptance on the total energy change. The energy has three terms: a Huber niche term at two neighbourhood scales (the mean soft
component of a site's nearest other sites against the reader's prediction for the cell placed there), a unary anchor to each cell's
one-call best guess, and a penalty on the drift of the soft components' first and second spatial moments. A swap is refused when
either cell would end farther than the cap from its original site. The GPU engine (engine_gpu) mirrors this module's graphs, caches
and arithmetic."""
import time

import numba as nb
import numpy as np
from scipy.spatial import cKDTree


def same_multiset(A, B):
    """True when the rows of A and B are the same multiset (sorted rows exactly equal)."""
    oa = np.lexsort(A.T[::-1])
    ob = np.lexsort(B.T[::-1])
    return bool(np.array_equal(A[oa], B[ob]))


def knn_nonself(pos, k, workers=8):
    """(N, k) indices and distances of the k nearest other sites of every site, sorted by distance (self removed by index)."""
    tree = cKDTree(pos)
    n = len(pos)
    d, idx = tree.query(pos, k=k + 1, workers=workers)
    self_ = idx == np.arange(n)[:, None]
    has = self_.any(1)
    drop = np.where(has[:, None], self_, np.zeros_like(self_))
    drop[~has, -1] = True
    keep = ~drop
    return idx[keep].reshape(n, k).astype(np.int64), d[keep].reshape(n, k)


def in_neighbours(nbr, n):
    """CSR of the reverse neighbour relation: for every site b, the sites v whose neighbour list holds b."""
    v = np.repeat(np.arange(n, dtype=np.int64), nbr.shape[1])
    b = nbr.ravel()
    o = np.argsort(b, kind="stable")
    indptr = np.zeros(n + 1, np.int64)
    np.add.at(indptr, b + 1, 1)
    return np.cumsum(indptr), v[o].astype(np.int64)


def proposal_graph(pos, k=30, workers=8):
    """Undirected union of every site's k nearest other sites, as CSR (both directions, unique edges)."""
    idx, _ = knn_nonself(pos, k, workers)
    n = len(pos)
    a = np.repeat(np.arange(n, dtype=np.int64), k)
    b = idx.ravel()
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = np.unique(lo * n + hi)
    ea, eb = key // n, key % n
    src = np.concatenate([ea, eb])
    dst = np.concatenate([eb, ea])
    o = np.argsort(src, kind="stable")
    indptr = np.zeros(n + 1, np.int64)
    np.add.at(indptr, src + 1, 1)
    return np.cumsum(indptr), dst[o].astype(np.int64)


def restrict_active(indptr, indices, active):
    """Proposal CSR restricted to edges whose two endpoints are active."""
    n = len(indptr) - 1
    src = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    keep = active[src] & active[indices]
    s2, d2 = src[keep], indices[keep]
    ip = np.zeros(n + 1, np.int64)
    np.add.at(ip, s2 + 1, 1)
    return np.cumsum(ip), d2.astype(np.int64)


@nb.njit(cache=True)
def _huber(z, delta):
    """Huber loss with threshold delta."""
    az = abs(z)
    return 0.5 * z * z if az <= delta else delta * (az - 0.5 * delta)


@nb.njit(cache=True)
def _site_energy(mrow, qrow, isig, delta):
    """Niche energy of one site: sum over components of the Huber loss of the scaled residual."""
    e = 0.0
    for k in range(mrow.shape[0]):
        e += _huber((mrow[k] - qrow[k]) * isig[k], delta)
    return e


@nb.njit(cache=True)
def env_all(cell_at, p, nbr):
    """Mean soft component of each site's neighbour list for the assignment cell_at."""
    n, r = nbr.shape
    K = p.shape[1]
    m = np.zeros((n, K))
    for a in range(n):
        for j in range(r):
            c = cell_at[nbr[a, j]]
            for k in range(K):
                m[a, k] += p[c, k]
        for k in range(K):
            m[a, k] /= r
    return m


@nb.njit(cache=True)
def site_energies(cell_at, m, q, isig, delta):
    """Niche energy of every site."""
    n = m.shape[0]
    e = np.zeros(n)
    for a in range(n):
        e[a] = _site_energy(m[a], q[cell_at[a]], isig, delta)
    return e


@nb.njit(cache=True, parallel=True)
def env_all_par(cell_at, p, nbr):
    """env_all with sites in parallel (same per-site summation order, bit-identical)."""
    n, r = nbr.shape
    K = p.shape[1]
    m = np.zeros((n, K))
    for a in nb.prange(n):
        for j in range(r):
            c = cell_at[nbr[a, j]]
            for k in range(K):
                m[a, k] += p[c, k]
        for k in range(K):
            m[a, k] /= r
    return m


@nb.njit(cache=True, parallel=True)
def site_energies_par(cell_at, m, q, isig, delta):
    """site_energies with sites in parallel (bit-identical)."""
    n = m.shape[0]
    e = np.zeros(n)
    for a in nb.prange(n):
        e[a] = _site_energy(m[a], q[cell_at[a]], isig, delta)
    return e


def moments_state(cell_at, p, S):
    """First / second moment sums and masses of the soft components for the assignment cell_at."""
    pc = p[cell_at]
    return pc.T @ S, pc.T @ (S * S).sum(1), p.sum(0)


def moments_of(SX, SXX, M):
    """Means and spreads of the soft components from their moment sums."""
    mu = SX / M[:, None]
    s = np.sqrt(np.maximum(SXX / M - (mu * mu).sum(1), 0.0))
    return mu, s


def moments_penalty(SX, SXX, M, mu0, s0):
    """Moment penalty: drift of each component's mean and spread from its starting value, weighted by its mass."""
    mu, s = moments_of(SX, SXX, M)
    return float((3.0 * M * ((mu - mu0) ** 2).sum(1) / (2.0 * s0 ** 2) + M * (s - s0) ** 2 / s0 ** 2).sum())


@nb.njit(cache=True)
def _moments_pen_k(SXk, SXXk, Mk, mu0k, s0k):
    """Moment penalty of one component."""
    mx = SXk[0] / Mk
    my = SXk[1] / Mk
    mz = SXk[2] / Mk
    v = SXXk / Mk - (mx * mx + my * my + mz * mz)
    s = np.sqrt(v) if v > 0.0 else 0.0
    dmu = (mx - mu0k[0]) ** 2 + (my - mu0k[1]) ** 2 + (mz - mu0k[2]) ** 2
    return 3.0 * Mk * dmu / (2.0 * s0k * s0k) + Mk * (s - s0k) ** 2 / (s0k * s0k)


def total_energy(cell_at, S, p, G, inv2tau2, scales, delta, mu0, s0, use_niche=True, use_unary=True, use_mom=True):
    """Exact energy from scratch (no caches): dict with the niche terms per scale, the unary term, the moment term and the total."""
    out = dict(niche=[], unary=0.0, moments=0.0)
    if use_niche:
        for q, isig, nbr in scales:
            m = env_all(cell_at, p, nbr)
            out["niche"].append(float(site_energies(cell_at, m, q, isig, delta).sum()))
    if use_unary:
        out["unary"] = float(((S - G[cell_at]) ** 2).sum() * inv2tau2)
    if use_mom:
        SX, SXX, M = moments_state(cell_at, p, S)
        out["moments"] = moments_penalty(SX, SXX, M, mu0, s0)
    out["total"] = float(sum(out["niche"]) + out["unary"] + out["moments"])
    return out


@nb.njit(cache=True)
def delta_niche(a, b, cell_at, p, q, isig, delta, m, e, nbr_r, in_ptr, in_idx, stamp, tag, buf_site, buf_m, buf_e):
    """Niche energy change of swapping the cells at sites a and b at one scale; new caches of the affected sites go to the buffers."""
    K = p.shape[1]
    r = nbr_r.shape[1]
    ca = cell_at[a]
    cb = cell_at[b]
    n_aff = 0
    dE = 0.0
    for pass_ in range(3):
        if pass_ == 0:
            lo = in_ptr[a]
            hi = in_ptr[a + 1]
        elif pass_ == 1:
            lo = in_ptr[b]
            hi = in_ptr[b + 1]
        else:
            lo = 0
            hi = 2
        for t in range(lo, hi):
            if pass_ == 0 or pass_ == 1:
                v = in_idx[t]
            elif t == 0:
                v = a
            else:
                v = b
            if stamp[v] == tag:
                continue
            stamp[v] = tag
            wa = 0.0
            wb = 0.0
            for j in range(r):
                u = nbr_r[v, j]
                if u == a:
                    wa = 1.0
                elif u == b:
                    wb = 1.0
            coef = (wa - wb) / r
            if v == a:
                cv = cb
            elif v == b:
                cv = ca
            else:
                cv = cell_at[v]
            for k in range(K):
                buf_m[n_aff, k] = m[v, k] + coef * (p[cb, k] - p[ca, k])
            ev = _site_energy(buf_m[n_aff], q[cv], isig, delta)
            buf_e[n_aff] = ev
            buf_site[n_aff] = v
            dE += ev - e[v]
            n_aff += 1
    return dE, n_aff


@nb.njit(cache=True)
def delta_total(a, b, cell_at, S, p, G, inv2tau2, use_niche, use_unary, use_mom,
                q1, isig1, m1, e1, nbr1, in1p, in1i, q2, isig2, m2, e2, nbr2, in2p, in2i,
                delta, SX, SXX, M, mu0, s0, stamp1, stamp2, tag, bs1, bm1, be1, bs2, bm2, be2):
    """Total energy change of swapping the cells at sites a and b (nothing applied), with its parts."""
    K = p.shape[1]
    ca = cell_at[a]
    cb = cell_at[b]
    d1 = 0.0
    d2 = 0.0
    n1 = 0
    n2 = 0
    if use_niche:
        d1, n1 = delta_niche(a, b, cell_at, p, q1, isig1, delta, m1, e1, nbr1, in1p, in1i, stamp1, tag, bs1, bm1, be1)
        d2, n2 = delta_niche(a, b, cell_at, p, q2, isig2, delta, m2, e2, nbr2, in2p, in2i, stamp2, tag, bs2, bm2, be2)
    dun = 0.0
    if use_unary:
        uo = 0.0
        un = 0.0
        for x in range(3):
            uo += (S[a, x] - G[ca, x]) ** 2 + (S[b, x] - G[cb, x]) ** 2
            un += (S[b, x] - G[ca, x]) ** 2 + (S[a, x] - G[cb, x]) ** 2
        dun = (un - uo) * inv2tau2
    dmom = 0.0
    if use_mom:
        sa2 = S[a, 0] ** 2 + S[a, 1] ** 2 + S[a, 2] ** 2
        sb2 = S[b, 0] ** 2 + S[b, 1] ** 2 + S[b, 2] ** 2
        tmpX = np.empty(3)
        for k in range(K):
            dpk = p[cb, k] - p[ca, k]
            if dpk == 0.0:
                continue
            old = _moments_pen_k(SX[k], SXX[k], M[k], mu0[k], s0[k])
            for x in range(3):
                tmpX[x] = SX[k, x] + dpk * (S[a, x] - S[b, x])
            new = _moments_pen_k(tmpX, SXX[k] + dpk * (sa2 - sb2), M[k], mu0[k], s0[k])
            dmom += new - old
    return d1 + d2 + dun + dmom, n1, n2, d1, d2, dun, dmom


@nb.njit(cache=True)
def apply_swap(a, b, cell_at, S, p, use_mom, m1, e1, m2, e2, SX, SXX, n1, n2, bs1, bm1, be1, bs2, bm2, be2):
    """Apply an evaluated swap: caches of the affected sites, moment sums, assignment."""
    K = p.shape[1]
    ca = cell_at[a]
    cb = cell_at[b]
    for i in range(n1):
        v = bs1[i]
        for k in range(K):
            m1[v, k] = bm1[i, k]
        e1[v] = be1[i]
    for i in range(n2):
        v = bs2[i]
        for k in range(K):
            m2[v, k] = bm2[i, k]
        e2[v] = be2[i]
    if use_mom:
        sa2 = S[a, 0] ** 2 + S[a, 1] ** 2 + S[a, 2] ** 2
        sb2 = S[b, 0] ** 2 + S[b, 1] ** 2 + S[b, 2] ** 2
        for k in range(K):
            dpk = p[cb, k] - p[ca, k]
            for x in range(3):
                SX[k, x] += dpk * (S[a, x] - S[b, x])
            SXX[k] += dpk * (sa2 - sb2)
    cell_at[a] = cb
    cell_at[b] = ca


@nb.njit(cache=True)
def sweep(cell_at, orig_site, S, p, G, inv2tau2, use_niche, use_unary, use_mom,
          q1, isig1, m1, e1, nbr1, in1p, in1i, q2, isig2, m2, e2, nbr2, in2p, in2i,
          delta, SX, SXX, M, mu0, s0, prop_ptr, prop_idx, order, u_pick, u_acc, T, cap2,
          stamp1, stamp2, tag0, bs1, bm1, be1, bs2, bm2, be2, terms):
    """One sweep over the given site order: one proposal per site, cap check, exact energy change, Metropolis acceptance."""
    n_prop = 0
    n_cap = 0
    n_acc = 0
    n_up = 0
    tag = tag0
    for t in range(order.shape[0]):
        a = order[t]
        deg = prop_ptr[a + 1] - prop_ptr[a]
        if deg == 0:
            continue
        j = int(u_pick[t] * deg)
        if j >= deg:
            j = deg - 1
        b = prop_idx[prop_ptr[a] + j]
        n_prop += 1
        ca = cell_at[a]
        cb = cell_at[b]
        oa = orig_site[ca]
        ob = orig_site[cb]
        da = 0.0
        db = 0.0
        for x in range(3):
            da += (S[b, x] - S[oa, x]) ** 2
            db += (S[a, x] - S[ob, x]) ** 2
        if da > cap2 or db > cap2:
            n_cap += 1
            continue
        tag += 1
        dU, n1, n2, dn1, dn2, dun, dmom = delta_total(a, b, cell_at, S, p, G, inv2tau2, use_niche, use_unary, use_mom,
                                                      q1, isig1, m1, e1, nbr1, in1p, in1i, q2, isig2, m2, e2, nbr2, in2p, in2i,
                                                      delta, SX, SXX, M, mu0, s0, stamp1, stamp2, tag, bs1, bm1, be1, bs2, bm2, be2)
        if dU <= 0.0 or u_acc[t] < np.exp(-dU / T):
            apply_swap(a, b, cell_at, S, p, use_mom, m1, e1, m2, e2, SX, SXX, n1, n2, bs1, bm1, be1, bs2, bm2, be2)
            n_acc += 1
            terms[0] += dn1
            terms[1] += dn2
            terms[2] += dun
            terms[3] += dmom
            if dU > 0.0:
                n_up += 1
                terms[4] += dU
    return n_prop, n_cap, n_acc, n_up, tag


class Engine:
    """Graphs, targets and caches of one reseating run. Cells and sites are indexed 0..n-1; cell i starts at site i."""
    def __init__(self, S, p, G, scales, isigs, qs, active, delta=1.345, T=1.0, cap_um=512.0, k_prop=30, use_niche=True,
                 use_unary=True, use_mom=True, workers=8, log=print, tau_um=None):
        t0 = time.time()
        self.n, self.K = p.shape
        self.S = np.ascontiguousarray(S, dtype=np.float64)
        self.p = np.ascontiguousarray(p, dtype=np.float64)
        self.G = np.ascontiguousarray(G, dtype=np.float64)
        self.delta, self.T, self.cap2 = float(delta), float(T), float(cap_um) ** 2
        self.use = (bool(use_niche), bool(use_unary), bool(use_mom))
        self.active = np.asarray(active, bool)
        self.log = log
        self.scales = []
        kmax = max(scales)
        nbr_all, dist_all = knn_nonself(self.S, kmax, workers)
        self.radii = {}
        for r, isig, q in zip(scales, isigs, qs):
            nbr = np.ascontiguousarray(nbr_all[:, :r])
            ip, ii = in_neighbours(nbr, self.n)
            self.scales.append(dict(r=r, nbr=nbr, in_ptr=ip, in_idx=ii, isig=np.ascontiguousarray(isig, dtype=np.float64),
                                    q=np.ascontiguousarray(q, dtype=np.float64), max_in=int(np.diff(ip).max())))
            self.radii[r] = dict(median_um=float(np.median(dist_all[:, r - 1])), p90_um=float(np.percentile(dist_all[:, r - 1], 90)),
                                 max_in_degree=int(np.diff(ip).max()), mean_in_degree=float(np.diff(ip).mean()))
        pp, pi = proposal_graph(self.S, k_prop, workers)
        self.prop_ptr, self.prop_idx = restrict_active(pp, pi, self.active)
        dega = np.diff(self.prop_ptr)[self.active]
        self.cell_at = np.arange(self.n, dtype=np.int64)
        self.orig_site = np.arange(self.n, dtype=np.int64)
        d0 = ((self.S - self.G) ** 2).sum(1)
        self.tau2 = float(d0.mean() / 3.0) if tau_um is None else float(tau_um) ** 2
        self.inv2tau2 = 1.0 / (2.0 * self.tau2)
        SX, SXX, M = moments_state(self.cell_at, self.p, self.S)
        self.M = M
        self.mu0, self.s0 = moments_of(SX, SXX, M)
        self.SX, self.SXX = SX.copy(), SXX.copy()
        self._recompute_caches()
        self.info = dict(n_sites=self.n, n_active=int(self.active.sum()), n_passive=int((~self.active).sum()), K=self.K, scales=scales,
                         radii_um=self.radii, proposal_k=k_prop, active_degree_median=float(np.median(dega)) if len(dega) else 0.0,
                         active_degree_min=int(dega.min()) if len(dega) else 0, tau_G_um=float(np.sqrt(self.tau2)), tau_fixed=tau_um is not None, T=self.T,
                         delta=self.delta, cap_um=float(cap_um), use_niche=self.use[0], use_unary=self.use[1], use_moments=self.use[2],
                         setup_seconds=round(time.time() - t0, 1))
        log(f"  engine: {self.n:,} sites ({self.info['n_active']:,} active / {self.info['n_passive']:,} passive), K {self.K}; graph radii "
            + ", ".join(f"r{r}: median {v['median_um']:.1f} um (p90 {v['p90_um']:.1f}), in-degree mean {v['mean_in_degree']:.1f} max "
                        f"{v['max_in_degree']}" for r, v in self.radii.items())
            + f"; proposal degree (active) median {self.info['active_degree_median']:.0f} min {self.info['active_degree_min']}; tau_G "
              f"{np.sqrt(self.tau2):.1f} um; T {self.T}; cap {cap_um:.0f} um; terms niche {self.use[0]} unary {self.use[1]} moments {self.use[2]}")

    def _recompute_caches(self):
        """Recompute the neighbourhood means and site energies from scratch."""
        for sc in self.scales:
            sc["m"] = env_all_par(self.cell_at, self.p, sc["nbr"])
            sc["e"] = site_energies_par(self.cell_at, sc["m"], sc["q"], sc["isig"], self.delta)

    def energy(self):
        """Exact energy of the current assignment."""
        return total_energy(self.cell_at, self.S, self.p, self.G, self.inv2tau2, [(s["q"], s["isig"], s["nbr"]) for s in self.scales],
                            self.delta, self.mu0, self.s0, *self.use)

    def run(self, n_sweeps=8, seed=20261019, on_sweep=None):
        """Run n_sweeps sweeps on the CPU; sweep sw draws its randomness from default_rng([seed, sw]). Returns the per-sweep record."""
        s1, s2 = self.scales
        n = self.n
        stamp1 = np.zeros(n, np.int64)
        stamp2 = np.zeros(n, np.int64)
        bs1 = np.zeros(s1["max_in"] * 2 + 2, np.int64)
        bm1 = np.zeros((len(bs1), self.K))
        be1 = np.zeros(len(bs1))
        bs2 = np.zeros(s2["max_in"] * 2 + 2, np.int64)
        bm2 = np.zeros((len(bs2), self.K))
        be2 = np.zeros(len(bs2))
        act = np.flatnonzero(self.active).astype(np.int64)
        E0 = self.energy()
        traj = [dict(sweep=0, **{k: v for k, v in E0.items()})]
        self.log(f"  sweep 0: U {E0['total']:.3f} (niche {[round(x, 3) for x in E0['niche']]}, unary {E0['unary']:.3f}, moments "
                 f"{E0['moments']:.3f})")
        U_run = E0["total"]
        tag = 0
        for sw in range(1, n_sweeps + 1):
            t0 = time.time()
            rng = np.random.default_rng([seed, sw])
            order = rng.permutation(act).astype(np.int64)
            u_pick = rng.random(len(order))
            u_acc = rng.random(len(order))
            terms = np.zeros(5)
            n_prop, n_cap, n_acc, n_up, tag = sweep(
                self.cell_at, self.orig_site, self.S, self.p, self.G, self.inv2tau2, self.use[0], self.use[1], self.use[2],
                s1["q"], s1["isig"], s1["m"], s1["e"], s1["nbr"], s1["in_ptr"], s1["in_idx"],
                s2["q"], s2["isig"], s2["m"], s2["e"], s2["nbr"], s2["in_ptr"], s2["in_idx"],
                self.delta, self.SX, self.SXX, self.M, self.mu0, self.s0, self.prop_ptr, self.prop_idx, order, u_pick, u_acc, self.T,
                self.cap2, stamp1, stamp2, tag, bs1, bm1, be1, bs2, bm2, be2, terms)
            U_run += terms[0] + terms[1] + terms[2] + terms[3]
            niche = []
            for sc in self.scales:
                m_new = env_all_par(self.cell_at, self.p, sc["nbr"])
                assert np.allclose(m_new, sc["m"], atol=1e-9), "m cache drift"
                sc["m"] = m_new
                sc["e"] = site_energies_par(self.cell_at, m_new, sc["q"], sc["isig"], self.delta)
                niche.append(float(sc["e"].sum()))
            SXc, SXXc, Mc = moments_state(self.cell_at, self.p, self.S)
            assert np.allclose(SXc, self.SX, rtol=1e-9, atol=1e-6) and np.allclose(SXXc, self.SXX, rtol=1e-9, atol=1e-3), "moment drift"
            self.SX, self.SXX = SXc, SXXc
            E = dict(niche=niche if self.use[0] else [],
                     unary=float(((self.S - self.G[self.cell_at]) ** 2).sum() * self.inv2tau2) if self.use[1] else 0.0,
                     moments=moments_penalty(SXc, SXXc, Mc, self.mu0, self.s0) if self.use[2] else 0.0)
            E["total"] = float(sum(E["niche"]) + E["unary"] + E["moments"])
            drift = U_run - E["total"]
            assert abs(drift) <= 1e-6 * max(1.0, abs(E["total"])), f"energy drift {drift}"
            U_run = E["total"]
            disp = np.linalg.norm(self.S[self.site_of()] - self.S[self.orig_site], axis=1)
            moved = disp > 0
            rec = dict(sweep=sw, proposals=int(n_prop), cap_rejected=int(n_cap), accepted=int(n_acc), accepted_uphill=int(n_up),
                       uphill_dU_sum=float(terms[4]), d_niche=[float(terms[0]), float(terms[1])], d_unary=float(terms[2]),
                       d_moments=float(terms[3]), drift=float(drift), cells_displaced=int(moved.sum()),
                       disp_um=dict(median_moved=float(np.median(disp[moved])) if moved.any() else 0.0,
                                    p90_moved=float(np.percentile(disp[moved], 90)) if moved.any() else 0.0, max=float(disp.max())),
                       seconds=round(time.time() - t0, 1), **{k: v for k, v in E.items()})
            traj.append(rec)
            self.log(f"  sweep {sw}: proposals {n_prop:,}, cap-rejected {n_cap:,}, accepted {n_acc:,} ({n_up:,} uphill); U {E['total']:.3f} "
                     f"(niche {[round(x, 3) for x in E['niche']]}, unary {E['unary']:.3f}, moments {E['moments']:.3f}); drift {drift:.2e}; "
                     f"displaced {moved.sum():,} cells, median {rec['disp_um']['median_moved']:.1f} um, p90 {rec['disp_um']['p90_moved']:.1f}, "
                     f"max {disp.max():.1f}; {rec['seconds']} s")
            if on_sweep is not None:
                on_sweep(sw, self)
        return traj

    def site_of(self):
        """site_of[c] = the site holding cell c."""
        s = np.empty(self.n, np.int64)
        s[self.cell_at] = np.arange(self.n)
        return s

    def output_layout(self, P_sites):
        """Row c = the position (from P_sites, the sample's rows) of the site now holding cell c."""
        return np.asarray(P_sites)[self.site_of()]
