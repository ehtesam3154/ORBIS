"""Certified reconstruction scorer: alignment of a predicted cell layout onto the true one and every reconstruction metric.

The truth and the prediction share their cells (row i = the same cell). Each layout is aligned by a rigid fit on the
count-weighted fine-type centroids (no scale), optionally size matched per axis, and scored: size, per-type sliced W2,
per-type F-score, type spread and type-centre error, cell-type contacts, niches, Moran's I agreement, pairwise distances,
contact F1, RSSD, per-cell SRC, LDDT, occupancy IoU and per-cell diagnostics.
"""
import hashlib
import json
import os
import platform
import sys
import time
import traceback

import numpy as np
import scipy
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, pdist
from scipy.spatial.transform import Rotation
from scipy.special import expit
from scipy.stats import pearsonr, rankdata, spearmanr

SEED_DIRS, N_DIRS = 0, 64
SEED_PAIRS, N_PAIRS = 1, 2_000_000
SEED_SUB, N_SUB = 2, 10_000
SEED_CORRUPT = 3
MIN_TYPE, MIN_SPREAD_TYPE, SQUEEZE = 20, 200, 0.8
K_CONTACT, K_NICHE, N_NICHES, K_MORAN, N_GENES = 15, 100, 15, 15, 50
CONTACT_PCTS = (0.01, 0.10)
LDDT_TAUS, LDDT_K, LDDT_CUT_U = (0.5, 1.0, 2.0, 4.0), 256, 15.0
LDDT_ALL_MAX, LDDT_QUERY_N = 250_000, 100_000
LDDT_PERFECT_SMOOTH = 0.25 * float(sum(expit(t) for t in LDDT_TAUS))
QUANTUM_UM = 1e-3
IOU_MIN_N, IOU_MIN_SHARE, IOU_MARGIN_UM = 200, 0.02, 1000.0
IOU_ORIGIN = "half_cell"
IOU_SLABS = "midpoint"
TOL_RANK, TOL_NULL = 1e-12, 1e-9
SD_FLAT = 1e-9
FLAT = 0.01
FOLD_GRID_N, FOLD_PHI_N, FOLD_N_START, FOLD_SEP_DEG, FOLD_CHUNK = 4000, 3600, 6, 5.0, 64
FOLD_PSI_N, FOLD_N_POLISH, FOLD_MAXFEV = 180, 2, 4000
FOLD_POLISH_CYCLES, FOLD_POLISH_DIREC, FOLD_POLISH_SIMPLEX, FOLD_POLISH_RTOL, FOLD_POLISH_ATOL = 12, 1e-3, 1e-4, 1e-12, 1e-15
FOLD_BAND = 2.0


def _workers():
    v = os.environ.get("ORBIS_WORKERS")
    return int(v) if v else max(1, len(os.sched_getaffinity(0)))
WORKERS = _workers()


def log(*a):
    print(*a, flush=True)


class RefusedInput(ValueError):
    """Raised for an input the scorer refuses to score; the message says why."""


def refuse(msg):
    """Raise RefusedInput with the reason."""
    raise RefusedInput("REFUSED: " + msg)


def refusal_reason(e):
    m = str(e)
    return m[len("REFUSED: "):] if m.startswith("REFUSED: ") else m


def md5_file(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def md5_codes(x):
    """md5 of integer codes as int64 bytes."""
    return hashlib.md5(np.ascontiguousarray(np.asarray(x, dtype=np.int64)).tobytes()).hexdigest()


def md5_array(a):
    """md5 of an array's dtype, shape and bytes."""
    a = np.ascontiguousarray(a)
    return hashlib.md5(str((a.dtype.str, a.shape)).encode() + a.tobytes()).hexdigest()


def environment():
    """Versions, threads and the scorer file of this run."""
    import sklearn
    try:
        import scanpy
        scv = scanpy.__version__
    except Exception:
        scv = None
    return dict(script=os.path.abspath(__file__), script_md5=md5_file(os.path.abspath(__file__)), python=sys.executable,
                python_version=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, scanpy=scv,
                cpu_affinity=sorted(os.sched_getaffinity(0)), kdtree_workers=WORKERS,
                threads={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")})


def as_coords(A, name, N=None, D=None):
    """Validated float64 coordinates (N, 2 or 3)."""
    A = np.asarray(A)
    if A.dtype.kind not in "fiu":
        refuse(f"{name}: dtype {A.dtype} is not numeric")
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[1] not in (2, 3):
        refuse(f"{name}: shape {A.shape}; expected (N, 2) or (N, 3)")
    if N is not None and A.shape[0] != N:
        refuse(f"{name}: {A.shape[0]} cells, the truth has {N} (different cell sets)")
    if D is not None and A.shape[1] != D:
        refuse(f"{name}: {A.shape[1]}-D coordinates, the truth is {D}-D")
    bad = ~np.isfinite(A).all(1)
    if bad.any():
        refuse(f"{name}: {int(bad.sum())} cells with NaN / inf coordinates (first rows {np.flatnonzero(bad)[:5].tolist()})")
    if A.shape[0] < 2 * MIN_TYPE:
        refuse(f"{name}: only {A.shape[0]} cells")
    return A


def as_codes(c, name, N):
    """Validated integer codes remapped monotonically to 0..K-1 (-1 = excluded from every type metric)."""
    if c is None:
        refuse(f"{name}: labels missing")
    c = np.asarray(c)
    if c.ndim != 1 or len(c) != N:
        refuse(f"{name}: labels shape {c.shape}, expected ({N},)")
    if c.dtype.kind not in "iu":
        refuse(f"{name}: labels must be integer codes (got {c.dtype}); encode names with np.unique(..., return_inverse=True)")
    c = c.astype(np.int64)
    if c.min() < -1:
        refuse(f"{name}: codes below -1 (-1 = excluded)")
    v = c >= 0
    if not v.any():
        refuse(f"{name}: fewer than 2 types with >= {MIN_TYPE} cells")
    out = c.copy()
    out[v] = np.unique(c[v], return_inverse=True)[1].ravel()
    if int((np.bincount(out[v]) >= MIN_TYPE).sum()) < 2:
        refuse(f"{name}: fewer than 2 types with >= {MIN_TYPE} cells")
    return out


def check_scale(X, Y, name, lo=0.1, hi=10.0):
    """Unit guard: the layout's RMS radius must lie within [lo, hi] x the truth's."""
    r = float(np.sqrt(((Y - Y.mean(0)) ** 2).sum(1).mean() / ((X - X.mean(0)) ** 2).sum(1).mean()))
    if not lo <= r <= hi:
        refuse(f"{name}: RMS radius {r:.4g} x the truth's - unit mismatch? (allowed {lo} .. {hi})")
    return r


def knn_excluding_self(Z, k, tree=None, query=None, chunk=400_000):
    """Indices of the k nearest other cells of each query cell."""
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


def spacing(X, n=50000, seed=0, tree=None):
    """Median distance from a cell to its nearest other position (the dataset spacing)."""
    q = X if len(X) <= n else X[np.random.default_rng(seed).choice(len(X), n, replace=False)]
    d, _ = (cKDTree(X) if tree is None else tree).query(q, k=8, workers=WORKERS)
    d = np.where(d[:, 1:] > 0, d[:, 1:], np.inf).min(1)
    return float(np.median(d[np.isfinite(d)]))


def knn_boundary_ties(Z, k, tree=None, n=50000, seed=0):
    """Share of cells whose k-th and (k+1)-th nearest other cells are equidistant."""
    q = np.arange(len(Z)) if len(Z) <= n else np.sort(np.random.default_rng(seed).choice(len(Z), n, replace=False))
    d, _ = (cKDTree(Z) if tree is None else tree).query(Z[q], k=k + 2, workers=WORKERS)
    return float(np.mean(d[:, k] == d[:, k + 1]))


def _type_sums(Z, codes, K, valid):
    S = np.zeros((K, Z.shape[1]))
    np.add.at(S, codes[valid], Z[valid])
    return S


def _orth_procrustes(Hm, cells):
    """Orthogonal Procrustes rotation (reflection allowed) with a deterministic completion for rank-deficient input."""
    U, s, Vt = np.linalg.svd(Hm)
    d = len(s)
    r = int(np.sum(s > TOL_RANK * s[0])) if s[0] > 0 else 0
    info = dict(singular_values=[float(v) for v in s], condition=float(s[-1] / s[0]) if s[0] > 0 else 0.0, rank=r)
    R = U @ Vt
    if r < d:
        tot = float((cells ** 2).sum())
        rel = float(np.sqrt(((cells @ U[:, r:]) ** 2).sum() / tot)) if tot > 0 else 0.0
        info.update(prediction_degenerate=True, null_spread_rel=rel)
        if rel > TOL_NULL:
            return None, info
        if np.linalg.det(R) < 0:
            U = U.copy()
            U[:, -1] *= -1.0
            R = U @ Vt
        info["completion"] = "det(R) = +1 (the undetermined directions carry no cells: the aligned layout is unique)"
    info["reflection"] = bool(np.linalg.det(R) < 0)
    return R, info


def _truth_centroid_check(cx, mx, w, what="truth"):
    """Refuse when the truth's weighted type centroids are degenerate."""
    s = np.linalg.svd((cx - mx) * np.sqrt(w)[:, None], compute_uv=False)
    cond = float(s[-1] / s[0]) if s[0] > 0 else 0.0
    if cond < TOL_RANK:
        refuse(f"the {what}'s type centroids are degenerate (all on a point / line / plane; singular values {s.tolist()}): no alignment "
               f"is determined")
    return cond


def fit_type_centroid(Y, Xt, codes):
    """Rigid fit (rotation, reflection, translation; no scale) on the count-weighted fine-type centroids."""
    valid = codes >= 0
    K = int(codes[valid].max()) + 1
    NF = np.bincount(codes[valid], minlength=K)
    cy, cx = _type_sums(Y, codes, K, valid), _type_sums(Xt, codes, K, valid)
    ok = NF > 0
    cy, cx, w = cy[ok] / NF[ok, None], cx[ok] / NF[ok, None], NF[ok] / int(valid.sum())
    if valid.all():
        my, mx = Y.mean(0), Xt.mean(0)
    else:
        my, mx = Y[valid].mean(0), Xt[valid].mean(0)
    tcond = _truth_centroid_check(cx, mx, w)
    R, info = _orth_procrustes(((cy - my) * w[:, None]).T @ (cx - mx), Y - my)
    if R is None:
        refuse(f"the prediction's type centroids span only {info['rank']} of {Y.shape[1]} dimensions and its cells spread outside "
               f"them (relative spread {info['null_spread_rel']:.2e} > {TOL_NULL}): the alignment is not determined")
    info.update(fit="type_centroid", dof="free", n_types=int(ok.sum()))
    info["truth_condition"] = tcond
    return R, my, mx, info


def fit_cell_free(Y, X):
    """Rigid fit on the cells themselves (diagnostic)."""
    mY, mX = Y.mean(0), X.mean(0)
    U, s, Vt = np.linalg.svd((Y - mY).T @ (X - mX))
    R = U @ Vt
    return (Y - mY) @ R + mX, dict(fit="cell", dof="free", reflection=bool(np.linalg.det(R) < 0), singular_values=[float(v) for v in s])


def fold_mid_low(m):
    """Fold of the first coordinate at the midline m: x -> m - |x - m|."""
    def f(Y):
        Y = Y.copy()
        Y[:, 0] = m - np.abs(Y[:, 0] - m)
        return Y
    return f


def fold_lr_own(axis, centre="median"):
    """Fold of one axis about each layout's own median (or mean)."""
    stat = np.median if centre == "median" else np.mean

    def f(Y):
        Y = Y.copy()
        Y[:, axis] = np.abs(Y[:, axis] - stat(Y[:, axis]))
        return Y
    return f


def make_fitter(Xt, codes, fold=None):
    """The type-centroid fit onto Xt (plus an optional fold) as one callable: Y -> (aligned Y, info)."""
    def fit(Y):
        R, my, mx, info = fit_type_centroid(Y, Xt, codes)
        Ya = (Y - my) @ R + mx
        if fold is not None:
            Ya = fold(Ya)
        return Ya, info
    return fit


def size_match(Ya, X, fit, flat=False):
    """Each axis rescaled about its mean to the truth's SD, then fitted again; None for a flat layout."""
    if flat:
        return None
    c0 = Ya.mean(0)
    sd = Ya.std(0)
    if (sd <= SD_FLAT * X.std(0)).any():
        return None
    return fit(c0 + (Ya - c0) * (X.std(0) / sd))[0]


def _fib_hemisphere(n, axis):
    """Equal-area Fibonacci lattice of unit vectors on a half sphere."""
    i = np.arange(n, dtype=np.float64) + 0.5
    h = 1.0 - i / n
    r = np.sqrt(np.maximum(0.0, 1.0 - h * h))
    ang = i * np.pi * (3.0 - np.sqrt(5.0))
    o = [b for b in range(3) if b != axis]
    G = np.zeros((n, 3))
    G[:, axis], G[:, o[0]], G[:, o[1]] = h, r * np.cos(ang), r * np.sin(ang)
    return G


def _basis_perp(g):
    """Orthonormal basis of the plane orthogonal to a unit vector."""
    E = []
    for b in sorted(np.argsort(np.abs(g), kind="stable")[:2].tolist()):
        e = np.zeros(3)
        e[b] = 1.0
        for v in [g] + E:
            e = e - (e @ v) * v
        E.append(e / np.linalg.norm(e))
    return np.stack(E, axis=1)


def flat_layout(own):
    """True when a layout's smallest own principal size is below FLAT x its largest."""
    own = np.asarray(own, dtype=np.float64)
    return bool(own.max() <= 0 or own.min() < FLAT * own.max())


class FoldObjective:

    """Folded type-centroid error of the size-matched layout, as a function of the output rotation."""
    def __init__(self, Yh, Xt, codes, axis, centre):
        if Yh.shape[1] != 3:
            refuse("the folded fit needs 3-D coordinates")
        self.axis = axis
        self.stat = np.median if centre == "median" else np.mean
        self.o = [b for b in range(3) if b != axis]
        self.valid = codes >= 0
        cv = np.unique(codes[self.valid], return_inverse=True)[1].ravel()
        n = np.bincount(cv).astype(np.float64)
        self.w = n / n.sum()
        self.S = sp.csr_matrix((1.0 / n[cv], (cv, np.flatnonzero(self.valid))), shape=(len(n), len(codes)))
        self.Yh = np.ascontiguousarray(Yh, dtype=np.float64)
        self.cy = np.asarray(self.S @ self.Yh)
        cx = np.asarray(self.S @ Xt)
        self.T = cx[:, self.o] - self.w @ cx[:, self.o]
        xa = Xt[:, axis] - self.stat(Xt[:, axis])
        self.tf = np.asarray(self.S @ np.abs(xa))
        self.sx = Xt.std(0)
        self.sx[axis] = float(np.sqrt(np.mean(xa ** 2)))
        self.CY = np.cov(self.Yh.T, bias=True)
        self.s0 = float(np.sqrt(Xt.var(0).sum() / np.trace(self.CY)))
        own = own_sizes(self.Yh, Xt)
        self.k_lo = 1.0 / (FOLD_BAND * float(own.max()))
        self.k_hi = FOLD_BAND / float(own.min()) if own.min() > 0 else np.inf
        self.Jscale = float(self.w @ self.tf ** 2 + (self.w[:, None] * self.T ** 2).sum())
        self.nfev = 0

    def _k_phi(self, sl, b):
        k = np.where(sl > 0, np.clip(self.sx[b] / np.where(sl > 0, sl, 1.0), self.k_lo, self.k_hi), 0.0)
        return k, np.minimum(1.0, (self.s0 * sl / self.sx[b] / FLAT) ** 2)

    def fold_terms(self, G):
        out = []
        for s in range(0, len(G), FOLD_CHUNK):
            Gc = np.atleast_2d(G[s:s + FOLD_CHUNK])
            D = Gc @ self.Yh.T
            D = D - self.stat(D, axis=1)[:, None]
            M = np.asarray(self.S @ np.abs(D).T)
            k, phi = self._k_phi(np.sqrt(np.mean(D ** 2, axis=1)), self.axis)
            res = self.w @ ((self.tf[:, None] - M * k) ** 2)
            out.append(np.c_[phi * res, res, k, phi])
        return np.vstack(out)

    def lin_terms(self, Ucols, j):
        Ucols = np.atleast_2d(Ucols)
        c = self.cy @ Ucols.T
        c = c - self.w @ c
        k, phi = self._k_phi(np.sqrt(np.maximum(np.einsum("mi,ij,mj->m", Ucols, self.CY, Ucols), 0.0)), self.o[j])
        res = self.w @ ((self.T[:, [j]] - c * k) ** 2)
        return np.c_[phi * res, res, k, phi]

    def parts(self, R):
        return (self.fold_terms(R[:, self.axis])[0], self.lin_terms(R[:, self.o[0]], 0)[0], self.lin_terms(R[:, self.o[1]], 1)[0])

    def J_R(self, R):
        self.nfev += 1
        return float(sum(p[0] for p in self.parts(R)))

    def R_of(self, g, u0, u1):
        R = np.zeros((3, 3))
        R[:, self.axis], R[:, self.o[0]], R[:, self.o[1]] = g, u0, u1
        return R

    def inner_psi(self, g):
        E = _basis_perp(g)
        psi = np.arange(FOLD_PSI_N) * (np.pi / FOLD_PSI_N)
        c, s = np.cos(psi)[:, None], np.sin(psi)[:, None]
        U0, U1 = c * E[:, 0] + s * E[:, 1], -s * E[:, 0] + c * E[:, 1]
        t0 = {sg: self.lin_terms(sg * U0, 0)[:, 0] for sg in (1.0, -1.0)}
        t1 = {sg: self.lin_terms(sg * U1, 1)[:, 0] for sg in (1.0, -1.0)}
        best = None
        for s0 in (1.0, -1.0):
            for s1 in (1.0, -1.0):
                jj = t0[s0] + t1[s1]
                i = int(np.argmin(jj))
                if best is None or jj[i] < best[0]:
                    best = (float(jj[i]), s0 * U0[i], s1 * U1[i])
        return best


def _pick_starts(G, Jg, n, sep_deg, first=0):
    """Indices of the best separated grid directions (plus the start fit's axis)."""
    cs = np.cos(np.radians(sep_deg))
    keep = []
    for k in np.argsort(Jg, kind="stable"):
        if len(keep) >= n:
            break
        if all(abs(float(G[k] @ G[j])) < cs for j in keep):
            keep.append(int(k))
    return keep + ([first] if first not in keep else [])


def _nelder_mead(f, h, xatol, fatol):
    sim = np.vstack([np.zeros(3), h * np.eye(3)])
    return minimize(f, np.zeros(3), method="Nelder-Mead", options=dict(initial_simplex=sim, xatol=xatol, fatol=fatol, maxiter=FOLD_MAXFEV,
                                                                       maxfev=FOLD_MAXFEV))


def _rot(x):
    return Rotation.from_rotvec(x).as_matrix()


def _polish(ob, Rk, jk):
    """Final polish of a folded fit: cycles of Powell and Nelder-Mead until a cycle gains nothing measurable."""
    gains, nf = [], 0
    for _ in range(FOLD_POLISH_CYCLES):
        f = lambda x, Rk=Rk: ob.J_R(_rot(x) @ Rk)
        r1 = minimize(f, np.zeros(3), method="Powell", options=dict(direc=FOLD_POLISH_DIREC * np.eye(3), xtol=1e-10, ftol=1e-15,
                                                                     maxiter=FOLD_MAXFEV, maxfev=FOLD_MAXFEV))
        x1, j1 = (np.asarray(r1.x, dtype=np.float64), float(r1.fun)) if float(r1.fun) < jk else (np.zeros(3), jk)
        r2 = minimize(f, x1, method="Nelder-Mead", options=dict(initial_simplex=x1 + np.vstack([np.zeros(3), FOLD_POLISH_SIMPLEX * np.eye(3)]),
                                                                 xatol=1e-13, fatol=1e-16 * ob.Jscale, maxiter=FOLD_MAXFEV, maxfev=FOLD_MAXFEV))
        x2, j2 = (np.asarray(r2.x, dtype=np.float64), float(r2.fun)) if float(r2.fun) < j1 else (x1, j1)
        nf += int(r1.nfev) + int(r2.nfev)
        gain = jk - j2
        gains.append(gain)
        if gain > 0:
            Rk, jk = _rot(x2) @ Rk, j2
        if gain <= FOLD_POLISH_RTOL * jk + FOLD_POLISH_ATOL * ob.Jscale:
            return Rk, jk, True, gains, nf
    return Rk, jk, False, gains, nf


def fit_folded(Y, Xt, codes, axis, centre, um=1.0):
    """Rigid fit minimising the folded objective (grid search, Nelder-Mead, polish); returns the unfolded aligned layout and its record."""
    R0, my, mx, info0 = fit_type_centroid(Y, Xt, codes)
    Yh = (Y - my) @ R0
    ob = FoldObjective(Yh, Xt, codes, axis, centre)
    I3 = np.eye(3)
    j_start = ob.J_R(I3)
    sols = [(j_start, I3, True, 0, "spec-fit frame")]
    polish = []
    ea = np.zeros(3)
    ea[axis] = 1.0
    own = np.linalg.eigh(ob.CY)[1].T
    own = own * np.where(own[:, axis] < 0, -1.0, 1.0)[:, None]
    G = np.vstack([ea, own, _fib_hemisphere(FOLD_GRID_N, axis)])
    F = ob.fold_terms(G)
    inner = [ob.inner_psi(g) for g in G]
    Jg = F[:, 0] + np.array([x[0] for x in inner])
    ob.nfev += len(G)
    step = float(np.sqrt(2.0 * np.pi / FOLD_GRID_N))
    ks = _pick_starts(G, Jg, FOLD_N_START, FOLD_SEP_DEG)
    for k in ks + [k for k in range(4) if k not in ks]:
        Rk = ob.R_of(G[k], inner[k][1], inner[k][2])
        jk = ob.J_R(Rk)
        r = _nelder_mead(lambda x, Rk=Rk: ob.J_R(_rot(x) @ Rk), step, 1e-9, 1e-13 * ob.Jscale)
        if float(r.fun) <= jk:
            Rk, jk = _rot(r.x) @ Rk, float(r.fun)
        sols.append((jk, Rk, bool(r.success), int(r.nfev), "grid" if k >= 4 else ("spec-fit axis" if k == 0 else "own axis")))
    sols.sort(key=lambda s: s[0])
    out = []
    for jk, Rk, okk, nfk, src in sols[:FOLD_N_POLISH]:
        Rk, jk, conv, gains, nfp = _polish(ob, Rk, jk)
        out.append((jk, Rk, conv, nfk + nfp, src + " + polish"))
        polish.append([g * um * um for g in gains])
    sols = sorted(out + sols[FOLD_N_POLISH:], key=lambda s: s[0])
    grid = dict(kind="spec-fit axis + own principal axes + Fibonacci hemisphere, x in-plane angle", n=FOLD_GRID_N + 4, n_psi=FOLD_PSI_N,
                spacing_deg=float(np.degrees(step)), refine="Nelder-Mead over 3 rotation parameters (xatol 1e-9 rad)",
                polish=f"best {FOLD_N_POLISH}: cycles of Powell (initial directions {FOLD_POLISH_DIREC:g} rad) then Nelder-Mead (simplex "
                       f"{FOLD_POLISH_SIMPLEX:g} rad) until one cycle gains <= {FOLD_POLISH_RTOL:g} x J + {FOLD_POLISH_ATOL:g} x the no-signal J "
                       f"(at most {FOLD_POLISH_CYCLES} cycles)")
    jbest, R, ok_best = sols[0][0], sols[0][1].copy(), sols[0][2]
    g = R[:, axis]
    R[:, axis] = g * (1.0 if g[int(np.argmax(np.abs(g)))] >= 0 else -1.0)
    g = R[:, axis]
    Z = Yh @ R + mx
    fold = fold_lr_own(axis, centre)
    Zf = fold(Z)
    pf, p0_, p1_ = ob.parts(R)
    hh = 1e-3
    f0 = ob.J_R(R)
    E3 = np.eye(3) * hh
    H = np.zeros((3, 3))
    for i in range(3):
        H[i, i] = (ob.J_R(_rot(E3[i]) @ R) - 2.0 * f0 + ob.J_R(_rot(-E3[i]) @ R)) / hh ** 2
        for j in range(i + 1, 3):
            H[i, j] = H[j, i] = (ob.J_R(_rot(E3[i] + E3[j]) @ R) - ob.J_R(_rot(E3[i] - E3[j]) @ R) - ob.J_R(_rot(E3[j] - E3[i]) @ R)
                                 + ob.J_R(_rot(-E3[i] - E3[j]) @ R)) / (4.0 * hh ** 2)
    ev = np.linalg.eigvalsh(H)
    cond = float(max(ev[0], 0.0) / ev[-1]) if ev[-1] > 0 else 0.0
    cinfo = dict(condition=cond, condition_rule="smallest / largest eigenvalue of the folded objective's Hessian over the 3 rotation "
                 "parameters at the optimum (central differences, 1e-3 rad)", hessian_eigenvalues_um2_per_rad2=[float(v) * um * um for v in ev])
    Rp = R if np.linalg.det(R) > 0 else R @ np.diag([-1.0 if i == axis else 1.0 for i in range(3)])
    cmp = [s for s in sols[1:] if abs(float(s[1][:, axis] @ g)) < np.cos(np.radians(1.0))]
    ru_diff = None
    if cmp:
        ru_diff = float(np.abs(fold(Yh @ cmp[0][1] + mx) - Zf).max()) * um
    Xf = fold(Xt)
    Sv = ob.S
    j_rigid = float((ob.w * ((np.asarray(Sv @ Zf) - np.asarray(Sv @ Xf)) ** 2).sum(1)).sum())
    info = dict(fit="folded type centroids of the size-matched layout (each output axis scaled to the truth's spread, fold axis: RMS "
                    "about the own " + centre + "; each cell folded about the layout's own " + centre + ", then type means)",
                dof="free", fold_axis=int(axis), fold_centre=centre, flat_rule=f"phi = min(1, (relative spread / {FLAT:g})^2)",
                objective_um2=float(pf[0] + p0_[0] + p1_[0]) * um * um, objective_rms_um=float(np.sqrt(max(pf[0] + p0_[0] + p1_[0], 0.0))) * um,
                objective_scale_um2=ob.Jscale * um * um, objective_optimizer_um2=jbest * um * um, objective_at_spec_fit_um2=j_start * um * um,
                objective_rigid_um2=j_rigid * um * um,
                axis_scales={int(axis): float(pf[2]), int(ob.o[0]): float(p0_[2]), int(ob.o[1]): float(p1_[2])},
                flat_weights={int(axis): float(pf[3]), int(ob.o[0]): float(p0_[3]), int(ob.o[1]): float(p1_[3])},
                angle_to_spec_fit_deg=float(np.degrees(np.arccos(np.clip((np.trace(Rp) - 1.0) / 2.0, -1.0, 1.0)))),
                fold_axis_tilt_deg=float(np.degrees(np.arccos(min(1.0, abs(float(g[axis])))))),
                runner_up_objective_um2=(cmp[0][0] * um * um if cmp else None),
                runner_up_gap_rel=((cmp[0][0] - jbest) / max(abs(jbest), 1e-300) if cmp else None),
                runner_up_angle_deg=(float(np.degrees(np.arccos(min(1.0, abs(float(cmp[0][1][:, axis] @ g)))))) if cmp else None),
                runner_up_layout_maxdiff_um=ru_diff, grid=grid, n_starts=len(sols), nfev=int(ob.nfev),
                refine_converged=bool(all(s[2] for s in sols)), best_converged=bool(ok_best), best_source=sols[0][4],
                polish_gains_um2=polish, start_fit=info0, **cinfo)
    return Z, info


def size_match_folded(Z, X, codes, axis, centre, flat=False):
    """Folded views: per-axis spread match in the fitted frame (fold axis: RMS about the own centre), no refit."""
    if flat:
        return None, None
    stat = np.median if centre == "median" else np.mean
    c0, sd, tgt = Z.mean(0), Z.std(0), X.std(0)
    xa, za = X[:, axis] - stat(X[:, axis]), Z[:, axis] - stat(Z[:, axis])
    c0[axis], sd[axis], tgt[axis] = stat(Z[:, axis]), float(np.sqrt(np.mean(za ** 2))), float(np.sqrt(np.mean(xa ** 2)))
    if (sd <= SD_FLAT * tgt).any():
        return None, None
    Zs = c0 + (Z - c0) * (tgt / sd)
    v = codes >= 0
    o = [b for b in range(Z.shape[1]) if b != axis]
    Zs[:, o] += X[v][:, o].mean(0) - Zs[v][:, o].mean(0)
    return Zs, dict(rule="per-axis spread match in the fitted frame (fold axis: RMS about the own " + centre + "), no refit",
                    scales=(tgt / sd).tolist())


def corrupt(tag, X, fine_all, um_per_unit, seed=SEED_CORRUPT):
    """A corrupted copy of the truth (ruler layouts): @truth, @shuffle, @shrink<f>, @collapse, @noise<um>, @axis<a>x<f>."""
    rng = np.random.default_rng(seed)
    if tag == "@truth":
        return X.copy()
    if tag.startswith("@axis"):
        a, f = tag[5:].split("x")
        a, f = int(a), float(f)
        if not 0 <= a < X.shape[1]:
            raise ValueError(f"unknown ruler {tag}")
        Y = X.copy()
        Y[:, a] = X[:, a].mean() + f * (X[:, a] - X[:, a].mean())
        return Y
    if tag == "@shuffle":
        lab = fine_all
        Y = X.copy()
        for t in np.unique(lab):
            m = np.flatnonzero(lab == t)
            Y[m] = X[rng.permutation(m)]
        return Y
    if tag.startswith("@shrink"):
        f = float(tag[7:])
        return X.mean(0) + f * (X - X.mean(0))
    if tag == "@collapse":
        u, inv = np.unique(fine_all, return_inverse=True)
        cx = np.zeros((len(u), X.shape[1]))
        np.add.at(cx, inv, X)
        cx /= np.bincount(inv, minlength=len(u))[:, None]
        return cx[inv] + rng.normal(0, 1.0 / um_per_unit, X.shape)
    if tag.startswith("@noise"):
        return X + rng.normal(0, float(tag[6:]) / um_per_unit, X.shape)
    raise ValueError(f"unknown ruler {tag}")


class Truth:

    """Everything that depends on the truth only, computed once per test and view."""
    def __init__(self, X, fine, contact=None, cls=None, *, um_per_unit, E=None, name="truth", cls_strat=None, fine_all=None,
                 cls_all=None, cfg=None, verbose=True):
        t0 = time.time()
        cfg = dict(cfg or {})
        self.name, self.cfg, self.um = name, cfg, float(um_per_unit)
        self.X = as_coords(X, f"{name}: truth")
        self.N, self.D = self.X.shape
        if self.N <= K_NICHE:
            refuse(f"{name}: {self.N} cells; the niche metric needs more than {K_NICHE} cells (its {K_NICHE} nearest other cells)")
        if cfg.get("iou_mode") == "sections" and self.D != 3:
            refuse(f"{name}: section ids given for {self.D}-D coordinates; a 2-D layout is one section (omit the section ids)")
        if (np.ptp(self.X, axis=0) <= 0).any():
            refuse(f"{name}: a truth axis has zero extent ({np.ptp(self.X, axis=0).tolist()}); give the {self.D - 1}-D coordinates instead")
        raw_contact = fine if contact is None else contact
        raw_cls = raw_contact if cls is None else cls
        self.fine = as_codes(fine, f"{name}: fine labels", self.N)
        self.contact = as_codes(raw_contact, f"{name}: contact labels", self.N)
        self.cls = as_codes(raw_cls, f"{name}: class labels", self.N)
        if cls_strat is None:
            cls_strat = np.where(self.cls >= 0, self.cls, int(self.cls.max()) + 1)
        self.cls_strat = np.asarray(cls_strat, dtype=np.int64)
        self.fine_all = np.asarray(self.fine if fine_all is None else fine_all, dtype=np.int64)
        self.cls_all = np.asarray(self.cls if cls_all is None else cls_all, dtype=np.int64)
        for nm, a in (("cls_strat", self.cls_strat), ("fine_all", self.fine_all), ("cls_all", self.cls_all)):
            if len(a) != self.N:
                refuse(f"{name}: {nm} has {len(a)} entries, expected {self.N}")
        if self.cls_strat.min() < 0:
            refuse(f"{name}: stratification codes must be >= 0")
        self.tree = cKDTree(self.X)
        self.spacing_um = float(cfg["spacing_um"]) if cfg.get("spacing_um") is not None else spacing(self.X, tree=self.tree) * self.um
        self.taus_ell = [float(k) for k in cfg.get("taus_ell", (1, 2, 4, 8, 16, 32))]
        self.taus_um = [float(t) for t in cfg.get("taus_um", (50, 100, 200, 400))]
        self.info = dict(name=name, n_cells=self.N, dim=self.D, um_per_unit=self.um, truth_md5=md5_array(self.X),
                         fine_md5=md5_codes(fine), contact_md5=md5_codes(raw_contact), cls_md5=md5_codes(raw_cls),
                         codes_md5_rule="int64 md5 of the input codes; internally remapped to 0..K-1",
                         n_excluded=int((self.fine < 0).sum()))
        v = self.fine >= 0
        self.KF = int(self.fine[v].max()) + 1
        self.NF = np.bincount(self.fine[v], minlength=self.KF)
        self.FT = np.flatnonzero(self.NF >= MIN_TYPE)
        self.FT200 = np.flatnonzero(self.NF >= MIN_SPREAD_TYPE)
        self.order = np.argsort(self.fine, kind="stable")
        self.bounds = np.searchsorted(self.fine[self.order], np.arange(self.KF + 1))
        self.TH = np.random.default_rng(SEED_DIRS).normal(size=(N_DIRS, self.D))
        self.TH /= np.linalg.norm(self.TH, axis=1, keepdims=True)
        self.ftrees = {t: cKDTree(self.X[self._cells(t)]) for t in self.FT}
        self.info.update(w2_directions=N_DIRS, w2_directions_seed=SEED_DIRS, corruption_seed=SEED_CORRUPT, iou_subsample_seed=0,
                         axis_reference_pairs_seed=0, axis_spearman_quantum_um=QUANTUM_UM, iou_origin=IOU_ORIGIN, iou_slabs=IOU_SLABS,
                         taus_um=self.taus_um,
                         spacing_um=self.spacing_um, taus_ell=self.taus_ell, taus_ell_um=[k * self.spacing_um for k in self.taus_ell],
                         n_fine_types=int((self.NF > 0).sum()), n_fine_types_ge20=int(len(self.FT)),
                         fine_coverage_ge20=float(self.NF[self.FT].sum() / self.N), n_fine_types_ge200=int(len(self.FT200)))
        vc = self.contact >= 0
        self.K4 = int(self.contact[vc].max()) + 1
        self.N4 = np.bincount(self.contact[vc], minlength=self.K4)
        self.T4 = np.flatnonzero(self.N4 >= MIN_TYPE)
        self.off = ~np.eye(len(self.T4), dtype=bool)
        self.nb15 = knn_excluding_self(self.X, K_CONTACT, self.tree)
        self.MX = self.contact_matrix(self.nb15)
        self.like_truth = self.like_type(self.nb15)
        self.info.update(n_contact_types_ge20=int(len(self.T4)), crisp_truth=float(np.diag(self.MX).mean()), like_type_truth=self.like_truth,
                         knn15_boundary_tie_share=knn_boundary_ties(self.X, K_CONTACT, self.tree))
        vk = self.cls >= 0
        self.KC = int(self.cls[vk].max()) + 1
        self.NB = np.bincount(self.cls[vk], minlength=self.KC)
        self.TYPES = np.flatnonzero(self.NB >= MIN_TYPE)
        self.CNX, self.CEN, kinfo = self._kmeans(self.niche_comp(self.X, self.tree))
        self.info.update(n_classes_ge20=int(len(self.TYPES)), niche_kmeans=kinfo)
        self.E = None if E is None else np.asarray(E)
        if self.E is not None:
            if self.E.ndim != 2 or len(self.E) != self.N:
                refuse(f"{name}: expression shape {self.E.shape}, expected ({self.N}, G)")
            if not np.isfinite(self.E).all():
                refuse(f"{name}: expression has NaN / inf")
            self.Et = np.ascontiguousarray(self.E.T, dtype=np.float64)
            self.IX = self.morans(self.nb15)
            self.info.update(moran_genes=int(self.E.shape[1]), moran_truth_finite=int(np.isfinite(self.IX).sum()),
                             expression_md5=md5_array(self.E))
        n_pairs = int(cfg.get("n_pairs", N_PAIRS))
        r = np.random.default_rng(SEED_PAIRS)
        i, j = r.integers(0, self.N, n_pairs), r.integers(0, self.N, n_pairs)
        k = i != j
        self.pi, self.pj = i[k], j[k]
        self.pdx = np.linalg.norm(self.X[self.pi] - self.X[self.pj], axis=1)
        self.X01 = (self.X - self.X.min(0)) / (self.X.max(0) - self.X.min(0))
        self.pdx01 = np.linalg.norm(self.X01[self.pi] - self.X01[self.pj], axis=1)
        self.info.update(pairs_seed=SEED_PAIRS, pairs_drawn=n_pairs, pairs_used=int(len(self.pi)))
        n_sub = int(cfg.get("n_sub", N_SUB))
        NS = np.bincount(self.cls_strat)
        if self.N <= n_sub:
            self.SUB = np.arange(self.N)
        else:
            self.SUB = np.sort(np.concatenate([np.random.default_rng(SEED_SUB).choice(np.flatnonzero(self.cls_strat == t),
                                                                                     max(1, int(round(n_sub * NS[t] / self.N))), replace=False)
                                               for t in np.flatnonzero(NS > 0)]))
        self.sub_dx = pdist(self.X[self.SUB])
        self.sub_dx_has_zero = bool((self.sub_dx == 0).any())
        self.sub_thr_x = None if self.sub_dx_has_zero else dict(zip(CONTACT_PCTS, self._pair_quantile(self.sub_dx, list(CONTACT_PCTS))))
        self.sub_ranks = centred_row_ranks(self.X[self.SUB])
        self.info.update(sub_seed=SEED_SUB, sub_cells=int(len(self.SUB)), sub_pairs=int(len(self.sub_dx)))
        self.Xc = self.X - self.X.mean(0)
        lu = cfg.get("lddt_u_um")
        self.lddt_u = (float(lu) / self.um) if lu is not None else spacing(self.X, tree=self.tree)
        q = cfg.get("lddt_query")
        if q is None:
            q = np.arange(self.N) if self.N <= LDDT_ALL_MAX else np.sort(np.random.default_rng(int(cfg.get("lddt_seed", 20261014)) + 7)
                                                                        .choice(self.N, LDDT_QUERY_N, replace=False))
        self.lddt_p = lddt_partners(self.X, self.lddt_u, int(cfg.get("lddt_seed", 20261014)), query=np.asarray(q), tree=self.tree)
        t_ = self.lddt_p["tot"]
        self.info.update(lddt_u_um=self.lddt_u * self.um, lddt_u_rule=cfg.get("lddt_u_rule", "given (cfg lddt_u_um)" if lu is not None else
                                                                        "truth median nearest-neighbour spacing"),
                         lddt_cutoff_um=LDDT_CUT_U * self.lddt_u * self.um, lddt_seed=int(cfg.get("lddt_seed", 20261014)),
                         lddt_query_cells=int(len(q)), lddt_pairs=int(self.lddt_p["kept"].sum()),
                         lddt_partners_median=float(np.median(t_)), lddt_share_capped=float(np.mean(t_ > LDDT_K)),
                         lddt_cells_without_partner=int(np.sum(t_ == 0)), lddt_perfect_smooth=LDDT_PERFECT_SMOOTH)
        self.iou = None
        mode = cfg.get("iou_mode")
        grids = [float(g) for g in cfg.get("iou_grids_um", ())]
        if mode == "sections" and grids:
            self.iou = self._section_setup(np.asarray(cfg["sections"]), cfg.get("section_R"), cfg.get("section_t"), grids)
        elif mode == "volume" and grids:
            self.iou = dict(mode="volume", grids=grids)
        self.info["iou"] = None if self.iou is None else {k: v for k, v in self.iou.items() if not k.startswith("_")}
        rg = np.random.default_rng(0)
        ai, aj = rg.integers(0, self.N, 2_000_000), rg.integers(0, self.N, 2_000_000)
        kk = ai != aj
        self.ax_i, self.ax_j = ai[kk], aj[kk]
        self.landmarks = None
        if cfg.get("landmark_names") is not None:
            self.landmarks = landmark_setup(self.X, np.asarray(cfg["landmark_names"]), self.um)
        self.info["seconds"] = round(time.time() - t0, 1)
        if verbose:
            log(f"[truth] {name}: {self.N:,} cells, {self.D}-D, {self.um:g} um per unit | fine {len(self.FT)} types >= 20 "
                f"({self.info['fine_coverage_ge20']:.4f} of cells), contact {len(self.T4)}, classes {len(self.TYPES)} | LDDT u "
                f"{self.info['lddt_u_um']:.3f} um, {len(q):,} query cells, {self.info['lddt_pairs']:,} pairs | SUB {len(self.SUB):,} | "
                f"IoU {self.iou['mode'] if self.iou else None} | {self.info['seconds']} s")

    def _cells(self, t):
        return self.order[self.bounds[t]:self.bounds[t + 1]]

    @staticmethod
    def _pair_quantile(d, p):
        d = d[d != 0]
        nw = 2 * len(d)
        ps = np.atleast_1d(np.asarray(p, dtype=np.float64))
        h = ps * (nw - 1)
        lo = np.floor(h).astype(np.int64)
        hi = np.minimum(lo + 1, nw - 1)
        part = np.partition(d, np.unique(np.r_[lo // 2, hi // 2]))
        q = np.array([np.quantile(np.array([part[a // 2], part[b // 2]]), hh - a) for hh, a, b in zip(h, lo, hi)])
        return float(q[0]) if np.ndim(p) == 0 else q

    def contact_matrix(self, nb):
        k = nb.shape[1]
        src = np.repeat(self.contact, k)
        dst = self.contact[nb].ravel()
        ok = (src >= 0) & (dst >= 0)
        M = np.bincount(src[ok] * self.K4 + dst[ok], minlength=self.K4 * self.K4).reshape(self.K4, self.K4).astype(np.float64)
        M[self.N4 > 0] /= (self.N4[self.N4 > 0] * k)[:, None]
        return M[np.ix_(self.T4, self.T4)]

    def like_type(self, nb):
        q = self.contact >= 0
        return float(np.mean(self.contact[nb[q]] == self.contact[q][:, None]))

    def niche_comp(self, Z, tree=None, chunk=200_000):
        tree = cKDTree(Z) if tree is None else tree
        F = np.zeros((len(Z), len(self.TYPES)))
        col = np.full(self.KC, -1, np.int64)
        col[self.TYPES] = np.arange(len(self.TYPES))
        for s in range(0, len(Z), chunk):
            nb = knn_excluding_self(Z, K_NICHE, tree, query=np.arange(s, min(len(Z), s + chunk)))
            cc = self.cls[nb]
            cc = np.where(cc >= 0, col[np.maximum(cc, 0)], -1)
            rows = np.repeat(np.arange(len(nb)), K_NICHE)
            ok = cc.ravel() >= 0
            F[s:s + len(nb)] = np.bincount(rows[ok] * len(self.TYPES) + cc.ravel()[ok],
                                           minlength=len(nb) * len(self.TYPES)).reshape(len(nb), len(self.TYPES))
        return F / K_NICHE

    def _kmeans(self, F):
        from sklearn.cluster import KMeans
        import sklearn
        key = hashlib.md5((md5_array(F) + f"|{N_NICHES}|10|0|{sklearn.__version__}|{os.environ.get('OMP_NUM_THREADS')}").encode()).hexdigest()
        cdir = self.cfg.get("cache_dir")
        f = f"{cdir}/niche_{self.cfg.get('cache_tag', self.name)}_{key[:16]}.npz" if cdir else None
        if f and os.path.exists(f):
            z = np.load(f)
            return z["CNX"].astype(np.int64), z["CEN"], dict(source="cache", file=f, file_md5=md5_file(f), key=key, n_clusters=N_NICHES,
                                                             n_init=10, random_state=0, k_neighbours=K_NICHE)
        t0 = time.time()
        km = KMeans(n_clusters=N_NICHES, n_init=10, random_state=0).fit(F)
        CNX, CEN = km.labels_.astype(np.int64), km.cluster_centers_
        if f:
            os.makedirs(cdir, exist_ok=True)
            np.savez(f, CNX=CNX.astype(np.int32), CEN=CEN, key=key)
        return CNX, CEN, dict(source="fitted", file=f, key=key, seconds=round(time.time() - t0, 1), inertia=float(km.inertia_),
                              n_iter=int(km.n_iter_), n_clusters=N_NICHES, n_init=10, random_state=0, k_neighbours=K_NICHE)

    def morans(self, nb):
        from scanpy.metrics import morans_i
        n, k = nb.shape
        g = sp.csr_matrix((np.full(n * k, 1 / k), (np.repeat(np.arange(n), k), nb.ravel())), shape=(n, n))
        return np.asarray(morans_i(g, self.Et))

    def _section_setup(self, sec, R, t, grids):
        if len(sec) != self.N:
            refuse(f"{self.name}: section ids have {len(sec)} entries")
        R = np.eye(self.D) if R is None else np.asarray(R, dtype=np.float64)
        t = np.zeros(self.D) if t is None else np.asarray(t, dtype=np.float64)
        Xs = self.X @ R + t
        secs = np.unique(sec)
        zt = np.array([Xs[sec == s, -1].mean() for s in secs])
        o = np.argsort(zt)
        secs, zt = secs[o], zt[o]
        spread = max(float(np.ptp(Xs[sec == s, -1])) for s in secs)
        h = float(np.median(np.diff(zt))) if len(zt) > 1 else float("inf")
        planar = spread <= 0.01 * h
        groups, gdep = [], []
        for s_, z_ in zip(secs, zt):
            if groups and z_ - gdep[-1][0] < 1.0 / self.um:
                groups[-1].append(s_)
                gdep[-1].append(z_)
            else:
                groups.append([s_])
                gdep.append([z_])
        zs = np.array([float(np.mean(d)) for d in gdep])
        hs = float(np.median(np.diff(zs))) if len(zs) > 1 else float("inf")
        mids = (zs[1:] + zs[:-1]) / 2
        lo, hi = np.r_[zs[0] - hs / 2, mids], np.r_[mids, zs[-1] + hs / 2]
        slabs = [(np.array(g), float(z_), float(a), float(b)) for g, z_, a, b in zip(groups, zs, lo, hi)]
        gaps = np.diff(zs) if len(zs) > 1 else np.array([np.inf])
        return dict(mode="sections" if planar else "not_planar", grids=grids, n_sections=int(len(secs)), n_slabs=int(len(slabs)),
                    sections_merged=int(len(secs) - len(slabs)), section_gap_um=h * self.um, slab_gap_min_um=float(gaps.min()) * self.um,
                    slab_gap_max_um=float(gaps.max()) * self.um, max_within_section_depth_spread_um=spread * self.um, margin_um=IOU_MARGIN_UM,
                    _sec=sec, _secs=secs, _zt=zt, _h=h, _R=R, _t=t, _Xs=Xs, _MIN=Xs[:, :-1].min(0), _slabs=slabs)


def lddt_partners(Z, u, seed, query=None, tree=None, chunk=2000):
    """Partner sets of the per-cell LDDT: true neighbours within 15 u, at most 256 per cell."""
    n = len(Z)
    q = np.arange(n) if query is None else np.asarray(query)
    tree = cKDTree(Z) if tree is None else tree
    cut = LDDT_CUT_U * u
    rng = np.random.default_rng(seed)
    J = np.full((len(q), LDDT_K), -1, np.int64)
    D = np.zeros((len(q), LDDT_K))
    kept = np.zeros(len(q), np.int64)
    tot = np.zeros(len(q), np.int64)
    for s in range(0, len(q), chunk):
        idx = q[s:s + chunk]
        cand = tree.query_ball_point(Z[idx], r=cut, workers=WORKERS)
        for a, i in enumerate(idx):
            c = np.sort(np.asarray(cand[a], dtype=np.int64))
            d = np.linalg.norm(Z[c] - Z[i], axis=1)
            m = (d < cut) & (c != i)
            c, d = c[m], d[m]
            tot[s + a] = len(c)
            if len(c) > LDDT_K:
                pick = rng.choice(len(c), LDDT_K, replace=False)
                c, d = c[pick], d[pick]
            J[s + a, :len(c)] = c
            D[s + a, :len(c)] = d
            kept[s + a] = len(c)
    return dict(J=J, D=D, kept=kept, tot=tot, q=q, seed=seed)


def lddt_score(Q, p, u, chunk=4096):
    """Per-cell smooth and plain LDDT of a layout on the partner sets."""
    J, D, kept, q = p["J"], p["D"], p["kept"], p["q"]
    n = len(J)
    sm, pl = np.full(n, np.nan), np.full(n, np.nan)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        m = J[s:e] >= 0
        k = kept[s:e]
        ok = k > 0
        dp = np.linalg.norm(Q[np.where(m, J[s:e], 0)] - Q[q[s:e], None, :], axis=-1)
        ad = np.abs(dp - D[s:e])
        sm[s:e][ok] = ((0.25 * sum(expit(t - ad / u) for t in LDDT_TAUS)) * m).sum(1)[ok] / k[ok]
        pl[s:e][ok] = ((0.25 * sum((ad < t * u).astype(np.float64) for t in LDDT_TAUS)) * m).sum(1)[ok] / k[ok]
    return sm, pl


def landmark_setup(X, names, um):
    """Brain landmarks of the truth (bulb, dentate gyrus, cerebellum depths)."""
    DG, OB, CB = names == "04 DG-IMN Glut", names == "05 OB-IMN GABA", names == "29 CB Glut"
    if not (DG.any() and OB.any() and CB.any()):
        return None
    SG = -1.0 if np.median(X[OB, 2]) > np.median(X[CB, 2]) else 1.0
    f0 = np.percentile(SG * X[:, 2], 0.1)
    front = (SG * X[:, 2] - f0) < 1500.0 / um
    xr = float(np.percentile(X[OB & front, 0], 99.5))
    zt = SG * (X[:, 2] - np.median(X[DG, 2]))
    classes = [k for k in np.unique(names) if (names == k).sum() >= 5000]
    sub = {k: np.random.default_rng(0).choice(np.flatnonzero(names == k), min(20000, int((names == k).sum())), replace=False) for k in classes}
    return dict(DG=DG, OB=OB, CB=CB, SG=SG, front=front, xr=xr, zt=zt, classes=classes, sub=sub, um=um)


def landmarks(L, Y):
    """Landmark readout of an aligned brain layout."""
    from scipy.stats import wasserstein_distance
    um = L["um"]
    zp = L["SG"] * (Y[:, 2] - np.median(Y[L["DG"], 2]))
    e = np.abs(zp - L["zt"])
    w = [wasserstein_distance(zp[L["sub"][k]], L["zt"][L["sub"][k]]) for k in L["classes"]]
    return dict(phantom_bulb_share=float((Y[L["OB"] & L["front"], 0] > L["xr"] + 100.0 / um).mean()),
                depth_err_vs_dg_median_um=float(np.median(e)) * um, depth_within_300um=float((e < 300.0 / um).mean()),
                class_depth_w1_median_um=float(np.median(w)) * um, class_depth_w1_max_um=float(np.max(w)) * um,
                front_p02_um=float(np.percentile(zp, 0.2)) * um, cb_p5_um=float(np.percentile(zp[L["CB"]], 5)) * um,
                end_p998_um=float(np.percentile(zp, 99.8)) * um, bulb_to_cb_um=float(np.median(zp[L["CB"]]) - np.median(zp[L["OB"]])) * um)


def m_size(T, Z, own=None):
    """Size of the aligned layout relative to the truth: overall, per axis, along the truth's and its own principal axes."""
    X = T.X
    out = dict(size_rms=float(np.sqrt(((Z - Z.mean(0)) ** 2).sum(1).mean() / ((X - X.mean(0)) ** 2).sum(1).mean())),
               size_axis=(Z.std(0) / X.std(0)).tolist())
    ev, V = np.linalg.eigh(np.cov((X - X.mean(0)).T, bias=True))
    V = V[:, ::-1]
    out["size_pc"] = (((Z - Z.mean(0)) @ V).std(0) / ((X - X.mean(0)) @ V).std(0)).tolist()
    if own is None:
        if getattr(T, "_psd", None) is None:
            T._psd = principal_sds(X)
        own = principal_sds(Z) / T._psd
    out["size_own_pc"] = [float(v) for v in own]
    return out


def principal_sds(Z):
    """Sorted principal SDs of a layout."""
    return np.linalg.svd(Z - Z.mean(0), compute_uv=False) / np.sqrt(len(Z))


def m_w2(T, Z):
    """Per-type sliced Wasserstein-2 distance (64 directions) and its per-axis version, in um."""
    X = T.X
    num, numax = 0.0, np.zeros(T.D)
    for t in T.FT:
        m = T._cells(t)
        num += np.mean((np.sort(X[m] @ T.TH.T, 0) - np.sort(Z[m] @ T.TH.T, 0)) ** 2) * len(m)
        numax += np.mean((np.sort(X[m], 0) - np.sort(Z[m], 0)) ** 2, axis=0) * len(m)
    tot = T.NF[T.FT].sum()
    return dict(w2_um=float(np.sqrt(num / tot) * T.um), w2_axis_um=(np.sqrt(numax / tot) * T.um).tolist())


def m_fscore(T, Z):
    """Per-type F-score, precision and recall at every distance threshold, and the Chamfer distance."""
    labels = [f"{t:g}" for t in T.taus_um] + [f"{k:g}ell" for k in T.taus_ell]
    taus = [t / T.um for t in T.taus_um] + [k * T.spacing_um / T.um for k in T.taus_ell]
    P, R, F, w, CD = [[] for _ in taus], [[] for _ in taus], [[] for _ in taus], [], []
    for t in T.FT:
        m = np.flatnonzero(T.fine == t)
        d1, _ = T.ftrees[t].query(Z[m], k=1, workers=WORKERS)
        d2, _ = cKDTree(Z[m]).query(T.X[m], k=1, workers=WORKERS)
        CD.append((d1.mean() + d2.mean()) / 2)
        for a, tau in enumerate(taus):
            p, r = float(np.mean(d1 < tau)), float(np.mean(d2 < tau))
            P[a].append(p)
            R[a].append(r)
            F[a].append(2 * p * r / (p + r) if p + r > 0 else 0.0)
        w.append(len(m))
    out = {"chamfer_um": float(np.average(CD, weights=w)) * T.um}
    for a, k in enumerate(labels):
        out[f"f{k}"] = float(np.average(F[a], weights=w))
        out[f"p{k}"] = float(np.average(P[a], weights=w))
        out[f"r{k}"] = float(np.average(R[a], weights=w))
    return out


def m_type_spread(T, Z):
    """Spread ratio, squeezed share and type-centre error of the fine types with >= 200 cells."""
    if len(T.FT200) == 0:
        return dict(spread_median=None, squeezed_share=None, n_spread_types=0, centre_err_um_mean=None, centre_err_um_median=None)
    sr, ce, w = [], [], []
    for t in T.FT200:
        m = T._cells(t)
        yt, yp = T.X[m], Z[m]
        st = np.sqrt(((yt - yt.mean(0)) ** 2).sum(1).mean())
        sp_ = np.sqrt(((yp - yp.mean(0)) ** 2).sum(1).mean())
        sr.append(sp_ / st)
        ce.append(np.linalg.norm(yp.mean(0) - yt.mean(0)))
        w.append(len(m))
    sr, ce, w = np.array(sr), np.array(ce), np.array(w, float)
    o = np.argsort(sr)
    cw = np.cumsum(w[o]) / w.sum()
    return dict(spread_median=float(sr[o][np.searchsorted(cw, 0.5)]), squeezed_share=float(np.mean(sr < SQUEEZE)),
                n_spread_types=int(len(sr)), n_squeezed=int((sr < SQUEEZE).sum()),
                centre_err_um_mean=float(np.average(ce, weights=w)) * T.um, centre_err_um_median=float(np.median(ce)) * T.um)


def m_structure(T, Z):
    """Cell-type contact matrix agreement, like-type share, niche NMI / ARI and Moran's I agreement."""
    tree = cKDTree(Z)
    nb = knn_excluding_self(Z, K_CONTACT, tree)
    MY = T.contact_matrix(nb)
    out = dict(contact_r_offdiag=float(pearsonr(T.MX[T.off], MY[T.off])[0]), contact_mad=float(np.abs(T.MX - MY).mean()),
               crisp=float(np.diag(MY).mean()), crisp_truth=float(np.diag(T.MX).mean()), like_type=T.like_type(nb),
               like_type_truth=T.like_truth)
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, pairwise_distances_argmin
    cn = pairwise_distances_argmin(T.niche_comp(Z, tree), T.CEN)
    out.update(niche_nmi=float(normalized_mutual_info_score(T.CNX, cn)), niche_ari=float(adjusted_rand_score(T.CNX, cn)))
    if T.E is not None:
        IY = T.morans(nb)
        ok = np.isfinite(T.IX) & np.isfinite(IY)
        out.update(moran_r=float(pearsonr(T.IX[ok], IY[ok])[0]), moran_mad=float(np.abs(T.IX[ok] - IY[ok]).mean()),
                   moran_genes_used=int(ok.sum()))
    return out


def m_pairs(T, Z):
    """Spearman and Pearson correlation of 2,000,000 random pairwise distances, and the min-max normalised version."""
    dy = np.linalg.norm(Z[T.pi] - Z[T.pj], axis=1)
    rng_ = Z.max(0) - Z.min(0)
    out = dict(spearman=float(spearmanr(T.pdx, dy)[0]), pearson=float(pearsonr(T.pdx, dy)[0]))
    if (rng_ > 0).all():
        Z01 = (Z - Z.min(0)) / rng_
        out["remap"] = float(pearsonr(T.pdx01, np.linalg.norm(Z01[T.pi] - Z01[T.pj], axis=1))[0])
    else:
        out["remap"] = float("nan")
    return out


def centred_row_ranks(A, chunk=500):
    """Centred average ranks of every row of the distance matrix."""
    n = len(A)
    out = np.empty((n, n), np.float32)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        out[s:e] = rankdata(cdist(A[s:e], A), axis=1) - (n + 1) / 2.0
    return out


def per_cell_src(A, B, chunk=500, ra_all=None):
    """Per-cell Spearman correlation of distances to all cells."""
    out = np.empty(len(A))
    for s in range(0, len(A), chunk):
        e = min(len(A), s + chunk)
        ra = (rankdata(cdist(A[s:e], A), axis=1) - (len(A) + 1) / 2.0) if ra_all is None else ra_all[s:e].astype(np.float64)
        rb = rankdata(cdist(B[s:e], B), axis=1)
        rb -= rb.mean(1, keepdims=True)
        out[s:e] = (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
    return out


def contact_scores(dx, dy, ps, thr_x=None):
    """Contact F1 and precision at the closest fractions of pairs."""
    nz = (dx * dy) != 0
    dx_, dy_ = dx[nz], dy[nz]
    tx = [thr_x[p] for p in ps] if (thr_x is not None and nz.all()) else Truth._pair_quantile(dx_, list(ps))
    ty = Truth._pair_quantile(dy_, list(ps))
    out = {}
    for p, a_, b_ in zip(ps, tx, ty):
        a, b = dx_ < a_, dy_ < b_
        tp = int(np.sum(a & b))
        na, nb = int(a.sum()), int(b.sum())
        out[p] = ((2 * tp / (na + nb) if na + nb > 0 else 0.0), (tp / nb if nb > 0 else 0.0))
    return out


def rssd(A, B):
    """Residual of the optimal proper rotation between two centred point sets."""
    if A.shape[1] == 2:
        A, B = np.pad(A, ((0, 0), (0, 1))), np.pad(B, ((0, 0), (0, 1)))
    return float(Rotation.align_vectors(A, B)[1])


def m_legacy(T, Z):
    """Contact F1, Sum RSSD and per-cell SRC on the stratified subsample."""
    out = {}
    dy = pdist(Z[T.SUB])
    for p, (f1, pr) in contact_scores(T.sub_dx, dy, CONTACT_PCTS, T.sub_thr_x).items():
        tag = f"{int(round(p * 100))}pct"
        out[f"contact_f1_{tag}"], out[f"contact_precision_{tag}"] = float(f1), float(pr)
    Zc = Z - Z.mean(0)
    f = T.um / 1000.0
    out["sum_rssd_mm"] = float(sum(rssd(T.Xc[T.cls == t] * f, Zc[T.cls == t] * f) for t in T.TYPES))
    src = per_cell_src(T.X[T.SUB], Z[T.SUB], ra_all=T.sub_ranks)
    out.update(src_median=float(np.median(src)), src_mean=float(np.mean(src)), n_sub=int(len(T.SUB)))
    return out


def m_lddt(T, Z):
    """Mean and median per-cell LDDT."""
    sm, pl = lddt_score(Z, T.lddt_p, T.lddt_u)
    return dict(lddt_smooth_mean=float(np.nanmean(sm)), lddt_smooth_median=float(np.nanmedian(sm)),
                lddt_plain_mean=float(np.nanmean(pl)), lddt_plain_median=float(np.nanmedian(pl)))


def _occ_keys(P, LO, g):
    """Integer keys of the occupied grid cells."""
    idx = np.floor((P - LO) / g).astype(np.int64)
    if len(idx) == 0:
        return np.zeros(0, np.int64)
    if np.abs(idx).max() >= 2 ** 20:
        refuse(f"layout far outside the truth's extent (grid index {int(np.abs(idx).max())})")
    key = np.zeros(len(idx), np.int64)
    for a in range(idx.shape[1]):
        key = key * (2 ** 21) + (idx[:, a] + 2 ** 20)
    return np.unique(key)


def _iou(A, B):
    inter = len(np.intersect1d(A, B, assume_unique=True))
    return inter / max(len(A) + len(B) - inter, 1)


def _origin(MIN, g, um):
    LO = MIN - IOU_MARGIN_UM / um
    return LO - g / 2


def m_iou(T, Z):
    """Occupancy IoU per section slab (serial data) or per 3-D voxel grid (continuous data), and the section count error."""
    S = T.iou
    if S is None or S["mode"] == "not_planar":
        return {}
    out = {}
    if S["mode"] == "volume":
        MIN = T.X.min(0)
        types, cnt = np.unique(T.cls[T.cls >= 0], return_counts=True)
        keep = cnt / T.N >= IOU_MIN_SHARE
        for g_um in S["grids"]:
            g = g_um / T.um
            LO, LO2 = _origin(MIN, g, T.um), _origin(MIN, g / 2, T.um)
            vals = [_iou(_occ_keys(T.X[T.cls == tp], LO, g), _occ_keys(Z[T.cls == tp], LO, g)) for tp in types[keep]]
            out[f"iou_pop_{g_um:g}"] = float(np.average(vals, weights=cnt[keep]))
            out[f"iou_support_{g_um / 2:g}"] = _iou(_occ_keys(T.X, LO2, g / 2), _occ_keys(Z, LO2, g / 2))
        out["iou_origin"] = IOU_ORIGIN
        return out
    Xs, sec, MIN = S["_Xs"], S["_sec"], S["_MIN"]
    Zs = Z @ S["_R"] + S["_t"]
    rng = np.random.default_rng(0)
    pop = {g: [] for g in S["grids"]}
    sup = {g: [] for g in S["grids"]}
    lr = []
    for ids, z0, a, b in S["_slabs"]:
        mt = np.flatnonzero(sec == ids[0]) if len(ids) == 1 else np.flatnonzero(np.isin(sec, ids))
        ms = np.flatnonzero((Zs[:, -1] >= a) & (Zs[:, -1] < b))
        lr.append(abs(float(np.log2(max(len(ms), 1) / len(mt)))))
        n = min(len(mt), len(ms))
        if n < IOU_MIN_N:
            continue
        it = rng.choice(mt, n, replace=False)
        ip = rng.choice(ms, n, replace=False)
        PT, PS = Xs[it, :-1], Zs[ip, :-1]
        types, cnt = np.unique(T.cls[it], return_counts=True)
        keep = (cnt / n >= IOU_MIN_SHARE) & (types >= 0)
        for g_um in S["grids"]:
            g = g_um / T.um
            LO, LO2 = _origin(MIN, g, T.um), _origin(MIN, g / 2, T.um)
            sup[g_um].append(_iou(_occ_keys(PT, LO2, g / 2), _occ_keys(PS, LO2, g / 2)))
            vals = [_iou(_occ_keys(PT[T.cls[it] == tp], LO, g), _occ_keys(PS[T.cls[ip] == tp], LO, g)) for tp in types[keep]]
            if vals:
                pop[g_um].append(float(np.average(vals, weights=cnt[keep])))
    out["count_err"] = float(np.mean(lr))
    for g_um in S["grids"]:
        out[f"iou_pop_{g_um:g}"] = float(np.mean(pop[g_um])) if pop[g_um] else None
        out[f"iou_support_{g_um / 2:g}"] = float(np.mean(sup[g_um])) if sup[g_um] else None
    out["iou_sections_scored"] = int(len(sup[S["grids"][0]]))
    out["iou_origin"] = IOU_ORIGIN
    out["iou_slabs"] = IOU_SLABS
    return out


def quantize_um(v, um):
    """Coordinates rounded to 1 nm, in um."""
    return np.round(v * um / QUANTUM_UM) * QUANTUM_UM


def axis_stats(T, Z):
    """Per truth axis: SDs, size, correlations, median error and references."""
    X, um = T.X, T.um
    E = Z - X
    tot = float((E ** 2).mean(0).sum())
    out = []
    for a in range(T.D):
        x, y, e = X[:, a], Z[:, a], E[:, a]
        out.append(dict(axis=a, sd_true_um=float(x.std()) * um, sd_placed_um=float(y.std()) * um, size=float(y.std() / x.std()),
                        pearson=float(np.corrcoef(x, y)[0, 1]), spearman=float(spearmanr(quantize_um(x, um), quantize_um(y, um))[0]),
                        med_err_um=float(np.median(np.abs(e))) * um, even_um=float(np.median(np.abs(x[T.ax_i] - x[T.ax_j]))) * um,
                        const_um=float(np.median(np.abs(x - x.mean()))) * um, r2=float(1.0 - (e ** 2).mean() / x.var()),
                        mse_share=float((e ** 2).mean() / tot) if tot > 0 else 0.0))
    return out


def m_diag(T, Z):
    """Per-cell error diagnostics of an aligned layout."""
    e = Z - T.X
    e3 = np.linalg.norm(e, axis=1) * T.um
    return dict(err3d_median_um=float(np.median(e3)), err3d_mean_um=float(np.mean(e3)), within100um=float(np.mean(e3 < 100.0)),
                err_axis_median_um=(np.median(np.abs(e), 0) * T.um).tolist(), axes=axis_stats(T, Z))


def score_view(T, Ya, Ys, Yd=None, timing=True, own=None, own_sm=None):
    """Every metric of one aligned layout, its size-matched version and the cell-fit diagnostic layout."""
    res, tm = {}, {}

    def run(name, f, *a):
        t0 = time.time()
        r = f(*a)
        tm[name] = round(time.time() - t0, 2)
        return r
    sm = Ys is not None
    res["size_matched_available"] = sm
    res.update(run("size", m_size, T, Ya, own))
    res.update(run("w2", m_w2, T, Ya))
    res.update(run("fscore", m_fscore, T, Ya))
    if sm:
        res.update({f"{k}_sm": v for k, v in run("size_sm", m_size, T, Ys, own_sm).items()})
        res.update({f"{k}_sm": v for k, v in run("w2_sm", m_w2, T, Ys).items()})
        res.update({f"{k}_sm": v for k, v in run("fscore_sm", m_fscore, T, Ys).items()})
    res["type_spread"] = run("spread", m_type_spread, T, Ya)
    res["type_spread_sm"] = run("spread_sm", m_type_spread, T, Ys) if sm else None
    res.update(run("structure", m_structure, T, Ya))
    res.update(run("pairs", m_pairs, T, Ya))
    res.update(run("legacy", m_legacy, T, Ya))
    res.update(run("lddt", m_lddt, T, Ya))
    res.update(run("iou", m_iou, T, Ya))
    if sm:
        res.update({f"{k}_sm": v for k, v in run("iou_sm", m_iou, T, Ys).items()})
    res["diag_typefit"] = run("diag", m_diag, T, Ya)
    if Yd is not None:
        res["diag_cellfit"] = run("diag_cell", m_diag, T, Yd)
        res["type_spread_cellfit"] = m_type_spread(T, Yd)
    if T.landmarks is not None:
        res["landmarks"] = run("landmarks", landmarks, T.landmarks, Ya)
    if timing:
        res["seconds_by_block"] = tm
    return res


def build_truths(spec, views=None, cache_dir=None):
    """One Truth per view of the test."""
    views = views or spec["views"]
    X = spec["X"]
    L = spec["labels"]
    common = dict(um_per_unit=spec["um_per_unit"], E=spec.get("E"), cls_strat=spec["cls_strat"], fine_all=spec["fine_all"],
                  cls_all=spec["cls_all"])
    sp_um = spacing(X) * spec["um_per_unit"]
    spec["spacing_um"] = sp_um
    cfg = dict(taus_um=spec["taus_um"], spacing_um=sp_um, iou_grids_um=spec["iou_grids_um"], iou_mode=spec["iou_mode"],
               lddt_u_um=spec["lddt_u_um"] if spec["lddt_u_um"] is not None else sp_um, lddt_seed=spec["lddt_seed"], cache_dir=cache_dir,
               lddt_u_rule=("given (test card lddt_u_um)" if spec["lddt_u_um"] is not None else
                            "the dataset spacing: median nearest-neighbour distance of the unfolded truth"),
               landmark_names=spec["class_names"] if spec.get("landmarks") else None)
    if spec["iou_mode"] == "sections":
        cfg["sections"] = spec["sections"]
        sf = spec.get("section_frame")
        if isinstance(sf, tuple):
            cfg["section_R"], cfg["section_t"] = sf
        elif sf == "fit":
            cfg["section_R"], cfg["section_t"] = section_frame_fit(X, spec["sections"])
    out = {}
    if "main" in views:
        out["main"] = Truth(X, L["fine"], L["contact"], L["cls"], name=f"{spec['dataset']}:{spec.get('units')}:main",
                            cfg=dict(cfg, cache_tag=f"{spec['dataset']}_{spec.get('units')}_main"), **common)
    if "lr_folded" in views:
        lf = spec["lr_fold"]
        Xf = fold_lr_own(lf["axis"], lf["centre"])(X)
        cf = dict(cfg, cache_tag=f"{spec['dataset']}_{spec.get('units')}_lrfold", lddt_seed=spec["lddt_seed"] + 1, landmark_names=None)
        if cf["iou_mode"] == "sections" and isinstance(spec.get("section_frame"), tuple):
            cf["iou_mode"] = None
        out["lr_folded"] = Truth(Xf, L["fine"], L["contact"], L["cls"], name=f"{spec['dataset']}:{spec.get('units')}:lr_folded", cfg=cf, **common)
    return out


def section_frame_fit(X, sec):
    """Rotation taking the truth to a frame in which every section has a constant last coordinate."""
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


def load_layout(path, spec):
    """A layout npz (keys P, G, cc, iso) as model-unit coordinates, after the cell identity checks."""
    tag = os.path.basename(path)
    if not os.path.exists(path):
        refuse(f"{tag}: missing {path}")
    z = np.load(path)
    for k in ("P", "G", "cc", "iso"):
        if k not in z.files:
            refuse(f"{tag}: npz lacks '{k}' (cell identity / unit flag cannot be checked)")
    if not bool(z["iso"]):
        refuse(f"{tag}: not a true-scale (iso) layout")
    P = as_coords(z["P"], tag, N=spec["n"], D=spec["X"].shape[1])
    if not np.array_equal(z["G"], spec["pos"]):
        refuse(f"{tag}: G != the test prebuilt pos.npy - different cells or a different order")
    if not np.array_equal(np.asarray(z["cc"]).astype(np.int64), spec["cell_class"]):
        refuse(f"{tag}: cc != the test prebuilt cell_class - different cells or a different order")
    return P


FOLDED_VIEWS = ("lr_folded",)


def own_sizes(Z, X):
    """The layout's own sorted principal SDs over the truth's."""
    return principal_sds(Z) / principal_sds(X)


def align_views(spec, Y, views=None, label=""):
    """Every fit of every view of one layout: {view: (aligned, size-matched or None, cell-fit layout, alignment record)}."""
    views = views or spec["views"]
    X = spec["X"]
    Y = as_coords(Y, label, N=len(X), D=X.shape[1])
    check_scale(X, Y, label)
    own = own_sizes(Y, X)
    flat = flat_layout(own)
    L = spec["labels"]
    Yd, dinfo = fit_cell_free(Y, X)
    if spec.get("main_fold") is not None:
        Yd = spec["main_fold"](Yd)
    out = {}
    common = dict(size_own_pc=own.tolist(), flat_layout=flat, flat_rule=f"smallest own principal size < {FLAT:g} x the largest")
    if "main" in views:
        fit_main = make_fitter(X, L["fine"], spec.get("main_fold"))
        Ya, ainfo = fit_main(Y)
        Ys = size_match(Ya, X, fit_main, flat)
        out["main"] = (Ya, Ys, Yd, dict(main=ainfo, sizematch="per-axis SD rescale about the mean, then the same fit again", cellfit=dinfo,
                                        fold=None if spec.get("main_fold") is None else "dataset fold after every fit",
                                        size_own_pc_sm=None if Ys is None else own_sizes(Ys, X).tolist(), **common))
    for v in FOLDED_VIEWS:
        if v not in views:
            continue
        if spec.get("main_fold") is not None:
            refuse(f"view {v}: a dataset fold and an LR fold together are not defined")
        lf = spec["lr_fold"]
        f = fold_lr_own(lf["axis"], lf["centre"])
        Z, finfo = fit_folded(Y, X, L["fine"], lf["axis"], lf["centre"], spec["um_per_unit"])
        Zs, sinfo = size_match_folded(Z, X, L["fine"], lf["axis"], lf["centre"], flat)
        out[v] = (f(Z), None if Zs is None else f(Zs), f(Yd),
                  dict(main=finfo, then=f"LR fold of axis {lf['axis']} about each layout's own {lf['centre']} ({lf['rule']})",
                       sizematch="the fitted frame's size-matched layout (per-axis spread match, fold axis: RMS about the own centre; the "
                                 "fit already chose its rotation on it: no refit), then the fold", sizematch_info=sinfo,
                       size_own_pc_sm=None if Zs is None else own_sizes(Zs, X).tolist(),
                       cellfit=dinfo, cellfit_fold="cell fit, then each layout's own-centre fold (diagnostic only)", **common))
    return out


def score_layout(spec, truths, Y, views=None, label=""):
    """Every metric of one layout in every view."""
    views = views or spec["views"]
    out = {}
    for v, (Ya, Ys, Yd, info) in align_views(spec, Y, views, label).items():
        r = score_view(truths["lr_folded" if v in FOLDED_VIEWS else "main"], Ya, Ys, Yd, own=info["size_own_pc"],
                       own_sm=info["size_own_pc_sm"])
        r["alignment"] = info
        out[v] = r
    return out


def _jsonable(o):
    """Plain Python types for JSON."""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def score_row(spec, truths, meta, make, out_path, base, views=None):
    """Score one layout and write its JSON record; a refused or failing layout is recorded with its reason."""
    views = views or spec["views"]
    t0 = time.time()
    tag = meta["tag"]
    try:
        r = score_layout(spec, truths, make(), views, label=tag)
        rec = dict(base, row=meta, status="ok", views=_jsonable(r), seconds=round(time.time() - t0, 1))
        hv = spec.get("headline_view") if spec.get("headline_view") in r else ("main" if "main" in r else next(iter(r)))
        log(f"  {tag[:60]:60s} [{hv}] {_oneline(r[hv])} | {rec['seconds']} s")
    except RefusedInput as e:
        rec = dict(base, row=meta, status="refused", reason=refusal_reason(e), views=None, seconds=round(time.time() - t0, 1))
        log(f"  {tag[:60]:60s} REFUSED: {refusal_reason(e)}")
    except Exception as e:
        rec = dict(base, row=meta, status="error", reason=f"{type(e).__name__}: {e}", traceback=traceback.format_exc(), views=None,
                   seconds=round(time.time() - t0, 1))
        log(f"  {tag[:60]:60s} ERROR: {type(e).__name__}: {e}")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    json.dump(rec, open(out_path, "w"), indent=1)
    return rec


def _oneline(r):
    fs = " ".join(f"F@{k[1:]} {r[k]:.3f}" for k in ("f20", "f100", "f200") if k in r)
    w2s = r.get("w2_um_sm")
    return (f"size {r['size_rms']:.3f} ({'/'.join(f'{x:.2f}' for x in r['size_axis'])}) | W2 {r['w2_um']:.1f} "
            f"({'-' if w2s is None else f'{w2s:.1f}'}) um | {fs} | contact r {r['contact_r_offdiag']:.3f} | NMI {r['niche_nmi']:.3f} | "
            f"Spearman {r['spearman']:.3f} | SRC {r['src_median']:.3f} | F1 {r['contact_f1_1pct']:.3f}")
