#!/usr/bin/env python
"""Endpoint evaluator of the dissociated-cell application and its reference rows.

Frame: physical mm = training-ruler mm x S_CAL per axis.
Region membership: for each placed cell the k = 15 nearest training-support cells (all cells of the five training brains);
hit = the plurality unit of those 15 equals the cell's dissection unit and the nearest lies within R (500 um; 250 / 1,000 um).
Type geometry: per subclass with >= 200 support cells in >= 2 training brains and >= 50 placed cells, the equal-mass sliced
Wasserstein-2 distance (256 directions) between the placed cells and the support cells of that subclass, original and centred.
Reference rows (10 seeds): global random placement, type-conditional random placement, within-type permutation."""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import paths as P
S_CAL = np.array([1.0185775453536927, 1.2931099417291327, 0.9995762396395721])
ISO_MM = 13.45773983001709
N_UNITS = 23
K_NN = 15
R_UM = (500.0, 250.0, 1000.0)
SW2_DIRS, SW2_DIR_SEED, SW2_SEED, SW2_NMAX = 256, 25300, 25301, 1000
MIN_REF_CELLS, MIN_REF_BRAINS, MIN_QUERY = 200, 2, 50
REF_SEEDS = tuple(range(26001, 26011))
PARC_UNIT_CSV = f"{P.FROZEN}/parcellation_to_unit.csv"


def sw2_dirs():
    d = np.random.default_rng(SW2_DIR_SEED).normal(size=(SW2_DIRS, 3))
    return d / np.linalg.norm(d, axis=1, keepdims=True)


DIRS = sw2_dirs()


def plurality(L, chunk=200_000):
    N, k = L.shape
    win = np.empty(N, dtype=L.dtype)
    share = np.empty(N, dtype=np.float32)
    ar = np.arange(k)[None, :]
    for s in range(0, N, chunk):
        l = L[s:s + chunk]
        eq = (l[:, :, None] == l[:, None, :]).sum(2)
        j = (eq * 100 - ar).argmax(1)
        r = np.arange(len(l))
        win[s:s + chunk] = l[r, j]
        share[s:s + chunk] = eq[r, j] / k
    return win, share


class Support:
    def __init__(self, sdir, brains=(0, 1, 2, 3, 4), workers=4):
        self.sdir = sdir
        self.workers = workers
        ld = lambda n: np.load(f"{sdir}/support_{n}.npy")
        self.pos_cal = ld("pos_cal_mm")
        self.unit = ld("unit")
        self.parc = ld("parc")
        self.subclass = ld("subclass")
        self.brain = ld("brain")
        pm = pd.read_csv(PARC_UNIT_CSV)
        self.parc2unit = np.full(int(max(pm.parcellation_index.max(), self.parc.max())) + 2, -1, dtype=np.int16)
        self.parc2unit[pm.parcellation_index.to_numpy()] = pm.unit.to_numpy()
        self.sel = np.flatnonzero(np.isin(self.brain, brains))
        t0 = time.time()
        self.tree = cKDTree(self.pos_cal[self.sel].astype(np.float64), leafsize=32, balanced_tree=False, compact_nodes=False)
        self.build_seconds = round(time.time() - t0, 1)

    def __len__(self):
        return len(self.sel)

    def knn(self, Qcal, k=K_NN, self_rows=None):
        N = len(Qcal)
        kk = k + (1 if self_rows is not None else 0)
        d, ii = self.tree.query(Qcal, k=kk, workers=self.workers)
        rows = self.sel[ii]
        if self_rows is not None:
            is_self = rows == np.asarray(self_rows)[:, None]
            drop = np.where(is_self.any(1), is_self.argmax(1), kk - 1)
            keep = np.ones(rows.shape, dtype=bool)
            keep[np.arange(N), drop] = False
            rows, d = rows[keep].reshape(N, k), d[keep].reshape(N, k)
        return d, rows

    def membership(self, Qcal, cell_unit, self_rows=None, placed=None):
        N = len(Qcal)
        placed = np.ones(N, dtype=bool) if placed is None else np.asarray(placed, dtype=bool)
        win = np.full(N, -2, dtype=np.int16); share = np.zeros(N, dtype=np.float32); winf = np.full(N, -2, dtype=np.int16)
        d1 = np.full(N, np.inf)
        if placed.any():
            q = Qcal[placed]
            d, rows = self.knn(q, self_rows=None if self_rows is None else np.asarray(self_rows)[placed])
            w, sh = plurality(self.unit[rows].astype(np.int16))
            pf, _ = plurality(self.parc[rows].astype(np.int16))
            win[placed], share[placed], d1[placed] = w, sh, d[:, 0]
            winf[placed] = np.where(pf >= 0, self.parc2unit[np.clip(pf, 0, None)], -1)
        cu = np.asarray(cell_unit)
        hit = {R: placed & (win == cu) & (cu >= 0) & (d1 <= R / 1000.0) for R in R_UM}
        hit_fine = {R: placed & (winf == cu) & (cu >= 0) & (d1 <= R / 1000.0) for R in R_UM}
        return dict(d1=d1, win=win, win_fine=winf, share=share, placed=placed, hit=hit, hit_fine=hit_fine)


def summarize_membership(res, cell_unit, cell_roi=None, roi_names=None, weights=None):
    cu = np.asarray(cell_unit)
    w = np.ones(len(cu)) if weights is None else np.asarray(weights, dtype=np.float64)
    out = {}
    for R in R_UM:
        key = f"R{int(R)}"
        o = {}
        for name, h in (("hit", res["hit"][R]), ("hit_fine", res["hit_fine"][R])):
            h = h.astype(np.float64)
            per_unit, n_unit = [], []
            for u in range(N_UNITS):
                m = cu == u
                ws = w[m].sum()
                per_unit.append(float((h[m] * w[m]).sum() / ws) if ws > 0 else float("nan"))
                n_unit.append(int(m.sum()))
            pu = np.array(per_unit)
            o[name] = dict(macro=float(np.nanmean(pu)), micro=float((h * w).sum() / w.sum()), per_unit=per_unit, n_per_unit=n_unit)
            if name == "hit" and cell_roi is not None and weights is None:
                per_roi = {}
                for ri, rn in enumerate(roi_names):
                    m = np.asarray(cell_roi) == ri
                    if m.any():
                        per_roi[rn] = dict(n=int(m.sum()), hit=float(h[m].mean()))
                o[name]["per_roi"] = per_roi
        placed = res["placed"]
        o["placed_frac"] = float(np.average(placed, weights=w))
        o["outside_frac"] = float(np.average(~(res["d1"] <= R / 1000.0), weights=w))
        pl = res["placed"]
        o["micro_over_placed"] = float((res["hit"][R].astype(np.float64) * w)[pl].sum() / w[pl].sum()) if pl.any() else float("nan")
        o["support_confidence_mean"] = float(np.average(res["share"][pl], weights=w[pl])) if pl.any() else float("nan")
        out[key] = o
    return out


def sw2_um(A, B):
    pa, pb = np.sort(A @ DIRS.T, axis=0), np.sort(B @ DIRS.T, axis=0)
    return float(np.sqrt(np.mean((pa - pb) ** 2)) * 1000.0)


class RefClouds:
    def __init__(self, sup):
        self.sup = sup
        sc = sup.subclass[sup.sel]
        self.n_sub = int(sc.max()) + 1
        order = np.argsort(sc, kind="stable")
        b = np.searchsorted(sc[order], np.arange(self.n_sub + 1))
        self.rows = [sup.sel[order[b[s]:b[s + 1]]] for s in range(self.n_sub)]
        nb = np.zeros((self.n_sub, 5), dtype=np.int64)
        for s in range(self.n_sub):
            if len(self.rows[s]):
                nb[s] = np.bincount(sup.brain[self.rows[s]], minlength=5)
        self.n_per_brain = nb
        self.eligible = np.flatnonzero((nb >= MIN_REF_CELLS).sum(1) >= MIN_REF_BRAINS)
        self._pr = {}
        self.mean = {s: sup.pos_cal[self.rows[s]].astype(np.float64).mean(0) for s in self.eligible}

    def ref_perm(self, s):
        if s not in self._pr:
            n = len(self.rows[s])
            self._pr[s] = np.random.default_rng([SW2_SEED, 999999, int(s), 1]).choice(n, min(n, SW2_NMAX), replace=False)
        return self._pr[s]


def sw2_layout(refs, Ycal, cell_sub, cohort_key, valid=None):
    sup = refs.sup
    valid = np.ones(len(Ycal), dtype=bool) if valid is None else valid
    cs = np.asarray(cell_sub)
    rows = []
    order = np.argsort(np.where(valid, cs, -1), kind="stable")
    cs_sorted = np.where(valid, cs, -1)[order]
    for s in refs.eligible:
        lo, hi = np.searchsorted(cs_sorted, s, "left"), np.searchsorted(cs_sorted, s, "right")
        nq = hi - lo
        if nq < MIN_QUERY:
            continue
        iq = order[lo:hi]
        nr = len(refs.rows[s])
        n = min(nq, nr, SW2_NMAX)
        pq = np.random.default_rng([SW2_SEED, int(cohort_key), int(s), 0]).choice(nq, n, replace=False)
        pr = refs.ref_perm(s)[:n]
        A = Ycal[iq[pq]].astype(np.float64)
        B = sup.pos_cal[refs.rows[s][pr]].astype(np.float64)
        mA, mB = Ycal[iq].astype(np.float64).mean(0), refs.mean[s]
        rows.append((int(s), int(nq), int(nr), int(n), sw2_um(A, B), sw2_um(A - mA, B - mB)))
    return rows


def summarize_sw2(rows, n_cells_valid, n_subclass_total_cells=None):
    if not rows:
        return dict(n_eligible_subclasses=0)
    a = np.array(rows, dtype=np.float64)
    w = a[:, 1]
    return dict(n_eligible_subclasses=int(len(a)), cells_in_eligible=int(w.sum()), coverage=float(w.sum() / n_cells_valid),
                orig_macro=float(a[:, 4].mean()), centred_macro=float(a[:, 5].mean()),
                orig_micro=float((a[:, 4] * w).sum() / w.sum()), centred_micro=float((a[:, 5] * w).sum() / w.sum()),
                per_subclass=a.tolist())


def ref_global_random(sup, n, seed):
    idx = np.random.default_rng(seed).choice(len(sup.sel), n, replace=False)
    return sup.pos_cal[sup.sel[idx]].astype(np.float64)


def ref_type_conditional(refs, cell_sub, seed):
    cs = np.asarray(cell_sub)
    rng = np.random.default_rng(seed)
    Q = np.full((len(cs), 3), np.nan)
    placed = np.zeros(len(cs), dtype=bool)
    order = np.argsort(cs, kind="stable")
    cso = cs[order]
    for s in np.unique(cs):
        if s < 0 or s >= refs.n_sub or len(refs.rows[s]) == 0:
            continue
        lo, hi = np.searchsorted(cso, s, "left"), np.searchsorted(cso, s, "right")
        ic = order[lo:hi]
        pick = refs.rows[s][rng.integers(0, len(refs.rows[s]), len(ic))]
        Q[ic] = refs.sup.pos_cal[pick]
        placed[ic] = True
    return Q, placed


def ref_within_type_permutation(Ycal, cell_sub, seed):
    cs = np.asarray(cell_sub)
    rng = np.random.default_rng(seed)
    out = np.array(Ycal, copy=True)
    order = np.argsort(cs, kind="stable")
    cso = cs[order]
    for s in np.unique(cs):
        if s < 0:
            continue
        lo, hi = np.searchsorted(cso, s, "left"), np.searchsorted(cso, s, "right")
        ic = order[lo:hi]
        out[ic] = Ycal[ic[rng.permutation(len(ic))]]
    return out


def evaluate_layout(sup, refs, Ycal, cell_unit, cell_sub, cohort_key, cell_roi=None, roi_names=None, with_sw2=True, self_rows=None, placed=None):
    placed = np.ones(len(Ycal), dtype=bool) if placed is None else np.asarray(placed, dtype=bool)
    Q = np.where(placed[:, None], Ycal, 0.0)
    res = sup.membership(Q, cell_unit, self_rows=self_rows, placed=placed)
    out = dict(n_cells=int(len(Ycal)), membership=summarize_membership(res, cell_unit, cell_roi, roi_names))
    if with_sw2:
        out["sw2"] = summarize_sw2(sw2_layout(refs, Ycal, cell_sub, cohort_key, valid=placed), int(placed.sum()))
    return out, res
