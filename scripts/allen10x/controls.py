#!/usr/bin/env python
"""Controls of the evaluator on training cells only (no 10x file): chance (label-only prediction of the random hit rate), a (ceiling:
Allen cells at their own positions, leave-one-out on the five-brain support and on the Zhuang-only support), b (global permutation),
c (type-geometry floor from two halves of one support cloud, and the distance between the Zhuang and Allen brains); selftest.
usage: controls.py {selftest|chance|a|b|c} [--out DIR]"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
import numpy as np

import endpoints as EV
import units as EU
import paths as P

SDIR = P.SUPPORT
SEED_CTRL = 25500
N_CTRL = 200_000


def t():
    return time.strftime("%H:%M:%S")


def selftest():
    rng = np.random.default_rng(0)
    L = np.array([[3, 3, 5, 5, 7], [1, 2, 2, 1, 9], [4, 4, 4, 6, 6], [-1, -1, 2, 2, 2]])
    w, s = EV.plurality(L)
    assert w.tolist() == [3, 1, 4, 2] and np.allclose(s, [0.4, 0.4, 0.6, 0.6]), (w, s)
    w, _ = EV.plurality(np.array([[-1, -1, -1, 2, 2]]))
    assert w[0] == -1
    A = rng.normal(size=(500, 3))
    assert EV.sw2_um(A, A) == 0.0
    sh = EV.sw2_um(A, A + np.array([0.1, 0, 0]))
    assert abs(sh - 100.0 * np.sqrt(np.mean(EV.DIRS[:, 0] ** 2))) < 1e-9, sh
    assert np.allclose(np.linalg.norm(EV.DIRS, axis=1), 1.0) and EV.DIRS.shape == (256, 3)
    assert np.array_equal(EV.DIRS, EV.sw2_dirs())

    class FakeSup(EV.Support):
        def __init__(self):
            n = 4000
            self.pos_cal = rng.uniform(0, 1, size=(n, 3)).astype(np.float32)
            self.unit = (self.pos_cal[:, 0] * 4).astype(np.int8)
            self.parc = (self.unit.astype(np.int16) * 10 + 1)
            self.subclass = (self.pos_cal[:, 1] * 3).astype(np.int16)
            self.brain = rng.integers(0, 5, n).astype(np.int8)
            self.parc2unit = np.full(100, -1, np.int16)
            for u in range(4):
                self.parc2unit[u * 10 + 1] = u
            self.sel = np.arange(n)
            self.workers = 1
            from scipy.spatial import cKDTree
            self.tree = cKDTree(self.pos_cal.astype(np.float64))
    sup = FakeSup()
    q = sup.pos_cal[:50].astype(np.float64)
    d, rows = sup.knn(q, k=15, self_rows=np.arange(50))
    assert d.shape == (50, 15) and (rows != np.arange(50)[:, None]).all() and (d[:, 0] > 0).all()
    d0, rows0 = sup.knn(q, k=15)
    assert (rows0[:, 0] == np.arange(50)).all() and (d0[:, 0] == 0).all()
    qi = np.column_stack([np.array([0.125, 0.375, 0.625, 0.875]), np.full(4, 0.5), np.full(4, 0.5)])
    r = sup.membership(qi, np.array([0, 1, 2, 3]))
    assert r["win"].tolist() == [0, 1, 2, 3] and all(r["hit"][500.0]) is True or r["hit"][500.0].all()
    r2 = sup.membership(qi + 5.0, np.array([0, 1, 2, 3]))
    assert not any(r2["hit"][R].any() for R in EV.R_UM)
    r3 = sup.membership(qi, np.array([0, 1, 2, 3]), placed=np.array([True, False, True, False]))
    assert r3["hit"][500.0].tolist() == [True, False, True, False]
    refs = EV.RefClouds(sup)
    cell_sub = rng.integers(0, 3, 600).astype(np.int16)
    Q, pl = EV.ref_type_conditional(refs, cell_sub, 1)
    assert pl.all() and np.isfinite(Q).all()
    assert all((sup.subclass[np.argmin(((sup.pos_cal[None, :, :].astype(np.float64)[0] - q_) ** 2).sum(1))] == c) for q_, c in zip(Q[:20], cell_sub[:20]))
    Y = rng.uniform(size=(600, 3))
    P = EV.ref_within_type_permutation(Y, cell_sub, 3)
    for c in range(3):
        assert np.allclose(np.sort(P[cell_sub == c], axis=0), np.sort(Y[cell_sub == c], axis=0))
    G = EV.ref_global_random(sup, 100, 5)
    assert G.shape == (100, 3)
    Qt, plt_ = EV.ref_type_conditional(refs, cell_sub, 11)
    r_floor = EV.sw2_layout(refs, Qt, cell_sub, 1)
    r_shift = EV.sw2_layout(refs, Qt + 0.3, cell_sub, 1)
    assert len(r_floor) == 3 and len(r_shift) == 3
    f_o = np.mean([x[4] for x in r_floor]); s_o = np.mean([x[4] for x in r_shift]); f_c = np.mean([x[5] for x in r_floor]); s_c = np.mean([x[5] for x in r_shift])
    assert s_o > 3 * f_o and abs(s_c - f_c) < 1e-6, (f_o, s_o, f_c, s_c)
    assert EV.sw2_layout(refs, Qt, cell_sub, 1) == r_floor
    cs2 = rng.integers(0, 3, 5000).astype(np.int16)
    Y2 = rng.uniform(size=(5000, 3))
    a1, a1b, a2 = EV.sw2_layout(refs, Y2, cs2, 1), EV.sw2_layout(refs, Y2, cs2, 1), EV.sw2_layout(refs, Y2, cs2, 2)
    assert a1 == a1b and a1 != a2 and all(x[3] == 1000 for x in a1)
    print("selftest OK")


def load_support(brains=(0, 1, 2, 3, 4), workers=6):
    t0 = time.time()
    s = EV.Support(SDIR, brains=brains, workers=workers)
    print(f"[{t()}] support {len(s):,} cells (brains {brains}), KD-tree built in {s.build_seconds} s (load {time.time() - t0:.0f} s)", flush=True)
    return s


def allen_control_cells(sup):
    ok = np.flatnonzero((sup.brain == 4) & (sup.unit >= 0))
    pick = np.sort(np.random.default_rng(SEED_CTRL).choice(ok, N_CTRL, replace=False))
    return pick


def chance(out):
    sup = load_support()
    rng = np.random.default_rng(25501)
    idx = np.sort(rng.choice(len(sup), 1_000_000, replace=False))
    rows_q = sup.sel[idx]
    res = sup.membership(sup.pos_cal[rows_q].astype(np.float64), np.zeros(len(rows_q), np.int8))
    win = res["win"]
    q_vote = np.array([(win == u).mean() for u in range(23)])
    q_none = float((win == -1).mean())
    ctrl = allen_control_cells(sup)
    p_ctrl = np.bincount(sup.unit[ctrl], minlength=23) / len(ctrl)
    import csv
    roi = {r["region_of_interest_acronym"]: int(r["n_cells"]) for r in csv.DictReader(open(f"{P.FROZEN}/pool_by_roi.csv"))}
    p_pool = np.zeros(23)
    for r_, c in roi.items():
        p_pool[EU.UNIT_NAMES.index(EU.ROI_TO_UNIT[r_])] += c
    p_pool /= p_pool.sum()
    quo = json.load(open(f"{P.FROZEN}/strat_quotas.json"))
    p_strat = np.array(quo["n_final"], dtype=np.float64) / quo["N"]
    pred = {}
    for name, p in (("allen_control_cells", p_ctrl), ("primary_10x_cohort_expected_from_S1_roi_counts", p_pool), ("stratified_10x_cohort_quotas", p_strat)):
        pred[name] = dict(micro=float((p * q_vote).sum()), macro=float(q_vote[p > 0].mean()), p_unit=p.tolist())
    out_d = dict(q_vote=q_vote.tolist(), q_none=q_none, n_random_support_positions=len(idx), seed=25501, predictions=pred, unit_names=EU.UNIT_NAMES,
                 note="chance = P(plurality unit of the 15 nearest support cells == unit) at a position drawn at random from the support; micro = sum_u p_u q_vote_u; macro = mean over units with p_u > 0")
    json.dump(out_d, open(f"{out}/chance_prediction.json", "w"), indent=1)
    print(f"[{t()}] q_vote (units): " + " ".join(f"{EU.UNIT_NAMES[u]}={q_vote[u]:.4f}" for u in range(23)) + f" | none={q_none:.4f}")
    for k, v in pred.items():
        print(f"  predicted chance level  {k}: micro {v['micro']:.4f}  macro {v['macro']:.4f}")


def control_a(out):
    sup = load_support()
    ctrl = allen_control_cells(sup)
    Q = sup.pos_cal[ctrl].astype(np.float64)
    cu = sup.unit[ctrl]
    resA = {}
    t0 = time.time()
    r = sup.membership(Q, cu, self_rows=ctrl)
    resA["support_all5_loo"] = EV.summarize_membership(r, cu)
    print(f"[{t()}] (a) support = all five brains, leave-one-out ({time.time() - t0:.0f} s): " + ", ".join(f"R{int(R)} micro {resA['support_all5_loo'][f'R{int(R)}']['hit']['micro']:.4f} macro {resA['support_all5_loo'][f'R{int(R)}']['hit']['macro']:.4f}" for R in EV.R_UM), flush=True)
    pu = resA["support_all5_loo"]["R500"]["hit"]
    print("   per unit (R500): " + " ".join(f"{EU.UNIT_NAMES[u]}={pu['per_unit'][u]:.3f}" for u in range(23)))
    sup4 = load_support(brains=(0, 1, 2, 3))
    r4 = sup4.membership(Q, cu)
    resA["support_zhuang1to4"] = EV.summarize_membership(r4, cu)
    print(f"[{t()}] (a2) support = Zhuang 1-4 only (cross-brain): " + ", ".join(f"R{int(R)} micro {resA['support_zhuang1to4'][f'R{int(R)}']['hit']['micro']:.4f} macro {resA['support_zhuang1to4'][f'R{int(R)}']['hit']['macro']:.4f}" for R in EV.R_UM), flush=True)
    resA.update(n_cells=int(len(ctrl)), seed=SEED_CTRL, rule="Allen hemisphere cells with a unit, default_rng(25500).choice, at their own pos_cal_mm; label = the cell's own unit")
    json.dump(resA, open(f"{out}/control_a.json", "w"), indent=1)
    return sup, ctrl, resA


def control_b(out, sup=None, ctrl=None):
    sup = sup or load_support()
    ctrl = allen_control_cells(sup) if ctrl is None else ctrl
    Q = sup.pos_cal[ctrl].astype(np.float64)
    cu = sup.unit[ctrl]
    perm = np.random.default_rng(25502).permutation(len(ctrl))
    r = sup.membership(Q[perm], cu)
    s = EV.summarize_membership(r, cu)
    pred = json.load(open(f"{out}/chance_prediction.json"))["predictions"]["allen_control_cells"]
    print(f"[{t()}] (b) globally permuted positions: micro {s['R500']['hit']['micro']:.4f} (predicted {pred['micro']:.4f}), macro {s['R500']['hit']['macro']:.4f} (predicted {pred['macro']:.4f})", flush=True)
    json.dump(dict(measured=s, predicted=pred, seed=25502), open(f"{out}/control_b.json", "w"), indent=1)


def control_c(out):
    sup = load_support()
    refs = EV.RefClouds(sup)
    print(f"[{t()}] eligible subclasses (>= {EV.MIN_REF_CELLS} cells in >= {EV.MIN_REF_BRAINS} brains): {len(refs.eligible)} of {refs.n_sub}", flush=True)
    res = dict(n_eligible=int(len(refs.eligible)), floor=[], between_brain=None)
    for sd in EV.REF_SEEDS:
        rng = np.random.default_rng(sd)
        rows_o, rows_c = [], []
        for s in refs.eligible:
            rr = refs.rows[s]
            p = rng.permutation(len(rr))
            h = len(rr) // 2
            a, b = rr[p[:h]], rr[p[h:2 * h]]
            n = min(h, EV.SW2_NMAX)
            A, B = sup.pos_cal[a[:n]].astype(np.float64), sup.pos_cal[b[:n]].astype(np.float64)
            mA, mB = sup.pos_cal[a].astype(np.float64).mean(0), sup.pos_cal[b].astype(np.float64).mean(0)
            rows_o.append(EV.sw2_um(A, B)); rows_c.append(EV.sw2_um(A - mA, B - mB))
        res["floor"].append(dict(seed=sd, orig_macro=float(np.mean(rows_o)), centred_macro=float(np.mean(rows_c))))
    fo = np.array([f["orig_macro"] for f in res["floor"]]); fc = np.array([f["centred_macro"] for f in res["floor"]])
    res["floor_summary"] = dict(orig_macro_mean=float(fo.mean()), orig_macro_sd=float(fo.std(ddof=1)), centred_macro_mean=float(fc.mean()), centred_macro_sd=float(fc.std(ddof=1)))
    print(f"[{t()}] (c) finite-sample floor (two disjoint halves, n <= 1000, 10 seeds): original frame {fo.mean():.2f} +- {fo.std(ddof=1):.2f} um, centred {fc.mean():.2f} +- {fc.std(ddof=1):.2f} um", flush=True)
    rows_o, rows_c, ns = [], [], []
    for s in refs.eligible:
        rr = refs.rows[s]
        z, a = rr[sup.brain[rr] <= 3], rr[sup.brain[rr] == 4]
        if len(z) < EV.MIN_REF_CELLS or len(a) < EV.MIN_REF_CELLS:
            continue
        rng = np.random.default_rng([EV.SW2_SEED, 777, int(s)])
        n = min(len(z), len(a), EV.SW2_NMAX)
        zi, ai = rng.choice(len(z), n, replace=False), rng.choice(len(a), n, replace=False)
        A, B = sup.pos_cal[z[zi]].astype(np.float64), sup.pos_cal[a[ai]].astype(np.float64)
        mA, mB = sup.pos_cal[z].astype(np.float64).mean(0), sup.pos_cal[a].astype(np.float64).mean(0)
        rows_o.append(EV.sw2_um(A, B)); rows_c.append(EV.sw2_um(A - mA, B - mB)); ns.append(n)
    res["between_brain"] = dict(n_subclasses=len(rows_o), orig_macro=float(np.mean(rows_o)), centred_macro=float(np.mean(rows_c)),
                                note="Zhuang 1-4 pooled vs Allen hemisphere, per subclass with >= 200 cells in both; same SW2 settings; one fixed subsample")
    print(f"[{t()}] (c2) brain-to-brain (Zhuang 1-4 vs Allen hemisphere, {len(rows_o)} subclasses): original {np.mean(rows_o):.2f} um, centred {np.mean(rows_c):.2f} um", flush=True)
    json.dump(res, open(f"{out}/control_c.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["selftest", "chance", "a", "b", "c"])
    ap.add_argument("--out", default=P.FROZEN)
    A = ap.parse_args()
    os.makedirs(A.out, exist_ok=True)
    {"selftest": lambda: selftest(), "chance": lambda: chance(A.out), "a": lambda: control_a(A.out), "b": lambda: control_b(A.out), "c": lambda: control_c(A.out), "pipeline": lambda: pipeline_test()}[A.what]()
