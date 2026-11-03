#!/usr/bin/env python
"""Region membership by the official CCFv3 annotation at the translated positions: hit = annotation unit of the translated position equals
the cell's dissection unit and the nearest translating training cell lies within R (500 um; 250 / 1,000 um).  Statistics, aggregation
and paired contrasts as summarize.py / contrasts.py.
usage: ccf_membership.py layouts|refs|summary --kind primary|strat"""
import argparse
import json
import os
import types

import numpy as np

import ccf_common as C
import ccf_annotation as VA
import ccf_translate as VT
import endpoints as EV
import application as EP
import summarize as ES
import contrasts as AR
import units as EU

MEM = f"{C.V}/member"
R_KEYS = ("R500", "R250", "R1000")


def member_res(un, ok_brain, d1, cu, placed=None):
    N = len(un)
    placed = np.ones(N, bool) if placed is None else np.asarray(placed, bool)
    d = np.where(placed, d1, np.inf)
    hit = {R: placed & ok_brain & (un == cu) & (cu >= 0) & (d <= R / 1000.0) for R in EV.R_UM}
    return dict(d1=d, win=un, win_fine=un, share=np.ones(N, np.float32), placed=placed, hit=hit, hit_fine=hit)


def eval_positions(ann, ccf, d1, cu, placed=None):
    ids, un, ok = ann.lookup(np.where(np.isfinite(ccf), ccf, -1.0))
    inside = ok & (ids != 0)
    res = member_res(un, inside, d1, cu, placed)
    sm = EV.summarize_membership(res, cu)
    extra = dict(frac_outside_brain=float((~inside).mean()), frac_inside_no_unit=float((inside & (un < 0)).mean()))
    return sm, extra, res, un, inside


def sup_stub():
    sup = types.SimpleNamespace()
    sup.pos_cal = np.load(f"{C.SUPPORT}/support_pos_cal_mm.npy")
    sup.subclass = np.load(f"{C.SUPPORT}/support_subclass.npy")
    sup.brain = np.load(f"{C.SUPPORT}/support_brain.npy")
    sup.sel = np.arange(len(sup.brain))
    return sup


def run_layouts(kind):
    os.makedirs(MEM, exist_ok=True)
    ann = VA.Annotation()
    labels = EP.EvalLabels()
    for c in C.COHORTS:
        rows_c = np.load(f"{C.APP}/cohorts/{kind}{c}/rows.npy")
        cu = labels.unit[rows_c].astype(np.int16)
        cs = labels.subclass[rows_c]
        for arm in C.ARMS:
            for s in C.SEEDS:
                tag = f"{arm}_{kind}{c}_s{s}"
                z = np.load(f"{C.LAYOUT_CCF}/{tag}.npz")
                assert (z["rows"] == rows_c).all()
                ccf, d1 = z["ccf_xyz_mm"].astype(np.float64), z["d1_mm"].astype(np.float64)
                sm, extra, res, un, inside = eval_positions(ann, ccf, d1, cu)
                perm = []
                for sd in EV.REF_SEEDS:
                    idx = EV.ref_within_type_permutation(np.repeat(np.arange(len(cs), dtype=np.float64)[:, None], 3, 1), cs, sd)[:, 0].astype(np.int64)
                    rp = member_res(un[idx], inside[idx], d1[idx], cu)
                    sp = EV.summarize_membership(rp, cu)
                    perm.append(dict(seed=sd, **{k: dict(macro=v["hit"]["macro"], micro=v["hit"]["micro"]) for k, v in sp.items()}))
                row = dict(tag=tag, arm=arm, kind=kind, cohort=c, seed=s, n_cells=int(len(cu)), membership=sm, extra=extra, within_type_permutation=perm)
                C.jdump(row, f"{MEM}/{tag}.json")
                np.savez_compressed(f"{MEM}/{tag}_cells.npz", rows=rows_c, hit500=res["hit"][500.0], hit250=res["hit"][250.0], hit1000=res["hit"][1000.0], unit_ann=un.astype(np.int8),
                                    d1=d1.astype(np.float32))
                m = sm["R500"]["hit"]
                C.log(f"{tag}: R500 macro {m['macro']:.4f} micro {m['micro']:.4f}  outside brain {extra['frac_outside_brain']:.4f} no-unit {extra['frac_inside_no_unit']:.4f}  R3 macro {np.mean([p['R500']['macro'] for p in perm]):.4f}")


def run_refs(kind):
    os.makedirs(MEM, exist_ok=True)
    ann = VA.Annotation()
    labels = EP.EvalLabels()
    sup = sup_stub()
    refs = EV.RefClouds(sup)
    T = VT.Translator.load()
    C.log(f"translator ready ({T.build_seconds} s); eligible subclasses {len(refs.eligible)}")
    n = 100_000
    r1 = {}
    keep = {}
    for sd in EV.REF_SEEDS:
        Q = EV.ref_global_random(sup, n, sd)
        o = T.translate(Q)
        ids, un, ok = ann.lookup(o["ccf_xyz_mm"])
        r1[sd] = (un, ok & (ids != 0), o["d1_mm"])
        keep[f"R1_s{sd}_ccf"] = o["ccf_xyz_mm"].astype(np.float32)
        keep[f"R1_s{sd}_d1"] = o["d1_mm"].astype(np.float32)
    out = {}
    for c in C.COHORTS:
        rows_c = np.load(f"{C.APP}/cohorts/{kind}{c}/rows.npy")
        cu = labels.unit[rows_c].astype(np.int16)
        cs = labels.subclass[rows_c]
        R1, R2 = [], []
        for sd in EV.REF_SEEDS:
            un, ins, d1 = r1[sd]
            R1.append(dict(seed=sd, membership=EV.summarize_membership(member_res(un, ins, d1, cu), cu)))
            Q2, placed = EV.ref_type_conditional(refs, cs, sd)
            o = T.translate(np.where(placed[:, None], Q2, 0.0))
            ids, un2, ok = ann.lookup(o["ccf_xyz_mm"])
            r = member_res(un2, ok & (ids != 0), o["d1_mm"], cu, placed=placed)
            R2.append(dict(seed=sd, placed_frac=float(placed.mean()), membership=EV.summarize_membership(r, cu)))
            keep[f"R2_c{c}_s{sd}_ccf"] = o["ccf_xyz_mm"].astype(np.float32)
            keep[f"R2_c{c}_s{sd}_d1"] = o["d1_mm"].astype(np.float32)
            keep[f"R2_c{c}_s{sd}_placed"] = placed
        out[c] = dict(R1=R1, R2=R2)
        m1 = np.mean([r["membership"]["R500"]["hit"]["macro"] for r in R1]); m2 = np.mean([r["membership"]["R500"]["hit"]["macro"] for r in R2])
        C.log(f"{kind}{c}: R1 macro {m1:.4f}  R2 macro {m2:.4f}")
    C.jdump(out, f"{MEM}/refs_{kind}.json")
    np.savez_compressed(f"{MEM}/refs_ccf_{kind}.npz", **keep)


def fm(x, nd=4):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def run_summary(kind):
    import csv
    labels = EP.EvalLabels()
    rows = {}
    for arm in C.ARMS:
        for c in C.COHORTS:
            for s in C.SEEDS:
                rows[(arm, c, s)] = json.load(open(f"{MEM}/{arm}_{kind}{c}_s{s}.json"))
    refs = json.load(open(f"{MEM}/refs_{kind}.json"))
    res = dict(kind=kind, arms={}, references={}, paired={}, per_unit={}, n_layouts=len(rows))
    for arm in C.ARMS:
        sel = {(c, s): r for (a, c, s), r in rows.items() if a == arm}
        m = {}
        for R in R_KEYS:
            for stat in ("macro", "micro"):
                m[f"{R}_{stat}"] = ES.cohort_means({k: r["membership"][R]["hit"][stat] for k, r in sel.items()})
        m["R500_outside_support"] = ES.cohort_means({k: r["membership"]["R500"]["outside_frac"] for k, r in sel.items()})
        m["frac_outside_brain"] = ES.cohort_means({k: r["extra"]["frac_outside_brain"] for k, r in sel.items()})
        m["frac_inside_no_unit"] = ES.cohort_means({k: r["extra"]["frac_inside_no_unit"] for k, r in sel.items()})
        m["R3_macro_R500"] = ES.cohort_means({k: float(np.mean([p["R500"]["macro"] for p in r["within_type_permutation"]])) for k, r in sel.items()})
        m["R3_micro_R500"] = ES.cohort_means({k: float(np.mean([p["R500"]["micro"] for p in r["within_type_permutation"]])) for k, r in sel.items()})
        res["arms"][arm] = m
        pu = np.array([r["membership"]["R500"]["hit"]["per_unit"] for r in sel.values()], dtype=np.float64)
        res["per_unit"][arm] = np.nanmean(pu, axis=0).tolist()
    for ref in ("R1", "R2"):
        cm = {}
        for R in R_KEYS:
            for stat in ("macro", "micro"):
                v = {(c, 0): float(np.mean([r["membership"][R]["hit"][stat] for r in refs[str(c)][ref]])) for c in C.COHORTS}
                cm[f"{R}_{stat}"] = ES.cohort_means(v)
        if ref == "R2":
            cm["placed_frac"] = ES.cohort_means({(c, 0): float(np.mean([r["placed_frac"] for r in refs[str(c)][ref]])) for c in C.COHORTS})
        res["references"][ref] = cm
        res["per_unit"][ref] = np.nanmean(np.array([[r["membership"]["R500"]["hit"]["per_unit"] for r in refs[str(c)][ref]] for c in C.COHORTS], dtype=np.float64).reshape(-1, 23), axis=0).tolist()
    rows_k = {k: None for k in rows}
    db = ES.donor_bootstrap(MEM, kind, rows_k, labels)
    pairs = AR.PAIRS
    seed_sd = {}
    for a, b, note in pairs:
        pr = dict(A=a, B=b, note=note, endpoints={})
        for j, nm in ((0, "R500 macro hit rate"), (1, "R500 micro hit rate")):
            d = db[a][:, j] - db[b][:, j]
            lo, hi = ES.ci(d)
            pr["endpoints"][nm] = dict(mean=float(d.mean()), ci=[lo, hi], margin=AR.M_HIT, verdict=AR.classify(lo, hi, AR.M_HIT, True), primary=(j == 0), bootstrap="donor, B=1000 seed 25400")
        dm = [np.mean([rows[(a, c, s)]["membership"]["R500"]["hit"]["macro"] - rows[(b, c, s)]["membership"]["R500"]["hit"]["macro"] for c in C.COHORTS]) for s in C.SEEDS]
        pr["per_seed_macro_difference"] = [float(x) for x in dm]
        pr["two_sd_over_seeds_macro_difference"] = float(2 * np.std(dm, ddof=1))
        pr["overall"] = AR.overall({k: v["verdict"] for k, v in pr["endpoints"].items() if v["primary"]})
        res["paired"][f"{a}__minus__{b}"] = pr
    C.jdump(res, f"{C.RES}/ccf_member_{kind}.json")
    with open(f"{C.RES}/ccf_member_{kind}_per_unit.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unit"] + list(res["per_unit"].keys()))
        for u in range(23):
            w.writerow([EU.UNIT_NAMES[u]] + [fm(res["per_unit"][k][u]) for k in res["per_unit"]])
    for arm, m in res["arms"].items():
        C.log(f"{kind} {arm}: R500 macro {fm(m['R500_macro'][0])} +- {fm(m['R500_macro'][1])}  micro {fm(m['R500_micro'][0])}  R3 macro {fm(m['R3_macro_R500'][0])}  outside-brain {fm(m['frac_outside_brain'][0])} no-unit {fm(m['frac_inside_no_unit'][0])}")
    for ref, m in res["references"].items():
        C.log(f"{kind} {ref}: R500 macro {fm(m['R500_macro'][0])} micro {fm(m['R500_micro'][0])}")
    for k, pr in res["paired"].items():
        e = pr["endpoints"]["R500 macro hit rate"]
        C.log(f"{kind} paired {k}: macro {e['mean'] * 100:+.2f} pp [{e['ci'][0] * 100:+.2f}, {e['ci'][1] * 100:+.2f}] -> {e['verdict']}; overall {pr['overall']}; 2xSD over seeds {pr['two_sd_over_seeds_macro_difference'] * 100:.2f} pp")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["layouts", "refs", "summary"])
    ap.add_argument("--kind", default="primary", choices=list(C.KINDS))
    a = ap.parse_args()
    {"layouts": run_layouts, "refs": run_refs, "summary": run_summary}[a.cmd](a.kind)
