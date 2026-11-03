#!/usr/bin/env python
"""Placed position -> official CCF coordinate: physical mm -> mean official [AP, DV, ML] coordinate of the k = 15 nearest training
cells with a finite official CCF row (cKDTree); also the distance to the nearest and the 15th of them.
subcommands: validate (translation of the training cells themselves) | layouts (every layout -> application/layouts_ccf) | manifest"""
import argparse
import json
import os
import time

import numpy as np
from scipy.spatial import cKDTree

import ccf_common as C

WORKERS = int(os.environ.get("CCF_WORKERS", "5"))


class Translator:
    def __init__(self, pos_cal, ccf, donor_rows, k=C.K_NN, workers=WORKERS):
        self.rows = np.asarray(donor_rows, dtype=np.int64)
        self.k, self.workers = k, workers
        t0 = time.time()
        self.tree = cKDTree(np.asarray(pos_cal[self.rows], dtype=np.float64), leafsize=32, balanced_tree=False, compact_nodes=False)
        self.ccf = np.asarray(ccf[self.rows], dtype=np.float64)
        assert np.isfinite(self.ccf).all()
        self.build_seconds = round(time.time() - t0, 1)

    @classmethod
    def load(cls, k=C.K_NN):
        pos = np.load(f"{C.SUPPORT}/support_pos_cal_mm.npy")
        ccf = np.load(f"{C.CCF_DIR}/support_ccf_xyz_mm.npy")
        return cls(pos, ccf, np.flatnonzero(np.isfinite(ccf).all(1)), k)

    def translate(self, Qcal):
        d, ii = self.tree.query(np.asarray(Qcal, dtype=np.float64), k=self.k, workers=self.workers)
        nb = self.ccf[ii]
        m = nb.mean(1)
        spread = np.sqrt(((nb - m[:, None, :]) ** 2).sum(2).mean(1))
        return dict(ccf_xyz_mm=m, d1_mm=d[:, 0], d15_mm=d[:, -1], spread_mm=spread)

    def translate_training_ruler(self, Y_mm):
        return self.translate(np.asarray(Y_mm, dtype=np.float64) * C.S_CAL)


def stats(err):
    e = np.asarray(err)
    return dict(n=int(len(e)), median=float(np.median(e)), p90=float(np.percentile(e, 90)), p99=float(np.percentile(e, 99)), mean=float(e.mean()))


def err_block(pred, true):
    d = pred - true
    e3 = np.sqrt((d ** 2).sum(1))
    s = stats(e3)
    s["median_abs_axis_AP_DV_ML"] = np.median(np.abs(d), axis=0).tolist()
    s["median_signed_axis_AP_DV_ML"] = np.median(d, axis=0).tolist()
    return s, e3


def ceilings(ann, unit_true, unit_ann, d1):
    ok = unit_true >= 0
    out = {}
    for R in (500.0, 250.0, 1000.0):
        hit = (unit_ann == unit_true) & (d1 <= R / 1000.0) & ok
        per = [float(hit[unit_true == u].mean()) for u in range(23) if (unit_true == u).any()]
        out[f"R{int(R)}"] = dict(macro=float(np.mean(per)), micro=float(hit[ok].mean()), n_cells=int(ok.sum()), n_units=len(per))
    out["frac_unit_minus1"] = float((unit_ann[ok] < 0).mean())
    out["frac_d1_gt_500um"] = float((d1[ok] > 0.5).mean())
    return out


def validate():
    import ccf_annotation as VA
    t0 = time.time()
    pos = np.load(f"{C.SUPPORT}/support_pos_cal_mm.npy")
    ccf = np.load(f"{C.CCF_DIR}/support_ccf_xyz_mm.npy")
    brain = np.load(f"{C.SUPPORT}/support_brain.npy")
    unit = np.load(f"{C.SUPPORT}/support_unit.npy")
    donors = np.flatnonzero(np.isfinite(ccf).all(1))
    n_d = len(donors)
    C.log(f"donors {n_d:,} (per brain {np.bincount(brain[donors], minlength=5).tolist()}); MemAvailable {C.memavail_gb():.1f} GB")
    ann = VA.Annotation()
    rep = dict(created=C.t(), k=C.K_NN, n_donors=int(n_d), donors_per_brain=dict(zip(C.BRAINS, np.bincount(brain[donors], minlength=5).tolist())), seed=C.SEED_BASE)
    C.need_mem(6.0)
    H = int(round(0.01 * n_d))
    held = np.sort(np.random.default_rng(C.SEED_BASE).choice(n_d, H, replace=False))
    keep = np.ones(n_d, bool); keep[held] = False
    T = Translator(pos, ccf, donors[keep])
    q = donors[held]
    out = T.translate(pos[q])
    s_all, e3 = err_block(out["ccf_xyz_mm"], ccf[q].astype(np.float64))
    per = {}
    for b in range(5):
        m = brain[q] == b
        per[C.BRAINS[b]] = err_block(out["ccf_xyz_mm"][m], ccf[q][m].astype(np.float64))[0]
    _, ua, _ = ann.lookup(out["ccf_xyz_mm"])
    rep["V1"] = dict(n_held_out=int(H), tree_build_s=T.build_seconds, pooled=s_all, per_brain=per, d1_median_mm=float(np.median(out["d1_mm"])), d1_p99_mm=float(np.percentile(out["d1_mm"], 99)),
                     spread_median_mm=float(np.median(out["spread_mm"])), ceiling_membership=ceilings(ann, unit[q].astype(np.int16), ua, out["d1_mm"]),
                     ceiling_membership_note="annotation unit at the translated position vs the cell's own E unit, R = 500 um, cells with a unit label")
    rep["G_T1_median_le_0p5mm"] = bool(s_all["median"] <= 0.5)
    C.log(f"V1 hold-out {H:,}: median {s_all['median'] * 1000:.0f} um  p90 {s_all['p90'] * 1000:.0f} um  per-brain medians {[round(per[b]['median'] * 1000) for b in per]} um; "
          f"ceiling R500 macro {rep['V1']['ceiling_membership']['R500']['macro']:.4f} micro {rep['V1']['ceiling_membership']['R500']['micro']:.4f}; gate G-T1 {rep['G_T1_median_le_0p5mm']}")
    del T
    rep["V2"] = {}
    for b in range(5):
        C.need_mem(6.0)
        mem = brain[donors] == b
        T = Translator(pos, ccf, donors[~mem])
        qb = donors[mem]
        nq = int(round(0.01 * len(qb)))
        pick = np.sort(np.random.default_rng([C.SEED_BASE, b]).choice(len(qb), nq, replace=False))
        q = qb[pick]
        out = T.translate(pos[q])
        s_all, e3 = err_block(out["ccf_xyz_mm"], ccf[q].astype(np.float64))
        _, ua, _ = ann.lookup(out["ccf_xyz_mm"])
        rep["V2"][C.BRAINS[b]] = dict(n_query=nq, pool_donors=int(len(T.rows)), extra=bool(b == 4), pooled=s_all, d1_median_mm=float(np.median(out["d1_mm"])), d1_p99_mm=float(np.percentile(out["d1_mm"], 99)),
                                      frac_d1_gt_500um=float((out["d1_mm"] > 0.5).mean()), spread_median_mm=float(np.median(out["spread_mm"])),
                                      ceiling_membership=ceilings(ann, unit[q].astype(np.int16), ua, out["d1_mm"]))
        C.log(f"V2 leave out {C.BRAINS[b]}: n {nq:,}  median {s_all['median'] * 1000:.0f} um  p90 {s_all['p90'] * 1000:.0f} um  d1 median {np.median(out['d1_mm']) * 1000:.0f} um  "
              f"ceiling R500 macro {rep['V2'][C.BRAINS[b]]['ceiling_membership']['R500']['macro']:.4f} micro {rep['V2'][C.BRAINS[b]]['ceiling_membership']['R500']['micro']:.4f}")
        del T
    z = rep["V2"]
    rep["V2_zhuang_mean"] = {k: float(np.mean([z[C.BRAINS[b]]["ceiling_membership"]["R500"][k] for b in range(4)])) for k in ("macro", "micro")}
    rep["V2_zhuang_median_error_mm"] = float(np.mean([z[C.BRAINS[b]]["pooled"]["median"] for b in range(4)]))
    rep["wall_s"] = round(time.time() - t0, 1)
    rep["peak_rss_gb"] = round(C.peak_rss_gb(), 2)
    C.jdump(rep, f"{C.RES}/ccf_translation_validation.json")
    C.log(f"validation done in {rep['wall_s']} s, peak RSS {rep['peak_rss_gb']} GB")
    return rep


def layouts(kinds=C.KINDS):
    os.makedirs(C.LAYOUT_CCF, exist_ok=True)
    T = Translator.load()
    C.log(f"translator: donors {len(T.rows):,}, tree build {T.build_seconds} s")
    man = {}
    for tag in C.layout_names(kinds):
        src = f"{C.APP}/layouts/{tag}.npz"
        z = np.load(src)
        out = T.translate_training_ruler(z["Y_mm"])
        dst = f"{C.LAYOUT_CCF}/{tag}.npz"
        np.savez_compressed(dst, ccf_xyz_mm=out["ccf_xyz_mm"].astype(np.float32), d1_mm=out["d1_mm"].astype(np.float32), d15_mm=out["d15_mm"].astype(np.float32),
                            spread_mm=out["spread_mm"].astype(np.float32), rows=z["rows"])
        x = out["ccf_xyz_mm"]
        man[tag] = dict(layout_md5=C.md5f(src), ccf_md5=C.md5f(dst), n=int(len(x)), d1_median_mm=float(np.median(out["d1_mm"])), frac_d1_gt_500um=float((out["d1_mm"] > 0.5).mean()),
                        spread_median_mm=float(np.median(out["spread_mm"])), frac_ML_ge_5p7=float((x[:, 2] >= 5.7).mean()), median_xyz_mm=np.median(x, 0).tolist())
        C.log(f"{tag}: d1 median {man[tag]['d1_median_mm'] * 1000:.0f} um  >500um {man[tag]['frac_d1_gt_500um']:.4f}  spread {man[tag]['spread_median_mm'] * 1000:.0f} um  ML>=5.7 {man[tag]['frac_ML_ge_5p7']:.4f}")
    with open(f"{C.LAYOUT_CCF}/layouts_ccf_MD5SUMS.txt", "w") as f:
        for tag in man:
            f.write(f"{man[tag]['ccf_md5']}  {tag}.npz\n")
    C.jdump(dict(k=C.K_NN, frame="physical mm = Y_mm (training-ruler) x S_CAL", S_CAL=C.S_CAL.tolist(), axis_order="ccf_xyz_mm columns = official csv [x, y, z] = [AP, DV, ML] mm (CCFv3)",
                 donors=int(len(T.rows)), support_ccf_md5=C.md5f(f"{C.CCF_DIR}/support_ccf_xyz_mm.npy"), created=C.t(), layouts=man), f"{C.LAYOUT_CCF}/layouts_ccf_manifest.json")
    C.log(f"{len(man)} layouts translated; peak RSS {C.peak_rss_gb():.2f} GB")


def write_manifest():
    val = json.load(open(f"{C.RES}/ccf_translation_validation.json"))
    rep = json.load(open(f"{C.CCF_DIR}/ccf_support_report.json"))
    man = dict(rule="translated(q) = mean of the official CCFv3 [x, y, z] = [AP, DV, ML] mm of the k nearest donors; Euclidean in physical mm (training-ruler mm x S_CAL)", k=C.K_NN,
               S_CAL=C.S_CAL.tolist(), donors="support cells (support row order) with a finite official ABC CCF row", n_donors=rep["donors_total"], donors_per_brain={b: v["donors"] for b, v in rep["per_brain"].items()},
               hemisphere="all five stores lie on the LEFT side of the CCF midline (donor ML < 5.7 mm: " + ", ".join(f"{b} {v['frac_z_lt_5p7']:.3f}" for b, v in rep["per_brain"].items()) + ")",
               files={f: C.md5f(f"{C.CCF_DIR}/{f}") for f in ("support_ccf_xyz_mm.npy", "ccf_support_report.json")}, support_pos_cal_mm_md5=C.md5f(f"{C.SUPPORT}/support_pos_cal_mm.npy"),
               validation=dict(V1_within_support_holdout_1pct=val["V1"]["pooled"], V1_per_brain={b: v["median"] for b, v in val["V1"]["per_brain"].items()},
                               V2_leave_one_brain_out={b: v["pooled"] for b, v in val["V2"].items()}, gate_G_T1_median_le_0p5mm=val["G_T1_median_le_0p5mm"]),
               code="ccf_translate.py (Translator.load(); translate_training_ruler(Y_mm))", created=C.t())
    C.jdump(man, f"{C.CCF_DIR}/translation_manifest.json")
    C.log("wrote translation_manifest.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["validate", "layouts", "manifest"])
    a = ap.parse_args()
    {"validate": validate, "layouts": layouts, "manifest": write_manifest}[a.cmd]()
