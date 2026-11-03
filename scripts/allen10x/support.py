#!/usr/bin/env python
"""Training-support map: for every cell of the five training stores (abca1, abca2_ref1, abca3, abca4, allen638850_hemi0, in this order)
its calibrated position, brain, row, ABC parcellation, unit and subclass, plus a coverage report.  No 10x file is read.
usage: support.py [OUT_DIR_OF_THE_SUPPORT]   (default OUT_DIR/application/support)"""
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
import numpy as np
import pandas as pd

import paths as P
import units as EU

R3 = P.MERFISH_RAW
STORES = P.STORES
OUT = sys.argv[1] if len(sys.argv) > 1 else P.SUPPORT
TAX = P.TAXONOMY
S_CAL = np.array([1.0185775453536927, 1.2931099417291327, 0.9995762396395721])
BRAINS = ["abca1", "abca2_ref1", "abca3", "abca4", "allen638850_hemi0"]
N_EXPECT = {"abca1": 2846908, "abca2_ref1": 1227408, "abca3": 1585782, "abca4": 162578, "allen638850_hemi0": 1866925}
CCF = {"abca1": f"{R3}/Zhuang-ABCA-1-CCF_ccf_coordinates.csv", "abca2_ref1": f"{R3}/Zhuang-ABCA-2-CCF_ccf_coordinates.csv",
       "abca3": f"{R3}/Zhuang-ABCA-3-CCF_ccf_coordinates.csv", "abca4": f"{R3}/Zhuang-ABCA-4-CCF_ccf_coordinates.csv",
       "allen638850_hemi0": f"{P.ALLEN_MERFISH_RAW}/ccf_coordinates.csv"}
META = {"abca1": f"{R3}/Zhuang-ABCA-1_cell_metadata_with_cluster_annotation.csv", "abca2_ref1": f"{R3}/Zhuang-ABCA-2_cell_metadata_with_cluster_annotation.csv",
        "abca3": f"{R3}/Zhuang-ABCA-3_cell_metadata.csv", "abca4": f"{R3}/Zhuang-ABCA-4_cell_metadata.csv",
        "allen638850_hemi0": f"{P.ALLEN_MERFISH_RAW}/cell_metadata_with_cluster_annotation.csv"}


def t():
    return time.strftime("%H:%M:%S")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def read_lines(p):
    with open(p) as f:
        return [x.rstrip("\n") for x in f]


def main():
    os.makedirs(OUT, exist_ok=True)
    B = EU.build(verbose=False)
    pm = B["parc"]
    parc2unit = np.full(int(pm.parcellation_index.max()) + 1, -1, np.int16)
    parc2unit[pm.parcellation_index.to_numpy()] = pm.unit.to_numpy()
    tax = pd.read_csv(TAX)
    assert tax.cluster_alias.is_unique
    sub_names = sorted(tax["subclass"].unique())
    sub_code = {s: i for i, s in enumerate(sub_names)}
    A = int(tax.cluster_alias.max()) + 1
    alias2sub = np.full(A, -1, np.int16)
    alias2sub[tax.cluster_alias.to_numpy()] = tax["subclass"].map(sub_code).to_numpy().astype(np.int16)
    rep = dict(started=t(), brains=BRAINS, n_units=23, unit_names=EU.UNIT_NAMES, subclass_names=sub_names, S_CAL=S_CAL.tolist(),
               taxonomy_file=TAX, taxonomy_md5=md5(TAX), units_build=dict(crosswalk_md5=EU.CROSSWALK_MD5, graph_md5=md5(EU.GRAPH),
                                                                          membership_md5=md5(EU.MEMBERSHIP), e_units_md5=md5(EU.__file__)), per_brain={})
    P, Pc, BR, RW, PA, UN, SU = [], [], [], [], [], [], []
    for b_i, b in enumerate(BRAINS):
        t0 = time.time()
        meta = json.load(open(f"{STORES}/{b}/meta.json"))
        iso = float(meta["iso_mm"])
        pos = np.load(f"{STORES}/{b}/pos.npy").astype(np.float64)
        labels = np.array(read_lines(f"{STORES}/{b}/cell_labels.txt"), dtype=object)
        n = len(labels)
        assert n == N_EXPECT[b] == len(pos) == meta["n_cells"], (b, n, len(pos))
        assert len(set(labels.tolist())) == n, f"{b}: duplicated cell labels"
        pos_mm = pos * iso
        ccf = pd.read_csv(CCF[b], dtype={"cell_label": str})
        assert ccf.cell_label.is_unique
        ix = pd.Index(ccf.cell_label).get_indexer(pd.Index(labels.astype(str)))
        hit = ix >= 0
        parc = np.full(n, -1, np.int16)
        pv = ccf["parcellation_index"].to_numpy()[ix[hit]]
        ok_p = np.isfinite(pv.astype(np.float64))
        tmp = np.full(hit.sum(), -1, np.int16)
        tmp[ok_p] = pv[ok_p].astype(np.int16)
        parc[hit] = tmp
        cc = ccf[["z", "y", "x"]].to_numpy(dtype=np.float64)[ix[hit]]
        fin = np.isfinite(cc).all(1)
        aff = []
        for a in range(3):
            xs, ys = pos_mm[hit][fin, a], cc[fin, a]
            sl, ic = np.polyfit(xs, ys, 1)
            r = np.corrcoef(xs, ys)[0, 1]
            aff.append(dict(axis="ML DV AP".split()[a], slope=float(sl), intercept=float(ic), pearson=float(r), max_abs_resid_mm=float(np.abs(ys - (sl * xs + ic)).max()),
                            median_abs_resid_mm=float(np.median(np.abs(ys - (sl * xs + ic))))))
        unit = np.full(n, -1, np.int8)
        okp = parc >= 0
        unit[okp] = parc2unit[parc[okp]].astype(np.int8)
        meta_csv = pd.read_csv(META[b], usecols=["cell_label", "cluster_alias"], dtype={"cell_label": str})
        assert meta_csv.cell_label.is_unique
        jx = pd.Index(meta_csv.cell_label).get_indexer(pd.Index(labels.astype(str)))
        jh = jx >= 0
        alias = np.full(n, -1, np.int64)
        alias[jh] = meta_csv["cluster_alias"].to_numpy()[jx[jh]]
        sub = np.full(n, -1, np.int16)
        oka = (alias >= 0) & (alias < A)
        sub[oka] = alias2sub[alias[oka]]
        rep["per_brain"][b] = dict(
            n_cells=n, iso_mm=iso, ccf_csv=CCF[b], ccf_csv_md5=md5(CCF[b]), cell_label_found_in_ccf_csv=int(hit.sum()), parcellation_finite=int((parc >= 0).sum()),
            parcellation_is_unassigned_0=int((parc == 0).sum()), cells_with_unit=int((unit >= 0).sum()), frac_with_unit=float((unit >= 0).mean()),
            cells_with_label_but_no_unit=int(((parc >= 0) & (unit < 0)).sum()), meta_csv=META[b], meta_csv_md5=md5(META[b]),
            cell_label_found_in_meta=int(jh.sum()), with_cluster_alias_in_taxonomy=int((sub >= 0).sum()), frac_with_subclass=float((sub >= 0).mean()),
            affine_check_csv_ccf_vs_store_pos=aff, unit_counts={EU.UNIT_NAMES[k]: int((unit == k).sum()) for k in range(23)})
        print(f"[{t()}] {b}: n {n:,}  CCF row {int(hit.sum()):,}  parcellation {int((parc >= 0).sum()):,}  unit {int((unit >= 0).sum()):,} ({(unit >= 0).mean():.4f})  "
              f"subclass {int((sub >= 0).sum()):,}  affine slopes {[round(a['slope'], 4) for a in aff]} r {[round(a['pearson'], 6) for a in aff]}  max resid mm {[round(a['max_abs_resid_mm'], 4) for a in aff]}  "
              f"({time.time() - t0:.0f} s)", flush=True)
        P.append(pos_mm.astype(np.float32)); Pc.append((pos_mm * S_CAL).astype(np.float32)); BR.append(np.full(n, b_i, np.int8)); RW.append(np.arange(n, dtype=np.int32))
        PA.append(parc); UN.append(unit); SU.append(sub)
    arrs = dict(pos_mm=np.concatenate(P), pos_cal_mm=np.concatenate(Pc), brain=np.concatenate(BR), row=np.concatenate(RW), parc=np.concatenate(PA),
                unit=np.concatenate(UN), subclass=np.concatenate(SU))
    for k, v in arrs.items():
        np.save(f"{OUT}/support_{k}.npy", v)
    n = len(arrs["unit"])
    cnt = np.array([(arrs["unit"] == k).sum() for k in range(23)], dtype=np.int64)
    rep["n_total"] = int(n)
    rep["cells_with_unit_total"] = int(cnt.sum())
    rep["frac_with_unit_total"] = float(cnt.sum() / n)
    rep["unit_counts_total"] = {EU.UNIT_NAMES[k]: int(cnt[k]) for k in range(23)}
    rep["unit_fraction_among_cells_with_unit"] = {EU.UNIT_NAMES[k]: float(cnt[k] / cnt.sum()) for k in range(23)}
    rep["finished"] = t()
    rep["file_md5"] = {f"support_{k}.npy": md5(f"{OUT}/support_{k}.npy") for k in arrs}
    json.dump(rep, open(f"{OUT}/support_report.json", "w"), indent=1)
    print(f"[{t()}] done: {n:,} support cells; with unit {cnt.sum():,} ({cnt.sum() / n:.4f})", flush=True)
    print("unit fractions among cells with a unit:", {EU.UNIT_NAMES[k]: round(float(cnt[k] / cnt.sum()), 4) for k in range(23)})


if __name__ == "__main__":
    main()
