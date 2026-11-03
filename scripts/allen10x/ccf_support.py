#!/usr/bin/env python
"""Official ABC CCF coordinates of every training-support cell, in the support row order -> application/ccf_translation/support_ccf_xyz_mm.npy
([AP, DV, ML] CCFv3 mm, NaN = no CCF row) and a report.  No 10x file is read."""
import json
import os

import numpy as np
import pandas as pd

import ccf_common as C
import support as ES


def main():
    os.makedirs(C.CCF_DIR, exist_ok=True)
    sup_brain = np.load(f"{C.SUPPORT}/support_brain.npy")
    sup_row = np.load(f"{C.SUPPORT}/support_row.npy")
    n = len(sup_brain)
    assert n == 7689601, n
    out = np.full((n, 3), np.nan, np.float32)
    rep = dict(created=C.t(), n_support=int(n), axis_order="official csv columns [x, y, z] = [AP, DV, ML] (CCFv3 PIR, mm)", per_brain={})
    off = 0
    for b_i, b in enumerate(ES.BRAINS):
        labels = np.array(ES.read_lines(f"{ES.STORES}/{b}/cell_labels.txt"), dtype=object)
        nb = len(labels)
        assert nb == ES.N_EXPECT[b], (b, nb)
        sel = np.flatnonzero(sup_brain == b_i)
        assert len(sel) == nb and (sup_row[sel] == np.arange(nb)).all() and sel[0] == off and sel[-1] == off + nb - 1, f"{b}: support rows are not the store rows in order"
        ccf = pd.read_csv(ES.CCF[b], dtype={"cell_label": str})
        assert ccf.cell_label.is_unique
        ix = pd.Index(ccf.cell_label).get_indexer(pd.Index(labels.astype(str)))
        hit = ix >= 0
        xyz = np.full((nb, 3), np.nan, np.float64)
        xyz[hit] = ccf[["x", "y", "z"]].to_numpy(dtype=np.float64)[ix[hit]]
        fin = np.isfinite(xyz).all(1)
        xyz[~fin] = np.nan
        out[sel] = xyz.astype(np.float32)
        med = np.nanmedian(xyz, axis=0)
        rep["per_brain"][b] = dict(n_cells=int(nb), csv=ES.CCF[b], csv_md5=C.md5f(ES.CCF[b]), csv_rows=int(len(ccf)), cell_label_found=int(hit.sum()), finite_xyz=int(fin.sum()),
                                   donors=int(fin.sum()), median_xyz_mm=med.tolist(), p01_xyz_mm=np.nanpercentile(xyz, 1, axis=0).tolist(), p99_xyz_mm=np.nanpercentile(xyz, 99, axis=0).tolist(),
                                   frac_z_lt_5p7=float((xyz[fin, 2] < 5.7).mean()))
        C.log(f"{b}: n {nb:,}  csv rows {len(ccf):,}  found {int(hit.sum()):,}  finite {int(fin.sum()):,}  median [AP, DV, ML] {np.round(med, 3).tolist()}  frac ML<5.7 {rep['per_brain'][b]['frac_z_lt_5p7']:.4f}")
        off += nb
    fin_all = np.isfinite(out).all(1)
    rep["donors_total"] = int(fin_all.sum())
    np.save(f"{C.CCF_DIR}/support_ccf_xyz_mm.npy", out)
    rep["support_ccf_xyz_mm_md5"] = C.md5f(f"{C.CCF_DIR}/support_ccf_xyz_mm.npy")
    C.jdump(rep, f"{C.CCF_DIR}/ccf_support_report.json")
    C.log(f"donors {int(fin_all.sum()):,} of {n:,}; md5 {rep['support_ccf_xyz_mm_md5']}; peak RSS {C.peak_rss_gb():.2f} GB")


if __name__ == "__main__":
    main()
