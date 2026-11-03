#!/usr/bin/env python
"""Skeletons of the five training stores of the 10x model (abca1, abca2_ref1, abca3, abca4 and the Zhuang-side hemisphere of the Allen
MERFISH brain): positions, sections, classes and cell labels from the cell stores, metadata from the template store tree; the feature
slot is filled afterwards by fill_features.py.  usage: build_stores.py --cellstore_root DIR --template_root DIR --out_root DIR [--dry]"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HOST = os.uname().nodename
ISO = 13.45773983001709
ZH = ["abca1", "abca2_ref1", "abca3", "abca4"]
ALLEN = "allen638850_hemi0"
NPC = 200
PCNAMES = [f"sPC{i}" for i in range(1, NPC + 1)]
FEATURE_TEXT = ("Harmony-corrected PCs of the Symphony reference (symphonypy @ cbda5adc531b49f5cfcfa5c0ef97dd4668209575 on harmonypy): "
                "477 shared genes, counts normalised over the 477 panel genes (lib 1e4), log1p, z with the reference mean / SD, clip 10, PCA 200, "
                "Harmony batch = platform; rows in the row order of the template stores. FILL STATUS in a10x.fill_status")


def md5(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def link(src, dst):
    assert os.path.exists(src), src
    if os.path.lexists(dst):
        assert os.path.islink(dst), f"{dst} exists and is not a link"
        os.remove(dst)
    os.symlink(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cellstore_root", required=True, help="cell stores of the five MERFISH brains (abca1..abca4, allen638850)")
    ap.add_argument("--template_root", required=True, help="previous training-store tree (stores/<brain>/meta.json, allen/cell_ID.npy)")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    CS = a.cellstore_root; HK = a.template_root
    OUT = a.out_root
    MAPS = os.path.join(os.path.dirname(OUT.rstrip("/")), "maps")
    print(f"[build-stores] host {HOST}; cell stores {CS}; template tree {HK}; OUT {OUT}; maps {MAPS}", flush=True)
    if not a.dry:
        os.makedirs(OUT, exist_ok=True); os.makedirs(MAPS, exist_ok=True)
    report = {"host": HOST, "out": OUT, "stores": {}}

    src_dirs = {"abca1": ("abca1", "abca1"), "abca2_ref1": ("abca2", "abca2_ref1"), "abca3": ("abca3", "abca3"), "abca4": ("abca4", "abca4")}
    for s in ZH:
        base, posdir = src_dirs[s]
        d = f"{OUT}/{s}"
        old_meta_path = f"{HK}/stores/{s}/meta.json"
        assert os.path.exists(old_meta_path), old_meta_path
        m = json.load(open(old_meta_path))
        assert m["n_genes"] == NPC and "count_keep" not in m and "hemi" not in json.dumps(list(m.keys())), (s, "unexpected key in old meta")
        n = int(m["n_cells"])
        files = {"cell_labels.txt": f"{CS}/{base}/cell_labels.txt", "cls.npy": f"{CS}/{base}/cls.npy", "pos.npy": f"{CS}/{posdir}/pos.npy", "sec.npy": f"{CS}/{base}/sec.npy"}
        pos = np.load(files["pos.npy"], mmap_mode="r"); sec = np.load(files["sec.npy"], mmap_mode="r"); cls = np.load(files["cls.npy"], mmap_mode="r")
        assert len(pos) == len(sec) == len(cls) == n and pos.shape[1] == 3, (s, len(pos), len(sec), len(cls), n)
        assert int(sec.max()) + 1 == len(m["sections"]) == len(m["section_cells"]), (s, "sections")
        m2 = dict(m)
        m2["gene_names"] = list(PCNAMES); m2["feature"] = FEATURE_TEXT
        m2["a10x"] = dict(built_by="build_stores.py", built_on=HOST, built_at=time.strftime("%F %T"), fill_status="PENDING",
                          meta_from=old_meta_path, meta_from_md5=md5(old_meta_path),
                          non_feature_files="symlinks to " + ", ".join(sorted(set(os.path.dirname(v) for v in files.values()))))
        report["stores"][s] = dict(n_cells=n, n_sections=int(sec.max()) + 1, meta_from=old_meta_path, meta_from_md5=m2["a10x"]["meta_from_md5"])
        if not a.dry:
            os.makedirs(d, exist_ok=True)
            for k, v in files.items():
                link(v, f"{d}/{k}")
            json.dump(m2, open(f"{d}/meta.json", "w"), indent=1)
        print(f"[build-stores] {s}: {n:,} cells, {int(sec.max()) + 1} sections; files -> {sorted(set(os.path.dirname(v) for v in files.values()))}", flush=True)

    A = f"{CS}/allen638850"
    cm = json.load(open(f"{A}/meta.json"))
    pos = np.load(f"{A}/pos.npy"); sec = np.load(f"{A}/sec.npy"); cls = np.load(f"{A}/cls.npy"); hemi = np.load(f"{A}/hemi.npy")
    N = len(pos)
    assert N == len(sec) == len(cls) == len(hemi) == cm["n_cells"] == 3739961 and pos.dtype == np.float32
    mid = float(cm["midline_ml_iso"])
    assert np.array_equal(hemi, (pos[:, 0] > mid).astype(np.int8)), "hemi.npy is not the midline plane of pos.npy"
    keep = np.flatnonzero(hemi == 0).astype(np.int64)
    drop = np.flatnonzero(hemi == 1)
    assert pos[keep, 0].max() <= mid < pos[drop, 0].min()
    cid = np.load(f"{HK}/allen/cell_ID.npy")
    assert cid.shape == (N,) and np.array_equal(np.sort(cid), np.arange(N)), "cell_ID is not a permutation of the cell-store rows"
    inv = np.argsort(cid)
    s2_row = inv[keep].astype(np.int64)
    assert np.array_equal(cid[s2_row], keep)
    sec_h, cls_h, pos_h = sec[keep], cls[keep], pos[keep]
    nsec = int(sec.max()) + 1
    cnt = np.bincount(sec_h, minlength=nsec)
    assert (cnt > 0).all(), "a section lost all its cells"
    labels = open(f"{A}/cell_labels.txt").read().split("\n")
    if labels and labels[-1] == "":
        labels.pop()
    assert len(labels) == N
    lab_h = [labels[i] for i in keep]
    cen = np.array([pos_h[sec_h == s].mean(0) for s in range(nsec)]); ax = int(np.argmax(cen.std(0)))
    gap_mm = float(np.median(np.diff(np.sort(cen[:, ax]))) * ISO)
    cm2 = {k: v for k, v in cm.items() if k not in ("measured", "n_measured", "count_keep", "count_keep_note", "hemispheres", "hemi_share_other")}
    cm2.update(animal="Allen-MERFISH-C57BL6J-638850 (Zhuang-side hemisphere only)", n_cells=int(len(keep)), n_genes=NPC, gene_names=list(PCNAMES),
               section_cells=[int(c) for c in cnt], feature=FEATURE_TEXT)
    cm2["hemisphere"] = dict(kept="hemi.npy == 0 of cellstore_ccf/allen638850: ML <= midline_ml_iso in store coordinates (the side the Zhuang hemibrains cover)",
                             midline_ml_iso=mid, midline_mm=mid * ISO, n_kept=int(len(keep)), n_dropped=int(len(drop)), folded=False,
                             hemi_npy_in_this_store=False, kept_ML_max_mm=float(pos_h[:, 0].max() * ISO), dropped_ML_min_mm=float(pos[drop, 0].min() * ISO))
    cm2["a10x"] = dict(built_by="build_stores.py", built_on=HOST, built_at=time.strftime("%F %T"), fill_status="PENDING",
                       source_store=A, source_meta_md5=md5(f"{A}/meta.json"), cut_axis=ax, section_gap_mm_median=gap_mm,
                       count_keep_removed="the cell store's count_keep (0.2077) belongs to log2-count features; PCs must never be thinned",
                       maps=dict(cellstore_row="cellstore_row.npy: new row -> row of cellstore_ccf/allen638850",
                                 s2_row="s2_row.npy: new row -> row of template allen/feat.npy order (= the reference Allen embedding row order)"))
    report["stores"][ALLEN] = dict(n_cells=int(len(keep)), n_dropped=int(len(drop)), n_sections=nsec, min_cells_per_section=int(cnt.min()), cut_axis=ax,
                                   median_section_gap_mm=gap_mm, midline_mm=mid * ISO)
    print(f"[build-stores] {ALLEN}: kept {len(keep):,} of {N:,} (dropped {len(drop):,}); midline {mid:.7f} iso = {mid * ISO:.4f} mm; kept ML max {pos_h[:, 0].max() * ISO:.4f} mm, "
          f"dropped ML min {pos[drop, 0].min() * ISO:.4f} mm; {nsec} sections all non-empty (min {cnt.min():,} cells); cut axis {ax}; median section gap {gap_mm:.3f} mm", flush=True)
    if not a.dry:
        d = f"{OUT}/{ALLEN}"; os.makedirs(d, exist_ok=True)
        assert not os.path.exists(f"{d}/hemi.npy")
        np.save(f"{d}/pos.npy", pos_h); np.save(f"{d}/sec.npy", sec_h.astype(np.int32)); np.save(f"{d}/cls.npy", cls_h.astype(np.int16))
        open(f"{d}/cell_labels.txt", "w").write("\n".join(lab_h) + "\n")
        np.save(f"{d}/cellstore_row.npy", keep); np.save(f"{d}/s2_row.npy", s2_row)
        json.dump(cm2, open(f"{d}/meta.json", "w"), indent=1)
        open(f"{MAPS}/allen_s2_row_cell_labels.txt", "w").write("\n".join(labels[i] for i in cid) + "\n")
        for k, ref in (("pos.npy", pos_h), ("sec.npy", sec_h), ("cls.npy", cls_h), ("cellstore_row.npy", keep), ("s2_row.npy", s2_row)):
            assert np.array_equal(np.load(f"{d}/{k}"), ref), k
        assert open(f"{d}/cell_labels.txt").read().split("\n")[:-1] == lab_h
        assert len(open(f"{MAPS}/allen_s2_row_cell_labels.txt").read().split("\n")) - 1 == N
    sums = {}
    if not a.dry:
        for s in ZH + [ALLEN]:
            for f in sorted(os.listdir(f"{OUT}/{s}")):
                p = f"{OUT}/{s}/{f}"
                if os.path.exists(p):
                    sums[f"{s}/{f}"] = md5(p)
        sums["maps/allen_s2_row_cell_labels.txt"] = md5(f"{MAPS}/allen_s2_row_cell_labels.txt")
    report["md5"] = {k: v for k, v in sums.items() if not k.endswith("meta.json") or k.startswith(ALLEN) is False}
    report["md5_all"] = sums
    report["seconds"] = round(time.time() - t0)
    print(json.dumps(report["stores"], indent=1))
    for k, v in sums.items():
        print(f"{v}  {k}")
    if not a.dry:
        json.dump(report, open(f"{os.path.dirname(OUT.rstrip('/'))}/build_report.json", "w"), indent=1)
    print(f"[build-stores] done in {report['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
