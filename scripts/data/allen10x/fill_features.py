#!/usr/bin/env python
"""Fill the feature slot of the five training stores with the Symphony reference embedding: Zhuang stores = byte copies of the reference
brains' corrected PCs, Allen hemisphere = rows of the reference Allen embedding through s2_row; then checks of shape, finiteness,
row order (by label when given and by data against the template tree) and the md5 of every other store file.
usage: fill_features.py --stores DIR --template_root DIR --ref_dir REFERENCE_DIR"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np

HOST = os.uname().nodename
ZH = {"abca1": "abca1", "abca2_ref1": "abca2", "abca3": "abca3", "abca4": "abca4"}
ALLEN = "allen638850_hemi0"
NPC = 200
FAILS = []


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ok(c, msg):
    print(("PASS " if c else "FAIL ") + msg, flush=True)
    if not c:
        FAILS.append(msg)


def md5(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def finite_stats(arr, chunk=250000):
    n = 0; bad = 0; s1 = np.zeros(arr.shape[1]); s2 = np.zeros(arr.shape[1])
    for i in range(0, arr.shape[0], chunk):
        x = np.asarray(arr[i:i + chunk], dtype=np.float64)
        bad += int((~np.isfinite(x)).sum()); n += len(x); s1 += x.sum(0); s2 += (x * x).sum(0)
    m = s1 / n
    return bad, m, np.sqrt(np.maximum(s2 / n - m * m, 0))


def order_check(new_rows, old_rows, k_old=10, n=50000, seed=25001):
    a = np.asarray(old_rows[:, :k_old], dtype=np.float64); b = np.asarray(new_rows, dtype=np.float64)

    def cc(a_, b_):
        a_ = (a_ - a_.mean(0)) / (a_.std(0) + 1e-12); b_ = (b_ - b_.mean(0)) / (b_.std(0) + 1e-12)
        return np.abs(a_.T @ b_ / len(a_))
    r = cc(a, b)
    rng = np.random.default_rng(seed)
    r0 = cc(a, b[rng.permutation(len(b))])
    return r.max(1), r.argmax(1), r0.max(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template_root", required=True, help="previous training-store tree (row-order and file checks)")
    ap.add_argument("--stores", required=True, help="the stores root made by build_stores.py")
    ap.add_argument("--src", action="append", default=[], help="NAME=PATH, NAME in abca1 abca2 abca3 abca4 allen")
    ap.add_argument("--ref_dir", default="", help="the reference directory: <ref_dir>/stores/{abca1,abca2,abca3,abca4}/feat.npy and <ref_dir>/allen/feat.npy (expands to the five --src)")
    ap.add_argument("--src_labels", action="append", default=[], help="NAME=PATH: the cell labels of that source in the reference row order (one per line)")
    ap.add_argument("--n_check", type=int, default=50000)
    a = ap.parse_args()
    t0 = time.time()
    HK = a.template_root
    STORES = a.stores.rstrip("/"); MAPS = os.path.join(os.path.dirname(STORES), "maps")
    old = {"abca1": f"{HK}/stores/abca1/feat.npy", "abca2": f"{HK}/stores/abca2/feat.npy", "abca3": f"{HK}/stores/abca3/feat.npy", "abca4": f"{HK}/stores/abca4/feat.npy",
           "allen": f"{HK}/allen/feat.npy"}
    src = dict(s.split("=", 1) for s in a.src)
    if a.ref_dir and not src:
        src = {k: f"{a.ref_dir.rstrip('/')}/stores/{k}/feat.npy" for k in ("abca1", "abca2", "abca3", "abca4")}
        src["allen"] = f"{a.ref_dir.rstrip('/')}/allen/feat.npy"
    assert set(src) == {"abca1", "abca2", "abca3", "abca4", "allen"}, f"need --src for abca1 abca2 abca3 abca4 allen, got {sorted(src)}"
    status = "FILLED"
    labs = dict(s.split("=", 1) for s in a.src_labels)
    log(f"stores {STORES}; sources " + json.dumps(src))
    report = {"host": HOST, "stores": {}, "sources": src, "src_md5": {}}
    n_old = {"abca1": 2846908, "abca2": 1227408, "abca3": 1585782, "abca4": 162578, "allen": 3739961}

    S = {}
    for k, p in src.items():
        ok(os.path.exists(p), f"source {k} exists: {p}")
        m = np.load(p, mmap_mode="r"); S[k] = m
        ok(m.shape == (n_old[k], NPC), f"source {k} shape {m.shape} == ({n_old[k]}, {NPC}) (the old tree's row count)")
        ok(m.dtype in (np.float32, np.float64), f"source {k} dtype {m.dtype}")
    if FAILS:
        sys.exit(1)

    for st, key in ZH.items():
        d = f"{STORES}/{st}"
        ok(os.path.isdir(d), f"store dir {d}")
        dst = f"{d}/feat.npy"
        assert not os.path.islink(dst), f"{dst} is a symlink: refusing to write through it"
        t1 = time.time()
        if S[key].dtype == np.float32:
            shutil.copyfile(src[key], dst)
            ms, md = md5(src[key]), md5(dst)
            ok(ms == md, f"{st}: byte copy of {key}, md5 source {ms[:12]} == destination {md[:12]}")
            report["src_md5"][key] = ms
        else:
            F = np.lib.format.open_memmap(dst, mode="w+", dtype=np.float32, shape=S[key].shape)
            for i in range(0, S[key].shape[0], 250000):
                F[i:i + 250000] = np.asarray(S[key][i:i + 250000]).astype(np.float32)
            F.flush(); del F
            bad = sum(int((np.asarray(np.load(dst, mmap_mode="r")[i:i + 250000]) != np.asarray(S[key][i:i + 250000]).astype(np.float32)).sum()) for i in range(0, S[key].shape[0], 250000))
            ok(bad == 0, f"{st}: float64 source cast to float32, {bad} entries differ from the exact cast")
            report["src_md5"][key] = "float64 source (not byte-copied)"
        report["stores"][st] = {"feat_md5": md5(dst), "seconds": round(time.time() - t1)}
        log(f"{st}: filled in {time.time() - t1:.0f} s")

    d = f"{STORES}/{ALLEN}"
    s2_row = np.load(f"{d}/s2_row.npy"); cs_row = np.load(f"{d}/cellstore_row.npy")
    n = len(s2_row)
    meta_a = json.load(open(f"{d}/meta.json"))
    ok(n == meta_a["n_cells"] == 1866925, f"Allen hemisphere rows {n:,} == meta n_cells {meta_a['n_cells']:,} == 1,866,925")
    ok(not os.path.exists(f"{d}/hemi.npy"), "Allen store has NO hemi.npy")
    t1 = time.time()
    assert not os.path.islink(f"{d}/feat.npy"), "Allen feat.npy is a symlink: refusing to write through it"
    F = np.lib.format.open_memmap(f"{d}/feat.npy", mode="w+", dtype=np.float32, shape=(n, NPC))
    for i in range(0, n, 250000):
        r = s2_row[i:i + 250000]; o = np.argsort(r)
        X = np.asarray(S["allen"][r[o]]).astype(np.float32)
        blk = np.empty_like(X); blk[o] = X
        F[i:i + len(r)] = blk
    F.flush(); del F
    G = np.load(f"{d}/feat.npy", mmap_mode="r")
    bad = 0
    for i in range(0, n, 250000):
        r = s2_row[i:i + 250000]
        bad += int((np.asarray(G[i:i + 250000]) != np.asarray(S["allen"][r]).astype(np.float32)).any(1).sum())
    ok(bad == 0, f"Allen: all {n:,} rows of feat.npy equal source[s2_row] ({bad} differ)")
    report["stores"][ALLEN] = {"feat_md5": md5(f"{d}/feat.npy"), "seconds": round(time.time() - t1)}
    log(f"{ALLEN}: filled in {time.time() - t1:.0f} s")

    for st in list(ZH) + [ALLEN]:
        m = json.load(open(f"{STORES}/{st}/meta.json")); f = np.load(f"{STORES}/{st}/feat.npy", mmap_mode="r")
        pos = np.load(f"{STORES}/{st}/pos.npy", mmap_mode="r")
        ok(f.shape == (m["n_cells"], NPC) and f.dtype == np.float32 and len(pos) == m["n_cells"], f"{st}: feat {f.shape} {f.dtype}, pos rows {len(pos):,}, meta n_cells {m['n_cells']:,}")
        bad, mu, sd = finite_stats(f)
        ok(bad == 0, f"{st}: {bad} non-finite entries")
        report["stores"][st].update(n_cells=int(m["n_cells"]), pc_mean_abs_max=float(np.abs(mu).max()), pc_sd_first5=[round(float(x), 3) for x in sd[:5]], pc_sd_last=round(float(sd[-1]), 3))
        log(f"{st}: PC SD first 5 {np.round(sd[:5], 3).tolist()}, last {sd[-1]:.3f}; max |PC mean| {np.abs(mu).max():.3f}")

    for k, lp in labs.items():
        L = open(lp).read().split("\n")
        if L and L[-1] == "":
            L.pop()
        if k == "allen":
            exp = open(f"{MAPS}/allen_s2_row_cell_labels.txt").read().split("\n")[:-1]
        else:
            st = [s for s, kk in ZH.items() if kk == k][0]
            exp = open(os.path.realpath(f"{STORES}/{st}/cell_labels.txt")).read().split("\n")
            exp = exp[:-1] if exp and exp[-1] == "" else exp
        ok(len(L) == len(exp) and L == exp, f"row order by LABEL: reference {k} labels ({len(L):,}) equal the old tree's order ({len(exp):,})")
        report.setdefault("label_check", {})[k] = bool(L == exp)
    if not labs:
        log("NOTE: no --src_labels given: row order is checked by DATA only (cross-correlation with the old Harmony PCs below)")

    rng = np.random.default_rng(25001)
    for st in list(ZH) + [ALLEN]:
        n_st = json.load(open(f"{STORES}/{st}/meta.json"))["n_cells"]
        idx = np.sort(rng.choice(n_st, min(a.n_check, n_st), replace=False))
        new = np.asarray(np.load(f"{STORES}/{st}/feat.npy", mmap_mode="r")[idx])
        if st == ALLEN:
            oldrows = np.asarray(np.load(old["allen"], mmap_mode="r")[np.sort(s2_row[idx])])
            oldrows = oldrows[np.argsort(np.argsort(s2_row[idx]))]
        else:
            oldrows = np.asarray(np.load(old[ZH[st]], mmap_mode="r")[idx])
        best, arg, ctl = order_check(new, oldrows)
        report["stores"][st].update(order_best_abs_r_first10_old_pcs=[round(float(x), 3) for x in best], order_best_new_pc=[int(x) for x in arg], order_shuffled_control_max=round(float(ctl.max()), 3))
        ok(bool((best[:5] >= 0.9).all() and ctl.max() < 0.1), f"{st}: row order by data: best |r| of old PC1-5 vs the new PCs {np.round(best[:5], 3).tolist()} (>= 0.9), PC6-10 {np.round(best[5:], 3).tolist()}; shuffled control max {ctl.max():.3f} (< 0.1)")

    for st, key in ZH.items():
        base = "abca2" if key == "abca2" else st
        for f_, srcp in (("pos.npy", f"{R3}/cellstore_ccf/{'abca2_ref1' if st == 'abca2_ref1' else st}/pos.npy"), ("sec.npy", f"{R3}/cellstore_ccf/{base}/sec.npy"),
                         ("cls.npy", f"{R3}/cellstore_ccf/{base}/cls.npy"), ("cell_labels.txt", f"{R3}/cellstore_ccf/{base}/cell_labels.txt")):
            a_, b_ = md5(f"{STORES}/{st}/{f_}"), md5(srcp)
            ok(a_ == b_, f"{st}/{f_} md5 {a_[:12]} == template tree {srcp}")
        oldm = json.load(open(f"{HK}/stores/{st}/meta.json"))
        newm = json.load(open(f"{STORES}/{st}/meta.json"))
        diffk = sorted(k for k in set(oldm) | set(newm) if oldm.get(k) != newm.get(k))
        ok(set(diffk) <= {"gene_names", "feature", "a10x"}, f"{st}/meta.json differs from the old meta only in {diffk}")
    CA = f"{R3}/cellstore_ccf/allen638850"
    for f_, ref_f in (("pos.npy", "pos.npy"), ("sec.npy", "sec.npy"), ("cls.npy", "cls.npy")):
        ok(np.array_equal(np.load(f"{d}/{f_}"), np.load(f"{CA}/{ref_f}")[cs_row]), f"{ALLEN}/{f_} == cellstore_ccf/allen638850/{ref_f}[cellstore_row]")
    labels = open(f"{CA}/cell_labels.txt").read().split("\n")[:-1]
    ok(open(f"{d}/cell_labels.txt").read().split("\n")[:-1] == [labels[i] for i in cs_row], f"{ALLEN}/cell_labels.txt == cell-store labels[cellstore_row]")
    cid = np.load(f"{HK}/allen/cell_ID.npy")
    ok(np.array_equal(cid[s2_row], cs_row), f"s2_row and cellstore_row are consistent with the old cell_ID permutation (cell_ID[s2_row] == cellstore_row)")

    for st in list(ZH) + [ALLEN]:
        mp = f"{STORES}/{st}/meta.json"; m = json.load(open(mp))
        key = ZH.get(st, "allen")
        m["a10x"]["fill_status"] = status if not FAILS else "FAILED"
        m["a10x"]["filled_at"] = time.strftime("%F %T"); m["a10x"]["feat_source"] = src[key]
        m["a10x"]["feat_md5"] = report["stores"][st]["feat_md5"]; m["a10x"]["filled_on"] = HOST
        json.dump(m, open(mp, "w"), indent=1)
    with open(f"{os.path.dirname(STORES)}/fill_report.json", "w") as f:
        json.dump(report, f, indent=1)
    md = {f"{st}/feat.npy": report["stores"][st]["feat_md5"] for st in list(ZH) + [ALLEN]}
    log("feat.npy md5: " + json.dumps(md))
    log(f"{len(FAILS)} failure(s); {time.time() - t0:.0f} s")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
