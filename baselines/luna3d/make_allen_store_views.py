"""Assemble the Allen-fold store views LUNA-3D reads: the training stores with their Harmony-corrected PCs as features and the
Allen test store with its PCs written in store row order (checked row by row).

usage: python make_allen_store_views.py OUT REF1_META PREBUILT HARMONY_DIR STORE_ROOT
"""
import json, os, shutil, sys, time
import numpy as np
OUT, REF1_META, PRE, H, C = sys.argv[1:6]
os.makedirs(OUT, exist_ok=True); t0 = time.time()
def link(src, dst):
    assert os.path.exists(src), src
    if os.path.islink(dst):
        os.remove(dst)
    os.symlink(src, dst)
for s in ("abca1", "abca3", "abca4"):
    link(f"{H}/stores/{s}", f"{OUT}/{s}")
d = f"{OUT}/abca2_ref1"; os.makedirs(d, exist_ok=True)
link(f"{H}/stores/abca2/feat.npy", f"{d}/feat.npy"); link(f"{C}/abca2_ref1/pos.npy", f"{d}/pos.npy")
for f in ("cls.npy", "sec.npy", "cell_labels.txt"):
    link(f"{C}/abca2/{f}", f"{d}/{f}")
shutil.copyfile(REF1_META, f"{d}/meta.json"); m = json.load(open(f"{d}/meta.json"))
assert m["gene_names"] == json.load(open(f"{H}/stores/abca2/meta.json"))["gene_names"]
assert np.load(f"{d}/feat.npy", mmap_mode="r").shape[0] == np.load(f"{d}/pos.npy", mmap_mode="r").shape[0] == m["n_cells"]
d = f"{OUT}/allen638850"; os.makedirs(d, exist_ok=True)
cid = np.load(f"{PRE}/cell_ID.npy"); N = len(cid)
assert np.array_equal(np.sort(cid), np.arange(N)), "prebuilt cell_ID is not a permutation of the store rows"
src = np.load(f"{H}/allen/feat.npy", mmap_mode="r"); assert src.shape == (N, 200) and src.dtype == np.float32
F = np.lib.format.open_memmap(f"{d}/feat.npy", mode="w+", dtype=np.float32, shape=src.shape)
inv = np.argsort(cid)
for j in range(0, N, 250000):
    F[j:j + 250000] = src[inv[j:j + 250000]]
F.flush(); del F
G = np.load(f"{d}/feat.npy", mmap_mode="r"); bad = 0
for i in range(0, N, 250000):
    bad += int((np.asarray(G[cid[i:i + 250000]]) != np.asarray(src[i:i + 250000])).any(1).sum())
assert bad == 0, f"{bad} rows differ"
link(f"{C}/allen638850/pos.npy", f"{d}/pos.npy"); link(f"{C}/allen638850/cls.npy", f"{d}/cls.npy")
cm = json.load(open(f"{C}/allen638850/meta.json")); hm = json.load(open(f"{H}/allen/meta.json"))
cls_store = np.load(f"{d}/cls.npy")[cid]; cls_pre = np.load(f"{PRE}/cell_class.npy")
for k in ("measured", "n_measured", "count_keep", "count_keep_note"):
    cm.pop(k, None)
cm.update(n_genes=200, gene_names=hm["gene_names"], feature=hm["feature"],
          note_luna=("make_allen_store_views.py: feat.npy = the Harmony PCs (prebuilt row order) written to store row cell_ID[i]; "
                     "pos / cls = the Allen cell store (unfolded); classes = the cell store's"))
json.dump(cm, open(f"{d}/meta.json", "w"))
print(json.dumps({"out": OUT, "allen_cells": int(N), "rows_compared": int(N), "rows_differing": bad,
                  "prebuilt cell_class == store cls[cell_ID]": bool(np.array_equal(cls_store, cls_pre)),
                  "seconds": round(time.time() - t0)}), flush=True)
