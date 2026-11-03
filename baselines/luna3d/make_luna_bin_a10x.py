"""Dissociated-cell application: write LUNA-3D's binary training table from the five spatial brains (Zhuang mice 1-4 and the Allen
hemisphere) in the 200-feature reference space; every cell once, random ~5,000-cell graphs (seed 0), rows in LUNA's sort order.
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
FOLDS = {"a10x": (["abca1", "abca2_ref1", "abca3", "abca4", "allen638850_hemi0"], None)}
ID_NUM = {"allen638850_hemi0": 9}
ap = argparse.ArgumentParser()
ap.add_argument("--store_root", required=True)
ap.add_argument("--fold", type=lambda s: int(s) if s.isdigit() else s, required=True)
ap.add_argument("--graph", type=int, default=5000); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--keep_graphs", type=int, default=0); ap.add_argument("--out", required=True)
a = ap.parse_args()
RD = a.store_root
t0 = time.time()
train_animals, test_animal = FOLDS[a.fold]
out = a.out
rng = np.random.default_rng(a.seed)
meta = {s: json.load(open(f"{RD}/{s}/meta.json")) for s in train_animals}
genes = meta[train_animals[0]]["gene_names"]; classes = meta[train_animals[0]]["classes"]
assert all(m["a10x"]["fill_status"] == "FILLED" for m in meta.values()) or os.environ.get("A10X_ALLOW_REHEARSAL") == "1", "stores are not FILLED with the new features (set A10X_ALLOW_REHEARSAL=1 for the old-basis timing profile)"
assert len(genes) == 200 and all("count_keep" not in m for m in meta.values())
os.makedirs(out, exist_ok=True)
assert all(m["gene_names"] == genes and m["classes"] == classes for m in meta.values())
def graphs_of(animal, rows):
    rows = rows[rng.permutation(len(rows))]
    k = max(1, int(round(len(rows) / a.graph)))
    return [(f"{animal}_g{i:04d}", r) for i, r in enumerate(np.array_split(rows, k))]
sizes = {s: meta[s]["n_cells"] for s in train_animals}; tot = sum(sizes.values()); n_train = tot
names, A, ROWS, SEC = [], [], [], []
for ai, s in enumerate(train_animals):
    k = int(round(n_train * sizes[s] / tot)); assert k == sizes[s]
    rows = np.sort(rng.choice(sizes[s], k, replace=False))
    g = graphs_of(s, rows)
    if a.keep_graphs:
        g = g[:a.keep_graphs]
    for name, r in g:
        SEC.append(np.full(len(r), len(names), np.int32)); names.append(name); ROWS.append(r); A.append(np.full(len(r), ai, np.int8))
A, ROWS, SEC = np.concatenate(A), np.concatenate(ROWS), np.concatenate(SEC)
if not a.keep_graphs:
    assert all(np.array_equal(np.sort(ROWS[A == ai]), np.arange(sizes[s])) for ai, s in enumerate(train_animals)), "not every cell once"
perm = rng.permutation(len(A)); A, ROWS, SEC = A[perm], ROWS[perm], SEC[perm]
NUM = np.array([ID_NUM.get(s) or int(s[4]) for s in train_animals], np.int64)
INDEX = NUM[A] * 10_000_000 + ROWS
df = pd.DataFrame({"cell_section": np.array(names, dtype=object)[SEC], "pos": np.arange(len(A))}, index=pd.Index(INDEX))
o = df.sort_values("cell_section", ignore_index=False)["pos"].to_numpy()
if a.keep_graphs:
    np.save(f"{out}/presort_index.npy", INDEX)
A, ROWS, SEC, INDEX = A[o], ROWS[o], SEC[o], INDEX[o]
assert np.all(np.diff(A) >= 0) or sorted(train_animals) != train_animals
gp = np.argsort(np.array(genes))
assert [genes[i] for i in gp] == sorted(genes)
N, G = len(A), len(genes)
F = np.lib.format.open_memmap(f"{out}/feat.npy", mode="w+", dtype=np.float32, shape=(N, G))
P = np.empty((N, 3)); CL = np.empty(N, np.int16)
for ai, s in enumerate(train_animals):
    m = np.flatnonzero(A == ai); lo, hi = m[0], m[-1] + 1; assert hi - lo == len(m)
    t1 = time.time(); X = np.load(f"{RD}/{s}/feat.npy")
    pos = np.load(f"{RD}/{s}/pos.npy"); cls = np.load(f"{RD}/{s}/cls.npy"); iso = meta[s]["iso_mm"]
    for j in range(lo, hi, 100000):
        r = ROWS[j:min(j + 100000, hi)]; F[j:j + len(r)] = X[r][:, gp]
    P[lo:hi] = np.asarray(pos, dtype=np.float64)[ROWS[lo:hi]] * iso; CL[lo:hi] = np.asarray(cls)[ROWS[lo:hi]]
    del X; print(f"  {s}: {hi - lo:,} cells, rows {lo:,}-{hi:,} written ({time.time() - t1:.0f} s)", flush=True)
F.flush(); del F
np.save(f"{out}/pos.npy", P); np.save(f"{out}/index.npy", INDEX); np.save(f"{out}/section_id.npy", SEC); np.save(f"{out}/class_id.npy", CL)
per = {s: {"cells": int((A == ai).sum()), "of": int(sizes[s]), "graphs": int(len(np.unique(SEC[A == ai])))} for ai, s in enumerate(train_animals)}
json.dump({"fold": a.fold, "train_animals": train_animals, "test_animal": test_animal, "graph": a.graph, "seed": a.seed, "stores_fill_status": [meta[s]["a10x"]["fill_status"] for s in train_animals],
           "keep_graphs": a.keep_graphs, "cells": int(N), "graphs": len(names), "per_animal": per, "genes": sorted(genes),
           "sections": names, "classes": classes, "store_root": RD,
           "note": "rows in LUNA's process_data sort order; features unrounded float32, genes sorted (filter_genes)"},
          open(f"{out}/meta.json", "w"))
print(json.dumps({"out": out, "cells": int(N), "graphs": len(names), "per_animal": per, "seconds": round(time.time() - t0)}), flush=True)
