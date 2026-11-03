"""Write LUNA-3D's CSV inputs for one brain fold: a training CSV, a one-graph train_infos.csv and the test parts.

Every test cell appears once, in random ~5,000-cell graphs (seeded). The training subsample (--n_train) is drawn first, so its
size also fixes the random state of the test partition; the released test parts used the default 1,000,000.
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
from multiprocessing import Pool
FOLDS = {1: (["abca1", "abca3", "abca4"], "abca2"), 2: (["abca2", "abca3", "abca4"], "abca1"), "m12": (["abca1", "abca2_ref1"], "abca3"),
         "allen": (["abca1", "abca2_ref1", "abca3", "abca4"], "allen638850")}
ID_NUM = {"allen638850": 9}
ap = argparse.ArgumentParser()
ap.add_argument("--store_root", required=True, help="directory holding the cell stores")
ap.add_argument("--fold", type=lambda s: int(s) if s.isdigit() else s, required=True); ap.add_argument("--n_train", type=int, default=1000000)
ap.add_argument("--graph", type=int, default=5000); ap.add_argument("--part_graphs", type=int, default=50)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--smoke", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
RD = a.store_root
train_animals, test_animal = FOLDS[a.fold]
out = os.path.join(a.out, f"fold{a.fold}" + ("_smoke" if a.smoke else "")); os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(a.seed)
meta = {s: json.load(open(f"{RD}/{s}/meta.json")) for s in train_animals + [test_animal]}
genes = meta[test_animal]["gene_names"]; classes = meta[test_animal]["classes"]
assert all(m["gene_names"] == genes and m["classes"] == classes for m in meta.values())
def graphs_of(animal, rows):
    rows = rows[rng.permutation(len(rows))]
    k = max(1, int(round(len(rows) / a.graph)))
    return [(f"{animal}_g{i:04d}", r) for i, r in enumerate(np.array_split(rows, k))]
def frame(animal, sec_rows):
    feat = np.load(f"{RD}/{animal}/feat.npy", mmap_mode="r"); pos = np.load(f"{RD}/{animal}/pos.npy", mmap_mode="r")
    cls = np.load(f"{RD}/{animal}/cls.npy", mmap_mode="r"); iso = meta[animal]["iso_mm"]; num = ID_NUM.get(animal) or int(animal[4])
    rows = np.concatenate([r for _, r in sec_rows]); srt = np.argsort(rows); rs = rows[srt]
    X = np.empty((len(rows), len(genes)), np.float32)
    for s in range(0, feat.shape[0], 200000):
        lo, hi = np.searchsorted(rs, [s, s + 200000])
        if hi > lo:
            X[srt[lo:hi]] = np.asarray(feat[s:s + 200000])[rs[lo:hi] - s]
    P = np.asarray(pos, dtype=np.float64)[rows] * iso
    df = pd.DataFrame(X, columns=genes, index=pd.Index(num * 10_000_000 + rows, name=None))
    df["coord_X"], df["coord_Y"], df["coord_Z"] = P[:, 0], P[:, 1], P[:, 2]
    df["cell_section"] = np.concatenate([[s] * len(r) for s, r in sec_rows])
    df["cell_class"] = np.asarray(classes, dtype=object)[np.asarray(cls)[rows].astype(int)]
    return df
def to_text(args):
    df, header = args
    return df.to_csv(None, header=header, float_format="%.5g")
def write_csv(path, df):
    t = time.time(); step = 25000
    with Pool(12) as pool, open(path, "w") as f:
        for i, s in enumerate(pool.imap(to_text, ((df.iloc[j:j + step], j == 0) for j in range(0, len(df), step)))):
            f.write(s)
    print(f"  wrote {path}: {len(df):,} rows, {df['cell_section'].nunique()} graphs, {os.path.getsize(path) / 1e9:.2f} GB, "
          f"{time.time() - t:.0f} s", flush=True)
manifest = {"fold": a.fold, "train_animals": train_animals, "test_animal": test_animal, "graph": a.graph, "seed": a.seed,
            "n_train_requested": a.n_train, "genes": len(genes), "store_root": RD, "train": {}, "test_parts": []}
sizes = {s: meta[s]["n_cells"] for s in train_animals}; tot = sum(sizes.values())
n_train = min(a.n_train, tot) if not a.smoke else 30000
train = []
for s in train_animals:
    k = int(round(n_train * sizes[s] / tot)); rows = np.sort(rng.choice(sizes[s], k, replace=False))
    g = graphs_of(s, rows); train.append(frame(s, g))
    manifest["train"][s] = {"cells": int(k), "of": int(sizes[s]), "graphs": len(g)}
train = pd.concat(train); train = train.iloc[rng.permutation(len(train))]
write_csv(f"{out}/train.csv", train)
first = train["cell_section"].iloc[0]
write_csv(f"{out}/train_infos.csv", train[train["cell_section"] == first])
manifest["train_total"] = int(len(train)); manifest["train_graphs"] = int(train["cell_section"].nunique()); del train
n_test = meta[test_animal]["n_cells"]
g = graphs_of(test_animal, np.arange(n_test) if not a.smoke else rng.choice(n_test, 20000, replace=False))
for p in range(0, len(g), a.part_graphs):
    part = g[p:p + a.part_graphs]; df = frame(test_animal, part); df = df.iloc[rng.permutation(len(df))]
    path = f"{out}/test_part{p // a.part_graphs:02d}.csv"; write_csv(path, df)
    manifest["test_parts"].append({"path": path, "cells": int(len(df)), "graphs": len(part)})
manifest["test_total"] = int(sum(t["cells"] for t in manifest["test_parts"]))
json.dump(manifest, open(f"{out}/manifest.json", "w"), indent=1)
print(json.dumps({k: v for k, v in manifest.items() if k != "test_parts"}), flush=True)
