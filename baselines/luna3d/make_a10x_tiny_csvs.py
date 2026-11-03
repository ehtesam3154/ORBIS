"""Dissociated-cell application: the two one-graph CSVs LUNA's data module loads next to the binary table (train_infos.csv: one
seeded 5,000-cell training graph; test_tiny.csv: one 5,000-cell graph of the Allen hemisphere). The training run tests nothing.
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
ID_NUM = {"allen638850_hemi0": 9}
ap = argparse.ArgumentParser()
ap.add_argument("--store_root", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n", type=int, default=5000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = np.random.default_rng(a.seed)
man = {"store_root": a.store_root, "seed": a.seed, "files": {}}
def graph_csv(animal, name):
    m = json.load(open(f"{a.store_root}/{animal}/meta.json")); genes, classes, iso = m["gene_names"], m["classes"], m["iso_mm"]
    rows = np.sort(rng.choice(m["n_cells"], a.n, replace=False))
    feat = np.load(f"{a.store_root}/{animal}/feat.npy", mmap_mode="r"); pos = np.load(f"{a.store_root}/{animal}/pos.npy", mmap_mode="r")
    cls = np.load(f"{a.store_root}/{animal}/cls.npy", mmap_mode="r")
    num = ID_NUM.get(animal) or int(animal[4])
    P = np.asarray(pos, dtype=np.float64)[rows] * iso
    df = pd.DataFrame(np.asarray(feat[rows], dtype=np.float32), columns=genes, index=pd.Index(num * 10_000_000 + rows))
    df["coord_X"], df["coord_Y"], df["coord_Z"] = P[:, 0], P[:, 1], P[:, 2]
    df["cell_section"] = name
    df["cell_class"] = np.asarray(classes, dtype=object)[np.asarray(cls)[rows].astype(int)]
    return df
for fname, animal, gname in (("train_infos.csv", "abca1", "abca1_info"), ("test_tiny.csv", "allen638850_hemi0", "allen638850_hemi0_tiny")):
    df = graph_csv(animal, gname)
    df.to_csv(f"{a.out}/{fname}", float_format="%.5g")
    man["files"][fname] = {"animal": animal, "graph": gname, "cells": int(len(df)), "columns": int(df.shape[1] + 1)}
    print(f"wrote {a.out}/{fname}: {len(df):,} rows, 1 graph ({gname}), {os.path.getsize(f'{a.out}/{fname}') / 1e6:.1f} MB", flush=True)
json.dump(man, open(f"{a.out}/manifest.json", "w"), indent=1)
