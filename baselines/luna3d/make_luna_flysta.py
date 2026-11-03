"""Write LUNA-3D's inputs for specimens with one physical ruler (whole embryos, tissue blocks): the binary training table
(every training cell once, random ~5,000-cell graphs, seeded) and the test CSVs (one part per test prebuilt), plus
test_tiny.csv (one whole test graph) and train_infos.csv (one training graph).
"""
import argparse, json, os, time
from multiprocessing import Pool
import numpy as np
import pandas as pd
STORES = ["stores/16.32h_E12-14h_c", "stores/16.53h_E16-18h_b", "stores/16.57h_E16-18h_a", "stores/16.63h_E16-18h_e",
          "stores/17.61h_E14-16h_f", "stores_more/13.79h_E10-12h_d", "stores_more/15.75h_E12-14h_a", "stores_more/15.77h_E12-14h_b",
          "stores_more/18.39h_E2.5-3h_d", "stores_more/18.97h_E2.5-3h_c", "stores_more/19.84h_E18-20h_d"]
TEST = {"16.88h_E14-16h_d": 20, "17.34h_E14-16h_c": 21}
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--graph", type=int, default=5000); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--root", required=True, help="store root holding build_summary.json (+ optional build_summary_more.json)")
ap.add_argument("--stores", default=",".join(STORES), help="training stores, relative to --root, in order")
ap.add_argument("--test", default=",".join(f"{k}:{v}" for k, v in TEST.items()), help="test prebuilts NAME:ID_PREFIX (root/prebuilt_test_NAME)")
ap.add_argument("--fold", default="flysta", help="output names fold<FOLD>_allcells_bin / fold<FOLD>")
ap.add_argument("--n_genes", type=int, default=2000)
a = ap.parse_args()
F, STORES = a.root, a.stores.split(",")
TEST = {k: int(v) for k, v in (x.split(":") for x in a.test.split(","))}
t0 = time.time()
BIN, CSV = f"{a.out}/fold{a.fold}_allcells_bin", f"{a.out}/fold{a.fold}"
os.makedirs(BIN, exist_ok=True); os.makedirs(CSV, exist_ok=True)
rng = np.random.default_rng(a.seed)
B = json.load(open(f"{F}/build_summary.json")); BM = json.load(open(f"{F}/build_summary_more.json")) if os.path.exists(f"{F}/build_summary_more.json") else B
classes = list(BM["classes"]); assert classes[:len(B["classes"])] == B["classes"]
hvg = list(json.load(open(B["hvg_source"]))["hvg"]); assert len(set(hvg)) == len(hvg) == a.n_genes
assert not any(("," in g) or ('"' in g) for g in hvg)
meta = {s: json.load(open(f"{F}/{s}/meta.json")) for s in STORES}
for s, m in meta.items():
    assert m["gene_names"] == hvg and classes[:len(m["classes"])] == m["classes"], s
ISO = meta[STORES[0]]["iso_mm"]; assert all(m["iso_mm"] == ISO for m in meta.values())
EMB = [s.split("/")[1] for s in STORES]
def graphs_of(animal, rows):
    rows = rows[rng.permutation(len(rows))]
    k = max(1, int(round(len(rows) / a.graph)))
    return [(f"{animal}_g{i:04d}", r) for i, r in enumerate(np.array_split(rows, k))]
sizes = {s: meta[s]["n_cells"] for s in STORES}
names, A, ROWS, SEC = [], [], [], []
for ai, s in enumerate(STORES):
    rows = np.sort(rng.choice(sizes[s], sizes[s], replace=False))
    for name, r in graphs_of(EMB[ai], rows):
        SEC.append(np.full(len(r), len(names), np.int32)); names.append(name); ROWS.append(r); A.append(np.full(len(r), ai, np.int8))
A, ROWS, SEC = np.concatenate(A), np.concatenate(ROWS), np.concatenate(SEC)
assert all(np.array_equal(np.sort(ROWS[A == ai]), np.arange(sizes[s])) for ai, s in enumerate(STORES)), "not every cell once"
perm = rng.permutation(len(A)); A, ROWS, SEC = A[perm], ROWS[perm], SEC[perm]
INDEX = (A.astype(np.int64) + 1) * 10_000_000 + ROWS
df = pd.DataFrame({"cell_section": np.array(names, dtype=object)[SEC], "pos": np.arange(len(A))}, index=pd.Index(INDEX))
o = df.sort_values("cell_section", ignore_index=False)["pos"].to_numpy()
A, ROWS, SEC, INDEX = A[o], ROWS[o], SEC[o], INDEX[o]
gp = np.argsort(np.array(hvg)); assert [hvg[i] for i in gp] == sorted(hvg)
N, G = len(A), len(hvg)
Fm = np.lib.format.open_memmap(f"{BIN}/feat.npy", mode="w+", dtype=np.float32, shape=(N, G))
P = np.empty((N, 3)); CL = np.empty(N, np.int16)
for ai, s in enumerate(STORES):
    m = np.flatnonzero(A == ai); lo, hi = m[0], m[-1] + 1; assert hi - lo == len(m)
    X = np.load(f"{F}/{s}/feat.npy"); pos = np.load(f"{F}/{s}/pos.npy"); cls = np.load(f"{F}/{s}/cls.npy")
    assert X.dtype == np.float32 and X.shape == (sizes[s], G)
    Fm[lo:hi] = X[ROWS[lo:hi]][:, gp]
    P[lo:hi] = np.asarray(pos, dtype=np.float64)[ROWS[lo:hi]] * ISO; CL[lo:hi] = np.asarray(cls)[ROWS[lo:hi]]
    print(f"  {s}: {hi - lo:,} cells, rows {lo:,}-{hi:,}", flush=True)
Fm.flush(); del Fm
np.save(f"{BIN}/pos.npy", P); np.save(f"{BIN}/index.npy", INDEX); np.save(f"{BIN}/section_id.npy", SEC); np.save(f"{BIN}/class_id.npy", CL)
per = {s: {"cells": int((A == ai).sum()), "of": int(sizes[s]), "graphs": int(len(np.unique(SEC[A == ai]))), "id_prefix": ai + 1}
       for ai, s in enumerate(STORES)}
gsz = np.bincount(SEC)
json.dump({"fold": a.fold, "train_animals": STORES, "test_animals": list(TEST), "graph": a.graph, "seed": a.seed, "cells": int(N),
           "graphs": len(names), "graph_cells_min_max": [int(gsz.min()), int(gsz.max())], "per_animal": per, "genes": sorted(hvg),
           "sections": names, "classes": classes, "store_root": F, "iso_mm": ISO,
           "note": "rows in LUNA's process_data sort order; features unrounded float32 log2(1 + count), genes sorted (filter_genes); "
                   "pos = store pos x iso_mm (mm, marker-posed AP / DV / LR)"}, open(f"{BIN}/meta.json", "w"))
print(json.dumps({"bin": BIN, "cells": int(N), "graphs": len(names), "graph_cells": [int(gsz.min()), int(gsz.max())],
                  "seconds": round(time.time() - t0)}), flush=True)
def to_text(args):
    d, header = args
    return d.to_csv(None, header=header, float_format="%.5g")
def write_csv(path, d):
    t = time.time(); step = 5000
    with Pool(12) as pool, open(path, "w") as f:
        for s_ in pool.imap(to_text, ((d.iloc[j:j + step], j == 0) for j in range(0, len(d), step))):
            f.write(s_)
    print(f"  wrote {path}: {len(d):,} rows, {d['cell_section'].nunique()} graphs, {os.path.getsize(path) / 1e9:.2f} GB, "
          f"{time.time() - t:.0f} s", flush=True)
def frame(Xf, Pmm, cls_names, ids, sec_rows):
    rows = np.concatenate([r for _, r in sec_rows])
    d = pd.DataFrame(Xf[rows], columns=hvg, index=pd.Index(ids[rows], name=None))
    d["coord_X"], d["coord_Y"], d["coord_Z"] = Pmm[rows, 0], Pmm[rows, 1], Pmm[rows, 2]
    d["cell_section"] = np.concatenate([[s_] * len(r) for s_, r in sec_rows])
    d["cell_class"] = cls_names[rows]
    return d
manifest = {"fold": a.fold, "train_bin": BIN, "train_total": int(N), "train_graphs": len(names), "train": per, "genes": G,
            "graph": a.graph, "seed": a.seed, "iso_mm": ISO, "test_parts": []}
for p_, (emb, num) in enumerate(TEST.items()):
    pre = f"{F}/prebuilt_test_{emb}"
    Xf = np.load(f"{pre}/feat.npy"); n = len(Xf); assert Xf.shape == (n, G) and Xf.dtype == np.float32
    assert np.array_equal(np.load(f"{pre}/cell_ID.npy"), np.arange(n))
    Pmm = np.load(f"{pre}/pos_canonical.npy").astype(np.float64) * ISO
    cn = np.asarray(B["classes"], dtype=object)[np.load(f"{pre}/cell_class.npy")]
    g = graphs_of(emb, np.arange(n))
    d = frame(Xf, Pmm, cn, num * 10_000_000 + np.arange(n), g); d = d.iloc[rng.permutation(len(d))]
    path = f"{CSV}/test_part{p_:02d}.csv"; write_csv(path, d)
    manifest["test_parts"].append({"path": path, "embryo": emb, "id_prefix": num, "cells": int(n), "graphs": len(g),
                                   "graph_cells_min_max": [int(min(len(r) for _, r in g)), int(max(len(r) for _, r in g))]})
    if p_ == 0:
        first = d["cell_section"].iloc[0]; write_csv(f"{CSV}/test_tiny.csv", d[d["cell_section"] == first])
        manifest["test_tiny_graph"] = first
manifest["test_total"] = int(sum(t["cells"] for t in manifest["test_parts"]))
s0 = 0; m0 = np.flatnonzero(SEC == s0)
Xi = np.load(f"{BIN}/feat.npy", mmap_mode="r")[m0]
d = pd.DataFrame(np.asarray(Xi), columns=sorted(hvg), index=pd.Index(INDEX[m0]))
d["coord_X"], d["coord_Y"], d["coord_Z"] = P[m0, 0], P[m0, 1], P[m0, 2]
d["cell_section"] = names[s0]; d["cell_class"] = np.asarray(classes, dtype=object)[CL[m0]]
write_csv(f"{CSV}/train_infos.csv", d); manifest["train_infos_graph"] = names[s0]
json.dump(manifest, open(f"{CSV}/manifest.json", "w"), indent=1)
print(json.dumps({k: v for k, v in manifest.items() if k not in ("train",)}), flush=True)
print(f"DONE {time.time() - t0:.0f} s", flush=True)
