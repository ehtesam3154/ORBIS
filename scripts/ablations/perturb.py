"""Perturbed copy of a test prebuilt: every file is a symlink to the original except feat.npy (log2(1 + counts),
float32), which is rewritten. zero LEVEL: a random LEVEL share of the genes set to 0 in every cell (gene set drawn
once with default_rng(seed)). thin LEVEL: binomial thinning of every cell's counts with keep probability LEVEL
(counts = round(2^f - 1)), drawn with default_rng(seed) in fixed row chunks. A summary is written to perturb.json.
usage: python scripts/ablations/perturb.py SRC_DIR OUT_DIR zero|thin LEVEL [--seed 1031] [--chunk 50000]"""
import argparse
import json
import os
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("kind", choices=["zero", "thin"]); ap.add_argument("level", type=float)
ap.add_argument("--seed", type=int, default=1031); ap.add_argument("--chunk", type=int, default=50000)
A = ap.parse_args()
assert not os.path.exists(A.out), f"{A.out} exists"
os.makedirs(A.out + ".tmp", exist_ok=False)
for f in os.listdir(A.src):
    if f != "feat.npy":
        os.symlink(os.path.realpath(os.path.join(A.src, f)), os.path.join(A.out + ".tmp", f))
F = np.load(os.path.join(A.src, "feat.npy"), mmap_mode="r")
N, G = F.shape
O = np.lib.format.open_memmap(os.path.join(A.out + ".tmp", "feat.npy"), mode="w+", dtype=np.float32, shape=(N, G))
rng = np.random.default_rng(A.seed)
zero = np.zeros(G, bool)
if A.kind == "zero":
    zero[rng.choice(G, int(round(A.level * G)), replace=False)] = True
med_b, med_a, mass_tot, mass_zero = [], [], 0.0, 0.0
t0 = time.time()
for s in range(0, N, A.chunk):
    x = np.asarray(F[s:s + A.chunk], dtype=np.float64)
    c = np.round(np.exp2(x) - 1.0).clip(min=0)
    med_b.append(c.sum(1))
    if A.kind == "zero":
        mass_tot += c.sum(); mass_zero += c[:, zero].sum()
        c[:, zero] = 0.0
    else:
        c = rng.binomial(c.astype(np.int64), A.level).astype(np.float64)
    med_a.append(c.sum(1))
    O[s:s + A.chunk] = np.log2(1.0 + c).astype(np.float32)
O.flush(); del O
b, a = np.concatenate(med_b), np.concatenate(med_a)
rep = dict(src=A.src, out=A.out, kind=A.kind, level_requested=A.level, seed=A.seed, cells=int(N), genes=int(G),
           genes_zeroed=int(zero.sum()), share_genes_zeroed=float(zero.mean()),
           share_count_mass_in_zeroed_genes=(float(mass_zero / mass_tot) if A.kind == "zero" else None),
           median_counts_per_cell_before=float(np.median(b)), median_counts_per_cell_after=float(np.median(a)),
           realised_keep_total_counts=float(a.sum() / b.sum()), seconds=round(time.time() - t0, 1),
           built=time.strftime("%F %T"), script="scripts/ablations/perturb.py")
json.dump(rep, open(os.path.join(A.out + ".tmp", "perturb.json"), "w"), indent=1)
os.rename(A.out + ".tmp", A.out)
print("PERTURB_DONE", json.dumps(rep))
