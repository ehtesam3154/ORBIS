"""Truth (test prebuilt) row -> cell-store row map for a specimen whose prebuilt and store hold the same cells in different orders.

The prebuilt positions are mapped into the store frame with the store's frame_fit rotation and matched to the store positions exactly
(kd-tree, distance < 1e-4); ties are resolved to distinct rows. The map is a permutation, saved as an int npy.
"""
import argparse, json
import numpy as np
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--prebuilt", required=True, help="test prebuilt dir (pos.npy)")
ap.add_argument("--store", required=True, help="cell store dir (pos.npy, meta.json with frame_fit.rotation)")
ap.add_argument("--out", required=True, help="output npy")
a = ap.parse_args()
pre = np.load(f"{a.prebuilt}/pos.npy").astype(np.float64); st1 = np.load(f"{a.store}/pos.npy").astype(np.float64)
Rf = np.asarray(json.load(open(f"{a.store}/meta.json"))["frame_fit"]["rotation"])
mapped = (pre - pre.mean(0)) @ Rf + st1.mean(0); dk, ik = cKDTree(st1).query(mapped, k=8, workers=-1)
row = ik[:, 0].copy(); taken = np.zeros(len(st1), bool)
dup = np.flatnonzero(np.bincount(row, minlength=len(st1))[row] > 1); taken[row] = True; taken[row[dup]] = False
for i in dup:
    j = next(int(r) for r, dd in zip(ik[i], dk[i]) if dd < 1e-4 and not taken[r]); row[i] = j; taken[j] = True
assert np.linalg.norm(mapped - st1[row], axis=1).max() < 1e-4 and len(np.unique(row)) == len(row)
np.save(a.out, row)
print("saved", a.out, row.shape, flush=True)
