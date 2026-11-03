"""Convert a baseline's final coordinates (every row of the test specimen, store order, mm) to our layout format.

Input npz keys: pred (n x 3, mm in the training ruler), rows (must be 0..n-1). Output npz keys: P (pred / --mm_per_unit, in the
truth (prebuilt) row order, float32), G (prebuilt pos.npy, float32), cc (prebuilt cell_class.npy), iso (True).
The truth row -> store row map is --row_map (a permutation); without it the query was the prebuilt itself (identity map).
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("src", help="final_coords.npz of a baseline run")
ap.add_argument("out", help="output layout npz")
ap.add_argument("--truth_dir", required=True, help="test prebuilt dir (pos.npy, cell_class.npy)")
ap.add_argument("--mm_per_unit", type=float, required=True, help="mm per unit of the written P")
ap.add_argument("--row_map", default="", help="npy: truth row -> store row; omit when the query was the prebuilt")
ap.add_argument("--store_pos", default="", help="store pos.npy (its row count must equal the map length)")
a = ap.parse_args()
P_DIR, ISO = a.truth_dir, a.mm_per_unit
if not a.row_map:
    m = np.arange(np.load(f"{P_DIR}/pos.npy", mmap_mode="r").shape[0])
else:
    m = np.load(a.row_map)
n_store = np.load(a.store_pos, mmap_mode="r").shape[0] if a.store_pos else len(m)
assert len(m) == n_store and len(np.unique(m)) == n_store and m.min() == 0 and m.max() == n_store - 1, "truth -> store map is not a permutation"
z = np.load(a.src); pred, rows = np.asarray(z["pred"], dtype=np.float64), np.asarray(z["rows"])
assert len(rows) == n_store and np.array_equal(rows, np.arange(n_store)), "expects every store row, in store order"
assert np.isfinite(pred).all() and pred.shape == (n_store, 3), pred.shape
P = pred[m] / ISO
np.savez_compressed(a.out, P=P.astype(np.float32), G=np.load(f"{P_DIR}/pos.npy").astype(np.float32),
                    cc=np.load(f"{P_DIR}/cell_class.npy"), iso=True)
print("saved", a.out, P.shape, flush=True)
