"""Protocol R: pool LUNA-3D's per-graph test predictions into one layout of the test specimen, in our layout format.

LUNA predicts normalised, centred coordinates per graph in an arbitrary orientation. Protocol R uses only what our model is given:
the unit is the mean per-axis extent of LUNA's training graphs, all test graphs are pooled as output, and ONE rotation / reflection
(no scale) is fitted for the whole specimen against the truth in the training ruler. Output npz keys: P (layout in units of
--out_mm_per_unit, truth row order), G (truth pos.npy), cc (truth cell_class, int64), iso.
"""
import argparse, glob, json, os
import numpy as np, pandas as pd
from scipy.linalg import svd

ap = argparse.ArgumentParser()
ap.add_argument("--pred_root", required=True, help="LUNA run dir holding test_part*/**/metadata_pred.csv")
ap.add_argument("--train_table", required=True, help="training table: binary dir (pos.npy, section_id.npy) or a LUNA train.csv")
ap.add_argument("--truth_dir", required=True, help="test prebuilt dir (pos.npy, cell_class.npy)")
ap.add_argument("--row_map", required=True, help="npy: truth (prebuilt) row -> row of --truth_pos and of the cell ids")
ap.add_argument("--truth_pos", required=True, help="npy of truth positions indexed by --row_map (store pos.npy, or the prebuilt pos_canonical.npy)")
ap.add_argument("--truth_scale", type=float, required=True, help="mm per unit of --truth_pos")
ap.add_argument("--store_ruler", default="1,1,1", help="per-axis factors of the test store (calibrated mm)")
ap.add_argument("--train_ruler", default="1,1,1", help="per-axis factors of the training ruler")
ap.add_argument("--fold_meta", default="", help="store meta.json with midline_ml_iso: fit against the truth folded at the midline")
ap.add_argument("--id_prefix", type=int, required=True, help="LUNA cell id = id_prefix x 10,000,000 + row")
ap.add_argument("--mixed_parts", action="store_true", help="the run also holds graphs of another test specimen (other id prefix)")
ap.add_argument("--out_mm_per_unit", type=float, required=True, help="mm per unit of the written P")
ap.add_argument("--save_rotation", default="", help="optional npy path for the fitted rotation")
ap.add_argument("--out", required=True, help="output npz path")
a = ap.parse_args()
ROOT, SRC, NUM, ISO_P = a.pred_root, a.train_table, a.id_prefix, a.out_mm_per_unit
S_STORE = np.array([float(v) for v in a.store_ruler.split(",")]); S_REF = np.array([float(v) for v in a.train_ruler.split(",")])
cls = np.load(f"{a.truth_dir}/cell_class.npy").astype(np.int64); N = len(cls)
ROWS = np.load(a.row_map)
pre_of = np.full(int(ROWS.max()) + 1, -1); pre_of[ROWS] = np.arange(N)
Xc = np.load(a.truth_pos).astype(np.float64)[ROWS] * a.truth_scale * S_STORE
if a.fold_meta:
    FOLD_MID = float(json.load(open(a.fold_meta))["midline_ml_iso"]) * a.truth_scale * S_STORE[0]
    Xc[:, 0] = FOLD_MID - np.abs(Xc[:, 0] - FOLD_MID)
Xt = Xc / S_REF
if os.path.isdir(SRC):
    pos = np.load(f"{SRC}/pos.npy", mmap_mode="r"); gid = np.load(f"{SRC}/section_id.npy")
else:
    df = pd.read_csv(SRC, usecols=["coord_X", "coord_Y", "coord_Z", "cell_section"]); pos = df[["coord_X", "coord_Y", "coord_Z"]].to_numpy()
    gid = df["cell_section"].to_numpy()
gid = pd.factorize(np.asarray(gid))[0]
pos = np.asarray(pos, dtype=np.float64); o = np.argsort(gid, kind="stable"); g_sorted = gid[o]; cut = np.flatnonzero(np.diff(g_sorted)) + 1
ext = np.array([p.max(0) - p.min(0) for p in np.split(pos[o], cut)])
EXT = ext.mean(0)
print(f"LUNA training graphs {len(ext)}; mean per-axis extent (training-ruler mm) {EXT.round(3)} "
      f"(p10 {np.percentile(ext, 10, 0).round(2)}, p90 {np.percentile(ext, 90, 0).round(2)}); test truth extent "
      f"{(Xt.max(0) - Xt.min(0)).round(3)}", flush=True)
Pn = np.full((N, 3), np.nan); seen = np.zeros(N, int); graphs = 0
for f in sorted(glob.glob(f"{ROOT}/test_part*/**/metadata_pred.csv", recursive=True)):
    pr = pd.read_csv(f, index_col=0)
    if a.mixed_parts and (pr.index // 10_000_000 != NUM).all():
        continue
    assert (pr.index // 10_000_000 == NUM).all()
    i = pre_of[pr.index.to_numpy() % 10_000_000]; assert (i >= 0).all()
    pn = pr[["coord_X", "coord_Y", "coord_Z"]].to_numpy(); Pn[i] = pn - pn.mean(0); seen[i] += 1; graphs += 1
done = seen == 1
print(f"{graphs} LUNA graphs; {int(done.sum()):,} / {N:,} cells covered; seen twice {int((seen > 1).sum())}", flush=True)
assert (seen <= 1).all()
Xn = (Xt - Xt[done].mean(0)) / EXT
U, _, Vt = svd(Pn[done].T @ Xn[done]); R = U @ Vt
Y = (Pn @ R) * EXT
ang = float(np.degrees(np.arccos(np.clip((np.trace(R * np.sign(np.linalg.det(R))) - 1) / 2, -1, 1))))
dev = []
for f in sorted(glob.glob(f"{ROOT}/test_part*/**/metadata_pred.csv", recursive=True))[:400]:
    pr = pd.read_csv(f, index_col=0)
    if a.mixed_parts and (pr.index // 10_000_000 != NUM).all():
        continue
    i = pre_of[pr.index.to_numpy() % 10_000_000]
    b_ = Pn[i]; b = Xn[i] - Xn[i].mean(0); u, _, vt = svd(b_.T @ b); r = u @ vt; d = r.T @ R
    dev.append(float(np.degrees(np.arccos(np.clip((np.trace(d * np.sign(np.linalg.det(d))) - 1) / 2, -1, 1)))) + (180.0 if np.linalg.det(d) < 0 else 0))
dev = np.array(dev)
Ya = Y[done] - Y[done].mean(0); Xa = Xt[done] - Xt[done].mean(0)
print(f"global rotation {ang:.1f} deg, reflection {bool(np.linalg.det(R) < 0)}; per-graph deviation from it: median {np.median(dev):.1f} deg, "
      f"> 30 deg {np.mean(dev > 30):.3f}", flush=True)
if a.save_rotation:
    print(f"per-graph deviations (deg): {np.round(dev, 1).tolist()}", flush=True)
    print(f"R (LUNA frame -> data frame) = {R.round(3).tolist()}; largest |entry| per row {np.abs(R).max(1).round(3).tolist()}", flush=True)
    np.save(a.save_rotation, R)
print(f"size without rescaling (output RMS / truth RMS, training ruler, data axes): {(Ya.std(0) / Xa.std(0)).round(3)}", flush=True)
if done.all():
    np.savez_compressed(a.out, P=(Y / ISO_P).astype(np.float32), G=np.load(f"{a.truth_dir}/pos.npy"), cc=cls, iso=True)
    print("saved", a.out, flush=True)
