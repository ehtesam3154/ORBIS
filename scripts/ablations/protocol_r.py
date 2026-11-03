"""Protocol R for a unit-box model sampled with fixed independent windows: every window is min-max rescaled to
[-0.5, 0.5] per axis and centred, the pooled windows get ONE rotation / reflection onto the truth (no scale), and the
result is multiplied by the unit EXT = the mean per-axis extent of the model's own WHOLE training views (dunit JSON,
by_label.WHOLE.raw_extent_mean_store). --mode truth applies the same steps to the true positions (the bound).
The dunit JSON is written by scripts/ablations/unit_box_scale.py (OUT/dunit_summary.json), run on the checkpoint whose
view stream the unit-box model was trained on:
  python -m scripts.ablations.unit_box_scale --ckpt CKPT --stores DIR[,DIR...] --mm_per_unit MM --out OUT
usage: python scripts/ablations/protocol_r.py --draw NPZ --dunit JSON --out NPZ [--minmax 1] [--mode model|truth]"""
import argparse
import json

import numpy as np
from scipy.linalg import svd

ap = argparse.ArgumentParser()
ap.add_argument("--draw", required=True, help="layout written with --fixed_windows 1 (keys P, G, cc, iso, win)")
ap.add_argument("--dunit", required=True, help="JSON with by_label.WHOLE.raw_extent_mean_store")
ap.add_argument("--out", required=True, help="output npz")
ap.add_argument("--minmax", type=int, default=1, help="1 = min-max rescale every window")
ap.add_argument("--mode", default="model", choices=["model", "truth"], help="truth = the true positions through the same windows and steps")
A = ap.parse_args()
z = np.load(A.draw)
assert all(k in z.files for k in ("P", "G", "cc", "iso", "win")), z.files
win = z["win"].astype(np.int64); N = len(win); n_win = int(win.max()) + 1
Xt = z["G"].astype(np.float64)
EXT = np.array(json.load(open(A.dunit))["by_label"]["WHOLE"]["raw_extent_mean_store"], dtype=np.float64)
src = Xt if A.mode == "truth" else z["P"].astype(np.float64)
Pn = np.empty((N, 3)); raw_ext = []
for w in range(n_win):
    idx = np.flatnonzero(win == w)
    p = src[idx]
    lo, hi = p.min(0), p.max(0)
    raw_ext.append(hi - lo)
    if A.minmax or A.mode == "truth":
        p = (p - lo) / (hi - lo) - 0.5
    Pn[idx] = p - p.mean(0)
raw_ext = np.array(raw_ext)
Xn = (Xt - Xt.mean(0)) / EXT
U, _, Vt = svd(Pn.T @ Xn)
R = U @ Vt
Y = (Pn @ R) * EXT
ang = float(np.degrees(np.arccos(np.clip((np.trace(R * np.sign(np.linalg.det(R))) - 1) / 2, -1, 1))))
Ya, Xa = Y - Y.mean(0), Xt - Xt.mean(0)
info = dict(mode=A.mode, draw=A.draw, out=A.out, n_cells=int(N), n_windows=n_win, ext_model_units=EXT.tolist(), minmax=A.minmax,
            raw_window_extent_mean=raw_ext.mean(0).tolist(), raw_window_extent_min=raw_ext.min(0).tolist(), rotation_deg=ang,
            reflection=bool(np.linalg.det(R) < 0), size_rms_ratio=(Ya.std(0) / Xa.std(0)).tolist(),
            extent_ratio=((Y.max(0) - Y.min(0)) / (Xt.max(0) - Xt.min(0))).tolist())
np.savez_compressed(A.out, P=Y.astype(np.float32), G=z["G"], cc=z["cc"], iso=z["iso"], R=R, ext=EXT, win=z["win"])
print("PROTOCOL_R", json.dumps(info), flush=True)
