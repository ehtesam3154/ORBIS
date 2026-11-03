"""CeLEry-3D: CeLEry's coordinate regressor (patched to 3 outputs) trained on every reference cell, then applied to every query
cell. Defaults as released (hidden [30, 25, 15], batch 4, lr 1e-3); epochs = max(1, ceil(728,000 / steps per epoch)).
Writes final_coords.npz (pred in mm, query row order) to --out.
"""
import argparse, json, math, os, sys, time
import numpy as np, pandas as pd, anndata as ad
import CeLEry as cel
ap = argparse.ArgumentParser()
ap.add_argument("--ref_dirs", required=True); ap.add_argument("--query_dir", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--pcs", action="store_true"); ap.add_argument("--pc_ref_dirs", default=""); ap.add_argument("--pc_query", default="")
ap.add_argument("--pc_query_map", default=""); ap.add_argument("--iso_mm", type=float, default=0.0); ap.add_argument("--test", type=int, default=0)
a = ap.parse_args(); t0 = time.time(); os.makedirs(a.out, exist_ok=True)
say = lambda m: print(f"[{time.time() - t0:.0f} s] {m}", flush=True)
REFS = [d.rstrip("/") for d in a.ref_dirs.split(",")]
metas = [json.load(open(f"{d}/meta.json")) for d in REFS]
ISO = a.iso_mm if a.iso_mm else float(metas[0]["iso_mm"])
assert all(abs(float(m.get("iso_mm", ISO)) - ISO) < 1e-12 for m in metas)
rng = np.random.default_rng(0)
def lognorm_inplace(X, step=200000):
    for i in range(0, len(X), step):
        c = np.round(np.exp2(X[i:i + step]) - 1.0); s = c.sum(1, keepdims=True); s[s == 0] = 1.0
        X[i:i + step] = np.log1p(c / s * 1e4)
    return X
def zscore_inplace(X, step=200000):
    n = len(X); s1 = np.zeros(X.shape[1]); s2 = np.zeros(X.shape[1])
    for i in range(0, n, step):
        x = X[i:i + step].astype(np.float64); s1 += x.sum(0); s2 += (x * x).sum(0)
    mu = s1 / n; sd = np.sqrt(np.maximum(s2 / n - mu * mu, 0.0))
    if (sd == 0).any():
        say(f"WARNING: {int((sd == 0).sum())} features with zero sd (CeLEry's get_zscore would give NaN there)")
    for i in range(0, n, step):
        X[i:i + step] = ((X[i:i + step] - mu) / sd).astype(np.float32)
    return mu, sd
def feats(dirs_feat, rows_list, pc_mode):
    n = sum(len(r) for r in rows_list); g = np.load(dirs_feat[0], mmap_mode="r").shape[1]
    X = np.empty((n, g), np.float32); o = 0
    for p, r in zip(dirs_feat, rows_list):
        f = np.load(p, mmap_mode="r")
        for i in range(0, len(r), 200000):
            X[o + i:o + i + len(r[i:i + 200000])] = f[r[i:i + 200000]]
        o += len(r)
    return X if pc_mode else lognorm_inplace(X)
sizes = [np.load(f"{d}/pos.npy", mmap_mode="r").shape[0] for d in REFS]
rrows = [np.arange(n) if not a.test else np.sort(rng.choice(n, max(1, 20000 * n // sum(sizes)), replace=False)) for n in sizes]
Rfeat = [f"{c}/feat.npy" for c in (a.pc_ref_dirs.split(",") if a.pcs else REFS)]
XR = feats(Rfeat, rrows, a.pcs)
L = np.concatenate([np.load(f"{d}/pos.npy", mmap_mode="r")[r].astype(np.float64) * ISO * 1000.0 for d, r in zip(REFS, rrows)])
muR, sdR = zscore_inplace(XR)
say(f"reference {XR.shape} ({', '.join(f'{os.path.basename(d)} {len(r):,}' for d, r in zip(REFS, rrows))}), features {'200 Harmony PCs' if a.pcs else 'log-normalised genes'}, iso_mm {ISO}")
R = ad.AnnData(XR); R.obs = pd.DataFrame(L.astype(np.float32), columns=["x", "y", "z"], index=R.obs_names)
steps = math.ceil(len(XR) / 4); epochs = max(1, math.ceil(728000 / steps)) if not a.test else 1
say(f"training: batch 4 -> {steps:,} steps / epoch, epochs {epochs} ({steps * epochs:,} updates; rule max(1, ceil(728,000 / steps)))")
cel.Fit_cord(data_train=R, hidden_dims=[30, 25, 15], num_epochs_max=epochs, path=a.out, filename="celery3d")
cord = L.astype(np.float32); lo = cord.min(0) - 1; hi = cord.max(0) + 1
del R, XR
say("trained")
nq = np.load(f"{a.query_dir}/feat.npy", mmap_mode="r").shape[0]
qrows = np.arange(nq) if not a.test else np.sort(rng.choice(nq, 20000, replace=False))
if a.pcs:
    mq = np.load(a.pc_query_map); inv = np.empty(len(mq), np.int64); inv[mq] = np.arange(len(mq))
    XQ = np.asarray(np.load(a.pc_query, mmap_mode="r"))[inv[qrows]].astype(np.float32)
else:
    XQ = feats([f"{a.query_dir}/feat.npy"], [qrows], False)
muQ, sdQ = zscore_inplace(XQ)
Q = ad.AnnData(XQ)
pn = cel.Predict_cord(data_test=Q, path=a.out, filename="celery3d")
pred_um = pn.astype(np.float32) * (hi - lo) + lo
assert pred_um.shape == (len(qrows), 3) and np.isfinite(pred_um).all()
np.savez(f"{a.out}/final_coords.npz", pred=pred_um.astype(np.float64) / 1000.0, rows=qrows, units="mm (reference stores' frame)",
         norm_lo_um=lo, norm_hi_um=hi, z_mean_ref=muR, z_sd_ref=sdR, z_mean_query=muQ, z_sd_query=sdQ, epochs=epochs, steps_per_epoch=steps,
         n_ref=len(L), iso_mm=ISO, ref_dirs=np.array(REFS))
say(f"DONE {pred_um.shape} -> {a.out}/final_coords.npz")
