"""Unit-box scale of the A2 ablation: statistics of the training views of a checkpoint's own view stream (the trainer's
view generator with the checkpoint's view arguments and seed). Each view's positions are min-max rescaled per axis to
[-0.5, 0.5] and centred; per view it records the view label, specimen, cells, raw per-axis extent, and the per-axis RMS of
the unit-box and of the raw view. Only positions are read: view k is the (k div W)-th view of the generator of loader
worker k mod W (W = views_workers, worker w seeded seed * 100003 + 1000 w), the order in which the trainer's DataLoader
delivers them. Nothing is trained; CPU only. Writes OUT/dunit_views.npz (per-view table) and OUT/dunit_summary.json;
protocol R reads by_label.WHOLE.raw_extent_mean_store; the arm's data_rms is by_label.WHOLE.unit_rms_pooled_all_axes
(6 decimals) and its noise_scale = data_rms / 0.163.
usage: python -m scripts.ablations.unit_box_scale --ckpt CKPT --stores DIR[,DIR...] --mm_per_unit MM --out OUT [--n 1200]"""
import argparse
import json
import os
import time

import numpy as np
import torch

from orbis.data.views import FreshViews

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, help="checkpoint whose training view arguments define the stream")
ap.add_argument("--stores", required=True, help="the checkpoint's training cell stores, comma-separated, in training order")
ap.add_argument("--mm_per_unit", type=float, required=True, help="millimetres per store unit (extents are also reported in mm)")
ap.add_argument("--n", type=int, default=1200, help="views to draw")
ap.add_argument("--out", required=True, help="output directory")
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)
ISO = A.mm_per_unit

a = torch.load(A.ckpt, map_location="cpu", weights_only=True)["args"]
stores = A.stores.split(",")
rank, start_ep = 0, 0
seed = a["seed"] * 100003 + rank * 1009 + start_ep
kw = dict(mix=[float(x) for x in a["views_mix"].split(",")],
          n_range=tuple(int(x) for x in a["views_n"].split(",")), thin_min=a["views_thin"],
          whole_animals=[int(x) for x in a["views_whole"].split(",")] if a["views_whole"] else None,
          fold_axis=a["views_fold_axis"])
gens = [FreshViews(stores, seed=seed + 1000 * w, **kw) for w in range(max(int(a["views_workers"]), 1))]
rows = []
t0 = time.time()
for k in range(A.n):
    fv = gens[k % len(gens)]
    sel, animal, vt_k, _keep = fv.draw()
    pos = np.ascontiguousarray(fv.stores[animal].pos[sel])
    if fv.fold_axis >= 0:
        pos = np.array(pos, copy=True)
        pos[:, fv.fold_axis] = np.abs(pos[:, fv.fold_axis])
    p = torch.from_numpy(pos).float().numpy().astype(np.float64)
    lo, hi = p.min(0), p.max(0)
    ext = hi - lo
    q = (p - lo) / np.where(ext > 0, ext, np.nan) - 0.5
    q = q - q.mean(0)
    rms_unit = np.sqrt((q ** 2).mean(0))
    pc = p - p.mean(0)
    rms_raw = np.sqrt((pc ** 2).mean(0))
    rows.append((int(vt_k), int(animal), len(p), *ext, *rms_unit, *rms_raw, *lo, *hi))
    if (k + 1) % 200 == 0:
        print(f"[{time.time() - t0:5.0f} s] {k + 1} views", flush=True)
R = np.array(rows, dtype=np.float64)
vt, an, nc = R[:, 0].astype(int), R[:, 1].astype(int), R[:, 2].astype(int)
ext, rms_u, rms_r = R[:, 3:6], R[:, 6:9], R[:, 9:12]
np.savez(f"{A.out}/dunit_views.npz", view_type=vt, animal=an, cells=nc, extent_store=ext, rms_unit=rms_u, rms_raw=rms_r,
         lo=R[:, 12:15], hi=R[:, 15:18])
NAMES = {0: "WHOLE", 1: "SECTION_COR", 2: "SECTION_SAG", 3: "BLOCK_COR", 4: "BLOCK_SAG"}
AX = ["x(ML)", "y(DV)", "z(AP)"]


def pooled(r):
    """RMS over all coordinates: sqrt of the mean over views and axes of the per-view per-axis variance."""
    return float(np.sqrt((r ** 2).mean()))


def per_axis_pooled(r):
    """Per axis: sqrt of the mean over views of the per-view variance."""
    return np.sqrt((r ** 2).mean(0))


S = dict(n_views=int(A.n), stream=f"training views, seed {a['seed']}, {len(gens)} loader workers", stores=stores,
         iso_mm_per_unit=ISO, by_type={}, by_label={}, axes=AX)
print(f"\n{A.n} stream views drawn in {time.time() - t0:.0f} s; view types {dict(zip(*np.unique(vt, return_counts=True)))}; animals {dict(zip(*np.unique(an, return_counts=True)))}")
print(f"min raw extent per axis over ALL views (store units): {ext.min(0)}; over SECTION views: {ext[(vt == 1) | (vt == 2)].min(0) if ((vt == 1) | (vt == 2)).any() else None}")
for name, m in [("ALL", np.ones(len(vt), bool)), ("WHOLE", vt == 0), ("SECTION", (vt == 1) | (vt == 2)), ("BLOCK", (vt == 3) | (vt == 4))]:
    if not m.any():
        continue
    d = dict(n=int(m.sum()), cells_mean=float(nc[m].mean()),
             unit_rms_mean_over_views=rms_u[m].mean(0).tolist(), unit_rms_pooled_per_axis=per_axis_pooled(rms_u[m]).tolist(),
             unit_rms_pooled_all_axes=pooled(rms_u[m]), unit_rms_mean_of_axes=float(rms_u[m].mean()),
             raw_rms_pooled_per_axis=per_axis_pooled(rms_r[m]).tolist(), raw_rms_pooled_all_axes=pooled(rms_r[m]),
             raw_extent_mean_store=ext[m].mean(0).tolist(), raw_extent_mean_mm=(ext[m].mean(0) * ISO).tolist(),
             raw_extent_min_store=ext[m].min(0).tolist(), raw_extent_p10_store=np.percentile(ext[m], 10, 0).tolist(),
             raw_extent_p90_store=np.percentile(ext[m], 90, 0).tolist())
    S["by_type"][name] = d
    print(f"\n[{name}] n {d['n']}, mean cells {d['cells_mean']:.0f}")
    print(f"  unit-box RMS per axis (mean over views): {np.round(d['unit_rms_mean_over_views'], 4).tolist()}  pooled per axis: {np.round(d['unit_rms_pooled_per_axis'], 4).tolist()}")
    print(f"  unit-box RMS pooled over all coordinates {d['unit_rms_pooled_all_axes']:.4f} | mean of the 3 axis means {d['unit_rms_mean_of_axes']:.4f}")
    print(f"  raw RMS per axis pooled: {np.round(d['raw_rms_pooled_per_axis'], 4).tolist()} -> all coordinates {d['raw_rms_pooled_all_axes']:.4f}")
    print(f"  raw extent mean (store units): {np.round(d['raw_extent_mean_store'], 4).tolist()} = {np.round(d['raw_extent_mean_mm'], 3).tolist()} mm; min {np.round(d['raw_extent_min_store'], 5).tolist()}")
for k in range(5):
    m = vt == k
    if m.any():
        S["by_label"][NAMES[k]] = dict(n=int(m.sum()), unit_rms_pooled_all_axes=pooled(rms_u[m]), unit_rms_pooled_per_axis=per_axis_pooled(rms_u[m]).tolist(),
                                      raw_extent_mean_store=ext[m].mean(0).tolist(), raw_extent_min_store=ext[m].min(0).tolist())
S["nan_views"] = int(np.isnan(rms_u).any(1).sum())
json.dump(S, open(f"{A.out}/dunit_summary.json", "w"), indent=1)
print(f"\nNaN views (zero extent on some axis): {S['nan_views']}; wrote {A.out}/dunit_summary.json and dunit_views.npz")
