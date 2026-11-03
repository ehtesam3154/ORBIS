# Portions adapted from Penn Software REMAP (https://github.com/ShunzhouJiang/REMAP, commit b663e2b).
# Copyright (C) 2025 The Trustees of the University of Pennsylvania. All rights reserved.
# Used and modified under the REMAP COPYRIGHT AND PERMISSION NOTICE (non-profit research purposes only);
# the full notice, conditions and disclaimer are in LICENSE_REMAP.txt in this folder.
"""REMAP on any store layout (Allen, tissue blocks, embryos): reference = the given cell stores, query = every row of --query_dir
(a store or a test prebuilt). Options: a measured-gene subset (--genes_meta), zero-count handling, an externally corrected
input fed as the authors' tutorials do (--corr_*: harmony=False + obsm['corrected']), and the fit path (--fit). Writes
final_coords.npz (pred in mm, query row order) to --out.
"""
import argparse, gc, json, os, sys, threading, time
import numpy as np, torch
import anndata as ad
from remap.main import Fit_cord_single
import remap_lowmem as LM
ap = argparse.ArgumentParser()
ap.add_argument("--ref_dirs", required=True); ap.add_argument("--query_dir", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--iso_mm", type=float, default=0.0); ap.add_argument("--n_ref", type=int, default=0)
ap.add_argument("--genes_meta", default=""); ap.add_argument("--genes_key", default="measured")
ap.add_argument("--drop_zero_ref", type=int, default=0); ap.add_argument("--query_zero", default="abort", choices=["abort", "fill_mean"])
ap.add_argument("--harmony", type=int, default=0); ap.add_argument("--test", type=int, default=0)
ap.add_argument("--n_iter", type=int, default=0, help="tests only; 0 = official (3, or 1 with --test 1)")
ap.add_argument("--corr_dirs", default=""); ap.add_argument("--corr_query", default=""); ap.add_argument("--corr_query_map", default="")
ap.add_argument("--corr_pca", default=""); ap.add_argument("--fit", default="lowmem", choices=["lowmem", "lowmem2", "official"])
a = ap.parse_args()
t0 = time.time(); OUT = a.out; os.makedirs(OUT, exist_ok=True)
REFS = [d.rstrip("/") for d in a.ref_dirs.split(",")]; NAMES = [os.path.basename(d) for d in REFS]
assert len(set(NAMES)) == len(NAMES), NAMES
metas = [json.load(open(f"{d}/meta.json")) for d in REFS]
genes_all = list(metas[0]["gene_names"]); assert all(list(m["gene_names"]) == genes_all for m in metas), "stores differ in gene order"
ISO = a.iso_mm if a.iso_mm else float(metas[0]["iso_mm"])
assert all(abs(float(m.get("iso_mm", ISO)) - ISO) < 1e-12 for m in metas), "stores differ in iso_mm"
qmeta = json.load(open(f"{a.query_dir}/meta.json")) if os.path.exists(f"{a.query_dir}/meta.json") else {}
if "gene_names" in qmeta:
    assert list(qmeta["gene_names"]) == genes_all, "query gene order differs from the stores"
qfeat = f"{a.query_dir}/feat.npy"; nq, gq = np.load(qfeat, mmap_mode="r").shape
assert gq == len(genes_all), (gq, len(genes_all))
if a.genes_meta:
    v = json.load(open(a.genes_meta))[a.genes_key]
    GCOLS = np.flatnonzero(np.asarray(v, bool)) if not isinstance(v[0], str) else np.flatnonzero(np.isin(genes_all, v))
else:
    GCOLS = np.arange(len(genes_all))
genes = [genes_all[i] for i in GCOLS]
def nonzero_rows(path, n, step=200000):
    f = np.load(path, mmap_mode="r"); ok = np.empty(n, bool)
    for i in range(0, n, step):
        ok[i:i + step] = (np.asarray(f[i:i + step])[:, GCOLS] > 0).any(1)
    return ok
rng = np.random.default_rng(0); RROWS, DROPPED, src, L = {}, {}, [], []
sizes = {d: np.load(f"{d}/pos.npy", mmap_mode="r").shape[0] for d in REFS}
keep = {d: (nonzero_rows(f"{d}/feat.npy", n) if a.drop_zero_ref else np.ones(n, bool)) for d, n in sizes.items()}
tot = sum(int(k.sum()) for k in keep.values())
for d, nm in zip(REFS, NAMES):
    kept = np.flatnonzero(keep[d]); DROPPED[d] = np.flatnonzero(~keep[d])
    rows = np.sort(rng.choice(kept, int(round(a.n_ref * len(kept) / tot)), replace=False)) if a.n_ref else kept
    RROWS[d] = rows; src += [nm] * len(rows)
    L.append(np.asarray(np.load(f"{d}/pos.npy", mmap_mode="r")[rows], dtype=np.float64) * ISO)
loc = np.concatenate(L); del L
qrows_all = np.arange(nq) if not a.test else np.sort(rng.choice(nq, 20000, replace=False))
qok = nonzero_rows(qfeat, nq)[qrows_all]
if not qok.all() and a.query_zero == "abort":
    raise SystemExit(f"{int((~qok).sum())} query cells have zero counts over the used genes (REMAP drops them); rerun with --query_zero fill_mean")
qrows = qrows_all[qok]
def load_ref_counts(cols=None):
    fc = GCOLS if cols is None else GCOLS[np.asarray(cols)]
    out = np.empty((sum(len(r) for r in RROWS.values()), len(fc)), np.float32); o = 0
    for d, r in RROWS.items():
        LM.load_counts_into(f"{d}/feat.npy", r, out[o:o + len(r)], fc); o += len(r)
    return out
def load_q():
    return LM.load_counts_into(qfeat, qrows, np.empty((len(qrows), len(GCOLS)), np.float32), GCOLS)
Rdata = ad.AnnData(load_ref_counts()); Rdata.var_names = genes; Rdata.obs_names = [f"r{i}" for i in range(Rdata.n_obs)]
Rdata.obs["source"] = src; Rdata.obsm["spatial"] = loc
Qdata = ad.AnnData(load_q()); Qdata.var_names = genes; Qdata.obs_names = [f"q{i}" for i in range(Qdata.n_obs)]
print(f"reference {Rdata.n_obs:,} cells ({', '.join(f'{nm} {len(RROWS[d]):,} (dropped zero-count {len(DROPPED[d])})' for d, nm in zip(REFS, NAMES))}), "
      f"query {Qdata.n_obs:,} of {len(qrows_all):,} (zero-count left out: {int((~qok).sum())}), genes {len(genes)}, iso_mm {ISO}, "
      f"harmony {a.harmony}, loaded in {time.time() - t0:.0f} s", flush=True)
if a.harmony:
    def _rss():
        pk = 0.0
        while True:
            r = int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9; pk = max(pk, r)
            print(f"[rss] t={time.time() - t0:.0f}s rss={r:.1f}GB peak_since_start={pk:.1f}GB", flush=True); time.sleep(60)
    threading.Thread(target=_rss, daemon=True).start()
NITER = a.n_iter if a.n_iter else (1 if a.test else 3)
_hold = [Rdata, Qdata]; del Rdata, Qdata
Rdata, Qdata = LM.covet_init_lowmem(*_hold, OUT, load_ref_counts, load_q, batch_key="source", _hold=_hold,
                                    **({"epochs": 50} if a.test else {}))
if a.corr_dirs:
    P = np.load(a.corr_pca); assert [str(g) for g in P["genes"]] == genes, "pca.npz genes != the used genes (order)"
    comp, zm, sd, mu = (P[k].astype(np.float64) for k in ("components", "zmean", "sd", "mean"))
    def recon(pcs):
        out = np.empty((len(pcs), comp.shape[0]), np.float32)
        for i in range(0, len(pcs), 200000):
            x = (np.asarray(pcs[i:i + 200000], np.float64) @ comp.T + zm) * sd + mu
            out[i:i + 200000] = np.maximum(np.expm1(x), 0.0)
        return out
    CD = a.corr_dirs.split(","); assert len(CD) == len(REFS)
    for d, c in zip(REFS, CD):
        f = np.load(f"{c}/feat.npy", mmap_mode="r"); assert f.shape == (sizes[d], comp.shape[1]), (c, f.shape)
    Ecorr = np.concatenate([recon(np.load(f"{c}/feat.npy", mmap_mode="r")[RROWS[d]]) for d, c in zip(REFS, CD)])
    fq = np.load(a.corr_query, mmap_mode="r"); mq = np.load(a.corr_query_map); assert len(mq) == nq and len(np.unique(mq)) == nq
    inv = np.empty(nq, np.int64); inv[mq] = np.arange(nq)
    Qcorr = recon(np.asarray(fq)[inv[qrows]])
    for A_, E_ in ((Rdata, Ecorr), (Qdata, Qcorr)):
        s = E_.sum(1); assert (s >= 1).all(), "a reconstructed cell has < 1 count (REMAP's lib_norm would drop it)"
        A_.obsm["corrected"] = E_; A_.uns["corrected_var_names"] = list(genes); A_.X = None
    print(f"corrected input: reference {Ecorr.shape}, query {Qcorr.shape}; row sums ref {np.percentile(Ecorr.sum(1), [0, 50, 100]).round(1)}, "
          f"query {np.percentile(Qcorr.sum(1), [0, 50, 100]).round(1)} ({time.time() - t0:.0f} s)", flush=True)
    del Ecorr, Qcorr; gc.collect()
if a.harmony:
    pred = Fit_cord_single(Rdata, loc, Qdata, OUT, n_iter=NITER, harmony=True)
elif a.fit == "official":
    pred = Fit_cord_single(Rdata, loc, Qdata, OUT, n_iter=NITER, harmony=False)
elif a.fit == "lowmem2":
    import remap_lowmem_fit2 as LM2
    pred = LM2.fit_cord_single_lowmem2(Rdata, loc, Qdata, OUT, n_iter=NITER)
else:
    pred = LM.fit_cord_single_lowmem(Rdata, loc, Qdata, OUT, n_iter=NITER, harmony=False)
pred = np.asarray(pred, dtype=np.float64); assert pred.shape == (len(qrows), 3) and np.isfinite(pred).all(), pred.shape
full = np.empty((len(qrows_all), 3)); full[qok] = pred; full[~qok] = pred.mean(0)
np.savez(f"{OUT}/final_coords.npz", pred=full, rows=qrows_all, filled_rows=qrows_all[~qok], units="mm (reference stores' frame)",
         n_ref=sum(len(r) for r in RROWS.values()), ref_names=np.array(NAMES), iso_mm=ISO, harmony=a.harmony, n_genes=len(genes),
         **{f"ref_rows_{k}": RROWS[d] for k, d in enumerate(REFS)}, **{f"dropped_ref_{k}": DROPPED[d] for k, d in enumerate(REFS)})
print(f"DONE {full.shape} (filled {int((~qok).sum())}) in {time.time() - t0:.0f} s -> {OUT}/final_coords.npz", flush=True)
