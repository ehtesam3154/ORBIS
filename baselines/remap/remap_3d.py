# Portions adapted from Penn Software REMAP (https://github.com/ShunzhouJiang/REMAP, commit b663e2b).
# Copyright (C) 2025 The Trustees of the University of Pennsylvania. All rights reserved.
# Used and modified under the REMAP COPYRIGHT AND PERMISSION NOTICE (non-profit research purposes only);
# the full notice, conditions and disclaimer are in LICENSE_REMAP.txt in this folder.
"""REMAP on the brain folds: reference = cell stores of the training animals (cells, 3-D positions in mm), query = every cell of
the test animal's store (expression only). Writes final_coords.npz (pred in mm, store row order) in the run folder.

Default path = REMAP's released Fit_cord_single with its covariance initialisation; --lowmem_all 1 = remap_lowmem.py, the
memory-only, same-math version that fits every reference cell (harmony 0).
"""
import argparse, gc, json, os, sys, time
import numpy as np, torch
import anndata as ad, scanpy as sc
from remap.ENVI import ENVI
from remap.util import covet_pca, covet_pcafit
from remap.main import Fit_cord_single
ap = argparse.ArgumentParser(); ap.add_argument("--n_ref", type=int, default=400000, help="reference cells drawn uniformly, proportional per animal; 0 = every cell of every reference animal"); ap.add_argument("--test", type=int, default=0)
ap.add_argument("--ref", default="abca1,abca3,abca4"); ap.add_argument("--root", required=True, help="directory holding the cell stores"); ap.add_argument("--out_root", required=True, help="parent directory of the run folder"); ap.add_argument("--raw_dir", default="", help="atlas CCF coordinate tables (--coords ccf only)")
ap.add_argument("--coords", default="store", choices=["store", "ccf"], help="store = the cell stores' rigid CCF-oriented section-stack coordinates (our truth); ccf = the ABC atlas CCF-registered (atlas-warped) coordinates from raw/Zhuang-ABCA-N-CCF_ccf_coordinates.csv; reference cells without CCF are dropped")
ap.add_argument("--query", default="abca2", help="the test animal's cell store (every cell, expression only)")
ap.add_argument("--out_tag", default="", help="run folder suffix bl_refreg_<out_tag> instead of the reference-derived name")
ap.add_argument("--envi_cpu_data", type=int, default=0, help="1 = keep the niche model's training tensors in host memory and move each batch to the GPU (the official ENVI.train puts ~21 GB on the GPU; a 20 GB A4500 OOMs); same batches, same math")
ap.add_argument("--harmony", type=int, default=1, help="Fit_cord_single(harmony=...) (remap/main.py:128): 1 = Harmony; 0 = REMAP's own no-batch-correction option")
ap.add_argument("--lowmem_all", type=int, default=0, help="1 = remap_lowmem.py: memory-only, same-math pipeline that fits ALL reference cells (harmony 0 only)")
ap.add_argument("--n_iter", type=int, default=0, help="Fit_cord_single n_iter; 0 = official (3, or 1 with --test 1); other values for tests only")
a = ap.parse_args()
if a.lowmem_all:
    import remap_lowmem as LM
S = a.root; ISO = 13.45773983001709; REFS = a.ref.split(",")
OUT = (f"{a.out_root}/bl_refreg_" + (a.out_tag if a.out_tag else "m" + "".join(m[-1] for m in REFS)) + ("_ccf" if a.coords == "ccf" else "")
       + ("_test" if a.test else "")); os.makedirs(OUT, exist_ok=True)
RAW = a.raw_dir
def ccf_rows(m):
    import pandas as pd
    lab = pd.read_csv(f"{S}/{m}/cell_labels.txt", header=None, names=["cell_label"], dtype=str)
    c = pd.read_csv(f"{RAW}/Zhuang-ABCA-{m[-1]}-CCF_ccf_coordinates.csv", usecols=["cell_label", "x", "y", "z"], dtype={"cell_label": str})
    return lab.merge(c, on="cell_label", how="left")[["x", "y", "z"]].to_numpy(dtype=np.float64)
genes = json.load(open(f"{S}/abca1/meta.json"))["gene_names"]
rng = np.random.default_rng(0); t0 = time.time()
def counts(m, rows):
    f = np.load(f"{S}/{m}/feat.npy", mmap_mode="r"); out = np.empty((len(rows), f.shape[1]), np.float32)
    for i in range(0, len(rows), 200000):
        r = rows[i:i + 200000]; out[i:i + len(r)] = np.round(np.exp2(np.asarray(f[r], dtype=np.float32)) - 1.0)
    return out
sizes = {m: np.load(f"{S}/{m}/pos.npy", mmap_mode="r").shape[0] for m in REFS}
tot = sum(sizes.values()); X, L, src, RROWS = [], [], [], {}
for m, n in sizes.items():
    if a.n_ref:
        k = int(round(a.n_ref * n / tot)); rows = np.sort(rng.choice(n, k, replace=False))
    else:
        k = n; rows = np.arange(n)
    if a.coords == "ccf":
        C = ccf_rows(m); rows = rows[~np.isnan(C[rows, 0])]; k = len(rows); RROWS[m] = rows
        X.append(None if a.lowmem_all else counts(m, rows)); L.append(C[rows]); src += [m] * k; continue
    RROWS[m] = rows
    X.append(None if a.lowmem_all else counts(m, rows)); L.append(np.asarray(np.load(f"{S}/{m}/pos.npy", mmap_mode="r")[rows], dtype=np.float64) * ISO)
    src += [m] * k
def load_ref_counts(cols=None):
    out = np.empty((sum(len(r) for r in RROWS.values()), len(genes) if cols is None else len(cols)), np.float32); o = 0
    for m, r in RROWS.items():
        LM.load_counts_into(f"{S}/{m}/feat.npy", r, out[o:o + len(r)], cols); o += len(r)
    return out
Rdata = ad.AnnData(load_ref_counts() if a.lowmem_all else np.concatenate(X)); Rdata.var_names = genes; Rdata.obs_names = [f"r{i}" for i in range(Rdata.n_obs)]
Rdata.obs["source"] = src; loc = np.concatenate(L); Rdata.obsm["spatial"] = loc; del X, L
nq = np.load(f"{S}/{a.query}/pos.npy", mmap_mode="r").shape[0]
qrows = np.arange(nq) if not a.test else np.sort(rng.choice(nq, 20000, replace=False))
Qdata = ad.AnnData(counts(a.query, qrows)); Qdata.var_names = genes; Qdata.obs_names = [f"q{i}" for i in range(Qdata.n_obs)]
print(f"reference {Rdata.n_obs:,} cells ({', '.join(f'{m} {src.count(m):,}' for m in sizes)}), query {Qdata.n_obs:,}, "
      f"loaded in {time.time() - t0:.0f} s", flush=True)
def covet_init_chunked(st_data, sc_data, save_path, num_covet_genes=100, k_nearest=100, num_HVG=1000, covet_genes=[], stable=1e-6,
                       batch_key=None, epochs=10000, spatial_distribution="pois", sc_distribution="nb", log_input=0.1,
                       lib_size=False, covet_batch_size=256, train_batch_size=2048, pve=0.98, sample_gene=None, chunk=50000, _hold=None):
    os.makedirs(save_path, exist_ok=True)
    st_data_raw = st_data.copy(); sc_data_raw = sc_data.copy()
    if _hold is not None:
        _hold.clear()
    if st_data_raw.shape[0] > 200000:
        sample_gene = min(300, sample_gene) if sample_gene is not None else 300
    if sample_gene is not None:
        st_data.layers["log"] = np.log(st_data.X + 1)
        sc.pp.highly_variable_genes(st_data, n_top_genes=sample_gene, layer="log")
        st_data = st_data[:, st_data.var["highly_variable"]].copy()
    envi_model = ENVI(spatial_data=st_data, sc_data=sc_data, num_covet_genes=num_covet_genes, covet_genes=covet_genes,
                      k_nearest=k_nearest, spatial_distribution=spatial_distribution, sc_distribution=sc_distribution,
                      batch_key=batch_key, num_HVG=num_HVG, covet_batch_size=covet_batch_size, log_input=log_input,
                      lib_size=lib_size, stable=stable)
    print(f"ENVI built ({time.time() - t0:.0f} s)", flush=True)
    del st_data, sc_data
    envi_model.train(batch_size=train_batch_size, epochs=epochs)
    print(f"ENVI trained ({time.time() - t0:.0f} s)", flush=True)
    lat_sc = np.asarray(envi_model.sc_data.obsm["envi_latent"]); envi_model.sc_data = None; gc.collect()
    covet_train_pca, pca_model = covet_pca(envi_model.spatial_data.obsm["COVET"], pve=pve, flat=True)
    lat = lat_sc; parts = []
    with torch.no_grad():
        for i in range(0, len(lat), chunk):
            cs = envi_model.covet_decode(torch.from_numpy(lat[i:i + chunk]).to(envi_model.device).float()).detach().cpu().numpy()
            parts.append(covet_pcafit(np.matmul(cs, np.transpose(cs, (0, 2, 1))), pca_model, flat=True)); del cs
    covet_test_pca = np.concatenate(parts)
    np.save(f"{save_path}/st_covariance.npy", covet_train_pca); np.save(f"{save_path}/sc_covariance.npy", covet_test_pca)
    print(f"covariance PCA: {covet_train_pca.shape[1]} components ({time.time() - t0:.0f} s)", flush=True)
    del envi_model; gc.collect(); torch.cuda.empty_cache()
    st_data_raw.obsm["covet"] = covet_train_pca; sc_data_raw.obsm["covet"] = covet_test_pca
    return st_data_raw, sc_data_raw
def envi_train_cpu_data(self, lr=1e-4, batch_size=512, epochs=10000, verbose=64, lr_schedule=True,
                        device="cuda" if torch.cuda.is_available() else "cpu", seed=2025):
    from tqdm import trange
    print(f"Training ENVI on {device} (host-resident data)")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed_all(seed)
    self.device = device
    self.to(device)
    self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    spatial_data = torch.tensor(self.spatial_data.X, dtype=torch.float32)
    cov_data = torch.tensor(self.spatial_data.obsm["COVET_SQRT"], dtype=torch.float32)
    sc_data = torch.tensor(self.sc_data.X, dtype=torch.float32)
    tq = trange(epochs, desc="")
    for step in tq:
        if lr_schedule and step == int(epochs * 0.75):
            for g in self.optimizer.param_groups:
                g["lr"] *= 0.1
        spatial_idx = torch.randint(0, len(spatial_data), (min(batch_size, len(spatial_data)),))
        sc_idx = torch.randint(0, len(sc_data), (min(batch_size, len(sc_data)),))
        spatial_sample = spatial_data[spatial_idx].to(device)
        cov_sample = cov_data[spatial_idx].to(device)
        sc_sample = sc_data[sc_idx].to(device)
        self.optimizer.zero_grad()
        spatial_likelihood, covet_likelihood, sc_likelihood, kl = self.compute_loss(spatial_sample, cov_sample, sc_sample)
        loss = (-self.spatial_weight * spatial_likelihood - self.sc_weight * sc_likelihood
                - self.covet_weight * covet_likelihood + 2 * self.kl_weight * kl)
        if torch.isnan(loss).any():
            print("NaN encountered in loss. Step not applied.")
            continue
        loss.backward()
        self.optimizer.step()
        if verbose > 0 and step % verbose == 0:
            tq.set_description(f"Trn: spatial Loss: {spatial_likelihood.item():.5f}, SC Loss: {sc_likelihood.item():.5f}, "
                               f"Cov Loss: {covet_likelihood.item():.5f}, KL Loss: {kl.item():.5f}", refresh=True)
    del spatial_data, cov_data, sc_data
    print("Finished initializing neighboring gene-gene covariance estimation.")
    self.latent_rep()
    del self.spatial_data.obsm["COVET_SQRT"]
if a.envi_cpu_data:
    ENVI.train = envi_train_cpu_data
NITER = a.n_iter if a.n_iter else (1 if a.test else 3)
_hold = [Rdata, Qdata]; del Rdata, Qdata
if a.lowmem_all:
    load_q = lambda: LM.load_counts_into(f"{S}/{a.query}/feat.npy", qrows, np.empty((len(qrows), len(genes)), np.float32))
    Rdata, Qdata = LM.covet_init_lowmem(*_hold, OUT, load_ref_counts, load_q, batch_key="source", _hold=_hold,
                                        **({"epochs": 50} if a.test else {}))
    pred = LM.fit_cord_single_lowmem(Rdata, loc, Qdata, OUT, n_iter=NITER, harmony=bool(a.harmony))
else:
    Rdata, Qdata = covet_init_chunked(*_hold, OUT, batch_key="source", _hold=_hold, **({"epochs": 50} if a.test else {}))
    pred = Fit_cord_single(Rdata, loc, Qdata, OUT, n_iter=NITER, harmony=bool(a.harmony))
np.savez(f"{OUT}/final_coords.npz", pred=np.asarray(pred, dtype=np.float64), rows=qrows, units="mm (CCF-oriented store frame)",
         n_ref=Rdata.n_obs)
print(f"DONE {pred.shape} in {time.time() - t0:.0f} s -> {OUT}/final_coords.npz", flush=True)
if a.coords == "ccf":
    Tq = ccf_rows(a.query)[qrows]; ok = ~np.isnan(Tq[:, 0]); P_, T_ = np.asarray(pred, dtype=np.float64)[ok], Tq[ok]
    mm = lambda Z: (Z - Z.min(0)) / (Z.max(0) - Z.min(0))
    def pcorr(A, B):
        r = np.random.default_rng(0); v = []
        for _ in range(5):
            i, j = r.integers(0, len(A), 2000000), r.integers(0, len(A), 2000000)
            v.append(np.corrcoef(np.linalg.norm(A[i] - A[j], axis=1), np.linalg.norm(B[i] - B[j], axis=1))[0, 1])
        return float(np.mean(v)), float(np.std(v))
    Pc, Tc = P_ - P_.mean(0), T_ - T_.mean(0); U, _, Vt = np.linalg.svd(Pc.T @ Tc); Pr = Pc @ (U @ Vt)
    res = dict(n_eval=int(ok.sum()), pairwise_corr_no_rotation=pcorr(mm(T_), mm(P_)), pairwise_corr_rotated=pcorr(mm(Tc), mm(Pr)))
    np.savez(f"{OUT}/final_coords.npz", pred=np.asarray(pred, dtype=np.float64), rows=qrows, ccf_truth=Tq, units="mm (CCF atlas space)",
             n_ref=Rdata.n_obs, metrics=json.dumps(res))
    print("CCF EVAL", json.dumps(res), flush=True)
