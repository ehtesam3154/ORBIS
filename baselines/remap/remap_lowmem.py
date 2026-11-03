# Portions adapted from Penn Software REMAP (https://github.com/ShunzhouJiang/REMAP, commit b663e2b).
# Copyright (C) 2025 The Trustees of the University of Pennsylvania. All rights reserved.
# Used and modified under the REMAP COPYRIGHT AND PERMISSION NOTICE (non-profit research purposes only);
# the full notice, conditions and disclaimer are in LICENSE_REMAP.txt in this folder.
"""Memory-only, same-math versions of REMAP's covariance initialisation, niche-model training, covariance PCA, query decoding and
preprocessing (arrays in chunks or row files instead of whole in RAM / GPU). Each function mirrors the released code path.
"""
import gc, mmap, os, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, scanpy as sc, torch, sklearn.neighbors, scipy.special
import anndata as ad
from tqdm import trange
from sklearn.utils.extmath import stable_cumsum
import sys
from remap.ENVI import ENVI
ENVI_module = sys.modules["remap.ENVI"]
from remap.ENVI_utils import batch_knn, matrix_square_root
from remap.fit_functions import Fit_cord, Predict_cord, pred_transform, Fit_covet, Predict_covet
from remap.util import use_corrected_expression
T0 = time.time()
CHUNK = 32768
KEEP_FILES = False
WORK = {}
def mem(tag):
    d = {}
    for line in open("/proc/self/status"):
        p = line.split()
        if p and p[0] in ("VmRSS:", "VmHWM:"):
            d[p[0][:-1]] = int(p[1]) * 1024 / 1e9
    g = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    print(f"[mem] {tag}: t={time.time() - T0:.0f}s rss={d['VmRSS']:.1f}GB stage_peak_rss={d['VmHWM']:.1f}GB "
          f"gpu_peak={g:.1f}GB", flush=True)
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
    except OSError:
        pass
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
class RowFile:
    def __init__(self, path, n, shape, create):
        self.path, self.n, self.shape = path, n, tuple(shape)
        self.rb = int(np.prod(self.shape)) * 4
        if create:
            with open(path, "wb") as f:
                f.truncate(n * self.rb)
        self.fd = os.open(path, os.O_RDWR if create else os.O_RDONLY)
    def write(self, i0, arr):
        mv = memoryview(np.ascontiguousarray(arr, dtype=np.float32)).cast("B")
        done, off = 0, i0 * self.rb
        while done < len(mv):
            done += os.pwrite(self.fd, mv[done:], off + done)
    def read(self, i0, i1):
        out = np.empty((i1 - i0,) + self.shape, np.float32)
        mv = memoryview(out).cast("B")
        done, off = 0, i0 * self.rb
        while done < len(mv):
            done += os.preadv(self.fd, [mv[done:]], off + done)
        return out
    def view(self):
        return np.memmap(self.path, np.float32, "r", shape=(self.n,) + self.shape)
    def close(self, remove=False):
        os.close(self.fd)
        if remove:
            os.remove(self.path)
class RowReader:
    def __init__(self, rf, nthreads=32):
        self.fd = os.open(rf.path, os.O_RDONLY)
        os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_RANDOM)
        self.rb, self.nt, self.ex = rf.rb, nthreads, ThreadPoolExecutor(nthreads)
    def _rd(self, mv, idx, j0):
        for j, i in enumerate(idx):
            o, done = (j0 + j) * self.rb, 0
            while done < self.rb:
                done += os.preadv(self.fd, [mv[o + done:o + self.rb]], int(i) * self.rb + done)
    def submit(self, idx, out_np):
        mv = memoryview(out_np).cast("B")
        return [self.ex.submit(self._rd, mv, idx[p], int(p[0])) for p in np.array_split(np.arange(len(idx)), self.nt) if len(p)]
    def close(self):
        self.ex.shutdown()
        os.close(self.fd)
def load_counts_into(feat_path, rows, out, cols=None, step=200000):
    f = np.load(feat_path, mmap_mode="r")
    for i in range(0, len(rows), step):
        r = rows[i:i + step]
        c = np.round(np.exp2(np.asarray(f[r], dtype=np.float32)) - 1.0)
        out[i:i + len(r)] = c if cols is None else c[:, cols]
        f._mmap.madvise(mmap.MADV_DONTNEED)
    return out
def _official_batches(n, covet_batch_size):
    if covet_batch_size is None or covet_batch_size >= n:
        return [(0, n)]
    parts = np.array_split(np.arange(n), np.ceil(n / covet_batch_size))
    return [(int(p[0]), int(p[-1]) + 1) for p in parts]
def _chunks(batches, target):
    out, cur = [], []
    for b in batches:
        cur.append(b)
        if cur[-1][1] - cur[0][0] >= target:
            out.append((cur[0][0], cur[-1][1], cur)); cur = []
    if cur:
        out.append((cur[0][0], cur[-1][1], cur))
    return out
class ExactF32Sum:
    B = 8192
    def __init__(self):
        self.s, self.carry, self.n = np.float32(0), np.empty(0, np.float32), 0
    def add(self, flat):
        self.n += flat.size
        if self.carry.size:
            flat = np.concatenate([self.carry, flat])
        nb = flat.size // self.B
        if nb:
            bs = np.add.reduce(flat[:nb * self.B].reshape(nb, self.B), axis=1)
            self.s = np.add.accumulate(np.concatenate([np.array([self.s], np.float32), bs]))[-1]
        self.carry = flat[nb * self.B:].copy()
    def mean(self):
        s = self.s if not self.carry.size else np.float32(self.s + np.add.reduce(self.carry))
        return s.dtype.type(s / np.intp(self.n))
def get_covet_lowmem(spatial_data, k, g, genes, covet_distribution, spatial_key="spatial", batch_key=-1, covet_pseudocount=1,
                     covet_batch_size=None):
    spatial_data = spatial_data.copy()
    spatial_data.layers["log"] = np.log(spatial_data.X + 1)
    if g == -1:
        covet_gene_set = np.arange(spatial_data.shape[-1])
        spatial_data.var.highly_variable = True
    else:
        sc.pp.highly_variable_genes(spatial_data, n_top_genes=g, layer="log")
        if g == 0:
            spatial_data.var.highly_variable = False
        if len(genes) > 0:
            spatial_data.var["highly_variable"][genes] = True
    covet_gene_set = np.where(np.asarray(spatial_data.var.highly_variable))[0]
    covet_genes = spatial_data.var_names[covet_gene_set]
    print(f"Number of genes used for calculating neighboring gene-gene covariance: {len(covet_genes)}")
    expression_data = spatial_data[:, spatial_data.var.highly_variable].X
    if covet_pseudocount > 0:
        expression_data = np.log(expression_data + covet_pseudocount)
    coords, n = spatial_data.obsm[spatial_key], expression_data.shape[0]
    batched = not (batch_key == -1 or batch_key not in spatial_data.obs.columns)
    batch = spatial_data.obs[batch_key] if batched else None
    del spatial_data
    gc.collect()
    if not batched:
        knn_graph = sklearn.neighbors.kneighbors_graph(coords, n_neighbors=k, mode="distance", n_jobs=-1).tocoo()
        knn_graph.data = knn_graph.data.astype(np.float32)
        knn_graph_index = np.reshape(np.asarray(knn_graph.col), [n, k])
        w_dtype = np.float32
        del knn_graph
    else:
        knn_graph_index, weighted_index = batch_knn(coords, batch, k)
        w_dtype = weighted_index.dtype
        del weighted_index
    gc.collect()
    mean_expr = expression_data.mean(axis=0, dtype=np.float32)
    gn = expression_data.shape[1]
    mem("covet: HVG + kNN")
    def raw_covet(c0, c1):
        weighted_index = np.ones((c1 - c0, k), dtype=w_dtype) / k
        wdm = ((mean_expr[None, None, :] - expression_data[knn_graph_index[np.arange(c0, c1)]]).astype(np.float32)
               * np.sqrt(weighted_index.astype(np.float32))[:, :, None]
               * np.sqrt(1.0 / (1.0 - np.sum(np.square(weighted_index.astype(np.float32)), axis=-1)))[:, None, None])
        covet = np.matmul(wdm.transpose([0, 2, 1]), wdm)
        return covet.astype("float32")
    batches = _official_batches(n, covet_batch_size)
    chunks = _chunks(batches, CHUNK)
    acc = ExactF32Sum()
    for c0, c1, _ in chunks:
        acc.add(raw_covet(c0, c1).reshape(-1))
    cmean = acc.mean()
    eye = cmean * 0.00001 * np.expand_dims(np.identity(gn, dtype=np.float32), axis=0)
    print(f"covet.mean() = {cmean!r} over {acc.n:,} values ({time.time() - T0:.0f} s)", flush=True)
    d = WORK["dir"]
    f_cov = RowFile(f"{d}/COVET.f32", n, (gn, gn), True)
    f_sqrt = RowFile(f"{d}/COVET_SQRT.f32", n, (gn, gn), True)
    for c0, c1, bl in chunks:
        covet = raw_covet(c0, c1)
        covet = covet + eye
        f_cov.write(c0, covet)
        if covet_distribution == "OT":
            sq = np.empty_like(covet)
            for b0, b1 in bl:
                sq[b0 - c0:b1 - c0] = matrix_square_root(covet[b0 - c0:b1 - c0])
        else:
            sq = np.copy(covet)
        f_sqrt.write(c0, sq)
        del covet, sq
    WORK["cov"], WORK["sqrt"] = f_cov, f_sqrt
    del expression_data, knn_graph_index
    gc.collect()
    mem(f"covet: COVET + COVET_SQRT written ({len(batches)} square-root batches)")
    return f_cov.view(), f_sqrt.view(), covet_genes
def envi_train_lowmem(self, lr=1e-4, batch_size=512, epochs=10000, verbose=64, lr_schedule=True,
                      device="cuda" if torch.cuda.is_available() else "cpu", seed=2025):
    print(f"Training ENVI on {device}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed_all(seed)
    self.device = device
    self.to(device)
    self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    spatial_data = torch.tensor(self.spatial_data.X, dtype=torch.float32, device=device)
    f_sqrt = WORK["sqrt"]
    sc_data = torch.tensor(self.sc_data.X, dtype=torch.float32, device=device)
    n_sp, n_sc = len(spatial_data), len(sc_data)
    idx = [(torch.randint(0, n_sp, (min(batch_size, n_sp),)), torch.randint(0, n_sc, (min(batch_size, n_sc),))) for _ in range(epochs)]
    rd, depth = RowReader(f_sqrt), 4
    bufs = [torch.empty((min(batch_size, n_sp),) + f_sqrt.shape, dtype=torch.float32, pin_memory=torch.cuda.is_available())
            for _ in range(depth)]
    pend = {}
    def prefetch(s):
        if s < epochs:
            pend[s] = rd.submit(idx[s][0].numpy(), bufs[s % depth].numpy())
    for s in range(depth - 1):
        prefetch(s)
    tq = trange(epochs, desc="")
    for step in tq:
        prefetch(step + depth - 1)
        if lr_schedule and step == int(epochs * 0.75):
            for gr in self.optimizer.param_groups:
                gr["lr"] *= 0.1
        spatial_idx, sc_idx = idx[step]
        for fu in pend.pop(step):
            fu.result()
        spatial_sample = spatial_data[spatial_idx]
        cov_sample = bufs[step % depth].to(device)
        sc_sample = sc_data[sc_idx]
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
    rd.close()
    del spatial_data, sc_data, bufs, idx
    print("Finished initializing neighboring gene-gene covariance estimation.")
    self.latent_rep()
    del self.spatial_data.obsm["COVET_SQRT"]
class ExactPCA:
    def __init__(self, pve=0.98):
        self.pve = pve
    def fit(self, chunks):
        n, shift = 0, None
        for X in chunks:
            X = X.astype(np.float64)
            if shift is None:
                shift = X.mean(0); p = X.shape[1]; S = np.zeros(p); G = np.zeros((p, p))
            X -= shift
            S += X.sum(0); G += X.T @ X; n += len(X)
        dlt = S / n
        self.mean_ = shift + dlt
        C = (G - n * np.outer(dlt, dlt)) / (n - 1)
        C = (C + C.T) / 2
        ev, V = np.linalg.eigh(C)
        ev, Vt = ev[::-1].copy(), np.ascontiguousarray(V[:, ::-1].T)
        ev[ev < 0.0] = 0.0
        j = np.argmax(np.abs(Vt), axis=1)
        Vt *= np.sign(Vt[np.arange(len(Vt)), j])[:, None]
        ratio = ev / ev.sum()
        cum = stable_cumsum(ratio)
        k = int(np.searchsorted(cum, self.pve, side="right") + 1)
        self.n_samples_, self.n_components_, self.components_ = n, k, Vt[:k].copy()
        self.explained_variance_, self.explained_variance_ratio_ = ev[:k], ratio[:k]
        self.cum_at_cut = (float(cum[k - 2]) if k > 1 else 0.0, float(cum[k - 1]))
        return self
    def transform(self, X):
        return ((X.astype(np.float64) - self.mean_) @ self.components_.T).astype(np.float32)
def _upper(cov):
    iu = np.triu_indices(cov.shape[-1])
    return cov[:, iu[0], iu[1]]
def covet_init_lowmem(st_data, sc_data, save_path, load_st_X, load_sc_X, num_covet_genes=100, k_nearest=100, num_HVG=1000,
                      covet_genes=[], stable=1e-6, batch_key=None, epochs=10000, spatial_distribution="pois",
                      sc_distribution="nb", log_input=0.1, lib_size=False, covet_batch_size=256, train_batch_size=2048, pve=0.98,
                      sample_gene=None, _hold=None):
    os.makedirs(save_path, exist_ok=True)
    WORK["dir"] = f"{save_path}/lowmem_covet"; os.makedirs(WORK["dir"], exist_ok=True)
    if _hold is not None:
        _hold.clear()
    st_cov_path, sc_cov_path = f"{save_path}/st_covariance.npy", f"{save_path}/sc_covariance.npy"
    if os.path.exists(st_cov_path) and os.path.exists(sc_cov_path):
        print("Loading precomputed neighboring gene-gene covariance")
        st_data.obsm["covet"] = np.load(st_cov_path); sc_data.obsm["covet"] = np.load(sc_cov_path)
        return st_data, sc_data
    mem("covet_init: start")
    var0, n_st = st_data.var.copy(), st_data.n_obs
    if n_st > 200000:
        sample_gene = min(300, sample_gene) if sample_gene is not None else 300
    if sample_gene is not None:
        logm = np.empty(st_data.shape, np.float32)
        for i in range(0, n_st, 200000):
            logm[i:i + 200000] = np.log(st_data.X[i:i + 200000] + 1)
        st_data.X = None; sc_data.X = None; gc.collect()
        tmp = ad.AnnData(obs=st_data.obs[[]], var=st_data.var.copy()); tmp.layers["log"] = logm
        sc.pp.highly_variable_genes(tmp, n_top_genes=sample_gene, layer="log")
        hv = np.asarray(tmp.var["highly_variable"])
        st_sub = ad.AnnData(X=load_st_X(np.where(hv)[0]), obs=st_data.obs.copy(), var=tmp.var[hv].copy(), uns=tmp.uns,
                            obsm={kk: v for kk, v in st_data.obsm.items()})
        st_sub.layers["log"] = logm[:, hv]
        del tmp, logm; gc.collect()
        sc_data.X = load_sc_X()
        if KEEP_FILES:
            WORK["st_sub"] = st_sub.copy()
        del st_sub.layers["log"]
        gc.collect()
        mem("covet_init: HVG(300) subset")
    else:
        st_sub = st_data
    ENVI_module.get_covet = get_covet_lowmem
    assert ENVI_module.__name__ == "remap.ENVI" and ENVI.__init__.__globals__["get_covet"] is get_covet_lowmem
    try:
        envi_model = ENVI(spatial_data=st_sub, sc_data=sc_data, num_covet_genes=num_covet_genes, covet_genes=covet_genes,
                          k_nearest=k_nearest, spatial_distribution=spatial_distribution, sc_distribution=sc_distribution,
                          batch_key=batch_key, num_HVG=num_HVG, covet_batch_size=covet_batch_size, log_input=log_input,
                          lib_size=lib_size, stable=stable)
    finally:
        ENVI_module.get_covet = sys.modules["remap.ENVI_utils"].get_covet
    print(f"ENVI built ({time.time() - T0:.0f} s)", flush=True)
    del st_sub
    gc.collect()
    mem("ENVI built")
    envi_train_lowmem(envi_model, batch_size=train_batch_size, epochs=epochs)
    print(f"ENVI trained ({time.time() - T0:.0f} s)", flush=True)
    lat_sc = np.asarray(envi_model.sc_data.obsm["envi_latent"])
    if KEEP_FILES:
        WORK["envi_state"] = {kk: v.detach().cpu().clone() for kk, v in envi_model.state_dict().items()}
        WORK["lat_sc"] = lat_sc.copy()
    envi_model.sc_data = None
    del envi_model.spatial_data.obsm["envi_latent"]
    gc.collect()
    mem("ENVI trained + latents")
    f_cov = WORK["cov"]
    rng_ = [(i, min(i + CHUNK, f_cov.n)) for i in range(0, f_cov.n, CHUNK)]
    pca = ExactPCA(pve).fit(_upper(f_cov.read(i0, i1)) for i0, i1 in rng_)
    covet_train_pca = np.empty((f_cov.n, pca.n_components_), np.float32)
    for i0, i1 in rng_:
        covet_train_pca[i0:i1] = pca.transform(_upper(f_cov.read(i0, i1)))
    print(f"covariance PCA (exact, float64 covariance): {pca.n_components_} components, cumulative explained variance "
          f"{pca.cum_at_cut[0]:.6f} -> {pca.cum_at_cut[1]:.6f} around pve {pve} ({time.time() - T0:.0f} s)", flush=True)
    mem("reference covariance PCA")
    covet_test_pca = np.empty((len(lat_sc), pca.n_components_), np.float32)
    with torch.no_grad():
        for p in np.array_split(np.arange(len(lat_sc)), 16):
            if not len(p):
                continue
            cs = envi_model.covet_decode(torch.from_numpy(lat_sc[p[0]:p[-1] + 1]).to(envi_model.device).float()).detach().cpu().numpy()
            cov = np.matmul(cs, np.transpose(cs, (0, 2, 1)))
            del cs
            covet_test_pca[p[0]:p[-1] + 1] = pca.transform(_upper(cov))
            del cov
    np.save(st_cov_path, covet_train_pca); np.save(sc_cov_path, covet_test_pca)
    print(f"covariance PCA: {covet_train_pca.shape[1]} components ({time.time() - T0:.0f} s)", flush=True)
    WORK["pca"] = pca
    del envi_model, lat_sc
    gc.collect(); torch.cuda.empty_cache()
    if not KEEP_FILES:
        WORK["cov"].close(remove=True); WORK["sqrt"].close(remove=True)
    mem("query covariance")
    if st_data.X is None:
        st_data.X = load_st_X(None)
    st_data.var = var0
    for kk in list(st_data.layers.keys()):
        del st_data.layers[kk]
    st_data.obsm["covet"] = covet_train_pca; sc_data.obsm["covet"] = covet_test_pca
    mem("covet_init done (counts re-read)")
    return st_data, sc_data
def _zscore_inplace(adata, step):
    X = adata.X; n = np.intp(X.shape[0])
    genemean = np.mean(X, axis=0)
    arrmean = np.add.reduce(X, axis=0, keepdims=True)
    arrmean = np.true_divide(arrmean, n, out=arrmean, casting="unsafe", subok=False)
    acc = None
    for i in range(0, X.shape[0], step):
        x = X[i:i + step] - arrmean
        x = np.multiply(x, x, out=x)
        acc = np.add.reduce(x, axis=0) if acc is None else np.add.reduce(np.concatenate([acc[None], x]), axis=0)
        del x
    ret = np.true_divide(acc, np.maximum(n - 0, 0), out=acc, casting="unsafe", subok=False)
    genesd = np.sqrt(ret, out=ret)
    for i in range(0, X.shape[0], step):
        x = (X[i:i + step] - genemean) / genesd
        x[np.where(x > 6)] = 6
        X[i:i + step] = x
    adata.genemean, adata.genesd, adata.standardize = genemean, genesd, True
def preprocess_inplace(st_data, sc_data, step=200000):
    common_genes = np.intersect1d(st_data.var.index, sc_data.var.index)
    out = []
    for a_ in (st_data, sc_data):
        cols = a_.var.index.get_indexer(common_genes)
        if len(cols) != a_.n_vars:
            raise NotImplementedError("preprocess_inplace: query and reference gene sets differ")
        X = a_.X
        if not np.array_equal(cols, np.arange(a_.n_vars)):
            for i in range(0, X.shape[0], step):
                X[i:i + step] = X[i:i + step][:, cols]
        b = ad.AnnData(X=X, obs=a_.obs, obsm={kk: v for kk, v in a_.obsm.items()}, var=pd.DataFrame(index=common_genes))
        a_.X = None
        out.append(b)
    for b in out:
        b.var_names = [i.upper() for i in list(b.var_names)]
        b.var["genename"] = b.var.index.astype("str")
        b.var_names_make_unique()
        X = b.X
        counts = np.concatenate([np.sum(X[i:i + step], axis=1) for i in range(0, X.shape[0], step)])
        if (counts < 1).any():
            raise NotImplementedError("preprocess_inplace: cells with < 1 count would be dropped by normalize_per_cell")
        b.obs["n_counts"] = counts
        counts += counts == 0
        counts /= 10000
        for i in range(0, X.shape[0], step):
            X[i:i + step] /= counts[i:i + step, np.newaxis]
            np.log1p(X[i:i + step], out=X[i:i + step])
        b.uns["log1p"] = {"base": None}
    for b in out:
        _zscore_inplace(b, step)
    return out[0], out[1]
import remap.datasetgenemap as _dgm
class _WrapLocationNoCopy(_dgm.wrap_gene_location):
    def __init__(self, datainput, label, covetseq):
        self.data_tensor = torch.from_numpy(np.asarray(datainput, dtype=np.float32))
        try:
            cord = label.to_numpy().astype('float32')
        except:
            cord = label.astype('float32')
        self.coord_dim = cord.shape[1]
        self.covetseq = torch.from_numpy(covetseq.astype('float32'))
        self.num_spots = self.data_tensor.shape[0]
        cmin = cord.min(axis=0)
        cmax = cord.max(axis=0)
        cord_norm = (cord - cmin) / (cmax - cmin)
        self.coord_tensor = torch.from_numpy(cord_norm.astype('float32'))
class _WrapCovetNoCopy(_dgm.wrap_gene_covet):
    def __init__(self, datainput, label, covetseq, trainloc=None):
        self.data_tensor = torch.from_numpy(np.asarray(datainput, dtype=np.float32))
        try:
            cord = label.to_numpy().astype('float32')
        except:
            cord = label.astype('float32')
        if trainloc is not None:
            try:
                trainloc = trainloc.to_numpy().astype('float32')
            except:
                trainloc = trainloc.astype('float32')
        self.coord_dim = cord.shape[1]
        self.covet_tensor = torch.from_numpy(covetseq.astype("float32"))
        self.num_spots = datainput.shape[0]
        means = cord.mean(axis=0) if trainloc is None else trainloc.mean(axis=0)
        stds = cord.std(axis=0) * 0.25 if trainloc is None else trainloc.std(axis=0) * 0.25
        self.coords_norm = torch.from_numpy(((cord - means) / stds).astype("float32"))
def fit_cord_single_lowmem(Rdata, location_data, Qdata, path_name, hidden_dims=[400, 200, 100], n_iter=3, batch_size=256,
                           harmony=False):
    if harmony:
        raise NotImplementedError("--lowmem_all 1 implements the harmony=False path only")
    ff = sys.modules["remap.fit_functions"]
    orig = (ff.wrap_gene_location, ff.wrap_gene_covet)
    ff.wrap_gene_location, ff.wrap_gene_covet = _WrapLocationNoCopy, _WrapCovetNoCopy
    try:
        return _fit_cord_single_body(Rdata, location_data, Qdata, path_name, hidden_dims, n_iter, batch_size)
    finally:
        ff.wrap_gene_location, ff.wrap_gene_covet = orig
def _fit_cord_single_body(Rdata, location_data, Qdata, path_name, hidden_dims, n_iter, batch_size):
    cuda = 'GPU' if torch.cuda.is_available() else 'CPU'
    print(f"Model is on {cuda}")
    norm = True
    Rdata = use_corrected_expression(Rdata)
    Qdata = use_corrected_expression(Qdata)
    covet_train_pca = Rdata.obsm['covet']
    covet_test_pca = Qdata.obsm['covet']
    if norm:
        Rdata, Qdata = preprocess_inplace(Rdata, Qdata)
    location_data = np.array(location_data)
    mem("preprocess (in place)")
    for i in range(n_iter):
        file_name_loc = f"location_iter{i}"
        model_train = Fit_cord(data_train=Rdata, location_data=location_data, covet_train=covet_train_pca, hidden_dims=hidden_dims,
                               num_epochs_max=500, number_error_try=15, batch_size=batch_size, path=path_name, filename=file_name_loc)
        mem(f"Fit_cord {i}")
        pred_loc_test = Predict_cord(data_test=Qdata, covet_test=covet_test_pca, out_dim=location_data.shape[1], path=path_name,
                                     filename=file_name_loc)
        pred_loc_train = Predict_cord(data_test=Rdata, covet_test=covet_train_pca, out_dim=location_data.shape[1], path=path_name,
                                      filename=file_name_loc)
        mem(f"Predict_cord {i} (query + reference)")
        pred_loc_test_transform = pred_transform(pred_cord=pred_loc_test, train_cord=np.array(location_data))
        pred_loc_train_transform = pred_transform(pred_cord=pred_loc_train, train_cord=np.array(location_data))
        if i != n_iter - 1:
            file_name_covet = f"covariance_iter{i}"
            model_train = Fit_covet(data_train=Rdata, location_data=pred_loc_train_transform, covet_train=covet_train_pca,
                                    hidden_dims=hidden_dims, num_epochs_max=500, number_error_try=15, batch_size=batch_size,
                                    path=path_name, filename=file_name_covet)
            mem(f"Fit_covet {i}")
            pred_covet_test = Predict_covet(data_test=Qdata, location_test=pred_loc_test_transform, out_dim=covet_train_pca.shape[1],
                                            location_train=pred_loc_train_transform, path=path_name, filename=file_name_covet)
            pred_covet_train = Predict_covet(data_test=Rdata, location_test=pred_loc_train_transform, out_dim=covet_train_pca.shape[1],
                                             path=path_name, filename=file_name_covet)
            mem(f"Predict_covet {i} (query + reference)")
            covet_test_pca = pred_covet_test
            covet_train_pca = pred_covet_train
    np.save(f"{path_name}/remap_loc.npy", pred_loc_test_transform)
    return pred_loc_test_transform
