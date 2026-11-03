#!/usr/bin/env python
"""Symphony reference on the five spatial training brains (477 shared genes, all cells).

stage-a   reference mean / SD of the log-normalised expression over all cells, PCA of the uncentred Gram matrix of the clipped z-scores
          (200 loadings, signs aligned to a sign-reference PCA), projection of every cell -> raw PCs per brain
harmony   Harmony integration of the raw PCs with batch = platform through symphonypy.pp.harmony_integrate, then the corrected
          embedding per brain, the soft assignments, the Harmony state and the light reference AnnData for symphonypy.tl.map_embedding

Transform per cell: c = 2^f - 1 -> log1p(c / max(sum_477(c), 1) * 1e4) -> z = clip((x - mean) / sd, -10, 10) -> PC = z @ V."""
import argparse
import datetime
import json
import logging
import multiprocessing as mp
import os
import platform as _platform
import sys
import threading
import time

import numpy as np

import symphony_common as C

CH = 50000
SRC = None
MU = SD = COMP = None
OUT = None


def log(*a):
    print(datetime.datetime.now().strftime("%H:%M:%S"), *a, flush=True)


def _chunk(t):
    b, lo, hi = t
    _, path, cols = SRC[b]
    return C.lognorm(C.read_counts(path, cols, lo, hi)) + (None,)


def w1(t):
    b, lo, hi = t
    _, path, cols = SRC[b]
    c = C.read_counts(path, cols, lo, hi)
    x, s = C.lognorm(c)
    d = None
    if lo == 0:
        dev = np.abs(c - np.rint(c))
        d = dict(max_abs_dev_from_integer=float(dev.max()), frac_entries_dev_gt_1e3=float((dev > 1e-3).mean()),
                 frac_entries_dev_gt_1e2=float((dev > 1e-2).mean()), max_count=float(c.max()), frac_zero=float((c == 0).mean()),
                 median_library_size=float(np.median(s)), cells_examined=int(hi - lo))
    return hi - lo, x.sum(0), (x * x).sum(0), int((s == 0).sum()), int(((s > 0) & (s < 1)).sum()), d


def w2(t):
    x, _ = _chunk(t)[:2]
    zr = (x - MU) / SD
    z = np.clip(zr, -C.CLIP, C.CLIP)
    return z.sum(0), z.T @ z, int((np.abs(zr) > C.CLIP).sum())


def w3(t):
    b, lo, hi = t
    x, _ = _chunk(t)[:2]
    z = np.clip((x - MU) / SD, -C.CLIP, C.CLIP)
    p = (z @ COMP).astype(np.float32)
    mm = np.load(f"{OUT}/raw/{SRC[b][0]}.npy", mmap_mode="r+")
    mm[lo:hi] = p
    mm.flush()
    return hi - lo


def run_pass(fn, tasks, workers, name):
    out = []
    t0 = time.time()
    with mp.get_context("fork").Pool(workers) as pool:
        for i, r in enumerate(pool.imap(fn, tasks, chunksize=1)):
            out.append(r)
            if (i + 1) % max(1, len(tasks) // 10) == 0:
                log(f"  {name}: {i + 1}/{len(tasks)} chunks ({time.time() - t0:.0f} s)")
    return out


def cmd_stage_a(a):
    """Reference statistics, PCA and raw PCs of every cell."""
    global SRC, MU, SD, COMP, OUT
    C.configure(a.stores_dir, a.allen_dir, a.panel, a.sign_pca)
    OUT = a.out
    os.makedirs(f"{OUT}/raw", exist_ok=True)
    SRC, gn = C.sources()
    sizes = {}
    for name, path, cols in SRC:
        f = np.load(path, mmap_mode="r")
        assert f.dtype == np.float32 and f.ndim == 2 and f.shape[1] == (1122 if cols is not None else C.NGENES), (name, f.shape, f.dtype)
        assert f.shape[0] == C.EXPECTED_CELLS[name], (name, f.shape)
        sizes[name] = f.shape[0]
        log(f"input {name}: {f.shape} {f.dtype}  {path}")
    N = sum(sizes.values())
    tasks = [(b, lo, min(lo + CH, sizes[nm])) for b, (nm, _, _) in enumerate(SRC) for lo in range(0, sizes[nm], CH)]
    log(f"cells per brain {sizes}  total {N:,}  genes {len(gn)}  chunk {CH}  tasks {len(tasks)}  workers {a.workers}")
    t0 = time.time()
    r1 = run_pass(w1, tasks, a.workers, "pass1")
    n = sum(r[0] for r in r1)
    assert n == N
    s1 = np.zeros(C.NGENES); s2 = np.zeros(C.NGENES)
    for r in r1:
        s1 += r[1]; s2 += r[2]
    MU = s1 / n
    SD = np.sqrt(np.maximum(s2 / n - MU * MU, 0)) + 1e-8
    floor_zero = sum(r[3] for r in r1); floor_small = sum(r[4] for r in r1)
    intstat = {SRC[t[0]][0]: r[5] for t, r in zip(tasks, r1) if r[5] is not None}
    log(f"pass 1 done: {n:,} cells ({time.time() - t0:.0f} s); library size 0 (floor max(sum,1) hit, all-zero cells): {floor_zero:,}; 0<sum<1: {floor_small:,}")
    r2 = run_pass(w2, tasks, a.workers, "pass2")
    z1 = np.zeros(C.NGENES); G = np.zeros((C.NGENES, C.NGENES)); nclip = 0
    for r in r2:
        z1 += r[0]; G += r[1]; nclip += r[2]
    log(f"pass 2 done ({time.time() - t0:.0f} s); entries clipped at +-{C.CLIP}: {nclip:,} of {N * C.NGENES:,} ({nclip / (N * C.NGENES):.2e})")
    zm = z1 / n
    cov = (G - n * np.outer(zm, zm)) / (n - 1)
    w, V = np.linalg.eigh(G)
    o = np.argsort(w)[::-1]; w, V = w[o], V[:, o]
    comp = V[:, :C.NPC].copy()
    old = np.load(C.SIGN_PCA, allow_pickle=True)
    assert list(old["genes"]) == gn
    cos_old = np.einsum("ij,ij->j", comp, old["components"])
    flip = cos_old < 0
    comp[:, flip] *= -1
    cos_old = np.abs(cos_old)
    evr_unc = w / w.sum()
    var_c = np.einsum("ij,ij->j", comp, cov @ comp)
    evr_c = var_c / np.trace(cov)
    wc, Vc = np.linalg.eigh(cov); oc = np.argsort(wc)[::-1]; wc, Vc = wc[oc], Vc[:, oc]
    cos_cent = np.abs(np.einsum("ij,ij->j", comp, Vc[:, :C.NPC]))
    COMP = comp
    np.savez(f"{OUT}/pca.npz", genes=np.array(gn), mean=MU, sd=SD, clip=C.CLIP, lib=C.LIB, components=comp, eigenvalues_uncentred=w,
             explained_variance_ratio_uncentred=evr_unc, explained_variance_ratio_centred_of_these_pcs=evr_c,
             postclip_mean_NOT_APPLIED=zm, sign_flipped_to_match_old=flip, abs_cos_to_old_components=cos_old,
             n_cells=n, brains=np.array(list(sizes)), brain_sizes=np.array(list(sizes.values())))
    diag = dict(cells=sizes, N=N, genes=len(gn), mean_max_abs_diff_vs_old=float(np.abs(MU - old["mean"]).max()),
                sd_max_abs_diff_vs_old=float(np.abs(SD - old["sd"]).max()),
                n_sign_flipped_to_match_old=int(flip.sum()), abs_cos_new_vs_old_components_min=float(cos_old.min()),
                abs_cos_new_vs_old_components_median=float(np.median(cos_old)),
                abs_cos_uncentred_vs_centred_new_min=float(cos_cent.min()),
                explained_uncentred_200=float(evr_unc[:C.NPC].sum()), explained_centred_of_same_pcs_200=float(evr_c.sum()),
                old_explained_centred_200=float(old["explained_variance_ratio"][:C.NPC].sum()),
                evr_uncentred_first5=evr_unc[:5].tolist(), postclip_mean_norm=float(np.linalg.norm(zm)),
                postclip_mean_projected_norm=float(np.linalg.norm(zm @ comp)),
                floor_hits_all_zero_cells=floor_zero, floor_hits_0_lt_sum_lt_1=floor_small, clipped_entries=nclip,
                stored_value_checks_first_chunk_per_brain=intstat, feature_meaning="float32 log2(1 + raw count); counts = 2^f - 1 (float64, not rounded)")
    log("PCA:", json.dumps({k: v for k, v in diag.items() if not isinstance(v, dict)}))
    for nm, k in sizes.items():
        mm = np.lib.format.open_memmap(f"{OUT}/raw/{nm}.npy", mode="w+", dtype=np.float32, shape=(k, C.NPC)); del mm
    run_pass(w3, tasks, a.workers, "pass3")
    diag["seconds_total"] = round(time.time() - t0)
    diag["peak_rss_main_gb"] = C.rss_gb()[1]
    json.dump(diag, open(f"{OUT}/stage_a.json", "w"), indent=1)
    log("STAGE_A_DONE", f"{time.time() - t0:.0f} s")


class RssLog(threading.Thread):
    def __init__(self, path, every=10):
        super().__init__(daemon=True); self.path, self.every, self.phase, self.stop = path, every, "init", False

    def run(self):
        with open(self.path, "a") as fh:
            while not self.stop:
                cur, hwm = C.rss_gb()
                fh.write(f"{datetime.datetime.now().strftime('%H:%M:%S')} {self.phase} rss_gb={cur:.2f} peak_gb={hwm:.2f}\n"); fh.flush()
                time.sleep(self.every)


def cmd_harmony(a):
    """Harmony integration of the raw PCs and the saved reference."""
    import anndata as ad
    import pandas as pd
    import symphonypy as sp
    import symphonypy._utils as su
    from importlib.metadata import version
    out = a.out or a.ref
    os.makedirs(out, exist_ok=True)
    rl = RssLog(f"{out}/rss_trace.txt"); rl.start()
    t_all = time.time()
    pc = np.load(f"{a.ref}/pca.npz", allow_pickle=True)
    genes, mu, sd, comp = [str(g) for g in pc["genes"]], pc["mean"], pc["sd"], pc["components"]
    meta = {}
    for b in C.BRAINS:
        mm = np.load(f"{a.ref}/raw/{b}.npy", mmap_mode="r"); meta[b] = (mm.offset, mm.shape[0]); del mm
    sizes = [meta[b][1] for b in C.BRAINS]
    assert dict(zip(C.BRAINS, sizes)) == C.EXPECTED_CELLS, sizes
    offs = np.cumsum([0] + sizes); Nfull = int(offs[-1])
    rows = None
    if a.n_sub:
        rows = np.sort(np.random.default_rng(a.sub_seed).choice(Nfull, a.n_sub, replace=False))
        np.save(f"{out}/rows.npy", rows)
    n = len(rows) if rows is not None else Nfull
    rl.phase = "load"
    X = np.empty((n, C.NPC), np.float32)
    brain_of = np.empty(n, np.int8)
    p = 0
    per_brain = []
    for j, b in enumerate(C.BRAINS):
        off, nrow = meta[b]
        r = (rows[(rows >= offs[j]) & (rows < offs[j + 1])] - offs[j]) if rows is not None else None
        nb = len(r) if r is not None else nrow
        q = p
        with open(f"{a.ref}/raw/{b}.npy", "rb") as fh:
            for i in range(0, nrow, 500000):
                m = min(500000, nrow - i)
                fh.seek(off + i * C.NPC * 4)
                blk = np.fromfile(fh, dtype=np.float32, count=m * C.NPC).reshape(m, C.NPC)
                if r is None:
                    X[q:q + m] = blk; q += m
                else:
                    sel = r[(r >= i) & (r < i + m)] - i
                    X[q:q + len(sel)] = blk[sel]; q += len(sel)
        assert q - p == nb
        brain_of[p:p + nb] = j; per_brain.append(nb); p += nb
    assert p == n
    cum = np.cumsum([0] + per_brain)
    log(f"loaded raw PCs {X.shape} {X.dtype}; cells per brain {dict(zip(C.BRAINS, per_brain))}; rss/peak GB {C.rss_gb()}")
    plat = np.where(brain_of == 4, "allen", "zhuang")
    obs = pd.DataFrame({"platform": pd.Categorical(plat), "brain": pd.Categorical(np.array(C.BRAINS)[brain_of], categories=list(C.BRAINS))},
                       index=pd.Index(np.arange(n).astype(str)))
    var = pd.DataFrame({"mean": mu, "std": sd, "highly_variable": np.ones(C.NGENES, bool)}, index=pd.Index(genes))
    nb_ = {str(k): int(v) for k, v in obs["platform"].value_counts().items()}
    try:
        import threadpoolctl
        tp = threadpoolctl.threadpool_info()
    except Exception:
        tp = []
    log(f"RUN Harmony [{a.path} path]: N={n:,} d={C.NPC} batch platform {nb_}  seed {a.seed}  affinity CPUs {len(os.sched_getaffinity(0))}  "
        f"OMP/OPENBLAS threads env {os.environ.get('OMP_NUM_THREADS')}/{os.environ.get('OPENBLAS_NUM_THREADS')}  blas {[(x.get('internal_api'), x.get('num_threads')) for x in tp]}")
    rl.phase = "harmony"
    t0 = time.time()
    adata = ad.AnnData(obs=obs, var=var)
    adata.obsm["X_pca"] = X
    adata.varm["PCs"] = comp
    adata.uns["log1p"] = {"base": None}
    del X
    cap = {}
    orig = su.run_harmony

    def run_harmony_logged(*args, **kw):
        cap["kwargs_passed_by_symphonypy"] = {k: (v if np.isscalar(v) or isinstance(v, str) else type(v).__name__) for k, v in kw.items()}
        t = time.time()
        ho = orig(*args, **kw)
        cap["harmony_seconds"] = round(time.time() - t)
        cap["Y_harmonypy"] = np.array(ho.Y)
        cap["objective_harmony"] = [float(v) for v in ho.objective_harmony]
        cap["objective_kmeans"] = [float(v) for v in ho.objective_kmeans]
        cap["kmeans_rounds"] = [int(v) for v in ho.kmeans_rounds]
        cap["K"] = int(ho.K)
        return ho

    su.run_harmony = run_harmony_logged
    sp.pp.harmony_integrate(adata, key="platform", flavor="python", ref_basis_source="X_pca", ref_basis_adjusted="X_pca_harmony",
                            ref_basis_loadings="PCs", verbose=True, random_seed=a.seed)
    su.run_harmony = orig
    h = adata.uns["harmony"]
    Zc, R = adata.obsm["X_pca_harmony"], h["R"]
    t_h = time.time() - t0
    rl.phase = "save"
    cur, hwm = C.rss_gb()
    obj = cap["objective_harmony"]
    eff = dict(path=a.path, N=n, d=C.NPC, K_effective=int(h["K"]), K_requested="min(round(N/30),100)=%d" % min(round(n / 30.0), 100),
               sigma_effective_unique=sorted(set(np.round(np.asarray(h["sigma"], float), 6).tolist())), theta="harmonypy default 2 per batch level (not overridden)",
               lamb="harmonypy default None = dynamic estimation, alpha 0.2", block_size=0.05, max_iter_harmony=10, max_iter_kmeans=4,
               epsilon_cluster=1e-3, epsilon_harmony=1e-2, batch_prop_cutoff=1e-5, random_state=a.seed,
               iterations_run=len(obj) - 1, objective_harmony=obj, kmeans_rounds=cap["kmeans_rounds"], converged_symphonypy_flag=bool(h["converged"]),
               batch_levels_np_unique_order=["allen", "zhuang"], batch_counts=nb_, harmony_seconds=cap["harmony_seconds"], fit_plus_postprocessing_seconds=round(t_h),
               rss_gb_after_fit_and_postprocessing=cur, peak_rss_gb_so_far=hwm, bytes_per_cell_peak=hwm * 1e9 / n,
               threadpool=[(x.get("internal_api"), x.get("num_threads"), os.path.basename(x.get("filepath", ""))) for x in tp],
               affinity_cpus=len(os.sched_getaffinity(0)))
    log("EFFECTIVE", json.dumps(eff))
    log(f"corrected {Zc.shape} {Zc.dtype} order={'F' if Zc.flags.f_contiguous else 'C'}; R {R.shape} {R.dtype}")
    for j, b in enumerate(C.BRAINS):
        s, e = int(cum[j]), int(cum[j + 1])
        d = f"{out}/allen" if b == "allen" else f"{out}/stores/{b}"
        os.makedirs(d, exist_ok=True); os.makedirs(f"{out}/harmony_R", exist_ok=True)
        np.save(f"{d}/feat.npy", np.ascontiguousarray(Zc[s:e], dtype=np.float32))
        np.save(f"{out}/harmony_R/{b}.npy", np.ascontiguousarray(R[:, s:e].T, dtype=np.float32))
        log(f"saved {b}: {e - s:,} rows -> {d}/feat.npy and harmony_R/{b}.npy")
    np.savez(f"{out}/harmony_state.npz", Nr=h["Nr"], C=h["C"], K=h["K"], sigma=h["sigma"], Y_harmonypy_dxK=cap["Y_harmonypy"],
             objective_harmony=np.array(obj), objective_kmeans=np.array(cap["objective_kmeans"]), kmeans_rounds=np.array(cap["kmeans_rounds"]),
             batch_levels=np.array(["allen", "zhuang"]), N_b=np.array([nb_["allen"], nb_["zhuang"]]), Pr_b=np.array([nb_["allen"], nb_["zhuang"]]) / n,
             theta=np.array([2.0, 2.0]), seed=a.seed)
    ref = ad.AnnData(X=np.zeros((0, C.NGENES), np.float32), var=var.copy())
    ref.varm["PCs"] = comp
    ref.uns["harmony"] = dict(Nr=np.asarray(h["Nr"]), C=np.asarray(h["C"]), K=int(h["K"]), sigma=np.asarray(h["sigma"], np.float32),
                              ref_basis_loadings=str(h["ref_basis_loadings"]), ref_basis_adjusted=str(h["ref_basis_adjusted"]), vars_use=str(h["vars_use"]),
                              harmony_kwargs={k: (int(v) if isinstance(v, (int, np.integer)) else v) for k, v in h["harmony_kwargs"].items()},
                              converged=bool(h["converged"]))
    ref.uns["a10x_provenance"] = dict(cells=int(n), brains=list(C.BRAINS), cells_per_brain=[int(x) for x in per_brain],
                                      transform="2^f-1 -> /max(sum477,1)*1e4 -> log1p -> (x-mean)/std clip 10 -> @PCs (no post-clip centring)",
                                      note="light reference for symphonypy.tl.map_embedding (use_genes_column=None or 'highly_variable'); R and per-cell embeddings are stored beside it")
    ref.write_h5ad(f"{out}/symphony_ref.h5ad")
    man = dict(built=datetime.datetime.now().isoformat(timespec="seconds"), host=_platform.node(), command=" ".join(sys.argv),
               python=sys.version.split()[0], versions={k: version(k) for k in ("numpy", "scanpy", "anndata", "harmonypy", "symphonypy", "scipy", "pandas", "scikit-learn")},
               symphonypy_commit="cbda5adc531b49f5cfcfa5c0ef97dd4668209575", effective=eff, run_harmony_kwargs_seen=cap.get("kwargs_passed_by_symphonypy"),
               total_seconds=round(time.time() - t_all), peak_rss_gb=C.rss_gb()[1], rows_subset=bool(rows is not None), sub_seed=a.sub_seed if rows is not None else None,
               transform_pca=json.load(open(f"{a.ref}/stage_a.json")) if os.path.exists(f"{a.ref}/stage_a.json") else None)
    json.dump(man, open(f"{out}/reference_manifest.json", "w"), indent=1, default=str)
    rl.stop = True
    log("HARMONY_STAGE_DONE", f"total {time.time() - t_all:.0f} s, peak RSS {C.rss_gb()[1]:.1f} GB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage-a"); s.add_argument("--out", required=True); s.add_argument("--workers", type=int, default=12); s.set_defaults(f=cmd_stage_a)
    s.add_argument("--stores_dir", required=True, help="directory of the four Zhuang cell stores (abca1..abca4, 1,122 genes)")
    s.add_argument("--allen_dir", required=True, help="Allen MERFISH store with the 477 panel genes")
    s.add_argument("--panel", required=True, help="text file with the 477 panel gene ids")
    s.add_argument("--sign_pca", required=True, help="pca.npz whose components fix the sign of each new component")
    s = sub.add_parser("harmony"); s.add_argument("--ref", required=True); s.add_argument("--out", default=None); s.add_argument("--n_sub", type=int, default=0)
    s.add_argument("--sub_seed", type=int, default=25001); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--path", choices=("symphonypy",), default="symphonypy"); s.set_defaults(f=cmd_harmony)
    args = ap.parse_args()
    args.f(args)
