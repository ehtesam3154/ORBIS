"""Tangram-3D: map query cells onto a seeded uniform subsample of reference cells (mode 'cells', density prior 'uniform'),
query in fixed chunks; each query cell takes the 3-D position of its most probable reference cell. Writes final_coords.npz
(pred in mm, query row order, top-5 reference cells and probabilities) to --out.
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, anndata as ad, scanpy as sc, scipy.sparse as sp, torch
import tangram as tg
ap = argparse.ArgumentParser()
ap.add_argument("--ref_dirs", required=True); ap.add_argument("--query_dir", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n_ref", type=int, required=True); ap.add_argument("--chunk", type=int, required=True)
ap.add_argument("--corr_dirs", default=""); ap.add_argument("--corr_query", default=""); ap.add_argument("--corr_query_map", default="")
ap.add_argument("--corr_pca", default=""); ap.add_argument("--genes_meta", default=""); ap.add_argument("--genes_key", default="measured")
ap.add_argument("--iso_mm", type=float, default=0.0); ap.add_argument("--probe", type=int, default=0); ap.add_argument("--epochs", type=int, default=1000); ap.add_argument("--max_chunks", type=int, default=0)
a = ap.parse_args(); t0 = time.time(); os.makedirs(a.out, exist_ok=True)
say = lambda m: print(f"[{time.time() - t0:.0f} s] {m}", flush=True)
REFS = [d.rstrip("/") for d in a.ref_dirs.split(",")]; metas = [json.load(open(f"{d}/meta.json")) for d in REFS]
genes_all = list(metas[0]["gene_names"]); ISO = a.iso_mm if a.iso_mm else float(metas[0]["iso_mm"])
if a.genes_meta:
    v = json.load(open(a.genes_meta))[a.genes_key]; GC = np.flatnonzero(np.asarray(v, bool)) if not isinstance(v[0], str) else np.flatnonzero(np.isin(genes_all, v))
else:
    GC = np.arange(len(genes_all))
genes = [genes_all[i] for i in GC]
if a.corr_pca:
    P = np.load(a.corr_pca); assert [str(g) for g in P["genes"]] == genes
    comp, zm, sdv, mu = (P[k].astype(np.float64) for k in ("components", "zmean", "sd", "mean"))
    recon = lambda pcs: np.maximum(np.expm1((np.asarray(pcs, np.float64) @ comp.T + zm) * sdv + mu), 0.0).astype(np.float32)
def rows_seq(f, r, cols=None, B=200000):
    m = np.load(f, mmap_mode="r"); out = []
    for i in range(0, m.shape[0], B):
        rr = r[(r >= i) & (r < i + B)] - i
        if len(rr):
            blk = np.array(m[i:i + B]); out.append(blk[rr] if cols is None else blk[rr][:, cols])
    return np.concatenate(out) if out else np.empty((0, m.shape[1] if cols is None else len(cols)), m.dtype)
counts = lambda f, r: np.round(np.exp2(rows_seq(f, r, GC).astype(np.float32)) - 1.0)
sizes = [np.load(f"{d}/pos.npy", mmap_mode="r").shape[0] for d in REFS]; off = np.cumsum([0] + sizes)
pick = np.sort(np.random.default_rng(0).choice(off[-1], a.n_ref, replace=False))
rrows = [pick[(pick >= off[k]) & (pick < off[k + 1])] - off[k] for k in range(len(REFS))]
CD = a.corr_dirs.split(",") if a.corr_dirs else None
XR = np.concatenate([recon(rows_seq(f"{CD[k]}/feat.npy", r)) if CD else counts(f"{d}/feat.npy", r) for k, (d, r) in enumerate(zip(REFS, rrows))])
LR = np.concatenate([np.load(f"{d}/pos.npy", mmap_mode="r")[r].astype(np.float64) * ISO for d, r in zip(REFS, rrows)])
SP = ad.AnnData(XR); SP.var_names = genes; SP.obs_names = [f"r{i}" for i in range(len(XR))]
say(f"reference subsample {XR.shape} of {off[-1]:,} training cells ({', '.join(f'{os.path.basename(d)} {len(r):,}' for d, r in zip(REFS, rrows))})")
nq = np.load(f"{a.query_dir}/feat.npy", mmap_mode="r").shape[0]
perm = np.random.default_rng(1).permutation(nq); chunks = np.array_split(perm, max(1, int(np.ceil(nq / a.chunk))))
if a.corr_query:
    mq = np.load(a.corr_query_map); inv = np.empty(len(mq), np.int64); inv[mq] = np.arange(len(mq)); QALL = np.load(a.corr_query)[inv]
else:
    QALL = counts(f"{a.query_dir}/feat.npy", np.arange(nq))
say(f"query matrix in RAM {QALL.shape}")
pred = np.full((nq, 3), np.nan); arg = np.full(nq, -1, np.int64); pmax = np.full(nq, np.nan, np.float32)
top_i = np.full((nq, 5), -1, np.int64); top_p = np.full((nq, 5), np.nan, np.float32)
todo = chunks[:a.max_chunks] if a.max_chunks else chunks
say(f"query {nq:,} cells in {len(chunks)} chunks of ~{a.chunk}; running {len(todo)}")
for ci, rows in enumerate(todo):
    r = np.sort(rows)
    XQ = recon(QALL[r]) if a.corr_query else QALL[r]
    SC = ad.AnnData(sp.csr_matrix(XQ)); SC.var_names = genes; SC.obs_names = [f"q{i}" for i in r]
    sc.pp.normalize_total(SC)
    SPc = SP.copy(); tg.pp_adatas(SC, SPc, genes=None)
    torch.manual_seed(1000 + ci); np.random.seed(1000 + ci); torch.cuda.reset_peak_memory_stats()
    tc = time.time()
    M = tg.map_cells_to_space(SC, SPc, mode="cells", density_prior="uniform", num_epochs=a.probe or a.epochs, device="cuda:0", verbose=False)
    X = np.asarray(M.X); j = X.argmax(1); arg[r] = j; pmax[r] = X[np.arange(len(r)), j]
    t5 = np.argpartition(-X, 5, axis=1)[:, :5]; o = np.argsort(-np.take_along_axis(X, t5, 1), 1); top_i[r] = np.take_along_axis(t5, o, 1)
    top_p[r] = np.take_along_axis(X, top_i[r], 1); pred[r] = LR[j]
    say(f"chunk {ci + 1}/{len(todo)}: {len(r)} cells x {SPc.n_obs} ref x {len(SPc.uns['training_genes'])} genes, {time.time() - tc:.1f} s, "
        f"GPU peak alloc {torch.cuda.max_memory_allocated() / 1e9:.2f} GB, reserved {torch.cuda.max_memory_reserved() / 1e9:.2f} GB, "
        f"median max-prob {np.median(pmax[r]):.4f}")
    del M, X, SC, SPc
if a.probe:
    sys.exit(0)
assert np.isfinite(pred).all() and (arg >= 0).all()
np.savez(f"{a.out}/final_coords.npz", pred=pred, rows=np.arange(nq), units="mm (reference stores' frame)", ref_names=np.array(REFS),
         **{f"ref_rows_{k}": rr for k, rr in enumerate(rrows)}, argmax_ref=arg, argmax_prob=pmax, top5_ref=top_i, top5_prob=top_p,
         n_ref=a.n_ref, chunk=a.chunk, iso_mm=ISO, num_epochs=a.epochs)
say(f"DONE -> {a.out}/final_coords.npz")
