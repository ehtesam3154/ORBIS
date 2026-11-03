"""novoSpaRc-3D, atlas-assisted (settings of the authors' Drosophila tutorial): target = seeded uniform subsample of training
cells with their 3-D positions, query in seeded chunks; position = the transport-plan-weighted mean of the target positions.
--part i/n runs one share of the chunks; --merge N joins the parts into final_coords.npz (pred in mm, query row order).
"""
import argparse, json, os, resource, sys, time
import numpy as np, pandas as pd, anndata as ad
import novosparc
ap = argparse.ArgumentParser()
ap.add_argument("--ref_dirs", required=True); ap.add_argument("--query_dir", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n_ref", type=int, required=True); ap.add_argument("--chunk", type=int, required=True)
ap.add_argument("--corr_dirs", default=""); ap.add_argument("--corr_query", default=""); ap.add_argument("--corr_query_map", default="")
ap.add_argument("--corr_pca", default=""); ap.add_argument("--genes_meta", default=""); ap.add_argument("--genes_key", default="measured")
ap.add_argument("--iso_mm", type=float, default=0.0); ap.add_argument("--max_chunks", type=int, default=0)
ap.add_argument("--part", default="0/1"); ap.add_argument("--merge", type=int, default=0)
a = ap.parse_args(); t0 = time.time(); os.makedirs(a.out, exist_ok=True)
say = lambda m: print(f"[{time.time() - t0:.0f} s] {m}", flush=True)
nq = np.load(f"{a.query_dir}/feat.npy", mmap_mode="r").shape[0]
if a.merge:
    pred = np.full((nq, 3), np.nan); arg = np.full(nq, -1, np.int64); pmax = np.full(nq, np.nan, np.float32)
    for i in range(a.merge):
        z = np.load(f"{a.out}/part_{i}.npz"); r = z["rows"]; pred[r] = z["pred"]; arg[r] = z["argmax_ref"]; pmax[r] = z["argmax_prob"]
        keep = {k: z[k] for k in z.files if k.startswith("ref_rows_") or k in ("ref_names", "n_ref", "chunk", "iso_mm")}
    assert np.isfinite(pred).all() and (arg >= 0).all(), "a part is missing cells"
    np.savez(f"{a.out}/final_coords.npz", pred=pred, rows=np.arange(nq), argmax_ref=arg, argmax_prob=pmax, units="mm (reference stores' frame)", **keep)
    say(f"merged {a.merge} parts -> {a.out}/final_coords.npz"); sys.exit(0)
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
def lognorm(X):
    X = np.asarray(X, np.float64); s = X.sum(1, keepdims=True); s[s == 0] = 1.0
    return np.log1p(X / s * 1e4)
sizes = [np.load(f"{d}/pos.npy", mmap_mode="r").shape[0] for d in REFS]; off = np.cumsum([0] + sizes)
pick = np.sort(np.random.default_rng(0).choice(off[-1], a.n_ref, replace=False))
rrows = [pick[(pick >= off[k]) & (pick < off[k + 1])] - off[k] for k in range(len(REFS))]
CD = a.corr_dirs.split(",") if a.corr_dirs else None
ATLAS = lognorm(np.concatenate([recon(rows_seq(f"{CD[k]}/feat.npy", r)) if CD else counts(f"{d}/feat.npy", r) for k, (d, r) in enumerate(zip(REFS, rrows))]))
LOC = np.concatenate([np.load(f"{d}/pos.npy", mmap_mode="r")[r].astype(np.float64) * ISO for d, r in zip(REFS, rrows)])
say(f"target {LOC.shape} / atlas {ATLAS.shape} from {off[-1]:,} training cells ({', '.join(f'{os.path.basename(d)} {len(r):,}' for d, r in zip(REFS, rrows))})")
perm = np.random.default_rng(1).permutation(nq); chunks = np.array_split(perm, max(1, int(np.ceil(nq / a.chunk))))
pi, pn = (int(x) for x in a.part.split("/")); mine = list(range(pi, len(chunks), pn))
if a.max_chunks:
    mine = mine[:a.max_chunks]
if a.corr_query:
    mq = np.load(a.corr_query_map); inv = np.empty(len(mq), np.int64); inv[mq] = np.arange(len(mq)); fq = np.load(a.corr_query, mmap_mode="r")
say(f"query {nq:,} cells in {len(chunks)} chunks of ~{a.chunk}; this worker (part {a.part}) runs {len(mine)}")
myrows = np.sort(np.concatenate([chunks[ci] for ci in mine])) if mine else np.empty(0, np.int64)
if a.corr_query:
    QX = recon(np.asarray(fq[np.sort(inv[myrows])])[np.argsort(np.argsort(inv[myrows]))])
else:
    QX = counts(f"{a.query_dir}/feat.npy", myrows)
qpos = np.full(nq, -1, np.int64); qpos[myrows] = np.arange(len(myrows))
say(f"this worker's query rows read once: {QX.shape}")
R, PR, AR, PM = [], [], [], []
for k, ci in enumerate(mine):
    r = np.sort(chunks[ci]); tc = time.time()
    XQ = lognorm(QX[qpos[r]])
    D = ad.AnnData(X=XQ, obs=pd.DataFrame(index=[str(i) for i in r]), var=pd.DataFrame(index=genes))
    tissue = novosparc.cm.Tissue(dataset=D, locations=LOC)
    tissue.setup_reconstruction(markers_to_use=np.arange(len(genes)), atlas_matrix=ATLAS, num_neighbors_s=5, num_neighbors_t=5, verbose=False)
    tissue.reconstruct(alpha_linear=0.8, epsilon=5e-3, verbose=False)
    gw = np.asarray(tissue.gw, np.float64); rs = gw.sum(1, keepdims=True); rs = np.where(rs < 1e-12, 1.0, rs); gwn = gw / rs
    R.append(r); PR.append(gwn @ LOC); j = gwn.argmax(1); AR.append(j); PM.append(gwn[np.arange(len(r)), j].astype(np.float32))
    say(f"chunk {k + 1}/{len(mine)} (#{ci}): {len(r)} cells x {len(LOC)} targets x {len(genes)} genes, {time.time() - tc:.1f} s, "
        f"plan mass {gw.sum():.4f}, finite {np.isfinite(gw).all()}, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB")
    del tissue, gw, gwn, D, XQ
np.savez(f"{a.out}/part_{pi}.npz", rows=np.concatenate(R), pred=np.concatenate(PR), argmax_ref=np.concatenate(AR), argmax_prob=np.concatenate(PM),
         ref_names=np.array(REFS), n_ref=a.n_ref, chunk=a.chunk, iso_mm=ISO, **{f"ref_rows_{k}": rr for k, rr in enumerate(rrows)})
say(f"DONE part {a.part}: {sum(len(r) for r in R):,} cells -> {a.out}/part_{pi}.npz")
