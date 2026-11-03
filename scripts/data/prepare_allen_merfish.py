"""Allen MERFISH whole mouse brain (MERFISH-C57BL6J-638850): cell store, test prebuilt, and Harmony-corrected features for the
Zhuang training mice and the Allen test brain.

Data: Allen Brain Cell (ABC) atlas, https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/ (release manifest
releases/20241115/manifest.json). Files needed in $DATA_DIR/raw/allen_merfish/ (renamed as listed):
  C57BL6J-638850-raw.h5ad                     <- expression_matrices/MERFISH-C57BL6J-638850/20230830/C57BL6J-638850-raw.h5ad
  cell_metadata_with_cluster_annotation.csv   <- metadata/MERFISH-C57BL6J-638850/20241115/views/cell_metadata_with_cluster_annotation.csv
  reconstructed_coordinates.csv               <- metadata/MERFISH-C57BL6J-638850-CCF/20231215/reconstructed_coordinates.csv
  ccf_coordinates.csv                         <- metadata/MERFISH-C57BL6J-638850-CCF/20231215/ccf_coordinates.csv
Needs the Zhuang mouse outputs of prepare_abc_mice.sh first (cell stores, the mouse-2 test prebuilt, the training prebuilt).
Command: DATA_DIR=... bash scripts/data/prepare_allen_merfish.sh   (outputs in $DATA_DIR/brains/)

Sub-commands:
  store    every Allen cell with section-stack coordinates, in the Zhuang stores' format: the 1,122-gene layout (the 477 genes
           Allen measures hold log2(1 + count), the others 0), stack coordinates mapped rigidly onto CCF [ML, DV, AP], then
           scaled per axis onto mouse 1's ruler, / 13.4577 mm
  test     the Allen test prebuilt: one graph per section, the 477 measured genes, positions = the published stack
           coordinates folded about the CCF midline plane, / 13.4577 mm
  harmony  477 shared genes of Zhuang mice 1-4 (every cell) and the Allen test cells: counts -> library size 1e4 -> log1p ->
           z-score (all cells) clipped at 10 -> PCA (K = 200) -> Harmony with batch = platform; writes the corrected stores,
           the Allen test folder and the training prebuilt with K input features"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

ISO = 13.45773983001709
MID_ML = 5.7


def slopes(d, raw_dir):
    """Per-axis OLS slope of the CCF coordinate (mm) on the store coordinate (mm), over cells with a CCF region."""
    m = json.load(open(f"{d}/meta.json")); lab = pd.read_csv(f"{d}/cell_labels.txt", header=None, dtype=str)[0]
    pos = np.load(f"{d}/pos.npy").astype(np.float64) * ISO
    ccf = pd.read_csv(f"{raw_dir}/{m['animal']}-CCF_ccf_coordinates.csv", usecols=["cell_label", "x", "y", "z", "parcellation_index"],
                      dtype={"cell_label": str}).set_index("cell_label")
    has = lab.isin(ccf.index).to_numpy(); sub = ccf.loc[lab[has]]; ok = sub["parcellation_index"].to_numpy() > 0
    C, P = sub[["z", "y", "x"]].to_numpy()[ok], pos[has][ok]
    return np.array([np.polyfit(P[:, k], C[:, k], 1)[0] for k in range(3)])


def cmd_store(a):
    """Allen cell store: feat, pos, sec, cls, hemi (side of the midline plane), cell_labels.txt, meta.json."""
    import anndata as ad
    from scipy import sparse
    t0 = time.time()
    ref = json.load(open(f"{a.ref_store}/meta.json")); ours = ref["gene_names"]
    s_ref = slopes(a.ref_store, a.zhuang_raw_dir)
    meta = pd.read_csv(f"{a.raw_dir}/cell_metadata_with_cluster_annotation.csv", usecols=["cell_label", "brain_section_label", "class"],
                       dtype={"cell_label": str})
    rec = pd.read_csv(f"{a.raw_dir}/reconstructed_coordinates.csv", dtype={"cell_label": str})
    rec.columns = ["cell_label", "mx", "my", "mz", "rec_pi"]
    ccf = pd.read_csv(f"{a.raw_dir}/ccf_coordinates.csv", dtype={"cell_label": str})
    ccf.columns = ["cell_label", "cx", "cy", "cz", "pi"]
    cells = meta.merge(rec, on="cell_label", how="inner").merge(ccf, on="cell_label", how="left")
    assert cells["cell_label"].is_unique
    M = cells[["mx", "my", "mz"]].to_numpy(np.float64)
    fit = cells["cx"].notna().to_numpy() & (cells["pi"].fillna(0).to_numpy() > 0)
    C = cells.loc[fit, ["cz", "cy", "cx"]].to_numpy(np.float64)
    mM, mC = M[fit].mean(0), C.mean(0)
    U, _, Vt = np.linalg.svd((M[fit] - mM).T @ (C - mC)); Rm = U @ Vt
    mapped = (M - mM) @ Rm + mC
    resid = np.sqrt(((mapped[fit] - C) ** 2).sum(1))
    fit_info = dict(n_fit=int(fit.sum()), det=float(np.linalg.det(Rm)), rotation=Rm.tolist(), mean_meta=mM.tolist(),
                    mean_ccf=mC.tolist(), residual_mm_median=float(np.median(resid)),
                    residual_mm_p90=float(np.percentile(resid, 90)),
                    per_axis_rms_mm=np.sqrt(((mapped[fit] - C) ** 2).mean(0)).tolist())
    P = mapped / ISO
    s = np.array([np.polyfit(P[fit][:, k] * ISO, C[:, k], 1)[0] for k in range(3)])
    D = s / s_ref; c0 = P.mean(0); P = (P - c0) * D + c0
    s_after = np.array([np.polyfit(P[fit][:, k] * ISO, C[:, k], 1)[0] for k in range(3)])
    assert np.all(np.abs(D - 1.0) < 0.5), f"implausible axis scale {D}"
    mid_iso = (MID_ML / ISO - c0[0]) * D[0] + c0[0]
    fit_info["axis_scale"] = dict(D=D.tolist(), centre_iso=c0.tolist(), slopes=s.tolist(), ref_slopes=s_ref.tolist(),
                                  ref_store=a.ref_store, slopes_after=s_after.tolist())
    print(f"[allen] axis scale D {np.round(D, 3).tolist()}", flush=True)
    hemi = (P[:, 0] > mid_iso).astype(np.int8)
    h = ad.read_h5ad(f"{a.raw_dir}/C57BL6J-638850-raw.h5ad", backed="r")
    row = pd.Index(h.obs_names.astype(str)).get_indexer(cells["cell_label"])
    keep = row >= 0
    cells, row, P, hemi = cells.loc[keep].reset_index(drop=True), row[keep], P[keep], hemi[keep]
    var_ids = pd.Index([str(v).split(".")[0] for v in (h.var["gene_identifier"] if "gene_identifier" in h.var else h.var_names)])
    src = var_ids.get_indexer(ours)
    measured = src >= 0
    print(f"[allen] {int(keep.sum()):,} cells in the raw h5ad; our genes measured: {measured.sum()} of {len(ours)}", flush=True)
    N, G = len(cells), len(ours)
    classes = json.load(open(a.class_list)); enc = {c_: i for i, c_ in enumerate(classes)}
    assert set(cells["class"].astype(str)) <= set(classes)
    secs = sorted(cells["brain_section_label"].unique()); sid = {s_: i for i, s_ in enumerate(secs)}
    os.makedirs(a.out_dir, exist_ok=True)
    feat = np.lib.format.open_memmap(f"{a.out_dir}/feat.npy", mode="w+", dtype=np.float32, shape=(N, G))
    np.save(f"{a.out_dir}/pos.npy", P.astype(np.float32))
    np.save(f"{a.out_dir}/sec.npy", cells["brain_section_label"].map(sid).to_numpy(np.int32))
    np.save(f"{a.out_dir}/cls.npy", cells["class"].astype(str).map(enc).to_numpy(np.int16))
    np.save(f"{a.out_dir}/hemi.npy", hemi)
    cols = src[measured]; dst = np.flatnonzero(measured)
    for s0 in range(0, N, a.rows_per_block):
        r = row[s0:s0 + a.rows_per_block]; o = np.argsort(r)
        X = h.X[r[o]]
        X = X.toarray() if sparse.issparse(X) else np.asarray(X)
        X = np.log2(1.0 + X[:, cols].astype(np.float32))
        out = np.zeros((len(r), G), dtype=np.float32); out[o[:, None], dst[None, :]] = X
        feat[s0:s0 + len(r)] = out
    feat.flush()
    json.dump(dict(animal="Allen-MERFISH-C57BL6J-638850", n_cells=int(N), n_genes=int(G), gene_names=list(ours),
                   measured=measured.astype(int).tolist(), n_measured=int(measured.sum()), sections=secs,
                   section_cells=[int((cells["brain_section_label"] == s_).sum()) for s_ in secs], classes=classes,
                   iso_mm=ISO, axes="stack coords rigidly mapped to CCF [z, y, x] = [ML, DV, AP], / iso_mm, per-axis scaled to mouse 1",
                   frame_fit=fit_info, has_ccf=int(cells["cx"].notna().sum()), feature="log2(1 + raw count); unmeasured genes 0",
                   hemispheres="hemi.npy: side of the midline plane in store coordinates (ML > midline_ml_iso = 1); 0 = the Zhuang side",
                   hemi_share_other=float(hemi.mean()), midline_ml_iso=float(mid_iso),
                   cell_labels_file="cell_labels.txt", platform="Vizgen MERSCOPE (Allen)", platform_id=1),
              open(f"{a.out_dir}/meta.json", "w"), indent=1)
    cells["cell_label"].to_csv(f"{a.out_dir}/cell_labels.txt", index=False, header=False)
    h.file.close()
    print(f"[allen] wrote {a.out_dir}: {N:,} cells, {len(secs)} sections ({time.time() - t0:.0f} s)", flush=True)


def save_infos_like(src, dst, n_genes):
    """Copy an infos.pkl with the input width set to n_genes."""
    import torch
    i = torch.load(src, map_location="cpu", weights_only=False)
    i.num_genes = n_genes
    i.input_dims = dict(i.input_dims, node_features_dimensions=n_genes); i.output_dims = dict(i.output_dims, node_features_dimensions=n_genes)
    torch.save(i, dst)


def cmd_test(a):
    """Allen test prebuilt (store rows grouped by section; the 477 measured genes; folded stack coordinates / ISO)."""
    S = a.store; meta = json.load(open(f"{S}/meta.json"))
    cols = np.flatnonzero(np.asarray(meta["measured"]) > 0); gn = [meta["gene_names"][c] for c in cols]
    lab = pd.read_csv(f"{S}/cell_labels.txt", header=None, names=["cell_label"], dtype=str)
    rec = pd.read_csv(f"{a.raw_dir}/reconstructed_coordinates.csv", dtype={"cell_label": str}); rec.columns = ["cell_label", "x", "y", "z", "rec_pi"]
    M = lab.merge(rec, on="cell_label", how="left")[["x", "y", "z"]].to_numpy(np.float64)
    assert len(M) == len(lab) and not np.isnan(M).any(), "stack coordinates missing for store cells"
    sec = np.load(f"{S}/sec.npy").astype(np.int64); cls = np.load(f"{S}/cls.npy").astype(np.int64)
    order = np.argsort(sec, kind="stable"); M, sec, cls = M[order], sec[order], cls[order]
    for s_ in np.unique(sec):
        assert np.ptp(M[sec == s_, 2]) < 1e-9, f"section {s_} is not one stack plane"
    ff = meta["frame_fit"]; Rm = np.asarray(ff["rotation"]); mM = np.asarray(ff["mean_meta"]); mC = np.asarray(ff["mean_ccf"])
    x_mid = mM[0] + (MID_ML - mC[0] - (M[:, 1] - mM[1]) * Rm[1, 0] - (M[:, 2] - mM[2]) * Rm[2, 0]) / Rm[0, 0]
    far = float((M[:, 0] > x_mid).mean())
    P = M.copy(); P[:, 0] = x_mid + np.abs(M[:, 0] - x_mid)
    fold = dict(axis=0, mid_ml_ccf=MID_ML, rot_col0=Rm[:, 0].tolist(), mean_meta=mM.tolist(), mean_ccf=mC.tolist(), units="mm",
                rule="x' = x_mid(y, z) + |x - x_mid(y, z)|, x_mid = mean_meta[0] + (mid_ml_ccf - mean_ccf[0] - (y - mean_meta[1]) rot_col0[1] - (z - mean_meta[2]) rot_col0[2]) / rot_col0[0]")
    names_sec = [meta["sections"][s_] for s_ in np.unique(sec)]
    sizes = [int((sec == s_).sum()) for s_ in np.unique(sec)]; offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    d = a.out_dir; os.makedirs(d, exist_ok=True)
    np.save(f"{d}/pos.npy", (P / ISO).astype(np.float32)); np.save(f"{d}/cell_class.npy", cls)
    np.save(f"{d}/cell_ID.npy", order.astype(np.int64)); np.save(f"{d}/offsets.npy", offsets)
    f = np.load(f"{S}/feat.npy", mmap_mode="r"); n = len(order)
    out = np.lib.format.open_memmap(f"{d}/feat.npy", mode="w+", dtype=np.float32, shape=(n, len(cols)))
    for i in range(0, n, 100000):
        sl = slice(i, min(n, i + 100000))
        x = np.asarray(f[np.sort(order[sl])], dtype=np.float32)
        x = x[np.argsort(np.argsort(order[sl]))]
        out[sl] = x[:, cols]
    out.flush()
    save_infos_like(f"{a.infos_from}/infos.pkl", f"{d}/infos.pkl", len(cols))
    json.dump(dict(animal="Allen-MERFISH-C57BL6J-638850", coord_source="reconstructed (section-stack) coordinates, mm, rigid frame as published, folded at the CCF midline along x; / ISO",
                   label_key="class", n_graphs=len(sizes), total_cells=int(len(P)), n_genes=len(cols), gene_names=gn, pos_dim=3,
                   coord_stats=dict(min=P.min(0).tolist(), max=P.max(0).tolist(), span=np.ptp(P, 0).tolist()),
                   section_names=names_sec, sizes=sizes, fold=fold, far_side_share=far,
                   thinning=dict(keep=1.0, seed=None, rule="binomial(round(2^f - 1), keep) once, then log2(1 + c)"),
                   store_rows="cell_ID.npy = row of the Allen cell store"), open(f"{d}/meta.json", "w"), indent=1)
    print(f"[allen] wrote {d}: {len(P):,} cells, {len(sizes)} sections, far side {far:.3f}", flush=True)


def lognorm_chunks(path, cols, B=200000):
    """log1p of library-size-normalised counts (1e4) of the 477 genes, in row blocks of B."""
    f = np.load(path, mmap_mode="r")
    for i in range(0, f.shape[0], B):
        x = np.asarray(f[i:i + B], dtype=np.float64); x = x[:, cols] if cols is not None else x
        c = np.exp2(x) - 1.0
        yield np.log1p(c / np.maximum(c.sum(1, keepdims=True), 1.0) * 1e4)


def pca_basis(sources, K):
    """Per-gene mean / sd over all cells, covariance of the clipped z-scores and its top-K eigenvectors."""
    n, s1, s2 = 0, np.zeros(477), np.zeros(477)
    for _, p, cols in sources:
        for x in lognorm_chunks(p, cols):
            n += len(x); s1 += x.sum(0); s2 += (x * x).sum(0)
    mu = s1 / n; sd = np.sqrt(np.maximum(s2 / n - mu * mu, 0)) + 1e-8
    z1, zz = np.zeros(477), np.zeros((477, 477))
    for _, p, cols in sources:
        for x in lognorm_chunks(p, cols):
            z = np.clip((x - mu) / sd, -10, 10); z1 += z.sum(0); zz += z.T @ z
    zm = z1 / n; cov = (zz - n * np.outer(zm, zm)) / (n - 1)
    w, V = np.linalg.eigh(cov); w, V = w[::-1], V[:, ::-1]; evr = w / w.sum()
    return mu, sd, zm, V[:, :K], evr


def project(p, cols, mu, sd, zm, comp):
    """Uncorrected PCs of every cell of one source."""
    return np.concatenate([((np.clip((x - mu) / sd, -10, 10) - zm) @ comp).astype(np.float32) for x in lognorm_chunks(p, cols)])


def cmd_harmony(a):
    """PCA of the 477 shared genes, Harmony over platform, and the corrected stores / test folder / training prebuilt."""
    import harmonypy as hm
    t0 = time.time(); K = a.k; OUT = a.out; ST = a.stores_dir; AL = a.allen_test
    names = json.load(open(f"{ST}/abca1/meta.json"))["gene_names"]
    am = json.load(open(f"{AL}/meta.json")); GN = am["gene_names"]
    COLS = np.array(sorted(names.index(x) for x in GN))
    assert len(COLS) == 477 and [names[c] for c in COLS] == GN
    SRC = [(m, f"{ST}/{m}/feat.npy", COLS) for m in ("abca1", "abca2", "abca3", "abca4")] + [("allen", f"{AL}/feat.npy", None)]
    for d in (OUT, f"{OUT}/raw", f"{OUT}/allen", f"{OUT}/train_prebuilt"):
        os.makedirs(d, exist_ok=True)
    mu, sd, zm, comp, evr = pca_basis(SRC, K)
    np.savez(f"{OUT}/pca.npz", genes=np.array(GN), mean=mu, sd=sd, clip=10.0, zmean=zm, components=comp, explained_variance_ratio=evr)
    print(f"PCA: {K} PCs explain {evr[:K].sum():.3f} ({time.time() - t0:.0f} s)", flush=True)
    P, sizes = [], []
    for name, p, cols in SRC:
        q = project(p, cols, mu, sd, zm, comp); np.save(f"{OUT}/raw/{name}.npy", q); P.append(q); sizes.append((name, len(q)))
    P = np.concatenate(P); batch = np.concatenate([["zhuang" if nm != "allen" else "allen"] * k for nm, k in sizes])
    print(f"projected {P.shape} ({time.time() - t0:.0f} s)", flush=True)
    t1 = time.time()
    ho = hm.run_harmony(P.astype(np.float64), pd.DataFrame({"platform": batch}), "platform")
    Zc = np.asarray(ho.Z_corr); Zc = (Zc.T if Zc.shape[0] != len(P) else Zc).astype(np.float32); del ho
    harmony_seconds = round(time.time() - t1)
    hpc = [f"hPC{i + 1}" for i in range(K)]
    feature = f"Harmony-corrected PCs (477 shared genes, lib 1e4, log1p, z clip 10, PCA {K}, harmonypy, batch = platform)"
    k0 = 0
    for name, k in sizes:
        zc = Zc[k0:k0 + k]; k0 += k
        d = f"{OUT}/allen" if name == "allen" else f"{OUT}/stores/{name}"
        os.makedirs(d, exist_ok=True); np.save(f"{d}/feat.npy", zc)
        src_dir = AL if name == "allen" else f"{ST}/{name}"
        files = ("pos.npy", "cell_class.npy", "offsets.npy", "cell_ID.npy") if name == "allen" else [f for f in os.listdir(src_dir) if f not in ("feat.npy", "meta.json")]
        for f in files:
            if not os.path.lexists(f"{d}/{f}"):
                os.symlink(os.path.realpath(f"{src_dir}/{f}"), f"{d}/{f}")
        m = json.load(open(f"{src_dir}/meta.json")); m.update(n_genes=K, gene_names=hpc, feature=feature)
        json.dump(m, open(f"{d}/meta.json", "w"), indent=1)
        if name == "allen":
            save_infos_like(f"{AL}/infos.pkl", f"{d}/infos.pkl", K)
    d = f"{OUT}/stores/abca2_ref1"; os.makedirs(d, exist_ok=True)
    links = {"feat.npy": f"{OUT}/stores/abca2/feat.npy", "pos.npy": f"{a.ref1_store}/pos.npy"}
    for f in ("cell_labels.txt", "cls.npy", "sec.npy"):
        links[f] = f"{ST}/abca2/{f}"
    for f, s in links.items():
        if not os.path.lexists(f"{d}/{f}"):
            os.symlink(os.path.realpath(s), f"{d}/{f}")
    m = json.load(open(f"{OUT}/stores/abca2/meta.json"))
    m["frame_fit"] = json.load(open(f"{a.ref1_store}/meta.json"))["frame_fit"]
    m["note_ruler"] = "positions = the mouse-2 store on mouse 1's per-axis ruler; features = the corrected mouse-2 PCs"
    json.dump(m, open(f"{d}/meta.json", "w"), indent=1)
    TP = a.train_prebuilt
    for f in os.listdir(TP):
        if f != "infos.pkl" and not os.path.lexists(f"{OUT}/train_prebuilt/{f}"):
            os.symlink(os.path.realpath(f"{TP}/{f}"), f"{OUT}/train_prebuilt/{f}")
    save_infos_like(f"{TP}/infos.pkl", f"{OUT}/train_prebuilt/infos.pkl", K)
    log = dict(harmony_seconds=harmony_seconds, K=K, explained_variance=float(evr[:K].sum()), cells={nm: k for nm, k in sizes}, genes=len(GN),
               batch="platform (zhuang vs allen)", harmonypy=getattr(hm, "__version__", "2.0.0"),
               recipe="2^f-1 -> lib 1e4 -> log1p -> z (all cells) clip 10 -> PCA K -> harmonypy defaults", total_seconds=round(time.time() - t0))
    json.dump(log, open(f"{OUT}/harmony.json", "w"), indent=1)
    print("harmony done", json.dumps(log), flush=True)


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("store")
    p.add_argument("--raw_dir", required=True); p.add_argument("--zhuang_raw_dir", required=True, help="DATA_DIR of the Zhuang mice (CCF files)")
    p.add_argument("--ref_store", required=True, help="mouse 1's cell store (gene order and per-axis ruler)")
    p.add_argument("--class_list", required=True); p.add_argument("--out_dir", required=True); p.add_argument("--rows_per_block", type=int, default=50000)
    p = sp.add_parser("test")
    p.add_argument("--store", required=True, help="the Allen cell store"); p.add_argument("--raw_dir", required=True)
    p.add_argument("--infos_from", required=True, help="a Zhuang test prebuilt whose infos.pkl is copied (input width set to 477)")
    p.add_argument("--out_dir", required=True)
    p = sp.add_parser("harmony")
    p.add_argument("--stores_dir", required=True, help="folder with the Zhuang cell stores abca1..abca4")
    p.add_argument("--ref1_store", required=True, help="the mouse-2 store on mouse 1's ruler (abca2_ref1)")
    p.add_argument("--allen_test", required=True, help="the Allen test prebuilt"); p.add_argument("--train_prebuilt", required=True)
    p.add_argument("--k", type=int, default=200); p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"store": cmd_store, "test": cmd_test, "harmony": cmd_harmony}[a.cmd](a)


if __name__ == "__main__":
    main()
