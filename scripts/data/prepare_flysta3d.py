"""Flysta3D-v2 fly embryos: training cell stores (11 embryos), two held-out test prebuilts and the infos-only training prebuilt.

Data: Flysta3D-v2 Stereo-seq embryos, STOmics DB STDS0000398 (https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000398/;
portal https://db.cngb.org/stomics/flysta3d-v2/). Files needed in $DATA_DIR/raw/flysta3d/ (the 13 embryo cell-bin h5ad files):
  16.32h_E12-14h_c_new_final_modified.h5ad  16.53h_E16-18h_b_new_final_modified.h5ad  16.57h_E16-18h_a_new_final_modified.h5ad
  16.63h_E16-18h_e_new_final_modified.h5ad  17.61h_E14-16h_f_all_final_modified.h5ad  16.88h_E14-16h_d_new_final_modified.h5ad
  17.34h_E14-16h_c_all_final_modified.h5ad  13.79h_E10-12h_d_new_final_modified.h5ad  15.75h_E12-14h_a_all_final_modified.h5ad
  15.77h_E12-14h_b_new_final_modified.h5ad  18.39h_E2.5-3h_d_new_final_modified.h5ad  18.97h_E2.5-3h_c_new_final_modified.h5ad
  19.84h_E18-20h_d_new_final_modified.h5ad
Command: DATA_DIR=... bash scripts/data/prepare_flysta3d.sh   (outputs in $DATA_DIR/embryos/ and $DATA_DIR/embryos_panel/)

Frame: each embryo is posed by its own expression (virtual-embryo convention): AP = first principal axis of the cell cloud, its
sign set by anterior (lab, Dfd, oc, ems) vs posterior (Abd-B, abd-A, cad) marker centroids; DV = dorsal (Hand, pnr) minus
ventral (elav, nSyb, posterior half) centroid, orthogonalised to AP; LR = AP x DV. Coordinates are align_spatial x scale (um),
centred at the median, rotated into (AP, DV, LR), divided by one unit U set from the five first training embryos so that their
mean RMS radius per axis is 0.163. Features (--features all): the genes present in all 13 files; training stores hold raw
counts (uint8, clipped at 255), test prebuilts log2(1 + min(count, 255)) as float32. --features panel: log2(1 + count) of the
2,000 seurat-flavour highly variable genes of the five training embryos (needs scanpy), absent genes 0. Test embryos keep their truth in the section frame (pos.npy);
pos_canonical.npy is the posed copy.
Writes <out>/stores/<5 embryos>, stores_more/<6 embryos>, prebuilt_test_<2 embryos>, prebuilt_infos_train,
build_summary.json."""
import argparse
import glob
import json
import os
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from scipy import sparse

EMB = ["16.32h_E12-14h_c_new_final_modified.h5ad", "16.53h_E16-18h_b_new_final_modified.h5ad",
       "16.57h_E16-18h_a_new_final_modified.h5ad", "16.63h_E16-18h_e_new_final_modified.h5ad",
       "17.61h_E14-16h_f_all_final_modified.h5ad", "16.88h_E14-16h_d_new_final_modified.h5ad",
       "17.34h_E14-16h_c_all_final_modified.h5ad"]
TEST = {"16.88h_E14-16h_d_new_final_modified.h5ad", "17.34h_E14-16h_c_all_final_modified.h5ad"}
MORE = ["13.79h_E10-12h_d_new_final_modified.h5ad", "15.75h_E12-14h_a_all_final_modified.h5ad",
        "15.77h_E12-14h_b_new_final_modified.h5ad", "18.39h_E2.5-3h_d_new_final_modified.h5ad",
        "18.97h_E2.5-3h_c_new_final_modified.h5ad", "19.84h_E18-20h_d_new_final_modified.h5ad"]
AP_ANTERIOR = ["lab", "Dfd", "oc", "ems"]
AP_POSTERIOR = ["Abd-B", "abd-A", "cad"]
DORSAL = ["Hand", "pnr"]
VENTRAL = ["elav", "nSyb"]
DATA_RMS = 0.163
FRAME_KEYS = ("AP", "DV", "LR", "roll_deg", "ap_tilt_deg", "ap_flip", "extent_ap_um", "extent_dv_um", "extent_lr_um",
              "transverse_radius_um", "stack_extent_um", "dorsal_markers", "ventral_markers", "lr_mirror")


def read_embryo(path):
    """Raw counts (cells x genes, CSR), gene names, section codes, positions in um and cell-type annotation of one embryo."""
    h = h5py.File(path, "r")
    n, G = [int(v) for v in h["layers/counts_X"].attrs["shape"]]
    X = sparse.csc_matrix((h["layers/counts_X/data"][:], h["layers/counts_X/indices"][:],
                           h["layers/counts_X/indptr"][:]), shape=(n, G)).tocsr()
    genes = np.array([g.decode() if isinstance(g, bytes) else str(g) for g in h["var/_index"][:]])
    sec_code = h["obs/slices/codes"][:].astype(np.int64)
    P = h["obsm/align_spatial"][:].astype(np.float64)
    scale = float(h["uns/spatial/scale"][()]); unit = h["uns/spatial/scale_unit"][()]
    unit = unit.decode() if isinstance(unit, bytes) else str(unit)
    lvl = "Annotation_1_cell_type"
    ann = (np.array([s.decode() if isinstance(s, bytes) else str(s) for s in h[f"obs/{lvl}/categories"][:]]),
           h[f"obs/{lvl}/codes"][:].astype(np.int64))
    ia = [x.decode() if isinstance(x, bytes) else str(x) for x in h["obs/IA/categories"][:]]
    h.close()
    assert unit == "um", unit
    return dict(n_cells=n, X=X, genes=genes, sec_code=sec_code, P=P * scale, ann=ann, ia=ia)


def libnorm(X):
    """Per-cell library size to 250, then log1p (used only to locate the marker centroids)."""
    Y = X.astype(np.float64).copy()
    lib = np.maximum(np.asarray(Y.sum(1)).ravel(), 1.0)
    Y = sparse.diags(250.0 / lib) @ Y
    Y.data = np.log1p(Y.data)
    return Y.tocsr().astype(np.float32)


def centroid3(Y, gi, name, Q, mask=None):
    """Expression-weighted 3-D centroid of one marker gene in the centred frame Q (None if the gene is absent or silent)."""
    if name not in gi:
        return None
    w = np.asarray(Y[:, gi[name]].todense()).ravel().astype(np.float64)
    if mask is not None:
        w = w * mask
    if w.sum() <= 0:
        return None
    return (w[:, None] * Q).sum(0) / w.sum()


def build_frame(E, Y):
    """Marker-posed body frame (AP, DV, LR unit vectors) of one embryo, its sections in depth order and frame statistics."""
    P = E["P"]; z = P[:, 2]
    secs = np.unique(E["sec_code"])
    sec_z = np.array([np.median(z[E["sec_code"] == s]) for s in secs])
    sec_n = np.array([int((E["sec_code"] == s).sum()) for s in secs])
    o = np.argsort(sec_z); secs, sec_z, sec_n = secs[o], sec_z[o], sec_n[o]
    Q = P - np.median(P, 0)
    w, V = np.linalg.eigh(np.cov(Q.T))
    AP = V[:, int(np.argmax(w))]
    gi = {g: i for i, g in enumerate(E["genes"])}
    ca = {g: float(centroid3(Y, gi, g, Q) @ AP) for g in AP_ANTERIOR if g in gi}
    cp = {g: float(centroid3(Y, gi, g, Q) @ AP) for g in AP_POSTERIOR if g in gi}
    ap_flip = bool(np.mean(list(cp.values())) < np.mean(list(ca.values())))
    if ap_flip:
        AP = -AP
    a = Q @ AP
    post = (a > np.median(a)).astype(np.float64)
    dc = {g: centroid3(Y, gi, g, Q) for g in DORSAL}
    vc = {g: centroid3(Y, gi, g, Q, mask=post) for g in VENTRAL}
    dc = {k: v for k, v in dc.items() if v is not None}; vc = {k: v for k, v in vc.items() if v is not None}
    dv = np.mean(list(dc.values()), 0) - np.mean(list(vc.values()), 0)
    dv = dv - (dv @ AP) * AP
    DV = dv / float(np.linalg.norm(dv))
    LR = np.cross(AP, DV)
    roll = float(np.degrees(np.arcsin(abs(DV[2]))))
    ap_tilt = float(np.degrees(np.arcsin(abs(AP[2]))))
    u_inplane = np.cross(AP, np.array([0.0, 0.0, 1.0]))
    u_inplane = u_inplane / np.linalg.norm(u_inplane)
    lo_a, hi_a = np.percentile(a, [0.5, 99.5])
    t = Q @ u_inplane
    lo_t, hi_t = np.percentile(t, [0.5, 99.5])
    R = (hi_t - lo_t) / 2.0
    return dict(AP=AP, DV=DV, LR=LR, secs=secs, sec_z=sec_z, ap_flip=ap_flip,
                dorsal_markers=sorted(dc), ventral_markers=sorted(vc), roll_deg=roll, ap_tilt_deg=ap_tilt,
                transverse_radius_um=float(R), extent_ap_um=float(hi_a - lo_a),
                extent_dv_um=float(np.ptp(np.percentile(Q @ DV, [0.5, 99.5]))),
                extent_lr_um=float(np.ptp(np.percentile(Q @ LR, [0.5, 99.5]))),
                stack_extent_um=float(sec_z.max() - sec_z.min()),
                lr_mirror="uncorrected (bilateral symmetry; recorded, not fixed)")


def tag_of(fn):
    """Embryo tag from a file name, e.g. 16.32h_E12-14h_c."""
    return os.path.basename(fn).split("_new")[0].split("_all")[0]


def pose(raw, fn):
    """Read one embryo and pose it: centred um positions (section frame Q and body frame C), sections, class names, frame."""
    e = read_embryo(os.path.join(raw, fn))
    tag = tag_of(fn)
    assert e["ia"] == [f"{tag.split('h_')[0]}h"], (tag, e["ia"])
    f = build_frame(e, libnorm(e["X"]))
    Q = e["P"] - np.median(e["P"], 0)
    C = np.stack([Q @ f["AP"], Q @ f["DV"], Q @ f["LR"]], 1)
    names, codes = e["ann"]
    cls_names = np.where(codes >= 0, names[np.maximum(codes, 0)], "unannotated")
    rank = {int(s): i for i, s in enumerate(f["secs"])}
    sec = np.array([rank[int(s)] for s in e["sec_code"]], dtype=np.int64)
    frame = {k: (f[k].tolist() if isinstance(f[k], np.ndarray) else f[k]) for k in FRAME_KEYS}
    print(f"{tag}: {e['n_cells']:,} cells, {len(f['secs'])} sections, roll {f['roll_deg']:.0f} deg, ap_flip {f['ap_flip']}", flush=True)
    return dict(e=e, tag=tag, C=C, Q=Q, cls_names=cls_names, sec=sec, sec_depth=np.asarray(f["sec_z"], float), frame=frame,
                age=float(tag.split("h_")[0]))


def panel_counts(e, panel):
    """Raw counts of the panel genes, clipped at 255, as uint8."""
    gi = {g: i for i, g in enumerate(e["genes"])}
    C = e["X"][:, np.array([gi[g] for g in panel])].toarray()
    assert np.allclose(C, np.round(C)) and C.min() >= 0
    return np.minimum(C, 255).astype(np.uint8)


def hvg_features(e, hvg):
    """log2(1 + count) of the 2,000 split genes (genes absent from this embryo are 0) and the number absent."""
    gi = {g: i for i, g in enumerate(e["genes"])}
    cols = np.array([gi.get(g, -1) for g in hvg])
    Xh = e["X"][:, np.maximum(cols, 0)].toarray().astype(np.float32)
    Xh[:, cols < 0] = 0.0
    return np.log2(1.0 + Xh).astype(np.float32), int((cols < 0).sum())


def split_hvgs(raw):
    """The 2,000 seurat-flavour highly variable genes of the five training embryos (library size 1e4, log1p), over the genes
    shared by the seven embryos of the 16.3-17.6 h cluster; also checks that the seeded test choice gives the two test embryos."""
    import anndata as ad
    import scanpy as sc
    fs = sorted(EMB)
    test = sorted(np.random.default_rng(0).choice(len(fs), 2, replace=False).tolist())
    assert {fs[i] for i in test} == TEST, "seeded test choice differs"
    A = [ad.read_h5ad(os.path.join(raw, f)) for f in fs]
    genes = sorted(set.intersection(*[set(a.var_names) for a in A]))

    def counts(a):
        C = a[:, genes].layers["counts_X"]
        return C.toarray() if hasattr(C, "toarray") else np.asarray(C)
    tr = ad.AnnData(X=np.concatenate([counts(A[i]) for i in range(len(fs)) if i not in test]).astype(np.float32))
    tr.var_names = genes
    sc.pp.normalize_total(tr, target_sum=1e4); sc.pp.log1p(tr)
    sc.pp.highly_variable_genes(tr, n_top_genes=2000, flavor="seurat")
    return list(tr.var_names[tr.var["highly_variable"]])


def save_infos(d, n_cells, name, note, n_genes, classes):
    """Write the infos.pkl that the trainer and sampler read."""
    infos = SimpleNamespace(name=name, num_cell_class=len(classes), num_genes=n_genes,
                            cell_class_decoder={i: c for i, c in enumerate(classes)},
                            num_cell_to_region_mapping_dict={n_cells: "embryo"},
                            input_dims={"node_features_dimensions": n_genes, "diffusion_time_dimensions": 1},
                            output_dims={"node_features_dimensions": n_genes, "diffusion_time_dimensions": 0},
                            pos_dim=3, meta=dict(note=note))
    torch.save(infos, os.path.join(d, "infos.pkl"))


def write_store(d, v, U, classes, genes, feat, feature):
    """Write one training cell store (positions in model units, features, sections, classes, meta)."""
    os.makedirs(d, exist_ok=True)
    cidx = {c: i for i, c in enumerate(classes)}
    n = int(v["e"]["n_cells"])
    np.save(f"{d}/pos.npy", (v["C"] / U).astype(np.float32)); np.save(f"{d}/feat.npy", feat); np.save(f"{d}/sec.npy", v["sec"])
    np.save(f"{d}/cls.npy", np.array([cidx[c] for c in v["cls_names"]], dtype=np.int64))
    with open(f"{d}/cell_labels.txt", "w") as fh:
        fh.write("\n".join(f"{v['tag']}_{i}" for i in range(n)) + "\n")
    sd = np.diff(np.sort(v["sec_depth"])) if len(v["sec_depth"]) > 1 else np.array([0.0])
    meta = dict(animal=f"Flysta3D-v2 {v['tag']}", n_cells=n, n_genes=len(genes), gene_names=genes,
                sections=[str(s) for s in range(len(v["sec_depth"]))],
                section_cells=np.bincount(v["sec"], minlength=len(v["sec_depth"])).tolist(), classes=list(classes),
                iso_mm=U / 1000.0, axes="marker-posed [AP, DV, LR], physical um / U",
                sec_depth_um=v["sec_depth"].tolist(), sec_gap_um=float(np.median(sd)), sagittal=False,
                whole_ok=True, feature=feature, frame=v["frame"], platform_id=0, age_h=v["age"])
    json.dump(meta, open(f"{d}/meta.json", "w"), indent=1)


FORE = {"pharynx", "esophagus", "pharynx/esophagus", "proventriculus"}
HIND = {"hindgut", "anal pad", "Malpighian tubule"}
CNS = {"brain/ventral nerve cord", "brain/ventral nerve cord_inner", "brain/ventral nerve cord_outer"}
DORSAL_TISSUE = {"dorsal trunk", "fat body", "tracheal system", "dorsal epidermis", "epidermis_dorsal"}


def frame_check(v):
    """Median AP of foregut vs hindgut cells and median DV of CNS vs dorsal-tissue cells (the embryo's frame sanity check)."""
    C, names = v["C"], v["cls_names"]
    med = lambda S, ax: float(np.median(C[np.isin(names, list(S)), ax])) if np.isin(names, list(S)).any() else float("nan")
    fa, ha, cv, dv = med(FORE, 0), med(HIND, 0), med(CNS, 1), med(DORSAL_TISSUE, 1)
    rms = np.sqrt(((C - C.mean(0)) ** 2).mean(0))
    return dict(roll_deg=v["frame"]["roll_deg"], ap_flip=v["frame"]["ap_flip"], rms_um=rms.round(1).tolist(), foregut_ap=fa,
                hindgut_ap=ha, cns_dv=cv, dorsal_dv=dv, frame_ok=bool(fa < ha and cv < dv))


def main():
    """Build the stores, the test prebuilts and the infos-only training prebuilt."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="folder with the 13 *_final_modified.h5ad files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--features", default="all", choices=["all", "panel"],
                    help="all = every gene present in all 13 files (stores: uint8 counts); panel = the 2,000 split genes (log2)")
    args = ap.parse_args()
    files = sorted(glob.glob(f"{args.raw}/*_final_modified.h5ad"))
    assert sorted(os.path.basename(f) for f in files) == sorted(EMB + MORE), "expected exactly the 13 embryo files"
    emb = {fn: pose(args.raw, fn) for fn in EMB + MORE}
    if args.features == "all":
        genes = sorted(set.intersection(*[set(v["e"]["genes"].tolist()) for v in emb.values()]))
        feature = "raw counts (uint8, clipped at 255) of the genes present in all 13 files; uint8 stores are read as counts"
        store_feat = lambda e: (panel_counts(e, genes), 0)
        test_feat = lambda e, o: np.log2(1.0 + panel_counts(e, genes)[o].astype(np.float32)).astype(np.float32)
    else:
        genes = split_hvgs(args.raw)
        feature = "log2(1 + raw count), split HVGs"
        store_feat = lambda e: hvg_features(e, genes)
        test_feat = lambda e, o: hvg_features(e, genes)[0][o]
    print(f"{args.features} features: {len(genes):,} genes", flush=True)
    classes = sorted(set().union(*[set(emb[fn]["cls_names"]) for fn in EMB]))
    cidx = {c: i for i, c in enumerate(classes)}
    train = [fn for fn in EMB if fn not in TEST]
    rms = np.mean([np.sqrt((emb[fn]["C"] ** 2).mean()) for fn in train])
    U = float(rms / DATA_RMS)
    print(f"{len(classes)} classes; training RMS radius per axis {rms:.1f} um -> U = {U:.1f} um per model unit", flush=True)
    os.makedirs(args.out, exist_ok=True)
    summary = dict(U_um_per_unit=U, data_rms=DATA_RMS, classes=list(classes), embryos={})
    for fn in EMB:
        v = emb[fn]; e = v["e"]; tag = v["tag"]
        feat, missing = store_feat(e)
        summary["embryos"][tag] = dict(n_cells=int(e["n_cells"]), n_sections=len(v["sec_depth"]), test=fn in TEST,
                                       **({"hvg_missing": missing} if args.features == "panel" else {}),
                                       **{k: v["frame"][k] for k in ("roll_deg", "ap_flip", "extent_ap_um")})
        if fn not in TEST:
            write_store(os.path.join(args.out, "stores", tag), v, U, classes, genes, feat, feature)
            continue
        d = os.path.join(args.out, f"prebuilt_test_{tag}"); os.makedirs(d, exist_ok=True)
        o = np.argsort(v["sec"], kind="stable")
        cls = np.array([cidx[c] for c in v["cls_names"]], dtype=np.int64)
        sec_cells = np.bincount(v["sec"], minlength=len(v["sec_depth"])).tolist()
        np.save(f"{d}/pos.npy", (v["Q"] / U).astype(np.float32)[o]); np.save(f"{d}/pos_canonical.npy", (v["C"] / U).astype(np.float32)[o])
        np.save(f"{d}/feat.npy", test_feat(e, o)); np.save(f"{d}/cell_class.npy", cls[o])
        np.save(f"{d}/cell_ID.npy", o.astype(np.int64))
        np.save(f"{d}/offsets.npy", np.concatenate([[0], np.cumsum(sec_cells)]).astype(np.int64))
        meta = dict(n_graphs=len(sec_cells), total_cells=int(e["n_cells"]), n_genes=len(genes), pos_dim=3,
                    section_names=[f"{tag}_sec{s:02d}" for s in range(len(sec_cells))], sizes=sec_cells,
                    note="held-out embryo, cells grouped by section in depth order", um_per_unit=U, frame=v["frame"], age_h=v["age"])
        json.dump(meta, open(f"{d}/meta.json", "w"), indent=1)
        save_infos(d, int(e["n_cells"]), os.path.basename(d), "Flysta3D test embryo", len(genes), classes)
    d = os.path.join(args.out, "prebuilt_infos_train"); os.makedirs(d, exist_ok=True)
    v = emb[train[0]]; o = np.argsort(v["sec"], kind="stable"); sc = np.bincount(v["sec"]).tolist()
    np.save(f"{d}/pos.npy", (v["C"] / U).astype(np.float32)[o]); np.save(f"{d}/feat.npy", test_feat(v["e"], o))
    np.save(f"{d}/cell_class.npy", np.array([cidx[c] for c in v["cls_names"]], dtype=np.int64)[o])
    np.save(f"{d}/cell_ID.npy", o.astype(np.int64)); np.save(f"{d}/offsets.npy", np.concatenate([[0], np.cumsum(sc)]).astype(np.int64))
    json.dump(dict(n_graphs=len(sc), total_cells=int(len(o)), n_genes=len(genes), pos_dim=3, section_names=[f"s{i}" for i in range(len(sc))],
                   sizes=sc, note="infos only (training uses the cell stores)"), open(f"{d}/meta.json", "w"), indent=1)
    save_infos(d, int(len(o)), "prebuilt_infos_train", "infos for the Flysta trainer", len(genes), classes)
    classes = list(classes)
    more = dict(U_um_per_unit=U, embryos={})
    for fn in MORE:
        v = emb[fn]
        new = sorted(set(v["cls_names"]) - set(classes))
        classes += new
        feat, missing = store_feat(v["e"])
        write_store(os.path.join(args.out, "stores_more", v["tag"]), v, U, classes, genes, feat, feature)
        more["embryos"][v["tag"]] = dict(n_cells=int(v["e"]["n_cells"]), n_sections=len(v["sec_depth"]), hvg_missing=missing,
                                         new_classes=new, **frame_check(v))
    if args.features == "all":
        summary.update(panel_genes=genes, panel_note="genes present in all 13 release files")
    else:
        more["classes"] = classes
        json.dump(more, open(os.path.join(args.out, "build_summary_more.json"), "w"), indent=1)
    json.dump(summary, open(os.path.join(args.out, "build_summary.json"), "w"), indent=1)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
