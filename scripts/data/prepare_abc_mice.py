"""Zhuang ABC-atlas MERFISH mice 1-4: test prebuilts, infos-only training prebuilt and training cell stores.

Data: Allen Brain Cell (ABC) atlas, public bucket https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/ (release manifest
releases/20241115/manifest.json). Files needed in $DATA_DIR/raw/abc_mice/ (renamed as listed):
  Zhuang-ABCA-N-raw.h5ad                                 <- expression_matrices/Zhuang-ABCA-N/20230830/Zhuang-ABCA-N-raw.h5ad (N = 1..4)
  Zhuang-ABCA-N-CCF_ccf_coordinates.csv                  <- metadata/Zhuang-ABCA-N-CCF/20230830/ccf_coordinates.csv (N = 1..4)
  Zhuang-ABCA-N_cell_metadata_with_cluster_annotation.csv <- metadata/Zhuang-ABCA-N/20241115/views/cell_metadata_with_cluster_annotation.csv (N = 1, 2)
  Zhuang-ABCA-N_cell_metadata.csv                        <- metadata/Zhuang-ABCA-N/20241115/cell_metadata.csv (N = 3, 4)
Command: DATA_DIR=... bash scripts/data/prepare_abc_mice.sh   (runs the sub-commands below in order; outputs in $DATA_DIR/brains/)

Sub-commands:
  annotate     class of mice 3 / 4 looked up by cluster_alias in mice 1 / 2's annotation; writes their annotated metadata and
               iso_stats_abcaN.json (coordinate minimum, span 13.4577 mm on every axis)
  classes      classes.json = the sorted class names over the four mice (one shared encoding)
  prebuilt     per-section prebuilt of one mouse (metadata coordinates, log2(1 + count), sections split into <= --chunk_max_cells graphs)
  mixed        a prebuilt of mixed graphs (sections, blocks of consecutive sections, whole-brain subsets) drawn from a prebuilt
  iso          the same prebuilt in true proportions: every axis divided by the largest training span (13.4577 mm)
  store        per-mouse cell store: every cell, metadata coordinates mapped rigidly onto the CCF axes [ML, DV, AP], / 13.4577 mm
  rescale      per-axis ruler: scale a store about its centroid by D = slope(store) / slope(reference store), slopes of CCF on
               store coordinates; with --out the rescaled copy is written to a new folder (other files linked)
  rereference  move an already rescaled store onto another reference store's ruler (new folder)"""
import argparse
import json
import os
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd

ISO = 13.45773983001709


def annot_csv(annot_dir, animal):
    """Path of a mouse's annotated cell metadata."""
    return os.path.join(annot_dir, f"Zhuang-ABCA-{animal}_cell_metadata_with_cluster_annotation.csv")


def meta_csv(a):
    """Annotated metadata of the mouse a.animal: the downloaded file for mice 1 / 2, the annotate output for mice 3 / 4."""
    return annot_csv(a.annot_dir if a.animal in ("3", "4") else a.raw_dir, a.animal)


def cmd_annotate(a):
    """Annotate mice 3 and 4 with mice 1 / 2's class per cluster_alias and write their coordinate statistics."""
    import anndata as ad
    amap = pd.concat([pd.read_csv(annot_csv(a.raw_dir, x), usecols=["cluster_alias", "class"]) for x in ("1", "2")]
                     ).dropna().drop_duplicates("cluster_alias").set_index("cluster_alias")["class"]
    g1 = list(ad.read_h5ad(f"{a.raw_dir}/Zhuang-ABCA-1-raw.h5ad", backed="r").var_names)
    os.makedirs(a.out_dir, exist_ok=True)
    for x in ("3", "4"):
        m = pd.read_csv(f"{a.raw_dir}/Zhuang-ABCA-{x}_cell_metadata.csv", dtype={"cell_label": str})
        m["class"] = m["cluster_alias"].map(amap)
        print(f"ABCA-{x}: {len(m):,} cells, class found for {m['class'].notna().mean():.1%}", flush=True)
        m = m.dropna(subset=["class", "x", "y", "z"])
        m[["cell_label", "brain_section_label", "class", "x", "y", "z"]].to_csv(annot_csv(a.out_dir, x), index=False)
        ga = list(ad.read_h5ad(f"{a.raw_dir}/Zhuang-ABCA-{x}-raw.h5ad", backed="r").var_names)
        assert ga == g1, f"ABCA-{x} gene order differs from ABCA-1"
        c = m[["x", "y", "z"]].to_numpy(float)
        json.dump({"min": c.min(0).tolist(), "max": c.max(0).tolist(), "span": [ISO] * 3}, open(f"{a.out_dir}/iso_stats_abca{x}.json", "w"))


def cmd_classes(a):
    """Write the shared class list (sorted names over the four mice)."""
    s = set()
    for x in "1234":
        s |= set(pd.read_csv(annot_csv(a.annot_dir if x in "34" else a.raw_dir, x), usecols=["class"])["class"].dropna().astype(str))
    json.dump(sorted(s), open(a.out, "w"), indent=1)


def make_graph_index(cells, chunk_max_cells, seed):
    """One graph per section; sections larger than chunk_max_cells are split at random into bounded chunks."""
    rng = np.random.default_rng(seed)
    graphs = []
    section_sizes = {}
    for sec, sub in cells.groupby("cell_section", sort=True):
        idx = sub.index.to_numpy(np.int64)
        if chunk_max_cells and len(idx) > chunk_max_cells:
            shuffled = idx.copy()
            rng.shuffle(shuffled)
            for ci, start in enumerate(range(0, len(shuffled), chunk_max_cells)):
                chunk = np.sort(shuffled[start:start + chunk_max_cells])
                name = f"{sec}__chunk{ci:03d}"
                graphs.append((name, chunk))
                section_sizes[name] = int(len(chunk))
        else:
            name = str(sec)
            graphs.append((name, idx))
            section_sizes[name] = int(len(idx))
    return graphs, section_sizes


def cmd_prebuilt(a):
    """Per-section prebuilt of one mouse: feat.npy, pos.npy, cell_class.npy, cell_ID.npy, offsets.npy, infos.pkl, meta.json."""
    import anndata as ad
    import torch
    from scipy import sparse
    stem = f"Zhuang-ABCA-{a.animal}"
    paths = {"h5ad": os.path.join(a.raw_dir, f"{stem}-raw.h5ad"), "metadata": meta_csv(a),
             "ccf": os.path.join(a.raw_dir, f"{stem}-CCF_ccf_coordinates.csv")}
    for p in paths.values():
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    meta = pd.read_csv(paths["metadata"], usecols=["cell_label", "brain_section_label", "class", "x", "y", "z"], dtype={"cell_label": str})
    meta = meta.rename(columns={"brain_section_label": "cell_section", "class": "cell_class", "x": "coord_X", "y": "coord_Y", "z": "coord_Z"})
    meta["cell_class"] = meta["cell_class"].astype(str)
    cells = meta[["cell_label", "cell_section", "cell_class", "coord_X", "coord_Y", "coord_Z"]]
    cells = pd.concat([sub for _, sub in cells.groupby("cell_section", sort=True)], axis=0, ignore_index=True)
    h = ad.read_h5ad(paths["h5ad"], backed="r")
    row_idx = pd.Index(h.obs_names.astype(str)).get_indexer(cells["cell_label"].astype(str))
    keep = row_idx >= 0
    if not bool(keep.all()):
        print(f"[prebuilt] dropping {int((~keep).sum())} cells absent from h5ad obs", flush=True)
        cells = cells.loc[keep].reset_index(drop=True)
        row_idx = row_idx[keep]
    gene_names = list(h.var_names)
    coords = cells[["coord_X", "coord_Y", "coord_Z"]].to_numpy(np.float32)
    if a.coord_stats_json:
        stats = json.load(open(a.coord_stats_json))
    else:
        lo, hi = coords.min(axis=0), coords.max(axis=0)
        stats = {"min": lo.tolist(), "max": hi.tolist(), "span": np.where(hi > lo, hi - lo, 1.0).tolist()}
    coords = ((coords - np.asarray(stats["min"], dtype=np.float32)) / np.asarray(stats["span"], dtype=np.float32) - 0.5).astype(np.float32)
    if a.write_coord_stats_json:
        os.makedirs(os.path.dirname(os.path.abspath(a.write_coord_stats_json)), exist_ok=True)
        json.dump(stats, open(a.write_coord_stats_json, "w"), indent=1)
    classes = sorted(cells["cell_class"].astype(str).unique())
    enc = {label: i for i, label in enumerate(classes)}
    decoder = {i: label for label, i in enc.items()}
    graphs, section_sizes = make_graph_index(cells, a.chunk_max_cells, a.seed)
    sizes = [len(idx) for _, idx in graphs]
    offsets = np.zeros(len(sizes) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(sizes)
    total = int(offsets[-1])
    G = len(gene_names)
    os.makedirs(a.out_dir, exist_ok=True)
    feat = np.lib.format.open_memmap(os.path.join(a.out_dir, "feat.npy"), mode="w+", dtype=np.float32, shape=(total, G))
    pos = np.lib.format.open_memmap(os.path.join(a.out_dir, "pos.npy"), mode="w+", dtype=np.float32, shape=(total, 3))
    cc = np.lib.format.open_memmap(os.path.join(a.out_dir, "cell_class.npy"), mode="w+", dtype=np.int64, shape=(total,))
    cid = np.lib.format.open_memmap(os.path.join(a.out_dir, "cell_ID.npy"), mode="w+", dtype=np.int64, shape=(total,))
    cell_classes = cells["cell_class"].astype(str).map(enc).to_numpy(np.int64)
    for j, (_, local_idx) in enumerate(graphs):
        s, e = int(offsets[j]), int(offsets[j + 1])
        source_rows = row_idx[local_idx]
        order = np.argsort(source_rows)
        X = h.X[source_rows[order], slice(0, G)]
        X = X.toarray().astype(np.float32, copy=False) if sparse.issparse(X) else np.asarray(X, dtype=np.float32)
        np.log2(1.0 + X, out=X)
        feat[s:e] = X[np.argsort(order)]
        pos[s:e] = coords[local_idx]
        cc[s:e] = cell_classes[local_idx]
        cid[s:e] = local_idx.astype(np.int64)
    h.file.close()
    feat.flush(); pos.flush(); cc.flush(); cid.flush()
    np.save(os.path.join(a.out_dir, "offsets.npy"), offsets)
    meta = {"animal": f"Zhuang-ABCA-{a.animal}", "coord_source": "metadata", "label_key": "class", "n_graphs": len(graphs),
            "total_cells": total, "n_genes": G, "gene_names": gene_names, "pos_dim": 3, "coord_stats": stats, "max_sections": 0,
            "section_mode": "first", "cells_per_section": 0, "chunk_max_cells": a.chunk_max_cells,
            "section_names": [name for name, _ in graphs], "sizes": sizes, "raw_paths": paths}
    json.dump(meta, open(os.path.join(a.out_dir, "meta.json"), "w"), indent=1)
    infos = SimpleNamespace(name="abca3d_prebuilt", num_cell_class=len(decoder), num_genes=int(G), cell_class_decoder=decoder,
                            num_cell_to_region_mapping_dict={int(v): str(k) for k, v in section_sizes.items()},
                            input_dims={"node_features_dimensions": int(G), "diffusion_time_dimensions": 1},
                            output_dims={"node_features_dimensions": int(G), "diffusion_time_dimensions": 0}, pos_dim=3, meta=meta)
    torch.save(infos, os.path.join(a.out_dir, "infos.pkl"))
    print(f"[prebuilt] wrote {a.out_dir}: {len(graphs)} graphs, {total:,} cells, {G} genes", flush=True)


def mixed_graphs(names, off, pos, a):
    """Graph list of the mixed prebuilt: (name, source rows) for the dense section graphs (every --dense_stride-th), random
    subsets of blocks of consecutive sections (two widths) and random whole-brain subsets; also the number of block graphs."""
    rng = np.random.default_rng(a.seed)
    sec_rows, sec_z = {}, {}
    for i, nm in enumerate(names):
        s, e = int(off[i]), int(off[i + 1])
        sec = nm.split("__chunk")[0]
        sec_rows.setdefault(sec, []).append(np.arange(s, e, dtype=np.int64))
        sec_z.setdefault(sec, float(pos[s, 2]))
    for k in sec_rows:
        sec_rows[k] = np.concatenate(sec_rows[k])
    secs = sorted(sec_rows, key=lambda s: sec_z[s])
    graphs = []
    for i, nm in enumerate(names):
        if a.dense_stride > 1 and (i % a.dense_stride) != 0:
            continue
        graphs.append((f"dense__{nm}", np.arange(int(off[i]), int(off[i + 1]), dtype=np.int64)))
    n_block = 0
    for k, stride, reps in ((a.block_k, a.block_stride, a.block_reps), (a.block_k2, a.block2_stride, a.block2_reps)):
        if k and k < len(secs):
            for start in range(0, len(secs) - k + 1, stride):
                pool = np.concatenate([sec_rows[s] for s in secs[start:start + k]])
                for rep in range(reps):
                    take = rng.choice(pool, size=min(a.cells, len(pool)), replace=False)
                    graphs.append((f"block{k}_s{start:03d}_r{rep}", np.sort(take)))
                    n_block += 1
    all_rows = np.concatenate([sec_rows[s] for s in secs])
    for rep in range(a.volume_reps):
        take = rng.choice(all_rows, size=min(a.cells, len(all_rows)), replace=False)
        graphs.append((f"volume_r{rep:02d}", np.sort(take)))
    return graphs, n_block


def cmd_mixed(a):
    """Mixed-graph prebuilt: the graphs of mixed_graphs(), rows copied from --src."""
    import torch
    src_meta = json.load(open(f"{a.src}/meta.json"))
    off = np.load(f"{a.src}/offsets.npy")
    pos = np.load(f"{a.src}/pos.npy", mmap_mode="r")
    feat = np.load(f"{a.src}/feat.npy", mmap_mode="r")
    cc = np.load(f"{a.src}/cell_class.npy", mmap_mode="r")
    cid = np.load(f"{a.src}/cell_ID.npy", mmap_mode="r")
    names = src_meta["section_names"]
    G = int(src_meta["n_genes"])
    assert len(names) == len(off) - 1
    graphs, n_block = mixed_graphs(names, off, pos, a)
    sizes = [len(idx) for _, idx in graphs]
    new_off = np.zeros(len(sizes) + 1, dtype=np.int64)
    new_off[1:] = np.cumsum(sizes)
    total = int(new_off[-1])
    print(f"[mixed] {len(graphs)} graphs / {total:,} rows", flush=True)
    os.makedirs(a.out, exist_ok=True)
    nf = np.lib.format.open_memmap(f"{a.out}/feat.npy", mode="w+", dtype=np.float32, shape=(total, G))
    npo = np.lib.format.open_memmap(f"{a.out}/pos.npy", mode="w+", dtype=np.float32, shape=(total, pos.shape[1]))
    ncc = np.lib.format.open_memmap(f"{a.out}/cell_class.npy", mode="w+", dtype=np.int64, shape=(total,))
    nci = np.lib.format.open_memmap(f"{a.out}/cell_ID.npy", mode="w+", dtype=np.int64, shape=(total,))
    for j, (nm, idx) in enumerate(graphs):
        s, e = int(new_off[j]), int(new_off[j + 1])
        nf[s:e] = feat[idx]
        npo[s:e] = pos[idx]
        ncc[s:e] = cc[idx]
        nci[s:e] = cid[idx]
    for arr in (nf, npo, ncc, nci):
        arr.flush()
    np.save(f"{a.out}/offsets.npy", new_off)
    meta = dict(src_meta)
    meta.update(n_graphs=len(graphs), total_cells=total, section_names=[nm for nm, _ in graphs], sizes=sizes,
                mixed=dict(cells=a.cells, block_k=a.block_k, block_stride=a.block_stride, block_reps=a.block_reps,
                           volume_reps=a.volume_reps, keep_dense=True, n_block=n_block, src=a.src),
                note="mixed graphs (sections, blocks, whole-brain subsets)")
    json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=1)
    infos = torch.load(f"{a.src}/infos.pkl", map_location="cpu", weights_only=False)
    infos.num_cell_to_region_mapping_dict = {int(n): str(nm) for nm, n in zip(meta["section_names"], sizes)}
    infos.meta = meta
    torch.save(infos, f"{a.out}/infos.pkl")


def cmd_iso(a):
    """Copy of a prebuilt with true proportions: pos x (span / largest span) of the training statistics; other files linked."""
    span = np.array(json.load(open(a.coord_stats_json))["span"], dtype=np.float64)
    f = (span / span.max()).astype(np.float32)
    os.makedirs(a.out, exist_ok=True)
    for fn in os.listdir(a.src):
        if fn != "pos.npy" and not os.path.exists(f"{a.out}/{fn}"):
            os.symlink(os.path.abspath(f"{a.src}/{fn}"), f"{a.out}/{fn}")
    p = np.load(f"{a.src}/pos.npy", mmap_mode="r")
    q = np.lib.format.open_memmap(f"{a.out}/pos.npy", mode="w+", dtype=np.float32, shape=p.shape)
    for i in range(0, p.shape[0], 1_000_000):
        q[i:i + 1_000_000] = np.asarray(p[i:i + 1_000_000]) * f
    q.flush()
    print(f"[iso] {a.out}: {p.shape}, factors {f}", flush=True)


def cmd_store(a):
    """Cell store of one mouse: feat (log2(1 + count)), pos (metadata coordinates rigidly mapped onto CCF [ML, DV, AP], / ISO),
    sec, cls (shared class list), cell_labels.txt, meta.json (with the rigid fit)."""
    import anndata as ad
    from scipy import sparse
    stem = f"Zhuang-ABCA-{a.animal}"
    meta = pd.read_csv(meta_csv(a), usecols=["cell_label", "brain_section_label", "class", "x", "y", "z"],
                       dtype={"cell_label": str})
    ccf = pd.read_csv(f"{a.raw_dir}/{stem}-CCF_ccf_coordinates.csv", usecols=["cell_label", "x", "y", "z", "parcellation_index"],
                      dtype={"cell_label": str}).rename(columns={"x": "cx", "y": "cy", "z": "cz"})
    cells = meta.merge(ccf, on="cell_label", how="left")
    assert cells["cell_label"].is_unique, "duplicate cell labels after the CCF join"
    M = cells[["x", "y", "z"]].to_numpy(np.float64)
    fit = cells["cx"].notna().to_numpy() & (cells["parcellation_index"].fillna(0).to_numpy() > 0)
    C = cells.loc[fit, ["cz", "cy", "cx"]].to_numpy(np.float64)
    mM, mC = M[fit].mean(0), C.mean(0)
    U, _, Vt = np.linalg.svd((M[fit] - mM).T @ (C - mC)); Rm = U @ Vt
    mapped = (M - mM) @ Rm + mC
    resid = np.sqrt(((mapped[fit] - C) ** 2).sum(1))
    fit_info = dict(n_fit=int(fit.sum()), det=float(np.linalg.det(Rm)), rotation=Rm.tolist(), mean_meta=mM.tolist(), mean_ccf=mC.tolist(),
                    residual_mm_median=float(np.median(resid)), residual_mm_p90=float(np.percentile(resid, 90)),
                    per_axis_rms_mm=np.sqrt(((mapped[fit] - C) ** 2).mean(0)).tolist())
    h = ad.read_h5ad(f"{a.raw_dir}/{stem}-raw.h5ad", backed="r")
    row = pd.Index(h.obs_names.astype(str)).get_indexer(cells["cell_label"])
    keep = row >= 0
    print(f"[store] ABCA-{a.animal}: metadata {len(meta):,}, with CCF {int(cells['cx'].notna().sum()):,}, in h5ad {int(keep.sum()):,}", flush=True)
    cells, row, mapped = cells.loc[keep].reset_index(drop=True), row[keep], mapped[keep]
    N, G = len(cells), h.shape[1]
    classes = json.load(open(a.class_list)); enc = {c: i for i, c in enumerate(classes)}
    assert set(cells["class"].astype(str)) <= set(classes), "class missing from the shared list"
    secs = sorted(cells["brain_section_label"].unique()); sid = {s: i for i, s in enumerate(secs)}
    os.makedirs(a.out_dir, exist_ok=True)
    feat = np.lib.format.open_memmap(f"{a.out_dir}/feat.npy", mode="w+", dtype=np.float32, shape=(N, G))
    pos = np.lib.format.open_memmap(f"{a.out_dir}/pos.npy", mode="w+", dtype=np.float32, shape=(N, 3))
    sec = np.lib.format.open_memmap(f"{a.out_dir}/sec.npy", mode="w+", dtype=np.int32, shape=(N,))
    cls = np.lib.format.open_memmap(f"{a.out_dir}/cls.npy", mode="w+", dtype=np.int16, shape=(N,))
    pos[:] = (mapped / ISO).astype(np.float32)
    sec[:] = cells["brain_section_label"].map(sid).to_numpy(np.int32)
    cls[:] = cells["class"].astype(str).map(enc).to_numpy(np.int16)
    for s in range(0, N, a.rows_per_block):
        r = row[s:s + a.rows_per_block]; o = np.argsort(r)
        X = h.X[r[o]]
        X = X.toarray() if sparse.issparse(X) else np.asarray(X)
        X = np.log2(1.0 + X.astype(np.float32))
        out = np.empty_like(X); out[o] = X
        feat[s:s + len(r)] = out
    feat.flush(); pos.flush(); sec.flush(); cls.flush()
    json.dump(dict(animal=stem, n_cells=int(N), n_genes=int(G), gene_names=list(map(str, h.var_names)),
                   sections=secs, section_cells=[int((cells["brain_section_label"] == s_).sum()) for s_ in secs],
                   classes=classes, iso_mm=ISO, axes="metadata coords rigidly mapped to CCF [z, y, x] = [ML, DV, AP], / iso_mm",
                   frame_fit=fit_info, has_ccf=int(cells["cx"].notna().sum()), feature="log2(1 + raw count)",
                   cell_labels_file="cell_labels.txt"), open(f"{a.out_dir}/meta.json", "w"), indent=1)
    cells["cell_label"].to_csv(f"{a.out_dir}/cell_labels.txt", index=False, header=False)
    h.file.close()
    print(f"[store] wrote {a.out_dir}: {N:,} cells x {G} genes, {len(secs)} sections", flush=True)


def slopes(d, raw_dir):
    """Per-axis OLS slope of the CCF coordinate (mm) on the store coordinate (mm), over cells with a CCF region."""
    m = json.load(open(f"{d}/meta.json")); lab = pd.read_csv(f"{d}/cell_labels.txt", header=None, dtype=str)[0]
    pos = np.load(f"{d}/pos.npy").astype(np.float64) * ISO
    ccf = pd.read_csv(f"{raw_dir}/{m['animal']}-CCF_ccf_coordinates.csv", usecols=["cell_label", "x", "y", "z", "parcellation_index"],
                      dtype={"cell_label": str}).set_index("cell_label")
    has = lab.isin(ccf.index).to_numpy(); sub = ccf.loc[lab[has]]; ok = sub["parcellation_index"].to_numpy() > 0
    C, P = sub[["z", "y", "x"]].to_numpy()[ok], pos[has][ok]
    return np.array([np.polyfit(P[:, k], C[:, k], 1)[0] for k in range(3)]), m


def cmd_rescale(a):
    """Scale each axis of a store about its centroid by D = slope(store) / slope(reference store); records D in meta.json."""
    if a.out:
        os.makedirs(a.out, exist_ok=False)
        for name in os.listdir(a.store):
            src = os.path.join(a.store, name)
            if name in ("pos.npy", "meta.json"):
                shutil.copyfile(src, os.path.join(a.out, name))
            else:
                os.symlink(os.path.realpath(src), os.path.join(a.out, name))
    store = a.out or a.store
    s_ref, _ = slopes(a.ref_store, a.raw_dir); s, m = slopes(store, a.raw_dir)
    assert "axis_scale" not in m["frame_fit"], "already rescaled"
    D = s / s_ref
    pos = np.load(f"{store}/pos.npy", mmap_mode="r+"); c = np.asarray(pos, dtype=np.float64).mean(0)
    pos[:] = ((np.asarray(pos, dtype=np.float64) - c) * D + c).astype(np.float32); pos.flush()
    m["frame_fit"]["axis_scale"] = dict(D=D.tolist(), centre_iso=c.tolist(), slopes=s.tolist(), ref_slopes=s_ref.tolist(), ref_store=a.ref_store)
    json.dump(m, open(f"{store}/meta.json", "w"), indent=1)
    print(f"[rescale] {m['animal']}: D {np.round(D, 3).tolist()}", flush=True)


def cmd_rereference(a):
    """Rescale an already rescaled store from its reference's ruler to another reference's ruler, into a new folder."""
    m = json.load(open(f"{a.store}/meta.json")); sc = m["frame_fit"]["axis_scale"]
    s_old_ref = np.asarray(sc["ref_slopes"], dtype=np.float64); c = np.asarray(sc["centre_iso"], dtype=np.float64)
    s_ref, _ = slopes(a.ref_store, a.raw_dir)
    f = s_old_ref / s_ref
    os.makedirs(a.out, exist_ok=False)
    for name in os.listdir(a.store):
        if name not in ("pos.npy", "meta.json"):
            os.symlink(os.path.realpath(os.path.join(a.store, name)), os.path.join(a.out, name))
    pos = np.load(f"{a.store}/pos.npy").astype(np.float64)
    np.save(f"{a.out}/pos.npy", ((pos - c) * f + c).astype(np.float32))
    m["frame_fit"]["axis_scale"] = dict(D=(np.asarray(sc["D"]) * f).tolist(), centre_iso=c.tolist(), slopes=sc["slopes"], ref_slopes=s_ref.tolist(),
                                        ref_store=a.ref_store, rereferenced_from=dict(store=a.store, ref_store=sc["ref_store"], factor=f.tolist()))
    json.dump(m, open(f"{a.out}/meta.json", "w"), indent=1)
    print(f"[rereference] {m['animal']}: factor {np.round(f, 4).tolist()}", flush=True)


def cmd_rowmap(a):
    """Map each row of a test prebuilt to the row of the same cell in that mouse's cell store (int64, written to --out).
    --match labels: join by cell label (the prebuilt's rows are the section-sorted metadata rows found in the h5ad; cell_ID
    indexes them). --match geometry: nearest store cell after carrying the prebuilt positions into the store frame (the
    store's rigid fit, then its per-axis ruler if it has one); cells that claim the same store row are re-assigned, in row
    order, to their nearest free store row within 1e-4."""
    if a.match == "labels":
        import anndata as ad
        m = pd.read_csv(meta_csv(a), usecols=["cell_label", "brain_section_label"], dtype={"cell_label": str})
        m = pd.concat([s for _, s in m.groupby("brain_section_label", sort=True)], axis=0, ignore_index=True)
        obs = pd.Index(ad.read_h5ad(f"{a.raw_dir}/Zhuang-ABCA-{a.animal}-raw.h5ad", backed="r").obs_names.astype(str))
        m = m.loc[obs.get_indexer(m["cell_label"]) >= 0].reset_index(drop=True)
        lab = m["cell_label"].to_numpy()[np.load(f"{a.prebuilt}/cell_ID.npy")]
        row = pd.Index(pd.read_csv(f"{a.store}/cell_labels.txt", header=None, dtype=str)[0]).get_indexer(lab).astype(np.int64)
        assert (row >= 0).all(), "prebuilt cells missing from the store"
    else:
        from scipy.spatial import cKDTree
        pre = np.load(f"{a.prebuilt}/pos.npy").astype(np.float64); st = np.load(f"{a.store}/pos.npy").astype(np.float64)
        ff = json.load(open(f"{a.store}/meta.json"))["frame_fit"]; Rf = np.asarray(ff["rotation"])
        if "axis_scale" in ff:
            c = np.asarray(ff["axis_scale"]["centre_iso"]); D = np.asarray(ff["axis_scale"]["D"])
            mapped = (pre - pre.mean(0)) @ Rf + c
            mapped = (mapped - c) * D + c
        else:
            mapped = (pre - pre.mean(0)) @ Rf + st.mean(0)
        dk, ik = cKDTree(st).query(mapped, k=8, workers=-1)
        row = ik[:, 0].copy(); taken = np.zeros(len(st), bool)
        dup = np.flatnonzero(np.bincount(row, minlength=len(st))[row] > 1); taken[row] = True; taken[row[dup]] = False
        for i in dup:
            j = next(int(r) for r, dd in zip(ik[i], dk[i]) if dd < 1e-4 and not taken[r]); row[i] = j; taken[j] = True
        print(f"[rowmap] max distance {np.linalg.norm(mapped - st[row], axis=1).max():.2e}", flush=True)
    assert len(np.unique(row)) == len(row), "row map is not one-to-one"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.save(a.out, row.astype(np.int64))


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("annotate"); p.add_argument("--raw_dir", required=True); p.add_argument("--out_dir", required=True)
    p = sp.add_parser("classes"); p.add_argument("--raw_dir", required=True); p.add_argument("--annot_dir", required=True)
    p.add_argument("--out", required=True)
    p = sp.add_parser("prebuilt")
    p.add_argument("--animal", required=True, choices=["1", "2", "3", "4"]); p.add_argument("--raw_dir", required=True)
    p.add_argument("--annot_dir", default="", help="folder of the annotate output (mice 3 / 4)")
    p.add_argument("--chunk_max_cells", type=int, default=0, help="0 = one graph per section; > 0 splits larger sections")
    p.add_argument("--coord_stats_json", default="", help="coordinate statistics to normalise with (default: this mouse's own)")
    p.add_argument("--write_coord_stats_json", default="", help="write this mouse's coordinate statistics here")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--out_dir", required=True)
    p = sp.add_parser("mixed")
    p.add_argument("--src", required=True); p.add_argument("--out", required=True)
    p.add_argument("--cells", type=int, default=20000); p.add_argument("--dense_stride", type=int, default=1)
    p.add_argument("--block_k", type=int, default=64); p.add_argument("--block_stride", type=int, default=8); p.add_argument("--block_reps", type=int, default=3)
    p.add_argument("--block_k2", type=int, default=0); p.add_argument("--block2_stride", type=int, default=8); p.add_argument("--block2_reps", type=int, default=2)
    p.add_argument("--volume_reps", type=int, default=12); p.add_argument("--seed", type=int, default=0)
    p = sp.add_parser("iso"); p.add_argument("--src", required=True); p.add_argument("--out", required=True)
    p.add_argument("--coord_stats_json", required=True)
    p = sp.add_parser("store")
    p.add_argument("--animal", required=True, choices=["1", "2", "3", "4"]); p.add_argument("--raw_dir", required=True)
    p.add_argument("--annot_dir", default=""); p.add_argument("--out_dir", required=True); p.add_argument("--class_list", required=True)
    p.add_argument("--rows_per_block", type=int, default=50000)
    p = sp.add_parser("rescale"); p.add_argument("--store", required=True); p.add_argument("--ref_store", required=True)
    p.add_argument("--raw_dir", required=True); p.add_argument("--out", default="", help="write the rescaled copy here instead of in place")
    p = sp.add_parser("rereference"); p.add_argument("--store", required=True); p.add_argument("--ref_store", required=True)
    p.add_argument("--raw_dir", required=True); p.add_argument("--out", required=True)
    p = sp.add_parser("rowmap"); p.add_argument("--animal", required=True, choices=["1", "2", "3", "4"])
    p.add_argument("--match", required=True, choices=["labels", "geometry"]); p.add_argument("--prebuilt", required=True)
    p.add_argument("--store", required=True); p.add_argument("--out", required=True)
    p.add_argument("--raw_dir", default=""); p.add_argument("--annot_dir", default="")
    a = ap.parse_args()
    {"annotate": cmd_annotate, "classes": cmd_classes, "prebuilt": cmd_prebuilt, "mixed": cmd_mixed, "iso": cmd_iso, "store": cmd_store,
     "rescale": cmd_rescale, "rereference": cmd_rereference, "rowmap": cmd_rowmap}[a.cmd](a)


if __name__ == "__main__":
    main()
