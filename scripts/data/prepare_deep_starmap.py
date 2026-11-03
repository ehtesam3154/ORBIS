"""Deep-STARmap mouse-brain blocks: training cell stores and held-out test prebuilts, one directory per fold.

Data: Deep-STARmap thick-tissue blocks, Zenodo record 16783355 (https://doi.org/10.5281/zenodo.16783355).
Files needed in $DATA_DIR/raw/deep_starmap/: Brain_Deep_STARmap_spatial.csv, Brain_Deep_STARmap_expression_matrix.csv,
Brain_Deep_STARmap_replicate_spatial.csv, Brain_Deep_STARmap_replicate_expression_matrix.csv.
Command: DATA_DIR=... bash scripts/data/prepare_deep_starmap.sh   (outputs in $DATA_DIR/blocks/)

Fold A trains on the main block and tests on the replicate block; fold B the reverse. Positions are voxel indices times the
voxel size (0.32 x 0.32 x 0.70 um), centred at the block median, divided by one unit per fold chosen so that the training
block's RMS radius per axis is 0.163. Features are log2(1 + count) of all 1,017 genes.
Writes <out>/fold{A,B}/: stores/STARmap_<train block>/, prebuilt_infos_train/, prebuilt_test_STARmap_<test block>/,
genes.json, build_summary.json."""
import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

VOXEL_UM = np.array([0.32, 0.32, 0.70])
DATA_RMS = 0.163
BLOCKS = {"main": "Brain_Deep_STARmap", "replicate": "Brain_Deep_STARmap_replicate"}
FOLDS = {"A": ("main", "replicate"), "B": ("replicate", "main")}


def read_blocks(raw):
    """Read both blocks: counts, voxel positions, class labels, log2 features, median-centred positions in um."""
    D, header = {}, {}
    for k, b in BLOCKS.items():
        S = pd.read_csv(f"{raw}/{b}_spatial.csv")
        with open(f"{raw}/{b}_expression_matrix.csv") as fh:
            hl = fh.readline().rstrip("\n")
        header[k] = hl
        E = pd.read_csv(f"{raw}/{b}_expression_matrix.csv", dtype=np.int32)
        X = E.to_numpy()
        assert len(S) == len(X), f"{b}: {len(S)} spatial rows vs {len(X)} expression rows"
        assert X.min() >= 0, f"{b}: negative counts"
        vox = S[["x", "y", "z"]].to_numpy(np.float64)
        assert np.array_equal(vox, np.round(vox)), f"{b}: non-integer voxel positions"
        D[k] = dict(name=b, genes=list(E.columns), X=X, um=vox * VOXEL_UM, vox=vox,
                    main=S["FUSEmap_main_level"].astype(str).to_numpy(), sub=S["FUSEmap_sub_level"].astype(str).to_numpy())
        print(f"[read] {b}: {len(S):,} cells, {X.shape[1]} genes", flush=True)
    genes = D["main"]["genes"]
    assert D["replicate"]["genes"] == genes and header["main"] == header["replicate"], "gene order differs between the blocks"
    assert len(genes) == len(set(genes)) == 1017
    classes = sorted(set(D["main"]["main"]) | set(D["replicate"]["main"]))
    classes_sub = sorted(set(D["main"]["sub"]) | set(D["replicate"]["sub"]))
    ci, si = {c: i for i, c in enumerate(classes)}, {c: i for i, c in enumerate(classes_sub)}
    for k, d in D.items():
        d["cls"] = np.array([ci[c] for c in d["main"]], dtype=np.int64)
        d["cls_sub"] = np.array([si[c] for c in d["sub"]], dtype=np.int64)
        d["feat"] = np.log2(1.0 + d["X"].astype(np.float32)).astype(np.float32)
        d["centre_um"] = np.median(d["um"], 0)
        d["C"] = d["um"] - d["centre_um"]
        lo, hi = d["um"].min(0), d["um"].max(0)
        assert 143.0 - 0.5 <= hi[2] - lo[2] <= 145.0 + 0.5, f"{d['name']}: z span {hi[2] - lo[2]:.1f} um outside 143-145 um"
    assert len(classes) == 19 and len(classes_sub) == 137
    return D, genes, classes, classes_sub


def save_infos(dst, n_cells, name, note, sizes, genes, classes):
    """Write the infos.pkl that the trainer and sampler read (gene count, class decoder, graph sizes)."""
    infos = SimpleNamespace(name=name, num_cell_class=len(classes), num_genes=len(genes),
                            cell_class_decoder={i: c for i, c in enumerate(classes)},
                            num_cell_to_region_mapping_dict={int(s): f"g{i:03d}" for i, s in enumerate(sizes)},
                            input_dims={"node_features_dimensions": len(genes), "diffusion_time_dimensions": 1},
                            output_dims={"node_features_dimensions": len(genes), "diffusion_time_dimensions": 0},
                            pos_dim=3, meta=dict(note=note, n_cells=int(n_cells)))
    torch.save(infos, os.path.join(dst, "infos.pkl"))


def write_fold(out, fold, T, Q, genes, classes, classes_sub, graph):
    """Write one fold: the training store, the infos-only training prebuilt and the test prebuilt."""
    root = os.path.join(out, f"fold{fold}"); os.makedirs(root, exist_ok=True)
    rms_axis = np.sqrt((T["C"] ** 2).mean(0))
    rms = float(np.sqrt((T["C"] ** 2).mean()))
    U = rms / DATA_RMS
    posT = (T["C"] / U).astype(np.float32); posQ = (Q["C"] / U).astype(np.float32)
    span = lambda p: (p.max(0) - p.min(0)).astype(float)
    m_rms_axis = np.sqrt((posT.astype(np.float64) ** 2).mean(0)); m_rms = float(np.sqrt((posT.astype(np.float64) ** 2).mean()))
    tr, te = FOLDS[fold]
    tag_tr = f"STARmap_{tr}"; sd = os.path.join(root, "stores", tag_tr); os.makedirs(sd, exist_ok=True)
    n = len(posT)
    np.save(f"{sd}/pos.npy", posT); np.save(f"{sd}/feat.npy", T["feat"]); np.save(f"{sd}/sec.npy", np.zeros(n, np.int64))
    np.save(f"{sd}/cls.npy", T["cls"]); np.save(f"{sd}/cls_sub.npy", T["cls_sub"])
    with open(f"{sd}/cell_labels.txt", "w") as fh:
        fh.write("\n".join(f"{tag_tr}_{i}" for i in range(n)) + "\n")
    meta = dict(animal=f"Deep-STARmap {T['name']} (Zenodo 16783355)", n_cells=int(n), n_genes=len(genes),
                gene_names=genes, sections=["block"], section_cells=[int(n)], classes=classes, classes_sub=classes_sub,
                iso_mm=U / 1000.0, axes="image [x, y, z] of the block (x, y in the block plane; z = depth), physical um centred at "
                "the median, / U", sec_depth_um=[0.0], sec_gap_um=0.0, sagittal=False, whole_ok=True,
                feature="log2(1 + raw count), all 1,017 genes in the release's column order", platform_id=0,
                voxel_um=VOXEL_UM.tolist(), centre_um=T["centre_um"].tolist(),
                note="continuous tissue: ONE section holds every cell; WHOLE views only")
    json.dump(meta, open(f"{sd}/meta.json", "w"), indent=1)
    dd = os.path.join(root, "prebuilt_infos_train"); os.makedirs(dd, exist_ok=True)
    o = np.random.default_rng(0).permutation(n); parts = np.array_split(o, int(np.ceil(n / graph))); o = np.concatenate(parts)
    sizes = [len(p) for p in parts]
    np.save(f"{dd}/pos.npy", posT[o]); np.save(f"{dd}/feat.npy", T["feat"][o]); np.save(f"{dd}/cell_class.npy", T["cls"][o])
    np.save(f"{dd}/cell_ID.npy", o.astype(np.int64)); np.save(f"{dd}/offsets.npy", np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64))
    json.dump(dict(n_graphs=len(sizes), total_cells=int(n), n_genes=len(genes), pos_dim=3, section_names=[f"g{i:03d}" for i in range(len(sizes))],
                   sizes=sizes, note="infos only (training uses the cell store); random partition of the training block"),
              open(f"{dd}/meta.json", "w"), indent=1)
    save_infos(dd, n, "prebuilt_infos_train", f"infos for the Deep-STARmap fold-{fold} trainer", sizes, genes, classes)
    tag_te = f"STARmap_{te}"; td = os.path.join(root, f"prebuilt_test_{tag_te}"); os.makedirs(td, exist_ok=True)
    m = len(posQ)
    np.save(f"{td}/pos.npy", posQ); np.save(f"{td}/pos_canonical.npy", posQ); np.save(f"{td}/feat.npy", Q["feat"])
    np.save(f"{td}/cell_class.npy", Q["cls"]); np.save(f"{td}/cell_class_sub.npy", Q["cls_sub"])
    np.save(f"{td}/cell_ID.npy", np.arange(m, dtype=np.int64)); np.save(f"{td}/offsets.npy", np.array([0, m], dtype=np.int64))
    json.dump(dict(n_graphs=1, total_cells=int(m), n_genes=len(genes), pos_dim=3, section_names=["block"], sizes=[int(m)],
                   note=f"held-out block {Q['name']}, every cell, table row order; truth in its own imaging frame, centred at its "
                   f"median, in fold {fold}'s training unit", um_per_unit=U, voxel_um=VOXEL_UM.tolist(), centre_um=Q["centre_um"].tolist(),
                   classes=classes, classes_sub=classes_sub), open(f"{td}/meta.json", "w"), indent=1)
    save_infos(td, m, os.path.basename(td), "Deep-STARmap test block", [m], genes, classes)
    json.dump({"hvg": genes}, open(f"{root}/genes.json", "w"))
    summ = dict(fold=fold, train=T["name"], test=Q["name"], U_um_per_unit=U, data_rms=DATA_RMS, iso_mm=U / 1000.0,
                classes=classes, classes_sub=classes_sub, hvg_source=f"{root}/genes.json", voxel_um=VOXEL_UM.tolist(),
                train_rms_um_axis=rms_axis.tolist(), train_rms_um=rms, train_rms_model_axis=m_rms_axis.tolist(), train_rms_model=m_rms,
                train_span_model=span(posT).tolist(), test_span_model=span(posQ).tolist(),
                train_span_um=span(T["um"]).tolist(), test_span_um=span(Q["um"]).tolist(), n_train=int(n), n_test=int(m))
    json.dump(summ, open(f"{root}/build_summary.json", "w"), indent=1)
    print(f"[fold {fold}] unit {U:.2f} um; wrote {sd} ({n:,}), {dd} ({len(sizes)} graphs), {td} ({m:,})", flush=True)


def main():
    """Build both folds."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="folder with the four Brain_Deep_STARmap*.csv files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--graph", type=int, default=20000, help="cells per graph of prebuilt_infos_train")
    args = ap.parse_args()
    D, genes, classes, classes_sub = read_blocks(args.raw)
    for fold, (tr, te) in FOLDS.items():
        write_fold(args.out, fold, D[tr], D[te], genes, classes, classes_sub, args.graph)
    print("done", flush=True)


if __name__ == "__main__":
    main()
