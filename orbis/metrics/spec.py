"""Scoring specification of one test set, built from plain paths and the test card (orbis/metrics/cards/<test>.json).

Test kinds:
  brain   serial whole brains (ABC-atlas mice, Allen 638850), calibrated millimetres: the truth is the test brain's cell
          store positions (in the test prebuilt's row order) x ISO x S_STORE; every layout is first aligned in the training
          ruler (truth / S_REF), multiplied by S_REF, then scored. Fine labels = supertype, contact labels = subclass,
          niche labels = class.
  block   continuous Deep-STARmap blocks: truth = prebuilt pos x mm; fine = sub classes, contact / niche = main classes.
  embryo  Flysta3D embryos: truth = pos_canonical x U (um); views main and lr_folded (headline: lr_folded).
"""
import json
import os
import time

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from orbis.metrics.scorer import (K_MORAN, N_GENES, WORKERS, fold_mid_low, knn_excluding_self, make_fitter, md5_codes, md5_file,
                                  refuse)


def _np(path, mmap=None):
    return np.load(path, mmap_mode=mmap)


def _sections(prebuilt):
    """Section id per cell: graphs merged by section name ('<section>__chunkNNN'), ids = sorted unique names."""
    meta = json.load(open(f"{prebuilt}/meta.json"))
    off = np.load(f"{prebuilt}/offsets.npy")
    names = meta["section_names"]
    base = [n.split("__chunk")[0] for n in names]
    uniq = sorted(set(base))
    sid = np.array([uniq.index(b) for b in base])
    return np.repeat(sid, np.diff(off))


def _abc_labels(rule, test, prebuilt, rows, c, labels_txt, metadata):
    """Subclass and supertype codes in the truth's row order (np.unique of the label strings), from the cell store's
    cell_labels.txt and the atlas cell metadata. rule 'annotation': metadata = [cell metadata with cluster annotation];
    rule 'alias': metadata = [annotation table holding the cluster aliases, cell metadata, cell metadata with annotation]."""
    import pandas as pd
    import torch
    lab = pd.read_csv(labels_txt, header=None, names=["cell_label"], dtype=str).iloc[rows]
    cols = ["cell_label", "class", "subclass", "supertype"]
    if rule == "annotation":
        meta = pd.read_csv(metadata[0], usecols=cols, dtype=str)
    elif rule == "alias":
        LV = ["class", "subclass", "supertype", "cluster"]
        alias = (pd.read_csv(metadata[0], usecols=["cluster_alias"] + LV, dtype=str)
                 .drop_duplicates("cluster_alias").set_index("cluster_alias"))
        meta = pd.read_csv(metadata[1], usecols=["cell_label", "cluster_alias"], dtype=str)
        meta = meta.merge(pd.read_csv(metadata[2], usecols=["cell_label", "class"],
                                      dtype=str), on="cell_label", how="left")
        for lv in ("subclass", "supertype"):
            meta[lv] = meta["cluster_alias"].map(alias[lv])
        meta = meta[cols]
    else:
        refuse(f"label rule {rule} (annotation | alias)")
    df = lab.reset_index(drop=True).merge(meta, on="cell_label", how="left")
    if len(df) != len(c):
        refuse(f"{test}: label merge changed the row count ({len(df)} vs {len(c)})")
    dec = torch.load(f"{prebuilt}/infos.pkl", map_location="cpu", weights_only=False).cell_class_decoder
    agree = float(np.mean(df["class"].astype(str).to_numpy() == np.array([dec[int(k)] for k in c])))
    if agree <= 0.999:
        refuse(f"{test}: fine labels are not in the truth's order (class agreement {agree:.5f})")
    return {lv: np.unique(df[lv].astype(str).to_numpy(), return_inverse=True)[1].ravel().astype(np.int64) for lv in ("subclass", "supertype")}, agree


def _geometric_rows(prebuilt, store_dir):
    """Exact geometric prebuilt -> store row match through the store's frame-fit rotation."""
    pre = np.load(f"{prebuilt}/pos.npy").astype(np.float64)
    st1 = np.load(f"{store_dir}/pos.npy").astype(np.float64)
    Rf = np.asarray(json.load(open(f"{store_dir}/meta.json"))["frame_fit"]["rotation"])
    mapped = (pre - pre.mean(0)) @ Rf + st1.mean(0)
    dk, ik = cKDTree(st1).query(mapped, k=8, workers=WORKERS)
    row = ik[:, 0].copy()
    taken = np.zeros(len(st1), bool)
    dup = np.flatnonzero(np.bincount(row, minlength=len(st1))[row] > 1)
    taken[row] = True
    taken[row[dup]] = False
    for i in dup:
        j = next(int(r) for r, dd in zip(ik[i], dk[i]) if dd < 1e-4 and not taken[r])
        row[i] = j
        taken[j] = True
    if not (np.linalg.norm(mapped - st1[row], axis=1).max() < 1e-4 and len(np.unique(row)) == len(row)):
        refuse("prebuilt -> store row map not exact")
    return row


def morans_knn(Z, V, k=K_MORAN):
    """scanpy morans_i on the row-normalised k-NN graph (self excluded) of Z; V = genes x cells."""
    from scanpy.metrics import morans_i
    nb = knn_excluding_self(Z, k)
    n = len(Z)
    g = sp.csr_matrix((np.full(n * k, 1.0 / k), (np.repeat(np.arange(n), k), nb.ravel())), shape=(n, n))
    return np.asarray(morans_i(g, np.ascontiguousarray(V, dtype=np.float64)))


def select_moran_genes(feat_path, Xg, sec, n_genes=N_GENES, k=K_MORAN):
    """The n_genes genes with the highest Moran's I on the truth: on 8 evenly spaced sections, in-plane (serial data), or on
    100,000 random cells (continuous data). Returns (idx, E float32, Moran's I of the picked genes)."""
    feat = np.load(feat_path, mmap_mode="r")
    G = feat.shape[1]
    if sec is not None:
        secs = np.unique(sec)
        pick = secs[np.linspace(0, len(secs) - 1, 8).round().astype(int)]
        rows = np.flatnonzero(np.isin(sec, pick))
    else:
        rows = np.sort(np.random.default_rng(0).choice(len(Xg), min(100000, len(Xg)), replace=False))
    Esub = np.asarray(feat[rows], dtype=float)
    keep = Esub.std(0) > 0
    Is = np.zeros(G)
    if sec is not None:
        n = 0
        for s in pick:
            m = sec[rows] == s
            Is[keep] += morans_knn(Xg[rows][m][:, :2], Esub[m][:, keep].T, k) * m.sum()
            n += m.sum()
        Is /= n
    else:
        Is[keep] = morans_knn(Xg[rows], Esub[:, keep].T, k)
    idx = np.sort(np.argsort(-Is)[:n_genes])
    return idx, np.asarray(feat[:, idx], dtype=np.float32), Is[idx]


def _gene_matrix(card, test_dir, Xg, sec, cache_dir):
    """Expression of the Moran genes: the card's gene list when it has one, else the selection rule (cached under cache_dir)."""
    feat_path = f"{test_dir}/feat.npy"
    if card.get("moran_gene_idx") is not None:
        idx = np.asarray(card["moran_gene_idx"], dtype=np.int64)
        feat = np.load(feat_path, mmap_mode="r")
        return np.asarray(feat[:, idx], dtype=np.float32), dict(source="test card gene list", idx=[int(i) for i in idx])
    f = f"{cache_dir}/genes_{card['test']}.npz" if cache_dir else None
    if f and os.path.exists(f):
        z = np.load(f)
        return z["E"], dict(source="cache", file=f, md5=md5_file(f), idx=[int(i) for i in z["idx"]])
    t0 = time.time()
    idx, E, I = select_moran_genes(feat_path, Xg, sec)
    if f:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(f, idx=idx, E=E, I_pick=I)
    return E, dict(source="selected (highest Moran's I on the truth)", file=f, md5=None if f is None else md5_file(f), idx=[int(i) for i in idx],
                   seconds=round(time.time() - t0, 1))


def _cal_prepare(X, fine, S_REF, FM, mm):
    """Calibrated brains: a layout (model units) is aligned onto the truth in the training ruler (X / S_REF), then multiplied by S_REF."""
    def prepare(P, X=X, S_REF=S_REF, FM=FM):
        Xr = X / S_REF
        fit = make_fitter(Xr, fine, None if FM is None else fold_mid_low(FM / float(S_REF[0])))
        return fit(P * mm)[0] * S_REF
    return prepare


def dataset_spec(card, test_dir, cache_dir=None, store_dir=None, store_rows=None, cell_metadata=None, need_expression=True):
    """Everything needed to score layouts of one test: truth (native units), labels (md5-checked against the card), views, rules."""
    t0 = time.time()
    ds = card["test"]
    pre = test_dir
    kind = card["kind"]
    pos = _np(f"{pre}/pos.npy")
    c = _np(f"{pre}/cell_class.npy").astype(np.int64)
    sec = _sections(pre) if kind in ("brain", "embryo") else None
    spec = dict(dataset=ds, kind=kind, label=card.get("label"), prebuilt=pre, pos=pos, cell_class=c, n=len(pos), provenance={})
    prov = spec["provenance"]
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    def cached_codes(name, md5, make):
        f = f"{cache_dir}/codes_{ds}_{name}.npy" if cache_dir else None
        if f and os.path.exists(f):
            a = np.load(f)
            if md5_codes(a) == md5:
                return a, "cache"
        a = make()
        if md5 is not None and md5_codes(a) != md5:
            refuse(f"{ds} {name}: derived codes md5 {md5_codes(a)} != test card {md5}")
        if f:
            np.save(f, a)
        return a, "derived"

    if kind == "brain":
        units = "cal"
        mm = float(card["mm_per_unit"])
        cal = card["calibration"]
        if store_dir is None:
            refuse(f"{ds}: the calibrated brain truth needs --store_dir (the test brain's cell store)")
        rule = card["store_rows"]
        if rule == "file":
            if store_rows is None:
                refuse(f"{ds}: --store_rows (prebuilt -> store row map, .npy) is required for this test")
            rows = np.load(store_rows)
            if card.get("store_rows_md5") is not None and md5_codes(rows) != card["store_rows_md5"]:
                refuse(f"{ds}: store row map md5 {md5_codes(rows)} != test card")
        elif rule == "cell_id":
            rows = np.load(f"{pre}/cell_ID.npy")
        else:
            rows, _ = cached_codes("rows", card["store_rows_md5"], lambda: _geometric_rows(pre, store_dir))
        lb = card["labels"]
        if md5_codes(c) != lb["class_md5"]:
            refuse(f"{ds}: cell_class md5 != test card")
        lab = {}

        def make(lv):
            if "_abc" not in lab:
                if not cell_metadata:
                    refuse(f"{ds}: --cell_metadata is required to derive the {lv} labels")
                lab["_abc"] = _abc_labels(lb["rule"], ds, pre, rows, c, f"{store_dir}/cell_labels.txt", cell_metadata)[0]
            return lab["_abc"][lv]
        sub, s1 = cached_codes("subclass", lb["subclass_md5"], lambda: make("subclass"))
        sup, s2 = cached_codes("supertype", lb["supertype_md5"], lambda: make("supertype"))
        prov["labels"] = dict(levels=dict(fine="supertype", contact="subclass", cls="class"), subclass=s1, supertype=s2,
                              md5=dict(class_=md5_codes(c), subclass=md5_codes(sub), supertype=md5_codes(sup)))
        import torch
        dec = torch.load(f"{pre}/infos.pkl", map_location="cpu", weights_only=False).cell_class_decoder
        names = np.array([dec[int(k)] for k in range(int(c.max()) + 1)])[c]
        um = 1000.0
        S_STORE, S_REF = np.asarray(cal["s_store"]), np.asarray(cal["s_ref"])
        ISO = float(cal["iso_mm_per_unit"])
        st_meta = json.load(open(f"{store_dir}/meta.json"))
        if "axis_scale" in st_meta["frame_fit"] and not np.allclose(st_meta["frame_fit"]["axis_scale"]["ref_slopes"], S_STORE):
            refuse(f"{ds}: store ruler != s_store")
        X = np.load(f"{store_dir}/pos.npy").astype(np.float64)[rows] * ISO * S_STORE
        fold, FM = None, None
        if cal.get("fold_at_midline"):
            FM = float(st_meta["midline_ml_iso"]) * ISO * S_STORE[0]
            X[:, 0] = FM - np.abs(X[:, 0] - FM)
            fold = fold_mid_low(FM)
        prepare = _cal_prepare(X, sup, S_REF, FM, mm)
        rule = ("cal: X = store pos[rows] x ISO x S_STORE" + (" folded at m - |x - m|" if FM else "") +
                "; Y = fit(P x mm onto X / S_REF) x S_REF, then the main fit in calibrated mm")
        spec.update(units=units, um_per_unit=um, X=X, prepare=prepare, main_fold=fold, units_rule=rule,
                    labels=dict(fine=sup, contact=sub, cls=c), fine_all=sup, cls_all=c, cls_strat=c, class_names=names,
                    views=["main"], headline_view="main", taus_um=(50, 100, 200, 400), iou_grids_um=(200,), iou_mode="sections",
                    lddt_u_um=None, lddt_seed=20261014, landmarks=True, _cal=(S_REF, FM, mm))
        spec["sections"] = sec
        spec["section_frame"] = "fit"
        Xg = pos.astype(np.float64) * mm
    elif kind == "block":
        mm = float(card["mm_per_unit"])
        cs = _np(f"{pre}/cell_class_sub.npy").astype(np.int64)
        for nm, a in (("main", c), ("sub", cs)):
            if md5_codes(a) != card["labels"][f"{nm}_md5"]:
                refuse(f"{ds}: {nm} labels md5 != test card")
        X = pos.astype(np.float64) * mm
        spec.update(units="raw", um_per_unit=1000.0, X=X, prepare=(lambda P: P * mm), main_fold=None,
                    units_rule="X = prebuilt pos x mm; Y = P x mm",
                    labels=dict(fine=cs, contact=c, cls=c), fine_all=cs, cls_all=c, cls_strat=c, class_names=None, views=["main"], headline_view="main",
                    taus_um=(50, 100, 200, 400), iou_grids_um=(50, 200), iou_mode="volume", lddt_u_um=None,
                    lddt_seed=20261014, landmarks=False, sections=None, section_frame=None)
        prov["labels"] = dict(levels=dict(fine="sub", contact="main", cls="main"), md5=dict(main=md5_codes(c), sub=md5_codes(cs)))
        Xg = X
    elif kind == "embryo":
        U = float(card["um_per_unit"])
        can = _np(f"{pre}/pos_canonical.npy")
        if md5_codes(c) != card["labels"]["class_md5"]:
            refuse(f"{ds}: labels md5 != test card")
        X = can.astype(np.float64) * U
        Xsec = pos.astype(np.float64) * U
        mY, mX = X.mean(0), Xsec.mean(0)
        Uu, _, Vt = np.linalg.svd((X - mY).T @ (Xsec - mX))
        Rs = Uu @ Vt
        ts = mX - mY @ Rs
        resid = float(np.abs(X @ Rs + ts - Xsec).max())
        if resid > 1e-3:
            refuse(f"{ds}: pos_canonical is not a rigid image of pos.npy (max residual {resid} um)")
        spec.update(units="um", um_per_unit=1.0, X=X, prepare=(lambda P: P * U), main_fold=None,
                    units_rule="X = pos_canonical x U (anterior-posterior / dorsal-ventral / left-right axes, um); Y = P x U",
                    labels=dict(fine=c, contact=c, cls=c), fine_all=c, cls_all=c, cls_strat=c, class_names=None,
                    views=["main", "lr_folded"], headline_view="lr_folded",
                    lr_fold=dict(axis=2, centre="median", mid=float(np.median(X[:, 2])),
                                 rule="|LR - median LR| of each layout ('mid' = the truth's median)"),
                    taus_um=(5, 10, 20, 40), iou_grids_um=(20,), iou_mode="sections",
                    lddt_u_um=float(card["lddt_u_um"]), lddt_seed=int(card["lddt_seed"]), landmarks=False, sections=sec,
                    section_frame=(Rs, ts))
        prov["section_frame"] = dict(rule="rigid map pos_canonical -> pos.npy (section frame)", max_residual_um=resid)
        prov["labels"] = dict(levels=dict(fine="Annotation_1", contact="Annotation_1", cls="Annotation_1"), md5=md5_codes(c))
        prov["lddt"] = dict(u_um=float(card["lddt_u_um"]), source="test card lddt_u_um")
        Xg = Xsec / 1000.0
    else:
        refuse(f"test kind {kind} (brain | block | embryo)")
    if need_expression:
        E, einfo = _gene_matrix(card, pre, Xg, sec, cache_dir)
        spec["E"] = E
        prov["moran_genes"] = einfo
    prov["seconds"] = round(time.time() - t0, 1)
    return spec


def restrict(spec, keep):
    """The specification of a subset of the test cells (keep = sorted row indices into the test prebuilt): every per-cell field
    restricted, the calibrated alignment rebuilt on the kept cells, the embryo midline recomputed on them."""
    keep = np.asarray(keep, dtype=np.int64)
    N = int(spec["n"])
    if not (len(keep) > 0 and np.all(np.diff(keep) > 0) and keep[0] >= 0 and keep[-1] < N):
        refuse("keep: not sorted unique rows of the test prebuilt")
    S = dict(spec)
    for k in ("pos", "cell_class", "X", "fine_all", "cls_all", "cls_strat"):
        S[k] = np.asarray(spec[k])[keep]
    S["labels"] = {k: np.asarray(v)[keep] for k, v in spec["labels"].items()}
    for k in ("class_names", "E", "sections"):
        if spec.get(k) is not None:
            S[k] = np.asarray(spec[k])[keep]
    S["n"] = int(len(keep))
    if "lr_fold" in spec:
        S["lr_fold"] = dict(spec["lr_fold"], mid=float(np.median(S["X"][:, 2])))
    if spec.get("_cal") is not None:
        S_REF, FM, mm = spec["_cal"]
        S["prepare"] = _cal_prepare(S["X"], S["labels"]["fine"], S_REF, FM, mm)
    S["provenance"] = dict(spec["provenance"], subset=dict(n_kept=int(len(keep)), n_full=N, keep_md5=md5_codes(keep)))
    return S
