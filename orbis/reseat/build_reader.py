"""Reader builder CLI: fits the niche reader on the training specimens and writes one reader npz per test set.

The reader npz holds q32 / q128 (predicted neighbourhood compositions of every test cell, float32), sigma32 / sigma128 (held-out RMS
residual per component), p (the test cells' K32 posteriors), keep (input feature columns) and panel (excluded gene panel).
Training stores are directories with feat.npy and pos.npy (sec.npy for serial sections); the order of --stores matters (the reader
rows are drawn store by store from one random generator). Held-out rows: sections with id % 10 == 0 (--split sections), 20 % of the
400-um xy tiles of the first store (--split tiles), or one named specimen (--split specimen).

usage: python -m orbis.reseat.build_reader --name NAME --stores DIR [DIR ...] (--mm_per_unit F | --um_per_unit F) --panel first|largest|none
       --split sections|tiles|specimen --tests NAME=DIR [NAME=DIR ...] --out_dir OUT [--sections] [--pos_dirs DIR ...]"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True, help="reader name: outputs k32_<name>.npz, reader_<name>_<test>.npz, reader_<name>.json")
ap.add_argument("--stores", nargs="+", required=True, help="training store directories (feat.npy, pos.npy[, sec.npy]), in this order")
ap.add_argument("--pos_dirs", nargs="+", default=None, help="directories holding pos.npy / sec.npy when they differ from --stores (one per store)")
ap.add_argument("--mm_per_unit", type=float, default=None, help="positions in um = pos x mm_per_unit x 1000")
ap.add_argument("--um_per_unit", type=float, default=None, help="positions in um = pos x um_per_unit")
ap.add_argument("--sections", action="store_true", help="serial sections: neighbours within each section (in-plane), sec.npy required")
ap.add_argument("--panel", required=True, choices=["first", "largest", "none"], help="specimen of the Moran gene panel (none: no panel)")
ap.add_argument("--split", required=True, choices=["sections", "tiles", "specimen"], help="held-out rows of the reader fits")
ap.add_argument("--val_specimen", default="", help="--split specimen: held-out store (directory name)")
ap.add_argument("--inner_specimen", default="", help="--split specimen: early-stopping store of the sigma step (directory name)")
ap.add_argument("--tests", nargs="+", required=True, help="NAME=DIR: test sets (DIR holds feat.npy with the training stores' columns)")
ap.add_argument("--chunked_standardiser", action="store_true", help="standardise in chunks of 100,000 rows (C-ordered float32 output)")
ap.add_argument("--out_dir", required=True)
ap.add_argument("--threads", type=int, default=8)
A = ap.parse_args()
assert (A.mm_per_unit is None) != (A.um_per_unit is None), "give exactly one of --mm_per_unit / --um_per_unit"
assert A.split != "sections" or A.sections, "--split sections needs --sections"
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = str(A.threads)
os.environ["NUMPY_MADVISE_HUGEPAGE"] = "0"
import json
import time

import numpy as np
import torch

from orbis.reseat import reader as NR

torch.set_num_threads(A.threads)
NR.WORKERS = A.threads
NR.Gene.chunked = A.chunked_standardiser
P = NR.P


def load_stores():
    """Training stores: name, features (memmap), positions in um, section ids (or None)."""
    pos_dirs = A.pos_dirs or A.stores
    assert len(pos_dirs) == len(A.stores)
    st = []
    for d, pd in zip(A.stores, pos_dirs):
        pos = np.load(f"{pd}/pos.npy").astype(np.float64)
        pos_um = pos * A.mm_per_unit * 1000.0 if A.mm_per_unit is not None else pos * A.um_per_unit
        sec = np.load(f"{pd}/sec.npy").astype(np.int64) if A.sections else None
        if A.sections and A.pos_dirs and os.path.exists(f"{d}/sec.npy"):
            assert np.array_equal(np.load(f"{d}/sec.npy"), np.load(f"{pd}/sec.npy")), d
        st.append(dict(name=os.path.basename(os.path.normpath(d)), feat=np.load(f"{d}/feat.npy", mmap_mode="r"), pos_um=pos_um, sec=sec))
    return st


def main():
    """Panel -> reader rows -> K32 -> targets -> sigma reader on held-out rows -> final reader -> one reader npz per test set."""
    os.makedirs(A.out_dir, exist_ok=True)
    tests_cfg = [t.split("=", 1) for t in A.tests]
    stores = load_stores()
    G = stores[0]["feat"].shape[1]
    assert all(s["feat"].shape[1] == G for s in stores)

    def geo_of(s):
        if A.sections:
            return NR.section_geo(s["pos_um"], s["sec"])[0]
        return NR.Geo(s["pos_um"])
    if A.panel == "none":
        pid = np.zeros(0, np.int64)
        pinfo = dict(rule="none", panel_idx=[])
    else:
        ps = stores[0] if A.panel == "first" else max(stores, key=lambda s: len(s["pos_um"]))
        geo = geo_of(ps)
        nbr = NR.knn_sets(geo, NR.MORAN_K, np.arange(len(ps["pos_um"])))
        I = NR.moran_all(np.array(ps["feat"], dtype=np.float32), nbr)
        order = np.argsort(-I, kind="stable")
        pid = np.sort(order[:NR.PANEL_N])
        pinfo = dict(rule=f"{NR.PANEL_N} highest Moran's I on {ps['name']}, {NR.MORAN_K}-NN, row-normalised, self excluded",
                     panel_idx=pid.tolist(), moran_rank50=float(I[order[NR.PANEL_N - 1]]), moran_rank51=float(I[order[NR.PANEL_N]]),
                     moran_median_all=float(np.median(I)), specimen=ps["name"])
        P(f"panel: {ps['name']} {len(nbr):,} cells; top-50 Moran {I[order[0]]:.3f} .. {I[order[NR.PANEL_N - 1]]:.3f} (51st {I[order[NR.PANEL_N]]:.3f}, median {np.median(I):.3f})")
    keep = np.setdiff1d(np.arange(G), pid)
    ntot = sum(len(s["pos_um"]) for s in stores)
    rg = np.random.default_rng(0)
    rows = []
    for s in stores:
        n = len(s["pos_um"])
        k = int(round(NR.N_READER * n / ntot))
        if ntot <= NR.N_READER:
            rows.append(np.arange(n))
        else:
            rows.append(np.sort(rg.choice(n, min(k, n), replace=False)))
    Fr = np.concatenate([np.asarray(s["feat"][r], dtype=np.float32) for s, r in zip(stores, rows)])
    P(f"{A.name}: stores {[s['name'] for s in stores]} cells {[len(s['pos_um']) for s in stores]}; reader rows {[len(r) for r in rows]}; genes/features {G} "
      f"(panel {len(pid)} excluded -> {len(keep)}); Fr {Fr.nbytes / 1e9:.1f} GB")
    if A.split == "sections":
        secr = np.concatenate([s["sec"][r] for s, r in zip(stores, rows)])
        va = secr % 10 == 0
        inner_of = lambda nv: secr[nv] % 10 == 5
        split_rule = "held-out = sections % 10 == 0; sigma-step inner = sections % 10 == 5"
    elif A.split == "tiles":
        Yt = stores[0]["pos_um"][rows[0]]
        tile = np.floor(Yt[:, 0] / 400).astype(int) * 1000 + np.floor(Yt[:, 1] / 400).astype(int)
        ut = np.unique(tile)
        vt = np.random.default_rng(0).choice(ut, int(round(0.2 * len(ut))), replace=False)
        va = np.isin(tile, vt)
        it = np.random.default_rng(1).choice(np.setdiff1d(ut, vt), int(round(0.2 * len(ut))), replace=False)
        inner_of = lambda nv: np.isin(tile[nv], it)
        split_rule = f"held-out = 20 % of the 400-um xy tiles (seed 0): {len(vt)} / {len(ut)} tiles; sigma-step inner = {len(it)} further tiles (seed 1)"
    else:
        emb = np.concatenate([[s["name"]] * len(r) for s, r in zip(stores, rows)])
        va = emb == A.val_specimen
        inner_of = lambda nv: emb[nv] == A.inner_specimen
        split_rule = f"held-out = specimen {A.val_specimen}; sigma-step inner = specimen {A.inner_specimen}"
    assert va.any() and (~va).any(), "empty held-out or fit rows"
    sub = np.sort(np.random.default_rng(0).choice(len(Fr), min(NR.N_FIT_K32, len(Fr)), replace=False))
    k32 = NR.K32(keep).fit(Fr[sub], seed=0)
    kf = f"{A.out_dir}/k32_{A.name}.npz"
    assert not os.path.exists(kf), f"{kf} exists"
    kmd5 = k32.save(kf)
    P(f"{A.name}: K32 fitted on {len(sub):,} cells: PCA-30 variance {k32.info['pca_var_explained']:.3f}, GMM converged {k32.info['gmm_converged']} in "
      f"{k32.info['gmm_n_iter']} iterations; saved {kf} md5 {kmd5}")
    Y, rad_all = [], {}
    for s, r in zip(stores, rows):
        t0 = time.time()
        p_all = k32.transform(s["feat"])
        geo = geo_of(s)
        (m32, m128), rad, short = NR.env_targets(geo, p_all, r)
        Y.append(np.concatenate([m32, m128], 1))
        rad_all[s["name"]] = dict(radii=rad, short=short, mean_max_posterior=float(p_all.max(1).mean()), component_mass=p_all.sum(0).round(1).tolist())
        P(f"{A.name}: {s['name']} targets: 32-NN radius median {rad[32]['median_um']:.1f} um (p90 {rad[32]['p90_um']:.1f}), 128-NN {rad[128]['median_um']:.1f} um "
          f"(p90 {rad[128]['p90_um']:.1f}); short rows {short}; mean max posterior {p_all.max(1).mean():.3f} ({time.time() - t0:.0f} s)")
        del p_all
    Y = np.concatenate(Y)
    blocks = [np.arange(0, 32), np.arange(32, 64)]
    nv = np.flatnonzero(~va)
    inner = inner_of(nv)
    P(f"{A.name}: sigma reader: fit rows {len(nv):,} (inner early-stopping rows {int(inner.sum()):,}), held-out rows {int(va.sum()):,}")
    pr, pinf = NR.fit_readers(Fr[nv][:, keep], Y[nv], inner, {"heldout": Fr[va][:, keep]}, blocks)
    qh = np.concatenate([NR.simplex(pr[pinf["pick_by_val"][j]]["heldout"][:, b]) for j, b in enumerate(blocks)], 1)
    res = Y[va].astype(np.float64) - qh
    sigma = np.sqrt((res ** 2).mean(0))
    hold = [NR.comp_scores(Y[va][:, b], qh[:, b]) for b in blocks]
    P(f"{A.name}: held-out rows: rho {[round(h['rho'], 3) for h in hold]} R2 {[round(h['R2'], 3) for h in hold]}; sigma r32 median {np.median(sigma[:32]):.4f} "
      f"(min {sigma[:32].min():.4f}, max {sigma[:32].max():.4f}), r128 median {np.median(sigma[32:]):.4f}")
    del pr, qh, res
    tests = {}
    for nm, td in tests_cfg:
        Ft = np.load(f"{td}/feat.npy", mmap_mode="r")
        assert Ft.shape[1] == G, (td, Ft.shape, G)
        tests[nm] = np.asarray(Ft[:, keep], dtype=np.float32)
        P(f"{A.name}: prediction set {nm}: {Ft.shape[0]:,} cells ({td})")
    fr, finf = NR.fit_readers(Fr[:, keep], Y, va, tests, blocks)
    saved = {}
    for nm, td in tests_cfg:
        q = np.concatenate([NR.simplex(fr[finf["pick_by_val"][j]][nm][:, b]) for j, b in enumerate(blocks)], 1).astype(np.float32)
        pt = k32.transform(np.load(f"{td}/feat.npy", mmap_mode="r"))
        qf = f"{A.out_dir}/reader_{A.name}_{nm}.npz"
        assert not os.path.exists(qf), f"{qf} exists"
        np.savez(qf, q32=q[:, :32], q128=q[:, 32:], sigma32=sigma[:32], sigma128=sigma[32:], p=pt, keep=keep, panel=pid)
        saved[nm] = dict(path=qf, md5=NR.md5(qf), n=int(len(q)), mean_max_posterior_test=float(pt.max(1).mean()))
        P(f"{A.name}: saved {qf} md5 {saved[nm]['md5']}: q ({len(q):,} x 64), sigma (64), p under K32_{A.name} (mean max posterior {pt.max(1).mean():.3f})")
    here = os.path.dirname(os.path.abspath(__file__))
    out = dict(name=A.name, stores=[s["name"] for s in stores], tests=[t for t, _ in tests_cfg], n_reader_rows=[int(len(r)) for r in rows], n_features=int(G),
               panel=pinfo, split=split_rule, sections=bool(A.sections), k32=dict(path=kf, md5=kmd5, **k32.info), targets=rad_all,
               sigma32=sigma[:32].tolist(), sigma128=sigma[32:].tolist(), heldout_scores=hold,
               sigma_reader=dict(pick=pinf["pick_by_val"], ridge=pinf["ridge"].get("alpha_counts"), mlp_best_epoch=pinf["mlp"].get("best_epoch")),
               final_reader=dict(pick=finf["pick_by_val"], ridge_alpha=finf["ridge"].get("alpha_counts"), ridge_val=finf["ridge"]["val"], mlp_val=finf["mlp"]["val"],
                                 mlp_best_epoch=finf["mlp"].get("best_epoch")), outputs=saved, threads=A.threads,
               chunked_standardiser=bool(A.chunked_standardiser),
               code=dict(build_reader=NR.md5(os.path.abspath(__file__)), reader=NR.md5(f"{here}/reader.py")))
    jf = f"{A.out_dir}/reader_{A.name}.json"
    json.dump(out, open(jf, "w"), indent=1, default=float)
    P(f"{A.name}: READER DONE -> {jf}")


if __name__ == "__main__":
    main()
