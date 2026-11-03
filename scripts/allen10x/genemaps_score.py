#!/usr/bin/env python
"""Gene maps rebuilt from placed 10x cells, scored against the MERFISH reference maps (Spearman over voxels, mean over the frozen genes)
in three ways: fill (whole brain, uncovered voxels take the gene's mean), own (each layout on the voxels it covers), common (two
layouts on the voxels both cover).  Rows: ours_readout, baseline_readout, their within-subclass permutations, global random placement,
type-conditional random placement; paired differences with a whole-donor bootstrap (B = 200).  Run genemaps_genes.py and
genemaps_fields.py first.  usage: genemaps_score.py [--test] [--B 200] [--workers 24]"""
import argparse, json, os, time
from types import SimpleNamespace
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
import pandas as pd
from multiprocessing import Pool

import genemaps_fields as E3
import paths as PA
from genemaps_fields import APP, SUP, CACHE, EV, sums_at_mask, field, corr_cols, rankdata, log, md5, FILL_MIN, GATE_CEIL, GATE_RAND

FIELDS = f"{CACHE}/fields"
XQ = f"{CACHE}/genemaps_10x_norm_primary.npz"
ROWS = ["ours_readout", "baseline_readout", "ours_readout_perm", "baseline_readout_perm", "global_random", "typecond_random"]
COMPS = [("ours_readout", "baseline_readout"), ("ours_readout", "global_random"), ("ours_readout", "typecond_random"),
         ("ours_readout", "ours_readout_perm"), ("baseline_readout", "baseline_readout_perm")]


def combine(parts, nb, ix):
    c = sum(parts[i][0] for i in ix); n = sum(parts[i][1] for i in ix)
    return c, n, sum(parts[i][2] * nb[i] for i in ix) / sum(nb[i] for i in ix)


def ceiling_c(ref):
    out = f"{CACHE}/genemaps_ceiling.npz"
    if os.path.exists(out):
        return dict(np.load(out))
    z = np.load(ref); mask = z["mask"]
    grid, fz, Xs, Ps, ndrop = E3.read_zhuang()
    assert list(fz.ensembl_id) == list(z["ensembl"])
    parts = [sums_at_mask(grid, mask, P, Xb, np.ones(Xb.shape[0])) for P, Xb in zip(Ps, Xs)]
    nb = [Xb.shape[0] for Xb in Xs]
    pooled, _ = field(*combine(parts, nb, [0, 1, 2, 3]))
    assert np.allclose(pooled, z["ref"], rtol=1e-4, atol=1e-6), "pooled Zhuang field differs from the frozen reference"
    a, b = combine(parts, nb, [0]), combine(parts, nb, [1, 2, 3])
    idx = np.flatnonzero((a[0] >= FILL_MIN) & (b[0] >= FILL_MIN))
    ra, rb = a[1][idx] / a[0][idx, None], b[1][idx] / b[0][idx, None]
    res = dict(sp=corr_cols(rankdata(ra, axis=0), rankdata(rb, axis=0)), pe=corr_cols(ra, rb), common_frac=np.array(len(idx) / len(mask)),
               cov=np.array([(a[0] >= FILL_MIN).mean(), (b[0] >= FILL_MIN).mean()]))
    np.savez_compressed(out, **res)
    log(f"ceiling (common voxels): macro Spearman {np.nanmean(res['sp']):.4f}, Pearson {np.nanmean(res['pe']):.4f}, common frac {len(idx) / len(mask):.3f}")
    return res


G_ = {}


def init_worker(ref, xq):
    E3.init_worker(ref, xq)
    G_.update(E3.G_)
    lab = np.load(f"{APP}/pool_eval_labels.npz", allow_pickle=True)
    G_["sub"], G_["donor"] = lab["subclass"], lab["donor"]
    pcal = np.load(f"{SUP}/support_pos_cal_mm.npy"); brain = np.load(f"{SUP}/support_brain.npy"); ssub = np.load(f"{SUP}/support_subclass.npy")
    G_["pcal"] = pcal
    G_["sel5"] = np.flatnonzero(np.isin(brain, (0, 1, 2, 3, 4)))
    selz = np.flatnonzero(np.isin(brain, (0, 1, 2, 3)))
    sc = ssub[selz]; order = np.argsort(sc, kind="stable"); nsub = int(ssub.max()) + 1
    bb = np.searchsorted(sc[order], np.arange(nsub + 1))
    G_["refs"] = SimpleNamespace(n_sub=nsub, rows=[selz[order[bb[s]:bb[s + 1]]] for s in range(nsub)], sup=SimpleNamespace(pos_cal=pcal))
    G_["crow"] = {c: np.load(f"{APP}/cohorts/primary{c}/rows.npy") for c in [1, 2, 3, 4, 5]}


def place(kind, c, s=None, sd=None):
    if kind in ("global_random", "typecond_random"):
        r = G_["crow"][c]
        if kind == "global_random":
            idx = np.random.default_rng(sd).choice(len(G_["sel5"]), len(r), replace=False)
            return G_["pcal"][G_["sel5"][idx]].astype(np.float64), r
        return EV.ref_type_conditional(G_["refs"], G_["sub"][r], sd)[0].astype(np.float64), r
    arm = kind.replace("_perm", "")
    L = np.load(f"{APP}/layouts/{arm}_primary{c}_s{s}.npz")
    Yc = L["Y_mm"].astype(np.float64) * EV.S_CAL; r = L["rows"]
    if kind.endswith("_perm"):
        Yc = EV.ref_within_type_permutation(Yc, G_["sub"][r], sd)
    return Yc, r


def lay(pos, rows, w=None):
    ii = G_["rowpos"].get_indexer(rows)
    w = np.ones(len(rows)) if w is None else np.asarray(w, np.float64)
    keep = np.isfinite(pos).all(1) & G_["ok"][ii]
    return sums_at_mask(G_["grid"], G_["mask"], pos[keep], G_["Xq"][ii[keep]], w[keep])


def s_fill(Lt):
    fld, ff = field(*Lt)
    s, p = E3.score(fld, G_["ref"], G_["ref_rank"])
    return s, p, ff


def _on(Lt, idx, rR, R):
    raw = Lt[1][idx] / Lt[0][idx, None]
    return corr_cols(rankdata(raw, axis=0), rR), corr_cols(raw, R)


def s_own(Lt):
    idx = np.flatnonzero(Lt[0] >= FILL_MIN)
    R = G_["ref"][idx]
    s, p = _on(Lt, idx, rankdata(R, axis=0), R)
    return s, p, len(idx) / len(Lt[0])


def s_common(A, B):
    idx = np.flatnonzero((A[0] >= FILL_MIN) & (B[0] >= FILL_MIN))
    R = G_["ref"][idx]; rR = rankdata(R, axis=0)
    sa, pa = _on(A, idx, rR, R); sb, pb = _on(B, idx, rR, R)
    return sa, sb, pa, pb, len(idx) / len(A[0])


def gate_job(args):
    c, sd = args
    P = lay(*place("global_random", c, None, sd))
    return (c, sd), (s_fill(P), s_own(P))


def run_job(args):
    c, s, rseeds = args
    out, cnts, M = {}, {}, {}
    for arm in ("ours_readout", "baseline_readout"):
        M[arm] = lay(*place(arm, c, s))
        out[("fill", arm, 0)] = s_fill(M[arm]); out[("own", arm, 0)] = s_own(M[arm])
        cnt, num, gm = M[arm]
        cov = cnt >= FILL_MIN
        fld = np.where(cov[:, None], num / np.where(cov, cnt, 1.0)[:, None], np.nan).astype(np.float32)
        np.savez(f"{FIELDS}/{arm}_primary{c}_s{s}.npz", cnt=cnt.astype(np.float32), field_nofill=fld, gmean=gm, mask=G_["mask"])
        cnts[f"{arm}_c{c}_s{s}"] = cnt.astype(np.float32)
    O = M["ours_readout"]
    out[("common", "ours_readout|baseline_readout", 0)] = s_common(O, M["baseline_readout"])
    for sd in rseeds:
        for arm in ("ours_readout", "baseline_readout"):
            P = lay(*place(arm + "_perm", c, s, sd))
            out[("fill", arm + "_perm", sd)] = s_fill(P); out[("own", arm + "_perm", sd)] = s_own(P)
            out[("common", f"{arm}|{arm}_perm", sd)] = s_common(M[arm], P)
        for kind in ("global_random", "typecond_random"):
            P = lay(*place(kind, c, None, sd))
            out[("fill", kind, sd)] = s_fill(P); out[("own", kind, sd)] = s_own(P)
            out[("common", f"ours_readout|{kind}", sd)] = s_common(O, P)
            cnts[f"{kind}_c{c}_sd{sd}"] = P[0].astype(np.float32)
    return (c, s), out, cnts


def boot_job(args):
    b, c, wmap, seeds = args

    def wl(kind, s=None, sd=None):
        pos, rows = place(kind, c, s, sd)
        return lay(pos, rows, wmap[G_["donor"][rows]])
    T = wl("typecond_random", sd=EV.REF_SEEDS[b % 10])
    tf = float(np.nanmean(s_fill(T)[0]))
    out = {}
    for s in seeds:
        O, L = wl("ours_readout", s), wl("baseline_readout", s)
        co, ct = s_common(O, L), s_common(O, T)
        out[s] = dict(fill_o=float(np.nanmean(s_fill(O)[0])), fill_l=float(np.nanmean(s_fill(L)[0])), fill_t=tf,
                      com_ol_o=float(np.nanmean(co[0])), com_ol_l=float(np.nanmean(co[1])), com_ot_o=float(np.nanmean(ct[0])), com_ot_t=float(np.nanmean(ct[1])))
    return (b, c), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    cohorts = [1] if a.test else [1, 2, 3, 4, 5]
    seeds = [1000] if a.test else [1000, 1001, 1002]
    rseeds = list(EV.REF_SEEDS)[:2] if a.test else list(EV.REF_SEEDS)
    B = 4 if a.test else a.B
    res = PA.RESULTS + ("/test" if a.test else "")
    os.makedirs(res, exist_ok=True); os.makedirs(FIELDS, exist_ok=True)
    ref = E3.stage_a()
    z = np.load(ref); ens, sym = z["ensembl"], z["symbol"]
    ceil = ceiling_c(ref)
    assert os.path.exists(XQ), XQ
    P = Pool(a.workers, initializer=init_worker, initargs=(ref, XQ))
    t0 = time.time()
    gout = dict(P.imap_unordered(gate_job, [(c, sd) for c in cohorts for sd in rseeds], chunksize=1))
    g_fill = float(np.mean([np.nanmean(v[0][0]) for v in gout.values()])); g_own = float(np.mean([np.nanmean(v[1][0]) for v in gout.values()]))
    c_fill, c_com = float(np.nanmean(z["ceil_sp"])), float(np.nanmean(ceil["sp"]))
    msg = (f"SANITY: ceiling fill {c_fill:.4f} / common {c_com:.4f} (gate >= {GATE_CEIL}); global random fill {g_fill:.4f} / own-coverage {g_own:.4f} "
           f"(gate |.| <= {GATE_RAND}); {time.time() - t0:.0f} s")
    log(msg)
    if min(c_fill, c_com) < GATE_CEIL or max(abs(g_fill), abs(g_own)) > GATE_RAND:
        P.close()
        log("STOPPED at the sanity gate; no model row computed")
        return
    t0 = time.time()
    R, CN = {}, {}
    for cs, out, cnts in P.imap_unordered(run_job, [(c, s, rseeds) for c in cohorts for s in seeds], chunksize=1):
        R[cs] = out; CN.update(cnts)
        log(f"  run {cs} done ({time.time() - t0:.0f} s)")
    np.savez_compressed(f"{FIELDS}/{'test_' if a.test else ''}counts_all.npz", mask=z["mask"], **CN)
    log(f"unweighted rows done in {time.time() - t0:.0f} s")
    lab = np.load(f"{APP}/pool_eval_labels.npz", allow_pickle=True); donor = lab["donor"]
    crow = {c: np.load(f"{APP}/cohorts/primary{c}/rows.npy") for c in cohorts}
    don_all = np.unique(donor[np.concatenate([crow[c] for c in cohorts])])
    rng = np.random.default_rng(27001)
    bj = []
    for b in range(B):
        cnt = np.bincount(rng.integers(0, len(don_all), len(don_all)), minlength=len(don_all))
        wmap = np.zeros(donor.max() + 1); wmap[don_all] = cnt
        bj += [(b, c, wmap, seeds) for c in cohorts]
    t0 = time.time()
    BO = dict(P.imap_unordered(boot_job, bj, chunksize=1))
    P.close()
    log(f"bootstrap ({B} reps x {len(cohorts)} cohorts) done in {time.time() - t0:.0f} s")
    runs = [(c, s) for c in cohorts for s in seeds]
    boot = {k: [] for k in ("fill_baseline", "fill_tc", "common_baseline", "common_tc")}
    for b in range(B):
        d = [BO[(b, c)][s] for c, s in runs]
        boot["fill_baseline"].append(np.mean([x["fill_o"] - x["fill_l"] for x in d])); boot["fill_tc"].append(np.mean([x["fill_o"] - x["fill_t"] for x in d]))
        boot["common_baseline"].append(np.mean([x["com_ol_o"] - x["com_ol_l"] for x in d])); boot["common_tc"].append(np.mean([x["com_ot_o"] - x["com_ot_t"] for x in d]))
    ci = {k: np.percentile(v, [2.5, 97.5]) for k, v in boot.items()}
    ks = lambda row: [0] if row in ("ours_readout", "baseline_readout") else rseeds
    sd1 = lambda v: float(v.std(ddof=1)) if len(v) > 1 else np.nan

    def pr(sc, row, i):
        f = (lambda v: float(np.nanmean(v))) if i < 2 else float
        return np.array([np.mean([f(R[cs][(sc, row, sd)][i]) for sd in ks(row)]) for cs in runs])

    def pg(sc, row, i):
        return np.nanmean(np.stack([np.nanmean(np.stack([R[cs][(sc, row, sd)][i] for sd in ks(row)]), 0) for cs in runs]), 0)

    def cpr(A, Bn, i):
        return np.array([np.mean([float(np.nanmean(R[cs][("common", f"{A}|{Bn}", sd)][i])) if i < 4 else R[cs][("common", f"{A}|{Bn}", sd)][4]
                                  for sd in ks(Bn if Bn != "baseline_readout" else "ours_readout")]) for cs in runs])

    def cpg(A, Bn, i):
        k = ks(Bn if Bn != "baseline_readout" else "ours_readout")
        return np.nanmean(np.stack([np.nanmean(np.stack([R[cs][("common", f"{A}|{Bn}", sd)][i] for sd in k]), 0) for cs in runs]), 0)
    gt = pd.DataFrame(dict(ensembl_id=ens, gene_symbol=sym))
    S = [dict(scoring="fill", kind="ceiling", row="Zhuang mouse 1 vs mice 2-4", a_macro_spearman=c_fill, a_macro_pearson=float(np.nanmean(z["ceil_pe"])),
              note=f"fill frac {z['ceil_fill'].round(3).tolist()}"),
         dict(scoring="common", kind="ceiling", row="Zhuang mouse 1 vs mice 2-4", a_macro_spearman=c_com, a_macro_pearson=float(np.nanmean(ceil["pe"])),
              common_voxel_frac=float(ceil["common_frac"]), note=f"coverage m1 / m2-4 {ceil['cov'].round(3).tolist()}; own = common for the ceiling")]
    gt["spearman_ceiling_fill"] = z["ceil_sp"]; gt["spearman_ceiling_common"] = ceil["sp"]
    for sc in ("fill", "own"):
        for row in ROWS:
            v = pr(sc, row, 0)
            S.append(dict(scoring=sc, kind="row", row=row, a_macro_spearman=v.mean(), sd_over_runs=sd1(v), a_macro_pearson=pr(sc, row, 1).mean(),
                          coverage=pr("own", row, 2).mean(), fill_frac=pr("fill", row, 2).mean(), n_runs=len(runs)))
            gt[f"{sc}_spearman_{row}"] = pg(sc, row, 0)
    cimap = {("fill", "baseline_readout"): "fill_baseline", ("fill", "typecond_random"): "fill_tc", ("common", "baseline_readout"): "common_baseline",
             ("common", "typecond_random"): "common_tc"}
    for sc in ("common", "fill", "own"):
        for A, Bn in COMPS:
            if sc == "common":
                va, vb = cpr(A, Bn, 0), cpr(A, Bn, 1); ga, gb = cpg(A, Bn, 0), cpg(A, Bn, 1)
                gt[f"common_spearman_{A}_vs_{Bn}__{A}"] = ga; gt[f"common_spearman_{A}_vs_{Bn}__{Bn}"] = gb
                frac = cpr(A, Bn, 4).mean()
            else:
                va, vb = pr(sc, A, 0), pr(sc, Bn, 0); ga, gb = pg(sc, A, 0), pg(sc, Bn, 0); frac = np.nan
            d, dg = va - vb, ga - gb
            k = cimap.get((sc, Bn)) if A == "ours_readout" else None
            S.append(dict(scoring=sc, kind="comparison", row=f"{A} - {Bn}", a_macro_spearman=va.mean(), b_macro_spearman=vb.mean(), diff=d.mean(),
                          sd_over_runs=sd1(d), ci95_lo=ci[k][0] if k else np.nan, ci95_hi=ci[k][1] if k else np.nan, B=B if k else np.nan,
                          runs_a_higher=int((d > 0).sum()), n_runs=len(runs), genes_a=int((dg > 0).sum()), genes_b=int((dg < 0).sum()),
                          genes_tie=int((dg == 0).sum()), common_voxel_frac=frac))
    SD = pd.DataFrame(S)
    SD.to_csv(f"{res}/genemaps_summary.csv", index=False); gt.to_csv(f"{res}/genemaps_genes.csv", index=False)
    SD[SD.scoring == "fill"].to_csv(f"{res}/genemaps_fill_summary.csv", index=False)
    gt[["ensembl_id", "gene_symbol", "spearman_ceiling_fill"] + [c for c in gt.columns if c.startswith("fill_")]].to_csv(f"{res}/genemaps_fill_genes.csv", index=False)
    json.dump(dict(runs=runs, boot=boot, ref=ref, ref_md5=md5(ref), xq=XQ, xq_md5=md5(XQ), gate=msg, test=a.test,
                   per_run={f"{sc}|{row}": pr(sc, row, 0).tolist() for sc in ("fill", "own") for row in ROWS},
                   per_run_common={f"{A}|{Bn}": [cpr(A, Bn, 0).tolist(), cpr(A, Bn, 1).tolist()] for A, Bn in COMPS}),
              open(f"{res}/genemaps_runs.json", "w"), default=float)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        log("\n" + SD.drop(columns=["note"]).to_string())


if __name__ == "__main__":
    main()
