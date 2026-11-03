#!/usr/bin/env python
"""The dissociated-cell application: cohorts -> (sampler outputs) -> layouts -> freeze -> evaluation labels -> evaluation.

subcommands:
  pool_unit                       dissection unit of every pool cell (needed only to draw the region-balanced cohorts)
  cohorts  --kind primary|strat   rows / features / ids / manifest of the five cohorts of a kind
  overlap                         pairwise overlaps of the ten cohorts
  layout   --kind K               label-free pose readout of every sampler output -> layouts/<arm>_<kind><c>_s<seed>.npz
  freeze   --kind K               md5 of every sampler output and layout of a kind (the evaluator refuses to run without it)
  labels                          evaluation labels of every pool cell (dissection region, unit, donor, subclass)
  evaluate --kind K --freeze_md5 MD5
Sampler outputs are raw/ours_<kind><c>_s<seed>.npz (P in the model's units, cell_ID) and raw/baseline_<kind><c>_s<seed>.npz
(P_norm, graph_id, cell_ID, sidecar .json with train_unit_EXT)."""
import argparse
import hashlib
import json
import os
import time

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
import numpy as np

import endpoints as EV
import units as EU
import paths as P
import pose_readout as PR

HOST = os.uname().nodename
APP = P.APP
POOL = P.POOL
MAPPED = P.MAPPED
SUPPORT = P.SUPPORT
LANDMARKS = P.LANDMARKS
N_POOL, N_COHORT = 2_341_350, 100_000
SEEDS = (1000, 1001, 1002)
C_TRAIN = np.array([3.9245159881199005, 3.8842347851115746, 7.38013117305142])
ARMS = ("ours_raw", "ours_readout", "baseline_readout")


def t():
    return time.strftime("%H:%M:%S")


def md5f(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def md5a(a):
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def jdump(o, path):
    def conv(x):
        if isinstance(x, dict):
            return {str(k): conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        if isinstance(x, np.ndarray):
            return conv(x.tolist())
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        return x
    with open(path, "w") as f:
        json.dump(conv(o), f, indent=1)


def pool_labels():
    lab = np.load(f"{POOL}/cell_labels.npy")
    assert len(lab) == N_POOL, len(lab)
    return lab


def build_pool_unit():
    import pandas as pd
    lab = pool_labels()
    df = pd.read_csv(f"{POOL}/EVAL_ONLY_cell_metadata.csv.gz", usecols=["cell_label", "region_of_interest_acronym"], dtype=str)
    assert len(df) == N_POOL and (df.cell_label.to_numpy() == lab).all(), "EVAL_ONLY row order != pool order"
    u = df.region_of_interest_acronym.map(EU.ROI_TO_UNIT)
    assert u.notna().all(), sorted(df.region_of_interest_acronym[u.isna()].unique())
    unit = u.map({n: i for i, n in enumerate(EU.UNIT_NAMES)}).to_numpy().astype(np.int8)
    os.makedirs(f"{APP}/cohorts", exist_ok=True)
    np.save(f"{APP}/cohorts/pool_unit.npy", unit)
    print(f"[{t()}] pool_unit: {np.bincount(unit, minlength=23).tolist()}", flush=True)
    return unit


def draw_primary(c, remaining):
    return np.sort(np.random.default_rng(25000 + c).choice(remaining, N_COHORT, replace=False))


def draw_strat(c, pool_unit, used, quotas):
    rng = np.random.default_rng(25000 + c)
    picks = []
    for r in range(23):
        avail = np.flatnonzero((pool_unit == r) & ~used)
        picks.append(rng.choice(avail, int(quotas[r]), replace=False))
    return np.sort(np.concatenate(picks))


def build_cohorts(kind):
    lab_md5 = md5f(f"{POOL}/cell_labels.npy")
    mapped_sums = {l.split()[1]: l.split()[0] for l in open(f"{MAPPED}/MD5SUMS.txt") if l.strip()}
    assert mapped_sums["cell_labels.npy"] == lab_md5, "mapped_v3 rows are not in pool order"
    X = np.load(f"{MAPPED}/X_pca_harmony_f32.npy", mmap_mode="r")
    assert X.shape == (N_POOL, 200) and X.dtype == np.float32
    used = np.zeros(N_POOL, dtype=bool)
    if kind == "strat":
        pool_unit = np.load(f"{APP}/cohorts/pool_unit.npy") if os.path.exists(f"{APP}/cohorts/pool_unit.npy") else build_pool_unit()
        quotas = np.array(json.load(open(f"{P.FROZEN}/strat_quotas.json"))["n_final"])
        assert quotas.sum() == N_COHORT
        quo_md5 = md5f(f"{P.FROZEN}/strat_quotas.json")
    names = [f"sPC{i}" for i in range(1, 201)]
    for c in range(1, 6):
        d = f"{APP}/cohorts/{kind}{c}"
        os.makedirs(d, exist_ok=True)
        rows = draw_primary(c, np.flatnonzero(~used)) if kind == "primary" else draw_strat(c, pool_unit, used, quotas)
        assert len(rows) == N_COHORT and len(np.unique(rows)) == N_COHORT and not used[rows].any()
        used[rows] = True
        F = np.ascontiguousarray(X[rows])
        assert F.dtype == np.float32 and np.isfinite(F).all()
        np.save(f"{d}/rows.npy", rows.astype(np.int64))
        np.save(f"{d}/ids.npy", rows.astype(np.int64))
        np.save(f"{d}/features.npy", F)
        json.dump(names, open(f"{d}/names.json", "w"))
        man = dict(kind=kind, cohort=c, seed=25000 + c, n=int(len(rows)), rows_md5=md5f(f"{d}/rows.npy"), features_md5=md5f(f"{d}/features.npy"),
                   pool_cell_labels_md5=lab_md5, mapped_features_file="X_pca_harmony_f32.npy", mapped_features_md5=mapped_sums["X_pca_harmony_f32.npy"],
                   rule=("primary: rows = sort(default_rng(25000 + c).choice(remaining, 100000, replace=False)), remaining = pool rows not used by cohorts < c; no label read"
                         if kind == "primary" else
                         "strat: rng = default_rng(25000 + c); for the 23 units in order: rng.choice(sorted pool rows of unit r not used by earlier stratified cohorts, n_r, replace=False); sorted union; "
                         f"n_r from strat_quotas.json md5 {quo_md5}"),
                   created=time.strftime("%F %T"), host=HOST, script_md5=md5f(os.path.abspath(__file__)))
        jdump(man, f"{d}/manifest.json")
        print(f"[{t()}] {kind}{c}: {len(rows):,} rows, rows md5 {man['rows_md5']}, features md5 {man['features_md5']}", flush=True)
    assert used.sum() == 5 * N_COHORT


def cohort_overlap():
    rows = {f"{k}{c}": np.load(f"{APP}/cohorts/{k}{c}/rows.npy") for k in ("primary", "strat") for c in range(1, 6)}
    names = list(rows)
    M = np.zeros((len(names), len(names)), dtype=np.int64)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            M[i, j] = len(np.intersect1d(rows[a], rows[b], assume_unique=True))
    for k in ("primary", "strat"):
        allr = np.concatenate([rows[f"{k}{c}"] for c in range(1, 6)])
        assert len(np.unique(allr)) == 5 * N_COHORT, f"{k}: cohorts are not disjoint"
    jdump(dict(names=names, overlap=M), f"{APP}/cohorts/cohort_overlap.json")
    print("primary-vs-strat overlap (cells):", {n: int(M[i, 5:].sum()) for i, n in enumerate(names[:5])} if len(names) == 10 else M, flush=True)


def rot_angle_between(R1, R2):
    D = R1 @ R2.T
    return PR.rotation_angle_deg(D) if np.linalg.det(D) > 0 else None


def pose_row(pose, cond):
    keys = ("det", "angle_deg", "n_supported", "n_supported_cells", "sv_ratio_target_cov", "sv_ratio_pred_cov", "sv_ratio_cross", "resid_rms")
    return dict({k: pose[k] for k in keys}, conditions=cond, R=pose["R"], t=pose["t"], c=pose["c"], scale=pose["scale"])


def make_layouts(feats, P_ours, Pn_base, graph_id, ext, lm, iso_mm=EV.ISO_MM):
    labels = PR.assign(feats, lm["centers"])
    out, diag = {}, {}
    n_q = np.bincount(labels, minlength=len(lm["landmark"]))
    diag["cluster_counts"] = n_q
    if P_ours is not None:
        P = np.asarray(P_ours, dtype=np.float64) * iso_mm
        out["ours_raw"] = (P + C_TRAIN, None)
        pz = PR.fit_pose(lm, labels, P, scale=None)
        cond = PR.conditions(pz)
        out["ours_readout"] = (PR.apply_pose(P, pz), pose_row(pz, cond))
        diag["ours_rotation_angle_deg"] = pz["angle_deg"]
        diag["ours_det"] = pz["det"]
    if Pn_base is not None:
        pz = PR.fit_pose(lm, labels, Pn_base, scale=np.asarray(ext, dtype=np.float64))
        cond = PR.conditions(pz)
        Y = PR.apply_pose(Pn_base, pz)
        out["baseline_readout"] = (Y, pose_row(pz, cond))
        per_graph = []
        for g in np.unique(graph_id):
            m = graph_id == g
            try:
                pg = PR.fit_pose(lm, labels[m], Pn_base[m], scale=np.asarray(ext, dtype=np.float64))
            except Exception as e:
                per_graph.append(dict(graph=int(g), error=str(e)))
                continue
            ang = rot_angle_between(pg["R"], pz["R"])
            per_graph.append(dict(graph=int(g), n_supported=pg["n_supported"], det=pg["det"], angle_to_pooled_deg=ang, mirrored_vs_pooled=ang is None,
                                  ok=bool(PR.conditions(pg)["all"])))
        diag["baseline_per_graph_pose"] = per_graph
    return out, labels, diag


def run_layouts(kind, cohorts, seeds, allow_partial=False):
    lm = PR.load_landmarks(LANDMARKS)
    lm_md5 = md5f(LANDMARKS)
    os.makedirs(f"{APP}/layouts", exist_ok=True)
    for c in cohorts:
        d = f"{APP}/cohorts/{kind}{c}"
        feats = np.load(f"{d}/features.npy")
        rows = np.load(f"{d}/rows.npy")
        for s in seeds:
            zo = np.load(f"{APP}/raw/ours_{kind}{c}_s{s}.npz") if os.path.exists(f"{APP}/raw/ours_{kind}{c}_s{s}.npz") else None
            zl = np.load(f"{APP}/raw/baseline_{kind}{c}_s{s}.npz") if os.path.exists(f"{APP}/raw/baseline_{kind}{c}_s{s}.npz") else None
            if (zo is None or zl is None) and not allow_partial:
                raise SystemExit(f"raw output missing for {kind}{c} s{s} (ours {zo is not None}, baseline {zl is not None}); use --allow_partial only for a deliberate partial run")
            if zo is not None:
                assert (zo["cell_ID"] == rows).all()
            Pn, gid, ext = None, None, None
            if zl is not None:
                assert (zl["cell_ID"] == rows).all()
                gid = zl["graph_id"]
                Pn = np.array(zl["P_norm"], dtype=np.float64)
                for g in np.unique(gid):
                    Pn[gid == g] -= Pn[gid == g].mean(0)
                ext = json.load(open(f"{APP}/raw/baseline_{kind}{c}_s{s}.npz.json"))["train_unit_EXT"]
            outs, labels, diag = make_layouts(feats, None if zo is None else zo["P"], Pn, gid, ext, lm)
            for arm, (Y, prow) in outs.items():
                path = f"{APP}/layouts/{arm}_{kind}{c}_s{s}.npz"
                np.savez_compressed(path, Y_mm=Y.astype(np.float32), rows=rows, cluster=labels.astype(np.int16))
                info = dict(arm=arm, kind=kind, cohort=c, seed=s, n=int(len(Y)), landmarks_md5=lm_md5, pose=prow, diagnostics={k: v for k, v in diag.items() if k != "cluster_counts"},
                            raw_input=f"raw/{'ours' if arm.startswith('ours') else 'baseline'}_{kind}{c}_s{s}.npz", layout_md5=md5f(path),
                            frame="training-ruler mm [ML, DV, AP], unfolded; ours_raw = sampler P x iso_mm + c_train (no rotation / landmarks)")
                jdump(info, path + ".json")
                msg = "" if prow is None else f" det {prow['det']:+.0f} angle {prow['angle_deg']:.2f} deg supported {prow['n_supported']} cond {prow['conditions']['all']}"
                print(f"[{t()}] layout {arm} {kind}{c} s{s}: Y mean {Y.mean(0).round(3).tolist()} std {Y.std(0).round(3).tolist()}{msg}", flush=True)


def run_freeze(kind):
    files = sorted(f for sub in ("raw", "layouts") for f in (os.path.join(sub, x) for x in os.listdir(f"{APP}/{sub}")) if f.endswith(".npz") and f"_{kind}" in f)
    man = {f: md5f(f"{APP}/{f}") for f in files}
    path = f"{APP}/freeze_manifest_{kind}.json"
    jdump(dict(kind=kind, created=time.strftime("%F %T"), host=HOST, n_files=len(man), md5=man), path)
    print(f"[{t()}] freeze manifest {kind}: {len(man)} files, md5 {md5f(path)}  ({path})", flush=True)


def build_eval_labels():
    import pandas as pd
    lab = pool_labels()
    tax = pd.read_csv(P.TAXONOMY)
    sub_names = sorted(tax["subclass"].unique())
    code = {s: i for i, s in enumerate(sub_names)}
    A = int(tax.cluster_alias.max()) + 1
    alias2sub = np.full(A, -1, np.int16)
    alias2sub[tax.cluster_alias.to_numpy()] = tax["subclass"].map(code).to_numpy().astype(np.int16)
    df = pd.read_csv(f"{POOL}/EVAL_ONLY_cell_metadata.csv.gz", usecols=["cell_label", "region_of_interest_acronym", "donor_label", "cluster_alias", "subclass"], dtype={"cell_label": str})
    assert len(df) == N_POOL and (df.cell_label.to_numpy() == lab).all()
    roi_names = sorted(df.region_of_interest_acronym.unique())
    roi = df.region_of_interest_acronym.map({n: i for i, n in enumerate(roi_names)}).to_numpy().astype(np.int8)
    unit = df.region_of_interest_acronym.map(EU.ROI_TO_UNIT).map({n: i for i, n in enumerate(EU.UNIT_NAMES)}).to_numpy().astype(np.int8)
    donors = sorted(df.donor_label.unique())
    donor = df.donor_label.map({n: i for i, n in enumerate(donors)}).to_numpy().astype(np.int32)
    alias = df.cluster_alias.to_numpy().astype(np.int64)
    sub = np.full(N_POOL, -1, np.int16)
    ok = (alias >= 0) & (alias < A)
    sub[ok] = alias2sub[alias[ok]]
    pre = df["subclass"].to_numpy()
    mine = np.array(sub_names, dtype=object)[np.clip(sub, 0, None)]
    agree = float(((mine == pre) & (sub >= 0)).mean())
    np.savez_compressed(f"{APP}/pool_eval_labels.npz", roi=roi, unit=unit, donor=donor, subclass=sub, roi_names=np.array(roi_names), donor_names=np.array(donors),
                        subclass_names=np.array(sub_names))
    print(f"[{t()}] eval labels: {len(roi_names)} ROIs, {len(donors)} donors, subclass found {int((sub >= 0).sum()):,}, join agrees with the pre-joined column on {agree:.6f} of cells", flush=True)


class EvalLabels:
    def __init__(self, path=None):
        z = np.load(path or f"{APP}/pool_eval_labels.npz")
        self.roi, self.unit, self.donor, self.subclass = z["roi"], z["unit"], z["donor"], z["subclass"]
        self.roi_names, self.donor_names, self.subclass_names = list(z["roi_names"]), list(z["donor_names"]), list(z["subclass_names"])


def check_support_subclass_names(sup_dir=SUPPORT):
    rep = json.load(open(f"{sup_dir}/support_report.json"))
    return rep["subclass_names"]


def eval_one_layout(sup, refs, Y_mm, rows, labels, cohort_key, tag, with_refs=True, out_dir=None, arm=None):
    cu, cs, cr = labels.unit[rows], labels.subclass[rows], labels.roi[rows]
    Ycal = np.asarray(Y_mm, dtype=np.float64) * EV.S_CAL
    row, res = EV.evaluate_layout(sup, refs, Ycal, cu, cs, cohort_key, cell_roi=cr, roi_names=labels.roi_names)
    row.update(tag=tag, arm=arm)
    if with_refs:
        perm = []
        for sd in EV.REF_SEEDS:
            Yp = EV.ref_within_type_permutation(Ycal, cs, sd)
            r2 = sup.membership(Yp, cu)
            sm = EV.summarize_membership(r2, cu)
            perm.append(dict(seed=sd, **{k: dict(macro=v["hit"]["macro"], micro=v["hit"]["micro"]) for k, v in sm.items()}))
        row["within_type_permutation"] = perm
    if out_dir:
        np.savez_compressed(f"{out_dir}/{tag}_cells.npz", rows=rows, win=res["win"], win_fine=res["win_fine"], d1=res["d1"].astype(np.float32), share=res["share"],
                            hit500=res["hit"][500.0], hit250=res["hit"][250.0], hit1000=res["hit"][1000.0], hit_fine500=res["hit_fine"][500.0])
    return row


def eval_reference_rows(sup, refs, rows, labels, cohort_key, tag, out_dir=None):
    cu, cs, cr = labels.unit[rows], labels.subclass[rows], labels.roi[rows]
    out = dict(tag=tag, R1=[], R2=[])
    for sd in EV.REF_SEEDS:
        Q = EV.ref_global_random(sup, len(rows), sd)
        row, _ = EV.evaluate_layout(sup, refs, Q, cu, cs, cohort_key)
        out["R1"].append(dict(seed=sd, **row))
        Q2, placed = EV.ref_type_conditional(refs, cs, sd)
        row2, _ = EV.evaluate_layout(sup, refs, np.where(placed[:, None], Q2, 0.0), cu, cs, cohort_key, placed=placed)
        row2["placed_frac"] = float(placed.mean())
        out["R2"].append(dict(seed=sd, **row2))
    return out


def run_evaluate(kind, cohorts, seeds, freeze_md5, arms=ARMS, refs_only=False):
    fm = f"{APP}/freeze_manifest_{kind}.json"
    assert os.path.exists(fm) and md5f(fm) == freeze_md5, "the freeze manifest is missing or its md5 differs from the one recorded at the freeze"
    man = json.load(open(fm))["md5"]
    if not os.path.exists(f"{APP}/pool_eval_labels.npz"):
        build_eval_labels()
    labels = EvalLabels()
    assert labels.subclass_names == check_support_subclass_names(), "subclass coding of the 10x labels != the support's"
    sup = EV.Support(SUPPORT, brains=(0, 1, 2, 3, 4), workers=6)
    refs = EV.RefClouds(sup)
    od = f"{APP}/eval"
    os.makedirs(od, exist_ok=True)
    for c in cohorts:
        rows = np.load(f"{APP}/cohorts/{kind}{c}/rows.npy")
        if not os.path.exists(f"{od}/refs_{kind}{c}.json"):
            jdump(eval_reference_rows(sup, refs, rows, labels, c, f"refs_{kind}{c}"), f"{od}/refs_{kind}{c}.json")
            print(f"[{t()}] reference rows {kind}{c} done", flush=True)
        if refs_only:
            continue
        for s in seeds:
            for arm in arms:
                lp = f"layouts/{arm}_{kind}{c}_s{s}.npz"
                if lp not in man:
                    if os.path.exists(f"{APP}/{lp}"):
                        raise SystemExit(f"{lp} exists but is NOT in the freeze manifest: refusing to evaluate an unfrozen layout")
                    continue
                assert md5f(f"{APP}/{lp}") == man[lp], f"{lp} changed after the freeze"
                z = np.load(f"{APP}/{lp}")
                assert (z["rows"] == rows).all()
                info = json.load(open(f"{APP}/{lp}.json"))
                tag = f"{arm}_{kind}{c}_s{s}"
                row = eval_one_layout(sup, refs, z["Y_mm"], rows, labels, c, tag, out_dir=od, arm=arm)
                row.update(kind=kind, cohort=c, seed=s, pose=info.get("pose"), diagnostics=info.get("diagnostics"))
                jdump(row, f"{od}/{tag}.json")
                m = row["membership"]["R500"]["hit"]
                print(f"[{t()}] {tag}: membership R500 macro {m['macro']:.4f} micro {m['micro']:.4f} | SW2 orig {row['sw2'].get('orig_macro', float('nan')):.1f} centred {row['sw2'].get('centred_macro', float('nan')):.1f} um "
                      f"({row['sw2'].get('n_eligible_subclasses')} subclasses)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pool_unit", "cohorts", "overlap", "layout", "freeze", "labels", "evaluate"])
    ap.add_argument("--kind", default="primary", choices=["primary", "strat"])
    ap.add_argument("--cohorts", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--freeze_md5", default="")
    ap.add_argument("--refs_only", action="store_true")
    ap.add_argument("--allow_partial", action="store_true")
    a = ap.parse_args()
    if a.cmd == "pool_unit":
        build_pool_unit()
    elif a.cmd == "cohorts":
        build_cohorts(a.kind)
    elif a.cmd == "overlap":
        cohort_overlap()
    elif a.cmd == "layout":
        run_layouts(a.kind, a.cohorts, a.seeds, a.allow_partial)
    elif a.cmd == "freeze":
        run_freeze(a.kind)
    elif a.cmd == "labels":
        build_eval_labels()
    elif a.cmd == "evaluate":
        run_evaluate(a.kind, a.cohorts, a.seeds, a.freeze_md5, refs_only=a.refs_only)


if __name__ == "__main__":
    main()
