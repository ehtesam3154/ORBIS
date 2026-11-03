#!/usr/bin/env python
"""Summary of the dissociated-cell application from the per-layout evaluation files: per arm the mean over the 3 sampler seeds, then
mean +- SD over the 5 cohorts; reference rows; paired ours_readout - baseline_readout differences (donor bootstrap for the hit
rates, subclass bootstrap for the type geometry; B = 1000).  usage: summarize.py --kind primary|strat --out DIR"""
import argparse
import csv
import json
import os

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
import numpy as np

import application as EP
import paths as P
import units as EU

ARMS = ("ours_raw", "ours_readout", "baseline_readout")
B_BOOT = 1000
MARGIN_HIT_PP, MARGIN_SW2_UM = 2.0, 25.0


def load_layout_rows(od, kind, cohorts, seeds, arms):
    rows, excluded = {}, []
    for arm in arms:
        for c in cohorts:
            for s in seeds:
                p = f"{od}/{arm}_{kind}{c}_s{s}.json"
                if not os.path.exists(p):
                    continue
                r = json.load(open(p))
                if arm != "ours_raw" and r.get("pose") and not r["pose"]["conditions"]["all"]:
                    excluded.append((arm, c, s, "pose not identifiable"))
                    continue
                rows[(arm, c, s)] = r
    return rows, excluded


def cohort_means(vals):
    by_c = {}
    for (c, s), v in vals.items():
        by_c.setdefault(c, []).append(v)
    cm = np.array([np.mean(v) for v in by_c.values()])
    sd_seed = np.array([np.std(v, ddof=1) for v in by_c.values() if len(v) > 1])
    return float(cm.mean()), float(cm.std(ddof=1)) if len(cm) > 1 else float("nan"), float(sd_seed.mean()) if len(sd_seed) else float("nan"), len(cm)


def fmt(x, nd=4):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def donor_bootstrap(od, kind, rows, labels, seed=25400):
    nd = len(labels.donor_names)
    rng = np.random.default_rng(seed)
    Wd = np.stack([np.bincount(rng.integers(0, nd, nd), minlength=nd) for _ in range(B_BOOT)]).astype(np.float64)
    acc = {}
    for (arm, c, s), r in rows.items():
        z = np.load(f"{od}/{arm}_{kind}{c}_s{s}_cells.npz")
        rr = z["rows"]
        cu = np.where(labels.unit[rr] >= 0, labels.unit[rr], 23)
        cd = labels.donor[rr]
        H = np.zeros((nd, 24)); M = np.zeros((nd, 24))
        np.add.at(H, (cd, cu), z["hit500"].astype(np.float64)); np.add.at(M, (cd, cu), 1.0)
        num, den = Wd @ H, Wd @ M
        macro = np.nanmean(np.where(den[:, :23] > 0, num[:, :23] / np.maximum(den[:, :23], 1e-12), np.nan), axis=1)
        micro = num.sum(1) / np.maximum(den.sum(1), 1e-12)
        acc.setdefault((arm, c), []).append(np.stack([macro, micro], 1))
    out = {}
    for (arm, c), lst in acc.items():
        out.setdefault(arm, []).append(np.mean(lst, axis=0))
    return {arm: np.mean(v, axis=0) for arm, v in out.items()}


def subclass_bootstrap(rows, arms, which, seed=25401):
    per = {}
    for (arm, c, s), r in rows.items():
        ps = r["sw2"].get("per_subclass") or []
        per[(arm, c, s)] = {int(x[0]): x[4 if which == "orig" else 5] for x in ps}
    if not per:
        return None
    common = set.intersection(*[set(v) for v in per.values()])
    common = sorted(common)
    if not common:
        return None
    V = {}
    for arm in arms:
        bycoh = {}
        for (a, c, s), d in per.items():
            if a == arm:
                bycoh.setdefault(c, []).append([d[k] for k in common])
        if bycoh:
            V[arm] = np.mean([np.mean(v, axis=0) for v in bycoh.values()], axis=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(common), size=(B_BOOT, len(common)))
    return {arm: v[idx].mean(1) for arm, v in V.items()}, len(common)


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="primary")
    ap.add_argument("--cohorts", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(EP.SEEDS))
    ap.add_argument("--out", default=P.RESULTS)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    a = ap.parse_args()
    od = f"{EP.APP}/eval"
    rows, excluded = load_layout_rows(od, a.kind, a.cohorts, a.seeds, a.arms)
    labels = EP.EvalLabels()
    res = dict(kind=a.kind, cohorts=a.cohorts, seeds=a.seeds, n_layouts=len(rows), excluded_pose_not_identifiable=excluded, arms={}, references={}, paired={})
    md = [f"# Application summary: {a.kind} cohorts\n", f"layouts evaluated: {len(rows)}; excluded as 'pose not identifiable': {excluded or 'none'}\n"]
    head = []
    for arm in a.arms:
        sel = {(c, s): r for (ar, c, s), r in rows.items() if ar == arm}
        if not sel:
            continue
        m = {}
        for R in ("R500", "R250", "R1000"):
            for kk in ("hit", "hit_fine"):
                for stat in ("macro", "micro"):
                    m[f"{R}_{kk}_{stat}"] = cohort_means({k: r["membership"][R][kk][stat] for k, r in sel.items()})
        m["R500_outside"] = cohort_means({k: r["membership"]["R500"]["outside_frac"] for k, r in sel.items()})
        m["R500_confidence"] = cohort_means({k: r["membership"]["R500"]["support_confidence_mean"] for k, r in sel.items()})
        for stat in ("orig_macro", "centred_macro", "orig_micro", "centred_micro", "coverage", "n_eligible_subclasses"):
            m[f"sw2_{stat}"] = cohort_means({k: r["sw2"].get(stat, float("nan")) for k, r in sel.items()})
        perm = {k: np.mean([p["R500"]["macro"] for p in r["within_type_permutation"]]) for k, r in sel.items() if r.get("within_type_permutation")}
        if perm:
            m["R3_permutation_macro_R500"] = cohort_means(perm)
        res["arms"][arm] = m
        head.append((arm, m))
    md.append("## Headline (mean over seeds, then mean +- SD over cohorts; [seed SD within cohort])\n")
    md.append("| arm | R500 macro | R500 micro | R250 macro | R1000 macro | fine-vote macro | outside R500 | SW2 orig macro (um) | SW2 centred macro (um) | eligible subclasses | R3 permutation macro |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for arm, m in head:
        f = lambda k, nd=4: f"{fmt(m[k][0], nd)} +- {fmt(m[k][1], nd)} [{fmt(m[k][2], nd)}]"
        md.append(f"| {arm} | {f('R500_hit_macro')} | {f('R500_hit_micro')} | {f('R250_hit_macro')} | {f('R1000_hit_macro')} | {f('R500_hit_fine_macro')} | {f('R500_outside')} | "
                  f"{f('sw2_orig_macro', 1)} | {f('sw2_centred_macro', 1)} | {fmt(m['sw2_n_eligible_subclasses'][0], 0)} | {f('R3_permutation_macro_R500') if 'R3_permutation_macro_R500' in m else '-'} |")
    refs = {}
    for c in a.cohorts:
        p = f"{od}/refs_{a.kind}{c}.json"
        if os.path.exists(p):
            refs[c] = json.load(open(p))
    md.append("\n## Reference rows (10 seeds per cohort, then mean +- SD over cohorts)\n")
    md.append("| reference | R500 macro | R500 micro | micro over placed cells | placed fraction | SW2 orig macro (um) | SW2 centred macro (um) |")
    md.append("|---|---|---|---|---|---|---|")
    for key, name in (("R1", "R1 global random placement"), ("R2", "R2 type-conditional random placement")):
        per = {}
        for c, rr in refs.items():
            L = rr[key]
            per[c] = dict(macro=np.mean([x["membership"]["R500"]["hit"]["macro"] for x in L]), micro=np.mean([x["membership"]["R500"]["hit"]["micro"] for x in L]),
                          mop=np.mean([x["membership"]["R500"]["micro_over_placed"] for x in L]), placed=np.mean([x["membership"]["R500"]["placed_frac"] for x in L]),
                          o=np.mean([x["sw2"].get("orig_macro", np.nan) for x in L]), cen=np.mean([x["sw2"].get("centred_macro", np.nan) for x in L]))
        if per:
            g = lambda k, nd=4: (lambda v: f"{fmt(float(np.mean(v)), nd)} +- {fmt(float(np.std(v, ddof=1)) if len(v) > 1 else float('nan'), nd)}")([p[k] for p in per.values()])
            res["references"][key] = {k: float(np.mean([p[k] for p in per.values()])) for k in ("macro", "micro", "mop", "placed", "o", "cen")}
            md.append(f"| {name} | {g('macro')} | {g('micro')} | {g('mop')} | {g('placed')} | {g('o', 1)} | {g('cen', 1)} |")
    md.append("\n## Per-unit macro components (R500 hit rate, mean over layouts)\n")
    md.append("| unit | " + " | ".join(arm for arm, _ in head) + " |")
    md.append("|---|" + "---|" * len(head))
    per_unit = {}
    for arm, _ in head:
        L = [r["membership"]["R500"]["hit"]["per_unit"] for (ar, c, s), r in rows.items() if ar == arm]
        per_unit[arm] = np.nanmean(np.array(L, dtype=float), axis=0)
    for u in range(23):
        md.append(f"| {EU.UNIT_NAMES[u]} | " + " | ".join(fmt(float(per_unit[arm][u]), 3) for arm, _ in head) + " |")
    with open(f"{a.out}/summary_{a.kind}_per_unit.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["unit"] + [arm for arm, _ in head])
        for u in range(23):
            w.writerow([EU.UNIT_NAMES[u]] + [float(per_unit[arm][u]) for arm, _ in head])
    roi_tab = {}
    for arm, _ in head:
        acc = {}
        for (ar, c, s), r in rows.items():
            if ar == arm:
                for rn, v in r["membership"]["R500"]["hit"].get("per_roi", {}).items():
                    acc.setdefault(rn, []).append((v["hit"], v["n"]))
        roi_tab[arm] = {rn: float(np.mean([x[0] for x in v])) for rn, v in acc.items()}
    if roi_tab and any(roi_tab.values()):
        names = sorted(set().union(*[set(v) for v in roi_tab.values()]))
        with open(f"{a.out}/summary_{a.kind}_per_roi.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["roi"] + [arm for arm, _ in head])
            for rn in names:
                w.writerow([rn] + [roi_tab[arm].get(rn, float("nan")) for arm, _ in head])
    md.append("\n## Readout conditions and truth-free diagnostics\n")
    md.append("| arm | cohort | seed | det | rotation (deg) | supported clusters | SV ratio | conditions | baseline graphs mirrored vs pooled | baseline median / max per-graph angle (deg) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for (arm, c, s), r in sorted(rows.items()):
        if r.get("pose"):
            p = r["pose"]
            pg = (r.get("diagnostics") or {}).get("baseline_per_graph_pose") if arm == "baseline_readout" else None
            if pg:
                angs = [g["angle_to_pooled_deg"] for g in pg if g.get("angle_to_pooled_deg") is not None]
                mir = sum(1 for g in pg if g.get("mirrored_vs_pooled"))
                extra = f"{mir}/{len(pg)} | {fmt(float(np.median(angs)), 2) if angs else '-'} / {fmt(float(np.max(angs)), 2) if angs else '-'}"
            else:
                extra = "- | -"
            md.append(f"| {arm} | {c} | {s} | {p['det']:+.0f} | {p['angle_deg']:.2f} | {p['n_supported']} | {p['sv_ratio_target_cov']:.3f} | {p['conditions']['all']} | {extra} |")
    if "ours_readout" in res["arms"] and "baseline_readout" in res["arms"]:
        db = donor_bootstrap(od, a.kind, rows, labels)
        md.append("\n## Paired difference ours_readout - baseline_readout (bootstrap 95 % interval; margins 2 pp / 25 um)\n")
        md.append("| statistic | difference | 95 % interval | beyond margin? |")
        md.append("|---|---|---|---|")
        for j, nm in ((0, "R500 macro hit rate (donor bootstrap)"), (1, "R500 micro hit rate (donor bootstrap)")):
            d = db["ours_readout"][:, j] - db["baseline_readout"][:, j]
            lo, hi = ci(d)
            res["paired"][nm] = dict(mean=float(d.mean()), ci=[lo, hi])
            md.append(f"| {nm} | {100 * d.mean():+.2f} pp | [{100 * lo:+.2f}, {100 * hi:+.2f}] | {'ours above margin' if lo > MARGIN_HIT_PP / 100 else ('baseline above margin' if hi < -MARGIN_HIT_PP / 100 else 'inside / overlapping the +-2 pp margin')} |")
        for which in ("orig", "centred"):
            sb = subclass_bootstrap(rows, ("ours_readout", "baseline_readout"), which)
            if sb is None:
                continue
            V, ncommon = sb
            d = V["ours_readout"] - V["baseline_readout"]
            lo, hi = ci(d)
            res["paired"][f"SW2 {which} macro"] = dict(mean=float(d.mean()), ci=[lo, hi], n_common_subclasses=ncommon)
            md.append(f"| SW2 {which} macro (subclass bootstrap, {ncommon} common subclasses) | {d.mean():+.1f} um (negative = ours closer to the MERFISH clouds) | [{lo:+.1f}, {hi:+.1f}] | "
                      f"{'ours better than margin' if hi < -MARGIN_SW2_UM else ('baseline better than margin' if lo > MARGIN_SW2_UM else 'inside / overlapping the +-25 um margin')} |")
    ctl = P.FROZEN
    try:
        ca = json.load(open(f"{ctl}/control_a.json")); cc = json.load(open(f"{ctl}/control_c.json"))
        md.append(f"\n## Controls to read the numbers against\n\n- membership ceiling (Allen cells at their own positions, leave-one-out, five-brain support): micro {ca['support_all5_loo']['R500']['hit']['micro']:.4f}, macro {ca['support_all5_loo']['R500']['hit']['macro']:.4f}; "
                  f"cross-brain (Zhuang 1-4 support only): micro {ca['support_zhuang1to4']['R500']['hit']['micro']:.4f}, macro {ca['support_zhuang1to4']['R500']['hit']['macro']:.4f}\n"
                  f"- SW2 finite-sample floor (two halves of one support subclass cloud, n <= 1000): original {cc['floor_summary']['orig_macro_mean']:.1f} um, centred {cc['floor_summary']['centred_macro_mean']:.1f} um; "
                  f"between-brain (Zhuang 1-4 vs Allen hemisphere): original {cc['between_brain']['orig_macro']:.1f} um, centred {cc['between_brain']['centred_macro']:.1f} um\n")
    except Exception as e:
        md.append(f"\n(controls files not found: {e})\n")
    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/summary_{a.kind}.md", "w") as f:
        f.write("\n".join(md) + "\n")
    EP.jdump(res, f"{a.out}/summary_{a.kind}.json")
    print("\n".join(md))


if __name__ == "__main__":
    main()
