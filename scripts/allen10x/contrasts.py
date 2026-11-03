#!/usr/bin/env python
"""Paired contrasts of the dissociated-cell application with their 95 % intervals and the reading against the fixed margins
(2 points on the macro hit rate, 25 um on the type geometry): ours_readout - baseline_readout, ours_raw - baseline_readout,
ours_raw - ours_readout.  usage: contrasts.py [--kinds primary strat] --out DIR  ->  DIR/contrasts.json"""
import argparse
import os
import time

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
import numpy as np

import application as EP
import paths as P
import summarize as ES

PAIRS = [("ours_readout", "baseline_readout", "primary contrast"),
         ("ours_raw", "baseline_readout", "added contrast"),
         ("ours_raw", "ours_readout", "added contrast")]
M_HIT, M_SW2 = ES.MARGIN_HIT_PP / 100.0, ES.MARGIN_SW2_UM


def classify(lo, hi, margin, higher_is_better):
    a_side = lo > margin if higher_is_better else hi < -margin
    b_side = hi < -margin if higher_is_better else lo > margin
    if a_side:
        return "A beyond"
    if b_side:
        return "B beyond"
    if lo >= -margin and hi <= margin:
        return "tie"
    return "inconclusive"


def overall(verdicts):
    a = [k for k, v in verdicts.items() if v == "A beyond"]
    b = [k for k, v in verdicts.items() if v == "B beyond"]
    if a and b:
        return "trade-off"
    if a:
        return "A better"
    if b:
        return "B better"
    if all(v == "tie" for v in verdicts.values()):
        return "tie"
    return "inconclusive"


def contrasts(kind, cohorts, seeds, arms=ES.ARMS):
    od = f"{EP.APP}/eval"
    rows, excluded = ES.load_layout_rows(od, kind, cohorts, seeds, list(arms))
    labels = EP.EvalLabels()
    db = ES.donor_bootstrap(od, kind, rows, labels)
    out = dict(excluded_pose_not_identifiable=excluded, pairs=[])
    sb_cache = {}
    for a, b, note in PAIRS:
        if a not in db or b not in db:
            continue
        pr = dict(A=a, B=b, note=note, endpoints={})
        for j, nm in ((0, "R500 macro hit rate"), (1, "R500 micro hit rate")):
            d = db[a][:, j] - db[b][:, j]
            lo, hi = ES.ci(d)
            pr["endpoints"][nm] = dict(mean=float(d.mean()), ci=[lo, hi], margin=M_HIT, unit="fraction (x100 = pp)", verdict=classify(lo, hi, M_HIT, True), primary=(j == 0), bootstrap="donor, B=1000 seed 25400")
        for which in ("orig", "centred"):
            sb = ES.subclass_bootstrap(rows, (a, b), which)
            if sb is None:
                continue
            V, ncommon = sb
            d = V[a] - V[b]
            lo, hi = ES.ci(d)
            pr["endpoints"][f"SW2 {which} macro"] = dict(mean=float(d.mean()), ci=[lo, hi], margin=M_SW2, unit="um (negative = A closer to the MERFISH clouds)", verdict=classify(lo, hi, M_SW2, False), primary=True,
                                                      n_common_subclasses=ncommon, bootstrap="subclass, B=1000 seed 25401")
        pr["overall"] = overall({k: v["verdict"] for k, v in pr["endpoints"].items() if v["primary"]})
        out["pairs"].append(pr)
    return out


def main():
    """Contrasts of every kind -> OUT/contrasts.json."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["primary", "strat"])
    ap.add_argument("--cohorts", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(EP.SEEDS))
    ap.add_argument("--out", default=P.RESULTS)
    a = ap.parse_args()
    res = dict(created=time.strftime("%F %T"), app=EP.APP, kinds={})
    for kind in a.kinds:
        res["kinds"][kind] = dict(contrasts=contrasts(kind, a.cohorts, a.seeds))
    os.makedirs(a.out, exist_ok=True)
    EP.jdump(res, f"{a.out}/contrasts.json")
    for kind, v in res["kinds"].items():
        for pr in v["contrasts"]["pairs"]:
            e = pr["endpoints"]["R500 macro hit rate"]
            print(f"{kind}: {pr['A']} - {pr['B']}: R500 macro {100 * e['mean']:+.2f} pp [{100 * e['ci'][0]:+.2f}, {100 * e['ci'][1]:+.2f}] {e['verdict']}")


if __name__ == "__main__":
    main()
