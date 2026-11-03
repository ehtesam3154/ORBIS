"""Sampling versus averaging on one test: root-mean-square errors of one-call outputs and of 50-step samples, and the scatter
between samples of the same cell, every layout aligned onto the truth by the scorer's own fit (headline view).

  python -m orbis.metrics.spread --card CARD --test_dir TEST [brain arguments as orbis.metrics.score] \
      --guesses G1.npz ... --samples S1.npz ... --out SPREAD.json

err_avg = RMS error of the one-call outputs, err_smp = RMS error of the samples, spread = RMS distance between two samples of a
cell / sqrt 2 (all pairs), spread_bg = the same between one-call outputs, trvar = total positional variance of the truth (um^2).
"""
import argparse
import json
import os
import sys
import time
from itertools import combinations

import numpy as np

from orbis.metrics.matched import headline_truth
from orbis.metrics.score import add_test_args, build
from orbis.metrics.scorer import align_views, load_layout, log


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_test_args(ap)
    ap.add_argument("--guesses", nargs="+", required=True, help="one-call outputs (npz)")
    ap.add_argument("--samples", nargs="+", required=True, help="50-step samples (npz)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    card, spec = build(a, need_expression=False)
    hv, Xv = headline_truth(spec)
    um = spec["um_per_unit"]
    L = {}
    medians = []
    for kind, files in (("best guess", a.guesses), ("sample", a.samples)):
        L[kind] = []
        for f in files:
            Y = spec["prepare"](load_layout(f, spec))
            Ya = align_views(spec, Y, [hv], label=os.path.basename(f))[hv][0]
            medians.append(dict(npz=os.path.basename(f), median_um=float(np.median(np.linalg.norm(Ya - Xv, axis=1) * um))))
            L[kind].append(Ya)
    mse = lambda P: float(np.mean(np.sum((P - Xv) ** 2, 1)) * um ** 2)
    pair = lambda Ps: float(np.mean([np.mean(np.sum((a - b) ** 2, 1)) for a, b in combinations(Ps, 2)]) * um ** 2 / 2)
    r = dict(test=card["test"], view=hv, n_cells=len(Xv),
             err_avg_um=float(np.sqrt(np.mean([mse(P) for P in L["best guess"]]))),
             err_smp_um=float(np.sqrt(np.mean([mse(P) for P in L["sample"]]))),
             spread_um=float(np.sqrt(pair(L["sample"]))),
             spread_bg_um=float(np.sqrt(pair(L["best guess"]))),
             trvar_um2=float(np.sum(np.var(Xv, 0)) * um ** 2),
             median_errors=medians, seconds=round(time.time() - t0, 1))
    r["spread_over_err_avg"] = r["spread_um"] / r["err_avg_um"]
    r["err_smp_over_err_avg"] = r["err_smp_um"] / r["err_avg_um"]
    r["avg_left_share"] = r["err_avg_um"] ** 2 / r["trvar_um2"]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(r, open(a.out, "w"), indent=1)
    log(f"[spread] {card['test']}: err_avg {r['err_avg_um']:.1f} um, err_smp {r['err_smp_um']:.1f} um, spread {r['spread_um']:.1f} um")
    return 0


if __name__ == "__main__":
    sys.exit(main())
