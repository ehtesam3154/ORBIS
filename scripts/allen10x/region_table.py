"""Per-region hit rate and per-subclass type geometry of the uniform cohorts, from the per-layout evaluation files.

Region hit rate (R = 500 um) per dissection unit: mean over the 15 runs (5 cohorts x 3 sampler seeds) of each arm.
Type geometry: per subclass, the sliced Wasserstein-2 distance averaged over the runs, kept for the subclasses scored in all 15 runs.
usage: region_table.py --out DIR [--arms ours_readout baseline_readout] [--names Ours Baseline]
writes DIR/units.csv and DIR/subclass_sw2.csv and prints the macro numbers."""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

import paths as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=P.RESULTS)
    ap.add_argument("--arms", nargs=2, default=["ours_readout", "baseline_readout"])
    ap.add_argument("--names", nargs=2, default=["Ours", "Baseline"])
    a = ap.parse_args()
    E = P.APP
    UNITS = json.load(open(f"{E}/support/support_report.json"))["unit_names"]
    SUBN = np.load(f"{E}/pool_eval_labels.npz")["subclass_names"]
    ARMS = dict(zip(a.names, a.arms))
    hit, n_unit, sw2 = {}, None, {}
    for name, arm in ARMS.items():
        files = sorted(glob.glob(f"{E}/eval/{arm}_primary[1-5]_s100[0-2].json"))
        assert len(files) == 15, files
        J = [json.load(open(f)) for f in files]
        hit[name] = np.mean([j["membership"]["R500"]["hit"]["per_unit"] for j in J], axis=0)
        n_unit = np.sum([j["membership"]["R500"]["hit"]["n_per_unit"] for j in J], axis=0) / 15
        rows = pd.concat([pd.DataFrame(j["sw2"]["per_subclass"], columns=["sub", "n", "n_support", "n_used", "sw2", "sw2_centred"])
                          for j in J])
        g = rows.groupby("sub")
        sw2[name] = g[["sw2", "n"]].mean()[g.size() == len(files)]
        print(f"{name:8s} runs {len(files)}  macro hit {hit[name].mean():.3f}  macro SW2 {sw2[name].sw2.mean():.0f} um")
    o, b = a.names
    U = pd.DataFrame({"unit": UNITS, "n_cells": n_unit.round().astype(int), o: hit[o], b: hit[b]})
    S = sw2[o].join(sw2[b], lsuffix="_ours", rsuffix="_base", how="inner")
    S = pd.DataFrame({"subclass": SUBN[S.index.astype(int)], "n_cells": S.n_ours.round().astype(int),
                      o: S.sw2_ours.to_numpy(), b: S.sw2_base.to_numpy()})
    os.makedirs(a.out, exist_ok=True)
    U.to_csv(f"{a.out}/units.csv", index=False); S.to_csv(f"{a.out}/subclass_sw2.csv", index=False)
    print(f"\nregion hit rate (macro over {len(U)} regions): {o} {U[o].mean():.3f}  {b} {U[b].mean():.3f}"
          f"  | {o} higher in {(U[o] > U[b]).sum()} of {len(U)}")
    print(f"type geometry SW2 (mean over {len(S)} subclasses): {o} {S[o].mean():.0f} um  {b} {S[b].mean():.0f} um"
          f"  | {o} lower for {(S[o] < S[b]).sum()} of {len(S)} subclasses")


if __name__ == "__main__":
    main()
