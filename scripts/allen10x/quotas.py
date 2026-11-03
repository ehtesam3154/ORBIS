#!/usr/bin/env python
"""Quotas of the five region-balanced cohorts from the training cells only: q_r = share of unit r among the training-support cells
with a unit; n_r = 100,000 q_r by largest remainder; a unit with 5 n_r above its pool size is fixed at floor(pool / 5) and the
freed cells are redistributed over the other units in proportion to q_r, repeated until no unit exceeds its cap.
usage: quotas.py [support_report.json]  ->  DATA_DIR/allen10x/frozen/strat_quotas.json"""
import csv
import json
import os
import sys

import numpy as np

import paths as P
import units as EU

N_COHORT, N_SETS = 100_000, 5


def largest_remainder(target, total, free):
    base = np.floor(target).astype(np.int64) * free
    rem = int(total - base.sum())
    frac = (target - np.floor(target)) * free
    order = np.argsort(-frac, kind="stable")
    add = np.zeros_like(base)
    add[order[:rem]] = 1
    return base + add * free


def quotas(q, N, cap):
    q = np.asarray(q, dtype=np.float64)
    free = np.ones(len(q), dtype=np.int64)
    n = np.zeros(len(q), dtype=np.int64)
    remaining = N
    rounds = []
    while True:
        qq = q * free
        qq = qq / qq.sum()
        nf = largest_remainder(remaining * qq, remaining, free)
        over = (free == 1) & (nf > cap)
        rounds.append(dict(free_units=int(free.sum()), remaining=int(remaining), over=[EU.UNIT_NAMES[k] for k in np.flatnonzero(over)]))
        if not over.any():
            n[free == 1] = nf[free == 1]
            break
        n[over] = cap[over]
        remaining -= int(cap[over].sum())
        free[over] = 0
    assert n.sum() == N and (n <= cap).all()
    return n, rounds


def main():
    sup = json.load(open(sys.argv[1] if len(sys.argv) > 1 else f"{P.SUPPORT}/support_report.json"))
    cnt = np.array([sup["unit_counts_total"][u] for u in EU.UNIT_NAMES], dtype=np.float64)
    q = cnt / cnt.sum()
    roi = {r["region_of_interest_acronym"]: int(r["n_cells"]) for r in csv.DictReader(open(f"{P.FROZEN}/pool_by_roi.csv"))}
    assert sum(roi.values()) == 2_341_350 and all(r in EU.ROI_TO_UNIT for r in roi)
    elig = np.zeros(23, dtype=np.int64)
    for r, c in roi.items():
        elig[EU.UNIT_NAMES.index(EU.ROI_TO_UNIT[r])] += c
    cap = elig // N_SETS
    n0 = largest_remainder(N_COHORT * q, N_COHORT, np.ones(23, dtype=np.int64))
    n, rounds = quotas(q, N_COHORT, cap)
    per_brain = {}
    for b, v in sup["per_brain"].items():
        c = np.array([v["unit_counts"][u] for u in EU.UNIT_NAMES], dtype=np.float64)
        per_brain[b] = (c / c.sum()).round(5).tolist()
    out = dict(N=N_COHORT, n_cohorts=N_SETS, unit_names=EU.UNIT_NAMES, q=q.tolist(), n_initial=n0.tolist(), eligible_v3=elig.tolist(), cap_per_cohort=cap.tolist(),
               n_final=n.tolist(), infeasible_at_start=[EU.UNIT_NAMES[k] for k in np.flatnonzero(5 * n0 > elig)], rounds=rounds,
               shortfall_cells_redistributed=int((n0 - np.minimum(n0, cap)).clip(0).sum()), per_training_brain_unit_fraction=per_brain,
               support_report=sys.argv[1] if len(sys.argv) > 1 else "support_report.json")
    json.dump(out, open(f"{P.FROZEN}/strat_quotas.json", "w"), indent=1)
    print(f"{'unit':18s} {'q':>8s} {'n0':>7s} {'elig_v3':>8s} {'cap':>7s} {'n_final':>8s} {'5n/elig':>8s}")
    for k in range(23):
        print(f"{EU.UNIT_NAMES[k]:18s} {q[k]:8.4f} {n0[k]:7d} {elig[k]:8d} {cap[k]:7d} {n[k]:8d} {5 * n[k] / elig[k]:8.3f}")
    print("infeasible at start:", out["infeasible_at_start"], "| rounds:", rounds, "| sum", int(n.sum()))


if __name__ == "__main__":
    main()
