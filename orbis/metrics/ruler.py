"""Within-type-shuffle ruler of a test: the true positions permuted among the cells of each fine type (seeds 26001-26010),
aligned like a method, and the per-cell error (median distance to the true position, headline view) of the aligned layout
(raw) and of its size-matched layout (matched). The ruler is the mean over the seeds; placement = 1 - error / ruler.

  python -m orbis.metrics.ruler --card CARD --test_dir TEST [brain / subset arguments as orbis.metrics.score] --out RULER.json
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

from orbis.metrics.matched import errors, headline_truth
from orbis.metrics.score import add_test_args, build
from orbis.metrics.scorer import corrupt, log

SEEDS = list(range(26001, 26011))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_test_args(ap)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    card, spec = build(a, need_expression=False)
    hv, Xv = headline_truth(spec)
    X = spec["X"]
    um = float(spec["um_per_unit"])
    fine = np.asarray(spec["fine_all"])
    rec = dict(test=card["test"], units=spec.get("units"), view=hv, n_cells=int(len(X)), um_per_unit=um, n_fine_codes=int(len(np.unique(fine))),
               seeds=SEEDS, per_seed=[])
    for s in SEEDS:
        t1 = time.time()
        Y = corrupt("@shuffle", spec["X"], spec["fine_all"], um, seed=s)
        err_raw, err_sm, info = errors(spec, Y, hv, Xv, f"RULER_shuffle_{s}")
        rec["per_seed"].append(dict(seed=s, err_raw_um=err_raw, err_sm_um=err_sm, flat_layout=info.get("flat_layout"),
                                    layout_md5=hashlib.md5(np.ascontiguousarray(Y).tobytes()).hexdigest(), seconds=round(time.time() - t1, 1)))
        log(f"[ruler] {card['test']} seed {s}: raw {err_raw:.3f} um | matched {err_sm if err_sm is None else round(err_sm, 3)} um")
    raw = np.array([r["err_raw_um"] for r in rec["per_seed"]])
    sm = np.array([r["err_sm_um"] for r in rec["per_seed"]], dtype=float)
    rec.update(ruler_raw_um=float(raw.mean()), ruler_raw_sd_um=float(raw.std(ddof=1)), ruler_matched_um=float(sm.mean()),
               ruler_matched_sd_um=float(sm.std(ddof=1)), n_seeds=len(SEEDS), seconds=round(time.time() - t0, 1))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=1)
    log(f"[ruler] {card['test']}: raw {rec['ruler_raw_um']:.3f} +- {rec['ruler_raw_sd_um']:.3f} um | matched {rec['ruler_matched_um']:.3f} um")
    return 0


if __name__ == "__main__":
    sys.exit(main())
