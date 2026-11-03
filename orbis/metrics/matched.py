"""Per-cell error at matched size: for each layout, the median distance between predicted and true positions of the aligned
layout (raw) and of the scorer's size-matched layout (matched), in the test's headline view, in micrometres.

  python -m orbis.metrics.matched --card CARD --test_dir TEST [brain / subset arguments as orbis.metrics.score] \
      --layouts L1.npz [L2.npz ...] --out_dir OUT

One JSON per layout (<out_dir>/<layout file stem>.json). The raw value equals the scorer's diag_typefit err3d_median_um.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from orbis.metrics.score import add_test_args, build
from orbis.metrics.scorer import FOLDED_VIEWS, align_views, fold_lr_own, load_layout, log, md5_file


def headline_truth(spec):
    """The truth in the headline view (folded about its own midline in a folded view) and the view name."""
    hv = spec["headline_view"]
    X = spec["X"]
    Xv = fold_lr_own(spec["lr_fold"]["axis"], spec["lr_fold"]["centre"])(X) if hv in FOLDED_VIEWS else X
    return hv, Xv


def errors(spec, Y, hv, Xv, label):
    """(raw error, matched error or None, alignment info) of a layout in native units after spec['prepare']."""
    um = float(spec["um_per_unit"])
    Ya, Ys, Yd, info = align_views(spec, Y, views=[hv], label=label)[hv]
    err_raw = float(np.median(np.linalg.norm(Ya - Xv, axis=1) * um))
    err_sm = None if Ys is None else float(np.median(np.linalg.norm(Ys - Xv, axis=1) * um))
    return err_raw, err_sm, info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_test_args(ap)
    ap.add_argument("--layouts", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args(argv)
    card, spec = build(a, need_expression=False)
    hv, Xv = headline_truth(spec)
    os.makedirs(a.out_dir, exist_ok=True)
    for f in a.layouts:
        t1 = time.time()
        stem = os.path.basename(f)[:-4] if f.endswith(".npz") else os.path.basename(f)
        Y = spec["prepare"](load_layout(f, spec))
        err_raw, err_sm, info = errors(spec, Y, hv, Xv, stem)
        rec = dict(tag=stem, test=card["test"], units=spec.get("units"), view=hv, npz=os.path.abspath(f), npz_md5=md5_file(f), err_raw_um=err_raw,
                   err_sm_um=err_sm, flat_layout=info.get("flat_layout"), size_own_pc=info.get("size_own_pc"),
                   size_own_pc_sm=info.get("size_own_pc_sm"), seconds=round(time.time() - t1, 1))
        json.dump(rec, open(f"{a.out_dir}/{stem}.json", "w"), indent=1)
        log(f"[matched] {stem}: raw {err_raw:.2f} um | matched size {err_sm if err_sm is None else round(err_sm, 2)} um | {rec['seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
