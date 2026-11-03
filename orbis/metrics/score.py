"""Scoring CLI: score sampled layouts (npz with keys P, G, cc, iso) or corrupted copies of the truth on one test set.

  python -m orbis.metrics.score --card orbis/metrics/cards/<test>.json --test_dir TEST_PREBUILT \
      [--store_dir STORE --store_rows ROWS.npy --cell_metadata CSV [CSV ...]]  (whole brains) \
      [--keep KEEP.npy]  (score a subset of the test cells; layouts then hold exactly those cells) \
      --layouts L1.npz [L2.npz ...] | --rulers @shuffle [@truth ...]  --out_dir OUT [--cache CACHE]

One JSON per layout: <out_dir>/<layout file stem>.json (rulers: RULER_<name>.json), with the truth summary, the environment and
every metric of every view. The truth is built once per call; the niche clustering, the derived label codes and selected genes
are cached in --cache.
"""
import argparse
import json
import os
import sys
import time

from orbis.metrics.scorer import (SEED_CORRUPT, _jsonable, build_truths, corrupt, environment, load_layout, log, md5_file, score_row)
from orbis.metrics.spec import dataset_spec, restrict


def build(a, need_expression=True):
    """The specification of the test (optionally restricted to --keep) from the CLI arguments."""
    import numpy as np
    card = json.load(open(a.card))
    spec = dataset_spec(card, a.test_dir, cache_dir=a.cache, store_dir=a.store_dir, store_rows=a.store_rows,
                        cell_metadata=a.cell_metadata, need_expression=need_expression)
    if a.keep:
        spec = restrict(spec, np.load(a.keep))
    return card, spec


def add_test_args(ap):
    """The arguments that define a test set (shared by every metrics CLI)."""
    ap.add_argument("--card", required=True, help="test card JSON (orbis/metrics/cards/<test>.json)")
    ap.add_argument("--test_dir", required=True, help="test prebuilt directory")
    ap.add_argument("--store_dir", default=None, help="whole brains: the test brain's cell store (pos.npy, meta.json, cell_labels.txt)")
    ap.add_argument("--store_rows", default=None, help="whole brains: prebuilt -> store row map (.npy), when the card asks for one")
    ap.add_argument("--cell_metadata", nargs="*", default=None, help="whole brains: atlas cell metadata CSV(s), in the order the card's label rule needs")
    ap.add_argument("--keep", default=None, help="sorted row indices (.npy) of a subset of the test cells")
    ap.add_argument("--cache", default=None, help="cache directory (niche clustering, label codes, gene selection)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_test_args(ap)
    ap.add_argument("--layouts", nargs="*", default=[], help="layout npz files (keys P, G, cc, iso)")
    ap.add_argument("--rulers", nargs="*", default=[], help="corruptions of the truth: @truth @shuffle @shrink<f> @collapse @noise<um> @axis<a>x<f>")
    ap.add_argument("--views", default="", help="comma list (default: the test's views)")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args(argv)
    if not a.layouts and not a.rulers:
        ap.error("give --layouts and / or --rulers")
    t0 = time.time()
    card, spec = build(a)
    views = a.views.split(",") if a.views else spec["views"]
    for v in views:
        if v not in spec["views"]:
            ap.error(f"view {v} not defined for {card['test']} (defined: {spec['views']})")
    truths = build_truths(spec, views, a.cache)
    os.makedirs(a.out_dir, exist_ok=True)
    base = dict(dataset=card["test"], units=spec["units"], units_rule=spec["units_rule"], card=os.path.basename(a.card), card_md5=md5_file(a.card),
                environment=environment(), dataset_provenance=_jsonable(spec["provenance"]),
                truth={v: _jsonable(t.info) for v, t in truths.items()})
    log(f"[score] {card['test']} ({spec['units']}): {len(a.layouts)} layouts, {len(a.rulers)} rulers, views {views}, truth in "
        f"{time.time() - t0:.0f} s")
    n_err = 0
    for tag in a.rulers:
        meta = dict(role="ruler_corruption", tag=tag, seed=SEED_CORRUPT)
        rec = score_row(spec, truths, meta, (lambda tag=tag: corrupt(tag, spec["X"], spec["fine_all"], spec["um_per_unit"])),
                        f"{a.out_dir}/{tag.replace('@', 'RULER_')}.json", base, views)
        n_err += rec["status"] == "error"
    for f in a.layouts:
        stem = os.path.basename(f)[:-4] if f.endswith(".npz") else os.path.basename(f)
        meta = dict(tag=stem, npz=os.path.abspath(f), npz_md5=md5_file(f))
        rec = score_row(spec, truths, meta, (lambda f=f: spec["prepare"](load_layout(f, spec))), f"{a.out_dir}/{stem}.json", base, views)
        n_err += rec["status"] == "error"
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
