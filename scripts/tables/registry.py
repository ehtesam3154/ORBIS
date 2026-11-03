"""Results registry read by the table builders, and the per-layout metrics of the paper's tables.

The registry is one JSON file. Paths inside it are absolute or relative to the registry file's directory.

{
  "tests": {
    "<test>": {                                   (abca3, abca2, abca1, allen638850, dstarA, dstarB, flysta2_c, flysta2_d)
      "view": "main" | "lr_folded",              headline view of the test
      "ruler": "<orbis.metrics.ruler JSON>",
      "methods": {"<baseline name>": [DRAW], ...},       one run per baseline
      "ours": [{"raw": [DRAW, ...], "reseated": [DRAW, ...]}, ...],   one entry per training, seed 0 first; draws = sampler seeds
      "best_guess": [DRAW, ...],                  one-call outputs of the seed-0 training
      "sampling": "<orbis.metrics.spread JSON>"
    }
  },
  "names": {"baselines": ["<baseline name>", ...],      table order of the baselines
            "reference": "<baseline name>",                the baseline printed next to our model in the per-training, control and
                                                           one-call tables
            "prior_work": "<method name>"},                the method whose per-cell SRC and RSSD definitions the tables use
  "controls": {"test": "abca2", "rulers": {"truth": SCORER, "shrink0.5": SCORER, "axis0x0.75": SCORER, "noise200": SCORER,
                                          "shuffle": SCORER, "collapse": SCORER}},
  "ablations": {
    "training": {"<arm>": {"<test>": [DRAW, ...]}},          A1 A2 A3 A4 A5 A6S A6L A7B1 A7B2 A7E3 A7E6 A8 A9S A9L A10
    "sampler":  {"<variant>": {"<test>": [DRAW, ...]}},      S1K1 S1K2 S1K5 S1K10 S1K25 S1K100 S2CHK S2FIX S3UNI S4G5k S4G10k S4G40k
    "input":    {"<variant>": {"<test>": [DRAW, ...]}},      R1Z25 R1Z50 R2T50 R2T25 A4R2T50 A4R2T25
    "reseated": {"<variant>": {"<test>": [DRAW]}},           first draw of a variant after reseating (mice 2 and 3)
    "subsets":  {"<percent>": {"<test>": {"ruler": "<subset ruler JSON>", "alone": [DRAW, ...], "among_all": [DRAW, ...]}}}
  }
}

DRAW = {"scorer": "<orbis.metrics.score JSON>", "matched": "<orbis.metrics.matched JSON>" (optional)}; SCORER = a scorer JSON.

Aggregation (the paper's): a training's value = the mean over its draws; a test's value = mean (SD, ddof 1) over trainings;
a geometry's value = per training the mean over its tests, then mean (SD) over trainings. Placement = 1 - per-cell error /
ruler (the within-type-shuffle ruler of the same test or subset).
"""
import json
import os

import numpy as np

_CACHE = {}


def load(path):
    """The registry with every path resolved."""
    base = os.path.dirname(os.path.abspath(path))
    reg = json.load(open(path))

    def res(o):
        if isinstance(o, dict):
            return {k: (os.path.join(base, v) if k in ("scorer", "matched", "ruler", "sampling") and isinstance(v, str) else res(v))
                    for k, v in o.items()}
        if isinstance(o, list):
            return [res(v) for v in o]
        return o
    reg = res(reg)
    if "controls" in reg:
        reg["controls"]["rulers"] = {k: os.path.join(base, v) for k, v in reg["controls"]["rulers"].items()}
    return reg


def jload(path):
    if path not in _CACHE:
        _CACHE[path] = json.load(open(path))
    return _CACHE[path]


def view_of(path, view):
    """The scorer JSON's metrics in the headline view."""
    V = jload(path)["views"]
    return V.get(view) or V["main"]


def layout_metrics(draw, view, ruler):
    """Every table metric of one layout (scorer JSON + optional matched-size JSON + the ruler JSON)."""
    v = view_of(draw["scorer"], view)
    size = [float(x) for x in v["size_axis"]]
    ts = v.get("type_spread_sm")
    per_cell = float(v["diag_typefit"]["err3d_median_um"])
    m = dict(size=size, size_err=float(np.mean([abs(1.0 - x) for x in size])),
             w2_sm=float(v["w2_um_sm"]) if v.get("w2_um_sm") is not None else None,
             f8ell_sm=None if v.get("f8ell_sm") is None else float(v["f8ell_sm"]),
             centre=None if not ts else float(ts["centre_err_um_mean"]),
             squeezed=None if not ts else float(ts["squeezed_share"]),
             per_cell_raw=per_cell, nmi=float(v["niche_nmi"]),
             contact_r_offdiag=None if v.get("contact_r_offdiag") is None else float(v["contact_r_offdiag"]),
             moran_r=None if v.get("moran_r") is None else float(v["moran_r"]),
             src_median=None if v.get("src_median") is None else float(v["src_median"]),
             sum_rssd_mm=None if v.get("sum_rssd_mm") is None else float(v["sum_rssd_mm"]))
    m["placement_raw"] = None if ruler is None else 1.0 - per_cell / jload(ruler)["ruler_raw_um"]
    mf = draw.get("matched")
    m["per_cell_matched"] = None
    if mf:
        mj = jload(mf)
        m["per_cell_matched"] = None if mj["err_sm_um"] is None else float(mj["err_sm_um"])
    for i in range(3):
        m[f"size_{'xyz'[i]}"] = size[i]
    return m


def mean_over(draws, key, view, ruler):
    """A row's value of one metric: the mean over its draws (None if any draw lacks it)."""
    vals = [layout_metrics(d, view, ruler)[key] for d in draws]
    if any(x is None for x in vals):
        return None
    return float(np.mean(vals))


def mean_axis(draws, view, ruler):
    """Per coordinate axis, the mean over the draws of SD(layout) / SD(truth)."""
    return np.mean([layout_metrics(d, view, ruler)["size"] for d in draws], 0)
