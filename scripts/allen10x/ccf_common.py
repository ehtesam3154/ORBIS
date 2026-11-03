#!/usr/bin/env python
"""Constants and small utilities of the official-annotation scorer (placed cell -> official Allen CCF coordinate -> annotation unit)."""
import hashlib
import json
import os
import time

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
import numpy as np

import paths as P

APP = P.APP
V = P.CCF_VALIDATION
RES = P.RESULTS
SUPPORT = P.SUPPORT
CCF_DIR = f"{APP}/ccf_translation"
LAYOUT_CCF = f"{APP}/layouts_ccf"
POOL = P.POOL
ANNOT = f"{P.FROZEN}/annotation_25.nrrd"
GRAPH = f"{P.FROZEN}/ccf_structure_graph_1.json"
S_CAL = np.array([1.0185775453536927, 1.2931099417291327, 0.9995762396395721])
K_NN = 15
SEED_BASE = 25400
ARMS = ("ours_raw", "ours_readout", "baseline_readout")
KINDS = ("primary", "strat")
COHORTS = (1, 2, 3, 4, 5)
SEEDS = (1000, 1001, 1002)
REF_SEEDS = tuple(range(26001, 26011))
BRAINS = ["abca1", "abca2_ref1", "abca3", "abca4", "allen638850_hemi0"]
N_POOL = 2_341_350
AXES = ("AP", "DV", "ML")


def t():
    return time.strftime("%H:%M:%S")


def log(*a):
    print(f"[{t()}]", *a, flush=True)


def md5f(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def memavail_gb():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable:"):
            return int(l.split()[1]) / 1e6
    return float("nan")


def need_mem(gb, floor=14.0, wait_s=600):
    t0 = time.time()
    while memavail_gb() - gb < floor:
        if time.time() - t0 > wait_s:
            raise SystemExit(f"MemAvailable {memavail_gb():.1f} GB < need {gb:.1f} + floor {floor:.1f} GB after {wait_s} s")
        time.sleep(20)


def jdump(o, path):
    def conv(x):
        if isinstance(x, dict):
            return {str(k): conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        if isinstance(x, np.ndarray):
            return conv(x.tolist())
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return float(x)
        if isinstance(x, np.bool_):
            return bool(x)
        return x
    with open(path, "w") as f:
        json.dump(conv(o), f, indent=1)


def peak_rss_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def layout_names(kinds=KINDS, arms=ARMS, cohorts=COHORTS, seeds=SEEDS):
    return [f"{a}_{k}{c}_s{s}" for k in kinds for a in arms for c in cohorts for s in seeds]
