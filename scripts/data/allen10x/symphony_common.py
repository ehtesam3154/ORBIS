"""Shared pieces of the Symphony reference build and the query mapping: input locations, the expression transform
(stored log2(1 + count) -> counts -> library size over the 477 panel genes to 1e4 -> log1p) and small utilities.
The input locations are set by the calling script from its command line (configure)."""
import json

import numpy as np

ST = None
AL = None
SIGN_PCA = None
PANEL = None
BRAINS = ("abca1", "abca2", "abca3", "abca4", "allen")
ZHUANG = BRAINS[:4]
EXPECTED_CELLS = {"abca1": 2846908, "abca2": 1227408, "abca3": 1585782, "abca4": 162578, "allen": 3739961}
LIB = 1e4
CLIP = 10.0
NPC = 200
NGENES = 477


def configure(stores_dir, allen_dir, panel, sign_pca=None):
    """Set the input locations: Zhuang cell stores (1,122 genes), Allen 477-gene store, panel gene list, sign-reference PCA."""
    global ST, AL, PANEL, SIGN_PCA
    ST, AL, PANEL, SIGN_PCA = stores_dir, allen_dir, panel, sign_pca


def panel_cols():
    """Column indices and Ensembl ids of the 477 panel genes inside the 1,122-gene Zhuang stores."""
    names = json.load(open(f"{ST}/abca1/meta.json"))["gene_names"]
    for m in ZHUANG[1:]:
        assert json.load(open(f"{ST}/{m}/meta.json"))["gene_names"] == names, m
    panel = [x.strip() for x in open(PANEL) if x.strip()]
    cols = np.array(sorted(names.index(x) for x in panel))
    gn = [names[c] for c in cols]
    assert len(cols) == NGENES and json.load(open(f"{AL}/meta.json"))["gene_names"] == gn
    return cols, gn


def sources():
    """(brain, feature file, panel columns or None) in the reference row order."""
    cols, gn = panel_cols()
    src = [(m, f"{ST}/{m}/feat.npy", cols) for m in ZHUANG] + [("allen", f"{AL}/feat.npy", None)]
    return src, gn


def counts_from_feat(x):
    """Stored log2(1 + count) -> count in float64, not rounded."""
    return np.exp2(np.asarray(x, dtype=np.float64)) - 1.0


def lognorm(c):
    """Counts (n, 477) -> log1p(c / max(library over the 477 genes, 1) * 1e4); also returns the library sizes."""
    s = c.sum(1, keepdims=True)
    return np.log1p(c / np.maximum(s, 1.0) * LIB), s[:, 0]


def read_counts(path, cols, lo=None, hi=None, idx=None):
    f = np.load(path, mmap_mode="r")
    x = f[lo:hi] if idx is None else f[idx]
    x = x[:, cols] if cols is not None else x
    return counts_from_feat(x)


def rss_gb():
    """Current and peak resident memory of this process in GB."""
    cur = hwm = 0.0
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            cur = int(line.split()[1]) / 1e6
        elif line.startswith("VmHWM:"):
            hwm = int(line.split()[1]) / 1e6
    return cur, hwm
