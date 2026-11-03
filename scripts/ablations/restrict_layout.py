"""A stored layout restricted to the cells of a subset (keep_idx.npy of scripts/ablations/make_subsets.py): the
unperturbed model's layout on exactly the cells the subset keeps.
usage: python scripts/ablations/restrict_layout.py LAYOUT_NPZ KEEP_IDX_NPY OUT_NPZ"""
import sys

import numpy as np

z = np.load(sys.argv[1]); k = np.load(sys.argv[2])
np.savez_compressed(sys.argv[3], P=z['P'][k], G=z['G'][k], cc=np.asarray(z['cc'])[k], iso=z['iso'])
