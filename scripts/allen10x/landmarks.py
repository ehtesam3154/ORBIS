#!/usr/bin/env python
"""Expression landmarks of the label-free pose readout on the five training stores: k-means (64 clusters) of the training cells'
features and the mean training position of each cluster (pose_readout.fit_landmarks).  usage: landmarks.py OUT_DIR"""
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
import numpy as np

import paths as P
import pose_readout as PR

STORES_ROOT = P.STORES
SUPPORT = P.SUPPORT
BRAINS = ["abca1", "abca2_ref1", "abca3", "abca4", "allen638850_hemi0"]
FROZEN_PR_MD5 = "8117725bf6318ac362a7ebb60448aa35"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


class ConcatFeat:
    def __init__(self, paths):
        self.mm = [np.load(p, mmap_mode="r") for p in paths]
        self.n = [len(m) for m in self.mm]
        self.off = np.concatenate([[0], np.cumsum(self.n)])
        assert all(m.dtype == np.float32 and m.shape[1] == 200 for m in self.mm)

    def __len__(self):
        return int(self.off[-1])

    def __getitem__(self, idx):
        idx = np.asarray(idx)
        out = np.empty((len(idx), 200), dtype=np.float32)
        for k, m in enumerate(self.mm):
            sel = np.flatnonzero((idx >= self.off[k]) & (idx < self.off[k + 1]))
            if len(sel):
                out[sel] = m[idx[sel] - self.off[k]]
        return out


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    feat = ConcatFeat([f"{STORES_ROOT}/{b}/feat.npy" for b in BRAINS])
    pos = np.load(f"{SUPPORT}/support_pos_mm.npy")
    brain = np.load(f"{SUPPORT}/support_brain.npy")
    assert len(feat) == len(pos) == len(brain) == 7_689_601, (len(feat), len(pos))
    assert (np.bincount(brain) == np.array(feat.n)).all()
    rng = np.random.default_rng(1)
    ii = np.sort(rng.choice(len(feat), 3000, replace=False))
    iso = {b: json.load(open(f"{STORES_ROOT}/{b}/meta.json"))["iso_mm"] for b in BRAINS}
    posm = [np.load(f"{STORES_ROOT}/{b}/pos.npy", mmap_mode="r") for b in BRAINS]
    rows = feat[ii]
    bad_f = bad_p = 0
    for i, row in zip(ii, rows):
        b = int(brain[i]); r = int(i - feat.off[b])
        bad_f += int(np.any(np.asarray(feat.mm[b][r]) != row))
        bad_p += int(np.abs(pos[i] - np.asarray(posm[b][r], dtype=np.float64) * iso[BRAINS[b]]).max() > 1e-4)
    print(f"spot check (3000 rows): feature mismatches {bad_f}, position mismatches {bad_p}", flush=True)
    assert bad_f == 0 and bad_p == 0
    print(f"[{time.strftime('%H:%M:%S')}] fitting KMeans(64, random_state=25000) on a uniform 1,000,000-cell sample of {len(feat):,} cells ...", flush=True)
    lm = PR.fit_landmarks(feat, pos, verbose=1)
    path = f"{out}/landmarks_k64.npz"
    PR.save_landmarks(path, lm)
    km = lm["kmeans"]
    sidx = lm["sample_idx"]
    per_brain = {BRAINS[b]: int((brain[sidx] == b).sum()) for b in range(5)}
    rep = dict(frozen_module_md5=FROZEN_PR_MD5, landmarks_file=path, landmarks_md5=md5(path), kmeans=km, converged=bool(km["n_iter"] < km["params"]["max_iter"]),
               sample_per_brain=per_brain, sample_idx_md5=hashlib.md5(sidx.astype(np.int64).tobytes()).hexdigest(), n_sample_per_cluster=lm["n_sample"].tolist(),
               landmark_training_mm=lm["landmark"].tolist(), stores=BRAINS, store_feat_files=[f"{STORES_ROOT}/{b}/feat.npy" for b in BRAINS],
               store_feat_sizes=feat.n, python=sys.version.split()[0], numpy=np.__version__, wall_seconds=round(time.time() - t0, 1),
               threads={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
    json.dump(rep, open(f"{out}/landmarks_k64_report.json", "w"), indent=1)
    print(f"[{time.strftime('%H:%M:%S')}] KMeans: sklearn {km['sklearn']} n_iter {km['n_iter']} (max_iter {km['params']['max_iter']}) converged {rep['converged']} inertia {km['inertia']:.6g} "
          f"cells fitted {km['n_cells_fitted']:,} of {km['n_train_total']:,}; per brain {per_brain}; cluster sizes min {lm['n_sample'].min()} max {lm['n_sample'].max()}; "
          f"landmarks md5 {rep['landmarks_md5']}; {rep['wall_seconds']} s", flush=True)


if __name__ == "__main__":
    main()
