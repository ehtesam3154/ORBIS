"""Nested random subsets (50 % > 25 % > 10 % of the cells) of a test prebuilt, written as loadable prebuilt dirs.
perm = default_rng(1031).permutation(N) of the original rows; subset P = sort(perm[:round(P / 100 N)]), so every subset
keeps the original row order and sub10 is inside sub25 is inside sub50. Every per-cell .npy is subset with the same
rows; offsets = per original graph the number of kept rows; meta.json gets total_cells / sizes updated and an
r3_subset block; infos.pkl is copied; keep_idx.npy = the kept rows (int64, original order).
Output: OUT_ROOT/<NAME>_sub<P>/ and the summary OUT_ROOT/<NAME>.json.
usage: python scripts/ablations/make_subsets.py SRC_DIR OUT_ROOT NAME"""
import hashlib
import json
import os
import shutil
import sys

import numpy as np

SEED = 1031
PCTS = (50, 25, 10)


def md5_file(path, chunk=1 << 24):
    """md5 of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def keep_sets(N):
    """The kept rows of every subset."""
    perm = np.random.default_rng(SEED).permutation(N)
    return {p: np.sort(perm[:int(round(p / 100.0 * N))]).astype(np.int64) for p in PCTS}


def subset_npy(src, dst, keep, chunk=20000):
    """Write src[keep] to dst and check it bitwise."""
    a = np.load(src, mmap_mode="r")
    out = np.lib.format.open_memmap(dst, mode="w+", dtype=a.dtype, shape=(len(keep),) + a.shape[1:])
    for s in range(0, len(keep), chunk):
        out[s:s + chunk] = a[keep[s:s + chunk]]
    out.flush()
    del out
    b = np.load(dst, mmap_mode="r")
    for s in range(0, len(keep), 200000):
        assert np.array_equal(np.asarray(b[s:s + 200000]), np.asarray(a[keep[s:s + 200000]])), (dst, s)


def build(src, out_root, name):
    """All subsets of one prebuilt."""
    off = np.load(f"{src}/offsets.npy").astype(np.int64)
    N = int(off[-1])
    meta = json.load(open(f"{src}/meta.json"))
    assert int(meta["total_cells"]) == N, (meta["total_cells"], N)
    files = sorted(os.listdir(src))
    per_cell = []
    for fn in files:
        if fn.endswith(".npy") and fn != "offsets.npy":
            sh = np.load(f"{src}/{fn}", mmap_mode="r").shape
            assert sh[0] == N, f"{name} {fn}: first dim {sh[0]} != N {N}: cannot subset"
            per_cell.append(fn)
    other = [f for f in files if f not in per_cell and f not in ("offsets.npy", "meta.json")]
    assert set(other) <= {"infos.pkl"}, f"{name}: unhandled files {other}"
    cid = np.load(f"{src}/cell_ID.npy", mmap_mode="r")
    cid_is_arange = bool(np.array_equal(np.asarray(cid), np.arange(N)))
    K = keep_sets(N)
    assert set(K[10]) <= set(K[25]) <= set(K[50])
    summ = dict(ds=name, source=src, source_realpath=os.path.realpath(src), N=N, seed=SEED,
                rule="perm = np.random.default_rng(1031).permutation(N); keep_P = np.sort(perm[:round(P/100 * N)]) (nested prefixes, original row order)",
                per_cell_files=per_cell, copied=other, cell_ID_is_arange=cid_is_arange, subsets={})
    for p in PCTS:
        keep = K[p]
        d = f"{out_root}/{name}_sub{p}"
        os.makedirs(d, exist_ok=True)
        for fn in per_cell:
            subset_npy(f"{src}/{fn}", f"{d}/{fn}", keep)
        if cid_is_arange:
            assert np.array_equal(np.load(f"{d}/cell_ID.npy"), keep), "cell_ID != keep_idx"
        cnt = np.searchsorted(keep, off[1:]) - np.searchsorted(keep, off[:-1])
        off_new = np.r_[0, np.cumsum(cnt)].astype(off.dtype)
        assert off_new[-1] == len(keep)
        np.save(f"{d}/offsets.npy", off_new)
        np.save(f"{d}/keep_idx.npy", keep)
        for fn in other:
            shutil.copyfile(os.path.realpath(f"{src}/{fn}"), f"{d}/{fn}")
        m = dict(meta)
        m["total_cells"] = int(len(keep))
        if "sizes" in m:
            assert len(m["sizes"]) == len(cnt) and list(map(int, m["sizes"])) == np.diff(off).tolist()
            m["sizes"] = [int(c) for c in cnt]
        m["r3_subset"] = dict(source=src, pct=p, seed=SEED, n_source=N, n_kept=int(len(keep)), keep_idx="keep_idx.npy",
                              keep_idx_md5=md5_file(f"{d}/keep_idx.npy"), rule=summ["rule"],
                              note="every per-cell array = source[keep_idx]")
        json.dump(m, open(f"{d}/meta.json", "w"), indent=1)
        n_empty = int((cnt == 0).sum())
        summ["subsets"][p] = dict(dir=d, n_kept=int(len(keep)), share=len(keep) / N, keep_idx_md5=m["r3_subset"]["keep_idx_md5"],
                                  n_graphs=int(len(cnt)), n_empty_graphs=n_empty, min_graph=int(cnt.min()), max_graph=int(cnt.max()),
                                  pos_md5=md5_file(f"{d}/pos.npy"), cell_class_md5=md5_file(f"{d}/cell_class.npy"))
        print(f"{name} sub{p}: kept {len(keep):,} of {N:,} cells ({len(keep) / N:.4f}); graphs {len(cnt)} (cells per graph {cnt.min()}..{cnt.max()}, "
              f"empty {n_empty}) -> {d}", flush=True)
    os.makedirs(out_root, exist_ok=True)
    json.dump(summ, open(f"{out_root}/{name}.json", "w"), indent=1)


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2], sys.argv[3])
