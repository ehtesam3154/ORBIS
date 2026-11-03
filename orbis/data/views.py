"""Training views drawn afresh at every step from per-specimen cell stores (feat.npy, pos.npy, sec.npy, cls.npy, meta.json).

A view is WHOLE (n random cells of one specimen), SECTION (one section) or BLOCK (n random cells of k consecutive sections).
Its label tells the network how it was made (WHOLE 0, SECTION coronal / sagittal 1 / 2, BLOCK coronal / sagittal 3 / 4);
sampling always uses WHOLE. Counts are binomially thinned with a keep probability drawn per view.
"""
import json
import os

import numpy as np
import torch

from orbis.data.holder import to_batch, to_device

WHOLE, SECTION, BLOCK = 0, 1, 2
N_KINDS = 3
N_VIEW_TYPES = 5


def view_label(kind, sagittal):
    """Integer label of a view kind; planar kinds also encode whether the specimen was cut sagitally."""
    return 0 if kind == WHOLE else (1 if kind == SECTION else 3) + int(bool(sagittal))


class CellStore:
    """One specimen's cells: features (memory-mapped), positions, section index and class per cell."""

    def __init__(self, d):
        self.dir = d
        self.meta = json.load(open(os.path.join(d, "meta.json")))
        self.feat = np.load(os.path.join(d, "feat.npy"), mmap_mode="r")
        self.counts = self.feat.dtype == np.uint8
        self.pos = np.load(os.path.join(d, "pos.npy"))
        self.sec = np.load(os.path.join(d, "sec.npy"))
        self.cls = np.load(os.path.join(d, "cls.npy"))
        self.n = len(self.pos)
        order = np.argsort(self.sec, kind="stable")
        bounds = np.searchsorted(self.sec[order], np.arange(self.sec.max() + 2))
        self.sec_rows = [order[bounds[s]:bounds[s + 1]] for s in range(self.sec.max() + 1)]
        self.n_sec = len(self.sec_rows)
        if "sec_depth_um" in self.meta:
            dep = np.asarray(self.meta["sec_depth_um"], dtype=float)
            assert len(dep) == self.n_sec, f"sec_depth_um has {len(dep)} entries for {self.n_sec} sections in {d}"
            self.sec_order = np.argsort(dep)
            self.sagittal = bool(self.meta.get("sagittal", False))
        else:
            cen = np.array([self.pos[r].mean(0) for r in self.sec_rows])
            ax = int(np.argmax(cen.std(0)))
            self.sec_order = np.argsort(cen[:, ax])
            self.sagittal = ax == 0
            assert ax in (0, 2), f"unexpected cut axis {ax} in {d}"
        self.whole_ok = self.meta.get("whole_ok", True)


class FreshViews:
    """Random views of a set of cell stores from one numpy random stream."""

    def __init__(self, store_dirs, mix=(0.5, 0.25, 0.25), n_range=(10000, 20000), thin_min=0.55, block_k=(2, 64),
                 whole_animals=None, seed=0, fold_axis=-1):
        self.stores = [CellStore(d) for d in store_dirs]
        self.fold_axis = int(fold_axis)
        assert self.fold_axis in (-1, 0, 1, 2)
        self.mix = np.asarray(mix, dtype=float) / np.sum(mix)
        assert len(self.mix) == N_KINDS, "mix = probabilities of WHOLE, SECTION, BLOCK"
        self.n_range, self.thin_min, self.block_k = n_range, float(thin_min), block_k
        sizes = np.array([s.n for s in self.stores], dtype=float)
        wa = whole_animals if whole_animals is not None else [i for i, s in enumerate(self.stores) if s.whole_ok]
        w = np.zeros(len(self.stores))
        w[wa] = sizes[wa]
        self.p_whole = w / w.sum()
        self.p_any = sizes / sizes.sum()
        self.rng = np.random.default_rng(seed)
        self.n_genes = self.stores[0].feat.shape[1]
        assert all(s.feat.shape[1] == self.n_genes for s in self.stores)
        assert all(s.meta["gene_names"] == self.stores[0].meta["gene_names"] for s in self.stores), "gene order differs"

    def draw(self):
        """One view: sorted rows, store index, view label, keep probability for thinning."""
        r = self.rng
        vt = int(r.choice(N_KINDS, p=self.mix))
        n = int(r.integers(self.n_range[0], self.n_range[1] + 1))
        if vt == WHOLE:
            a = int(r.choice(len(self.stores), p=self.p_whole))
            S = self.stores[a]
            rows = r.choice(S.n, min(n, S.n), replace=False)
        elif vt == SECTION:
            a = int(r.choice(len(self.stores), p=self.p_any))
            S = self.stores[a]
            si = int(r.integers(S.n_sec))
            cand = S.sec_rows[si]
            n = self.n_range[1]
            rows = cand if len(cand) <= n else r.choice(cand, n, replace=False)
        else:
            a = int(r.choice(len(self.stores), p=self.p_any))
            S = self.stores[a]
            k = int(r.integers(self.block_k[0], min(self.block_k[1], S.n_sec) + 1))
            start = int(r.integers(0, S.n_sec - k + 1))
            cand = np.concatenate([S.sec_rows[s] for s in S.sec_order[start:start + k]])
            rows = cand if len(cand) <= n else r.choice(cand, n, replace=False)
        keep = float(r.uniform(self.thin_min, 1.0)) if self.thin_min < 1.0 else 1.0
        return np.sort(rows), a, view_label(vt, self.stores[a].sagittal), keep

    def item(self):
        """One view as CPU tensors; with fold_axis >= 0 positions are folded across that axis (x -> |x|)."""
        rows, a, vt, keep = self.draw()
        S = self.stores[a]
        pos = np.ascontiguousarray(S.pos[rows])
        if self.fold_axis >= 0:
            pos = np.array(pos, copy=True)
            pos[:, self.fold_axis] = np.abs(pos[:, self.fold_axis])
        f = torch.from_numpy(np.ascontiguousarray(S.feat[rows]))
        return dict(feat=f if S.counts else f.float(), pos=torch.from_numpy(pos).float(),
                    cls=torch.from_numpy(S.cls[rows].astype(np.int64)), rows=torch.from_numpy(rows.astype(np.int64)),
                    vt=vt, keep=keep, animal=a, counts=bool(S.counts))


class FreshViewDataset(torch.utils.data.IterableDataset):
    """Endless stream of fresh views; DataLoader worker w draws from seed + 1000 w."""

    def __init__(self, store_dirs, seed=0, **kw):
        super().__init__()
        self.store_dirs, self.seed, self.kw = list(store_dirs), int(seed), kw

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        fv = FreshViews(self.store_dirs, seed=self.seed + 1000 * (wi.id if wi is not None else 0), **self.kw)
        while True:
            yield fv.item()


def to_gpu_batch(item, dev):
    """A view as a dense one-graph batch on dev (masked, centred), with thinned features log2(1 + counts) and its label.

    Count stores (uint8) are thinned directly; log2(1 + count) stores are converted back to counts first."""
    bt = to_device(to_batch(item["feat"], item["pos"], item["cls"], item["rows"]), dev)
    keep = float(item["keep"])
    if item.get("counts", False):
        c = bt.node_features.float()
        if keep < 1.0:
            c = torch.binomial(c, torch.full_like(c, keep))
        bt.node_features = torch.log2(1.0 + c) * bt.node_mask.unsqueeze(-1).to(c.dtype)
    elif keep < 1.0:
        c = torch.round(torch.exp2(bt.node_features) - 1.0).clamp_min(0)
        c = torch.binomial(c, torch.full_like(c, keep))
        bt.node_features = torch.log2(1.0 + c) * bt.node_mask.unsqueeze(-1).to(c.dtype)
    bt.view_type = torch.tensor([int(item["vt"])], device=dev, dtype=torch.long)
    bt.thin, bt.animal = keep, int(item["animal"])
    return bt


def unit_box_positions(pos, mask):
    """Rescale one view's positions per axis to [-0.5, 0.5] (min-max on this view), then re-centre to mean 0.

    pos (1, N, 3), mask (1, N) bool; computed in float64. A zero-extent axis stays at 0.
    Returns (positions (1, N, 3), per-axis extent (3,), number of zero-extent axes)."""
    m = mask[0]
    p = pos[0][m].double()
    lo, hi = p.min(0).values, p.max(0).values
    ext = hi - lo
    guard = ext <= 0
    q = (p - lo) / torch.where(guard, torch.ones_like(ext), ext) - 0.5
    q = torch.where(guard, torch.zeros_like(q), q)
    q = q - q.mean(0, keepdim=True)
    out = torch.zeros_like(pos)
    out[0][m] = q.to(pos.dtype)
    return out, ext, int(guard.sum())
