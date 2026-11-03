#!/usr/bin/env python
"""The official Allen CCFv3 (2017) annotation volume at 25 um ([AP, DV, ML], voxel index = floor(mm / 0.025)), its lookup and the
structure id -> unit table.  `python ccf_annotation.py check` writes the convention check."""
import json
import sys

import numpy as np

import ccf_common as C
import units as EU

VOX_MM = 0.025


class Annotation:
    def __init__(self, path=C.ANNOT):
        import nrrd
        self.vol, self.header = nrrd.read(path)
        self.vol = np.ascontiguousarray(self.vol.astype(np.uint32))
        self.shape = np.array(self.vol.shape)
        B = EU.build(verbose=False)
        self.nodes = B["nodes"]
        self.unit_names = B["unit_names"]
        graph_ids = np.array(sorted(self.nodes), dtype=np.int64)
        unit_of = np.full(len(graph_ids), -1, np.int16)
        pos = {int(i): k for k, i in enumerate(graph_ids)}
        for u, ids in enumerate(B["unit_ids"]):
            for i in ids:
                unit_of[pos[int(i)]] = u
        self.graph_ids, self.unit_of_graph = graph_ids, unit_of
        self.present, inv = np.unique(self.vol, return_inverse=True)
        self.inv = inv.reshape(self.vol.shape).astype(np.int32)
        gi = np.searchsorted(graph_ids, self.present.astype(np.int64))
        gi_c = np.clip(gi, 0, len(graph_ids) - 1)
        self.in_graph = (graph_ids[gi_c] == self.present.astype(np.int64))
        self.unit_present = np.where(self.in_graph, unit_of[gi_c], -1).astype(np.int16)
        self.unit_present[self.present == 0] = -1
        self.acr_present = [self.nodes[int(i)]["acronym"] if g else ("" if i == 0 else f"?{int(i)}") for i, g in zip(self.present, self.in_graph)]
        self.unit_vol = self.unit_present[self.inv].astype(np.int8)

    def index(self, xyz_mm):
        x = np.asarray(xyz_mm, dtype=np.float64)
        fin = np.isfinite(x).all(1)
        idx = np.floor(np.where(fin[:, None], x, -1.0) / VOX_MM).astype(np.int64)
        ok = fin & (idx >= 0).all(1) & (idx < self.shape[None, :]).all(1)
        return idx, ok

    def lookup(self, xyz_mm):
        idx, ok = self.index(xyz_mm)
        cmp_ = np.full(len(idx), -1, np.int64)
        cmp_[ok] = self.inv[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
        ids = np.zeros(len(idx), np.uint32)
        ids[ok] = self.present[cmp_[ok]]
        unit = np.full(len(idx), -1, np.int16)
        unit[ok] = self.unit_present[cmp_[ok]]
        return ids, unit, ok

    def compact(self, xyz_mm):
        idx, ok = self.index(xyz_mm)
        cmp_ = np.full(len(idx), -1, np.int64)
        cmp_[ok] = self.inv[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
        return cmp_

    def descendants_mask(self, root_acronym):
        rid = [i for i, n in self.nodes.items() if n["acronym"] == root_acronym]
        assert len(rid) == 1, (root_acronym, rid)
        d = set(EU.descendants(self.nodes, rid[0]))
        return np.array([int(i) in d for i in self.present]), rid[0]


def centroid_mm(mask):
    ii = np.argwhere(mask)
    return ((ii.mean(0) + 0.5) * VOX_MM).tolist()


def check(out_json=f"{C.RES}/ccf_annotation_check.json"):
    A = Annotation()
    rep = dict(created=C.t(), annotation_file=C.ANNOT, annotation_md5=C.md5f(C.ANNOT), shape=A.shape.tolist(), header={k: str(v) for k, v in A.header.items()},
               n_ids_present=int(len(A.present)), ids_not_in_graph=[int(i) for i, g in zip(A.present, A.in_graph) if not g and i != 0],
               voxels_nonzero=int((A.vol != 0).sum()), voxels_in_a_unit=int((A.unit_vol >= 0).sum()), voxels_nonzero_no_unit=int(((A.vol != 0) & (A.unit_vol < 0)).sum()))
    nz = A.vol != 0
    mir = nz[:, :, ::-1]
    dice = float(2 * (nz & mir).sum() / (nz.sum() + mir.sum()))
    un = {n: centroid_mm(A.unit_vol == k) for k, n in enumerate(A.unit_names) if n in ("OLF", "TH", "CB", "HY", "MB")}
    isoc, iso_id = A.descendants_mask("Isocortex")
    iso_vol = isoc[A.inv]
    un["Isocortex"] = centroid_mm(iso_vol)
    p1 = dict(shape_ok=bool(tuple(A.shape) == (528, 320, 456)), mirror_dice_about_ML_5p7=dice, centroids_AP_DV_ML_mm=un,
              AP_order_OLF_lt_TH_lt_CB=bool(un["OLF"][0] < un["TH"][0] < un["CB"][0]), DV_order_Isocortex_lt_HY=bool(un["Isocortex"][1] < un["HY"][1]))
    p1["pass"] = bool(p1["shape_ok"] and dice >= 0.98 and p1["AP_order_OLF_lt_TH_lt_CB"] and p1["DV_order_Isocortex_lt_HY"])
    rep["truth_free"] = p1
    xyz = np.load(f"{C.CCF_DIR}/support_ccf_xyz_mm.npy", mmap_mode="r")
    unit = np.load(f"{C.SUPPORT}/support_unit.npy")
    ok = np.flatnonzero(np.isfinite(np.asarray(xyz[:, 0])) & (unit >= 0))
    rng = np.random.default_rng(C.SEED_BASE)
    pick = np.sort(rng.choice(ok, 200_000, replace=False))
    X = np.asarray(xyz[pick], dtype=np.float64)
    u0 = unit[pick].astype(np.int16)
    extent = A.shape * VOX_MM
    import itertools
    res = {}
    for perm in itertools.permutations(range(3)):
        for flips in itertools.product((0, 1), repeat=3):
            Xp = X[:, list(perm)].copy()
            for a in range(3):
                if flips[a]:
                    Xp[:, a] = extent[a] - Xp[:, a]
            _, un_, okk = A.lookup(Xp)
            res[f"perm{perm}_flip{flips}"] = dict(agree=float((un_ == u0).mean()), inside=float(okk.mean()))
    ident = res["perm(0, 1, 2)_flip(0, 0, 0)"]
    others = {k: v["agree"] for k, v in res.items() if k != "perm(0, 1, 2)_flip(0, 0, 0)"}
    best_other = max(others.items(), key=lambda kv: kv[1])
    p2 = dict(n_cells=int(len(pick)), identity_agreement=ident["agree"], identity_inside_volume=ident["inside"], best_other=dict(name=best_other[0], agreement=best_other[1]),
              gap_to_best_other=ident["agree"] - best_other[1], all=res)
    p2["pass"] = bool(ident["agree"] >= 0.80 and ident["agree"] - best_other[1] >= 0.2)
    sup_brain = np.load(f"{C.SUPPORT}/support_brain.npy")[pick]
    _, un_id, _ = A.lookup(X)
    p2["identity_agreement_per_brain"] = {C.BRAINS[b]: float((un_id[sup_brain == b] == u0[sup_brain == b]).mean()) for b in range(5) if (sup_brain == b).any()}
    rep["support_cell_axis_check"] = p2
    rep["PASS"] = bool(p1["pass"] and p2["pass"])
    C.jdump(rep, out_json)
    C.log(f"truth-free pass {p1['pass']} (mirror Dice {dice:.4f}, AP order {p1['AP_order_OLF_lt_TH_lt_CB']}, DV order {p1['DV_order_Isocortex_lt_HY']}); "
          f"axis check: identity {ident['agree']:.4f}, best other {best_other[0]} {best_other[1]:.4f}, gap {p2['gap_to_best_other']:.4f} -> pass {p2['pass']}; OVERALL {rep['PASS']}")
    return rep


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check()
