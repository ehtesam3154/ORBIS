#!/usr/bin/env python
"""The 23 anatomical units (fixed unions of Allen CCF structures), the dissection-region -> unit table and the join of the
ABC parcellation of every training cell to a unit.  `python units.py` writes the unit tables to DATA_DIR/allen10x/frozen."""
import csv
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

import paths as P
CROSSWALK = f"{P.FROZEN}/roi_ccf_crosswalk.csv"
CROSSWALK_MD5 = "c005a25ce140c29cf383112e22899100"
GRAPH = f"{P.FROZEN}/ccf_structure_graph_1.json"
MEMBERSHIP = f"{P.MERFISH_RAW}/ccf2020_parcellation_to_parcellation_term_membership_acronym.csv"

UNITS = [
    ("MO-FRP", ["MO-FRP", "MOp"]),
    ("SS-GU-VISC", ["SS-GU-VISC", "SSp"]),
    ("AUD-TEa-PERI-ECT", ["AUD-TEa-PERI-ECT", "AUD", "TEa-PERI-ECT"]),
    ("VIS-PTLp", ["VIS-PTLp", "VIS"]),
    ("ACA", ["ACA"]), ("PL-ILA-ORB", ["PL-ILA-ORB"]), ("AI", ["AI"]), ("RSP", ["RSP"]),
    ("OLF", ["OLF"]), ("HIP", ["HIP"]), ("RHP", ["RHP", "ENT"]), ("CTXsp", ["CTXsp"]),
    ("STRd", ["STRd"]), ("STRv", ["STRv"]), ("LSX", ["LSX"]), ("sAMY", ["sAMY"]), ("PAL", ["PAL"]),
    ("TH", ["TH"]), ("HY", ["HY"]), ("MB", ["MB"]), ("P", ["P"]), ("MY", ["MY"]), ("CB", ["CB"]),
]
UNIT_NAMES = [u for u, _ in UNITS]
ROI_TO_UNIT = {r: u for u, rs in UNITS for r in rs}


def md5f(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def load_graph(path=GRAPH):
    msg = json.load(open(path))["msg"]
    nodes = {}

    def walk(n, parent):
        nodes[n["id"]] = dict(acronym=n["acronym"], name=n["name"], parent=parent, children=[c["id"] for c in n.get("children", [])],
                              st_level=n.get("st_level"))
        for c in n.get("children", []):
            walk(c, n["id"])

    for n in msg:
        walk(n, None)
    return nodes


def descendants(nodes, root_id):
    out, stack = [], [root_id]
    while stack:
        i = stack.pop()
        out.append(i)
        stack.extend(nodes[i]["children"])
    return out


def build(verbose=True):
    assert md5f(CROSSWALK) == CROSSWALK_MD5, "crosswalk md5 changed"
    cw = pd.read_csv(CROSSWALK)
    assert len(cw) == 29 and cw.region.is_unique
    nodes = load_graph()
    acr2id = {}
    for i, n in nodes.items():
        acr2id.setdefault(n["acronym"], []).append(i)
    roi_roots = {}
    for _, r in cw.iterrows():
        acs = str(r["ccf_component_acronyms"]).split(";")
        ids = [int(x) for x in str(r["ccf_structure_root_ids"]).split(";")]
        assert len(acs) == len(ids)
        for a, i in zip(acs, ids):
            assert a in acr2id and len(acr2id[a]) == 1, f"acronym {a} not unique in the graph"
            assert acr2id[a][0] == i, f"{r['region']}: graph id of {a} is {acr2id[a][0]}, crosswalk says {i}"
        roi_roots[r["region"]] = list(zip(acs, ids))
    assert set(roi_roots) == set(ROI_TO_UNIT), (set(roi_roots) ^ set(ROI_TO_UNIT))
    unit_ids, unit_roots = [], []
    for u, rois in UNITS:
        roots = sorted({x for r in rois for x in roi_roots[r]}, key=lambda t: t[1])
        ids = set()
        for a, i in roots:
            ids.update(descendants(nodes, i))
        for r in rois:
            s = set()
            for a, i in roi_roots[r]:
                s.update(descendants(nodes, i))
            assert s <= ids
        unit_ids.append(ids)
        unit_roots.append(roots)
    seen = {}
    for k, ids in enumerate(unit_ids):
        for i in ids:
            assert i not in seen, f"structure {nodes[i]['acronym']} ({i}) is in units {UNIT_NAMES[seen[i]]} and {UNIT_NAMES[k]}"
            seen[i] = k
    acr_unit = {nodes[i]["acronym"]: k for i, k in seen.items()}
    pm = pd.read_csv(MEMBERSHIP)
    unit_of_parc = np.full(len(pm), -1, dtype=np.int16)
    how = []
    unmatched = []
    for j, r in enumerate(pm.itertuples(index=False)):
        sub, st = str(r.substructure), str(r.structure)
        if sub in acr_unit:
            unit_of_parc[j] = acr_unit[sub]; how.append("substructure")
        elif sub in acr2id:
            how.append("substructure_in_graph_no_unit")
        elif st in acr_unit:
            unit_of_parc[j] = acr_unit[st]; how.append("structure_fallback")
        elif st in acr2id:
            how.append("structure_in_graph_no_unit")
        elif str(r.division) in acr_unit and str(r.division) in UNIT_NAMES:
            unit_of_parc[j] = acr_unit[str(r.division)]; how.append("division_fallback")
        else:
            how.append("unmatched"); unmatched.append((int(r.parcellation_index), sub, st, str(r.division)))
    pm = pm.assign(unit=unit_of_parc, join=how)
    out = dict(unit_names=UNIT_NAMES, roi_to_unit=ROI_TO_UNIT, unit_roots=unit_roots, unit_ids=unit_ids, nodes=nodes, acr_unit=acr_unit,
               parc=pm, unmatched=unmatched, crosswalk=cw, roi_roots=roi_roots)
    if verbose:
        print(f"[units] graph nodes {len(nodes)}; 23 units; structures with a unit {len(seen)}; parcellations {len(pm)}: "
              f"unit assigned {int((unit_of_parc >= 0).sum())}, by join {pd.Series(how).value_counts().to_dict()}", flush=True)
    return out


if __name__ == "__main__":
    B = build()
    res = P.FROZEN
    nodes, pm = B["nodes"], B["parc"]
    rows = []
    for k, (u, rois) in enumerate(UNITS):
        ids = sorted(B["unit_ids"][k])
        rows.append(dict(unit_index=k, unit=u, member_roi_names=";".join(rois), root_acronyms=";".join(a for a, _ in B["unit_roots"][k]),
                         root_ids=";".join(str(i) for _, i in B["unit_roots"][k]), n_structures_incl_descendants=len(ids),
                         n_parcellations=int((pm.unit == k).sum()),
                         structure_acronyms=";".join(sorted(nodes[i]["acronym"] for i in ids))))
    pd.DataFrame(rows).to_csv(f"{res}/units_23.csv", index=False)
    pd.DataFrame([dict(roi=r, unit=u, unit_index=UNIT_NAMES.index(u), root_acronyms=";".join(a for a, _ in B["roi_roots"][r]),
                       root_ids=";".join(str(i) for _, i in B["roi_roots"][r])) for r, u in ROI_TO_UNIT.items()]).to_csv(f"{res}/roi_to_unit_29.csv", index=False)
    pm[["parcellation_index", "division", "structure", "substructure", "unit", "join"]].to_csv(f"{res}/parcellation_to_unit.csv", index=False)
    pd.DataFrame(B["unmatched"], columns=["parcellation_index", "substructure", "structure", "division"]).to_csv(f"{res}/parcellation_unmatched_to_graph.csv", index=False)
    print("unit sizes (parcellations):", {UNIT_NAMES[k]: int((pm.unit == k).sum()) for k in range(23)})
    print("join summary:", pm["join"].value_counts().to_dict())
    print("parcellations without a unit by division:", pm[pm.unit < 0].division.value_counts().to_dict())
    print("unmatched to graph:", B["unmatched"][:20])
