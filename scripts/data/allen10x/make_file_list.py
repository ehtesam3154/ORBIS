#!/usr/bin/env python
"""List every WMB-10Xv3 raw expression file (v3 only, *-raw.h5ad) of the official ABC release manifest with its URL, size and md5.
usage: make_file_list.py MANIFEST.json   ->  OUT_DIR/v3_raw_file_list.{tsv,json}  (DATA_DIR/allen_10x is the download target)"""
import json, shutil, sys, hashlib, datetime, os
mf = sys.argv[1]
m = json.load(open(mf))
md5 = hashlib.md5(open(mf, "rb").read()).hexdigest()
fl = m["file_listing"]["WMB-10Xv3"]["expression_matrices"]
rows = []
for k in sorted(fl):
    f = fl[k]["raw"]["files"]["h5ad"]
    assert f["relative_path"].endswith("-raw.h5ad") and "WMB-10Xv3" in f["relative_path"] and "v2" not in f["relative_path"].lower() and "Multi" not in f["relative_path"]
    rows.append(dict(name=k, url=f["url"], bytes=int(f["size"]), md5_manifest=f["file_hash"], release=f["version"], relative_path=f["relative_path"]))
tot = sum(r["bytes"] for r in rows)
OUT = os.environ["OUT_DIR"]
with open(f"{OUT}/v3_raw_file_list.tsv", "w") as fh:
    for r in rows:
        fh.write(f"{r['name']}\t{r['url']}\t{r['bytes']}\t{r['md5_manifest']}\n")
du = shutil.disk_usage(os.path.join(os.environ["DATA_DIR"], "raw"))
info = dict(manifest=m["version"], manifest_file=mf, manifest_md5=md5, manifest_resource_uri=m["resource_uri"],
            expression_release=sorted({r["release"] for r in rows}), n_files=len(rows), total_bytes=tot, total_gb=tot / 1e9, total_gib=tot / 2**30,
            free_bytes_before_download=du.free, free_tb_before_download=du.free / 1e12,
            time=datetime.datetime.now().isoformat(timespec="seconds"), files=rows)
json.dump(info, open(f"{OUT}/v3_raw_file_list.json", "w"), indent=1)
lines = [f"ABC release manifest {m['version']} ({mf}, md5 {md5}); expression release(s) {info['expression_release']}; {len(rows)} WMB-10Xv3 RAW h5ad files (log2 files, v2, Multiome excluded)"]
for r in rows:
    lines.append(f"  {r['name']:24s} {r['bytes']:>16,d} B  md5(manifest) {r['md5_manifest']}  {r['url']}")
lines.append(f"TOTAL {tot:,d} B = {tot / 1e9:.2f} GB = {tot / 2**30:.2f} GiB")
lines.append(f"FREE before download: {du.free:,d} B = {du.free / 1e12:.2f} TB (size {du.total / 1e12:.2f} TB, used {du.used / 1e12:.2f} TB); download uses {100 * tot / du.free:.1f} % of the free space")
print("\n".join(lines))
