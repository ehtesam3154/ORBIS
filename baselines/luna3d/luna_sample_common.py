"""Helpers of the expression-only LUNA-3D sampler: file-open audit for the sidecar, the seeded graph partition, md5 and memory helpers.
"""
import hashlib
import json
import os
import re
import resource
import sys
import numpy as np
_OPENED = []
_IGNORE = re.compile(r"(\.py|\.pyc|\.pyi|\.pth|\.cfg|\.typed|\.mo|\.cache)$|\.so(\.\d+)*$|^/(proc|sys|dev|etc|usr|lib|lib64|run|var)/|__pycache__|/site-packages/|/dist-packages/")
def install_audit_hook():
    def hook(event, args):
        if event == "open":
            p = args[0]
            if isinstance(p, (str, bytes)) and not isinstance(p, int):
                p = os.fsdecode(p)
                if not _IGNORE.search(p):
                    _OPENED.append((p, str(args[1]) if len(args) > 1 else ""))
    sys.addaudithook(hook)
def opened_files():
    return sorted(set(_OPENED))
def md5_file(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
def md5_array(a):
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()
def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0
def partition(n, size, seed):
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(int(n))
    k = max(1, int(round(n / size)))
    return [np.asarray(g, dtype=np.int64) for g in np.array_split(perm, k)]
def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=str)
