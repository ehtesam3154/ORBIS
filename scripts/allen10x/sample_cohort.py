"""Expression-only sampling of one 10x cohort with the shared-state sampler of orbis.sample (no positions, labels or sections are read).

The network sees each cell's feature row (the 200 mapped reference PCs), the noisy positions it generates itself, the time and an
all-True mask; cell_class is a constant 0 and cell_ID the opaque input id (the network reads neither).
Output npz: P (N x 3 float32, rows in input order, the model's units), cell_ID, iso; sidecar <out>.json with the run settings.
usage (from the ORBIS root): python -m scripts.allen10x.sample_cohort --ckpt CK --features F.npy --ids I.npy
       --feature_names_json NAMES.json --train_meta STORE/meta.json --sample_seed 1000 --out OUT.npz [--steps 50] [--schedule edm] [--group 20000]"""
import argparse
import hashlib
import json
import time

import numpy as np
import torch

from orbis.model.load import load_checkpoint
from orbis.sample.sampler import STEP_EPS, rf_sample_global


def md5_file(path):
    """md5 of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    """Sample one layout of the given cells."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--features", required=True, help="N x G float32 .npy (columns named by --feature_names_json)")
    ap.add_argument("--ids", required=True, help="N opaque integer cell ids (.npy)")
    ap.add_argument("--feature_names_json", required=True, help="JSON list of the G column names of --features")
    ap.add_argument("--train_meta", required=True, help="meta.json of a training store: its gene_names must equal the column names")
    ap.add_argument("--sample_seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--schedule", default="edm")
    ap.add_argument("--group", type=int, default=20000)
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args()
    t_start = time.time()
    dev = torch.device(f"cuda:{a.gpu}")
    F = np.load(a.features)
    ids = np.load(a.ids)
    assert F.ndim == 2 and F.dtype == np.float32 and np.isfinite(F).all(), "features must be finite N x G float32"
    N, G = F.shape
    assert ids.shape == (N,) and np.issubdtype(ids.dtype, np.integer) and len(np.unique(ids)) == N, "ids: N unique integers"
    names = json.load(open(a.feature_names_json))
    assert names == json.load(open(a.train_meta))["gene_names"], "column order differs from the training stores'"
    net, args = load_checkpoint(a.ckpt, dev)
    assert int(args.get("gene_columns_end", 0)) - int(args.get("gene_columns_start", 0)) == G, "feature count differs from the checkpoint's"
    assert bool(args.get("xt_precond", 0)) or bool(args.get("loss_mse", 0)), "shared state: needs xt_precond 1 or loss_mse 1"
    xs = float(args.get("noise_scale", 1.0))
    data_rms = float(args["data_rms"])
    Ft = torch.from_numpy(np.ascontiguousarray(F)).to(dev)
    C = torch.zeros(N, dtype=torch.long, device=dev)
    I = torch.from_numpy(ids.astype(np.int64)).to(dev)

    def batch_of(w):
        n = len(w)
        if n == N:
            return Ft.unsqueeze(0), torch.ones(1, n, dtype=torch.bool, device=dev), C.view(1, n, 1), I.view(1, n, 1)
        return Ft[w].unsqueeze(0), torch.ones(1, n, dtype=torch.bool, device=dev), C[w].view(1, n, 1), I[w].view(1, n, 1)

    t0 = time.time()
    Y = rf_sample_global(net, batch_of, N, a.steps, STEP_EPS, dev, noise_scale=xs, pos_dim=3, schedule=a.schedule,
                         data_rms=data_rms, group=a.group, seed=a.sample_seed)
    torch.cuda.synchronize()
    P = Y.cpu().numpy().astype(float).astype(np.float32)
    assert P.shape == (N, 3) and np.isfinite(P).all()
    np.savez_compressed(a.out, P=P, cell_ID=ids.astype(np.int64), iso=True)
    side = dict(ckpt=a.ckpt, ckpt_md5=md5_file(a.ckpt), features_md5=md5_file(a.features), N=int(N), G=int(G), sample_seed=a.sample_seed,
                steps=a.steps, schedule=a.schedule, group=a.group, noise_scale=xs, data_rms=data_rms,
                P_md5=hashlib.md5(np.ascontiguousarray(P).tobytes()).hexdigest(), sample_seconds=round(time.time() - t0, 2),
                wall_seconds=round(time.time() - t_start, 2), device=torch.cuda.get_device_name(dev), torch=torch.__version__)
    json.dump(side, open(a.out + ".json", "w"), indent=1)
    print(f"sampled {N:,} cells, seed {a.sample_seed}, {a.steps} steps ({a.schedule}), windows of <= {a.group}: {side['sample_seconds']} s -> {a.out}")


if __name__ == "__main__":
    main()
