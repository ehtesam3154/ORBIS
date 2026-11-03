"""Sampling CLI: writes one layout of every cell of a test prebuilt per step count K, as <out_dir>/<tag>_steps<K>.npz
(keys P = sampled positions, G = positions stored in the prebuilt, cc = cell classes, iso; fixed windows add win)
and a run record <out_dir>/<tag>.json.
usage: python -m orbis.sample.run --ckpt CKPT --test_dir DIR --out_dir OUT --tag TAG --steps 50 --schedule edm
       --calibrate 0 --global_state 1 --sample_seed 1000 [--group 20000] [--global_k1 1] [--fixed_windows 1 --independent_windows 0|1]"""
import argparse
import json
import os
import time

import numpy as np
import torch

from orbis.data.prebuilt import PrebuiltGraphs
from orbis.model.load import load_checkpoint
from orbis.sample.chunks import sample_chunks
from orbis.sample.fixed_windows import rf_sample_fixedwin
from orbis.sample.sampler import STEP_EPS, rf_sample_global


def parse_args(argv=None):
    """Command-line arguments."""
    ap = argparse.ArgumentParser(description="Sample layouts of a test prebuilt with a trained checkpoint.")
    ap.add_argument("--ckpt", required=True, help="checkpoint file")
    ap.add_argument("--test_dir", required=True, help="test prebuilt directory")
    ap.add_argument("--out_dir", required=True, help="output directory for <tag>_steps<K>.npz and <tag>.json")
    ap.add_argument("--tag", required=True, help="output name")
    ap.add_argument("--steps", type=int, nargs="+", default=[1], help="number of steps K (one layout per value)")
    ap.add_argument("--gpu", default="0", help="CUDA device index")
    ap.add_argument("--mm", type=float, default=0.0, help="millimetres per unit of the test positions (recorded only)")
    ap.add_argument("--schedule", default="uniform", choices=["uniform", "edm"], help="step spacing: uniform in t or EDM spacing")
    ap.add_argument("--calibrate", type=int, default=0, choices=[0], help="kept for compatibility; must be 0")
    ap.add_argument("--global_state", type=int, default=0,
                    help="1 = one shared state over all cells, random windows redrawn at every step; 0 = chunks stitched on anchor cells")
    ap.add_argument("--global_k1", type=int, default=0, help="with --global_state 1: K = 1 also uses the shared-state sampler")
    ap.add_argument("--sample_seed", type=int, default=1000, help="seed of the shared-state and fixed-window samplers (the chunk sampler seeds chunk k with 1000 + k)")
    ap.add_argument("--group", type=int, default=20000, help="cells per window")
    ap.add_argument("--fixed_windows", type=int, default=0, help="with --global_state 1: 1 = windows drawn once per sample (K > 1)")
    ap.add_argument("--independent_windows", type=int, default=0,
                    help="with --fixed_windows 1: 1 = every window re-centred on its own (independent samples); 0 = shared state")
    return ap.parse_args(argv)


def main(argv=None):
    """Sample every requested step count and write the layouts."""
    a = parse_args(argv)
    dev = torch.device(f"cuda:{a.gpu}")
    net, args = load_checkpoint(a.ckpt, dev)
    xs = float(args.get("noise_scale", 1.0))
    data_rms = float(args["data_rms"])
    xpre = bool(args.get("xt_precond", 0))
    abs_frame = bool(args.get("loss_mse", 0)) or float(args.get("frame_anchor", 0.0)) > 0
    ds = PrebuiltGraphs(a.test_dir)
    N = int(ds.offsets[-1])
    os.makedirs(a.out_dir, exist_ok=True)
    res = {"ckpt": a.ckpt, "test_dir": a.test_dir, "mm": a.mm, "schedule": a.schedule, "global_state": a.global_state,
           "global_k1": a.global_k1, "group": a.group, "sample_seed": a.sample_seed, "fixed_windows": a.fixed_windows,
           "independent_windows": a.independent_windows}
    F = torch.from_numpy(np.asarray(ds.feat[:], dtype=np.float32)).to(dev)
    C = torch.from_numpy(np.asarray(ds.cc[:])).to(dev); I = torch.from_numpy(np.asarray(ds.cid[:])).to(dev)

    def batch_of(w):
        n = len(w)
        if n == len(F):
            return F.unsqueeze(0), torch.ones(1, n, dtype=torch.bool, device=dev), C.view(1, n, 1), I.view(1, n, 1)
        return F[w].unsqueeze(0), torch.ones(1, n, dtype=torch.bool, device=dev), C[w].view(1, n, 1), I[w].view(1, n, 1)

    for K in a.steps:
        t0 = time.time(); extra = {}
        if a.global_state and (K > 1 or a.global_k1):
            assert xpre or abs_frame, "shared state: needs a checkpoint with xt_precond 1 or loss_mse 1"
            if a.fixed_windows:
                Y, extra["win"] = rf_sample_fixedwin(net, batch_of, N, K, STEP_EPS, dev, noise_scale=xs, pos_dim=3, schedule=a.schedule,
                                                     data_rms=data_rms, group=a.group, seed=a.sample_seed,
                                                     independent=bool(a.independent_windows))
            else:
                Y = rf_sample_global(net, batch_of, N, K, STEP_EPS, dev, noise_scale=xs, pos_dim=3, schedule=a.schedule,
                                     data_rms=data_rms, group=a.group, seed=a.sample_seed)
            P_all = Y.cpu().numpy().astype(float)
        else:
            P_all = sample_chunks(net, ds, N, K, dev, xs, data_rms, a.schedule, align=not xpre)
        G_all = np.asarray(ds.pos[:], dtype=float)
        res[f"steps{K}"] = dict(cells=N, seconds=round(time.time() - t0, 1))
        print(f"{a.tag} steps={K}: {json.dumps(res[f'steps{K}'])}", flush=True)
        np.savez_compressed(os.path.join(a.out_dir, f"{a.tag}_steps{K}.npz"), P=P_all.astype(np.float32), G=G_all.astype(np.float32),
                            cc=np.asarray(ds.cc[:]), iso=True, **extra)
    json.dump(res, open(os.path.join(a.out_dir, f"{a.tag}.json"), "w"), indent=1)
    print("SAMPLE_DONE")


if __name__ == "__main__":
    main()
