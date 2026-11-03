"""Left-right unfold of an embryo layout sampled by a model trained on views folded across the left-right midplane
(axis 2). The model outputs |LR| - c, with c the training cells' mean |LR|; every cell gets |LR| = max(LR_out + c, 0)
and a random side drawn with SEED. Writes <name>_unfold<SEED>k<number of stores>_steps<K>.npz next to the input.
usage: python -m orbis.sample.unfold LAYOUT_NPZ SEED STORE_DIR[,STORE_DIR,...]   (the model's own training stores)"""
import os
import sys

import numpy as np


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    path = argv[0]
    seed = int(argv[1])
    sdirs = argv[2].split(",")
    c = float(np.mean(np.concatenate([np.abs(np.load(d + "/pos.npy")[:, 2]) for d in sdirs])))
    z = dict(np.load(path))
    P = z["P"].astype(np.float64)
    a = np.maximum(P[:, 2] + c, 0.0)
    side = np.where(np.random.default_rng(seed).random(len(P)) < 0.5, -1.0, 1.0)
    P[:, 2] = side * a
    z["P"] = P.astype(np.float32)
    tag = os.path.basename(path)[:-len(".npz")]
    suf = f"{seed}k{len(sdirs)}"
    out = tag.replace("_steps", f"_unfold{suf}_steps") if "_steps" in tag else tag + f"_unfold{suf}"
    np.savez_compressed(os.path.join(os.path.dirname(path), f"{out}.npz"), **z)
    print(f"{tag}: c = {c:.5f} (training mean |LR| over {len(sdirs)} stores), side seed {seed}; |LR| < 0 clipped for "
          f"{np.mean(P[:, 2] == 0):.2%}; wrote {out}", flush=True)


if __name__ == "__main__":
    main()
