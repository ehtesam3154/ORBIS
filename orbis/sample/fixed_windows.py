"""Fixed-window sampler: as the shared-state sampler, but the partition into windows is drawn once per sample."""
import numpy as np
import torch

from orbis.sample.sampler import denoise, time_grid


@torch.no_grad()
def rf_sample_fixedwin(net, batch_of, N, K, eps, dev, noise_scale=1.0, pos_dim=3, schedule="uniform", data_rms=0.0,
                       group=20000, seed=0, independent=False):
    """The windows (about `group` cells each) are drawn once, right after the initial noise, and kept for every step.
    independent=False: one global state, re-centred as a whole after every step.
    independent=True: every window is re-centred on its own after every step (each window is its own sample).
    Returns (Y (N, pos_dim) on `dev`, window index of every cell (N,) int16)."""
    assert K > 1, "fixed windows: K > 1"
    g = torch.Generator().manual_seed(int(seed))
    Y = torch.randn(N, pos_dim, generator=g) * noise_scale
    Y = (Y - Y.mean(0, keepdim=True)).to(dev)
    ts = time_grid(K, schedule, noise_scale, data_rms)
    if ts[0] > 0:
        Y = Y * (1.0 - ts[0])
    n_win = int(np.ceil(N / group))
    wins_np = [np.sort(w) for w in np.array_split(torch.randperm(N, generator=g).numpy(), n_win)]
    wins = [torch.from_numpy(w).to(dev) for w in wins_np]
    win_id = np.empty(N, dtype=np.int16)
    for i, w in enumerate(wins_np):
        win_id[w] = i
    assert (np.bincount(win_id, minlength=n_win) == np.array([len(w) for w in wins_np])).all() and len(np.unique(np.concatenate(wins_np))) == N
    sizes = np.array([len(w) for w in wins_np])
    print(f"[fixed windows] N {N:,}: {n_win} windows of {sizes.min():,}-{sizes.max():,} cells, "
          f"{'independent' if independent else 'shared state'}, K {K}, schedule {schedule}, seed {seed}", flush=True)

    def denoise_all(Yc, tc):
        Yh = torch.empty_like(Yc)
        for w in wins:
            X, mask, cc, cid = batch_of(w)
            yw = Yc[w]; mu = yw.mean(0, keepdim=True)
            Dx = denoise(net, X, (yw - mu).unsqueeze(0), mask, cc, cid, tc)[0]
            Yh[w] = Dx + mu
        return Yh

    def recentre(Yn):
        if not independent:
            return Yn - Yn.mean(0, keepdim=True)
        for w in wins:
            Yn[w] = Yn[w] - Yn[w].mean(0, keepdim=True)
        return Yn

    for k in range(K):
        t, t_next = ts[k], ts[k + 1]
        Yh = denoise_all(Y, t)
        v = (Yh - Y) / (1.0 - t + eps)
        Yn = recentre(Y + (t_next - t) * v)
        Y = Yn
    return Y, win_id
