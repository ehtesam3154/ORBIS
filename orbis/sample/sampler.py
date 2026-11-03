"""Samplers of the flow model: the shared-state sampler over all cells and the single-graph Euler sampler."""
import numpy as np
import torch

from orbis.data.holder import DataHolder, remove_mean_with_mask
from orbis.sample.procrustes import _procrustes_align

STEP_EPS = 1e-3


def edm_t_steps(K, noise_scale, data_rms, rho=7.0, sigma_max=80.0, sigma_min=0.002, sigma_data=0.5):
    """EDM step spacing on the linear path t = s / (s + sigma); K times from about 0 to about 1, plus t = 1."""
    i = torch.arange(K, dtype=torch.float64)
    a, b = (sigma_max / sigma_data) ** (1 / rho), (sigma_min / sigma_data) ** (1 / rho)
    r = (a + i / max(K - 1, 1) * (b - a)) ** rho
    t = noise_scale / (noise_scale + r * data_rms)
    return torch.cat([t, torch.ones(1, dtype=torch.float64)]).tolist()


def time_grid(K, schedule, noise_scale, data_rms):
    """The K + 1 times of a K-step sample: uniform in t (also for K = 1) or EDM spacing."""
    if K == 1 or schedule == "uniform":
        return [k / K for k in range(K)] + [1.0]
    return edm_t_steps(K, noise_scale, data_rms or noise_scale)


def denoise(net, X, positions, mask, cell_class, cell_ID, t):
    """One network call: the clean-layout prediction for one dense graph at time t."""
    tt = torch.full((positions.shape[0], 1), float(t), device=X.device)
    z = DataHolder(node_features=X, positions=positions, node_mask=mask, cell_class=cell_class, cell_ID=cell_ID,
                   diffusion_time=tt, t=tt).mask()
    return net(z).positions


@torch.no_grad()
def rf_sample_global(net, batch_of, N, K, eps, dev, noise_scale=1.0, pos_dim=3, schedule="uniform", data_rms=0.0,
                     group=20000, seed=0):
    """One sample of all N cells from a single shared state. At every step the cells are split into a fresh random
    partition of windows of about `group` cells; each window is centred, denoised and written back.
    batch_of(rows) returns (features, mask, cell_class, cell_ID) of those rows as one dense graph on `dev`."""
    g = torch.Generator().manual_seed(int(seed))
    Y = torch.randn(N, pos_dim, generator=g) * noise_scale
    Y = (Y - Y.mean(0, keepdim=True)).to(dev)
    ts = time_grid(K, schedule, noise_scale, data_rms)
    if ts[0] > 0:
        Y = Y * (1.0 - ts[0])
    n_win = int(np.ceil(N / group))

    def denoise_all(Yc, tc):
        Yh = torch.empty_like(Yc)
        wins = np.array_split(torch.randperm(N, generator=g).numpy(), n_win)
        for w in wins:
            w = torch.from_numpy(np.sort(w)).to(dev)
            X, mask, cc, cid = batch_of(w)
            yw = Yc[w]; mu = yw.mean(0, keepdim=True)
            Dx = denoise(net, X, (yw - mu).unsqueeze(0), mask, cc, cid, tc)[0]
            Yh[w] = Dx + mu
        return Yh

    for k in range(K):
        t, t_next = ts[k], ts[k + 1]
        Yh = denoise_all(Y, t)
        if K == 1:
            return Yh
        v = (Yh - Y) / (1.0 - t + eps)
        Yn = Y + (t_next - t) * v
        Yn = Yn - Yn.mean(0, keepdim=True)
        Y = Yn
    return Y


@torch.no_grad()
def rf_sample_xt(net, X, mask, cell_class, cell_ID, K, eps, noise_scale=1.0, align=False, pos_dim=3,
                 schedule="uniform", data_rms=0.0):
    """Euler sampler of one dense graph started from noise of SD noise_scale (torch's global generator). K = 1 returns
    the t = 0 prediction. align=True: for t > 0 each prediction is rotated onto the current state before the step."""
    B, N = mask.shape
    m3 = mask.unsqueeze(-1)
    Y = remove_mean_with_mask(torch.randn(B, N, pos_dim, device=X.device) * noise_scale * m3, mask)
    ts = time_grid(K, schedule, noise_scale, data_rms)
    if ts[0] > 0:
        Y = Y * (1.0 - ts[0])

    def predict(Yc, tc, k):
        Yh = denoise(net, X, Yc * m3, mask, cell_class, cell_ID, tc)
        if K == 1:
            return Yh
        Yh = Yh * m3
        if align and k > 0:
            Yh = Yh.clone()
            for b in range(B):
                m = mask[b]
                src, ref = Yh[b][m] - Yh[b][m].mean(0), Yc[b][m] - Yc[b][m].mean(0)
                Yh[b][m] = _procrustes_align(src, ref) + Yc[b][m].mean(0)
        return Yh

    for k in range(K):
        t, t_next = ts[k], ts[k + 1]
        Yh = predict(Y, t, k)
        if K == 1:
            return Yh
        v = (Yh - Y) / (1.0 - t + eps)
        Y = remove_mean_with_mask((Y + (t_next - t) * v) * m3, mask)
    return Y
