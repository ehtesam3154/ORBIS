"""The linear flow path x_t = (1 - t) s n + t x0 and its training-time distribution."""
import torch

from orbis.data.holder import DataHolder, remove_mean_with_mask


class LinearPath:
    """t = 0 is noise, t = 1 is data. Training t: logit-normal(t_mu, t_sigma); with probability t_mix_p instead
    logit-normal(t_mix_mu, t_sigma); clamped to [eps, 1 - eps]. Noise n is standard normal, centred, times noise_scale s."""

    def __init__(self, eps=1e-3, t_mu=0.0, t_sigma=1.0, noise_scale=1.0, t_mix_p=0.0, t_mix_mu=2.32):
        self.eps = float(eps)
        self.t_mu = float(t_mu)
        self.t_sigma = float(t_sigma)
        self.noise_scale = float(noise_scale)
        self.t_mix_p = float(t_mix_p)
        self.t_mix_mu = float(t_mix_mu)

    def sample_t(self, B, device):
        """B training times, shape (B, 1)."""
        u = torch.randn(B, 1, device=device) * self.t_sigma + self.t_mu
        t0 = torch.sigmoid(u).clamp(self.eps, 1 - self.eps)
        if self.t_mix_p <= 0:
            return t0
        alt = torch.sigmoid(torch.randn(B, 1, device=device) * self.t_sigma + self.t_mix_mu).clamp(self.eps, 1 - self.eps)
        return torch.where(torch.rand(B, 1, device=device) < self.t_mix_p, alt, t0)

    def make_zt(self, batched, Y0=None, t=None):
        """Noise a batch (noise Y0 and time t drawn here unless given): returns (z = DataHolder at x_t, t (B, 1), Y0, Y1)."""
        Y1 = batched.positions
        mask = batched.node_mask
        Y0 = (torch.randn_like(Y1) if Y0 is None else Y0.to(Y1)) * mask.unsqueeze(-1)
        if self.noise_scale != 1.0:
            Y0 = Y0 * self.noise_scale
        Y0 = remove_mean_with_mask(Y0, mask)
        t = self.sample_t(Y1.size(0), Y1.device) if t is None else t
        tb = t.view(-1, 1, 1)
        Yt = ((1.0 - tb) * Y0 + tb * Y1) * mask.unsqueeze(-1)
        z = DataHolder(node_features=batched.node_features, positions=Yt, node_mask=mask, cell_class=batched.cell_class,
                       cell_ID=batched.cell_ID, diffusion_time=t, t=t).mask()
        return z, t, Y0, Y1
