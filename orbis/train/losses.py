"""Training losses on the predicted clean positions."""
import torch
import torch.nn as nn


class CoordMSELoss(nn.Module):
    """Mean squared coordinate error over the real cells, in the training frame."""

    def forward(self, pred, true):
        m = true.node_mask.unsqueeze(-1).to(pred.positions.dtype)
        return (((pred.positions - true.positions) * m) ** 2).sum() / (m.sum() * pred.positions.shape[-1])


class PlaneBalancedDistLoss(nn.Module):
    """Pairwise-distance squared error with within-plane and cross-plane pairs averaged separately and mixed
    w_within : 1 - w_within (plane = equal third coordinate of the true positions, within ztol); 2-D or one-plane
    graphs use the plain mean over all pairs."""

    def __init__(self, w_within: float = 0.5, ztol: float = 1e-4, eps: float = 1e-8):
        super().__init__()
        self.w = float(w_within)
        self.ztol = float(ztol)
        self.eps = float(eps)

    def _one(self, true_pos, pred_pos, mask):
        T = true_pos[mask]
        P = pred_pos[mask]
        n = T.shape[0]
        Dt = torch.cdist(T, T, p=2)
        Dp = torch.cdist(P, P, p=2)
        se = (Dp - Dt) ** 2
        if T.shape[1] < 3 or n < 2:
            return se.sum() / (n * n)
        z = T[:, 2]
        same = (z.unsqueeze(1) - z.unsqueeze(0)).abs() < self.ztol
        n_same = same.sum()
        n_within = n_same - n
        n_cross = n * n - n_same
        tot = se.sum()
        s_within = (se * same).sum()
        s_cross = tot - s_within
        if n_cross <= 0:
            return tot / (n * n)
        if n_within <= 0:
            return s_cross / n_cross
        return self.w * (s_within / (n_within + self.eps)) \
            + (1.0 - self.w) * (s_cross / (n_cross + self.eps))

    def forward(self, pred, true):
        losses = [self._one(t, p, m) for t, p, m in zip(true.positions, pred.positions, true.node_mask)]
        return torch.stack(losses).mean()
