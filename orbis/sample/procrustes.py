"""Orthogonal Procrustes alignment (rotation and reflection) of one centred point set onto another."""
import torch


def _procrustes_align(x_src: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
    """Return x_src @ R with R the orthogonal matrix minimising ||x_src @ R - x_ref||_F."""
    with torch.no_grad():
        M_detached = (x_src.detach().T @ x_ref.detach())
        M_max = M_detached.abs().max()
        if (not torch.isfinite(M_max).item()) or M_max.item() < 1e-6:
            return x_src
        U, _S, Vh = torch.linalg.svd(M_detached)
        R = (U @ Vh).detach()
    return x_src @ R
