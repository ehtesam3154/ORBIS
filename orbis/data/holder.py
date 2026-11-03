"""Per-graph tensor container, dense one-graph batches and masked mean removal."""
import torch


def apply_mask(tensor, mask, mask_dim=-1):
    """tensor * mask broadcast along the last dimension (None passes through)."""
    return tensor * mask.unsqueeze(mask_dim) if tensor is not None else None


def center_positions(positions, mask):
    """Subtract, in place and per graph, the mean position of the masked cells."""
    for i in range(positions.shape[0]):
        masked_positions = positions[i][mask[i]]
        positions[i][mask[i]] = masked_positions - masked_positions.mean(dim=0)
    return positions


class DataHolder:
    """Tensors of a batch of graphs: features (B, N, G), positions (B, N, D), time, labels and the cell mask (B, N)."""

    def __init__(self, positions, node_features, diffusion_time, cell_ID=None, cell_class=None, t_int=None, t=None,
                 node_mask=None):
        self.positions = positions
        self.node_features = node_features
        self.cell_class = cell_class
        self.cell_ID = cell_ID
        self.t_int = t_int
        self.t = t
        self.diffusion_time = diffusion_time if diffusion_time is not None else t
        self.node_mask = node_mask

    def mask(self, node_mask=None):
        """Zero masked-out cells and centre the positions of each graph on its real cells."""
        if node_mask is None:
            assert self.node_mask is not None
            node_mask = self.node_mask
        self.node_features = apply_mask(self.node_features, node_mask)
        self.positions = apply_mask(self.positions, node_mask)
        if self.positions is not None:
            self.positions = center_positions(self.positions, node_mask)
        self.cell_class = apply_mask(self.cell_class, node_mask)
        self.cell_ID = apply_mask(self.cell_ID, node_mask)
        return self


def to_batch(node_features, positions, cell_class, cell_ID):
    """One graph of N cells as a dense batch of size 1 (mask all True), masked and centred."""
    n = node_features.shape[0]
    return DataHolder(node_features=node_features.unsqueeze(0), positions=positions.unsqueeze(0).float(),
                      node_mask=torch.ones(1, n, dtype=torch.bool), cell_class=cell_class.view(1, n, 1),
                      cell_ID=cell_ID.view(1, n, 1), diffusion_time=None).mask()


def to_device(dh, dev):
    """Move every tensor of a DataHolder to dev."""
    for a in ("positions", "node_features", "cell_class", "cell_ID", "node_mask", "t", "t_int", "diffusion_time"):
        v = getattr(dh, a, None)
        if torch.is_tensor(v):
            setattr(dh, a, v.to(dev))
    return dh


def remove_mean_with_mask(x, node_mask):
    """x minus its per-graph mean over the real cells; x must be zero on masked-out cells."""
    m = node_mask.unsqueeze(-1)
    mean = torch.sum(x, dim=1, keepdim=True) / m.sum(1, keepdims=True)
    return x - mean * m
