"""Build the network a checkpoint was trained with and load its weights."""
import torch

from orbis.data.views import N_VIEW_TYPES
from orbis.model.dit import DiT3D, EDMScaling


def build_model(args, device="cpu"):
    """The network described by a checkpoint's args dict (absent keys take the training defaults)."""
    gene_dim = int(args.get("gene_columns_end", 1122)) - int(args.get("gene_columns_start", 0))
    net = DiT3D(gene_dim, pos_dim=int(args.get("pos_dim", 3)), hidden_dim=int(args.get("dit_hidden", 256)),
                n_layers=int(args.get("dit_layers", 8)), n_heads=int(args.get("dit_heads", 8)),
                n_view_types=int(args.get("n_view_types", N_VIEW_TYPES)) if args.get("view_label", 0) else 0,
                qk_norm=bool(int(args.get("qk_norm", 0)))).to(device)
    if args.get("xt_precond", 0):
        net = EDMScaling(net, float(args.get("noise_scale", 1.0)), float(args.get("data_rms", 0.0)) or 0.163).to(device)
    return net


def dit_of(net):
    """The DiT3D inside a network (to set its view_label)."""
    return next(m for m in net.modules() if isinstance(m, DiT3D))


def load_checkpoint(path, device="cpu", weights="state_dict"):
    """(network in eval mode, args dict) of a checkpoint; weights = 'state_dict' (EMA) or 'raw_state_dict'."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    net = build_model(ck["args"], device)
    net.load_state_dict(ck[weights])
    net.eval()
    return net, ck["args"]
