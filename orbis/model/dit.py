"""The denoiser: a DiT over cells for D-dimensional positions with a view-label input, and the EDM output scaling."""
import torch
import torch.nn as nn

from orbis.data.holder import DataHolder
from orbis.model.dit_backbone import DiTBackbone


class DiT3D(DiTBackbone):
    """DiT reading expression and the noisy positions x_t of every cell, predicting clean positions.

    pos_dim-dimensional position input and output; blocks run under bf16 autocast on CUDA (final layer in fp32);
    n_view_types > 0 adds a zero-initialised view-label embedding to the time embedding (label from self.view_label,
    WHOLE = 0 when None); qk_norm adds LayerNorm on queries and keys in every attention."""

    def __init__(self, gene_dim, pos_dim=3, hidden_dim=256, n_layers=8, n_heads=8, mlp_ratio=4.0, out_features_dim=32,
                 time_embed_dim=256, n_view_types=0, qk_norm=False):
        super().__init__(input_dims={"node_features_dimensions": int(gene_dim)}, n_layers=int(n_layers),
                         hidden_mlp_dims={}, hidden_dims={"output_features_to_pos_dims": int(out_features_dim)},
                         output_dims={}, dit_cfg={"hidden_dim": int(hidden_dim), "n_heads": int(n_heads),
                                                  "mlp_ratio": float(mlp_ratio), "n_layers": int(n_layers),
                                                  "time_embed_dim": int(time_embed_dim), "qk_norm": bool(qk_norm)})
        self.pos_dim = int(pos_dim)
        self.pos_embed[0] = nn.Linear(self.pos_dim, self.hidden_dim)
        self.final.linear = nn.Linear(self.hidden_dim, self.out_features_dim + self.pos_dim)
        self.bf16 = True
        self.n_view_types = int(n_view_types)
        self.view_embed = nn.Linear(self.n_view_types, self.hidden_dim, bias=False) if self.n_view_types else None
        if self.view_embed is not None:
            nn.init.zeros_(self.view_embed.weight)
        self.view_label = None

    def forward(self, data, **_unused_kwargs):
        """DataHolder in (node_features, positions = x_t, diffusion_time, node_mask); DataHolder out (centred positions)."""
        node_mask = data.node_mask
        tok = self.gene_embed(data.node_features)
        tok = tok + self.pos_embed(data.positions)
        tok = tok * node_mask.unsqueeze(-1).to(tok.dtype)
        t = data.diffusion_time
        if t.dim() > 1:
            t = t.view(-1)
        c = self.t_embed(t)
        if self.view_embed is not None:
            lab = self.view_label if self.view_label is not None else torch.zeros(c.shape[0], dtype=torch.long, device=c.device)
            c = c + self.view_embed(torch.nn.functional.one_hot(lab.view(-1).to(c.device), self.n_view_types).to(c.dtype))
        key_padding_mask = None if bool(node_mask.all()) else ~node_mask
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16 and tok.is_cuda):
            for block in self.blocks:
                tok = block(tok, c, key_padding_mask=key_padding_mask)
        out = self.final(tok.float(), c)
        features = out[..., :-self.pos_dim]
        positions = out[..., -self.pos_dim:]
        pad_mask_features = node_mask.unsqueeze(-1).to(features.dtype)
        features = features * pad_mask_features
        positions = positions * pad_mask_features
        n_real = node_mask.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1).to(positions.dtype)
        positions = positions - positions.sum(dim=1, keepdim=True) / n_real
        positions = positions * pad_mask_features
        return DataHolder(node_features=features, positions=positions, diffusion_time=data.diffusion_time,
                          cell_class=data.cell_class, cell_ID=data.cell_ID, t_int=data.t_int, t=data.t,
                          node_mask=node_mask)


class EDMScaling(nn.Module):
    """EDM output scaling around a denoiser that reads x_t (Karras et al. 2022), for x_t = t x0 + (1 - t) s n.

    With data RMS d and v = sqrt(t^2 d^2 + (1 - t)^2 s^2): the network reads x_t / v and the output is
    t d^2 / v^2 x_t + d (1 - t) s / v F."""

    def __init__(self, net, noise_scale=1.0, data_rms=0.163):
        super().__init__()
        self.net = net
        self.s, self.d = float(noise_scale), float(data_rms)
        self.pos_dim = getattr(net, "pos_dim", 3)

    def forward(self, z):
        """DataHolder in (positions = x_t, diffusion_time (B, 1)); DataHolder out with the scaled x0 estimate."""
        t = z.diffusion_time.reshape(z.diffusion_time.shape[0], -1)[:, :1].to(z.positions.dtype).view(-1, 1, 1)
        v = torch.sqrt((t * self.d) ** 2 + ((1.0 - t) * self.s) ** 2)
        xt = z.positions
        zin = DataHolder(node_features=z.node_features, positions=xt / v, diffusion_time=z.diffusion_time,
                         cell_class=z.cell_class, cell_ID=z.cell_ID, t_int=z.t_int, t=z.t, node_mask=z.node_mask)
        out = self.net(zin)
        m = z.node_mask.unsqueeze(-1).to(xt.dtype)
        F = out.positions
        out.positions = (t * self.d ** 2 / v ** 2 * xt + self.d * (1.0 - t) * self.s / v * F) * m
        return out
