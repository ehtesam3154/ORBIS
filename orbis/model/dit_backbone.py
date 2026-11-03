"""Diffusion Transformer backbone over cells (one token per cell)."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from orbis.data.holder import DataHolder
from orbis.model.sdpa_attention import SDPAMultiheadAttention


def _sinusoidal_time_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding (B, dim) of scalar times t."""
    if t.dim() > 1:
        t = t.view(-1)
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000.0)
        * torch.arange(0, half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimestepEmbedder(nn.Module):
    """Sinusoidal time embedding followed by a two-layer MLP."""

    def __init__(self, hidden_dim: int, sinusoidal_dim: int = 256):
        super().__init__()
        self.sinusoidal_dim = int(sinusoidal_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.sinusoidal_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(_sinusoidal_time_embed(t, self.sinusoidal_dim))


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN modulation x * (1 + scale) + shift, broadcast over tokens."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning: attention and MLP sublayers, each gated."""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_heads = int(n_heads)
        if self.hidden_dim % self.n_heads != 0:
            raise ValueError(
                f"hidden_dim {hidden_dim} must be divisible by n_heads {n_heads}"
            )

        self.norm1 = nn.LayerNorm(self.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = SDPAMultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.n_heads,
            qk_norm=qk_norm,
        )
        self.norm2 = nn.LayerNorm(self.hidden_dim, elementwise_affine=False, eps=1e-6)
        ff_hidden = int(self.hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, ff_hidden),
            nn.GELU(),
            nn.Linear(ff_hidden, self.hidden_dim),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 6 * self.hidden_dim),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        h = _modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(
            h, h, h,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + gate_msa.unsqueeze(1) * attn_out

        h = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(h)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x


class DiTFinalLayer(nn.Module):
    """Final adaLN modulation and linear projection to output features and positions."""

    def __init__(self, hidden_dim: int, out_features_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        self.linear = nn.Linear(hidden_dim, out_features_dim + 2)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = _modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiTBackbone(nn.Module):
    """DiT over cells: token = gene embedding + position embedding, time through adaLN-Zero."""

    def __init__(
        self,
        input_dims: Dict[str, int],
        n_layers: int,
        hidden_mlp_dims: Dict[str, int],
        hidden_dims: Dict[str, int],
        output_dims: Dict[str, int],
        dit_cfg=None,
    ):
        super().__init__()
        self.out_features_dim = int(hidden_dims["output_features_to_pos_dims"])

        def _g(k, default):
            if dit_cfg is None:
                return default
            return (
                dit_cfg.get(k, default)
                if hasattr(dit_cfg, "get")
                else getattr(dit_cfg, k, default)
            )

        self.hidden_dim = int(_g("hidden_dim", 256))
        self.n_heads = int(_g("n_heads", 8))
        self.mlp_ratio = float(_g("mlp_ratio", 4))
        self.n_layers = int(_g("n_layers", n_layers))
        self.time_embed_dim = int(_g("time_embed_dim", 256))
        self.qk_norm = bool(_g("qk_norm", False))

        gene_in = int(input_dims["node_features_dimensions"])

        self.gene_embed = nn.Sequential(
            nn.Linear(gene_in, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.pos_embed = nn.Sequential(
            nn.Linear(2, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.t_embed = TimestepEmbedder(self.hidden_dim, self.time_embed_dim)

        self.blocks = nn.ModuleList([
            DiTBlock(self.hidden_dim, self.n_heads, self.mlp_ratio, qk_norm=self.qk_norm)
            for _ in range(self.n_layers)
        ])

        self.final = DiTFinalLayer(self.hidden_dim, self.out_features_dim)

    def forward(self, data: DataHolder, **_unused_kwargs) -> DataHolder:
        node_mask = data.node_mask

        tok = self.gene_embed(data.node_features)
        tok = tok + self.pos_embed(data.positions)
        tok = tok * node_mask.unsqueeze(-1).to(tok.dtype)

        t = data.diffusion_time
        if t.dim() > 1:
            t = t.view(-1)
        c = self.t_embed(t)

        key_padding_mask = ~node_mask

        for block in self.blocks:
            tok = block(tok, c, key_padding_mask=key_padding_mask)

        out = self.final(tok, c)
        features = out[..., :-2]
        positions = out[..., -2:]

        pad_mask_features = node_mask.unsqueeze(-1).to(features.dtype)
        features = features * pad_mask_features
        positions = positions * pad_mask_features
        n_real = node_mask.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1).to(
            positions.dtype
        )
        positions = positions - positions.sum(dim=1, keepdim=True) / n_real
        positions = positions * pad_mask_features

        return DataHolder(
            node_features=features,
            positions=positions,
            diffusion_time=data.diffusion_time,
            cell_class=data.cell_class,
            cell_ID=data.cell_ID,
            t_int=data.t_int,
            t=data.t,
            node_mask=node_mask,
        )
