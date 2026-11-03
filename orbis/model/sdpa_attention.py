"""Multi-head attention through torch.nn.functional.scaled_dot_product_attention."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class SDPAMultiheadAttention(nn.Module):
    """Multi-head self-attention computed with scaled_dot_product_attention, with optional QK-norm."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim {embed_dim} must be divisible by num_heads "
                f"{num_heads} (per-head dim must be an integer)."
            )
        self.head_dim = self.embed_dim // self.num_heads
        self.kdim = int(kdim) if kdim is not None else self.embed_dim
        self.vdim = int(vdim) if vdim is not None else self.embed_dim
        self.dropout = float(dropout)

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, self.embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, self.embed_dim, bias=bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
            self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Attention output (B, N_q, embed_dim) and None."""
        if need_weights:
            raise RuntimeError(
                "SDPAMultiheadAttention does not return attention "
                "weights — exposing them forces materialising the "
                "N×N matrix and defeats the FlashAttention memory-"
                "efficient path. Use a dedicated weights-extracting "
                "module if you need that signal."
            )

        B, N_q, _ = query.shape
        N_kv = key.shape[1]
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(query).view(B, N_q, H, D).transpose(1, 2)
        k = self.k_proj(key  ).view(B, N_kv, H, D).transpose(1, 2)
        v = self.v_proj(value).view(B, N_kv, H, D).transpose(1, 2)
        if hasattr(self, "q_norm"):
            q, k = self.q_norm(q), self.k_norm(k)

        sdpa_mask: Optional[torch.Tensor] = None
        if key_padding_mask is not None or attn_mask is not None:
            mask_dtype = q.dtype
            bias_shape = (B, 1, 1, N_kv)
            sdpa_mask = torch.zeros(
                bias_shape, dtype=mask_dtype, device=q.device,
            )
            if key_padding_mask is not None:
                if key_padding_mask.dtype != torch.bool:
                    raise TypeError(
                        f"key_padding_mask must be bool (True at PAD); "
                        f"got dtype {key_padding_mask.dtype}."
                    )
                neg_inf = torch.finfo(mask_dtype).min
                pad_bias = torch.where(
                    key_padding_mask,
                    torch.tensor(neg_inf, dtype=mask_dtype, device=q.device),
                    torch.tensor(0.0,     dtype=mask_dtype, device=q.device),
                )
                sdpa_mask = sdpa_mask + pad_bias.view(B, 1, 1, N_kv)

            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    neg_inf = torch.finfo(mask_dtype).min
                    attn_bias = torch.where(
                        attn_mask,
                        torch.tensor(0.0,     dtype=mask_dtype, device=q.device),
                        torch.tensor(neg_inf, dtype=mask_dtype, device=q.device),
                    )
                else:
                    attn_bias = attn_mask.to(mask_dtype)
                if attn_bias.dim() == 2:
                    attn_bias = attn_bias.view(1, 1, N_q, N_kv)
                elif attn_bias.dim() == 3:
                    attn_bias = attn_bias.view(B, H, N_q, N_kv)
                else:
                    raise ValueError(
                        f"attn_mask must have rank 2 or 3; got rank "
                        f"{attn_bias.dim()}."
                    )
                sdpa_mask = sdpa_mask + attn_bias

        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=sdpa_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(B, N_q, self.embed_dim)
        out = self.out_proj(out)
        return out, None
