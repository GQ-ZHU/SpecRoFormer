"""
RoPE encoder layers.

Implement Rotary Position Embedding (RoPE) applied to m/z values and
the corresponding Transformer encoder layer used by the SpecRoFormer
models.
"""

import math
from math import sqrt

import torch
import torch.nn as nn


class RotaryEmbeddingBank(nn.Module):
    """
    Compute the cos / sin base frequencies used by RoPE.

    A different base frequency is used for each pair of channels in
    every attention head. The input is the raw m/z sequence.

    Parameters
    ----------
    dim : int
        Total embedding dimension.
    n_head : int
        Number of attention heads.
    base : float
        Base of the geometric progression of frequencies.

    Input
    -----
    mz : torch.Tensor
        Shape ``[B, L]``, raw m/z values.

    Output
    ------
    freqs : torch.Tensor
        Shape ``[B, L, n_head, d_head // 2]``, the rotation angles.
    """

    def __init__(self, dim, n_head, base=10000.0):
        super().__init__()
        self.dim = dim
        self.n_head = n_head
        self.d_head = dim // n_head

        assert self.d_head % 2 == 0, "d_head must be even for RoPE"

        # One distinct frequency per pair of channels (log-spaced).
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.d_head, 2).float()
                / self.d_head
            )
        )
        self.register_buffer("inv_freq", inv_freq)

        # Optional learnable scale for the mz -> angle mapping.
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, mz):
        B, L = mz.shape

        # Convert m/z to a stable scalar range via log1p and a
        # learnable scale.
        v = torch.log1p(mz) * self.scale  # [B, L]

        # Outer product with the inverse frequencies.
        freqs = torch.einsum("bl,d->bld", v, self.inv_freq)  # [B, L, d2]

        # Expand to heads: [B, L, n_head, d2]
        freqs = freqs.unsqueeze(2).repeat(1, 1, self.n_head, 1)

        return freqs


def apply_rotary_to_qk(q, k, freqs):
    """
    Apply the RoPE rotation to the query and key tensors.

    Parameters
    ----------
    q, k : torch.Tensor
        Shape ``[B, L, n_head, d_head]``.
    freqs : torch.Tensor
        Shape ``[B, L, n_head, d_head // 2]``.

    Returns
    -------
    q_rot, k_rot : torch.Tensor
        Rotated query and key tensors.
    """
    B, L, H, d_head = q.shape
    d2 = d_head // 2

    # Split into two halves for pair-wise rotation.
    q1 = q[..., :d2]
    q2 = q[..., d2:]
    k1 = k[..., :d2]
    k2 = k[..., d2:]

    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    q_rot = torch.cat(
        [q1 * cos - q2 * sin, q1 * sin + q2 * cos],
        dim=-1,
    )
    k_rot = torch.cat(
        [k1 * cos - k2 * sin, k1 * sin + k2 * cos],
        dim=-1,
    )
    return q_rot, k_rot


class RoPEMultiheadAttentionLayer(nn.Module):
    """Multi-head attention with RoPE applied to Q and K."""

    def __init__(self, dim, n_head, dropout=0.1):
        super().__init__()
        assert dim % n_head == 0

        self.dim = dim
        self.n_head = n_head
        self.d_head = dim // n_head

        # QKV projections (manual attention computation).
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Rotary embedding bank.
        self.rotary = RotaryEmbeddingBank(dim, n_head)

    def forward(
        self,
        x,
        mz,
        attn_mask=None,
        key_padding_mask=None,
        return_attention: bool = False,
    ):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``[B, L, D]`` (batch first).
        mz : torch.Tensor
            Shape ``[B, L]``, used as rotary positions.
        attn_mask : optional
            Not used here.
        key_padding_mask : torch.Tensor, optional
            Shape ``[B, L]``, True for padding.
        return_attention : bool
            If True, also return attention probabilities.
        """
        B, L, D = x.shape

        # Project to Q, K, V.
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to heads.
        q = q.reshape(B, L, self.n_head, self.d_head)
        k = k.reshape(B, L, self.n_head, self.d_head)
        v = v.reshape(B, L, self.n_head, self.d_head)

        # RoPE frequencies.
        freqs = self.rotary(mz)  # [B, L, n_head, d2]

        # Apply RoPE to Q and K.
        q, k = apply_rotary_to_qk(q, k, freqs)

        # Scaled dot-product attention.
        q = q.transpose(1, 2)  # [B, n_head, L, d_head]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_scores = torch.matmul(
            q, k.transpose(-2, -1)
        ) / sqrt(self.d_head)  # [B, H, L, L]

        # Key padding mask.
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,L]
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        out = torch.matmul(attn_probs, v)             # [B, H, L, d_head]
        out = out.transpose(1, 2).reshape(B, L, D)    # [B, L, D]
        out = self.out_proj(out)

        if return_attention:
            return out, attn_probs
        else:
            return out


class RoPEEncoderLayer(nn.Module):
    """Post-LN Transformer encoder layer with RoPE attention."""

    def __init__(self, dim, n_head, dim_ff, dropout=0.1, activation="gelu"):
        super().__init__()
        self.attn = RoPEMultiheadAttentionLayer(
            dim, n_head, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x,
        mz,
        key_padding_mask=None,
        return_attention: bool = False,
    ):
        if return_attention:
            attn_out, attn_probs = self.attn(
                x,
                mz,
                key_padding_mask=key_padding_mask,
                return_attention=True,
            )
        else:
            attn_out = self.attn(
                x,
                mz,
                key_padding_mask=key_padding_mask,
                return_attention=False,
            )
            attn_probs = None

        x = x + attn_out
        x = self.norm1(x)

        ff_out = self.ff(x)
        x = x + ff_out
        x = self.norm2(x)

        if return_attention:
            return x, attn_probs
        else:
            return x