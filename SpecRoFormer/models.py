"""
Model definitions for the SpecRoFormer family.

Only the models used in the final method and its ablations are kept:

- ``SpecRoFormer_RoPE``             : RoPE Transformer baseline.
- ``SpecRoFormer_LinearMZ``    : RoPE + linear m/z embedding.
- ``SpecRoFormer``  : final method, adds precursor
  m/z conditioning on top of the RoPE Transformer.
"""

import math
from typing import Literal, Union, Iterable, Tuple

import torch
import torch.nn as nn

from SpecRoFormer.utils.RoPEEncoderLayer import RoPEEncoderLayer


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

LAMBDA_MIN = math.pow(10, -3.0)
LAMBDA_MAX = math.pow(10, 3.0)


# ---------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------

class PrecursorMzEmbedding(nn.Module):
    """
    MLP embedding of the precursor m/z.

    The raw m/z is scaled to roughly [0, 1] before the MLP, so the
    input distribution is stable across datasets.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = None,
        mz_max: float = 2000.0,
        activation: str = "gelu",
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = embedding_dim

        self.mz_max = mz_max

        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, precursor_mz: torch.Tensor):
        """
        Parameters
        ----------
        precursor_mz : torch.Tensor
            Shape ``[B]``.

        Returns
        -------
        torch.Tensor
            Shape ``[B, D]``.
        """
        x = precursor_mz.unsqueeze(-1) / self.mz_max
        return self.net(x)


class MultiFeedForwardModule(nn.Module):
    """Configurable MLP with optional final dropout."""

    def __init__(
        self,
        input_size: int,
        hidden_size: Union[int, Iterable[int]],
        output_size: int,
        *,
        activation: Literal["relu", "selu", "gelu"] = "relu",
        dropout: float = 0.1,
        dropout_last_layer: bool = True,
    ):
        super(MultiFeedForwardModule, self).__init__()

        if activation == "relu":
            self._activation = nn.ReLU()
        elif activation == "selu":
            self._activation = nn.SELU()
        elif activation == "gelu":
            self._activation = nn.GELU()
        else:
            raise ValueError("activation must be relu or selu")

        if not hasattr(hidden_size, "__iter__"):
            if hidden_size is None:
                hidden_size = [output_size]
            else:
                hidden_size = [hidden_size]

        self._layers = []
        layer_dims = [input_size] + hidden_size + [output_size]

        for i in range(1, len(layer_dims) - 1):
            self._layers.append(
                nn.Linear(layer_dims[i - 1], layer_dims[i])
            )
            self._layers.append(self._activation)
            self._layers.append(nn.Dropout(dropout))

        self._layers.append(
            nn.Linear(layer_dims[-2], layer_dims[-1])
        )

        if dropout_last_layer:
            self._layers.append(nn.Dropout(dropout))

        self._layers = nn.Sequential(*self._layers)

    def forward(self, x):
        return self._layers(x)


class SinusodialMz(nn.Module):
    """Fixed sinusoidal embedding of m/z values."""

    def __init__(
        self,
        embedding_dim: int,
        *,
        lambda_params: Tuple[float, float] = (LAMBDA_MIN, LAMBDA_MAX),
    ) -> None:
        super(SinusodialMz, self).__init__()

        self.lambda_min, self.lambda_max = lambda_params
        self.lambda_div_value = self.lambda_max / self.lambda_min

        self.x = torch.arange(0, embedding_dim, 2)
        self.x = (
            2
            * math.pi
            * (
                self.lambda_min
                * self.lambda_div_value
                ** (self.x / (embedding_dim - 2))
            )
            ** -1
        )

    def forward(self, mz: torch.Tensor):
        self.x = self.x.to(mz.device)

        x = torch.einsum("bl,d->bld", mz, self.x)
        sin_embedding = torch.sin(x)
        cos_embedding = torch.cos(x)

        b, l, d = sin_embedding.shape
        x = torch.zeros(
            b, l, 2 * d, dtype=mz.dtype, device=mz.device
        )
        x[:, :, ::2] = sin_embedding
        x[:, :, 1::2] = cos_embedding
        return x


class SinusodialMzEmbedding(nn.Module):
    """Sinusoidal m/z embedding followed by a feed-forward projection."""

    def __init__(
        self,
        embedding_dim: int,
        *,
        lambda_params: Tuple[float, float] = (LAMBDA_MIN, LAMBDA_MAX),
        feedward_activation: Literal["relu", "selu", "gelu"] = "relu",
        dropout: float = 0.1,
        dropout_last_layer: bool = True,
    ):
        super(SinusodialMzEmbedding, self).__init__()

        if embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be even")

        self.embedding = SinusodialMz(
            embedding_dim, lambda_params=lambda_params
        )
        self.feedward_layers = MultiFeedForwardModule(
            embedding_dim,
            embedding_dim,
            embedding_dim,
            activation=feedward_activation,
            dropout=dropout,
            dropout_last_layer=dropout_last_layer,
        )

    def forward(self, mz: torch.Tensor):
        x = self.embedding(mz)
        x = self.feedward_layers(x)
        return x


class LinearMzEmbedding(nn.Module):
    """Linear projection of the scalar m/z to ``embedding_dim``."""

    def __init__(self, embedding_dim):
        super().__init__()
        self.linear = nn.Linear(1, embedding_dim)

    def forward(self, mz):
        return self.linear(mz.unsqueeze(-1))


class PeaksEmbedding(nn.Module):
    """
    Peak encoder combining sinusoidal m/z embedding and raw intensity.

    The intensity is concatenated as a single scalar channel and a
    small MLP is applied jointly on ``[mz_emb, intensity]``.
    """

    def __init__(
        self,
        embedding_dim: int,
        *,
        lambda_params: Tuple[float, float] = (LAMBDA_MIN, LAMBDA_MAX),
        feedward_activation: Literal["relu", "selu", "gelu"] = "relu",
        dropout: float = 0.1,
        dropout_last_layer: bool = False,
    ) -> None:
        super(PeaksEmbedding, self).__init__()

        self.mz_embedding = SinusodialMzEmbedding(
            embedding_dim,
            lambda_params=lambda_params,
            feedward_activation=feedward_activation,
            dropout=dropout,
            dropout_last_layer=dropout_last_layer,
        )
        self.intensity_embedding = MultiFeedForwardModule(
            embedding_dim + 1,
            embedding_dim,
            embedding_dim,
            activation=feedward_activation,
            dropout=dropout,
            dropout_last_layer=dropout_last_layer,
        )

    def forward(
        self,
        mz: torch.Tensor,
        intensity: torch.Tensor,
    ):
        mz_tensor = self.mz_embedding(mz)
        intensity_tensor = torch.unsqueeze(intensity, dim=-1)
        x = self.intensity_embedding(
            torch.cat([mz_tensor, intensity_tensor], dim=-1)
        )
        return x


class LinearPeaksEmbedding(nn.Module):
    """Peak encoder using a linear m/z embedding."""

    def __init__(
        self,
        embedding_dim: int,
        *,
        lambda_params: Tuple[float, float] = (LAMBDA_MIN, LAMBDA_MAX),
        feedward_activation: Literal["relu", "selu", "gelu"] = "relu",
        dropout: float = 0.1,
        dropout_last_layer: bool = False,
    ) -> None:
        super(LinearPeaksEmbedding, self).__init__()

        self.mz_embedding = LinearMzEmbedding(embedding_dim)
        self.intensity_embedding = MultiFeedForwardModule(
            embedding_dim + 1,
            embedding_dim,
            embedding_dim,
            activation=feedward_activation,
            dropout=dropout,
            dropout_last_layer=dropout_last_layer,
        )

    def forward(
        self,
        mz: torch.Tensor,
        intensity: torch.Tensor,
    ):
        mz_tensor = self.mz_embedding(mz)
        intensity_tensor = torch.unsqueeze(intensity, dim=-1)
        x = self.intensity_embedding(
            torch.cat([mz_tensor, intensity_tensor], dim=-1)
        )
        return x


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

class SpecRoFormer_RoPE(nn.Module):
    """
    RoPE Transformer baseline for MS/MS retrieval.

    Uses sinusoidal m/z embedding, RoPE encoder layers, and
    mask-aware mean pooling.
    """

    def __init__(
        self,
        embedding_dim: int,
        n_head: int,
        n_layer: int,
        dim_feedward: int,
        dim_target: int,
        *,
        lambda_params: Tuple[float, float] = (LAMBDA_MIN, LAMBDA_MAX),
        feedward_activation: Literal["relu", "selu", "gelu"] = "relu",
        dropout: float = 0.1,
        dropout_last_layer: bool = False,
        norm_first: bool = True,
    ) -> None:
        super(SpecRoFormer_RoPE, self).__init__()

        if embedding_dim % n_head != 0:
            raise ValueError("embedding must be divisible by n_head")

        self.embedding = PeaksEmbedding(
            embedding_dim,
            lambda_params=lambda_params,
            feedward_activation=feedward_activation,
            dropout=dropout,
            dropout_last_layer=dropout_last_layer,
        )

        if feedward_activation == "selu":
            # The Transformer encoder only supports gelu or relu.
            self.activation = "gelu"
        else:
            self.activation = feedward_activation

        if feedward_activation == "relu":
            self._activation = nn.ReLU()
        elif feedward_activation == "selu":
            self._activation = nn.SELU()
        elif feedward_activation == "gelu":
            self._activation = nn.GELU()
        else:
            raise ValueError(
                "activation must be relu or selu or gelu"
            )

        self.layers = nn.ModuleList(
            [
                RoPEEncoderLayer(
                    embedding_dim,
                    n_head,
                    dim_feedward,
                    dropout=dropout,
                )
                for _ in range(n_layer)
            ]
        )

        self._decoder = MultiFeedForwardModule(
            embedding_dim,
            dim_feedward,
            dim_target,
            activation=feedward_activation,
            dropout=dropout,
            dropout_last_layer=dropout_last_layer,
        )

    def forward(
        self,
        mz: torch.Tensor,
        intensity: torch.Tensor,
        mask: torch.Tensor,
    ):
        x = self.embedding(mz, intensity)

        for layer in self.layers:
            x = layer(x, mz, key_padding_mask=mask)

        # Mask-aware mean pooling.
        extended_mask = mask.unsqueeze(-1)
        masked_embeddings: torch.Tensor = x * (~extended_mask)
        sum_embeddings = masked_embeddings.sum(dim=1)
        num_valid_tokens = (~mask).sum(dim=1, keepdim=True)
        x = sum_embeddings / num_valid_tokens

        x = self._activation(self._decoder(x))
        return x


class SpecRoFormer_LinearMZ(nn.Module):
    """
    Ablation: RoPE Transformer with a linear m/z embedding instead
    of the sinusoidal one.
    """

    def __init__(
        self,
        embedding_dim: int,
        n_head: int,
        n_layer: int,
        dim_feedward: int,
        dim_target: int,
        *,
        lambda_params: Tuple[float, float] = (LAMBDA_MIN, LAMBDA_MAX),
        feedward_activation: Literal["relu", "selu", "gelu"] = "relu",
        dropout: float = 0.1,
        dropout_last_layer: bool = False,
        norm_first: bool = True,
    ) -> None:
        super(SpecRoFormer_LinearMZ, self).__init__()

        if embedding_dim % n_head != 0:
            raise ValueError("embedding must be divisible by n_head")

        self.embedding = LinearPeaksEmbedding(
            embedding_dim,
            lambda_params=lambda_params,
            feedward_activation=feedward_activation,
            dropout=dropout,
            dropout_last_layer=dropout_last_layer,
        )

        if feedward_activation == "selu":
            self.activation = "gelu"
        else:
            self.activation = feedward_activation

        if feedward_activation == "relu":
            self._activation = nn.ReLU()
        elif feedward_activation == "selu":
            self._activation = nn.SELU()
        elif feedward_activation == "gelu":
            self._activation = nn.GELU()
        else:
            raise ValueError(
                "activation must be relu or selu or gelu"
            )

        self.layers = nn.ModuleList(
            [
                RoPEEncoderLayer(
                    embedding_dim,
                    n_head,
                    dim_feedward,
                    dropout=dropout,
                )
                for _ in range(n_layer)
            ]
        )

        self._decoder = MultiFeedForwardModule(
            embedding_dim,
            dim_feedward,
            dim_target,
            activation=feedward_activation,
            dropout=dropout,
            dropout_last_layer=dropout_last_layer,
        )

    def forward(
        self,
        mz: torch.Tensor,
        intensity: torch.Tensor,
        mask: torch.Tensor,
    ):
        x = self.embedding(mz, intensity)

        for layer in self.layers:
            x = layer(x, mz, key_padding_mask=mask)

        extended_mask = mask.unsqueeze(-1)
        masked_embeddings: torch.Tensor = x * (~extended_mask)
        sum_embeddings = masked_embeddings.sum(dim=1)
        num_valid_tokens = (~mask).sum(dim=1, keepdim=True)
        x = sum_embeddings / num_valid_tokens

        x = self._activation(self._decoder(x))
        return x


class SpecRoFormer(nn.Module):
    """
    Final method: RoPE Transformer with precursor m/z conditioning.

    After the RoPE Transformer and mask-aware mean pooling, the
    precursor m/z is embedded and injected as a gated residual on
    the pooled representation.
    """

    def __init__(
        self,
        embedding_dim: int,
        n_head: int,
        n_layer: int,
        dim_feedward: int,
        dim_target: int,
        *,
        lambda_params: Tuple[float, float] = (LAMBDA_MIN, LAMBDA_MAX),
        feedward_activation: Literal["relu", "selu", "gelu"] = "relu",
        dropout: float = 0.1,
        dropout_last_layer: bool = False,
        norm_first: bool = True,
    ) -> None:
        super(SpecRoFormer, self).__init__()

        if embedding_dim % n_head != 0:
            raise ValueError("embedding must be divisible by n_head")

        self.embedding = PeaksEmbedding(
            embedding_dim,
            lambda_params=lambda_params,
            feedward_activation=feedward_activation,
            dropout=dropout,
            dropout_last_layer=dropout_last_layer,
        )

        if feedward_activation == "selu":
            self.activation = "gelu"
        else:
            self.activation = feedward_activation

        if feedward_activation == "relu":
            self._activation = nn.ReLU()
        elif feedward_activation == "selu":
            self._activation = nn.SELU()
        elif feedward_activation == "gelu":
            self._activation = nn.GELU()
        else:
            raise ValueError(
                "activation must be relu or selu or gelu"
            )

        self.layers = nn.ModuleList(
            [
                RoPEEncoderLayer(
                    embedding_dim,
                    n_head,
                    dim_feedward,
                    dropout=dropout,
                )
                for _ in range(n_layer)
            ]
        )

        self._decoder = MultiFeedForwardModule(
            embedding_dim,
            dim_feedward,
            dim_target,
            activation=feedward_activation,
            dropout=dropout,
            dropout_last_layer=dropout_last_layer,
        )

        # Precursor m/z conditioning.
        self.precursor_embedding = PrecursorMzEmbedding(
            embedding_dim=embedding_dim,
            mz_max=1000.0,
        )

        # Gate controlling the strength of the conditioning.
        self.mz_gate = nn.Parameter(torch.tensor(0.1))

        # Optional attention return flag, set by the caller at test time.
        self.return_attention = None

    def forward(
        self,
        mz: torch.Tensor,
        intensity: torch.Tensor,
        mask: torch.Tensor,
    ):
        all_attentions = []

        x = self.embedding(mz, intensity)

        for layer in self.layers:
            if self.return_attention:
                x, attn = layer(
                    x,
                    mz,
                    key_padding_mask=mask,
                    return_attention=True,
                )
                all_attentions.append(attn)
            else:
                x = layer(x, mz, key_padding_mask=mask)

        # Mask-aware mean pooling.
        extended_mask = mask.unsqueeze(-1)
        masked_embeddings: torch.Tensor = x * (~extended_mask)
        sum_embeddings = masked_embeddings.sum(dim=1)
        num_valid_tokens = (~mask).sum(dim=1, keepdim=True)
        x = sum_embeddings / num_valid_tokens

        # Global precursor conditioning.
        precursor_mz = mz[:, 0]  # [B]
        precursor_emb = self.precursor_embedding(
            precursor_mz
        )  # [B, D]

        # Gated residual injection.
        x = x + torch.tanh(self.mz_gate) * precursor_emb

        # Decoder head.
        x = self._activation(self._decoder(x))

        if self.return_attention:
            return x, all_attentions
        else:
            return x