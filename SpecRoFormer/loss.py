"""
Contrastive and structure-aware loss functions.

This module defines the loss functions used for training the
SpecRoFormer family of models:

- ``TanimotoScoreLoss``              : regress cosine similarity to
  structural (Tanimoto) similarity.
- ``SupConLoss``                     : supervised contrastive loss.
- ``SupConLossWithTanimotoScore``    : SupCon + Tanimoto regression.
- ``StructureAwareSupConWithTanimoto``: structure-aware SupCon where
  positive pairs are reweighted by Tanimoto similarity, plus a
  Tanimoto regression term.
"""

from typing import Literal

import numpy as np
import numpy.typing as npt
import torch
from torch import nn, device
import torch.nn.functional as F


class TanimotoScoreLoss(nn.Module):
    """
    Regress the cosine similarity between embedding views to the
    pairwise structural (Tanimoto) similarity of the corresponding
    compounds.

    Parameters
    ----------
    score_path : str
        Path to a precomputed Tanimoto similarity matrix (``.npy``).
    device : torch.device
        Device on which to allocate the target tensor.
    reduction : {"mean", "sum"}
        Reduction applied to the MSE loss.
    """

    def __init__(
        self,
        score_path: str,
        device: device,
        reduction: Literal["mean", "sum"] = "mean",
    ) -> None:
        super(TanimotoScoreLoss, self).__init__()

        # Lazy read to avoid loading the full matrix into memory.
        self.tanimoto_score: npt.NDArray[np.float32] = np.load(
            score_path, mmap_mode="r"
        )
        self.mse_loss = torch.nn.MSELoss(reduction=reduction)
        self.device = device

    def target_score(self, row: npt.NDArray):
        """Return the pairwise Tanimoto submatrix for the given rows."""
        return np.array(
            self.tanimoto_score[row][:, row], dtype=np.float32
        )

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor = None,
    ):
        n_views = features.shape[1]

        tanimoto_score = torch.tensor(
            self.target_score(labels.cpu().numpy())
        ).to(self.device)

        tanimoto_score = tanimoto_score.repeat_interleave(
            n_views, 0
        ).repeat_interleave(n_views, 1)

        normalized_feature = F.normalize(
            features.reshape(-1, features.shape[-1]),
            dim=-1,
        )

        pred_cosine_score = torch.matmul(
            normalized_feature, normalized_feature.T
        )

        return self.mse_loss(tanimoto_score, pred_cosine_score)


class SupConLoss(nn.Module):
    """
    Supervised contrastive loss.

    If both ``labels`` and ``mask`` are None, this degenerates into
    the SimCLR unsupervised loss.

    Reference
    ---------
    Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
    https://arxiv.org/pdf/2002.05709.pdf
    """

    def __init__(
        self,
        device: torch.device,
        temperature: float = 0.2,
        contrast_mode: Literal["one", "all"] = "all",
        base_temperature: float = 0.2,
        reduction: Literal["mean", "sum"] = "mean",
    ):
        super(SupConLoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.reduction = reduction
        self.leaky_relu = nn.LeakyReLU()

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor = None,
        mask: torch.Tensor = None,
    ):
        """
        Parameters
        ----------
        features : torch.Tensor
            Hidden vector of shape ``[bsz, n_views, ...]``.
        labels : torch.Tensor, optional
            Ground truth of shape ``[bsz]``.
        mask : torch.Tensor, optional
            Contrastive mask of shape ``[bsz, bsz]``, where
            ``mask[i, j] = 1`` if sample ``j`` has the same class as
            sample ``i``. Can be asymmetric.

        Returns
        -------
        loss : torch.Tensor
            Scalar loss.
        """
        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )

        if len(features.shape) > 3:
            features = features.view(
                features.shape[0], features.shape[1], -1
            )

        features = F.normalize(features, dim=-1)
        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(
                batch_size, dtype=torch.float32
            ).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError(
                    "Num of labels does not match num of features"
                )
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]

        # [batch, n_views, feature] -> [(batch, feature)] * n_views
        contrast_feature = torch.cat(
            torch.unbind(features, dim=1), dim=0
        )

        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(
                "Unknown mode: {}".format(self.contrast_mode)
            )

        # Inner-product similarity.
        anchor_dot_contrast = torch.div(
            self.leaky_relu(
                torch.matmul(anchor_feature, contrast_feature.T)
            ),
            self.temperature,
        )

        # Numerical stability: subtract the row-wise max before softmax.
        logits_max, _ = torch.max(
            anchor_dot_contrast, dim=1, keepdim=True
        )
        logits = anchor_dot_contrast - logits_max.detach()

        # Tile the mask across views.
        mask = mask.repeat(anchor_count, contrast_count)

        # Mask out self-contrast cases.
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count)
            .view(-1, 1)
            .to(self.device),
            0,
        )
        mask = mask * logits_mask

        # log_prob = logits - log(sum(exp(logits)))
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(
            exp_logits.sum(1, keepdim=True)
        )

        # Mean log-likelihood over positive pairs.
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = -(
            self.temperature / self.base_temperature
        ) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size)

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        else:
            raise ValueError(f"Unkown reduction: {self.reduction}")

        return loss


class StructureAwareSupConWithTanimoto(nn.Module):
    """
    Structure-aware supervised contrastive loss combined with a
    Tanimoto regression term.

    Positive pairs in the contrastive loss are reweighted by
    ``1 + beta * tanimoto``, so that structurally closer compounds
    contribute more strongly to the loss.
    """

    def __init__(
        self,
        alpha: int,
        score_path: str,
        device: torch.device,
        temperature: float = 0.2,
        contrast_mode: Literal["one", "all"] = "all",
        base_temperature: float = 0.2,
        reduction: Literal["mean", "sum"] = "mean",
        beta: float = 0.5,
    ) -> None:
        super().__init__()

        self.device = device
        self.beta = beta
        self.alpha = alpha

        # Full Tanimoto matrix, memory-mapped.
        self.tanimoto_score = np.load(score_path, mmap_mode="r")

        self.supcon_loss = StructureAwareSupervisedContrastiveLoss(
            device=device,
            temperature=temperature,
            contrast_mode=contrast_mode,
            base_temperature=base_temperature,
            reduction=reduction,
        )

        self.tanimotoscore_loss = TanimotoScoreLoss(
            score_path, device, reduction
        )

    def get_batch_tanimoto(self, labels: torch.Tensor):
        """
        Slice out the Tanimoto submatrix for the current batch labels.
        """
        idx = labels.cpu().numpy()
        sub_matrix = self.tanimoto_score[idx][:, idx]
        return torch.tensor(
            sub_matrix, dtype=torch.float32, device=self.device
        )

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        if labels is None:
            raise ValueError(
                "Structure-aware SupCon requires labels."
            )

        batch_tanimoto = self.get_batch_tanimoto(labels)

        supcontrastive_loss = self.supcon_loss(
            features=features,
            labels=labels,
            mask=mask,
            tanimoto=batch_tanimoto,
            beta=self.beta,
        )

        tanimoto_loss = self.tanimotoscore_loss(features, labels)

        return supcontrastive_loss + self.alpha * tanimoto_loss


class StructureAwareSupervisedContrastiveLoss(nn.Module):
    """
    Structure-aware supervised contrastive loss.

    Compared to ``SupConLoss``, the positive mask is additionally
    weighted by ``1 + beta * tanimoto`` so that structurally closer
    positives contribute more strongly.
    """

    def __init__(
        self,
        device: torch.device,
        temperature: float = 0.2,
        contrast_mode: Literal["one", "all"] = "all",
        base_temperature: float = 0.2,
        reduction: Literal["mean", "sum"] = "mean",
    ):
        super(StructureAwareSupervisedContrastiveLoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.reduction = reduction
        self.leaky_relu = nn.LeakyReLU()

    def forward(
        self,
        features,
        labels=None,
        mask=None,
        tanimoto=None,
        beta=0.5,
    ):
        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...]"
            )

        if len(features.shape) > 3:
            features = features.view(
                features.shape[0], features.shape[1], -1
            )

        features = F.normalize(features, dim=-1)
        batch_size = features.shape[0]
        n_views = features.shape[1]

        # ----- 1. Base label-level mask -----
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(
                batch_size, dtype=torch.float32
            ).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError(
                    "Num of labels does not match num of features"
                )
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        # ----- 2. Expand features for contrast -----
        contrast_count = n_views
        contrast_feature = torch.cat(
            torch.unbind(features, dim=1), dim=0
        )

        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(
                f"Unknown mode: {self.contrast_mode}"
            )

        # ----- 3. Compute logits -----
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature,
        )

        logits_max, _ = torch.max(
            anchor_dot_contrast, dim=1, keepdim=True
        )
        logits = anchor_dot_contrast - logits_max.detach()

        # ----- 4. Expand mask to view level -----
        mask = mask.repeat(anchor_count, contrast_count)

        # ----- 5. Remove self-contrast -----
        logits_mask = torch.ones_like(mask)
        logits_mask.scatter_(
            1,
            torch.arange(batch_size * anchor_count)
            .view(-1, 1)
            .to(self.device),
            0,
        )

        mask = mask * logits_mask

        # ----- 6. Structure-aware positive weighting -----
        if tanimoto is not None:
            # Expected shape: [B, B]
            tanimoto = tanimoto.repeat(n_views, n_views)
            assert (
                tanimoto.shape == mask.shape
            ), f"Shape mismatch: {tanimoto.shape} vs {mask.shape}"
            positive_weight = 1 + beta * tanimoto
            weighted_mask = mask * positive_weight
        else:
            weighted_mask = mask

        weighted_mask = weighted_mask * logits_mask

        # ----- 7. Compute log_prob -----
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(
            exp_logits.sum(1, keepdim=True) + 1e-12
        )

        # ----- 8. Weighted positive aggregation -----
        pos_weight_sum = weighted_mask.sum(1)
        valid = pos_weight_sum > 0

        mean_log_prob_pos = torch.zeros_like(pos_weight_sum)
        mean_log_prob_pos[valid] = (
            (weighted_mask[valid] * log_prob[valid]).sum(1)
            / (pos_weight_sum[valid] + 1e-12)
        )

        # ----- 9. Final loss -----
        loss = -(
            self.temperature / self.base_temperature
        ) * mean_log_prob_pos

        loss = loss.view(anchor_count, batch_size)

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

        return loss


class SupConLossWithTanimotoScore(nn.Module):
    """
    Combination of ``SupConLoss`` and ``TanimotoScoreLoss``.

    ``loss = supcon + alpha * tanimoto_regression``
    """

    def __init__(
        self,
        alpha: int,
        score_path: str,
        device: device,
        temperature: float = 0.2,
        contrast_mode: Literal["one", "all"] = "all",
        base_temperature: float = 0.2,
        reduction: Literal["mean", "sum"] = "mean",
    ) -> None:
        super(SupConLossWithTanimotoScore, self).__init__()

        self.supcon_loss = SupConLoss(
            device,
            temperature,
            contrast_mode,
            base_temperature,
            reduction,
        )
        self.tanimotoscore_loss = TanimotoScoreLoss(
            score_path, device, reduction
        )
        self.alpha = alpha

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor = None,
        mask: torch.Tensor = None,
    ):
        supcontrastive_loss = self.supcon_loss(
            features, labels, mask
        )
        tanimoto_loss = self.tanimotoscore_loss(features, labels)
        return supcontrastive_loss + self.alpha * tanimoto_loss