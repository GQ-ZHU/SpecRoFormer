"""
Mini-batch step functions.

Define the train / evaluation step functions used by the Trainer, plus
a top-1 accuracy metric used as a custom metric.
"""

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import device

from SpecRoFormer.type import BatchType


def step_train(
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    batch: BatchType,
    custom_fn: Callable[..., torch.Tensor] = None,
):
    """
    Run one training / validation step for a single batch.

    The batch contains a list of inputs ``x`` and the target ``y``.
    Each element of ``x`` corresponds to one replica / view of the
    spectrum and is expanded into a separate forward pass.

    If the model returns a tuple ``(z, (recon_pred, recon_target))``,
    the loss is assumed to be multi-task and is called with the
    additional reconstruction arguments.
    """
    x, y = batch
    y = y.to(device)

    pred = []
    recon_preds = []
    recon_targets = []
    is_multi_task = False

    for item in x:
        item = [d.to(device) for d in item]
        out = model(*item)

        # Support both single-value and tuple returns from forward.
        if isinstance(out, tuple):
            # out = (z, (recon_pred, recon_target))
            if len(out) == 2 and isinstance(out[1], tuple):
                z, (r_pred, r_target) = out
                recon_preds.append(r_pred)
                recon_targets.append(r_target)
                is_multi_task = True
            else:
                z = out[0]
        else:
            z = out

        pred.append(z.unsqueeze(dim=1))

    pred = torch.cat(pred, dim=1)

    # Choose how the loss is called depending on whether it is multi-task.
    if is_multi_task:
        loss: torch.Tensor = criterion(
            pred,
            y,
            recon_preds=recon_preds,
            recon_targets=recon_targets,
        )
    else:
        loss: torch.Tensor = criterion(pred, y)

    if custom_fn is not None:
        custom_metric = custom_fn(pred.detach(), y.detach())
        return loss, custom_metric

    return loss


def step_evaluate(
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    batch: BatchType,
    custom_fn: Callable[..., torch.Tensor] = None,
):
    """
    Run one evaluation step for a single batch.

    Same contract as ``step_train`` but without gradient computation;
    the caller is expected to wrap this in ``torch.no_grad()``.
    """
    x, y = batch
    y = y.to(device)

    pred = []
    recon_preds = []
    recon_targets = []

    for item in x:
        item = [d.to(device) for d in item]
        out = model(*item)

        # Support both single-value and tuple returns from forward.
        if isinstance(out, tuple):
            if len(out) == 2 and isinstance(out[1], tuple):
                z, (recon_pred, recon_target) = out
                recon_preds.append(recon_pred)
                recon_targets.append(recon_target)
            else:
                z = out[0]
        else:
            z = out

        pred.append(z.unsqueeze(dim=1))

    pred = torch.cat(pred, dim=1)

    # Multi-task loss.
    if recon_preds:
        loss: torch.Tensor = criterion(
            pred,
            y,
            recon_preds=recon_preds,
            recon_targets=recon_targets,
        )
    else:
        loss: torch.Tensor = criterion(pred, y)

    if custom_fn is not None:
        custom_metric = custom_fn(pred.detach(), y.detach())
        return loss, custom_metric

    return loss


def top1_accuracy(pred: torch.Tensor, y: torch.Tensor):
    """
    Compute top-1 retrieval accuracy over query / reference replicas.

    ``pred`` has shape ``[batch_size, replica_count, dim]``.
    Positive pairs are identified by matching labels in ``y``.
    """
    device = pred.device
    replica_count = pred.shape[1]
    batch_size = y.shape[0]

    feature = torch.cat(
        torch.unbind(pred, dim=1),
        dim=0,
    )

    y = y.view(-1, 1)
    mask = torch.eq(y, y.T).float().to(device)
    mask = mask.repeat(replica_count, replica_count)

    slash_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * replica_count).view(-1, 1).to(device),
        0,
    )

    mask = mask * slash_mask

    feature = F.normalize(feature, dim=-1)

    cosine_score = torch.matmul(feature, feature.T)
    cosine_score = cosine_score * slash_mask

    hit = torch.gather(
        mask,
        dim=1,
        index=torch.argmax(cosine_score, dim=1, keepdim=True),
    ).squeeze()

    return hit.sum() / hit.shape[0]