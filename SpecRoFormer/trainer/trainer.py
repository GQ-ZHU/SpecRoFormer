"""
Trainer and ModelTester.

``Trainer`` handles the training loop, validation, checkpointing,
early stopping, learning rate scheduling, and metric storage.

``ModelTester`` runs inference over a dataloader and optionally
collects Δm/z attention profiles and raw attention records for later
visualization.
"""

import math
import random
import os
from typing import Literal, Optional, Union, Sequence

import json

import numpy as np
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from SpecRoFormer.type import (
    StepTrain,
    StepVal,
    TrainerConfig,
    SchedulerConfig,
    DescriptionConfig,
    StepFuncConfig,
    StorageConfig,
    CustomMetricConfig,
)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(n: Literal[0, 1]):
    """Return ``cuda:n`` if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{n}")
    return torch.device("cpu")


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the
    values of the cosine function between the initial lr set in the
    optimizer to 0, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Returns:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    def lr_lambda(current_step):
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0,
            0.5
            * (
                1.0
                + math.cos(
                    math.pi * float(num_cycles) * 2.0 * progress
                )
            ),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_warmup_constant_schedule(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    last_epoch=-1,
):
    """Linear warmup followed by a constant learning rate."""

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch)

class Trainer:
    def __init__(
        self,
        model: Module,
        reduction: Literal["mean", "sum"],
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        criterion: Module,
        device: torch.device,
        step_fn: StepFuncConfig,
        desc_config: DescriptionConfig,
        storage_config: StorageConfig,
        trainer_config: TrainerConfig,
        scheduler_config: SchedulerConfig,
        custom_metric_config: Optional[CustomMetricConfig] = None,
        save_by_epoch: bool = False,
    ) -> None:
        """
        Trainer for training and validating the model.

        The top-1 model on the validation set is saved. Early stopping
        is applied based on either the validation loss or a custom
        metric, depending on ``custom_metric_config``.

        Parameters
        ----------
        model : Module
            The model to be trained.
        reduction : {"mean", "sum"}
            Whether the loss function reduces by mean or sum.
        train_loader : DataLoader
            Training data.
        val_loader : DataLoader
            Validation data.
        optimizer : Optimizer
            Optimizer.
        criterion : Module
            Loss function.
        device : torch.device
            Device used for training.
        step_fn : StepFuncConfig
            Mini-batch step functions for train and validation.
        desc_config : DescriptionConfig
            Log templates for train / validation / end.
        storage_config : StorageConfig
            Storage paths for model, losses, metrics, scheduler.
        trainer_config : TrainerConfig
            Training epochs, early stopping, progress bar settings.
        scheduler_config : SchedulerConfig
            Learning rate scheduler configuration.
        custom_metric_config : CustomMetricConfig, optional
            If provided, validation uses this custom metric instead of
            the criterion for model selection.
        save_by_epoch : bool
            If True, save the model at every epoch regardless of
            validation performance.
        """
        self.model = model
        self.reduction = reduction
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.step_fn = step_fn
        self.desc_config = desc_config
        self.storage_config = storage_config
        self.trainer_config = trainer_config
        self.scheduler_config = scheduler_config
        self.custom_metric_config = custom_metric_config

        if scheduler_config["only_warmup"]:
            self.scheduler = get_warmup_constant_schedule(
                optimizer,
                scheduler_config["warmup_steps"],
                scheduler_config["total_steps"],
            )
        else:
            self.scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                scheduler_config["warmup_steps"],
                scheduler_config["total_steps"],
            )

        self._lr_seq = []
        self._step_loss_seq = []

        self.save_by_epoch = save_by_epoch

    @property
    def lrs(self):
        return self._lrs.copy()

    @property
    def step_loss(self):
        return self._step_loss.copy()

    def step_value(
        self,
        value: int,
        size: int,
        reduction: Literal["mean", "sum"],
    ):
        """Scale a step metric according to the reduction mode."""
        if reduction == "mean":
            return value * size
        else:
            return value

    def process_epoch(
        self,
        epoch: int,
        dataloader: DataLoader,
        step_fn: Union[StepTrain, StepVal],
        is_train: bool = True,
    ):
        """
        Run one epoch of training or validation.

        Returns
        -------
        mean_loss : float
        mean_custom_metric : float
        """
        num = 0
        loss_sum = 0
        custom_metric_sum = 0

        pbar = dataloader
        if self.trainer_config["show_progress_bar"]:
            pbar = tqdm(dataloader, total=len(dataloader))
            if is_train:
                pbar.set_description(
                    f"Train: Epoch "
                    f"[{epoch + 1}/{self.trainer_config['n_epoch']}]"
                )
            else:
                pbar.set_description(
                    f"Validation: Epoch "
                    f"[{epoch + 1}/{self.trainer_config['n_epoch']}]"
                )

        if is_train:
            self.model.train()
        else:
            self.model.eval()

        for batch in pbar:
            size = len(batch[-1])
            num += size

            if self.custom_metric_config is None:
                step_loss = step_fn(
                    self.model,
                    self.criterion,
                    self.device,
                    batch,
                    None,
                )
            else:
                step_loss, custom_metric = step_fn(
                    self.model,
                    self.criterion,
                    self.device,
                    batch,
                    self.custom_metric_config["fn"],
                )
                custom_metric_sum += self.step_value(
                    custom_metric,
                    size,
                    self.reduction,
                )

            self._step_loss_seq.append(step_loss.item())

            loss_sum += self.step_value(
                step_loss.item(),
                size,
                self.reduction,
            )

            if is_train:
                self.optimizer.zero_grad()
                step_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    0.5,
                )

                self.optimizer.step()
                self._lr_seq.append(
                    self.optimizer.param_groups[0]["lr"]
                )
                self.scheduler.step()

            if self.trainer_config["show_progress_bar"]:
                if self.custom_metric_config is not None:
                    pbar.set_postfix_str(
                        f"{self.custom_metric_config['name']}="
                        f"{custom_metric.item():.3f}, "
                        f"loss={step_loss.item(): .4f}"
                    )
                else:
                    pbar.set_postfix(
                        loss=f"{step_loss.item(): .4f}"
                    )

        return loss_sum / num, custom_metric_sum / num

    def train(self):
        """Run the full training loop with validation and early stopping."""

        def checkpoint():
            torch.save(
                self.model.state_dict(),
                self.storage_config["model"],
            )

        def maximum(x, y):
            return x > y

        def minimum(x, y):
            return x < y

        loss_metrics = {
            "train": [],
            "validation": [],
        }

        custom_metrics = {
            "train": [],
            "validation": [],
        }

        best_model_loss = torch.inf
        best_custom_metric = torch.inf
        is_save = minimum

        if (
            self.custom_metric_config is not None
            and self.custom_metric_config["destination"] == "maximum"
        ):
            best_custom_metric = 0
            is_save = maximum

        early_stop = 0
        best_epoch = 0

        checkpoint()

        for epoch in range(self.trainer_config["n_epoch"]):
            train_loss, train_custom_metric = self.process_epoch(
                epoch,
                self.train_loader,
                self.step_fn["train"],
                True,
            )

            if self.custom_metric_config is not None:
                print(
                    self.desc_config["train"].format(
                        epoch + 1,
                        train_loss,
                        self.custom_metric_config["name"],
                        train_custom_metric,
                    )
                )
            else:
                print(
                    self.desc_config["train"].format(
                        epoch + 1,
                        train_loss,
                    )
                )

            with torch.no_grad():
                val_loss, val_custom_metric = self.process_epoch(
                    epoch,
                    self.val_loader,
                    self.step_fn["val"],
                    False,
                )

            if self.custom_metric_config is not None:
                if is_save(val_custom_metric, best_custom_metric):
                    early_stop = 0
                    best_epoch = epoch
                    best_model_loss = val_loss
                    best_custom_metric = val_custom_metric
                    print(
                        self.desc_config["val"].format(
                            epoch + 1,
                            val_loss,
                            self.custom_metric_config["name"],
                            val_custom_metric,
                        )
                    )
                    checkpoint()
                else:
                    early_stop += 1
            else:
                if val_loss < best_model_loss:
                    early_stop = 0
                    best_epoch = epoch
                    best_model_loss = val_loss
                    print(
                        self.desc_config["val"].format(
                            epoch + 1,
                            val_loss,
                        )
                    )
                    checkpoint()
                else:
                    early_stop += 1

            if self.save_by_epoch:
                checkpoint()
                print(f"model saved at epoch {epoch}")

            loss_metrics["train"].append(train_loss)
            loss_metrics["validation"].append(val_loss)
            custom_metrics["train"].append(train_custom_metric)
            custom_metrics["validation"].append(val_custom_metric)

            if early_stop == self.trainer_config["early_stop"]:
                print(f"early stop at epoch {epoch}")
                break

        print(f"best model saved in epoch {best_epoch}")

        if self.custom_metric_config is not None:
            print(
                self.desc_config["end"].format(
                    best_model_loss,
                    self.custom_metric_config["name"],
                    best_custom_metric,
                )
            )
        else:
            print(
                self.desc_config["end"].format(
                    best_model_loss,
                )
            )

        print("store train metrics")
        np.save(self.storage_config["loss"], loss_metrics)
        np.save(self.storage_config["custom"], custom_metrics)
        np.save(self.storage_config["lr"], self._lr_seq)
        np.save(self.storage_config["step_loss"], self._step_loss_seq)

class ModelTester:
    def __init__(
        self,
        model: Module,
        device: torch.device,
        show_progress_bar: bool = True,
        collect_attention_profile: bool = False,
        identifier_list: Optional[list] = None,
        target_indices: Optional[Sequence[int]] = None,
        save_last_layer_only: bool = True,
        profile_config: Optional[dict] = None,
        sample_metadata: Optional[dict[int, dict]] = None,
        save_attention: bool = False,
        attention_save_path: Optional[str] = None,
    ) -> None:
        """
        Test-time inference wrapper.

        Returns embeddings for all samples in the dataloader, and
        optionally collects Δm/z attention profiles and raw attention
        records for plotting and downstream analysis.

        Parameters
        ----------
        model : Module
        device : torch.device
        show_progress_bar : bool
            Show a tqdm progress bar during inference.
        collect_attention_profile : bool
            If True, collect compressed attention statistics per
            sample for plotting. When False, only embeddings are
            returned.
        identifier_list : list, optional
            Per-sample identifier list aligned with the dataloader
            order (e.g. SMILES, query id, ...).
        target_indices : Sequence[int], optional
            If provided, only these query indices are used for
            attention-profile collection. Embeddings are still
            computed for all samples.
        save_last_layer_only : bool
            In the current analysis setting, only the final Transformer
            layer is needed. This flag is kept for extensibility.
        profile_config : dict, optional
            Configuration used to compute Δm/z attention profiles.
        sample_metadata : dict, optional
            Optional metadata attached to each sample index, e.g.
            ``{17: {"group_label": "easy", "top1_hit": True, ...}}``.
        save_attention : bool
            If True, save raw attention records to disk.
        attention_save_path : str, optional
            Destination path for the saved attention records.
        """
        self.model = model
        self.device = device
        self.show_progress_bar = show_progress_bar

        # Whether to collect compressed attention statistics for plotting.
        # When False, the tester only returns embeddings.
        self.collect_attention_profile = collect_attention_profile

        # Optional per-sample identifier list (e.g., SMILES, query id, etc.).
        # The index in this list must align with the dataloader order.
        self.identifier_list = identifier_list

        # If target_indices is provided, only these query indices will be used
        # for attention-profile collection. Embeddings are still computed for all
        # samples unless the upper pipeline filters them earlier.
        self.target_indices: Optional[set[int]] = (
            None if target_indices is None else set(target_indices)
        )

        # In the current analysis setting, we only need the final Transformer layer.
        # This flag is kept for extensibility and explicit behavior control.
        self.save_last_layer_only: bool = save_last_layer_only

        # Optional metadata attached to each sample index.
        # Example:
        # sample_metadata = {
        #     17: {
        #         "group_label": "easy",
        #         "first_hit_rank": 1,
        #         "top1_hit": True,
        #         "top5_hit": True,
        #     }
        # }
        self.sample_metadata = sample_metadata or {}

        # Configuration used to compute Δm/z attention profiles.
        self.profile_config = profile_config or {
            "bin_size": 1.0,
            "max_delta": 200.0,
            "exclude_diagonal": False,
        }

        self.save_attention = save_attention
        self.attention_save_path = attention_save_path
        self.attention_records = []

    def _compute_delta_profile(
        self,
        attn: torch.Tensor,
        mz: torch.Tensor,
        mask: torch.Tensor,
    ):
        """
        Compute the Δm/z attention profile for a single sample.

        Parameters
        ----------
        attn : torch.Tensor
            Attention matrix of shape ``[L, L]``, typically after head
            averaging.
        mz : torch.Tensor
            m/z values of shape ``[L]``.
        mask : torch.Tensor
            Padding mask of shape ``[L]``, where True indicates padding.

        Returns
        -------
        bin_centers : np.ndarray
            Centers of Δm/z bins.
        avg_profile : np.ndarray
            Mean attention value within each Δm/z bin.
        valid_peak_count : int
            Number of valid (non-padding) peaks.
        """
        attn = attn.detach().cpu().numpy()
        mz = mz.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy().astype(bool)

        # Remove padding tokens before profile computation.
        valid = ~mask
        mz = mz[valid]
        attn = attn[valid][:, valid]

        valid_peak_count = len(mz)

        bin_size = self.profile_config["bin_size"]
        max_delta = self.profile_config["max_delta"]
        exclude_diagonal = self.profile_config["exclude_diagonal"]

        if valid_peak_count == 0:
            edges = np.arange(0.0, max_delta + bin_size, bin_size)
            centers = edges[:-1] + bin_size / 2
            return centers, np.zeros_like(centers), 0

        # Pairwise relative m/z difference matrix.
        delta = np.abs(mz[:, None] - mz[None, :])

        # Attention weights corresponding to each peak pair.
        weights = attn.copy()

        # Optional removal of diagonal self-attention.
        if exclude_diagonal:
            np.fill_diagonal(weights, 0.0)

        # Keep only pairs within the desired Δm/z range.
        valid_delta = delta <= max_delta
        delta_flat = delta[valid_delta]
        weight_flat = weights[valid_delta]

        # Bin Δm/z values and average attention inside each bin.
        edges = np.arange(0.0, max_delta + bin_size, bin_size)
        bin_ids = np.digitize(delta_flat, edges) - 1
        keep = (bin_ids >= 0) & (bin_ids < len(edges) - 1)

        bin_ids = bin_ids[keep]
        weight_flat = weight_flat[keep]

        sum_weights = np.bincount(
            bin_ids,
            weights=weight_flat,
            minlength=len(edges) - 1,
        )
        count_weights = np.bincount(
            bin_ids,
            minlength=len(edges) - 1,
        )

        avg_profile = np.divide(
            sum_weights,
            count_weights,
            out=np.zeros_like(sum_weights, dtype=float),
            where=count_weights > 0,
        )

        centers = edges[:-1] + bin_size / 2
        return centers, avg_profile, valid_peak_count

    def _build_profile_record(
        self,
        current_index: int,
        identifier: str,
        bin_centers: np.ndarray,
        avg_profile: np.ndarray,
        valid_peak_count: int,
    ) -> dict:
        """
        Build one sample-level record for downstream plotting and
        statistics.

        Saved statistics
        ----------------
        query_index :
            Original query index in the evaluated dataset.
        identifier :
            Human-readable identifier aligned with the query, such as
            SMILES or another sample key.
        valid_peak_count :
            Number of valid non-padding peaks used in profile
            computation.
        group_label :
            Optional difficulty label such as 'easy', 'medium', 'hard'.
        first_hit_rank :
            Rank position of the first correct retrieval result.
        top1_hit / top5_hit :
            Boolean retrieval indicators.
        profile_peak_value :
            Maximum value of the Δm/z profile.
        profile_peak_delta :
            Δm/z location at which the profile reaches its maximum.
        mean_attention :
            Mean value of the Δm/z profile across all bins.
        attention_entropy :
            Entropy of the normalized Δm/z profile.
        nonzero_bin_count :
            Number of bins whose average attention is greater than zero.
        delta_bins / delta_profile :
            Full binned profile used for later plotting and aggregation.
        """
        metadata = self.sample_metadata.get(current_index, {})

        if len(avg_profile) > 0:
            peak_idx = int(np.argmax(avg_profile))
            profile_peak_value = float(avg_profile[peak_idx])
            profile_peak_delta = float(bin_centers[peak_idx])
            mean_attention = float(np.mean(avg_profile))
            nonzero_bin_count = int(np.sum(avg_profile > 0))

            profile_sum = float(np.sum(avg_profile))
            if profile_sum > 0:
                p = avg_profile / profile_sum
                attention_entropy = float(
                    -(p * np.log(p + 1e-12)).sum()
                )
            else:
                attention_entropy = 0.0
        else:
            profile_peak_value = 0.0
            profile_peak_delta = -1.0
            mean_attention = 0.0
            nonzero_bin_count = 0
            attention_entropy = 0.0

        record = {
            "query_index": current_index,
            "identifier": identifier,
            "valid_peak_count": valid_peak_count,

            # Optional retrieval metadata for later regrouping.
            "group_label": metadata.get("group_label", None),
            "first_hit_rank": metadata.get("first_hit_rank", None),
            "top1_hit": metadata.get("top1_hit", None),
            "top5_hit": metadata.get("top5_hit", None),
            "top10_hit": metadata.get("top10_hit", None),

            # Compact summary statistics of the Δm/z profile.
            "profile_peak_value": profile_peak_value,
            "profile_peak_delta": profile_peak_delta,
            "mean_attention": mean_attention,
            "attention_entropy": attention_entropy,
            "nonzero_bin_count": nonzero_bin_count,

            # Full profile for plotting.
            "delta_bins": bin_centers.tolist(),
            "delta_profile": avg_profile.tolist(),
        }
        return record

    def _build_attention_record(
        self,
        current_index: int,
        identifier: str,
        smiles: str,
        attention: torch.Tensor,
        mz: torch.Tensor,
        intensity: torch.Tensor,
        mask: torch.Tensor,
        embedding: torch.Tensor,
    ):
        """
        Save raw attention information for later visualization.

        The attention tensor is intentionally saved BEFORE head
        averaging, so that later analyses can visualize either
        individual heads or averaged attention without rerunning
        inference.
        """
        metadata = self.sample_metadata.get(current_index, {})

        record = {
            "query_index": current_index,
            "identifier": identifier,
            "smiles": smiles,
            "valid_length": int(
                (~mask.numpy().astype(bool)).sum()
            ),
            "group_label": metadata.get("group_label"),
            "first_hit_rank": metadata.get("first_hit_rank"),
            "top1_hit": metadata.get("top1_hit"),
            "top5_hit": metadata.get("top5_hit"),
            "top10_hit": metadata.get("top10_hit"),

            # ---------- spectrum ----------
            "mz": mz.numpy(),
            "intensity": intensity.numpy(),
            "mask": mask.numpy(),

            # ---------- model ----------
            "attention": attention.numpy(),
            "embedding": embedding.numpy(),
        }

        return record

    def _save_attention_records(self):
        """Persist collected attention records to disk."""
        if not self.save_attention:
            return

        if self.attention_save_path is None:
            return

        os.makedirs(
            os.path.dirname(self.attention_save_path),
            exist_ok=True,
        )

        torch.save(
            self.attention_records,
            self.attention_save_path,
        )

        print(
            f"Attention records saved to "
            f"{self.attention_save_path}"
        )

    def test(self, dataloader: DataLoader):
        """
        Run inference on the dataloader.

        Returns
        -------
        embedding_array : np.ndarray
            Embeddings for all samples in the dataloader.
        profile_records : list[dict]
            Sample-level attention-profile records. Empty when
            ``collect_attention_profile`` is False.
        """
        self.model.eval()

        result = []
        profile_records = []
        sample_index = 0

        with torch.no_grad():
            pbar = dataloader
            if self.show_progress_bar:
                pbar = tqdm(
                    dataloader,
                    total=len(dataloader),
                    desc="embedding",
                )

            for x in pbar:
                # Only request attention tensors when profile collection
                # is enabled.
                self.model.return_attention = (
                    self.collect_attention_profile
                )
                attentions = None

                x = [d.to(self.device) for d in x]
                out = self.model(*x)

                if isinstance(out, tuple):
                    z = out[0]
                    attentions = out[1]
                else:
                    z = out

                result.append(z.detach().cpu().numpy())

                if (
                    self.collect_attention_profile
                    and attentions is not None
                ):
                    if isinstance(attentions, list):
                        attention_tensor_list = [
                            attn.detach().cpu()
                            for attn in attentions
                        ]
                    else:
                        attention_tensor_list = [
                            attentions.detach().cpu()
                        ]

                    batch_size = z.shape[0]

                    for i in range(batch_size):
                        current_index = sample_index

                        should_collect = (
                            self.target_indices is None
                            or current_index in self.target_indices
                        )

                        if should_collect:
                            identifier = f"sample_{current_index}"
                            smiles = None
                            if self.identifier_list is not None:
                                identifier = self.identifier_list[
                                    current_index
                                ]
                                smiles = self.identifier_list[
                                    current_index
                                ]

                            # In the current study, only the final
                            # layer is used. The branch is kept
                            # explicit for clarity and extensibility.
                            if self.save_last_layer_only:
                                sample_attn = (
                                    attention_tensor_list[-1][i]
                                )
                            else:
                                sample_attn = (
                                    attention_tensor_list[-1][i]
                                )

                            sample_mz = x[0][i].detach().cpu()
                            sample_intensity = (
                                x[1][i].detach().cpu()
                            )
                            sample_mask = x[2][i].detach().cpu()

                            if self.save_attention:
                                self.attention_records.append(
                                    self._build_attention_record(
                                        current_index=current_index,
                                        identifier=identifier,
                                        smiles=smiles,
                                        attention=(
                                            sample_attn.detach().cpu()
                                        ),
                                        mz=sample_mz,
                                        intensity=sample_intensity,
                                        mask=sample_mask,
                                        embedding=(
                                            z[i].detach().cpu()
                                        ),
                                    )
                                )

                            # If the attention tensor shape is
                            # [H, L, L], average across heads before
                            # computing the profile.
                            if sample_attn.ndim == 3:
                                sample_attn = sample_attn.mean(dim=0)

                            bin_centers, avg_profile, valid_peak_count = (
                                self._compute_delta_profile(
                                    sample_attn,
                                    sample_mz,
                                    sample_mask,
                                )
                            )

                            record = self._build_profile_record(
                                current_index=current_index,
                                identifier=identifier,
                                bin_centers=bin_centers,
                                avg_profile=avg_profile,
                                valid_peak_count=valid_peak_count,
                            )
                            profile_records.append(record)

                        sample_index += 1
                else:
                    sample_index += z.shape[0]

        self._save_attention_records()
        embedding_array = np.concatenate(result, axis=0)
        return embedding_array, profile_records
                           