"""
Type definitions.

TypedDict / type alias definitions shared across the training and
evaluation pipeline.
"""

from typing import (
    TypedDict,
    Sequence,
    Callable,
    Optional,
    Literal,
)

import torch
from torch import nn
from torch import device
import numpy as np
import numpy.typing as npt


# ---------------------------------------------------------------------
# Batch / step signatures
# ---------------------------------------------------------------------

BatchType = Sequence[torch.Tensor]

StepTrain = Callable[
    [
        nn.Module,
        nn.Module,
        device,
        BatchType,
        Optional[Callable[..., int]],
    ],
    Sequence[torch.Tensor],
]

StepVal = Callable[
    [
        nn.Module,
        nn.Module,
        device,
        BatchType,
        Optional[Callable[..., int]],
    ],
    Sequence[torch.Tensor],
]


# ---------------------------------------------------------------------
# Spectrum / token data
# ---------------------------------------------------------------------

class Peak(TypedDict):
    """A single peak with string m/z and intensity array."""

    mz: str
    intensity: npt.NDArray


class MetaData(TypedDict):
    """Peak list plus SMILES for a spectrum."""

    peaks: Sequence[Peak]
    smiles: str


class TokenSequence(TypedDict):
    """Tokenized representation of a spectrum."""

    mz: npt.NDArray[np.int32]
    intensity: npt.NDArray[np.float32]
    mask: npt.NDArray[np.bool_]
    smiles: str


# ---------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------

class OptimizerConfig(TypedDict):
    lr: float
    weight_decay: float


class TokenizerConfig(TypedDict):
    max_len: int
    show_progress_bar: bool


class TrainerConfig(TypedDict):
    n_epoch: int
    early_stop: int
    device: device
    show_progress_bar: bool


class SchedulerConfig(TypedDict):
    warmup_steps: int
    total_steps: int
    only_warmup: bool


class DescriptionConfig(TypedDict):
    train: str
    val: str
    end: str


class StepFuncConfig(TypedDict):
    train: StepTrain
    val: StepVal


class DataLoaderConfig(TypedDict):
    batch_size: int
    val_ratio: float


class StorageConfig(TypedDict):
    model: str
    lr: str
    step_loss: str
    loss: str
    custom: str


class CustomMetricConfig(TypedDict):
    name: str
    fn: Callable[..., int]
    destination: Literal["maximum", "minimum"]


# ---------------------------------------------------------------------
# Loss configuration
# ---------------------------------------------------------------------

class TanimotoLossConfig(TypedDict):
    score_path: str
    device: torch.device
    reduction: Optional[Literal["mean", "sum"]]


class SupConLossConfig(TypedDict):
    device: torch.device
    temperature: Optional[float]
    contrast_mode: Optional[Literal["one", "all"]]
    base_temperature: Optional[float]
    reduction: Optional[Literal["mean", "sum"]]


class SupConLossWithTanimotoScoreConfig(TypedDict):
    alpha: int
    score_path: str
    device: torch.device
    temperature: Optional[float]
    contrast_mode: Optional[Literal["one", "all"]]
    base_temperature: Optional[float]
    reduction: Optional[Literal["mean", "sum"]]


class SupConLossWithTanimotoAndReconConfig(TypedDict):
    alpha: float
    beta: float
    score_path: str
    device: torch.device
    temperature: Optional[float]
    contrast_mode: Optional[Literal["one", "all"]]
    base_temperature: Optional[float]
    reduction: Optional[Literal["mean", "sum"]]


# ---------------------------------------------------------------------
# Augmentation configuration
# ---------------------------------------------------------------------

class AugmentationConfig(TypedDict):
    """Data augmentation configuration for contrastive views."""

    # View-level control: whether to enable augmentation for a view.
    prob: float

    # Sub-spectrum consistency (core structural augmentation).
    subspec_prob: float
    subspec_drop_ratio: float

    # Noise-aware augmentation (low-intensity peak removal).
    low_intensity_prob: float
    removal_max: float
    removal_intensity: float

    # Intensity perturbation.
    rate_intensity: float


DefaultAugmentationConifg = AugmentationConfig(
    prob=0.5,
    removal_max=0.2,
    removal_intensity=0.3,
    rate_intensity=0.15,
)