"""
Training entry point.

Parse command-line arguments, load and tokenize the GNPS Orbitrap
training reference set, build train/validation datasets, and train
one or more of the supported models with the specified loss.
"""

import sys

sys.path.append("..")

import argparse
from pathlib import Path
from typing import Union, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from SpecRoFormer.trainer.trainer import Trainer, set_seed
from SpecRoFormer.trainer.fn import step_train, step_evaluate
from SpecRoFormer.data.datasets import TrainDataset
from SpecRoFormer.data.tokenizer import Tokenizer
from SpecRoFormer.utils.clean import get_classified_tokenset
from SpecRoFormer.models import (
    SpecRoFormer_RoPE,
    SpecRoFormer_LinearMZ,
    SpecRoFormer,
)
from SpecRoFormer.loss import (
    TanimotoScoreLoss,
    SupConLoss,
    SupConLossWithTanimotoScore,
    StructureAwareSupConWithTanimoto,
)
from SpecRoFormer.const import gnps
from SpecRoFormer.type import (
    AugmentationConfig,
    StorageConfig,
    StepFuncConfig,
    DescriptionConfig,
    TokenizerConfig,
    DataLoaderConfig,
    SchedulerConfig,
    TrainerConfig,
    OptimizerConfig,
    SupConLossConfig,
    TanimotoLossConfig,
    SupConLossWithTanimotoScoreConfig,
)


# ---------------------------------------------------------------------
# Loss curve plotting
# ---------------------------------------------------------------------

def plot_loss_from_npy(
    npy_path: Union[str, Path],
    save_dir: Union[str, Path],
    save_name: str,
    *,
    xlabel: Optional[str] = None,
    ylabel: str = "Loss",
    figsize=(8, 5),
    dpi=300,
):
    """
    Read a loss array from ``.npy`` and save a line plot.

    Supports two formats:

    1. Epoch loss: a dict mapping ``str -> list / ndarray``.
    2. Step loss : a 1D ndarray.

    Parameters
    ----------
    npy_path : str or Path
        Path to the ``.npy`` file containing the loss.
    save_dir : str or Path
        Directory where the figure is saved.
    save_name : str
        Output file name (``.png`` appended if missing).
    xlabel : str, optional
        X-axis label. Defaults to ``"Epoch"`` or ``"Step"`` depending
        on the detected format.
    ylabel : str
        Y-axis label.
    figsize : tuple
        Matplotlib figure size.
    dpi : int
        Output figure DPI.

    Returns
    -------
    save_path : Path
        Path to the saved figure.
    """
    npy_path = Path(npy_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not save_name.endswith(".png"):
        save_name = save_name + ".png"

    # Load data.
    loss = np.load(npy_path, allow_pickle=True)

    # Detect the epoch-loss dict format.
    if (
        isinstance(loss, np.ndarray)
        and loss.shape == ()
        and isinstance(loss.item(), dict)
    ):
        loss = loss.item()
        is_dict = True
    else:
        is_dict = False

    plt.figure(figsize=figsize)

    if is_dict:
        # Epoch loss: multiple curves.
        for name, values in loss.items():
            values = np.asarray(values)
            sns.lineplot(
                x=range(len(values)),
                y=values,
                label=name,
            )
        plt.xlabel(xlabel or "Epoch")
        plt.legend()
    else:
        # Step loss: single curve.
        loss = np.asarray(loss).reshape(-1)
        sns.lineplot(
            x=range(len(loss)),
            y=loss,
        )
        plt.xlabel(xlabel or "Step")

    plt.ylabel(ylabel)
    plt.tight_layout()

    save_path = save_dir / save_name
    plt.savefig(save_path, dpi=dpi)
    plt.close()

    return save_path


# ---------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="SpecRoFormer experiment config"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--is_augment",
        action="store_true",
        help="Whether to use data augmentation",
    )

    parser.add_argument(
        "--is_augment_plus",
        action="store_true",
        help="Whether to use advanced data augmentation",
    )

    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["SpecRoFormer_RoPE"],
        help="Method names",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size",
    )

    parser.add_argument(
        "--loss_type",
        type=str,
        default="SupConWithTanimotoLoss",
        choices=[
            "TanimotoLoss",
            "SupConLoss",
            "SupConWithTanimotoLoss",
            "StructureAwareSupConWithTanimoto",
        ],
        help="Type of loss function to use",
    )

    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help=(
            "Path to pretrained model weights (.pth). "
            "If not set, train from scratch."
        ),
    )

    args = parser.parse_args()
    print("Parsed arguments:")
    print("-" * 40)
    for k, v in vars(args).items():
        print(f"{k:>15} | {v}")
    print("-" * 40)

    return parser.parse_args()


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train(args):
    # -----------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------
    labels_dir = Path(
        "../data/labels"
    )
    labels_dir.mkdir(parents=True, exist_ok=True)

    train_spectra = np.load(
        gnps.ORBITRAP_TRAIN_REF, allow_pickle=True
    )
    unique_smiles = np.load(
        gnps.DIR / gnps.UNIQUE_SMILES, allow_pickle=True
    )

    print("Start loading the data ...")

    # Tokenization.
    tokenizer_config = TokenizerConfig(
        max_len=100,
        show_progress_bar=True,
    )
    tokenizer = Tokenizer(**tokenizer_config)
    sequences = tokenizer.tokenize_sequence(train_spectra)
    data, labels = get_classified_tokenset(unique_smiles, sequences)

    np.save(labels_dir / "data.npy", data)
    np.save(labels_dir / "labels.npy", labels)
    print("finish the tokenize process")

    data = np.load(
        labels_dir / "data.npy", allow_pickle=True
    ).item()
    labels = np.load(labels_dir / "labels.npy", allow_pickle=True)
    print("The data has been loaded successfully")

    # -----------------------------------------------------------------
    # Training parameters
    # -----------------------------------------------------------------
    seed = set_seed(args.seed)
    is_augment = args.is_augment

    if args.is_augment_plus:
        print("Using advanced augmentation strategy (aug_plus).")
        augment_config = AugmentationConfig(
            # Whether to augment this view (Dataset level).
            prob=0.5,

            # Sub-spectrum consistency (core).
            subspec_prob=0.3,
            subspec_drop_ratio=0.35,

            # Low-intensity peak removal (auxiliary).
            low_intensity_prob=0.4,
            removal_max=0.2,
            removal_intensity=0.1,

            # Intensity perturbation.
            rate_intensity=0.1,
        )
    else:
        print("Using basic augmentation strategy (aug).")
        augment_config = AugmentationConfig(
            # Whether to augment this view (Dataset level).
            prob=0.5,

            # Low-intensity peak removal (auxiliary).
            low_intensity_prob=0.4,
            removal_max=0.2,
            removal_intensity=0.1,

            # Intensity perturbation.
            rate_intensity=0.1,
        )

    # -----------------------------------------------------------------
    # Dataset and dataloader
    # -----------------------------------------------------------------
    dataloader_config = DataLoaderConfig(
        batch_size=args.batch_size,
        val_ratio=0.2,
    )

    length = len(labels)
    val_indices = np.random.choice(
        np.arange(length),
        int(dataloader_config["val_ratio"] * length),
        replace=False,
    )
    val_keys = labels[val_indices]
    train_keys = np.delete(labels, val_indices)

    train_dataset = TrainDataset(
        data=data,
        keys=train_keys,
        n_views=2,
        is_augment=is_augment,
        augment_config=augment_config,
    )

    val_dataset = TrainDataset(
        data=data,
        keys=val_keys,
        n_views=2,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=dataloader_config["batch_size"],
        shuffle=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=dataloader_config["batch_size"],
        shuffle=True,
    )

    # -----------------------------------------------------------------
    # Loop over methods
    # -----------------------------------------------------------------
    for model_name in args.methods:
        loss_type = args.loss_type
        model_backbone = "transformer"

        # Model.
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        if model_name == "SpecRoFormer_RoPE":
            model = SpecRoFormer_RoPE(
                512, 16, 4, 512, 512, feedward_activation="selu"
            )
        elif model_name == "SpecRoFormer_LinearMZ":
            model = SpecRoFormer_LinearMZ(
                512, 16, 4, 512, 512, feedward_activation="selu"
            )
        elif model_name == "SpecRoFormer":
            model = SpecRoFormer(
                512, 16, 4, 512, 512, feedward_activation="selu"
            )
        else:
            raise ValueError(f"Unknown model name {model_name}")

        if args.pretrained is not None:
            print(
                f"Loading pretrained weights from {args.pretrained}"
            )
            model_state = torch.load(
                args.pretrained, map_location=device
            )
            model.load_state_dict(model_state)
        else:
            print("Training from scratch (no pretrained weights)")

        model = model.to(device)
        print(model)

        # -----------------------------------------------------------------
        # Trainer
        # -----------------------------------------------------------------
        trainer_config = TrainerConfig(
            n_epoch=300,
            device=device,
            early_stop=30,
            show_progress_bar=True,
        )

        # Scheduler.
        n_step = trainer_config["n_epoch"] * len(train_dataloader)
        scheduler_config = SchedulerConfig(
            warmup_steps=int(n_step * 0.1),
            total_steps=n_step,
            only_warmup=True,
        )

        # -----------------------------------------------------------------
        # Loss
        # -----------------------------------------------------------------
        reduction = "mean"
        tanimoto_score_path = (
            "../data/"
            "MSBert/GNPS/tanimoto_score.npy"
        )

        if loss_type == "SupConLoss":
            loss_config = SupConLossConfig(
                device=device,
                temperature=0.05,
                base_temperature=0.01,
                contrast_mode="all",
                reduction=reduction,
            )
            criterion = SupConLoss(**loss_config)

        elif loss_type == "TanimotoLoss":
            loss_config = TanimotoLossConfig(
                score_path=tanimoto_score_path,
                device=device,
                reduction=reduction,
            )
            criterion = TanimotoScoreLoss(**loss_config)

        elif loss_type == "SupConWithTanimotoLoss":
            loss_config = SupConLossWithTanimotoScoreConfig(
                alpha=30,
                score_path=tanimoto_score_path,
                device=device,
                temperature=0.05,
                contrast_mode="all",
                base_temperature=0.05,
                reduction=reduction,
            )
            criterion = SupConLossWithTanimotoScore(**loss_config)

        elif loss_type == "StructureAwareSupConWithTanimoto":
            loss_config = SupConLossWithTanimotoScoreConfig(
                alpha=30,
                score_path=tanimoto_score_path,
                device=device,
                temperature=0.05,
                contrast_mode="all",
                base_temperature=0.05,
                reduction=reduction,
            )
            criterion = StructureAwareSupConWithTanimoto(
                **loss_config
            )

        else:
            raise ValueError(f"Unknown loss type {loss_type}")

        # -----------------------------------------------------------------
        # Optimizer
        # -----------------------------------------------------------------
        if args.pretrained is not None:
            print(
                "Fine-tuning: setting a lower learning rate "
                "for optimizer."
            )
            lr = 2e-5
        else:
            lr = 1e-4

        optimizer_config = OptimizerConfig(
            lr=lr,
            weight_decay=0.1,
        )
        optimizer = AdamW(model.parameters(), **optimizer_config)

        # -----------------------------------------------------------------
        # Description and step functions
        # -----------------------------------------------------------------
        desc_config = DescriptionConfig(
            train="train, epoch={}, loss={:.4f}",
            val="validation, epoch={}, loss={:.4f}, save the model",
            end="model train end, best model loss={:.4f}",
        )

        stepfunc_config = StepFuncConfig(
            train=step_train,
            val=step_evaluate,
        )

        # -----------------------------------------------------------------
        # Storage
        # -----------------------------------------------------------------
        suffix = ""
        if is_augment:
            if args.is_augment_plus:
                suffix = "-AugmentationPlus"
            else:
                suffix = "-Augmentation"

        if args.pretrained is not None:
            model_dir = Path(
                f"../data/model/"
                f"{model_name}-{model_backbone}-{loss_type}"
                f"{suffix}-bs{dataloader_config['batch_size']}-finetune"
            )
        else:
            model_dir = Path(
                f"../data/model/"
                f"{model_name}-{model_backbone}-{loss_type}"
                f"{suffix}-bs{dataloader_config['batch_size']}"
            )

        model_dir.mkdir(parents=True, exist_ok=True)

        storage_config = StorageConfig(
            model=model_dir / "model.ckpt",
            lr=model_dir / "lr.npy",
            step_loss=model_dir / "step_loss.npy",
            loss=model_dir / "epoch_loss.npy",
            custom=model_dir / "custom_metric.npy",
        )

        # -----------------------------------------------------------------
        # Train
        # -----------------------------------------------------------------
        torch.cuda.empty_cache()

        trainer = Trainer(
            model,
            reduction,
            train_dataloader,
            val_dataloader,
            optimizer,
            criterion,
            device,
            stepfunc_config,
            desc_config,
            storage_config,
            trainer_config,
            scheduler_config,
        )

        trainer.train()

        # Plot the training loss curves.
        plot_loss_from_npy(
            npy_path=model_dir / "epoch_loss.npy",
            save_dir=model_dir,
            save_name="epoch_loss",
        )

        plot_loss_from_npy(
            npy_path=model_dir / "step_loss.npy",
            save_dir=model_dir,
            save_name="step_loss",
        )


if __name__ == "__main__":
    args = parse_args()
    train(args)