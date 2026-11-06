"""
Hyperparameter search with Optuna.

Search over model size, optimizer, loss, and augmentation settings
for the SpecRoFormer family of models, and save the trial table plus
param importance to a single TSV file.
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Sequence, Optional, Literal

import json

import numpy as np
import numpy.typing as npt
import optuna
import optuna.importance
import pandas as pd
import torch
from matchms import Spectrum
from optuna.trial import Trial
from torch.optim import AdamW
from torch.utils.data import DataLoader

from SpecRoFormer.models import (
    SpecRoFormer_RoPE,
    SpecRoFormer_LinearMZ,
    SpecRoFormer,
)
from SpecRoFormer.trainer.trainer import Trainer, ModelTester, set_seed
from SpecRoFormer.trainer.fn import step_train, step_evaluate
from SpecRoFormer.type import (
    AugmentationConfig,
    OptimizerConfig,
    TokenizerConfig,
    DataLoaderConfig,
    StepFuncConfig,
    SchedulerConfig,
    TrainerConfig,
    SupConLossConfig,
    SupConLossWithTanimotoScoreConfig,
    TanimotoLossConfig,
    DescriptionConfig,
    StorageConfig,
)
from SpecRoFormer.data.datasets import (
    TrainDataset,
    TestDataset,
    TokenSequence,
)
from SpecRoFormer.data.tokenizer import Tokenizer
from SpecRoFormer.utils.clean import get_classified_tokenset
from SpecRoFormer.loss import (
    SupConLossWithTanimotoScore,
    SupConLoss,
    TanimotoScoreLoss,
)
from SpecRoFormer.utils.model import embedding, metric
from SpecRoFormer.const import gnps


# ---------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------

def build_model(
    model_name,
    n_head,
    n_layer,
    dim_target,
    feedward_activation,
):
    """Instantiate one of the supported models by name."""
    if model_name == "SpecRoFormer_RoPE":
        model = SpecRoFormer_RoPE(
            512,
            n_head,
            n_layer,
            512,
            dim_target,
            feedward_activation=feedward_activation,
        )
    elif model_name == "SpecRoFormer_LinearMZ":
        model = SpecRoFormer_LinearMZ(
            512,
            n_head,
            n_layer,
            512,
            dim_target,
            feedward_activation=feedward_activation,
        )
    elif model_name == "SpecRoFormer":
        model = SpecRoFormer(
            512,
            n_head,
            n_layer,
            512,
            dim_target,
            feedward_activation=feedward_activation,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model


# ---------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------

def objective(
    trial: Trial,
    seed: int,
    device: torch.device,
    model_backbone: Literal["transformer", "mamba"],
    model_name: str,
    loss_type: Literal[
        "SupConLoss", "TanimotoLoss", "SupConWithTanimotoLoss"
    ],
    data: dict,
    labels: npt.NDArray,
    metric_fn,
    show_progress_bar: Optional[bool] = True,
    augment_mode: Literal["none", "aug", "aug_plus"] = "none",
):
    """Optuna objective: train one configuration and return top-1."""
    torch.cuda.empty_cache()

    batch_size = trial.suggest_categorical("batch_size", [512])
    n_layer = trial.suggest_categorical(
        "n_layer", [2, 3, 4, 5, 6, 7, 8]
    )
    n_epoch = 60
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    dim_target = trial.suggest_categorical(
        "dim_target", [128, 256, 512]
    )
    weight_decay = trial.suggest_categorical(
        "weight_decay", [0.1, 0.05, 0.01]
    )
    feedward_activation = trial.suggest_categorical(
        "feedward_activation", ["gelu", "selu", "relu"]
    )

    desc = (
        f"model: {model_name}, "
        f"model_backbone: {model_backbone}, "
        f"loss_type: {loss_type}, "
        f"batch_size: {batch_size}, "
        f"n_layer: {n_layer}, "
        f"n_epoch: {n_epoch}, "
        f"lr: {lr}, "
        f"dim_target: {dim_target}, "
        f"weight_decay: {weight_decay}, "
        f"feedward_activation: {feedward_activation}"
    )

    # -------------------------------------------------------------
    # Augmentation
    # -------------------------------------------------------------
    augment_config = None
    is_augment = True

    if augment_mode == "aug":
        low_intensity_prob = trial.suggest_categorical(
            "low_intensity_prob", [0.6, 0.8, 1.0]
        )
        removal_max = trial.suggest_categorical(
            "removal_max", [0.1, 0.15, 0.2, 0.25]
        )
        removal_intensity = trial.suggest_categorical(
            "removal_intensity", [0.1, 0.2, 0.3]
        )
        rate_intensity = trial.suggest_categorical(
            "rate_intensity", [0.1, 0.15, 0.2]
        )

        augment_config = AugmentationConfig(
            prob=0.5,
            low_intensity_prob=low_intensity_prob,
            removal_max=removal_max,
            removal_intensity=removal_intensity,
            rate_intensity=rate_intensity,
        )

        desc += (
            f", aug=basic"
            f", low_intensity_prob={low_intensity_prob}"
            f", removal_max={removal_max}"
            f", removal_intensity={removal_intensity}"
            f", rate_intensity={rate_intensity}"
        )

    elif augment_mode == "aug_plus":
        subspec_prob = trial.suggest_float(
            "subspec_prob", 0.3, 0.8, step=0.1
        )
        subspec_drop_ratio = trial.suggest_float(
            "subspec_drop_ratio", 0.2, 0.5, step=0.05
        )
        low_intensity_prob = trial.suggest_float(
            "low_intensity_prob", 0.2, 0.4, step=0.05
        )
        removal_max = trial.suggest_float(
            "removal_max", 0.1, 0.3, step=0.05
        )
        removal_intensity = trial.suggest_float(
            "removal_intensity", 0.1, 0.3, step=0.05
        )
        rate_intensity = trial.suggest_float(
            "rate_intensity", 0.05, 0.2, step=0.05
        )

        augment_config = AugmentationConfig(
            prob=0.5,
            subspec_prob=subspec_prob,
            subspec_drop_ratio=subspec_drop_ratio,
            low_intensity_prob=low_intensity_prob,
            removal_max=removal_max,
            removal_intensity=removal_intensity,
            rate_intensity=rate_intensity,
        )

        desc += (
            f", aug=plus"
            f", subspec_prob={subspec_prob}"
            f", subspec_drop_ratio={subspec_drop_ratio}"
            f", low_intensity_prob={low_intensity_prob}"
            f", removal_max={removal_max}"
            f", removal_intensity={removal_intensity}"
            f", rate_intensity={rate_intensity}"
        )
    else:
        is_augment = False

    # -------------------------------------------------------------
    # Dataset and dataloader
    # -------------------------------------------------------------
    set_seed(seed)

    dataloader_config = DataLoaderConfig(
        batch_size=batch_size,
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
        shuffle=False,
    )

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------
    if model_backbone == "transformer":
        n_head = trial.suggest_categorical("n_head", [4, 8, 16])
        desc += f", n_head: {n_head}"
        model = build_model(
            model_name,
            n_head,
            n_layer,
            dim_target,
            feedward_activation,
        )
    else:
        raise ValueError(f"No such model {model_backbone}")

    model = model.to(device)

    # -------------------------------------------------------------
    # Trainer / scheduler
    # -------------------------------------------------------------
    trainer_config = TrainerConfig(
        n_epoch=n_epoch,
        device=device,
        early_stop=20,
        show_progress_bar=show_progress_bar,
    )

    n_step = trainer_config["n_epoch"] * len(train_dataloader)
    scheduler_config = SchedulerConfig(
        warmup_steps=int(n_step * 0.1),
        total_steps=n_step,
        only_warmup=True,
    )

    # -------------------------------------------------------------
    # Loss
    # -------------------------------------------------------------
    reduction = "mean"
    tanimoto_score_path = gnps.DIR / gnps.TANOMOTO_SCORE

    if loss_type == "TanimotoLoss":
        loss_config = TanimotoLossConfig(
            score_path=tanimoto_score_path,
            device=device,
            reduction=reduction,
        )
        criterion = TanimotoScoreLoss(**loss_config)
    else:
        temperature = trial.suggest_categorical(
            "temperature",
            [0.5, 0.25, 0.1, 0.07, 0.05, 0.01, 0.005],
        )
        base_temperature = trial.suggest_categorical(
            "base_temperature",
            [0.5, 0.25, 0.1, 0.07, 0.05, 0.01, 0.005],
        )

        desc += (
            f", temperature: {temperature}"
            f", base_temperature: {base_temperature}"
        )

        if loss_type == "SupConLoss":
            loss_config = SupConLossConfig(
                device=device,
                temperature=temperature,
                base_temperature=base_temperature,
                contrast_mode="all",
                reduction=reduction,
            )
            criterion = SupConLoss(**loss_config)
        else:
            alpha = trial.suggest_categorical(
                "alpha", [20, 30, 40, 50]
            )
            desc += f", alpha: {alpha}"

            loss_config = SupConLossWithTanimotoScoreConfig(
                alpha=alpha,
                score_path=tanimoto_score_path,
                device=device,
                temperature=temperature,
                contrast_mode="all",
                base_temperature=base_temperature,
                reduction=reduction,
            )
            criterion = SupConLossWithTanimotoScore(**loss_config)

    # -------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------
    optimizer_config = OptimizerConfig(
        lr=lr,
        weight_decay=weight_decay,
    )
    optimizer = AdamW(model.parameters(), **optimizer_config)

    # -------------------------------------------------------------
    # Description / step fn / storage
    # -------------------------------------------------------------
    desc_config = DescriptionConfig(
        train="train, epoch {}, loss={:.4f}",
        val="validation, epoch {}, loss={:.4f}",
        end="model train end, best model loss={:.4f}",
    )

    stepfunc_config = StepFuncConfig(
        train=step_train,
        val=step_evaluate,
    )

    suffix = ""
    if augment_mode == "aug_plus":
        suffix = "-AugmentationPlus"
    elif augment_mode == "aug":
        suffix = "-Augmentation"

    model_dir = Path(
        f"./data/hparam_search/model/"
        f"{model_name}-{model_backbone}-{loss_type}{suffix}"
    )
    model_dir.mkdir(parents=True, exist_ok=True)

    storage_config = StorageConfig(
        model=model_dir / "model.ckpt",
        lr=model_dir / "lr.npy",
        step_loss=model_dir / "step_loss.npy",
        loss=model_dir / "epoch_loss.npy",
        custom=model_dir / "custom_metric.npy",
    )

    # -------------------------------------------------------------
    # Train
    # -------------------------------------------------------------
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

    # -------------------------------------------------------------
    # Evaluate best checkpoint
    # -------------------------------------------------------------
    model = build_model(
        model_name,
        n_head,
        n_layer,
        dim_target,
        feedward_activation,
    )
    model_state = torch.load(storage_config["model"])
    model.load_state_dict(model_state)
    model = model.to(device)

    print(desc)
    print("calculate the metric")
    metric_value = metric_fn(model)
    print(metric_value)
    return metric_value


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":

    def metric_fn(
        model,
        device: torch.device,
        tokenizer: Tokenizer,
        query_spectra: Sequence[Spectrum],
        ref_spectra: Sequence[Spectrum],
        show_progress_bar: Optional[bool] = True,
    ):
        """Compute the top-1 retrieval metric on the given split."""
        tester = ModelTester(model, device, show_progress_bar)
        query_embedding, query_smiles, _ = embedding(
            tester, tokenizer, 512, query_spectra, show_progress_bar
        )
        ref_embedding, ref_smiles, _ = embedding(
            tester, tokenizer, 512, ref_spectra, show_progress_bar
        )
        _, values, _ = metric(
            [1, 5, 10],
            query_embedding,
            ref_embedding,
            query_smiles,
            ref_smiles,
            show_progress_bar,
        )
        return values[0]

    candidate_loss_fn = [
        "SupConLoss",
        "TanimotoLoss",
        "SupConWithTanimotoLoss",
    ]
    candidate_backbones = ["transformer", "mamba"]

    parser = ArgumentParser()
    parser.add_argument(
        "--loss",
        type=str,
        help="loss type",
        default="SupConLoss",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed",
        default=42,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="model",
        default="SpecRoFormer_RoPE",
    )
    parser.add_argument(
        "--model_backbone",
        type=str,
        help="model",
        default="transformer",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="if show progress",
    )
    parser.add_argument(
        "--augment_mode",
        type=str,
        default="none",
        choices=["none", "aug", "aug_plus"],
        help="augmentation mode",
    )

    args = parser.parse_args()

    if args.loss not in candidate_loss_fn:
        raise ValueError("input loss type not in candidate loss functions")
    if args.seed < 0:
        raise ValueError("input random seed must be non negative number")
    if args.model_backbone not in candidate_backbones:
        raise ValueError(
            "input model backbone must be transformer or mamba"
        )

    seed = args.seed
    show_progress_bar = args.progress
    augment_mode = args.augment_mode

    query_path = gnps.ORBITRAP_TRAIN_QUERY
    ref_path = gnps.ORBITRAP_TRAIN_REF
    query_spectra = np.load(query_path, allow_pickle=True)
    ref_spectra = np.load(ref_path, allow_pickle=True)
    unique_smiles = np.load(
        gnps.DIR / gnps.UNIQUE_SMILES, allow_pickle=True
    )
    print("read raw spectra success")

    tokenizer_config = TokenizerConfig(
        max_len=100,
        show_progress_bar=show_progress_bar,
    )
    tokenizer = Tokenizer(**tokenizer_config)
    query_sequences = tokenizer.tokenize_sequence(query_spectra)
    ref_sequences = tokenizer.tokenize_sequence(ref_spectra)
    print("tokenize the query and reference data success")

    data, labels = get_classified_tokenset(
        unique_smiles,
        ref_sequences,
        show_progress_bar,
    )
    query_dataset = TestDataset(query_sequences)
    ref_dataset = TestDataset(ref_sequences)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    study_name = (
        f"{args.model_name}-{args.model_backbone}-{args.loss}"
        f"-aug{augment_mode}-seed{seed}"
    )
    study = optuna.create_study(
        study_name=study_name, direction="maximize"
    )
    study.optimize(
        lambda trial: objective(
            trial,
            seed,
            device,
            args.model_backbone,
            args.model_name,
            args.loss,
            data,
            labels,
            lambda model: metric_fn(
                model,
                device,
                tokenizer,
                query_spectra,
                ref_spectra,
                show_progress_bar,
            ),
            show_progress_bar,
            augment_mode,
        ),
        n_trials=100,
        show_progress_bar=show_progress_bar,
    )

    params = study.best_params
    print(params)

    df: pd.DataFrame = study.trials_dataframe(
        attrs=("number", "value", "params", "state")
    )
    print(df)

    importance = optuna.importance.get_param_importances(study)
    for param, importance_value in importance.items():
        print(
            f"Parameter: {param}, "
            f"Importance: {importance_value:.4f}"
        )

    # Save all results to a single file.
    SAVE_DIR = Path("../data/hparam_search")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    fname = (
        f"hparam_{args.model_name}"
        f"_{args.model_backbone}"
        f"_{args.loss}"
        f"_aug{args.augment_mode}"
        f"_seed{args.seed}.tsv"
    )

    best_params = study.best_params
    best_value = study.best_value
    importance = optuna.importance.get_param_importances(study)

    fpath = SAVE_DIR / fname
    with open(fpath, "w") as f:
        f.write(f"# model: {args.model_name}\n")
        f.write(f"# model_backbone: {args.model_backbone}\n")
        f.write(f"# loss: {args.loss}\n")
        f.write(f"# augment_mode: {args.augment_mode}\n")
        f.write(f"# seed: {args.seed}\n")
        f.write(f"# n_trials: {len(study.trials)}\n")
        f.write(f"# best_value: {best_value:.6f}\n")
        f.write(f"# best_params: {json.dumps(best_params)}\n")
        f.write(
            f"# param_importance: {json.dumps(importance)}\n"
        )
        f.write("\n")
        df.to_csv(f, sep="\t", index=False)

    print(f"[Saved] Hyperparameter search results -> {fname}")