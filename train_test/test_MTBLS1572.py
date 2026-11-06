"""
Test the MTBLS1572 dataset.
"""

import sys

sys.path.append("..")

import argparse

import numpy as np
import pandas as pd
import torch

from SpecRoFormer.trainer.trainer import ModelTester
from SpecRoFormer.data.tokenizer import Tokenizer
from SpecRoFormer.utils.model import (
    search_with_spectra,
    load_model,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SpecRoFormer experiment config"
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

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
        default=["SpecRoFormer"],
        help="Method names",
    )

    parser.add_argument(
        "--batch_size", type=int, default=512, help="Batch size"
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
        "--is_finetune",
        action="store_true",
        help="Whether the model was fine-tuned",
    )

    args = parser.parse_args()
    print("MTBLS1572 dataset - Parsed arguments:")
    print("-" * 40)
    for k, v in vars(args).items():
        print(f"{k:>15} | {v}")
    print("-" * 40)

    return parser.parse_args()


def test(args):
    show_progress_bar = False
    is_augment = args.is_augment
    model_backbone = "transformer"
    loss_type = args.loss_type
    k_metric = [5, 1, 10]
    loader_batch_size = (
        512
        if torch.cuda.mem_get_info(0)[0] < 20 * 1024**3
        else 1024
    )
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    tokenizer = Tokenizer(100, show_progress_bar)
    train_batch = args.batch_size

    query_spectra = np.load(
        "../data/MSBert/"
        "MTBLS1572/query.npy",
        allow_pickle=True,
    )
    ref_spectra = np.load(
        "../data/MSBert/"
        "MTBLS1572/ref.npy",
        allow_pickle=True,
    )

    for model_name in args.methods:
        model = load_model(
            device,
            model_name,
            model_backbone,
            train_batch,
            loss_type,
            is_augment,
            args.is_finetune,
            args.is_augment_plus,
        )

        tester = ModelTester(model, device, show_progress_bar)

        df = search_with_spectra(
            "MTBLS1572",
            tester,
            k_metric,
            tokenizer,
            query_spectra,
            ref_spectra,
            loader_batch_size,
            show_progress_bar,
            train_batch,
            use_gpu_metric=True,
            calculate_avg_similarity=True,
        )
        print(df)

        txt_dir = (
            f"../data/model/{model_name}-{model_backbone}-{loss_type}"
        )
        if is_augment:
            if args.is_augment_plus:
                txt_dir += "-AugmentationPlus"
            else:
                txt_dir += "-Augmentation"
        txt_dir += f"-bs{train_batch}"
        if args.is_finetune:
            txt_dir += "-finetune"

        save_path = txt_dir + "/MTBLS1572.txt"
        df_percent = df * 100
        df_percent.to_csv(
            save_path,
            sep="\t",
            index=False,
        )


if __name__ == "__main__":
    args = parse_args()
    test(args)