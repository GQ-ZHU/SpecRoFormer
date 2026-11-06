"""
Test the GNPS dataset with 10 query/reference replicas.
"""

import sys

sys.path.append("..")

import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from SpecRoFormer.trainer.trainer import ModelTester
from SpecRoFormer.const import gnps
from SpecRoFormer.data.tokenizer import Tokenizer
from SpecRoFormer.utils.model import (
    search_with_spectra,
    search,
    load_model,
)
from train_test.write_to_txt import save_mean_std_txt


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
        default=["SpecRoFormer"],
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
        "--is_finetune",
        action="store_true",
        help="Whether the model was fine-tuned",
    )

    args = parser.parse_args()
    print("GNPS dataset - Parsed arguments:")
    print("-" * 40)
    for k, v in vars(args).items():
        print(f"{k:>15} | {v}")
    print("-" * 40)

    return parser.parse_args()


def test(args):
    spectra_paths = {
        "gnps": {
            "orbitrap": {
                "train": (
                    gnps.ORBITRAP_TRAIN_QUERY,
                    gnps.ORBITRAP_TEST_REF,
                ),
                "test": (
                    gnps.ORBITRAP_TEST_QUERY,
                    gnps.ORBITRAP_TEST_REF,
                ),
            },
            "qtof": {
                "test": (gnps.QTOF_TEST_QUERY, gnps.QTOF_TEST_REF),
            },
            "other": {
                "test": (gnps.OTHER_TEST_QUERY, gnps.OTHER_TEST_REF),
            },
        }
    }
    gnps_train_ref = np.load(
        gnps.ORBITRAP_TRAIN_REF, allow_pickle=True
    )

    show_progress_bar = True
    is_augment = args.is_augment
    model_backbone = "transformer"
    loss_type = args.loss_type
    replica_suffix = "-replication-{}"
    k_metric = [1, 5, 10]
    top_peaks = 100
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    if top_peaks == 100:
        loader_batch_size = (
            2800
            if torch.cuda.mem_get_info(0)[0] < 20 * 1024**3
            else 5600
        )
    else:
        loader_batch_size = (
            500
            if torch.cuda.mem_get_info(0)[0] < 20 * 1024**3
            else 1000
        )

    tokenizer = Tokenizer(top_peaks, show_progress_bar)
    train_batch = args.batch_size

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
        replica_df_seq = []

        for i in tqdm(range(10)):
            df_seq = []
            for db, db_metadata in spectra_paths.items():
                for desc, path_metadata in db_metadata.items():
                    for info, paths in path_metadata.items():
                        print(
                            "-" * 40,
                            f"{db}-{desc}-{info}",
                            "-" * 40,
                        )
                        query_path, ref_path = paths
                        query_path = query_path.with_stem(
                            query_path.stem
                            + replica_suffix.format(i + 1)
                        )
                        ref_path = ref_path.with_stem(
                            ref_path.stem
                            + replica_suffix.format(i + 1)
                        )

                        if db == "gnps" and desc == "orbitrap":
                            if info == "train":
                                query_path = gnps.ORBITRAP_TRAIN_QUERY

                            ref_spectra = np.load(
                                ref_path, allow_pickle=True
                            )
                            query_spectra = np.load(
                                query_path, allow_pickle=True
                            )
                            ref_spectra = np.hstack(
                                (gnps_train_ref, ref_spectra)
                            )
                            df = search_with_spectra(
                                f"{db}-{desc}-{info}",
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
                        else:
                            df = search(
                                f"{db}-{desc}-{info}",
                                tester,
                                k_metric,
                                tokenizer,
                                query_path,
                                ref_path,
                                loader_batch_size,
                                show_progress_bar,
                                512,
                                use_gpu_metric=True,
                                calculate_avg_similarity=True,
                            )
                        df_seq.append(df)
            df = pd.concat(df_seq, axis=0)
            print(df)
            replica_df_seq.append(df)

        data = [item.values for item in replica_df_seq]

        # Compute mean and std over the 10 replicas.
        data = np.array(data)
        mean = np.mean(data, axis=0) * 100
        std = np.std(data, axis=0) * 100

        print("Mean:\n", mean, mean.shape)
        print("Std:\n", std, std.shape)

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

        save_path = save_mean_std_txt(
            mean=mean,
            std=std,
            datasets=[
                "gnps-orbitrap-train",
                "gnps-orbitrap-test",
                "gnps-qtof-test",
                "gnps-other-test",
            ],
            metrics=replica_df_seq[0].columns.tolist(),
            save_dir=txt_dir,
            file_name="GNPS.txt",
            precision=2,
        )

        print(f"Saved to {save_path}")


if __name__ == "__main__":
    args = parse_args()
    test(args)