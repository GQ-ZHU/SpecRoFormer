"""
Utilities for writing evaluation results to text files.
"""

import os
from typing import List

import numpy as np


def save_mean_std_txt(
    mean: np.ndarray,
    std: np.ndarray,
    datasets: List[str],
    metrics: List[str],
    save_dir: str,
    file_name: str,
    precision: int = 2,
):
    """
    Save mean and std results to a txt file in table format.

    Parameters
    ----------
    mean : np.ndarray
        Shape ``(N_datasets, N_metrics)``.
    std : np.ndarray
        Shape ``(N_datasets, N_metrics)``.
    datasets : List[str]
        Dataset names, one per row.
    metrics : List[str]
        Metric names, one per column.
    save_dir : str
        Directory where the txt file is written.
    file_name : str
        Name of the output txt file.
    precision : int
        Decimal precision for mean / std values.

    Returns
    -------
    save_path : str
        Full path to the saved txt file.
    """
    # Safety checks.
    assert mean.shape == std.shape, "mean and std must have same shape"
    assert mean.shape[0] == len(datasets), "datasets number mismatch"
    assert mean.shape[1] == len(metrics), "metrics number mismatch"

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)

    fmt = f"{{:.{precision}f}}"

    with open(save_path, "w") as f:
        # Header.
        header = "Dataset".ljust(30)
        for metric in metrics:
            header += f"{metric} (mean±std)".ljust(25)
        f.write(header + "\n")

        # Content.
        for i, dataset in enumerate(datasets):
            line = dataset.ljust(30)
            for j in range(len(metrics)):
                value = (
                    f"{fmt.format(mean[i, j])}±"
                    f"{fmt.format(std[i, j])}"
                )
                line += value.ljust(25)
            f.write(line + "\n")

    return save_path