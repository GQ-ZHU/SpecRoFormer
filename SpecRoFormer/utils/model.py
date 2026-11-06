"""
Model loading and retrieval utilities.

Provide:
    - ``embedding``  : tokenize spectra and run inference.
    - ``metric``     : CPU retrieval metric (top-k hit rate).
    - ``metric_GPU`` : GPU retrieval metric with optional structural
      similarity calculation.
    - ``load_model`` : construct and load one of the supported models.
    - ``search`` / ``search_with_spectra`` : high-level retrieval
      pipelines.
    - ``most_similar`` / ``split_attention_by_hit`` : small helpers.
"""

from typing import Optional, Literal
from pathlib import Path
from collections.abc import Sequence

import os
import shutil

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from numba import njit, prange
from matchms import Spectrum
from torch.utils.data import DataLoader

from SpecRoFormer.data.datasets import TestDataset
from SpecRoFormer.data.tokenizer import Tokenizer
from SpecRoFormer.trainer.trainer import ModelTester
from SpecRoFormer.models import *
from SpecRoFormer.utils.clean import get_smiles
from SpecRoFormer.utils.similarity_matrix import (
    build_similarity_matrix_from_smiles,
)


# ---------------------------------------------------------------------
# Embedding pipeline
# ---------------------------------------------------------------------

def embedding(
    tester: ModelTester,
    tokenizer: Tokenizer,
    batch_size: int,
    spectra: Sequence[Spectrum],
    show_progress_bar: bool = True,
    collect_attention_profile: bool = False,
    identifier_list: Optional[Sequence[str]] = None,
    target_indices: Optional[Sequence[int]] = None,
    save_last_layer_only: bool = True,
    profile_config: Optional[dict] = None,
    sample_metadata: Optional[dict[int, dict]] = None,
    save_attention: bool = False,
    attention_save_path: Optional[str] = None,
):
    """
    Tokenize spectra, build dataloader, and run inference through ModelTester.

    Parameters
    ----------
    tester : ModelTester
        Wrapped model tester used for embedding extraction and optional
        attention-profile collection.
    tokenizer : Tokenizer
        Tokenizer used to convert spectra into model-ready sequences.
    batch_size : int
        Dataloader batch size.
    spectra : Sequence[Spectrum]
        Input spectra.
    show_progress_bar : bool
        Whether to show progress bars during tokenization / metadata extraction.
    collect_attention_profile : bool
        Whether to collect compressed attention-profile records instead of
        saving full attention matrices.
    identifier_list : Optional[Sequence[str]]
        Optional identifiers aligned with the query order. If None, SMILES
        extracted from the spectra will be used by default.
    target_indices : Optional[Sequence[int]]
        Optional subset of query indices for which attention profiles will
        be collected.
    save_last_layer_only : bool
        Whether to only use the final Transformer layer attention.
    profile_config : Optional[dict]
        Configuration used in Δm/z profile computation.
    sample_metadata : Optional[dict[int, dict]]
        Optional per-query metadata such as group labels and retrieval ranks.

    Returns
    -------
    embedding_array : np.ndarray
        Embeddings for all input spectra.
    smiles_seq : np.ndarray
        SMILES sequence aligned with the input order.
    profile_records : list[dict]
        Per-sample attention-profile records. Empty if
        ``collect_attention_profile`` is False.
    """
    sequences = tokenizer.tokenize_sequence(spectra)
    smiles_seq = get_smiles(sequences, show_progress_bar)

    test_dataset = TestDataset(sequences)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # If no external identifier list is provided, use SMILES by default.
    if identifier_list is None:
        identifier_list = smiles_seq

    # Pass profile-collection related configuration into the tester.
    tester.collect_attention_profile = collect_attention_profile
    tester.identifier_list = list(identifier_list)
    tester.target_indices = (
        None if target_indices is None else set(target_indices)
    )
    tester.save_last_layer_only = save_last_layer_only
    tester.profile_config = profile_config or tester.profile_config
    tester.sample_metadata = sample_metadata or {}
    tester.save_attention = save_attention
    tester.attention_save_path = attention_save_path

    embedding_array, profile_records = tester.test(test_dataloader)
    return embedding_array, smiles_seq, profile_records


# ---------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------

@njit
def cosine_similarity(A: npt.NDArray, B: npt.NDArray):
    """Row-wise cosine similarity between A and B, dense score matrix."""
    norm_A = np.sqrt(np.sum(A ** 2, axis=1)) + 1e-8
    norm_B = np.sqrt(np.sum(B ** 2, axis=1)) + 1e-8
    normalize_A = A / norm_A[:, np.newaxis]
    normalize_B = B / norm_B[:, np.newaxis]
    scores = np.dot(normalize_A, normalize_B.T)
    return scores


@njit(parallel=True)
def top_k_indices(score, top_k):
    """Return the indices of the top-k entries in each row of ``score``."""
    rows, cols = score.shape
    indices = np.empty((rows, top_k), dtype=np.int64)
    for i in prange(rows):
        row = score[i]
        sorted_idx = np.argsort(row)[::-1]
        indices[i] = sorted_idx[:top_k]
    return indices


@njit
def eq(a: npt.NDArray, b: npt.NDArray):
    """Numba-compatible element-wise equality."""
    return np.equal(a, b)


def metric(
    k_metric: list[int],
    query_embedding: npt.NDArray,
    ref_embedding: npt.NDArray,
    query_smiles: npt.NDArray,
    ref_smiles: npt.NDArray,
    show_progress_bar: bool = True,
    batch_size: Optional[int] = None,
):
    """
    CPU retrieval metric.

    Returns
    -------
    infos : list[str]
        Names of the top-k metrics.
    values : list[float]
        Corresponding hit rates.
    result_dict : dict
        Contains the top1 / top5 / top10 hit masks, the first-hit
        rank for each query, and the top1 prediction per query.
    """
    num_query = len(query_embedding)

    top1_mask = np.zeros(num_query, dtype=bool)
    top5_mask = np.zeros(num_query, dtype=bool)
    top10_mask = np.zeros(num_query, dtype=bool)

    # -1 indicates no hit.
    first_hit_rank = np.full(num_query, -1, dtype=int)

    # Save the top1 prediction for later manual inspection.
    top1_pred = np.empty(num_query, dtype=object)

    def init_count():
        return {f"top{k}": 0 for k in k_metric}

    def calculate_hit_count(start: int, end: int, indices: npt.NDArray):
        for query_index, ref_index in zip(range(start, end), indices):
            q = query_smiles[query_index]
            r = ref_smiles[ref_index]

            top1_pred[query_index] = r[0]

            # Find the first position where the query matches.
            hit_positions = np.where(r == q)[0]
            if len(hit_positions) > 0:
                first_hit_rank[query_index] = int(hit_positions[0]) + 1

            if q == r[0]:
                top1_mask[query_index] = True
            if q in r[:5]:
                top5_mask[query_index] = True
            if q in r[:10]:
                top10_mask[query_index] = True

            for k in k_metric:
                if q in r[:k]:
                    hit_count[f"top{k}"] += 1

    k_metric = sorted(list(set(k_metric)))
    k_max = max(k_metric)

    if batch_size is None:
        batch_size = num_query

    hit_count = init_count()

    start_seq = range(0, num_query, batch_size)
    end_seq = range(batch_size, num_query + batch_size, batch_size)
    pbar = zip(start_seq, end_seq)

    if show_progress_bar:
        pbar = tqdm(
            pbar,
            total=len(start_seq),
            desc="calculate hit and recall count",
        )

    for start, end in pbar:
        score = cosine_similarity(
            query_embedding[start:end], ref_embedding
        )
        indices = top_k_indices(score, k_max)
        calculate_hit_count(start, end, indices)

    for key in hit_count:
        hit_count[key] /= num_query

    infos = list(hit_count.keys())
    values = list(hit_count.values())

    result_dict = {
        "mask_dict": {
            "top1": top1_mask,
            "top5": top5_mask,
            "top10": top10_mask,
        },
        "first_hit_rank": first_hit_rank,
        "top1_pred": top1_pred,
    }

    return infos, values, result_dict


def metric_GPU(
    k_metric: list[int],
    query_embedding: npt.NDArray,
    ref_embedding: npt.NDArray,
    query_smiles: npt.NDArray,
    ref_smiles: npt.NDArray,
    show_progress_bar: bool = True,
    batch_size: Optional[int] = None,
    calculate_avg_similarity: bool = False,
    similarity_matrix: Optional[np.ndarray] = None,
):
    """
    GPU retrieval metric with optional average structural similarity.

    When ``calculate_avg_similarity`` is True, ``similarity_matrix``
    must be provided and must have shape
    ``[num_query, num_ref]``.
    """
    num_query = len(query_embedding)

    top1_mask = np.zeros(num_query, dtype=bool)
    top5_mask = np.zeros(num_query, dtype=bool)
    top10_mask = np.zeros(num_query, dtype=bool)

    # -1 indicates no hit.
    first_hit_rank = np.full(num_query, -1, dtype=int)

    # Save the top1 prediction for later manual inspection.
    top1_pred = np.empty(num_query, dtype=object)

    def init_count():
        return {f"top{k}": 0 for k in k_metric}

    def calculate_hit_count(start: int, end: int, indices: npt.NDArray):
        for query_index, ref_index in zip(range(start, end), indices):
            q = query_smiles[query_index]
            r = ref_smiles[ref_index]

            top1_pred[query_index] = r[0]

            # Find the first position where the query matches.
            hit_positions = np.where(r == q)[0]
            if len(hit_positions) > 0:
                first_hit_rank[query_index] = int(hit_positions[0]) + 1

            if q == r[0]:
                top1_mask[query_index] = True
            if q in r[:5]:
                top5_mask[query_index] = True
            if q in r[:10]:
                top10_mask[query_index] = True

            for k in k_metric:
                if q in r[:k]:
                    hit_count[f"top{k}"] += 1

            # Optional: average structural similarity.
            if calculate_avg_similarity:
                for k in k_metric:
                    retrieved_smiles = r[:k]
                    avg_similarity[f"avgsim@{k}"] += float(
                        similarity_matrix[
                            query_index, ref_index[:k]
                        ].mean()
                    )

    k_metric = sorted(list(set(k_metric)))
    k_max = max(k_metric)

    if batch_size is None:
        batch_size = num_query

    hit_count = init_count()

    avg_similarity = None
    if calculate_avg_similarity:
        if similarity_matrix is None:
            raise ValueError(
                "similarity_matrix must be provided "
                "when calculate_avg_similarity=True"
            )
        avg_similarity = {f"avgsim@{k}": 0.0 for k in k_metric}

    start_seq = range(0, num_query, batch_size)
    end_seq = range(batch_size, num_query + batch_size, batch_size)
    pbar = zip(start_seq, end_seq)

    if show_progress_bar:
        pbar = tqdm(
            pbar,
            total=len(start_seq),
            desc="calculate hit and recall count",
        )

    device = torch.device("cuda")
    ref_tensor = torch.tensor(ref_embedding, device=device)
    ref_tensor = F.normalize(ref_tensor, dim=1)

    for start, end in pbar:
        query_tensor = torch.tensor(
            query_embedding[start:end], device=device
        )
        query_tensor = F.normalize(query_tensor, dim=1)

        score = torch.matmul(query_tensor, ref_tensor.T)

        _, indices = torch.topk(score, k_max, dim=1)
        indices = indices.cpu().numpy()

        calculate_hit_count(start, end, indices)

    for key in hit_count:
        hit_count[key] /= num_query

    if calculate_avg_similarity:
        for key in avg_similarity:
            avg_similarity[key] /= num_query

    infos = list(hit_count.keys())
    values = list(hit_count.values())

    if calculate_avg_similarity:
        infos.extend(avg_similarity.keys())
        values.extend(avg_similarity.values())

    result_dict = {
        "mask_dict": {
            "top1": top1_mask,
            "top5": top5_mask,
            "top10": top10_mask,
        },
        "first_hit_rank": first_hit_rank,
        "top1_pred": top1_pred,
    }

    if calculate_avg_similarity:
        result_dict["avg_similarity"] = avg_similarity

    return infos, values, result_dict


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def load_model(
    device,
    model_name,
    model_backbone,
    batch_size,
    loss_type: Literal[
        "TanimotoLoss", "SupConLoss", "SupConWithTanimotoLoss"
    ],
    is_augment: bool = False,
    is_finetune: bool = False,
    is_augment_plus: bool = False,
):
    """
    Build and load one of the supported models.

    Only the following ``model_name`` values are supported:

    - ``"SpecRoFormer_RoPE"``
    - ``"SpecRoFormer_LinearMZ"``
    - ``"SpecRoFormer"`` (the final method, previously named
      ``SpecRoFormer``)
    """
    # root_dir = "/data/model"
    root_dir = Path(__file__).resolve().parents[2] / "data" / "model"
    folder_name = f"{model_name}-{model_backbone}-{loss_type}"

    if is_augment:
        if is_augment_plus:
            folder_name += "-AugmentationPlus"
        else:
            folder_name += "-Augmentation"

    folder_name += f"-bs{batch_size}"

    if is_finetune:
        folder_name += "-finetune"

    model_path = os.path.join(root_dir, folder_name, "model.ckpt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Instantiate the model.
    if model_name == "SpecRoFormer_RoPE":
        model = SpecRoFormer_RoPE(
            512, 16, 4, 512, 512, feedward_activation="selu"
        )
    elif model_name == "SpecRoFormer_LinearMZ":
        model = SpecRoFormer_LinearMZ(
            512, 16, 4, 512, 512, feedward_activation="selu"
        )
    elif model_name == "SpecRoFormer":
        # The final method, previously called
        # SpecRoFormer.
        model = SpecRoFormer(
            512, 16, 4, 512, 512, feedward_activation="selu"
        )
    else:
        raise ValueError(f"Unknown model name {model_name}")

    # Load weights.
    model_state = torch.load(model_path, map_location=device)
    model.load_state_dict(model_state)
    model = model.to(device)
    return model


# ---------------------------------------------------------------------
# High-level retrieval pipelines
# ---------------------------------------------------------------------

def search_with_spectra(
    desc: str,
    tester: ModelTester,
    k_metric: list[int],
    tokenizer: Tokenizer,
    query_spectra: Sequence[Spectrum],
    ref_spectra: Sequence[Spectrum],
    loader_batch_size: int,
    show_progress_bar: bool = True,
    batch_size: Optional[int] = None,
    use_gpu_metric: bool = False,
    return_report: bool = False,
    return_profile: bool = False,
    target_indices: Optional[Sequence[int]] = None,
    save_last_layer_only: bool = True,
    profile_config: Optional[dict] = None,
    sample_metadata: Optional[dict[int, dict]] = None,
    calculate_avg_similarity=False,
    save_attention: bool = False,
    attention_save_path: Optional[str] = None,
):
    """
    Search using in-memory spectra.

    Parameters
    ----------
    return_report : bool
        Whether to additionally return a per-query retrieval report.
    return_profile : bool
        Whether to additionally return a DataFrame containing sample-level
        attention-profile statistics.
    target_indices : Optional[Sequence[int]]
        If provided, only these query indices will be used to collect
        attention-profile records.
    save_last_layer_only : bool
        Whether to only use the final Transformer attention layer.
    profile_config : Optional[dict]
        Configuration used for Δm/z profile computation.
    sample_metadata : Optional[dict[int, dict]]
        Optional per-query metadata such as group labels and retrieval ranks.

    Returns
    -------
    df : pd.DataFrame
        Summary retrieval metrics.
    report_df : pd.DataFrame, optional
        Per-query retrieval results.
    profile_df : pd.DataFrame, optional
        Sample-level attention-profile table.
    """
    query_embedding, query_smiles, query_profile_records = embedding(
        tester,
        tokenizer,
        loader_batch_size,
        query_spectra,
        show_progress_bar=show_progress_bar,
        collect_attention_profile=return_profile,
        identifier_list=None,
        target_indices=target_indices,
        save_last_layer_only=save_last_layer_only,
        profile_config=profile_config,
        sample_metadata=sample_metadata,
        save_attention=save_attention,
        attention_save_path=attention_save_path,
    )

    ref_embedding, ref_smiles, _ = embedding(
        tester,
        tokenizer,
        loader_batch_size,
        ref_spectra,
        show_progress_bar=show_progress_bar,
        collect_attention_profile=False,
        identifier_list=None,
        target_indices=None,
        save_last_layer_only=True,
        profile_config=None,
        sample_metadata=None,
    )

    if use_gpu_metric:
        if calculate_avg_similarity:
            similarity_matrix = build_similarity_matrix_from_smiles(
                query_smiles,
                ref_smiles,
            )
            infos, values, result_dict = metric_GPU(
                k_metric,
                query_embedding,
                ref_embedding,
                query_smiles,
                ref_smiles,
                show_progress_bar,
                batch_size,
                calculate_avg_similarity=True,
                similarity_matrix=similarity_matrix,
            )
        else:
            infos, values, result_dict = metric_GPU(
                k_metric,
                query_embedding,
                ref_embedding,
                query_smiles,
                ref_smiles,
                show_progress_bar,
                batch_size,
            )
    else:
        infos, values, result_dict = metric(
            k_metric,
            query_embedding,
            ref_embedding,
            query_smiles,
            ref_smiles,
            show_progress_bar,
            batch_size,
        )

    df = pd.DataFrame([values], columns=infos, index=[desc])

    outputs = [df]

    report_df = None
    if return_report:
        mask_dict = result_dict["mask_dict"]
        first_hit_rank = result_dict["first_hit_rank"]
        top1_pred = result_dict["top1_pred"]

        report_df = pd.DataFrame({
            "query_index": np.arange(len(query_smiles)),
            "query_smiles": query_smiles,
            "top1_hit": mask_dict["top1"],
            "top5_hit": mask_dict["top5"],
            "top10_hit": mask_dict["top10"],
            "first_hit_rank": first_hit_rank,
            "top1_pred": top1_pred,
        })
        outputs.append(report_df)

    profile_df = None
    if return_profile:
        profile_df = pd.DataFrame(query_profile_records)
        outputs.append(profile_df)

    if len(outputs) == 1:
        return outputs[0]
    return tuple(outputs)


def search(
    desc: str,
    tester: ModelTester,
    k_metric: list[int],
    tokenizer: Tokenizer,
    query_path: Path,
    ref_path: Path,
    loader_batch_size: int,
    show_progress_bar: bool = True,
    batch_size: Optional[int] = None,
    use_gpu_metric: bool = False,
    return_report: bool = False,
    return_profile: bool = False,
    target_indices: Optional[Sequence[int]] = None,
    save_last_layer_only: bool = True,
    profile_config: Optional[dict] = None,
    sample_metadata: Optional[dict[int, dict]] = None,
    calculate_avg_similarity=False,
    save_attention: bool = False,
    attention_save_path: Optional[str] = None,
):
    """
    Search using spectra loaded from disk.

    Parameters
    ----------
    return_report : bool
        Whether to additionally return a per-query retrieval report.
    return_profile : bool
        Whether to additionally return a DataFrame containing sample-level
        attention-profile statistics.
    target_indices : Optional[Sequence[int]]
        If provided, only these query indices will be used to collect
        attention-profile records.
    save_last_layer_only : bool
        Whether to only use the final Transformer attention layer.
    profile_config : Optional[dict]
        Configuration used for Δm/z profile computation.
    sample_metadata : Optional[dict[int, dict]]
        Optional per-query metadata such as group labels and retrieval ranks.

    Returns
    -------
    df : pd.DataFrame
        Summary retrieval metrics.
    report_df : pd.DataFrame, optional
        Per-query retrieval results.
    profile_df : pd.DataFrame, optional
        Sample-level attention-profile table.
    """
    query_spectra = np.load(query_path, allow_pickle=True)
    ref_spectra = np.load(ref_path, allow_pickle=True)

    query_embedding, query_smiles, query_profile_records = embedding(
        tester,
        tokenizer,
        loader_batch_size,
        query_spectra,
        show_progress_bar=show_progress_bar,
        collect_attention_profile=return_profile,
        identifier_list=None,
        target_indices=target_indices,
        save_last_layer_only=save_last_layer_only,
        profile_config=profile_config,
        sample_metadata=sample_metadata,
        save_attention=save_attention,
        attention_save_path=attention_save_path,
    )

    ref_embedding, ref_smiles, _ = embedding(
        tester,
        tokenizer,
        loader_batch_size,
        ref_spectra,
        show_progress_bar=show_progress_bar,
        collect_attention_profile=False,
        identifier_list=None,
        target_indices=None,
        save_last_layer_only=True,
        profile_config=None,
        sample_metadata=None,
    )

    if use_gpu_metric:
        if calculate_avg_similarity:
            similarity_matrix = build_similarity_matrix_from_smiles(
                query_smiles,
                ref_smiles,
            )
            infos, values, result_dict = metric_GPU(
                k_metric,
                query_embedding,
                ref_embedding,
                query_smiles,
                ref_smiles,
                show_progress_bar,
                batch_size,
                calculate_avg_similarity=True,
                similarity_matrix=similarity_matrix,
            )
        else:
            infos, values, result_dict = metric_GPU(
                k_metric,
                query_embedding,
                ref_embedding,
                query_smiles,
                ref_smiles,
                show_progress_bar,
                batch_size,
            )
    else:
        infos, values, result_dict = metric(
            k_metric,
            query_embedding,
            ref_embedding,
            query_smiles,
            ref_smiles,
            show_progress_bar,
            batch_size,
        )

    df = pd.DataFrame([values], columns=infos, index=[desc])

    outputs = [df]

    report_df = None
    if return_report:
        mask_dict = result_dict["mask_dict"]
        first_hit_rank = result_dict["first_hit_rank"]
        top1_pred = result_dict["top1_pred"]

        report_df = pd.DataFrame({
            "query_index": np.arange(len(query_smiles)),
            "query_smiles": query_smiles,
            "top1_hit": mask_dict["top1"],
            "top5_hit": mask_dict["top5"],
            "top10_hit": mask_dict["top10"],
            "first_hit_rank": first_hit_rank,
            "top1_pred": top1_pred,
        })
        outputs.append(report_df)

    profile_df = None
    if return_profile:
        profile_df = pd.DataFrame(query_profile_records)
        outputs.append(profile_df)

    if len(outputs) == 1:
        return outputs[0]
    return tuple(outputs)


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def most_similar(
    query_embedding: npt.NDArray,
    ref_embedding: npt.NDArray,
    batch_size: int,
    show_progress_bar: bool = True,
):
    """Return the score and index of the top-1 reference per query."""
    start_seq = range(0, len(query_embedding), batch_size)
    end_seq = range(
        batch_size, len(query_embedding) + batch_size, batch_size
    )
    pbar = zip(start_seq, end_seq)

    if show_progress_bar:
        pbar = tqdm(
            pbar,
            total=len(start_seq),
            desc="processing",
        )

    scores = []
    most_similar_indices = []

    for start, end in pbar:
        cosine_score = cosine_similarity(
            query_embedding[start:end], ref_embedding
        )
        indices = top_k_indices(cosine_score, 1).flatten()

        for i, j in zip(range(cosine_score.shape[0]), indices):
            scores.append(cosine_score[i][j])
            most_similar_indices.append(j)

    return np.array(scores), np.array(most_similar_indices)


def split_attention_by_hit(
    attention_dir: str,
    correct_indices,
    wrong_indices,
    separator: str = "_",
):
    """
    Move saved attention files into ``correct`` / ``wrong``
    subdirectories based on the hit mask.

    Parameters
    ----------
    attention_dir : str
        Directory containing the saved attention files.
    correct_indices : iterable[int]
        Indices of correctly retrieved samples.
    wrong_indices : iterable[int]
        Indices of incorrectly retrieved samples.
    separator : str
        Separator used between the inchikey and the sample index in
        the file name (default ``"_"``).
    """
    attention_dir = Path(attention_dir)

    correct_dir = attention_dir / "correct"
    wrong_dir = attention_dir / "wrong"

    correct_dir.mkdir(exist_ok=True)
    wrong_dir.mkdir(exist_ok=True)

    correct_set = set(correct_indices)
    wrong_set = set(wrong_indices)

    for file in attention_dir.glob("*.pt"):
        stem = file.stem  # drop the ".pt" suffix

        # Parse the sample index after the last separator.
        if separator not in stem:
            continue

        index_str = stem.rsplit(separator, 1)[-1]

        try:
            index = int(index_str)
        except ValueError:
            continue

        if index in correct_set:
            shutil.move(str(file), correct_dir / file.name)
        elif index in wrong_set:
            shutil.move(str(file), wrong_dir / file.name)

        # Files not in either set are left untouched.