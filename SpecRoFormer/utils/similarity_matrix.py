"""
Fingerprint-based structural similarity utilities.

Build a Tanimoto similarity matrix between query and reference
SMILES using the Daylight fingerprint (2048 bits).
"""

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm


def build_fingerprint_cache(smiles_list):
    """
    Build a fingerprint cache keyed by SMILES.

    Parameters
    ----------
    smiles_list : array-like
        SMILES strings (may contain duplicates).

    Returns
    -------
    fp_cache : dict
        Mapping from SMILES to its RDKit fingerprint, or to None if
        the SMILES could not be parsed.
    """
    unique_smiles = np.unique(smiles_list)

    generator = rdFingerprintGenerator.GetRDKitFPGenerator(
        fpSize=2048
    )

    fp_cache = {}

    for smiles in tqdm(unique_smiles, desc="Build fingerprints"):
        if smiles is None:
            fp_cache[smiles] = None
            continue

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                fp_cache[smiles] = None
                continue

            fp_cache[smiles] = generator.GetFingerprint(mol)

        except Exception:
            fp_cache[smiles] = None

    return fp_cache


def build_similarity_matrix_from_smiles(query_smiles, ref_smiles):
    """
    Build the query-reference structural similarity matrix.

    Uses the Daylight fingerprint (2048 bits) and Tanimoto / Jaccard
    similarity.

    Parameters
    ----------
    query_smiles, ref_smiles : array-like
        SMILES sequences aligned with the query / reference order.

    Returns
    -------
    similarity_matrix : np.ndarray
        Shape ``[num_query, num_ref]``, float32.
    """
    query_smiles = np.asarray(query_smiles)
    ref_smiles = np.asarray(ref_smiles)

    # -------------------------------------------------------------
    # Unique SMILES
    # -------------------------------------------------------------
    query_unique, query_inverse = np.unique(
        query_smiles,
        return_inverse=True,
    )
    ref_unique, ref_inverse = np.unique(
        ref_smiles,
        return_inverse=True,
    )

    # -------------------------------------------------------------
    # Fingerprints
    # -------------------------------------------------------------
    all_smiles = np.concatenate([query_unique, ref_unique])
    fp_cache = build_fingerprint_cache(all_smiles)

    # -------------------------------------------------------------
    # Unique similarity matrix
    # -------------------------------------------------------------
    unique_similarity = np.zeros(
        (len(query_unique), len(ref_unique)),
        dtype=np.float32,
    )

    ref_fps = [fp_cache[s] for s in ref_unique]

    for i, q in enumerate(
        tqdm(query_unique, desc="Compute Tanimoto")
    ):
        fp_q = fp_cache[q]
        if fp_q is None:
            continue

        valid_indices = [
            idx for idx, fp in enumerate(ref_fps) if fp is not None
        ]
        valid_ref_fps = [ref_fps[idx] for idx in valid_indices]

        if len(valid_ref_fps) == 0:
            continue

        similarities = DataStructs.BulkTanimotoSimilarity(
            fp_q,
            valid_ref_fps,
        )

        unique_similarity[i, valid_indices] = similarities

    # -------------------------------------------------------------
    # Expand back to full shape
    # -------------------------------------------------------------
    similarity_matrix = unique_similarity[query_inverse][:, ref_inverse]

    return similarity_matrix.astype(np.float32)