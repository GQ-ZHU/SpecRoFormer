"""
Clean pretrain dataset.

Load each raw pretrain MGF, derive precursor m/z from SMILES for
in-silico spectra, set standard metadata fields, and filter out
spectra whose SMILES cannot be parsed or whose peaks are empty.
"""

import sys

sys.path.append("..")

import logging
import re

import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors
from matchms.importing import load_from_mgf, load_from_msp
from matchms.exporting import save_as_mgf, save_as_msp

from SpecRoFormer.utils.clean import count_annotations, is_annotated
from SpecRoFormer.const import pretrain_data


# Suppress noisy matchms warnings
logging.getLogger("matchms").setLevel(logging.ERROR)


# Proton mass in Dalton
PROTON_MASS = 1.007276466812


# ---------------------------------------------------------------------
# SMILES utilities
# ---------------------------------------------------------------------

def clean_cfm_smiles(smiles: str):
    """
    Clean non-standard SMILES produced by in-silico tools (e.g. CFM-ID).
    """
    if smiles is None:
        return None

    smiles = smiles.strip()

    # Drop illegal leading symbols such as ":"
    if smiles.startswith(":"):
        smiles = smiles[1:]

    if len(smiles) == 0:
        return None

    return smiles


def smiles_to_precursor_mz_robust(smiles: str):
    """
    Robust precursor m/z computation for in-silico SMILES.

    Returns
    -------
    float or None
        Precursor m/z if parsing succeeds, otherwise None.
    """
    smiles = clean_cfm_smiles(smiles)
    if smiles is None:
        return None

    # First attempt: normal parsing
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Second attempt: relaxed parsing without sanitization
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None

    exact_mass = Descriptors.ExactMolWt(mol)
    return exact_mass + PROTON_MASS


def get_smiles_from_spectrum(spectrum):
    """Return SMILES from spectrum metadata, trying several key names."""
    meta = spectrum.metadata
    for key in ["smiles", "SMILES", "smiles/inchi", "SMILES/INCHI"]:
        if key in meta:
            return meta[key]
    return None


# ---------------------------------------------------------------------
# Virtual spectrum filtering
# ---------------------------------------------------------------------

def is_annotated_virtual(spectra):
    """
    Filtering strategy for in-silico MS/MS spectra (e.g. CFM-ID).

    Keep spectra that have a usable SMILES for precursor computation
    and at least one peak.
    """
    filtered = []

    for s in tqdm(spectra):
        smiles = (
            s.get("smiles")
            or s.get("SMILES")
            or s.get("smiles/inchi")
            or s.get("SMILES/INCHI")
        )

        if smiles is None:
            continue

        precursor_mz = smiles_to_precursor_mz_robust(smiles)
        if precursor_mz is None:
            continue

        # Require at least one peak
        if len(s.peaks.mz) == 0:
            continue

        filtered.append(s)

    return filtered


# ---------------------------------------------------------------------
# Main loop over pretrain datasets
# ---------------------------------------------------------------------

for dataset_name, save_path in pretrain_data.PRETRAIN_DATASETS.items():
    print("\n================================================")
    print("loading dataset:", dataset_name)

    pretrain_dataset_path = (
        f"../data/pretrain/{dataset_name}.mgf"
    )
    pretrain_data_local = list(load_from_mgf(pretrain_dataset_path))

    valid_count = 0
    missing_smiles = 0

    for spectrum in pretrain_data_local:
        smiles = get_smiles_from_spectrum(spectrum)
        precursor_mz = smiles_to_precursor_mz_robust(smiles)

        if precursor_mz is None:
            missing_smiles += 1
            continue

        spectrum.set("precursor_mz", precursor_mz)
        spectrum.set("charge", 1)
        spectrum.set("ionmode", "positive")
        spectrum.set("precursor_type", "[M+H]+")

        valid_count += 1

    print(f"Added precursor_mz for {valid_count} spectra")
    print(f"Missing SMILES: {missing_smiles}")

    count_annotations(pretrain_data_local, "Pretrain Dataset")

    print("Filtering pretrain dataset...")
    pretrain_data_local = is_annotated_virtual(pretrain_data_local)
    count_annotations(pretrain_data_local, "Pretrain Dataset")

    np.save(save_path, pretrain_data_local)