"""
Deduplicate evaluation datasets against a training dataset using InChIKey.

Supported input formats
-----------------------
- npy  : list / dict-like spectra with smiles or inchikey keys
- mgf  : loaded via matchms
- msp  : loaded via matchms (with MassBank cleaning pipeline)
- tsv  : MassSpecGym-style tabular file
"""

import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from matchms import Spectrum
from matchms.importing import load_from_mgf, load_from_msp
from matchms.exporting import save_as_mgf


# Make SpecRoFormer importable
specembedding_path = "../../SpecRoFormer"
if specembedding_path not in sys.path:
    sys.path.append(specembedding_path)

from SpecRoFormer.utils.clean import (
    apply_filters,
    clean_metadata,
    clean_metadata2,
    seperate_spectra_by_ionmode,
    minimal_processing,
    filter_by_precursor_mz,
    is_annotated,
)


# ---------------------------------------------------------------------
# InChIKey normalization
# ---------------------------------------------------------------------

def normalize_inchikey(inchikey):
    """
    Use only the first connectivity block of InChIKey.

    Example
    -------
    ABCDEFGHIJKLMN-UHFFFAOYSA-N  ->  ABCDEFGHIJKLMN
    """
    if inchikey is None:
        return None

    if pd.isna(inchikey):
        return None

    inchikey = str(inchikey).strip().upper()

    if inchikey == "":
        return None

    return inchikey.split("-")[0]


def smiles_to_inchikey(smiles):
    """Convert a SMILES string to a normalized InChIKey (or None)."""
    if smiles is None or pd.isna(smiles):
        return None

    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None

        inchikey = Chem.MolToInchiKey(mol)
        return normalize_inchikey(inchikey)

    except Exception:
        return None


def get_identifier(metadata):
    """
    Return the compound identifier used for deduplication.

    Prefer inchikey if available, otherwise derive it from SMILES.
    """
    inchikey = metadata.get("inchikey", None)

    if inchikey is not None and not pd.isna(inchikey):
        return normalize_inchikey(inchikey)

    smiles = metadata.get("smiles", None)
    return smiles_to_inchikey(smiles)


# ---------------------------------------------------------------------
# Training set loading
# ---------------------------------------------------------------------

def build_train_set(train_path):
    """Build a set of compound identifiers from the training spectra."""
    train = np.load(train_path, allow_pickle=True)
    ids = set()

    for s in train:
        if isinstance(s, dict):
            meta = s
        else:
            meta = getattr(s, "metadata", {})

        key = get_identifier(meta)
        if key is not None:
            ids.add(key)

    return ids


# ---------------------------------------------------------------------
# Per-format deduplication
# ---------------------------------------------------------------------

def dedup_npy(path, train_ids):
    """Deduplicate a .npy spectrum array."""
    data = np.load(path, allow_pickle=True)
    keep = []
    removed = []

    for x in data:
        meta = x if isinstance(x, dict) else getattr(x, "metadata", {})
        key = get_identifier(meta)

        if key is None or key not in train_ids:
            keep.append(x)
        else:
            removed.append({
                "identifier": key,
                "smiles": meta.get("smiles", None),
            })

    root, ext = os.path.splitext(path)
    out = root + "_dedup.npy"
    np.save(out, np.array(keep, dtype=object))

    return len(data), len(keep), removed, out


def dedup_mgf(path, train_ids):
    """Deduplicate an .mgf file (via matchms)."""
    spectra = list(load_from_mgf(path))
    keep = []
    removed = []

    for s in spectra:
        meta = s.metadata
        key = get_identifier(meta)

        if key is None or key not in train_ids:
            keep.append(s)
        else:
            removed.append({
                "identifier": key,
                "smiles": meta.get("smiles", None),
            })

    root, ext = os.path.splitext(path)
    out = root + "_dedup.npy"
    np.save(out, np.asarray(keep, dtype=object))

    return len(spectra), len(keep), removed, out


def dedup_msp(path, train_ids):
    """
    Deduplicate a .msp / .msp_nist file.

    A MassBank-style cleaning pipeline is applied before deduplication.
    """
    # -----------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------
    spectra = list(load_from_msp(path))
    print(f"Raw spectra: {len(spectra)}")

    # -----------------------------------------------------------------
    # Original MassBank cleaning pipeline
    # -----------------------------------------------------------------
    spectra = [
        apply_filters(s)
        for s in tqdm(spectra, desc="apply_filters")
    ]
    spectra = [s for s in spectra if s is not None]

    spectra = [
        clean_metadata(s)
        for s in tqdm(spectra, desc="clean_metadata")
    ]
    spectra = [
        clean_metadata2(s)
        for s in tqdm(spectra, desc="clean_metadata2")
    ]

    positive, _ = seperate_spectra_by_ionmode(spectra)

    positive = [
        minimal_processing(s)
        for s in tqdm(positive, desc="minimal_processing")
    ]
    positive = [s for s in positive if s is not None]

    positive = filter_by_precursor_mz(positive)
    positive = is_annotated(positive)

    spectra = positive

    print(f"After original cleaning: {len(spectra)}")

    # -----------------------------------------------------------------
    # Deduplicate
    # -----------------------------------------------------------------
    keep = []
    removed = []

    for s in tqdm(spectra, desc="deduplicate"):
        meta = s.metadata
        key = get_identifier(meta)

        if key is None or key not in train_ids:
            keep.append(s)
        else:
            removed.append({
                "identifier": key,
                "smiles": meta.get("smiles", None),
            })

    print(f"Remaining spectra: {len(keep)}")

    root, ext = os.path.splitext(path)
    out = root + "_dedup.npy"
    np.save(out, np.asarray(keep, dtype=object))

    return len(spectra), len(keep), removed, out


def dedup_tsv(path, train_ids):
    """Deduplicate a MassSpecGym-style .tsv file."""
    df = pd.read_csv(path, sep="\t")

    keep = []
    removed = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        smiles = row["smiles"]
        key = smiles_to_inchikey(smiles)

        if key is not None and key in train_ids:
            removed.append({
                "identifier": key,
                "smiles": smiles,
            })
            continue

        # -------------------------------------------------------------
        # Peaks
        # -------------------------------------------------------------
        mz_col = "mzs" if "mzs" in df.columns else "mz"
        intensity_col = (
            "intensities" if "intensities" in df.columns else "intensity"
        )

        mz = np.asarray(
            str(row[mz_col]).split(","),
            dtype=np.float32,
        )
        intensity = np.asarray(
            str(row[intensity_col]).split(","),
            dtype=np.float32,
        )

        if len(mz) != len(intensity):
            continue

        if len(mz) == 0:
            continue

        spectrum = Spectrum(
            mz=mz,
            intensities=intensity,
            metadata={
                "smiles": row["smiles"],
                "inchikey": row["inchikey"],
                "precursor_mz": row["precursor_mz"],
                "adduct": row["adduct"],
                "instrument_type": row["instrument_type"],
            },
        )

        keep.append(spectrum)

    root, ext = os.path.splitext(path)
    out = root + "_dedup.npy"
    np.save(out, np.asarray(keep, dtype=object))

    return len(df), len(keep), removed, out


# ---------------------------------------------------------------------
# Main deduplication driver
# ---------------------------------------------------------------------

def deduplicate(train_path, dataset_paths):
    """
    Deduplicate each dataset in ``dataset_paths`` against ``train_path``.

    Writes two reports:
        - deduplication_report.xlsx
        - removed_compounds.xlsx
    """
    train_ids = build_train_set(train_path)

    reports = []
    all_removed = []

    for path in dataset_paths:
        ext = os.path.splitext(path)[1].lower()

        if ext == ".npy":
            total, remain, removed, out = dedup_npy(path, train_ids)
        elif ext == ".mgf":
            total, remain, removed, out = dedup_mgf(path, train_ids)
        elif ext == ".msp":
            total, remain, removed, out = dedup_msp(path, train_ids)
        elif ext == ".msp_nist":
            total, remain, removed, out = dedup_msp(path, train_ids)
        elif ext == ".tsv":
            total, remain, removed, out = dedup_tsv(path, train_ids)
        else:
            print(f"Skip unsupported file: {path}")
            continue

        reports.append({
            "Dataset": os.path.basename(path),
            "Original Spectra": total,
            "Remaining Spectra": remain,
            "Removed Spectra": total - remain,
            "Removed Compounds": len(set(
                x["identifier"] for x in removed if x["identifier"]
            )),
        })

        for r in removed:
            r["dataset"] = os.path.basename(path)
            all_removed.append(r)

        print(f"{path} -> {out}")

    pd.DataFrame(reports).to_excel(
        "deduplication_report.xlsx",
        index=False,
    )

    pd.DataFrame(all_removed).drop_duplicates().to_excel(
        "removed_compounds.xlsx",
        index=False,
    )

    print("Saved deduplication_report.xlsx")
    print("Saved removed_compounds.xlsx")


if __name__ == "__main__":
    train_path = (
        "../data/MSBert/GNPS/Orbitrap/train_ref.npy"
    )

    datasets = [
        # "../data/MSBert/GNPS/"
        # "Orbitrap_test_query_cleaned.mgf",
        # "../data/MSBert/GNPS/"
        # "Orbitrap_test_ref_cleaned.mgf",
        # "../data/MSBert/GNPS/"
        # "Other_cleaned.mgf",
        # "../data/MSBert/GNPS/"
        # "QTOF_cleaned.mgf",
        # "../data/MSBert/MoNA/"
        # "MoNA_Orbitrap_cleaned.msp",
        # "../data/MSBert/MoNA/"
        # "MoNA_QTOF_cleaned.msp",
        # "../data/"
        # "MassBank/MassBank.msp_NIST",
        # "../data/"
        # "massSpecGymData/MassSpecGym.tsv",
    ]

    deduplicate(train_path, datasets)