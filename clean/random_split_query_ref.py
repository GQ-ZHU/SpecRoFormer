"""
Random query/reference splitting utilities.

This module defines generic helpers to:
  - canonicalize SMILES
  - collect unique SMILES from spectra
  - split a spectra list into query/reference sets
  - rebuild query/reference splits for MSBERT, MassBank and MassSpecGym
"""

from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from matchms.importing import load_from_mgf

from SpecRoFormer.utils.clean import get_ref_query, get_unique_smiles


# ---------------------------------------------------------------------
# SMILES utilities
# ---------------------------------------------------------------------

def canonicalize_smiles(smiles):
    """
    Convert a SMILES string to canonical SMILES.

    Parameters
    ----------
    smiles : str

    Returns
    -------
    str or None
        Canonical SMILES if valid, otherwise None.
    """
    if smiles is None:
        return None

    if not isinstance(smiles, str):
        return None

    smiles = smiles.strip()

    if smiles == "":
        return None

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True,
    )


# ---------------------------------------------------------------------
# MSBERT dataset processing
# ---------------------------------------------------------------------

def process_msbert(
    gnps_root,
    mona_root,
    replica_num=10,
    replica_suffix="-replication-{}",
):
    """
    Process MSBERT datasets (GNPS + MoNA).

    Parameters
    ----------
    gnps_root : str or Path
        Directory containing GNPS cleaned MGF/NPY files.

    mona_root : str or Path
        Directory containing MoNA cleaned MSP/NPY files.

    replica_num : int
        Number of query/reference replicas to generate.

    replica_suffix : str
        Suffix template for replica files.
    """
    gnps_root = Path(gnps_root)
    mona_root = Path(mona_root)

    # ----------------------------------------------------------------
    # GNPS Orbitrap
    # ----------------------------------------------------------------
    print("=" * 80)
    print("Processing GNPS Orbitrap")
    print("=" * 80)

    train_ref = np.array(
        list(
            load_from_mgf(
                str(gnps_root / "Orbitrap_train_ref_cleaned.mgf")
            )
        ),
        dtype=object,
    )
    train_query = np.array(
        list(
            load_from_mgf(
                str(gnps_root / "Orbitrap_train_query_cleaned.mgf")
            )
        ),
        dtype=object,
    )

    test_ref = np.load(
        str(gnps_root / "Orbitrap_test_ref_cleaned_dedup.npy"),
        allow_pickle=True,
    )
    test_query = np.load(
        str(gnps_root / "Orbitrap_test_query_cleaned_dedup.npy"),
        allow_pickle=True,
    )

    orbitrap_all = np.concatenate(
        [train_ref, train_query, test_ref, test_query]
    )

    orbitrap_dir = gnps_root / "Orbitrap"
    orbitrap_dir.mkdir(parents=True, exist_ok=True)

    np.save(orbitrap_dir / "all.npy", orbitrap_all)
    np.save(orbitrap_dir / "train_ref.npy", np.array(train_ref, dtype=object))
    np.save(
        orbitrap_dir / "train_query.npy",
        np.array(train_query, dtype=object),
    )
    np.save(
        orbitrap_dir / "test_ref.npy",
        np.array(test_ref, dtype=object),
    )
    np.save(
        orbitrap_dir / "test_query.npy",
        np.array(test_query, dtype=object),
    )

    train_ref_smiles = get_unique_smiles(train_ref)

    test_spectra = np.concatenate([test_query, test_ref])
    test_smiles = get_unique_smiles(test_spectra)
    available_smiles = test_smiles - train_ref_smiles

    print(
        f"Unique SMILES in GNPS Orbitrap train_ref: "
        f"{len(train_ref_smiles)}"
        f"\nUnique SMILES in GNPS Orbitrap test: {len(test_smiles)}"
        f"\nAvailable SMILES for GNPS Orbitrap test split: "
        f"{len(available_smiles)}"
    )

    for seed in range(replica_num):
        query, ref = get_ref_query(
            test_spectra,
            available_smiles,
            seed=seed,
        )
        np.save(
            orbitrap_dir
            / f"test_query{replica_suffix.format(seed + 1)}.npy",
            query,
        )
        np.save(
            orbitrap_dir
            / f"test_ref{replica_suffix.format(seed + 1)}.npy",
            ref,
        )

    # ----------------------------------------------------------------
    # GNPS QTOF
    # ----------------------------------------------------------------
    print("=" * 80)
    print("Processing GNPS QTOF")
    print("=" * 80)

    qtof = np.load(
        str(gnps_root / "QTOF_cleaned_dedup.npy"),
        allow_pickle=True,
    )

    qtof_dir = gnps_root / "QTOF"
    qtof_dir.mkdir(parents=True, exist_ok=True)

    np.save(qtof_dir / "all.npy", qtof)

    query, ref = get_ref_query(
        qtof,
        get_unique_smiles(qtof),
        seed=0,
    )
    np.save(qtof_dir / "test_query.npy", query)
    np.save(qtof_dir / "test_ref.npy", ref)

    qtof_smiles = get_unique_smiles(qtof)
    available_smiles = qtof_smiles - train_ref_smiles

    print(
        f"Unique SMILES in GNPS Orbitrap train_ref: "
        f"{len(train_ref_smiles)}"
        f"\nUnique SMILES in GNPS QTOF test: {len(qtof_smiles)}"
        f"\nAvailable SMILES for GNPS QTOF test split: "
        f"{len(available_smiles)}"
    )

    for seed in range(replica_num):
        query, ref = get_ref_query(
            qtof,
            available_smiles,
            seed=seed,
        )
        np.save(
            qtof_dir / f"test_query{replica_suffix.format(seed + 1)}.npy",
            query,
        )
        np.save(
            qtof_dir / f"test_ref{replica_suffix.format(seed + 1)}.npy",
            ref,
        )

    # ----------------------------------------------------------------
    # GNPS Other
    # ----------------------------------------------------------------
    print("=" * 80)
    print("Processing GNPS Other")
    print("=" * 80)

    other = np.load(
        str(gnps_root / "Other_cleaned_dedup.npy"),
        allow_pickle=True,
    )

    other_dir = gnps_root / "Other"
    other_dir.mkdir(parents=True, exist_ok=True)

    np.save(other_dir / "all.npy", other)

    query, ref = get_ref_query(
        other,
        get_unique_smiles(other),
        seed=0,
    )
    np.save(other_dir / "test_query.npy", query)
    np.save(other_dir / "test_ref.npy", ref)

    other_smiles = get_unique_smiles(other)
    available_smiles = other_smiles - train_ref_smiles

    print(
        f"Unique SMILES in GNPS Orbitrap train_ref: "
        f"{len(train_ref_smiles)}"
        f"\nUnique SMILES in GNPS Other test: {len(other_smiles)}"
        f"\nAvailable SMILES for GNPS Other test split: "
        f"{len(available_smiles)}"
    )

    for seed in range(replica_num):
        query, ref = get_ref_query(
            other,
            available_smiles,
            seed=seed,
        )
        np.save(
            other_dir / f"test_query{replica_suffix.format(seed + 1)}.npy",
            query,
        )
        np.save(
            other_dir / f"test_ref{replica_suffix.format(seed + 1)}.npy",
            ref,
        )

    # ----------------------------------------------------------------
    # MoNA Orbitrap
    # ----------------------------------------------------------------
    print("=" * 80)
    print("Processing MoNA Orbitrap")
    print("=" * 80)

    mona_orbitrap = np.load(
        str(mona_root / "MoNA_Orbitrap_cleaned_dedup.npy"),
        allow_pickle=True,
    )
    mona_orbitrap_smiles = (
        get_unique_smiles(mona_orbitrap) - train_ref_smiles
    )

    print(
        f"Unique SMILES in GNPS Orbitrap train_ref: "
        f"{len(train_ref_smiles)}"
        f"\nUnique SMILES in MoNA Orbitrap test: "
        f"{len(get_unique_smiles(mona_orbitrap))}"
        f"\nAvailable SMILES for MoNA Orbitrap test split: "
        f"{len(mona_orbitrap_smiles)}"
    )

    query, ref = get_ref_query(
        mona_orbitrap,
        mona_orbitrap_smiles,
        seed=0,
    )
    np.save(mona_root / "orbitrap_query.npy", query)
    np.save(mona_root / "orbitrap_ref.npy", ref)

    # ----------------------------------------------------------------
    # MoNA QTOF
    # ----------------------------------------------------------------
    print("=" * 80)
    print("Processing MoNA QTOF")
    print("=" * 80)

    mona_qtof = np.load(
        str(mona_root / "MoNA_QTOF_cleaned_dedup.npy"),
        allow_pickle=True,
    )
    mona_qtof_smiles = (
        get_unique_smiles(mona_qtof) - train_ref_smiles
    )

    print(
        f"Unique SMILES in GNPS Orbitrap train_ref: "
        f"{len(train_ref_smiles)}"
        f"\nUnique SMILES in MoNA QTOF test: "
        f"{len(get_unique_smiles(mona_qtof))}"
        f"\nAvailable SMILES for MoNA QTOF test split: "
        f"{len(mona_qtof_smiles)}"
    )

    query, ref = get_ref_query(
        mona_qtof,
        mona_qtof_smiles,
        seed=0,
    )
    np.save(mona_root / "qtof_query.npy", query)
    np.save(mona_root / "qtof_ref.npy", ref)

    print("=" * 80)
    print("MSBERT processing finished.")
    print("=" * 80)


# ---------------------------------------------------------------------
# MassBank dataset processing
# ---------------------------------------------------------------------

def process_massbank(
    massbank_path,
    train_ref_path,
    output_dir,
    replica_num=10,
    replica_suffix="-replication-{}",
):
    """
    Rebuild MassBank query/reference split after deduplication.

    Parameters
    ----------
    massbank_path : str or Path
        MassBank_dedup.npy.

    train_ref_path : str or Path
        GNPS Orbitrap train_ref MGF file.

    output_dir : str or Path

    replica_num : int
        Number of query/reference replicas to generate.

    replica_suffix : str
        Suffix template for replica files.
    """
    massbank_path = Path(massbank_path)
    train_ref_path = Path(train_ref_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Loading data")
    print("=" * 80)

    spectra = np.load(massbank_path, allow_pickle=True)

    train_ref = np.array(
        list(load_from_mgf(str(train_ref_path))),
        dtype=object,
    )
    train_smiles = get_unique_smiles(train_ref)

    print(f"MassBank spectra: {len(spectra)}")
    print(f"Train compounds: {len(train_smiles)}")

    # ----------------------------------------------------------------
    # Split by instrument
    # ----------------------------------------------------------------
    print("=" * 80)
    print("Split by instrument")
    print("=" * 80)

    instrument2spectra = defaultdict(list)

    for spectrum in spectra:
        instrument = spectrum.get("instrument")
        if instrument is None:
            instrument = spectrum.get("instrument_type")
        if instrument is None:
            continue

        instrument = instrument.lower()

        if "orbitrap" in instrument:
            instrument2spectra["Orbitrap"].append(spectrum)
        elif (
            "qtof" in instrument
            or "q-tof" in instrument
            or "tof" in instrument
        ):
            instrument2spectra["QTOF"].append(spectrum)

    instrument2spectra["all"] = (
        instrument2spectra["Orbitrap"] + instrument2spectra["QTOF"]
    )

    print(
        "MassBank spectra by instrument:"
        f" Orbitrap: {len(instrument2spectra['Orbitrap'])}"
        f" QTOF: {len(instrument2spectra['QTOF'])}"
        f" All: {len(instrument2spectra['all'])}"
    )
    print()

    for name, spectra in instrument2spectra.items():
        print("=" * 80)
        print(name)
        print("=" * 80)

        spectra = np.asarray(spectra, dtype=object)
        smiles = get_unique_smiles(spectra)

        print(f"Original compounds: {len(smiles)}")

        available_smiles = list(smiles - train_smiles)
        print(f"Remaining compounds: {len(available_smiles)}")

        for i in range(replica_num):
            query, reference = get_ref_query(
                spectra=spectra,
                target_smiles=available_smiles,
                seed=i,
            )

            np.save(
                output_dir
                / f"{name}-query{replica_suffix.format(i + 1)}.npy",
                query,
            )
            np.save(
                output_dir
                / f"{name}-reference{replica_suffix.format(i + 1)}.npy",
                reference,
            )

        print(f"Saved {replica_num} replicas.")

    print()
    print("=" * 80)
    print("MassBank processing finished.")
    print("=" * 80)


# ---------------------------------------------------------------------
# MassSpecGym dataset processing
# ---------------------------------------------------------------------

def process_massspecgym(
    root_dir,
    train_ref_path,
    replica_num=10,
    replica_suffix="-replication-{}",
):
    """
    Randomly split MassSpecGym into query/reference sets.

    Parameters
    ----------
    root_dir : str or Path
        Directory containing MassSpecGym_dedup.npy.

    train_ref_path : str or Path
        GNPS Orbitrap train_ref MGF file.

    replica_num : int
        Number of query/reference replicas to generate.

    replica_suffix : str
        Suffix template for replica files.
    """
    root_dir = Path(root_dir)
    train_ref_path = Path(train_ref_path)

    print("=" * 80)
    print("Processing MassSpecGym")
    print("=" * 80)

    # ----------------------------------------------------------------
    # Load train reference
    # ----------------------------------------------------------------
    train_ref = np.array(
        list(load_from_mgf(str(train_ref_path))),
        dtype=object,
    )
    train_smiles = get_unique_smiles(train_ref)

    print(f"Train compounds: {len(train_smiles)}")

    # ----------------------------------------------------------------
    # Load MassSpecGym
    # ----------------------------------------------------------------
    spectra = np.load(
        root_dir / "MassSpecGym_dedup.npy",
        allow_pickle=True,
    )
    print(f"Total spectra: {len(spectra)}")

    instrument2spectra = defaultdict(list)

    for spectrum in tqdm(spectra, desc="Grouping by instrument"):
        instrument2spectra["all"].append(spectrum)

        instrument = spectrum.get("instrument_type")
        if instrument is None:
            continue
        if pd.isna(instrument):
            continue

        instrument = str(instrument).strip()
        if instrument == "" or instrument.lower() == "nan":
            continue

        instrument2spectra[instrument].append(spectrum)

    # ----------------------------------------------------------------
    # Random split
    # ----------------------------------------------------------------
    for instrument, spectra in instrument2spectra.items():
        print()
        print("=" * 80)
        print(instrument)
        print("=" * 80)

        spectra = np.asarray(spectra, dtype=object)
        smiles = get_unique_smiles(spectra)

        print(f"Original compounds: {len(smiles)}")

        available_smiles = list(smiles - train_smiles)
        print(f"After exclusion: {len(available_smiles)}")

        for i in range(replica_num):
            query, reference = get_ref_query(
                spectra,
                available_smiles,
                seed=i,
            )

            np.save(
                root_dir
                / f"{instrument}-query"
                f"{replica_suffix.format(i + 1)}.npy",
                query,
            )
            np.save(
                root_dir
                / f"{instrument}-reference"
                f"{replica_suffix.format(i + 1)}.npy",
                reference,
            )

        print(f"Saved {replica_num} replicas.")

    print()
    print("=" * 80)
    print("MassSpecGym processing finished.")
    print("=" * 80)


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # MSBERT: GNPS + MoNA
    process_msbert(
        gnps_root=(
            "../data/MSBert_dedup/GNPS"
        ),
        mona_root=(
            "../data/MSBert_dedup/MoNA"
        ),
    )

    # MassBank
    process_massbank(
        massbank_path=(
            "../data/MassBank_dedup/MassBank_dedup.npy"
        ),
        train_ref_path=(
            "../data/MSBert_dedup/GNPS/Orbitrap_train_ref_cleaned.mgf"
        ),
        output_dir=(
            "../data/MassBank_dedup"
        ),
    )

    # MassSpecGym
    process_massspecgym(
        root_dir=(
            "../data/massSpecGymData_dedup"
        ),
        train_ref_path=(
            "../data/MSBert_dedup/GNPS/Orbitrap_train_ref_cleaned.mgf"
        ),
    )