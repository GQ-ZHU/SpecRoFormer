"""
Clean MassSpecGym dataset.

Load the raw MassSpecGym TSV, convert each row into a matchms Spectrum,
group spectra by instrument type, and generate ten random
query/reference splits for each group.
"""

import sys

sys.path.append("..")

from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from matchms import Spectrum

from SpecRoFormer.utils.clean import get_ref_query, get_unique_smiles


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

MASS_SPEC_GYM_PATH = (
    "../data/"
    "massSpecGymData/MassSpecGym.tsv"
)
OUTPUT_DIR = Path(
    "../data/massSpecGymData"
)

TRAIN_REF_PATH = (
    "../data/MSBert/GNPS/"
    "Orbitrap/train_ref.npy"
)


# ---------------------------------------------------------------------
# 1. Load raw TSV
# ---------------------------------------------------------------------

df = pd.read_csv(MASS_SPEC_GYM_PATH, sep="\t")

key2index = {
    "mz": 1,
    "intensity": 2,
    "smiles": 3,
    "inchikey": 4,
    "precursor_mz": 8,
    "adduct": 9,
    "instrument_type": 10,
}

instrument2spectra = defaultdict(list)

for row in tqdm(df.values):
    mz = np.array(row[key2index["mz"]].split(","), dtype=np.float32)
    intensity = np.array(
        row[key2index["intensity"]].split(","),
        dtype=np.float32,
    )
    smiles = row[key2index["smiles"]]
    inchikey = row[key2index["inchikey"]]
    precursor_mz = row[key2index["precursor_mz"]]
    adduct = row[key2index["adduct"]]
    instrument_type = row[key2index["instrument_type"]]

    spectrum = Spectrum(
        mz=mz,
        intensities=intensity,
        metadata={
            "smiles": smiles,
            "inchikey": inchikey,
            "precursor_mz": precursor_mz,
            "adduct": adduct,
        },
    )

    instrument2spectra["all"].append(spectrum)

    if pd.isna(instrument_type):
        continue

    instrument2spectra[instrument_type].append(spectrum)

instrument2spectra = dict(instrument2spectra)


# ---------------------------------------------------------------------
# 2. Split each instrument group into query/reference sets
# ---------------------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
replica_suffix = "-replication-{}"

train_ref_spectra = np.load(TRAIN_REF_PATH, allow_pickle=True)
train_ref_smiles = get_unique_smiles(train_ref_spectra)

for instrument_type, spectra in instrument2spectra.items():
    spectra = np.asarray(spectra, dtype=object)

    smiles_seq = get_unique_smiles(spectra)
    print(len(smiles_seq))

    bool_indices = np.isin(smiles_seq, train_ref_smiles)
    excluded_smiles = smiles_seq[~bool_indices]
    print(len(excluded_smiles))

    for i in range(10):
        query, reference = get_ref_query(spectra, excluded_smiles)

        np.save(
            OUTPUT_DIR
            / (instrument_type + "-query" + replica_suffix.format(i + 1)),
            query,
        )
        np.save(
            OUTPUT_DIR
            / (
                instrument_type
                + "-reference"
                + replica_suffix.format(i + 1)
            ),
            reference,
        )