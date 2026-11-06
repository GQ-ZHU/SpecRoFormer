"""
Clean MassBank dataset.

Load the raw NIST-formatted MassBank file, apply filters and metadata
cleaning, separate spectra by ion mode and instrument type, then
generate ten random query/reference splits for each instrument group.
"""

import sys

sys.path.append("..")

from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from matchms.importing import load_from_msp

from SpecRoFormer.utils.clean import (
    minimal_processing,
    is_annotated,
    filter_by_precursor_mz,
    apply_filters,
    count_annotations,
    clean_metadata,
    clean_metadata2,
    seperate_spectra_by_ionmode,
    get_ref_query,
    get_unique_smiles,
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

MASSBANK_PATH = (
    "../data/MassBank/MassBank.msp_NIST"
)
MASSBANK_DIR = Path(
    "../data/MassBank"
)

TRAIN_REF_PATH = (
    "../data/MSBert/GNPS/"
    "Orbitrap/train_ref.npy"
)


# ---------------------------------------------------------------------
# 1. Load and clean spectra
# ---------------------------------------------------------------------

spectra = list(load_from_msp(MASSBANK_PATH))

after_filter = [apply_filters(s) for s in tqdm(spectra)]
after_filter = [s for s in tqdm(after_filter) if s is not None]

cleaned = [clean_metadata(s) for s in tqdm(after_filter)]
cleaned = [clean_metadata2(s) for s in tqdm(cleaned)]

positive, negative = seperate_spectra_by_ionmode(cleaned)

positive = [minimal_processing(s) for s in tqdm(positive)]
positive = [s for s in positive if s is not None]

count_annotations(positive, "peak num >= 5")
positive = filter_by_precursor_mz(positive)
count_annotations(positive, "10 < precursor_mz < 1000")

annotated = is_annotated(positive)
count_annotations(annotated, "annotated")


# ---------------------------------------------------------------------
# 2. Group spectra by instrument type
# ---------------------------------------------------------------------

instrument2spectra = defaultdict(list)
unknown_instruments = []

for s in tqdm(annotated):
    instrument = s.metadata.get("instrument", None)
    if instrument is None:
        instrument = s.metadata.get("instrument_type")

    if instrument is None:
        continue

    if "orbitrap" in instrument.lower():
        instrument2spectra["Orbitrap"].append(s)
    elif (
        "qtof" in instrument.lower()
        or "q-tof" in instrument.lower()
        or "tof" in instrument.lower()
    ):
        instrument2spectra["QTOF"].append(s)
    else:
        unknown_instruments.append(instrument)

instrument2spectra["all"] = (
    instrument2spectra["Orbitrap"] + instrument2spectra["QTOF"]
)
instrument2spectra = dict(instrument2spectra)


# ---------------------------------------------------------------------
# 3. Build 10 random query/reference splits for each instrument group
# ---------------------------------------------------------------------

MASSBANK_DIR.mkdir(parents=True, exist_ok=True)
replica_suffix = "-replication-{}"

train_ref_spectra = np.load(TRAIN_REF_PATH, allow_pickle=True)
train_ref_smiles = get_unique_smiles(train_ref_spectra)

for instrument_type, spectra in instrument2spectra.items():
    print(instrument_type)

    spectra = np.asarray(spectra, dtype=object)

    smiles_seq = get_unique_smiles(spectra)
    print(len(smiles_seq))

    bool_indices = np.isin(smiles_seq, train_ref_smiles)
    excluded_smiles = smiles_seq[~bool_indices]
    print(len(excluded_smiles))

    for i in range(10):
        query, reference = get_ref_query(spectra, excluded_smiles)

        np.save(
            MASSBANK_DIR
            / (instrument_type + "-query" + replica_suffix.format(i + 1)),
            query,
        )
        np.save(
            MASSBANK_DIR
            / (
                instrument_type
                + "-reference"
                + replica_suffix.format(i + 1)
            ),
            reference,
        )