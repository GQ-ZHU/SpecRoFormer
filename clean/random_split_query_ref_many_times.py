"""
Generate ten random query/reference splits for GNPS test datasets.

Each split excludes compounds that already appear in the GNPS
Orbitrap training reference set, so that test compounds do not overlap
with the training set.
"""

import sys

sys.path.append("..")

import numpy as np

from SpecRoFormer.const import gnps
from SpecRoFormer.utils.clean import get_ref_query, get_unique_smiles


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

TRAIN_REF_PATH = (
    "../data/MSBert/GNPS/"
    "Orbitrap/train_ref.npy"
)


# ---------------------------------------------------------------------
# Collect paths for GNPS Orbitrap / QTOF / Other test splits
# ---------------------------------------------------------------------

spectra_paths = {
    "gnps": {
        "orbitrap": {
            "train": (gnps.ORBITRAP_TRAIN_QUERY, gnps.ORBITRAP_TRAIN_REF),
            "test": (gnps.ORBITRAP_TEST_QUERY, gnps.ORBITRAP_TEST_REF),
        },
        "qtof": {
            "test": (gnps.QTOF_TEST_QUERY, gnps.QTOF_TEST_REF),
        },
        "other": {
            "test": (gnps.OTHER_TEST_QUERY, gnps.OTHER_TEST_REF),
        },
    },
}

(
    gnps_orbitrap_train_query_path,
    gnps_orbitrap_train_ref_path,
) = spectra_paths["gnps"]["orbitrap"]["train"]
(
    gnps_orbitrap_test_query_path,
    gnps_orbitrap_test_ref_path,
) = spectra_paths["gnps"]["orbitrap"]["test"]

gnps_qtof_test_query_path, gnps_qtof_test_ref_path = spectra_paths[
    "gnps"
]["qtof"]["test"]

gnps_other_test_query_path, gnps_other_test_ref_path = spectra_paths[
    "gnps"
]["other"]["test"]


# ---------------------------------------------------------------------
# Load GNPS Orbitrap train / test spectra
# ---------------------------------------------------------------------

gnps_orbitrap_train_query_spectra = np.load(
    gnps_orbitrap_train_query_path,
    allow_pickle=True,
)
gnps_orbitrap_train_ref_spectra = np.load(
    gnps_orbitrap_train_ref_path,
    allow_pickle=True,
)
gnps_orbitrap_test_query_spectra = np.load(
    gnps_orbitrap_test_query_path,
    allow_pickle=True,
)
gnps_orbitrap_test_ref_spectra = np.load(
    gnps_orbitrap_test_ref_path,
    allow_pickle=True,
)

gnps_qtof_test_query_spectra = np.load(
    gnps_qtof_test_query_path,
    allow_pickle=True,
)
gnps_qtof_test_ref_spectra = np.load(
    gnps_qtof_test_ref_path,
    allow_pickle=True,
)

gnps_other_test_query_spectra = np.load(
    gnps_other_test_query_path,
    allow_pickle=True,
)
gnps_other_test_ref_spectra = np.load(
    gnps_other_test_ref_path,
    allow_pickle=True,
)


# ---------------------------------------------------------------------
# Reassemble GNPS spectra
# ---------------------------------------------------------------------

gnps_orbitrap_train_spectra = np.concatenate(
    (
        gnps_orbitrap_train_query_spectra,
        gnps_orbitrap_train_ref_spectra,
    )
)

gnps_orbitrap_test_spectra = np.concatenate(
    (
        gnps_orbitrap_test_query_spectra,
        gnps_orbitrap_test_ref_spectra,
    )
)

gnps_orbitrap_spectra = np.concatenate(
    (
        gnps_orbitrap_train_spectra,
        gnps_orbitrap_test_spectra,
    )
)

gnps_qtof_spectra = np.concatenate(
    (
        gnps_qtof_test_query_spectra,
        gnps_qtof_test_ref_spectra,
    )
)

gnps_other_spectra = np.concatenate(
    (
        gnps_other_test_query_spectra,
        gnps_other_test_ref_spectra,
    )
)


# ---------------------------------------------------------------------
# Generate ten splits for GNPS Orbitrap / QTOF / Other test data
# ---------------------------------------------------------------------

replica_suffix = "-replication-{}"

spectra_seq = [
    gnps_orbitrap_test_spectra,
    gnps_qtof_spectra,
    gnps_other_spectra,
]

paths = [
    (gnps.ORBITRAP_TEST_QUERY, gnps.ORBITRAP_TEST_REF),
    (gnps.QTOF_TEST_QUERY, gnps.QTOF_TEST_REF),
    (gnps.OTHER_TEST_QUERY, gnps.OTHER_TEST_REF),
]

train_ref_spectra = np.load(TRAIN_REF_PATH, allow_pickle=True)
train_ref_smiles = get_unique_smiles(train_ref_spectra)

for i in range(10):
    for spectra, (query_path, ref_path) in zip(spectra_seq, paths):
        smiles_seq = get_unique_smiles(spectra)
        print("Before exclusion:", len(smiles_seq))

        bool_indices = np.isin(smiles_seq, train_ref_smiles)
        excluded_smiles = smiles_seq[~bool_indices]
        print("After exclusion:", len(excluded_smiles))

        query_spectra, ref_spectra = get_ref_query(
            spectra,
            excluded_smiles,
        )

        replica_query_path = query_path.with_stem(
            query_path.stem + replica_suffix.format(i + 1)
        )
        replica_ref_path = ref_path.with_stem(
            ref_path.stem + replica_suffix.format(i + 1)
        )

        np.save(replica_query_path, query_spectra)
        np.save(replica_ref_path, ref_spectra)