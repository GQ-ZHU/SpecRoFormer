"""
Process MSBERT dataset.

Merge cleaned MSBERT spectra, split MoNA into common/unique parts
relative to GNPS, build GNPS query/reference sets, and compute the
Tanimoto score matrix used for training.
"""

import sys

sys.path.append("..")

import numpy as np
import pandas as pd
from tqdm import tqdm
from matchms.importing import load_from_mgf, load_from_msp
from matchms.filtering.metadata_processing.add_fingerprint import (
    add_fingerprint,
)
from matchms.similarity import FingerprintSimilarity

from SpecRoFormer.const import gnps, mona
from SpecRoFormer.utils.clean import (
    count_annotations,
    get_ref_query,
    get_unique_smiles,
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

GNPS_ROOT = "../data/MSBert/GNPS"
MONA_ROOT = "../data/MSBert/MoNA"


# ---------------------------------------------------------------------
# 1. Merge cleaned GNPS Orbitrap spectra into a single array
# ---------------------------------------------------------------------

gnps_orbitrap_train_ref = list(
    load_from_mgf(f"{GNPS_ROOT}/Orbitrap_train_ref_cleaned.mgf")
)
gnps_orbitrap_train_query = list(
    load_from_mgf(f"{GNPS_ROOT}/Orbitrap_train_query_cleaned.mgf")
)
gnps_orbitrap_test_query = list(
    load_from_mgf(f"{GNPS_ROOT}/Orbitrap_test_query_cleaned.mgf")
)
gnps_orbitrap_test_ref = list(
    load_from_mgf(f"{GNPS_ROOT}/Orbitrap_test_ref_cleaned.mgf")
)

gnps_orbitrap = (
    gnps_orbitrap_train_ref
    + gnps_orbitrap_train_query
    + gnps_orbitrap_test_query
    + gnps_orbitrap_test_ref
)

np.save(f"{GNPS_ROOT}/Orbitrap/all.npy", gnps_orbitrap)


# ---------------------------------------------------------------------
# 2. Merge cleaned GNPS QTOF / Other spectra
# ---------------------------------------------------------------------

gnps_qtof = list(load_from_mgf(f"{GNPS_ROOT}/QTOF_cleaned.mgf"))
gnps_other = list(load_from_mgf(f"{GNPS_ROOT}/Other_cleaned.mgf"))

np.save(f"{GNPS_ROOT}/QTOF/all.npy", gnps_qtof)
np.save(f"{GNPS_ROOT}/Other/all.npy", gnps_other)


# ---------------------------------------------------------------------
# 3. Load cleaned MoNA spectra
# ---------------------------------------------------------------------

mona_orbitrap = list(
    load_from_msp(f"{MONA_ROOT}/MoNA_Orbitrap_cleaned.msp")
)
mona_qtof = list(load_from_msp(f"{MONA_ROOT}/MoNA_QTOF_cleaned.msp"))

count_annotations(gnps_orbitrap, "gnps orbitrap")
count_annotations(gnps_qtof, "gnps qtof")
count_annotations(gnps_other, "gnps other")
count_annotations(mona_orbitrap, "mona orbitrap")
count_annotations(mona_qtof, "mona qtof")


# ---------------------------------------------------------------------
# 4. Helper: extract unique SMILES from a list of spectra
# ---------------------------------------------------------------------

def get_compounds(spectra):
    """Return the unique SMILES set of a spectrum list."""
    smiles_seq = [s.get("smiles") for s in spectra]
    return np.unique(smiles_seq)


gnps_orbitrap_compounds = get_compounds(gnps_orbitrap)
gnps_qtof_compounds = get_compounds(gnps_qtof)
mona_orbitrap_compounds = get_compounds(mona_orbitrap)
mona_qtof_compounds = get_compounds(mona_qtof)

print(
    len(gnps_orbitrap_compounds),
    len(gnps_qtof_compounds),
    len(mona_orbitrap_compounds),
    len(mona_qtof_compounds),
)


# ---------------------------------------------------------------------
# 5. Split a dataset into "common" and "unique" parts
# ---------------------------------------------------------------------

def get_common_unique(spectra, target_common_compounds):
    """
    Split spectra into two groups based on whether their SMILES is in
    ``target_common_compounds``.

    Returns
    -------
    common_spectra : list
    common_compounds : np.ndarray
    unique_spectra : list
    unique_compounds : np.ndarray
    """
    common_spectra = []
    unique_spectra = []
    common_compounds = []
    unique_compounds = []

    for s in spectra:
        compound_name = s.get("smiles")
        if (
            compound_name is not None
            and compound_name in target_common_compounds
        ):
            common_spectra.append(s)
            common_compounds.append(compound_name)
        else:
            unique_spectra.append(s)
            unique_compounds.append(compound_name)

    return (
        common_spectra,
        np.unique(common_compounds),
        unique_spectra,
        np.unique(unique_compounds),
    )


# Split MoNA Orbitrap into: same compounds as GNPS Orbitrap / unique ones
(
    orbitrap_common_spectra,
    orbitrap_common_compounds,
    orbitrap_unique_spectra,
    orbitrap_unique_compounds,
) = get_common_unique(mona_orbitrap, gnps_orbitrap_compounds)

# Split MoNA QTOF into: same compounds as GNPS QTOF / unique ones
(
    qtof_common_spectra,
    qtof_common_compounds,
    qtof_unique_spectra,
    qtof_unique_compounds,
) = get_common_unique(mona_qtof, gnps_qtof_compounds)

np.save(mona.ORBITRAP_COMMON, orbitrap_common_spectra)
np.save(mona.ORBITRAP_UNIQUE, orbitrap_unique_spectra)
np.save(mona.QTOF_COMMON, qtof_common_spectra)
np.save(mona.QTOF_UNIQUE, qtof_unique_spectra)


# ---------------------------------------------------------------------
# 6. Save GNPS Orbitrap train / test splits
# ---------------------------------------------------------------------

np.save(gnps.ORBITRAP_TRAIN_QUERY, gnps_orbitrap_train_query)
np.save(gnps.ORBITRAP_TRAIN_REF, gnps_orbitrap_train_ref)
np.save(gnps.ORBITRAP_TEST_QUERY, gnps_orbitrap_test_query)
np.save(gnps.ORBITRAP_TEST_REF, gnps_orbitrap_test_ref)


# ---------------------------------------------------------------------
# 7. Split GNPS QTOF and Other into query/reference sets
# ---------------------------------------------------------------------

gnps_qtof_query, gnps_qtof_ref = get_ref_query(
    gnps_qtof,
    get_unique_smiles(gnps_qtof),
)
gnps_other_query, gnps_other_ref = get_ref_query(
    gnps_other,
    get_unique_smiles(gnps_other),
)

np.save(gnps.QTOF_TEST_QUERY, gnps_qtof_query)
np.save(gnps.QTOF_TEST_REF, gnps_qtof_ref)
np.save(gnps.OTHER_TEST_QUERY, gnps_other_query)
np.save(gnps.OTHER_TEST_REF, gnps_other_ref)


# ---------------------------------------------------------------------
# 8. Compute unique SMILES over all GNPS + MoNA data
# ---------------------------------------------------------------------

all_data = (
    gnps_orbitrap
    + gnps_qtof
    + gnps_other
    + mona_orbitrap
    + mona_qtof
)
all_unique_smiles = get_unique_smiles(all_data)
np.save(gnps.DIR / gnps.UNIQUE_SMILES, all_unique_smiles)


# ---------------------------------------------------------------------
# 9. Build a representative spectrum per unique SMILES and compute
#    the Tanimoto score matrix over representatives
# ---------------------------------------------------------------------

smiles_seq = np.array([s.get("smiles") for s in all_data])

indices = []
for smiles in tqdm(all_unique_smiles):
    index = np.where(smiles_seq == smiles)[0][0]
    # The fingerprint will be added at the corresponding index later.
    indices.append(index)

# print(indices)

metadata = pd.DataFrame(
    list(zip(all_unique_smiles, indices)),
    columns=["smiles", "ID"],
)

# Add molecular fingerprints to one representative spectrum per SMILES
for i in tqdm(metadata["ID"].values):
    all_data[i] = add_fingerprint(
        all_data[i],
        fingerprint_type="daylight",
        nbits=2048,
    )

spectrums_represent = [all_data[i] for i in metadata["ID"].values]

# Compute the fingerprint similarity matrix
similarity_measure = FingerprintSimilarity(similarity_measure="jaccard")
scores = similarity_measure.matrix(
    spectrums_represent,
    spectrums_represent,
)
np.save(gnps.DIR / gnps.TANOMOTO_SCORE, scores)


# ---------------------------------------------------------------------
# 10. Rebuild MoNA query/reference splits after excluding GNPS train ref
# ---------------------------------------------------------------------

mona_orbitrap = list(
    load_from_msp(f"{MONA_ROOT}/MoNA_Orbitrap_cleaned.msp")
)
mona_qtof = list(load_from_msp(f"{MONA_ROOT}/MoNA_QTOF_cleaned.msp"))

train_ref_spectra = np.load(
    f"{GNPS_ROOT}/Orbitrap/train_ref.npy",
    allow_pickle=True,
)
train_ref_smiles = get_unique_smiles(train_ref_spectra)

for dataset_name, spectra in [
    ("orbitrap", mona_orbitrap),
    ("qtof", mona_qtof),
]:
    smiles_seq = get_unique_smiles(spectra)
    print("Before exclusion:", len(smiles_seq))

    bool_indices = np.isin(smiles_seq, train_ref_smiles)
    excluded_smiles = smiles_seq[~bool_indices]
    print("After exclusion:", len(excluded_smiles))

    query_spectra, ref_spectra = get_ref_query(spectra, excluded_smiles)

    np.save(
        f"{MONA_ROOT}/{dataset_name}_query.npy",
        query_spectra,
    )
    np.save(
        f"{MONA_ROOT}/{dataset_name}_ref.npy",
        ref_spectra,
    )