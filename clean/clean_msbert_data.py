"""
Clean MSBERT dataset.

Load raw GNPS and MoNA spectra, filter out spectra without valid
chemical annotations, and save the cleaned spectra back to disk.
"""

import sys

sys.path.append("..")

from matchms.importing import load_from_mgf, load_from_msp
from matchms.exporting import save_as_mgf, save_as_msp

from SpecRoFormer.utils.clean import count_annotations, is_annotated


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

GNPS_ROOT = "../data/MSBert/GNPS"
MONA_ROOT = "../data/MSBert/MoNA"


# ---------------------------------------------------------------------
# Load raw GNPS spectra
# ---------------------------------------------------------------------

gnps_orbitrap_train_ref = list(
    load_from_mgf(f"{GNPS_ROOT}/Orbitrap_train_ref.mgf")
)
gnps_orbitrap_train_query = list(
    load_from_mgf(f"{GNPS_ROOT}/Orbitrap_train_query.mgf")
)
gnps_orbitrap_test_query = list(
    load_from_mgf(f"{GNPS_ROOT}/Orbitrap_test_query.mgf")
)
gnps_orbitrap_test_ref = list(
    load_from_mgf(f"{GNPS_ROOT}/Orbitrap_test_ref.mgf")
)

gnps_qtof = list(load_from_mgf(f"{GNPS_ROOT}/QTOF.mgf"))
gnps_other = list(load_from_mgf(f"{GNPS_ROOT}/Other.mgf"))


# ---------------------------------------------------------------------
# Load raw MoNA spectra
# ---------------------------------------------------------------------

mona_orbitrap = list(load_from_msp(f"{MONA_ROOT}/MoNA_Orbitrap.msp"))
mona_qtof = list(load_from_msp(f"{MONA_ROOT}/MoNA_QTOF.msp"))


# ---------------------------------------------------------------------
# Annotation statistics before filtering
# ---------------------------------------------------------------------

gnps_orbitrap = (
    gnps_orbitrap_train_ref
    + gnps_orbitrap_train_query
    + gnps_orbitrap_test_query
    + gnps_orbitrap_test_ref
)

count_annotations(gnps_orbitrap, "gnps orbitrap")
count_annotations(gnps_qtof, "gnps qtof")
count_annotations(gnps_other, "gnps other")
count_annotations(mona_orbitrap, "mona orbitrap")
count_annotations(mona_qtof, "mona qtof")


# ---------------------------------------------------------------------
# Filter out spectra without valid chemical annotations
# ---------------------------------------------------------------------

gnps_orbitrap_train_ref = is_annotated(gnps_orbitrap_train_ref)
gnps_orbitrap_train_query = is_annotated(gnps_orbitrap_train_query)
gnps_orbitrap_test_query = is_annotated(gnps_orbitrap_test_query)
gnps_orbitrap_test_ref = is_annotated(gnps_orbitrap_test_ref)

gnps_qtof = is_annotated(gnps_qtof)
gnps_other = is_annotated(gnps_other)

mona_orbitrap = is_annotated(mona_orbitrap)
mona_qtof = is_annotated(mona_qtof)


# ---------------------------------------------------------------------
# Annotation statistics after filtering
# ---------------------------------------------------------------------

gnps_orbitrap = (
    gnps_orbitrap_train_ref
    + gnps_orbitrap_train_query
    + gnps_orbitrap_test_query
    + gnps_orbitrap_test_ref
)

count_annotations(gnps_orbitrap, "gnps orbitrap")
count_annotations(gnps_qtof, "gnps qtof")
count_annotations(gnps_other, "gnps other")
count_annotations(mona_orbitrap, "mona orbitrap")
count_annotations(mona_qtof, "mona qtof")
# Note: only MoNA QTOF contains invalid records; after cleaning its
# size drops from 15834 to 7917.


# ---------------------------------------------------------------------
# Save cleaned spectra
# ---------------------------------------------------------------------

save_as_mgf(
    gnps_orbitrap_train_ref,
    f"{GNPS_ROOT}/Orbitrap_train_ref_cleaned.mgf",
)
save_as_mgf(
    gnps_orbitrap_train_query,
    f"{GNPS_ROOT}/Orbitrap_train_query_cleaned.mgf",
)
save_as_mgf(
    gnps_orbitrap_test_query,
    f"{GNPS_ROOT}/Orbitrap_test_query_cleaned.mgf",
)
save_as_mgf(
    gnps_orbitrap_test_ref,
    f"{GNPS_ROOT}/Orbitrap_test_ref_cleaned.mgf",
)

save_as_mgf(gnps_qtof, f"{GNPS_ROOT}/QTOF_cleaned.mgf")
save_as_mgf(gnps_other, f"{GNPS_ROOT}/Other_cleaned.mgf")

save_as_msp(mona_orbitrap, f"{MONA_ROOT}/MoNA_Orbitrap_cleaned.msp")
save_as_msp(mona_qtof, f"{MONA_ROOT}/MoNA_QTOF_cleaned.msp")