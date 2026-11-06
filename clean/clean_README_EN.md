# Data Cleaning and Preprocessing
This directory contains all scripts used to clean, filter, deduplicate, and split the raw MS/MS datasets into query/reference sets that are consumed by the training and evaluation pipeline of SpecRoFormer.

---
## Pipeline overview
The pipeline starts from raw public datasets (GNPS, MoNA, MassBank, MassSpecGym, MTBLS1572, in‑silico pretrain data) and produces `.npy` spectrum files organized by dataset / instrument / split.

The table below lists each stage, the script responsible for it, and the main artifacts it produces.

| Stage | Script | Main artifacts |
|---|---|---|
| 1. Raw filtering | `clean_msbert_data.py` | `*_cleaned.mgf`, `*_cleaned.msp` |
| 1. Raw filtering | `MassBank‑Clean.py` | MassBank query / reference `.npy` (10 replicas) |
| 1. Raw filtering | `MassSpecGymData‑Clean.py` | MassSpecGym query / reference `.npy` (10 replicas) |
| 1. Raw filtering | `clean_pretrain_dataset.py` | Pretrain dataset `.npy` |
| 2. Merge & split | `process_msbert_data.py` | `all.npy`, `train_*.npy`, `test_*.npy`, `tanimoto_score.npy` |
| 3. Deduplication | `deduplicate_by_inchikey.py` | `*_dedup.npy`, `deduplication_report.xlsx`, `removed_compounds.xlsx` |
| 4. Rebuild splits | `random_split_query_ref.py` | Rebuilt query / reference `.npy` (10 replicas) |
| 4. Rebuild splits | `random_split_query_ref_many_times.py` | GNPS test query / reference `.npy` (10 replicas) |

**Dependency chain.**
- Stage 1 → Stage 2: cleaned files feed the merge step.
- Stage 2 → Stage 3: merged arrays are the input of deduplication.
- Stage 3 → Stage 4: deduplicated files are the input of the split rebuilding.
- Stage 4 has two alternative scripts; use either one depending on whether you need to re‑run the MassBank cleaning pipeline.

---
## Files
### 1. `clean_msbert_data.py`
Load raw GNPS Orbitrap / QTOF / Other MGF files and raw MoNA Orbitrap / QTOF MSP files, then remove spectra that do not carry a valid chemical annotation (`smiles`, `inchikey`, etc.). Cleaned spectra are written back to disk as `*_cleaned.mgf` / `*_cleaned.msp`.

- **Input**: raw `GNPS/*.mgf`, `MoNA/*.msp`
- **Output**: `*_cleaned.mgf`, `*_cleaned.msp`
- **Key function**: `SpecRoFormer.utils.clean.is_annotated`
- **Notes**: only MoNA QTOF loses a large fraction of records (15834 → 7917) after filtering.

---
### 2. `process_msbert_data.py`
Merge cleaned MSBERT spectra into single `.npy` arrays per dataset, split MoNA into "common" / "unique" parts relative to GNPS, build the GNPS query/reference sets, and compute the Tanimoto score matrix between one representative spectrum per unique SMILES. Finally, rebuild MoNA query/reference splits while excluding compounds that already appear in the GNPS Orbitrap training reference set.

- **Input**: `*_cleaned.mgf`, `*_cleaned.msp`
- **Output**:
  - `GNPS/Orbitrap/all.npy`
  - `GNPS/QTOF/all.npy`
  - `GNPS/Other/all.npy`
  - `MoNA/{orbitrap,qtof}_{query,ref}.npy`
  - `MoNA/orbitrap_common.npy`, `MoNA/qtof_unique.npy`, ...
  - `GNPS/unique_smiles.npy`
  - `GNPS/tanimoto_score.npy`
- **Key functions**:
  - `SpecRoFormer.utils.clean.get_ref_query`
  - `SpecRoFormer.utils.clean.get_unique_smiles`
  - `matchms.similarity.FingerprintSimilarity`

---
### 3. `MassBank‑Clean.py`
Load the raw MassBank NIST file, apply the full MassBank cleaning pipeline (filters, metadata cleaning, ion mode separation, peak count and precursor m/z filtering, annotation check), group spectra by instrument type (`Orbitrap`, `QTOF`, `all`), and generate ten random query/reference replicas per group. Compounds present in the GNPS Orbitrap training reference set are excluded.

- **Input**: `MassBank/MassBank.msp_NIST`
- **Output**:
  - `MassBank/{Orbitrap,QTOF,all}-query‑replication‑{1..10}.npy`
  - `MassBank/{Orbitrap,QTOF,all}-reference‑replication‑{1..10}.npy`
- **Depends on**: `GNPS/Orbitrap/train_ref.npy`

---
### 4. `MassSpecGymData‑Clean.py`
Load the raw MassSpecGym TSV, convert each row into a matchms `Spectrum`, group spectra by instrument type (`all`, `Orbitrap`, `QTOF`, ...), and generate ten random query/reference replicas per group while excluding compounds in the GNPS Orbitrap training reference set.

- **Input**: `massSpecGymData/MassSpecGym.tsv`
- **Output**:
  - `massSpecGymData/{split}-query‑replication‑{1..10}.npy`
  - `massSpecGymData/{split}-reference‑replication‑{1..10}.npy`
- **Depends on**: `GNPS/Orbitrap/train_ref.npy`

---
### 5. `clean_pretrain_dataset.py`
Clean the in‑silico pretrain dataset (e.g. CFM‑ID generated MGF). For each spectrum, compute precursor m/z from SMILES using a robust parser that tolerates non‑standard SMILES produced by in‑silico tools, set `charge`, `ionmode`, and `precursor_type`, then filter out spectra whose SMILES cannot be parsed or whose peak list is empty. Cleaned spectra are stored per dataset as `.npy`.

- **Input**: `/mnt/.../Ms2Mol/data/pretrain/{dataset_name}.mgf`
- **Output**: one `.npy` per dataset, path defined by `SpecRoFormer.const.pretrain_data.PRETRAIN_DATASETS`
- **Key functions**:
  - `clean_cfm_smiles`
  - `smiles_to_precursor_mz_robust`
  - `is_annotated_virtual`

---
### 6. `deduplicate_by_inchikey.py`
Strict deduplication of evaluation datasets against a training dataset using the first block of the InChIKey (connectivity block). Supported input formats:

| Extension | Loader | Notes |
|--------------------|--------------------------------------|-------------------------------------------|
| `.npy`             | `np.load(allow_pickle=True)`         | spectrum objects or dict metadata         |
| `.mgf`             | `matchms.load_from_mgf`              |                                           |
| `.msp`             | `matchms.load_from_msp`              | runs the full MassBank cleaning first     |
| `.msp_nist`        | `matchms.load_from_msp`              | same as `.msp`                            |
| `.tsv`             | `pandas.read_csv(sep="\t")`          | MassSpecGym‑style, builds `Spectrum`      |

- **Input**: training reference `.npy` + a list of dataset paths
- **Output**:
  - `<dataset>_dedup.npy`
  - `deduplication_report.xlsx`
  - `removed_compounds.xlsx`
- **Key functions**:
  - `normalize_inchikey`
  - `smiles_to_inchikey`
  - `build_train_set`
  - `dedup_npy`, `dedup_mgf`, `dedup_msp`, `dedup_tsv`

`__main__` currently has the dataset list commented out on purpose; uncomment only the entries you want to process.

---
### 7. `random_split_query_ref.py`
Unified entry point for rebuilding the query/reference splits after deduplication. Provides three helpers:

- `process_msbert(gnps_root, mona_root)`
  Rebuild GNPS Orbitrap / QTOF / Other and MoNA Orbitrap / QTOF splits. Generates the "all" file per dataset and ten replicas for the GNPS test groups.

- `process_massbank(massbank_path, train_ref_path, output_dir)`
  Rebuild MassBank query/reference splits per instrument, ten replicas, excluding compounds in the training reference set.

- `process_massspecgym(root_dir, train_ref_path)`
  Rebuild MassSpecGym query/reference splits per instrument, ten replicas, excluding compounds in the training reference set.

- **Inputs**: `*_dedup.npy` produced by `deduplicate_by_inchikey.py`
- **Outputs**: `*_query‑replication‑{1..10}.npy` and `*_reference‑replication‑{1..10}.npy`
- **Key functions**:
  - `SpecRoFormer.utils.clean.get_ref_query`
  - `SpecRoFormer.utils.clean.get_unique_smiles`

`__main__` runs all three helpers in sequence with hardcoded paths. Adjust the paths to match your environment.

---
### 8. `random_split_query_ref_many_times.py`
Generate ten random query/reference splits for the GNPS test datasets (Orbitrap / QTOF / Other) while excluding compounds that already appear in the GNPS Orbitrap training reference set.

This script is complementary to `random_split_query_ref.py`: it works directly from the `.npy` arrays produced by `process_msbert_data.py`, without re‑running deduplication.

- **Inputs**: `GNPS/Orbitrap/{train,test}_*.npy`, `GNPS/QTOF/*.npy`, `GNPS/Other/*.npy`
- **Outputs**:
  - `GNPS/Orbitrap/test_query‑replication‑{1..10}.npy`
  - `GNPS/Orbitrap/test_ref‑replication‑{1..10}.npy`
  - `GNPS/QTOF/test_query‑replication‑{1..10}.npy`
  - `GNPS/QTOF/test_ref‑replication‑{1..10}.npy`
  - `GNPS/Other/test_query‑replication‑{1..10}.npy`
  - `GNPS/Other/test_ref‑replication‑{1..10}.npy`

---
## Recommended execution order
Below is the recommended order for a full pipeline run, starting from raw data. Steps that can be skipped if the corresponding intermediate files already exist are marked as "(optional)".

1. Raw filtering
```bash
python clean_msbert_data.py        # GNPS + MoNA raw filtering
python MassBank‑Clean.py           # MassBank raw cleaning
python MassSpecGymData‑Clean.py    # MassSpecGym raw cleaning
python clean_pretrain_dataset.py   # in‑silico pretrain cleaning
```

2. Merge and split GNPS + MoNA
```bash
python process_msbert_data.py
```

3. Strict deduplication
```bash
python deduplicate_by_inchikey.py
```

4. Rebuild splits after deduplication
```bash
python random_split_query_ref.py
# OR (only if you need to re‑split GNPS test sets without re‑dedup)
# python random_split_query_ref_many_times.py
```

### Notes on dependencies
- Steps 1 and 2 depend on `SpecRoFormer.utils.clean`. Make sure the package is importable (`sys.path.append("..")` is already set in each script).
- Step 3 (`deduplicate_by_inchikey.py`) produces `*_dedup.npy` files that step 4 consumes. If you skip step 3, step 4 must be run on the non‑dedup files by editing the paths in the script.
- `random_split_query_ref_many_times.py` is an alternative to `random_split_query_ref.py` that does not re‑run the MassBank cleaning pipeline; use it if you want a fast re‑split of the GNPS test sets.
- All output paths are hardcoded to the original project layout. Change them to match your local environment before running.

---
## Datasets used in this directory
| Dataset | Format | Source files |
|------------------|---------------|--------------------------------------------------|
| GNPS             | mgf           | `Orbitrap_*.mgf`, `QTOF.mgf`, `Other.mgf`        |
| MoNA             | msp           | `MoNA_Orbitrap.msp`, `MoNA_QTOF.msp`             |
| MassBank         | msp_nist      | `MassBank.msp_NIST`                              |
| MassSpecGym      | tsv           | `MassSpecGym.tsv`                                |
| MTBLS1572        | msp           | `q.msp`, `r.msp`                                 |
| Pretrain         | mgf           | `{dataset_name}.mgf` (in‑silico, CFM‑ID style)   |

Raw data files are not tracked in git. See `data/` for per‑dataset download links and folder structure.
