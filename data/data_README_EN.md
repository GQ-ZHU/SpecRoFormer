# Data
This directory holds all datasets used by SpecRoFormer for training and evaluation. Raw data files are **not** redistributed with the repository; only the directory skeleton and this README are kept.

---
## Overview
SpecRoFormer uses several public MS/MS datasets for training and cross‑dataset / cross‑instrument evaluation. Two levels of benchmarks are supported:

- **Processed benchmark (recommended for quick start):** preprocessed spectra, splits, and evaluation files, available on Zenodo. These files are **not** InChIKey‑deduplicated against the training set.
- **Strict benchmark (for unseen‑compound evaluation):** reconstructed from the raw sources released by SpecEmbedding, then passed through the cleaning pipeline in `clean/`, which performs strict InChIKey‑level deduplication.

| Dataset | Role | Source |
|---|---|---|
| GNPS (Orbitrap) | Training + evaluation | SpecEmbedding, figshare |
| GNPS (QTOF / Other) | Evaluation | SpecEmbedding, figshare |
| MoNA (Orbitrap / QTOF) | Evaluation | SpecEmbedding, figshare |
| MassBank (Orbitrap / QTOF / all) | Evaluation | SpecEmbedding, figshare |
| MassSpecGym (Orbitrap / QTOF / all) | Evaluation | SpecEmbedding, figshare |
| MTBLS1572 | Evaluation | SpecEmbedding, figshare |
| SimMS‑9M | Evaluation | CSU‑MS2, Zenodo |

---
## Download
### Option A — Processed benchmark (recommended)
Zenodo record:
‑ https://doi.org/10.5281/zenodo.19368733

The record provides several archives. Extract each archive into the corresponding directory under `data/`:

| Archive | Extract to |
|---|---|
| `GNPS.zip` | `data/MSBert/GNPS/` |
| `MoNA.zip` | `data/MSBert/MoNA/` |
| `MTBLS1572.zip` | `data/MSBert/MTBLS1572/` |
| `MassBank.zip` | `data/MassBank/` |
| `massSpecGymData.zip` | `data/massSpecGymData/` |
| SimMS‑9M archive | `data/PreTrained/simulated_dataset/` |

After extraction, each dataset directory should contain `.npy` spectrum files (see "Directory layout" below).

**Note.** These processed files are **not** InChIKey‑deduplicated against the training set. If you need strict unseen‑compound evaluation, use Option B.

---
### Option B — Rebuild from raw sources
Download the original data released by SpecEmbedding, place the raw files under `data/`, then run the cleaning pipeline in `clean/` to reproduce the splits and to perform strict InChIKey‑level deduplication.

Raw data sources:
‑ GNPS, MoNA, MassBank, MassSpecGym, MTBLS1572
  ‑ Paper: https://doi.org/10.1021/acs.analchem.5c02655
  ‑ Data: https://doi.org/10.6084/m9.figshare.28876751
‑ SimMS‑9M
  ‑ Paper: https://doi.org/10.1021/acs.analchem.5c01594
  ‑ Data: https://zenodo.org/records/17756840

Place the raw files as described in [`clean/README.md`](../clean/README.md), then run the pipeline in four stages:

1. **Raw filtering**
```bash
python clean_msbert_data.py            # GNPS + MoNA
python MassBank‑Clean.py               # MassBank
python MassSpecGymData‑Clean.py        # MassSpecGym
python clean_pretrain_dataset.py       # in‑silico (SimMS‑9M)
```

2. **Merge and split GNPS + MoNA**
```bash
python process_msbert_data.py
```

3. **Strict deduplication**
```bash
python deduplicate_by_inchikey.py
```

4. **Rebuild query / reference splits**
```bash
python random_split_query_ref.py
# OR, if you only need to re‑split the GNPS test sets without
# re‑running deduplication:
# python random_split_query_ref_many_times.py
```

See [`clean/README.md`](../clean/README.md) for full details on each script, inputs, outputs, and dependencies.

---
## Directory layout
After downloading (Option A) or rebuilding (Option B), the `data/` directory should look like this:

```text
data/
├── MSBert/
│   ├── GNPS/
│   │   ├── Orbitrap/
│   │   │   ├── all.npy
│   │   │   ├── train_ref.npy
│   │   │   ├── train_query.npy
│   │   │   ├── test_ref.npy
│   │   │   ├── test_query.npy
│   │   │   ├── test_ref‑replication‑{1..10}.npy
│   │   │   └── test_query‑replication‑{1..10}.npy
│   │   ├── QTOF/
│   │   │   ├── all.npy
│   │   │   ├── test_ref.npy
│   │   │   ├── test_query.npy
│   │   │   ├── test_ref‑replication‑{1..10}.npy
│   │   │   └── test_query‑replication‑{1..10}.npy
│   │   └── Other/
│   │       ├── all.npy
│   │       ├── test_ref.npy
│   │       ├── test_query.npy
│   │       ├── test_ref‑replication‑{1..10}.npy
│   │       └── test_query‑replication‑{1..10}.npy
│   ├── MoNA/
│   │   ├── orbitrap_query.npy
│   │   ├── orbitrap_ref.npy
│   │   ├── qtof_query.npy
│   │   └── qtof_ref.npy
│   └── MTBLS1572/
│       ├── query.npy
│       └── ref.npy
├── MassBank/
│   ├── Orbitrap‑query‑replication‑{1..10}.npy
│   ├── Orbitrap‑reference‑replication‑{1..10}.npy
│   ├── QTOF‑query‑replication‑{1..10}.npy
│   ├── QTOF‑reference‑replication‑{1..10}.npy
│   ├── all‑query‑replication‑{1..10}.npy
│   └── all‑reference‑replication‑{1..10}.npy
├── massSpecGymData/
│   ├── Orbitrap‑query‑replication‑{1..10}.npy
│   ├── Orbitrap‑reference‑replication‑{1..10}.npy
│   ├── QTOF‑query‑replication‑{1..10}.npy
│   ├── QTOF‑reference‑replication‑{1..10}.npy
│   ├── all‑query‑replication‑{1..10}.npy
│   └── all‑reference‑replication‑{1..10}.npy
└── PreTrained/
    └── simulated_dataset/           # SimMS‑9M
        ├── ...
```

Additional directories (`labels/`, `model/`) are created automatically when running training or evaluation, and are not part of this layout.

---
## Datasets
### GNPS
Large collection of MS/MS spectra from diverse compounds, acquired on Orbitrap, QTOF, and other instruments.
‑ **Training:** GNPS Orbitrap training split only.
‑ **Evaluation:** GNPS Orbitrap / QTOF / Other test splits.

### MoNA
MassBank of North America. Spectra measured on Orbitrap and QTOF instruments, covering a wide range of small molecules.
‑ **Evaluation only.**

### MassBank
International public repository of high‑quality MS/MS spectra.
‑ **Evaluation only.**

### MassSpecGym
Benchmark curated from Orbitrap and QTOF instruments (NeurIPS 2024 benchmark).
‑ **Evaluation only.**

### MTBLS1572
Small dataset of 134 endogenous metabolites acquired on a QTOF instrument with DDA.
‑ **Evaluation only.**

### SimMS‑9M
Large‑scale simulated dataset generated by CFM‑ID from ~3 million compounds under three collision energy settings (~9 million spectra). Also used as the in‑silico pretraining corpus.
‑ **Evaluation only.**

---
## Data processing
All datasets are processed through the scripts in `clean/`. The full pipeline, per‑script descriptions, and the dependency chain are documented in [`clean/README.md`](../clean/clean_README_EN.md).

The pipeline covers four stages:
1. Raw spectrum filtering and annotation checks.
2. Merging and splitting into query / reference sets.
3. Strict InChIKey‑based deduplication against the GNPS Orbitrap training reference.
4. Rebuilding the query / reference splits after deduplication.

---
## Benchmark protocol
‑ **Training set:** GNPS Orbitrap training subset only.
‑ **Evaluation sets:** GNPS test splits (Orbitrap / QTOF / Other), MoNA, MassBank, MassSpecGym, MTBLS1572, SimMS‑9M.
‑ **Splitting:** within each dataset‑instrument subset, spectra are split at the compound level into query and reference sets. When a compound has at least two spectra, one is randomly chosen as the query and the rest are assigned to the reference set; when only one spectrum is available, it is placed in the reference set only.
‑ **Replicas:** for large datasets (GNPS, MassBank, MassSpecGym), the split is repeated ten times. All ten replicas are released and evaluation results are reported as mean and standard deviation.
‑ **Strict benchmark:** compounds whose first‑block InChIKey overlaps with the GNPS Orbitrap training set are removed from all evaluation datasets. This yields the stricter benchmark used for unseen‑compound evaluation.

---
## File format
All spectrum files are NumPy arrays saved with `numpy.save`. Each element is a `matchms.Spectrum` object, loaded with:

```python
import numpy as np
spectra = np.load(
    "data/MSBert/GNPS/Orbitrap/train_ref.npy",
    allow_pickle=True,
)
```

Each `Spectrum` carries the standard matchms metadata (`smiles`, `inchikey`, `precursor_mz`, `charge`, `ionmode`, and so on) as well as the peak list.

---
## Notes
‑ **Raw data are not redistributed.** Only the directory skeleton and this README are tracked in the repository.
‑ **Please cite the original data sources** when using these datasets. See the "Datasets" section above for the corresponding papers.
‑ **Zenodo release:** the processed benchmark is available at https://doi.org/10.5281/zenodo.19368733. It does not include InChIKey‑level deduplication; use Option B if strict unseen‑compound evaluation is required.
