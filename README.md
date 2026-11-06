# SpecRoFormer

SpecRoFormer is a Transformer-based framework for learning generalizable representations of small-molecule MS/MS spectra. It models relative m/z relationships through RoPE, incorporates precursor m/z as a global conditioning signal, and uses fragmentation-aware augmentation to improve robustness under distribution shifts.

---

## Overview

Spectrum retrieval in tandem mass spectrometry (MS/MS) relies on learning representations that accurately reflect molecular structural similarity. However, fragmentation variability across experimental conditions and instrument biases introduce significant distribution shifts in spectral data, making it difficult to learn representations that can generalize to different instruments. In this study, we propose SpecRoFormer, a Transformer-based framework that learns generalizable MS/MS representations through relative spectral modeling and fragmentation-aware augmentation. The approach captures structure-relevant relationships between peaks while improving robustness to fragmentation variability and distribution shifts. We train SpecRoFormer on the GNPS dataset and evaluate its generalization across multiple external benchmarks, including MoNA, MassBank, and MassSpecGym. Across these datasets, SpecRoFormer achieves the best overall retrieval performance against strong baselines such as MSBERT, DreaMS, and SpecEmbedding. Furthermore, evaluations on the large-scale simulated dataset SimMS-9M further verify the robustness of SpecRoFormer under broader distribution shifts, consistently outperforming competing methods. These results demonstrate that modeling relative spectral relationships together with fragmentation-aware robustness is critical for learning generalizable MS/MS representations, enabling reliable cross-instrument spectrum retrieval.

![Overview of SpecRoFormer framework](figs/Figure-1-Overview_of_SpecRoFormer_framework.jpg)

For dataset download links, directory layout, and the full data processing pipeline, see [`data/README.md`](data/README.md).

---

## Key Features

- **RoPE for m/z**: Rotary Positional Encoding adapted to continuous m/z coordinates, so the attention mechanism naturally captures relative mass relationships between fragment ions.
- **Precursor m/z conditioning**: The precursor m/z is injected as an explicit global conditioning signal through a gated residual connection.
- **Fragmentation-aware augmentation**: Sub-spectrum sampling, low intensity peak removal, and intensity perturbation mimic realistic fragmentation variability.
- **Training objective**: Supervised contrastive learning + Tanimoto-similarity regression.

---

## Installation

### Prerequisites

- Python >= 3.9
- pip or conda
- CUDA-enabled GPU (optional but strongly recommended)

### Option A - Pip (recommended)

```bash
git clone <GITHUB_URL>
cd <repo-name>

# (1) Install CUDA-enabled PyTorch first if needed:
#    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# (2) One-shot setup

bash environment.sh
```

The script will:

1. Upgrade pip.
2. Install dependencies from `requirements.txt`.
3. Install the `SpecRoFormer` package in editable mode.
4. Run the environment self-check.

### Option B - Conda

```bash
conda env create -f environment.yml
conda activate specroformer
python scripts/check_env.py
```

### Option C - Manual pip

```bash
pip install -r requirements.txt
pip install -e .
python scripts/check_env.py
```

---

## Data preparation

See [`data/README.md`](data/data_README_EN.md) for full details on how to download or rebuild the benchmark datasets.

- **Training set**: GNPS Orbitrap training split only.
- **Evaluation sets**: GNPS Orbitrap / QTOF / Other test splits, MoNA, MassBank, MassSpecGym, MTBLS1572, SimMS-9M.

---

## Quick start

### Training

```bash
cd train_test
bash train.sh
```

Or directly:

```bash
python train.py \
    --is_augment \
    --batch_size 128 \
    --methods SpecRoFormer \
    --loss_type SupConWithTanimotoLoss
```

### Evaluation

```bash
cd train_test
bash test.sh
```

Or, evaluate on a specific benchmark:

```bash
python test_GNPS.py \
    --is_augment \
    --batch_size 128 \
    --methods SpecRoFormer \
    --loss_type SupConWithTanimotoLoss
```

Supported benchmark scripts:

- `test_GNPS.py`, `test_GNPS_dedup.py`
- `test_MoNA.py`, `test_MoNA_dedup.py`
- `test_MassBank.py`, `test_MassBank_dedup.py`
- `test_MassSpecGymData.py`, `test_MassSpecGymData_dedup.py`
- `test_MTBLS1572.py`
- `test_3M.py` (SimMS-9M)

### Hyperparameter search

```bash
cd train_test
bash hparam_search.sh
```

---

## Relation to prior work

SpecRoFormer is a modified and extended version of the open-source code released alongside:

> Pengxiang Xiong, Haitao Xu, Haijing Zheng.
> Supervised Contrastive Learning Leads to More Reasonable Spectral Embeddings.
> Analytical Chemistry, 2025, 97, 20137-20146.
> https://doi.org/10.1021/acs.analchem.5c02655

We sincerely thank the authors of SpecEmbedding for releasing their code and the benchmark construction protocol that this work builds on. The core differences in SpecRoFormer are:

- RoPE-based relative m/z modeling introduced on top of the original sinusoidal m/z encoding.
- Explicit precursor m/z conditioning via gated residual connection.
- Fragmentation-aware augmentation including sub-spectrum sampling.
- A combined contrastive + Tanimoto training objective.

---

## Not included in this repository

The following components are part of our internal analysis pipeline used to answer specific reviewer requests, and are therefore not released:

- Attention-profile analysis
- Clustering / t-SNE analysis
- Dataset statistics analysis
- Embedding-structure consistency analysis
- Retrieval-behavior analysis
- ROC-AUC analysis

Core method, training, evaluation, and data cleaning pipeline are all included and can be reproduced end-to-end.

---

## Reproducibility and Support

We have made our best effort to organize the code, provide standard comments/docstrings throughout, and test the released version to confirm that it runs as expected.

That said, if you encounter errors during reproduction, they are most likely caused by differences in environment configuration or package/function versions, or by other issues we did not anticipate. We cannot exhaustively enumerate every possible case or provide a fix for each of them, and we appreciate your understanding.

If this work is helpful for your own experiments, we would be grateful if you could cite it.

## Citation

If you find this work useful, please cite:

```bibtex
@article{zhu2026specroformer,
    title={SpecRoFormer: Relative Position-Aware Representation Learning for Small-Molecule MS/MS Spectra}, 
    author = {Guoqing Zhu and Shiting Lai and Xinyi Cao and Keyu Hu and Luojun Lin and Yuanlong Yu}, 
    journal={Analytical Chemistry}, 
    year={2026}, 
    doi={10.1021/acs.analchem.6c03195}
}
```

Please also cite the original SpecEmbedding paper if you use the benchmark construction protocol.

---

## Acknowledgements

This work is supported by the National Natural Science Foundation of China (Grant No. 62406071). We thank the authors of SpecEmbedding for their open-source release, and the creators of GNPS, MoNA, MassBank, MassSpecGym, MTBLS1572, and CSU-MS2 for making their datasets publicly available.

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---
