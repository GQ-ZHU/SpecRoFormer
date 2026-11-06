# SpecRoFormer

SpecRoFormer 是一个基于 Transformer 的框架，用于学习小分子 MS/MS 谱图的可泛化表征。它通过 RoPE 建模相对 m/z 关系，将前体离子 m/z 作为全局条件信号注入，并使用碎片感知增强来提升在分布偏移下的鲁棒性。

---

## 概述

串联质谱（MS/MS）中的谱图检索依赖于能够准确反映分子结构相似性的表征学习。然而，不同实验条件下的碎片化差异以及仪器偏差会引入显著的分布偏移，使得学习能够跨仪器泛化的表征变得困难。在本研究中，我们提出 SpecRoFormer——一个基于 Transformer 的框架，通过相对谱图建模与碎片感知增强来学习可泛化的 MS/MS 表征。该方法在捕获峰之间与结构相关的关系的同时，提升了对碎片化变异和分布偏移的鲁棒性。我们在 GNPS 数据集上训练 SpecRoFormer，并在多个外部基准（包括 MoNA、MassBank 和 MassSpecGym）上评估其泛化能力。在这些数据集上，SpecRoFormer 相比 MSBERT、DreaMS 和 SpecEmbedding 等强基线取得了最佳的整体检索性能。此外，在大规模模拟数据集 SimMS-9M 上的评估进一步验证了 SpecRoFormer 在更广泛分布偏移下的鲁棒性，其表现持续优于竞争方法。这些结果表明，将相对谱图关系建模与碎片感知鲁棒性相结合，对于学习可泛化的 MS/MS 表征至关重要，能够实现可靠的跨仪器谱图检索。

![SpecRoFormer 框架总览](figs/Figure-1-Overview_of_SpecRoFormer_framework.jpg)

有关数据集下载链接、目录结构以及完整的数据处理流程，请参阅 [`data/README.md`](data/data_README_ZH.md)。

---

## 核心特性

- **面向 m/z 的 RoPE**：将旋转位置编码（Rotary Positional Encoding）适配到连续的 m/z 坐标，使注意力机制能够自然地捕获碎片离子之间的相对质量关系。
- **前体离子 m/z 条件化**：通过门控残差连接，将前体离子 m/z 作为显式的全局条件信号注入。
- **碎片感知增强**：通过子谱采样、低强度峰去除和强度扰动来模拟真实的碎片化变异。
- **训练目标**：监督对比学习 + Tanimoto 相似度回归。

---

## 安装

### 环境要求

- Python >= 3.9
- pip 或 conda
- 支持 CUDA 的 GPU（可选，但强烈建议）

### 方式 A - Pip（推荐）

```bash
git clone <GITHUB_URL>
cd <repo-name>

# (1) 如有需要，先安装支持 CUDA 的 PyTorch：
#    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# (2) 一键式环境配置

bash environment.sh
```

该脚本将：

1. 升级 pip。
2. 从 `requirements.txt` 安装依赖。
3. 以可编辑模式安装 `SpecRoFormer` 包。
4. 运行环境自检。

### 方式 B - Conda

```bash
conda env create -f environment.yml
conda activate specroformer
python scripts/check_env.py
```

### 方式 C - 手动 pip

```bash
pip install -r requirements.txt
pip install -e .
python scripts/check_env.py
```

---

## 数据准备

有关如何下载或重建基准数据集的详细信息，请参阅 [`data/README.md`](data/README.md)。

- **训练集**：仅使用 GNPS Orbitrap 训练划分。
- **评估集**：GNPS Orbitrap / QTOF / Other 测试划分、MoNA、MassBank、MassSpecGym、MTBLS1572、SimMS-9M。

---

## 快速开始

### 训练

```bash
cd train_test
bash train.sh
```

或直接运行：

```bash
python train.py \
    --is_augment \
    --batch_size 128 \
    --methods SpecRoFormer \
    --loss_type SupConWithTanimotoLoss
```

### 评估

```bash
cd train_test
bash test.sh
```

或对特定基准进行评估：

```bash
python test_GNPS.py \
    --is_augment \
    --batch_size 128 \
    --methods SpecRoFormer \
    --loss_type SupConWithTanimotoLoss
```

支持的基准脚本：

- `test_GNPS.py`、`test_GNPS_dedup.py`
- `test_MoNA.py`、`test_MoNA_dedup.py`
- `test_MassBank.py`、`test_MassBank_dedup.py`
- `test_MassSpecGymData.py`、`test_MassSpecGymData_dedup.py`
- `test_MTBLS1572.py`
- `test_3M.py`（SimMS-9M）

### 超参数搜索

```bash
cd train_test
bash hparam_search.sh
```

---

## 与先前工作的关系

SpecRoFormer 是对以下开源代码的修改与扩展版本：

> Pengxiang Xiong, Haitao Xu, Haijing Zheng.
> Supervised Contrastive Learning Leads to More Reasonable Spectral Embeddings.
> Analytical Chemistry, 2025, 97, 20137-20146.
> https://doi.org/10.1021/acs.analchem.5c02655

我们衷心感谢 SpecEmbedding 的作者公开其代码以及本工作所基于的基准构建流程。SpecRoFormer 的核心差异包括：

- 在原有正弦 m/z 编码的基础上引入基于 RoPE 的相对 m/z 建模。
- 通过门控残差连接实现显式的前体离子 m/z 条件化。
- 包含子谱采样在内的碎片感知增强。
- 组合的对比学习 + Tanimoto 训练目标。

---

## 本仓库未包含的内容

以下组件属于我们用于回应特定审稿人意见的内部分析流程，因此不予发布：

- 注意力分布（Attention-profile）分析
- 聚类 / t-SNE 分析
- 数据集统计量分析
- 嵌入结构一致性分析
- 检索行为分析
- ROC-AUC 分析

核心方法、训练、评估与数据清洗流程均已包含在内，可端到端复现。

---

## 复现与支持

我们已尽力整理代码、补充标准注释，并对公布的版本进行了测试，确认其可以正常运行。

但需要说明的是，如果在复现过程中遇到报错，很可能是环境配置、依赖包或函数版本差异，以及一些我们未能预料的问题所致。我们无法穷尽列举所有可能的情况及对应的解决办法，敬请见谅。

如果本工作对你正在进行的实验有所帮助，欢迎引用我们的工作。


## 引用

如果您认为本工作有用，请引用：

```bibtex
@article{zhu2026specroformer,
    title={SpecRoFormer: Relative Position-Aware Representation Learning for Small-Molecule MS/MS Spectra}, 
    author = {Guoqing Zhu and Shiting Lai and Xinyi Cao and Keyu Hu and Luojun Lin and Yuanlong Yu}, 
    journal={Analytical Chemistry}, 
    year={2026}, 
    doi={10.1021/acs.analchem.6c03195}
}
```

如果您使用该基准构建流程，也请引用原始的 SpecEmbedding 论文。

---

## 致谢

本工作受国家自然科学基金资助（项目批准号：62406071）。我们感谢 SpecEmbedding 作者的开放源码发布，以及 GNPS、MoNA、MassBank、MassSpecGym、MTBLS1572 和 CSU-MS2 的创建者公开提供其数据集。

---

## 许可证

本项目基于 MIT 许可证发布。详情请参阅 [LICENSE](LICENSE)。

---
