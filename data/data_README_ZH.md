# 数据集说明（data/README.md）
本目录存放 SpecRoFormer 训练与评估所用全部数据集。**原始数据文件不会随代码仓库一同分发**，仓库内仅保留目录骨架与本说明文档。

---
## 概述
SpecRoFormer 使用多个公开二级质谱数据集开展训练，支持跨数据集、跨仪器的模型评估。项目提供两套基准评测方案：

‑ **预处理基准集（推荐快速上手）：** Zenodo上提供已经预处理完成的谱图、数据集划分文件与评估文件。该版本**未**针对训练集做 InChIKey 去重。
‑ **严格基准集（用于未知化合物评估）：** 基于 SpecEmbedding 发布的原始数据源重建，经过 `clean/` 目录下清洗流水线，执行严格 InChIKey 级别去重。

| 数据集 | 用途 | 数据源 |
|---|---|---|
| GNPS (Orbitrap) | 训练 + 评估 | SpecEmbedding, figshare |
| GNPS (QTOF / Other) | 评估 | SpecEmbedding, figshare |
| MoNA (Orbitrap / QTOF) | 评估 | SpecEmbedding, figshare |
| MassBank (Orbitrap / QTOF / all) | 评估 | SpecEmbedding, figshare |
| MassSpecGym (Orbitrap / QTOF / all) | 评估 | SpecEmbedding, figshare |
| MTBLS1572 | 评估 | SpecEmbedding, figshare |
| SimMS‑9M | 评估 | CSU‑MS2, Zenodo |

---
## 数据下载
### 方案A — 使用预处理基准集（推荐）
Zenodo 数据集条目：
‑ https://doi.org/10.5281/zenodo.19368733

该页面提供多个压缩包，请将每个压缩包解压至 `data/` 下对应子目录：

| 压缩包 | 解压目标路径 |
|---|---|
| `GNPS.zip` | `data/MSBert/GNPS/` |
| `MoNA.zip` | `data/MSBert/MoNA/` |
| `MTBLS1572.zip` | `data/MSBert/MTBLS1572/` |
| `MassBank.zip` | `data/MassBank/` |
| `massSpecGymData.zip` | `data/massSpecGymData/` |
| SimMS‑9M 压缩包 | `data/PreTrained/simulated_dataset/` |

解压完成后，各数据集目录下应存在 `.npy` 谱图文件，详见下方「目录结构」章节。

> **注意：** 预处理版本**没有**相对于训练集做 InChIKey 去重。如果你需要严格的未知化合物评估，请使用方案B。

---
### 方案B — 从原始数据源重建
下载 SpecEmbedding 发布的原始数据，放到 `data/` 目录，运行 `clean/` 下清洗流水线，生成数据集划分并执行严格 InChIKey 去重。

原始数据源：
‑ GNPS, MoNA, MassBank, MassSpecGym, MTBLS1572
  ‑ 论文：https://doi.org/10.1021/acs.analchem.5c02655
  ‑ 数据：https://doi.org/10.6084/m9.figshare.28876751
‑ SimMS‑9M
  ‑ 论文：https://doi.org/10.1021/acs.analchem.5c01594
  ‑ 数据：https://zenodo.org/records/17756840

按照 [`clean/README.md`](../clean/clean_README_ZH.md) 的说明摆放原始文件，分四步运行流水线：

1. **原始数据过滤**
```bash
python clean_msbert_data.py            # GNPS + MoNA
python MassBank‑Clean.py               # MassBank
python MassSpecGymData‑Clean.py        # MassSpecGym
python clean_pretrain_dataset.py       # 模拟数据集 SimMS‑9M
```

2. **合并并划分 GNPS + MoNA**
```bash
python process_msbert_data.py
```

3. **严格去重**
```bash
python deduplicate_by_inchikey.py
```

4. **重建查询集/参考集划分**
```bash
python random_split_query_ref.py
# 如果不需要重新跑过去重，仅重新切分GNPS测试集可执行：
# python random_split_query_ref_many_times.py
```

脚本输入输出、依赖关系详见 [`clean/README.md`](../clean/README.md)。

---
## 目录结构
下载（方案A）或重建（方案B）完成后，`data/` 目录结构如下：

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

训练、评估运行时会自动生成 `labels/`、`model/` 等目录，不属于本静态目录结构。

---
## 数据集介绍
### GNPS
包含大量不同化合物的二级质谱，采集仪器包括 Orbitrap、QTOF 以及其他类型仪器。
‑ **训练集：** 仅 GNPS Orbitrap 的训练子集。
‑ **评估集：** GNPS Orbitrap / QTOF / Other 的测试子集。

### MoNA
北美质谱库，包含 Orbitrap、QTOF测得的大量小分子谱图。
‑ **仅用于评估。**

### MassBank
国际公开高质量二级质谱库。
‑ **仅用于评估。**

### MassSpecGym
来自 NeurIPS 2024 的基准数据集，包含 Orbitrap、QTOF仪器采集谱图。
‑ **仅用于评估。**

### MTBLS1572
包含134种内源性代谢物的小数据集，QTOF仪器DDA模式采集。
‑ **仅用于评估。**

### SimMS‑9M
大规模模拟数据集，基于约300万化合物使用CFM‑ID在三种碰撞能量条件生成，合计约900万张谱图；也用作仿真预训练语料。
‑ **仅用于评估。**

---
## 数据处理流程
全部数据集由 `clean/` 目录脚本处理。完整流水线、脚本说明与依赖关系见 [`clean/README.md`](../clean/README.md)。

流水线分为4个阶段：
1. 原始谱过滤与元信息校验。
2. 数据合并，切分为查询集 / 参考集。
3. 基于InChIKey，相对于GNPS Orbitrap训练集做严格去重。
4. 去重完成后，重新生成查询集/参考集划分。

---
## 评测协议
‑ **训练集：** 仅 GNPS Orbitrap 训练子集。
‑ **评估集：** GNPS各测试子集（Orbitrap / QTOF / Other）、MoNA、MassBank、MassSpecGym、MTBLS1572、SimMS‑9M。
‑ **数据集切分规则：** 在每个数据集‑仪器分组内部，按化合物维度划分查询集与参考集。若一个化合物有多张谱，随机选一张作为查询，其余进入参考集；若只有一张谱，则直接放入参考集。
‑ **重复实验副本：** GNPS、MassBank、MassSpecGym等大数据集会重复切分10次，全部10套划分文件对外发布；评测结果取10次的均值与标准差。
‑ **严格基准集：** 将所有评估集中，InChIKey首段与GNPS Orbitrap训练集重合的化合物全部剔除，得到用于未知化合物评测的严格基准。

---
## 文件格式
所有谱图文件为 numpy.save 保存的NumPy数组。数组每一项是 `matchms.Spectrum` 对象，读取示例：

```python
import numpy as np
spectra = np.load(
    "data/MSBert/GNPS/Orbitrap/train_ref.npy",
    allow_pickle=True,
)
```

每个 Spectrum 对象包含 matchms 标准元数据（smiles、inchikey、precursor_mz、charge、ionmode等）以及峰列表。

---
## 注意事项
‑ **原始数据不随仓库分发。** 仓库仅维护目录骨架和本README。
‑ 使用数据集时请引用对应的原始文献，见上方「数据集介绍」部分。
‑ Zenodo预处理数据集地址：https://doi.org/10.5281/zenodo.19368733。该版本不含InChIKey去重；如需严格未知化合物评测请使用方案B。
