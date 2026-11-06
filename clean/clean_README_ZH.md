# 数据清洗与预处理
本目录存放全部脚本，用于对原始二级质谱数据集做清洗、过滤、去重，并切分为查询集/参考集，供给 SpecRoFormer 的训练与评估流水线使用。

---
## 流水线总览
流水线输入为公开原始数据集（GNPS、MoNA、MassBank、MassSpecGym、MTBLS1572、仿真预训练数据），输出按【数据集/仪器/划分】组织的 `.npy` 谱图文件。

下表列出每个阶段、对应脚本以及主要输出产物：

| 阶段 | 脚本 | 主要产物 |
|---|---|---|
| 1.原始过滤 | `clean_msbert_data.py` | `*_cleaned.mgf`、`*_cleaned.msp` |
| 1.原始过滤 | `MassBank‑Clean.py` | MassBank 查询/参考集 `.npy`（10组副本） |
| 1.原始过滤 | `MassSpecGymData‑Clean.py` | MassSpecGym 查询/参考集 `.npy`（10组副本） |
| 1.原始过滤 | `clean_pretrain_dataset.py` | 预训练数据集 `.npy` |
| 2.合并与切分 | `process_msbert_data.py` | `all.npy`、`train_*.npy`、`test_*.npy`、`tanimoto_score.npy` |
| 3.去重 | `deduplicate_by_inchikey.py` | `*_dedup.npy`、`deduplication_report.xlsx`、`removed_compounds.xlsx` |
| 4.重建划分 | `random_split_query_ref.py` | 重建后的查询/参考集 `.npy`（10组副本） |
| 4.重建划分 | `random_split_query_ref_many_times.py` | GNPS测试集查询/参考集 `.npy`（10组副本） |

**依赖关系链**
- 阶段1 → 阶段2：清洗后的文件作为合并步骤输入。
- 阶段2 → 阶段3：合并后的数组作为去重步骤输入。
- 阶段3 → 阶段4：去重后的文件作为重建划分输入。
- 阶段4提供两套可选脚本；根据你是否需要重新执行MassBank清洗流水线选择其一。

---
## 脚本说明
### 1. `clean_msbert_data.py`
读取GNPS Orbitrap / QTOF / Other原始MGF文件与MoNA Orbitrap / QTOF原始MSP文件；过滤掉缺少有效化学注释（smiles、inchikey等）的谱图。清洗完成的谱图保存为 `*_cleaned.mgf` / `*_cleaned.msp`。

- **输入**：原始 `GNPS/*.mgf`、`MoNA/*.msp`
- **输出**：`*_cleaned.mgf`、`*_cleaned.msp`
- **核心函数**：`SpecRoFormer.utils.clean.is_annotated`
- **备注**：MoNA QTOF过滤后数据量损失较大（15834 → 7917）。

---
### 2. `process_msbert_data.py`
将清洗后的MSBERT格式谱图按数据集合并为独立 `.npy` 数组；相对于GNPS把MoNA拆分为“共有”/“独有”子集；构建GNPS查询/参考集；对每个唯一SMILES取一张代表谱，计算Tanimoto相似度矩阵。最后重建MoNA查询/参考划分，剔除GNPS Orbitrap训练参考集中已出现的化合物。

- **输入**：`*_cleaned.mgf`、`*_cleaned.msp`
- **输出**：
  - `GNPS/Orbitrap/all.npy`
  - `GNPS/QTOF/all.npy`
  - `GNPS/Other/all.npy`
  - `MoNA/{orbitrap,qtof}_{query,ref}.npy`
  - `MoNA/orbitrap_common.npy`、`MoNA/qtof_unique.npy`……
  - `GNPS/unique_smiles.npy`
  - `GNPS/tanimoto_score.npy`
- **核心函数**：
  - `SpecRoFormer.utils.clean.get_ref_query`
  - `SpecRoFormer.utils.clean.get_unique_smiles`
  - `matchms.similarity.FingerprintSimilarity`

---
### 3. `MassBank‑Clean.py`
读取MassBank NIST原始文件，执行完整MassBank清洗流水线（过滤、元数据清洗、电离模式分离、峰数量与母离子m/z过滤、注释校验）；按仪器类型（Orbitrap、QTOF、all）分组，每组生成10套随机查询/参考集副本；剔除GNPS Orbitrap训练参考集中已存在的化合物。

- **输入**：`MassBank/MassBank.msp_NIST`
- **输出**：
  - `MassBank/{Orbitrap,QTOF,all}-query‑replication‑{1..10}.npy`
  - `MassBank/{Orbitrap,QTOF,all}-reference‑replication‑{1..10}.npy`
- **依赖**：`GNPS/Orbitrap/train_ref.npy`

---
### 4. `MassSpecGymData‑Clean.py`
读取MassSpecGym原始TSV文件，将每行转为matchms的`Spectrum`对象；按仪器类型（all、Orbitrap、QTOF等）分组；每组生成10套随机查询/参考集副本，剔除GNPS Orbitrap训练参考集中的化合物。

- **输入**：`massSpecGymData/MassSpecGym.tsv`
- **输出**：
  - `massSpecGymData/{split}-query‑replication‑{1..10}.npy`
  - `massSpecGymData/{split}-reference‑replication‑{1..10}.npy`
- **依赖**：`GNPS/Orbitrap/train_ref.npy`

---
### 5. `clean_pretrain_dataset.py`
清洗仿真预训练数据集（例如CFM‑ID生成的MGF）。针对每张谱图，使用鲁棒解析器从SMILES计算母离子m/z（兼容仿真工具输出的非标准SMILES），设置`charge`、`ionmode`、`precursor_type`；过滤SMILES解析失败或峰列表为空的谱图。清洗后按数据集保存为 `.npy`。

- **输入**：`/mnt/.../Ms2Mol/data/pretrain/{dataset_name}.mgf`
- **输出**：每个数据集输出一份 `.npy`，路径由 `SpecRoFormer.const.pretrain_data.PRETRAIN_DATASETS` 定义
- **核心函数**：
  - `clean_cfm_smiles`
  - `smiles_to_precursor_mz_robust`
  - `is_annotated_virtual`

---
### 6. `deduplicate_by_inchikey.py`
使用InChIKey首段（连通性块），将评估数据集相对于训练集做严格去重。支持输入格式：

| 后缀 | 读取器 | 说明 |
|--------------------|--------------------------------------|-------------------------------------------|
| `.npy`             | `np.load(allow_pickle=True)`         | Spectrum对象或者字典元数据 |
| `.mgf`             | `matchms.load_from_mgf`              | |
| `.msp`             | `matchms.load_from_msp`              | 会先执行完整MassBank清洗流程 |
| `.msp_nist`        | `matchms.load_from_msp`              | 和`.msp`逻辑一致 |
| `.tsv`             | `pandas.read_csv(sep="\t")`          | MassSpecGym格式，内部构建Spectrum |

- **输入**：训练参考集 `.npy` + 一批数据集路径
- **输出**：
  - `<dataset>_dedup.npy`
  - `deduplication_report.xlsx`
  - `removed_compounds.xlsx`
- **核心函数**：
  - `normalize_inchikey`
  - `smiles_to_inchikey`
  - `build_train_set`
  - `dedup_npy`、`dedup_mgf`、`dedup_msp`、`dedup_tsv`

脚本的`__main__`中数据集列表默认为注释状态；只取消你需要处理的条目注释。

---
### 7. `random_split_query_ref.py`
去重完成后重建查询/参考集划分的统一入口，提供3个工具函数：

- `process_msbert(gnps_root, mona_root)`
  重建GNPS Orbitrap / QTOF / Other 与 MoNA Orbitrap / QTOF 的划分；生成各数据集的all文件以及GNPS测试集的10组副本。

- `process_massbank(massbank_path, train_ref_path, output_dir)`
  按仪器重建MassBank查询/参考集，生成10组副本，剔除训练参考集内化合物。

- `process_massspecgym(root_dir, train_ref_path)`
  按仪器重建MassSpecGym查询/参考集，生成10组副本，剔除训练参考集内化合物。

- **输入**：`deduplicate_by_inchikey.py`产出的 `*_dedup.npy`
- **输出**：`*_query‑replication‑{1..10}.npy` 与 `*_reference‑replication‑{1..10}.npy`
- **核心函数**：
  - `SpecRoFormer.utils.clean.get_ref_query`
  - `SpecRoFormer.utils.clean.get_unique_smiles`

`__main__`会按顺序依次执行全部三个工具函数，路径为硬编码；请根据你的本地环境修改路径。

---
### 8. `random_split_query_ref_many_times.py`
为GNPS测试集（Orbitrap / QTOF / Other）生成10套随机查询/参考划分，剔除GNPS Orbitrap训练参考集已存在的化合物。

该脚本是 `random_split_query_ref.py` 的补充：直接读取 `process_msbert_data.py` 输出的 `.npy`，**不需要重新跑过去重流程**。

- **输入**：`GNPS/Orbitrap/{train,test}_*.npy`、`GNPS/QTOF/*.npy`、`GNPS/Other/*.npy`
- **输出**：
  - `GNPS/Orbitrap/test_query‑replication‑{1..10}.npy`
  - `GNPS/Orbitrap/test_ref‑replication‑{1..10}.npy`
  - `GNPS/QTOF/test_query‑replication‑{1..10}.npy`
  - `GNPS/QTOF/test_ref‑replication‑{1..10}.npy`
  - `GNPS/Other/test_query‑replication‑{1..10}.npy`
  - `GNPS/Other/test_ref‑replication‑{1..10}.npy`

---
## 推荐执行顺序
下面是从原始数据开始完整运行流水线的顺序；如果中间产物已存在，对应步骤可以跳过。

1. 原始过滤
```bash
python clean_msbert_data.py        # GNPS + MoNA raw filtering
python MassBank‑Clean.py           # MassBank raw cleaning
python MassSpecGymData‑Clean.py    # MassSpecGym raw cleaning
python clean_pretrain_dataset.py   # in‑silico pretrain cleaning
```

2. 合并切分 GNPS + MoNA
```bash
python process_msbert_data.py
```

3. 严格去重
```bash
python deduplicate_by_inchikey.py
```

4. 去重后重建数据集划分
```bash
python random_split_query_ref.py
# 或者（仅想重新切分GNPS测试集，不重新去重）
# python random_split_query_ref_many_times.py
```

### 依赖注意事项
- 步骤1、2依赖 `SpecRoFormer.utils.clean`，需要保证包可导入（脚本内部已经写好 `sys.path.append("..")`）。
- 步骤3（`deduplicate_by_inchikey.py`）产出的 `*_dedup.npy` 供步骤4读取。如果跳过步骤3，你需要修改脚本路径，让步骤4读取未去重文件。
- `random_split_query_ref_many_times.py` 是 `random_split_query_ref.py` 的替代方案，不会重新运行MassBank清洗；适合只想快速重切GNPS测试集的场景。
- 全部输出路径写死为项目原始目录结构；运行前修改为你本地环境路径。

---
## 本目录用到的数据集
| 数据集 | 格式 | 源文件 |
|------------------|---------------|--------------------------------------------------|
| GNPS             | mgf           | `Orbitrap_*.mgf`、`QTOF.mgf`、`Other.mgf`        |
| MoNA             | msp           | `MoNA_Orbitrap.msp`、`MoNA_QTOF.msp`             |
| MassBank         | msp_nist      | `MassBank.msp_NIST`                              |
| MassSpecGym      | tsv           | `MassSpecGym.tsv`                                |
| MTBLS1572        | msp           | `q.msp`、`r.msp`                                 |
| Pretrain         | mgf           | `{dataset_name}.mgf`（CFM‑ID风格仿真数据）   |

原始数据文件不纳入git版本管理。下载链接与文件夹结构参见上层 `data/` 的文档。
