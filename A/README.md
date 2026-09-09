# 时序图未来链接预测

> 参赛团队：ZenFlow

本项目是第六届计图人工智能挑战赛“基于图学习的动态推荐任务”的完整复现代码。给定历史时序交互图，以及测试查询中的源节点、时间戳和 100 个候选目标节点，模型输出每个候选在未来形成链接的相对概率。

项目包含 Dataset1 和 Dataset2 两条建模流水线，从赛事原始 CSV 开始自动完成数据检查、验证集构造、特征生成、模型训练、推理、结果校验及 ZIP 打包。

## 主要特点

- 使用 Jittor/JittorGeometric 训练时序图与图表示模型；
- 将未来链接预测建模为每行 100 个候选节点的学习排序问题；
- 融合时序统计、局部图结构、矩阵分解、LightGCN、协同过滤和树模型；
- Dataset2 使用滚动时间窗口和 OOF 预测训练多级重排序器；
- 大型特征采用稀疏矩阵、memmap 和分块计算；
- 支持阶段缓存、断点续跑和指定阶段强制重跑；
- 自动检查预测矩阵形状、数值范围并生成 `result.zip`。

## 目录结构

```text
.
├── run.py                         # 完整流水线入口
├── requirements.txt               # Python 依赖
├── scripts/
│   └── run_pipeline.sh            # Shell 运行入口
├── code/
│   ├── solution/
│   │   ├── base/                  # Dataset2 基础模型与公共特征
│   │   ├── dataset1/              # Dataset1 训练、特征和推理
│   │   └── dataset2/              # Dataset2 高层特征与重排序
│   └── training/
│       ├── base_models/           # Dataset2 基础模型训练
│       └── dataset2/              # OOF、校准及上层模型训练
└── README.md
```

运行过程中生成的模型、缓存和预测位于：

```text
outputs/
submissions/
```

这些目录已由 `.gitignore` 排除，可以从原始数据重新生成。

## 环境要求

推荐环境：

```text
Ubuntu 22.04
Python 3.10
CUDA 12.4
Jittor 1.3.10
16 GB 以上内存
具有充足空间的高速磁盘
```

Jittor 首次执行时需要编译算子，因此第一次启动通常明显慢于后续运行。

### 使用 venv 安装

Ubuntu 可使用系统提供的 `python3` 创建环境。系统依赖配置参考[Jittor 官方安装说明](https://github.com/Jittor/jittor#install)，
JittorGeometric 使用项目固定的[官方源码](https://github.com/AlgRUC/JittorGeometric/tree/ff7d8ffac7bf3d95cc1962e091c52dc5737492d4)：

```bash
sudo apt-get update
sudo apt-get install -y \
  python3 python3-dev python3-venv python3-pip \
  build-essential g++ libomp-dev git

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt
```

其中 Jittor 由 PyPI 安装；JittorGeometric 使用 `requirements.txt` 中固定的 GitHub 提交源码包构建安装。后者包含标准`setup.py`，因此不需要手动执行 `git clone` 和 `pip install .`。安装过程需要能够访问 GitHub。

`requirements.txt` 只能安装 Python 包，不能安装编译器、Python 开发头文件或 OpenMP 运行库，因此全新 Ubuntu 环境仍须先执行上述 `apt-get` 命令。

如Python、编译器和 OpenMP 已由审核环境提供，可以跳过系统依赖安装。

### 使用 Conda（可选）

Conda 不是运行本项目的必要条件。本地开发如需使用 Conda，可以执行：

```bash
conda create -n <env_name> python=3.10 -y
conda activate <env_name>
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt
```

### 检查环境

```bash
python3 --version
g++ --version
python3 -m jittor.test.test_example
python3 -c "from jittor_geometric.nn.models.craft import CRAFT; from jittor_geometric.dataloader.temporal_dataloader import TemporalDataLoader, get_neighbor_sampler; print('JittorGeometric CRAFT imports ok')"
python3 -c "import numpy, pandas, lightgbm, scipy; print('dependencies ok')"
```

如 CUDA 工具链不在标准位置，可以在运行前指定：

```bash
export cc_path=/usr/bin/g++
export nvcc_path=/usr/local/cuda/bin/nvcc
```

请根据审核机的实际 CUDA 安装路径调整 `nvcc_path`。CPU 环境可以完成 Jittor 基础安装测试，但本项目完整训练建议使用可用的 NVIDIA CUDA 环境。

## 数据准备

将原始数据集放在项目根目录：

```text
dataset1/
├── train.csv
└── test.csv

dataset2/
├── train.csv
└── test.csv
```

训练集字段：

```text
src,dst,time
```

Dataset2 训练集还包含赛事提供的 `split` 字段。测试集字段为：

```text
src,time,candidate_1,...,candidate_100
```

`run.py` 启动时会检查：

- 四个原始文件是否存在；
- CSV 表头是否符合预期；
- 测试文件是否包含 100 个候选列。

## 快速开始

### 完整运行

推荐使用 Shell 入口：

```bash
bash scripts/run_pipeline.sh
```

Shell 脚本默认调用 `python3`。如需指定其他 Python 3 解释器，可以覆盖 `PYTHON_BIN`：

```bash
PYTHON_BIN=/usr/bin/your_python3 bash scripts/run_pipeline.sh
```

等价的 Python 命令为：

```bash
python3 run.py --target all --seed 20260724
```

完整运行会依次训练 Dataset1 和 Dataset2，并最终生成：

```text
submissions/prediction/result.zip
```

`--seed` 是完整流水线的统一随机种子。所有 fold、子模型和随机采样阶段均直接
使用该值，不再附加阶段或 fold 偏移。

### 单独运行 Dataset1

```bash
python3 run.py --target dataset1
```

### 使用已有 checkpoint 推理

以 Dataset2 的 Tree 基础模型为例，训练完成后可跳过训练，直接读取指定
`--save_dir` 中的 `dataset2_tree_ranker.txt` 生成测试集候选分数：

```bash
PYTHONPATH=code python3 -m training.base_models.dataset2_tree \
  --dataset dataset2 \
  --data_dir . \
  --save_dir outputs/dataset2/training/base_models/tree/models \
  --output_dir outputs/dataset2/training/base_models/tree/predictions \
  --n_workers 2 \
  --seed 20260724 \
  --inference_only
```

该命令读取：

```text
outputs/dataset2/training/base_models/tree/models/dataset2_tree_ranker.txt
```

并生成：

```text
outputs/dataset2/training/base_models/tree/predictions/dataset2/dataset2_tree_scores.csv
```

完整集成推理依赖验证特征和多个上层 checkpoint，由
`python3 run.py --target dataset2` 自动按依赖顺序执行；已有且源码指纹一致的
阶段会被跳过。

### 仅重新打包

当两份预测 CSV 已经存在时，可以执行：

```bash
python3 run.py --target package
```

### 查看流水线阶段

```bash
python3 run.py --list-stages
```

### 强制重跑指定阶段

```bash
python3 run.py --target all --force-stage <stage>
```

例如：

```bash
python3 run.py --target all --force-stage validation_temporal_stack
python3 run.py --target all --force-stage final_candidate_ranker
```

`--force-stage` 可以重复指定：

```bash
python3 run.py --target all \
  --force-stage distribution_ranker_training \
  --force-stage distribution_ranker_inference
```

### 参数管理

顶层入口暴露影响全流程复现的 `--target`、`--seed` 和 `--force-stage`。学习率、
epoch、batch size、训练抽样量、模型维度等关键训练参数由对应模块的命令行参数
定义，默认值即为本方案提交配置，可通过 `python3 -m <module> --help` 查看。
特征窗口、固定融合系数和候选宽度等与本方案结构绑定的常量保留在实现中，避免
完整复现时因无意覆盖而改变模型结构。

## 任务建模

训练集中的时序交互表示为：

```text
e_i = (src_i, dst_i, time_i)
```

测试查询表示为：

```text
q_j = (src_j, time_j, candidates_j)
```

其中 `candidates_j` 包含 100 个候选目标节点。模型为每个候选计算：

```text
score(src_j, candidate_jk, time_j | history)
```

训练时，一条历史真实边的 `dst` 为正样本，其他候选为负样本。正样本在候选集合中的位置会随机化。模型最终优化候选集合内的相对排序，并以 MRR 作为主要评价指标。

## 评价指标与复现结果

主要指标为平均倒数排名（Mean Reciprocal Rank，MRR）。对第 `i` 条查询，将 100 个候选按预测分数从高到低排序，真实目标的名次记为 `rank_i`：

```text
MRR = (1 / N) * sum(1 / rank_i)
```

排名从 1 开始；分数相同时保持候选输入顺序，以保证计算稳定。Dataset2 使用严格按时间构造的三个滚动验证 fold：目标时间之前的边作为历史，目标边仅用于计算该 fold 的 MRR，不参与对应 fold 特征或模型的拟合。

本开源版本不将某次本地运行的中间验证分数声明为线上成绩。完整运行会在验证阶段的 `report.json` 和标准输出中记录 fold MRR；线上成绩以赛事评测系统对最终`result.zip` 的计算为准。离线时间切分、线上测试窗口及候选集合不同，因此离线MRR 与线上成绩并不要求完全一致。CUDA、Jittor 编译器版本以及 LightGBM 多线程归约还可能造成浮点末位差异。

## 总体架构

```text
train.csv（带真实目标的历史交互）
          │
          ├── Dataset1：时间切分、伪查询与模型训练
          │
          ├── Dataset2：滚动时间验证、严格 OOF 与模型训练
          │             ├── CRAFT / Tree / Temporal / GraphMF / LightGCN
          │             ├── Basket、候选分布和边际校准
          │             ├── 物品、多尺度和用户协同过滤
          │             └── 时间、锚点、分组和图传播重排序
          │
          └── 构建历史统计量、图表示和 checkpoint
                                      │
                                      ▼
test.csv（无标签查询：src、time、100 个候选 dst）
          │
          ├── 查询训练历史对应的时序与图特征
          ├── 构造无标签测试候选集合的转导式分布特征
          ├── 使用已训练模型分别生成 Dataset1 / Dataset2 分数
          └── 不作为真实交互加入历史图，不参与模型拟合
                                      │
                                      ▼
             172 维最终候选级排序、结果校验与 result.zip 打包
```

测试阶段不会读取或构造测试标签，测试候选也不会作为已经发生的真实边加入训练历史。除训练历史特征外，模型会对赛事提供的完整无标签候选集合统计候选频率、时间窗口分布和跨查询关系，用于离线批量转导推理。这些统计不参与有监督标签拟合，但意味着不同测试查询并非完全独立处理。

## Dataset1 模型

### 训练数据构造

Dataset1 从训练历史尾部切出伪测试窗口：

1. 窗口之前的边作为冻结历史；
2. 窗口中的真实边作为训练查询；
3. 真实目标节点作为正样本；
4. 从候选节点池构造负样本；
5. 正样本随机放入候选集合。

生产阶段使用完整训练集作为历史，对正式测试集的 100 个候选逐一打分。

### 时序与结构特征

Dataset1 候选特征主要包括：

- 源节点历史交互次数；
- 目标节点历史度数和独立源节点数；
- `(src, dst)` 历史交互次数；
- 最近交互时间和时间间隔；
- 多尺度时间衰减统计；
- 源节点近期交互目标；
- 两跳路径、共同邻居及局部结构支持；
- PPR/传播类结构分数；
- 候选在查询集合中的频率和关系统计。

特征实现主要位于：

```text
code/solution/dataset1/temporal_features.py
code/solution/dataset1/structural_features.py
code/solution/dataset1/reciprocal_features.py
```

### Jittor 图表示

Dataset1 使用 Jittor 训练图表示通道。默认配置：

| 参数           | 默认值 |
| -------------- | -----: |
| Embedding 维度 |     64 |
| 图传播层数     |      3 |
| 训练步数       |  1,500 |

图表示用于补充手工时序和结构统计，为结构位置相近的候选提供可学习的表示差异。

### 最终排序器

Dataset1 使用 LightGBM LambdaRank：

| 参数                  | 默认值 |
| --------------------- | -----: |
| 训练窗口数            |      1 |
| 每个查询采样负样本数  |     24 |
| Boosting 轮数         |    900 |
| Learning rate         |   0.06 |
| Num leaves            |     95 |
| Feature fraction      |   0.85 |
| Bagging fraction      |   0.85 |
| LambdaRank truncation |     30 |

推理后对每行 100 个候选进行 min-max 归一化。

## Dataset2 基础模型

Dataset2 首先训练多个互补的基础模型。

### CRAFT

CRAFT 使用 Jittor/JittorGeometric 建模时序邻居和节点状态：

- 按时间顺序构造交互；
- 使用时序邻居采样；
- 结合节点最近更新时间；
- 使用正负边训练；
- 根据验证指标提前停止。

默认 Epoch 上限为 100，batch size 为 200，early stopping patience 为 10。

### 候选级树模型

树模型使用源节点历史、目标热度、节点对历史、时间间隔和局部结构等统计特征训练 LightGBM。默认抽取 100,000 个训练交互，每个正样本配置 5 个负样本。

### 时序统计排序器

时序 LambdaRank 使用：

- 节点度数和独立邻居数；
- 源节点活跃度；
- 节点对历史次数；
- 最近交互时间；
- 多尺度时间衰减强度；
- 近期序列转移支持；
- 二部图协同支持；
- 候选频率及已见节点标记。

默认使用前 90% 历史作为冻结前缀，从剩余数据中抽取 20,000 个训练查询和5,000 个验证查询。

### GraphMF

GraphMF 使用 Jittor 学习 64 维源节点和目标节点向量：

```text
score(src, dst) = dot(embedding_src, embedding_dst) + bias_dst
```

隐向量分数与时序统计特征拼接后，再训练候选排序器。默认训练 5 个 Epoch，batch size 为 4,096，每个正样本使用 10 个负样本。

### LightGCN

LightGCN 在完整二部图的归一化邻接矩阵上传播 GraphMF 表示，并融合不同层的节点向量。LightGCN 相似度与时序统计特征拼接，再由独立排序器输出候选分数。

### 基础模型融合

各通道先逐行归一化，再按照固定权重融合：

```text
classical =
    0.50 × tree
  + 0.30 × CRAFT
  + 0.20 × statistical

temporal_blend =
    0.65 × temporal_ranker
  + 0.35 × classical

graph_blend =
    0.50 × temporal_blend
  + 0.50 × GraphMF

base =
    0.65 × graph_blend
  + 0.35 × LightGCN
```

基础模型输出位于：

```text
outputs/dataset2/production/base/dataset2.csv
```

## Dataset2 高层特征

### Basket 共现

按相同或相近 `(src, time)` 形成查询组，结合训练历史中的物品共现关系构造：

- 候选与高分候选的共现强度；
- 近期共现和邻居支持数量；
- 当前 basket 内的相对支持；
- 基础分数与共现分数的差异。

### 候选分布与时间窗口

从测试输入中的候选集合构造：

- 候选全局出现频率；
- 相对候选背景频率的偏移；
- 行内频率排名；
- 训练热度与测试候选热度的差异；
- 多个时间窗口内的局部候选频率。

### 边际校准

基础模型首先给出候选软分配，再使用候选边际频率进行校准，并生成加权事件统计。派生特征包括多尺度 Hawkes 强度、最近软事件时间、独立源节点数和累计软权重。

### 物品协同过滤

物品侧通道基于训练历史中的稀疏用户—物品矩阵，包括：

- Item cosine；
- EASE；
- 潜在因子；
- 候选相对协同排名。

### 多尺度协同过滤

在多个时间半衰期和流行度抑制强度下构造衰减交互矩阵，生成短期和长期Item-CF 通道。

### 用户协同过滤

寻找与当前源节点历史兴趣相近的用户，并根据相似用户对候选目标的历史交互进行投票。

### 分组与图传播

分组特征利用：

- 相同 `(src, time)` 查询组大小；
- 候选在组内其他行中的出现次数；
- 高分候选重复支持；
- 历史连接候选比例和冷启动候选比例；
- 各基础分数的 rank、Z-score 和 top gap。

图传播通道进一步补充高阶图结构关系。

## Dataset2 训练与验证

### 滚动时间窗口

从 Dataset2 训练集构建 3 个滚动验证环境。每个环境包含：

- 目标窗口开始前的冻结历史；
- 窗口内的真实交互；
- 宽度为 100 的模拟候选查询；
- 随机化的正样本候选位置；
- 按 `(src, time)` 保持完整的查询组。

### OOF 预测

每个验证环境内部划分 OOF fold：

```text
train_mask   = row_fold != target_fold
predict_mask = row_fold == target_fold
```

使用监督标签训练的模型不会使用目标 fold 的标签预测该 fold。流水线为 OOF 产物记录 benchmark hash、prediction fold、模型名和随机种子。

### 跨窗口训练

验证某个滚动窗口时，上层模型使用其他两个窗口训练；生产阶段则使用三个滚动窗口共同训练最终模型。

### 多级重排序

Dataset2 上层模型依次包括：

1. Distribution ranker；
2. Collaborative ranker；
3. Temporal stack；
4. 30,000/60,000 行 Anchor ensemble；
5. Group ranker；
6. Graph ranker；
7. Final candidate ranker。

`validation_temporal_stack` 使用 `cold-weight=0.15`，与下游固定读取的 `cross_window_cw0p15.npz` 保持一致。

## 最终候选级模型

最终模型为每个候选构造 172 维特征，主要来自：

- 分组和图传播基础分数；
- Anchor、Distribution、Collaborative 和 Temporal Stack 分数；
- 各分数的行内排名、Z-score 和 top gap；
- 原始时序与候选窗口特征；
- 物品、多尺度和用户协同通道；
- 查询组、冷启动和图结构上下文。

最终使用 LightGBM 候选级二分类模型：

| 参数                   | 默认值 |
| ---------------------- | -----: |
| 每个滚动 fold 抽样行数 | 30,000 |
| Boosting 轮数          |    180 |
| 正样本权重             |   50.0 |
| Learning rate          |   0.03 |
| Num leaves             |     63 |
| Min data in leaf       |    300 |
| Feature fraction       |   0.82 |
| Bagging fraction       |   0.90 |
| 最终模型融合权重       |   0.45 |

最终预测为：

```text
final_score =
    0.55 × group_graph_base
  + 0.45 × final_candidate_model
```

融合后执行行内 min-max 归一化。

## 完整流水线顺序

`python3 run.py --target all` 依次执行：

1. Dataset1 排序器训练和推理；
2. Dataset2 三组滚动验证基准构造；
3. 三组严格 OOF 特征生成；
4. Temporal Stack 验证模型；
5. 30,000/60,000 行 Anchor 验证模型；
6. Group feature、Group ranker 和 Graph ranker 验证；
7. CRAFT、Tree、Temporal、GraphMF 和 LightGCN 训练；
8. Dataset2 基础模型推理与融合；
9. Basket 和时序特征构造；
10. 候选分布和边际校准；
11. 物品、多尺度及用户协同通道；
12. Distribution ranker 训练和推理；
13. Collaborative ranker；
14. Temporal Stack、Anchor Ensemble 和 Group Ranker 生产推理；
15. 最终 172 维候选级模型训练和推理；
16. 输出检查和 ZIP 打包。

## 输出文件

最终目录：

```text
submissions/prediction/
├── dataset1.csv
├── dataset2.csv
├── manifest.json
└── result.zip
```

输出检查包括：

- Dataset1 预测形状为 `61,051 × 100`；
- Dataset2 预测形状为 `153,420 × 100`；
- 所有值均为有限浮点数；
- 所有值位于 `[0, 1]`；
- CSV 保留 8 位小数；
- ZIP 内包含 `dataset1.csv` 和 `dataset2.csv`。

`manifest.json` 记录两份预测文件及 ZIP 的 SHA256。

## 断点续跑

每个成功阶段会写入：

```text
outputs/state/<stage>.json
```

状态文件记录阶段名、完成时间、执行命令和源码指纹。源码指纹一致时，再次运行会跳过已完成阶段。

> [!note]
>
> 注意：不要只删除阶段产物而保留对应状态文件。需要重新生成时，应同时删除对应状态文件，或使用：
>
> ```bash
> python3 run.py --target all --force-stage <stage>
> ```
>
> 如需完全从头运行，应删除整个 `outputs/` 和 `submissions/` 后重新执行入口。

## 资源优化

Dataset2 会生成多组 `rows × 100 × features` 大型数组。工程实现采用：

- NumPy memmap；
- 分块特征构造和分块预测；
- SciPy 稀疏矩阵；
- 图邻居 Top-K 裁剪；
- EASE 物品集合裁剪；
- 训练查询抽样；
- 特征及模型缓存；
- 阶段级断点续跑。

数据规模增大时，可以在不改变特征族和训练逻辑的前提下调整 batch size、线程数、邻居上限、预测分块和训练采样规模。

## 可复现性

完整入口只使用一个基础随机种子，默认值为 `20260724`：

```bash
python3 run.py --target all --seed 20260724
```

该值传递给 Dataset1、Dataset2 基础模型、滚动验证、OOF 特征和上层排序器；所有模块的独立默认值也统一为 `20260724`，不使用额外偏移。实际执行命令记录在`outputs/state/<stage>.json`。

CUDA、Jittor 编译器及 LightGBM 多线程调度可能造成浮点末位差异，因此不要求不同机器生成完全相同的 ZIP 哈希，但输出结构和排序结果应保持一致。

由于先前在不同阶段使用了不同种子，之后整理源码时统一使用一个基础随机种子，因此复现结果可能与线上提交结果存在些许出入。


## 常见问题

### Jittor 首次启动较慢

首次运行会编译 CUDA/C++ 算子，这是正常现象。请不要在编译过程中终止任务。

### 某阶段提示文件不存在

首先检查其上游状态文件是否存在但产物已被删除：

```bash
python3 run.py --list-stages
find outputs/state -maxdepth 1 -type f | sort
```

然后强制重跑缺失产物对应的上游阶段。

## 第三方组件

本项目通过 `requirements.txt` 使用以下主要开源组件：

- [Jittor](https://github.com/Jittor/jittor)：张量计算和神经网络训练；
- [JittorGeometric](https://github.com/AlgRUC/JittorGeometric)：图神经网络与
  时序图组件，安装版本固定到 requirements 中记录的提交；
- [LightGBM](https://github.com/microsoft/LightGBM)：候选排序与树模型；
- NumPy、Pandas、SciPy、scikit-learn 和 NetworkX：数值计算、数据处理与
  图结构计算。

上述组件未作为源码复制进本仓库，使用时分别遵循其上游项目许可证。赛事原始数据不随源码发布，本项目也不附带第三方预训练权重。

## 许可证

代码许可证见 [LICENSE](./LICENSE)。赛事原始数据不包含在本许可证和源码包中。
