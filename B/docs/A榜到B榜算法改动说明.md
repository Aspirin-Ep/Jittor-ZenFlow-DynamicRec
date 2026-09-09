# A 榜到 B 榜算法改动说明

## 1. 总体原则

B 榜保留 A 榜的任务建模、模型家族、主要特征语义、训练目标和候选级排序逻辑。新增代码主要解决 Dataset3/4 节点与交互规模显著增大后，A 榜稠密或全量内存计算无法在约 18GB 内存下执行的问题。

B 榜审核源码统一放在 `code/`，同时保留 Dataset1/2 实现，因此仍可运行 A 榜流程：

```bash
python run.py --target a --seed 20260724
```

## 2. Dataset3 与 Dataset1

Dataset3 与 Dataset1 都是非二部时序图，B 榜直接复用以下核心：

- 历史时序统计、重复交互、衰减强度与候选热度特征；
- 互惠查询和局部结构特征；
- Jittor 图表示与 LightGCN 传播；
- 候选级 LightGBM 排序；
- 按查询分组并以 MRR 选择轮数，最终逐行归一化。

规模和数据接口调整：

- 使用官方 `split=0` 作为历史、`split=1` 作为有标签未来窗口；
- 节点数由数据动态推导，不再依赖 A 榜固定规模；
- 大型训练特征写入 memmap，分块构造并支持恢复；
- LightGCN 嵌入保存为磁盘 checkpoint。

## 3. Dataset4 与 Dataset2

Dataset4 与 Dataset2 都是二部时序推荐图。B 榜保留：

- CRAFT 时序邻居编码；
- Tree、Temporal、Hawkes、重复周期和物品转移特征；
- GraphMF 与 LightGCN；
- Basket/Group 重排；
- Candidate Distribution 与 Marginal Calibration；
- Item/User/Multiscale Collaborative Filtering 与 EASE 通道；
- 基础通道、上下文通道和最终候选级 LambdaRank 多层堆叠。

主要规模适配如下：

| 项目 | A 榜实现语义 | B 榜规模适配 |
|---|---|---|
| 数据读取 | Pandas/NumPy 时序边与查询 | CSV 流式预处理为紧凑整数 memmap |
| CRAFT | 两层两头时序邻居模型、BPR | 双向紧凑时间索引；每 epoch 抽样边；固定 30 邻居 |
| GraphMF | 二部图 BPR 表示 | 大 batch、5 个负样本、尾窗选择 epoch、epoch checkpoint |
| LightGCN | 归一化图传播 | SciPy 稀疏矩阵逐层传播并保存恢复点 |
| CF/EASE | 物品/用户/多尺度协同与 ridge EASE | 128 维 CountSketch 和投影空间 ridge 近似，磁盘映射中间量 |
| 排序训练 | 多 fold 候选级树模型 | 单时间前向链、120000 条训练查询、分块特征 |
| 推理 | 全量候选打分 | 每块 2500 行，逐块写 CSV 并保存字节级恢复点 |

CountSketch CF 与投影 EASE 保持协同/岭回归通道的建模语义，但不声称与 A 榜稠密实现数值等价。CRAFT 抽样量、epoch、负样本数、验证查询数和 fold 数属于公开的资源超参数差异。

## 4. 时间前向训练与标签边界

Dataset3/4 的有监督验证只使用官方训练集标签：

- 对应验证特征的图历史冻结在 `split=0`；
- `split=1` 只作为后续有标签未来窗口；
- Dataset4 将抽样查询按时间切为基础模型、上下文模型、元排序器和最终验证四段；
- 后层模型只使用更早时间段产生的前向分数，避免同一标签同时训练上下两层；
- 正式推理前，生产基础模型使用完整训练边重训。

## 5. 工程性调整

以下改动不改变模型目标：

- 将通用数据检查和流式 CSV 校验移入 `solution/common`；
- 增加阶段状态、源码/数据指纹和输出存在性检查；
- 使用 memmap、稀疏矩阵和分块写出控制峰值内存；
- 为 GraphMF、CRAFT、LightGCN 和正式推理增加恢复点；
- 限制 NumPy、SciPy、LightGBM 与特征线程数，避免 18GB 环境下线程缓冲膨胀；
- 将源码目录从 `src/` 行政性更名为审核所需的 `code/`。

逐阶段映射见 `docs/A_B核心算法映射.md`，完整模型和特征说明见 `docs/完整算法说明.md`。
