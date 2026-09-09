# A/B 榜核心算法映射

本文件用于说明 Dataset4 相对 Dataset2 的规模适配边界。B 榜保留模型家族、输入语义、训练目标和候选级堆叠逻辑；只缩减 epoch、采样量、fold 数、邻居数和中间表示规模，或将不可落盘的稠密计算替换为有明确误差边界的稀疏/投影实现。

| A 榜 Dataset2 阶段 | B 榜 Dataset4 实现 | 保持一致的核心 | 规模适配 |
|---|---|---|---|
| CRAFT | `dataset4/craft.py` | 两层两头时序邻居编码、BPR、最近更新时间、30邻居 | 每 epoch 采样边、验证抽样、紧凑 dst 词表 |
| Tree/Temporal | `train_component_rankers` | 候选级 LightGBM、时间冻结统计、独立模型 | 35维向量化核心特征（含Hawkes、重复周期和物品转移）、训练查询抽样 |
| GraphMF | `dataset4/model.py` | 64维二部图 BPR、hard negatives、AdamW | 大 batch、5个负样本、时间尾窗选 epoch |
| LightGCN | `propagate_lightgcn` | GraphMF 表示上的两层归一化二部图传播 | SciPy 稀疏传播、层级检查点 |
| Candidate Distribution | `candidate_distribution_ranker.txt` | 独立候选分布排序模型 | 单时间fold、候选频率特征白名单 |
| Marginal Calibration | `marginal_ranker.txt` | 独立边际/热度校准模型 | 单时间fold、向量化特征 |
| Item/User/Multiscale CF、EASE | `SketchCollaborative` | 用户物品协同、近期/全量、多尺度、ridge EASE通道 | 128维CountSketch与投影空间ridge近似；磁盘映射控制驻留内存 |
| Collaborative Stack | `collaborative_ranker.txt` | 独立协同通道排序器 | 单时间fold |
| Anchor Ensemble | `anchor_ranker.txt` | 基础模型和协同通道的可学习融合 | 单时间fold、无30k/60k双模型 |
| Group/Graph | `group_graph_ranker.txt` | Basket、重复候选、图/基础模型分组重排 | 分块计算、单时间fold |
| Temporal Stack | `temporal_stack_ranker.txt` | 多通道可学习时序堆叠 | 单时间fold |
| Final Candidate Ranker | `final_ranker.txt` | 候选级LambdaRank、按查询分组 | 61维、120k分层验证查询；按MRR早停并保存完整曲线、验证内容哈希、特征族gain和直接列消融诊断 |

## 时间前向训练

Dataset4 的抽样验证查询按时间排序，并划分为四段：

1. 基础模型拟合与早停；
2. 六个上下文模型拟合与早停；
3. 最终候选排序器拟合；
4. 最终 MRR 验证。

监督模型只在比其评估区间更早的标签段上拟合。生产基础模型会使用完整 split=1重训；生产上下文模型仍使用早期基础模型对后续标签段生成的前向分数训练，避免同一标签同时拟合上下两层。正式测试推理再使用所有生产重训模型。

第一阶段验证只更改最终候选排序器的轮数选择指标：仍以 LambdaRank 为训练目标，但在固定的第四段上直接按稳定候选顺序 MRR 早停。同折直接时序列消融使用相同指标；下层基础模型和六个上下文模型使用 NDCG/AP 早停，避免一次实验同时改变多个层级。

## 不声称数值等价的部分

Dataset4 的 CountSketch CF 和投影 EASE 是 Dataset2 稠密协同计算的规模近似，模型语义和输出通道一致，但不保证数值等价。CRAFT 每轮抽样、训练 epoch、GraphMF 负样本数、验证查询数和 fold 数属于公开的规模超参数差异。
