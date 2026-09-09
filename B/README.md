# ZenFlow 赛道一 B 榜代码审核材料

> 参赛队伍：ZenFlow

本目录是第六届计图人工智能挑战赛“基于图学习的动态推荐任务”的 B 榜代码审核材料。队伍记录的该版本线上成绩为 **1.4148**。由于此次运行产生的模型与中间文件已删除，当前审核材料随附的模型文件是后续再次运行时生成的。利用此模型文件产生的预测文件实际评分可能与线上记录最高得分有细微出入，但再次运行未对种子、超参数和实验环境做任何更改，因此可认为两次运行结果等价，可能存在的得分细微差异属正常误差。

## 目录内容

```text
.
├── run.py                          # A/B 榜统一运行入口，默认运行 B 榜
├── requirements.txt                # 固定版本 Python 依赖
├── README.md/.pdf                  # 正式说明文档
├── code/                           # Dataset1–4 全部算法源码
├── scripts/                        # Shell 启动和服务器环境脚本
├── checkpoints/                    # 最优结果生产推理使用的模型权重
├── docs/
│   ├── 完整算法说明.md/.pdf
│   ├── B榜最优结果复现说明.md/.pdf
│   ├── A榜到B榜算法改动说明.md/.pdf
│   └── A_B核心算法映射.md/.pdf
```

## 最优结果复现

本材料提供两种可复现路径。复核最优提交时，应直接加载归档权重，不重新训练：

```bash
python scripts/run_checkpoint_inference.py \
  --target b \
  --workers 8 \
  --chunk-rows 2500 \
  --cf-dim 128 \
  --seed 20260724
```

该命令加载 `checkpoints/` 中的 Dataset3 LambdaRank/GCN 与 Dataset4 GraphMF、CRAFT、组件模型、上下文模型和最终排序模型；只重建推理所需的紧凑数组、时间索引、稀疏传播与协同缓存。自动生成 `submissions/checkpoint_inference/result.zip`。从零训练复现仍可使用 `run.py`。

详细环境、数据布局、输出和断点续跑方法见 [B榜最优结果复现说明](docs/B榜最优结果复现说明.md)。算法细节见 [完整算法说明](/docs/完整算法说明.md) 。A/B 榜算法对应关系及规模适配见 [A榜到B榜算法改动说明](docs/A榜到B榜算法改动说明.md)。

## 数据合规说明

- 本目录不包含 `dataset1/`、`dataset2/`、`dataset3/`、`dataset4/` 或任何原始 CSV。
- `checkpoints/` 只包含复现推理所需的学习型模型权重，不包含由数据构建的邻接索引、memmap 特征、协同缓存或训练过程状态。
- 复现时由统一入口检查官方输入文件的表头、候选宽度和输出格式。
