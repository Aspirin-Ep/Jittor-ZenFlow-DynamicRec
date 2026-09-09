# B 榜最优结果复现说明

## 1. 复现目标

本目录保留产生最优结果所需的学习型模型 checkpoint。推荐复现路径为：从官方原始 CSV 构建必要的紧凑输入与推理索引，然后直接加载 checkpoint 输出 Dataset3、Dataset4 分数；该路径不执行任何监督模型训练。固定随机种子为 `20260724`。

队伍记录的该版本线上成绩为 1.4148，线上成绩以赛事评测系统记录为准。


## 2. 环境配置

推荐审核环境：Ubuntu 22.04、Python 3.10、CUDA 12.4 及兼容的 NVIDIA 驱动。依赖版本由根目录 `requirements.txt` 固定，其中 Jittor 为 1.3.10.0，JittorGeometric 使用固定源码提交；同时包含 NumPy、Pandas、SciPy、scikit-learn 和 LightGBM 等依赖。

本项目开发验证使用过 WSL2 和 Python 3.11；这不改变提交环境的推荐配置。首次启动 Jittor 时可能编译 CUDA/C++ 算子，耗时会高于缓存后的运行。

## 3. 数据布局

本审核材料不提供赛事数据。请把官方数据放在项目根目录：

```text
dataset3/
├── train.csv
└── test.csv
dataset4/
├── train.csv
└── test.csv
```

程序会检查训练表头 `src,dst,time,split`，并检查测试表包含 `src,time,c1,...,c100`。不要把数据集放入最终代码审核压缩包。

## 4. 安装依赖

在 Linux CPython 环境中执行：

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

详细的 CUDA、cuDNN 和无管理员权限环境配置见 `docs/完整算法说明.md`。

## 5. 直接加载 checkpoint 复现

确认根目录包含官方数据并保留 `checkpoints/` 后运行：

```bash
python scripts/run_checkpoint_inference.py \
  --target b \
  --workers 8 \
  --chunk-rows 2500 \
  --cf-dim 128 \
  --seed 20260724
```

该入口会先校验所有归档 checkpoint 的 SHA256，再加载以下权重：

- Dataset3：LambdaRank 与 LightGCN；
- Dataset4：GraphMF、CRAFT、4 个基础组件模型、6 个上下文模型和最终 LambdaRank。

为计算这些已保存模型的输入，它会构建紧凑 CSV 数组、时间历史索引、LightGCN 稀疏传播结果、协同过滤缓存和候选频次缓存。这些是确定性的推理派生物，不会拟合或更新任何学习型模型。可中断后用相同命令继续，派生缓存会按文件状态复用。

默认结果位置：

```text
submissions/checkpoint_inference/
├── dataset3.csv                # 157670 × 100
├── dataset4.csv                # 2322538 × 100
├── manifest-b.json              # 两个 CSV 与 ZIP 的 SHA256
└── result.zip                   # 内含 dataset3.csv、dataset4.csv
```

若只需复核单个数据集，可将 `--target b` 改为 `--target dataset3` 或 `--target dataset4`。`--max-test-rows N` 仅用于 Dataset4 冒烟检查，正式复现不要设置该参数。若两个 CSV 已生成但缺少 ZIP，可执行 `python scripts/run_checkpoint_inference.py --package-only` 单独校验并打包。

## 6. 从零完整复现（可选）

确认根目录没有旧的 `outputs/`、`submissions/` 后运行：

```bash
python run.py \
  --target b \
  --workers 8 \
  --chunk-rows 2500 \
  --cf-dim 128 \
  --seed 20260724
```

等价 Shell 入口：

```bash
bash scripts/run_pipeline.sh \
  --workers 8 \
  --chunk-rows 2500 \
  --cf-dim 128 \
  --seed 20260724
```

建议资源：不少于 18GB 内存、40GB 可用磁盘和 NVIDIA CUDA GPU。`chunk-rows=2500` 是 18GB 内存环境的安全配置；增加该值只影响分块效率，不改变模型结构。

也可以分别运行单个 B 榜数据集：

```bash
python run.py --target dataset3 --workers 8 --seed 20260724
python run.py --target dataset4 --workers 8 --chunk-rows 2500 --cf-dim 128 --seed 20260724
```

两条命令分别生成 `submissions/prediction/dataset3.csv` 和 `submissions/prediction/dataset4.csv`。两个数据集均完成后执行 `python run.py --target package-b` 生成 B 榜 `result.zip`。

## 7. 输出与校验

完整运行结束后生成：

```text
outputs/                         # 模型、checkpoint、可恢复缓存和状态
submissions/prediction/
├── dataset3.csv                # 157670 × 100
├── dataset4.csv                # 2322538 × 100
├── manifest-b.json             # CSV 与 ZIP 的 SHA256
└── result.zip                  # 内含 dataset3.csv、dataset4.csv
```

每行输出 100 个 `[0,1]` 分数并保留 8 位小数。统一入口会逐行检查行数、列数、有限值和范围，再生成 ZIP。

CUDA、Jittor 和 LightGBM 多线程归约可能导致浮点末位变化；判断复现成功应以输出结构、有限值、范围及排序效果为主。

## 8. 训练选择结果

- Dataset3 LambdaRank：验证 MRR 早停选择 187 轮；
- Dataset4 GraphMF：选择 5 个 epoch；
- Dataset4 CRAFT：选择 20 个 epoch；
- Dataset4 最终 LambdaRank：固定时间前向验证 MRR `0.493469`，选择 197 轮；
- Dataset4 验证查询：120000。

训练过程记录不属于最终复现所需文件，因此未随审核包提交。

## 9. 断点续跑

再次执行完全相同的命令时，程序会根据源码、数据文件信息、命令参数和产物状态跳过已完成阶段。Dataset4 的预处理、GraphMF、CRAFT、LightGCN、验证特征和正式推理均包含更细粒度恢复点。

若需要从零复验，应先把旧的 `outputs/` 和 `submissions/` 移出项目目录，而不是仅使用 `--force-stage`；后者仍可能复用阶段内部缓存。

`checkpoints/` 是供审核和权重留档的精简副本。直接加载模式将派生缓存写入 `outputs/checkpoint_inference/`；完整从零复现则使用 `outputs/` 下的训练产物和缓存。
