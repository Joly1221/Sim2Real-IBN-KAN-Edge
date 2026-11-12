# P3 KAN 适配器实验

## 官方代码与来源

- KAN 作者官方源码：[KindXiaoming/pykan](https://github.com/KindXiaoming/pykan)，本地目录 `official_pykan`，提交 `ecde4ec3274d3bef1ad737479cf126aed38ab530`。
- 官方 `KANLayer` 和样条实现保留原文件，未修改；许可证见 `official_pykan/LICENSE`。仅为导入官方代码，在本实验 `dependencies` 目录安装 scikit-learn 1.5.2、joblib 1.4.2、threadpoolctl 3.5.0，没有向原虚拟环境安装它们。
- 原 MLP 适配器代码 `../sim2real_mlp_adapter/mlp_adapter.py` 保持不变。`kan_adapter.py` 中替换的 MLP 代码保留为注释，并以【原 MLP 源代码】【新增】【修改】标注。

## 模型结构

从 `D:\Projects\Sim2real\runs\yolo26n_sem_50epochs_1024\weights\best.pt` 构建，而不是从 MLP 或 Conv-Adapter 权重构建。原始权重 SHA-256：`2f3b9d4650a6174dc801923dafc10ca389981118752cde384bc2b1b881c1e996`。

P3 特征 `B×64×128×128` → 原特征与 `[1×1卷积 64→16 → SiLU → 官方 KANLayer 16→16 → 1×1卷积 16→64]` 的输出残差相加 → 原语义分割分类器。KAN 对每个空间位置共享同一组参数；它不额外融合相邻位置。

KAN 层使用官方 `KANLayer`，5 个网格区间、3 阶、初始网格范围 `[-1,1]`，这是实验起点而非已优化设置。官方 KAN 的 `grid` 和 `mask` 固定，不能和可训练系数一起解除冻结。KAN 的 `scale_base`、`scale_sp` 和 `coef` 等可学习参数参与训练；此实现没有使用真实图像更新网格或参数。

## 文件

- `kan_adapter.py`：接入和冻结逻辑，修改处有中文标注。
- `build_kan.py`：从原始仿真 `best.pt` 构造初始模型；原权重和现有模型不受影响。
- `yolo26n_sem_kan_init.pt`：包含完整原模型与新 KAN 分支的**未训练初始权重**。
- `verify_kan.py`：初始等价、形状、冻结、梯度和合成资源检查。
- `train_kan.py`：仿真训练脚本；`--smoke` 是小样本流程检查。2026-11-10 已启动一次完整训练，实时状态见 `runs/kan_sim_1024/progress.json`。
- `grid_probe_smoke.json`：前 8 张仿真训练图像、1024 推理预处理下的初始网格范围抽查；非随机样本，且不包括训练增强，不能代表整体或真实域分布。

## 已完成的检查

未训练时，1024 输入下 P3 和适配器输入输出均为 `1×64×128×128`，与原模型最大输出差为 0。可训练参数 4,688 个；两次合成优化器更新中 KAN 系数收到梯度，原模型参数、BN 缓冲区及固定网格/掩码不变。RTX 4090 上一次合成 1024、batch 4 的前向及反向峰值 allocated 约 892.5 MiB；这不是完整训练的显存峰值或速度基准。

8 张仿真训练图、4 张仿真验证图、256 输入的一轮 smoke 训练已完成，审计见 `runs/smoke/adapter_audit.json`。真实图像没有参与训练；smoke 指标不用于判断模型效果。8 张原仿真图初始输入特征抽查中，约 0.057% 的值超出 KAN 名义网格区间 `[-1,1]`；官方样条网格有扩展边界和基础函数，此数字不等于样条贡献为零的比例。正式实验前仍应检查更多仿真场景及训练增强下的分布。

## 运行检查

在项目根目录执行：

```powershell
.\.venv-yolo\Scripts\python.exe .\sim2real_kan_adapter\verify_kan.py
.\.venv-yolo\Scripts\python.exe .\sim2real_kan_adapter\train_kan.py --smoke
```

2026-11-10 已启动 `train_kan.py --epochs 10`：1024、batch 4、AdamW、最多 10 轮、patience 5、seed 42，只用仿真数据。当前正在运行，无需重复启动；实时进度记录在 `runs/kan_sim_1024/progress.json`，控制台日志为 `training_kan_20261110.log`。每轮会核对原模型及 KAN 固定网格和掩码；完整训练结束后才会生成最终审计。

用户指定的数据源为 `D:\Projects\Sim2real\data\simulated_data\Traindata`。为沿用 MLP 的配置，训练实际读取其已转换副本 `D:\Projects\Sim2real\data\simulated_data\yolo26_semantic_data`：训练和验证图像、标签的文件名集合一致；2026-11-10 每个集合随机抽查 20 个文件，共 80 个文件，哈希均一致。此前 2026-11-06 已对数据副本做过全量检查，但不能将那次检查当作本次全部文件的重新核对。真实图像只在模型固定后用于预测或评价。不可直接使用普通 `YOLO(...).train()`，它可能重建模型或重置冻结状态。
