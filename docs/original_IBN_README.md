# 浅层 C3k2 的 IBN 仿真实验

【新增文档，2026-11-11】本目录从原始仿真 `best.pt` 构建完整的 YOLO26n-sem + IBN 模型，只在第一处浅层 C3k2 中把 `BN(32)` 换为 `IN(16) + BN(16)`。原 Ultralytics、Conv-Adapter、KAN、边缘分支程序及原始权重均保留，不在原文件内做修改。

**当前交付的是初始化模型和可运行程序；尚未完成正式训练，也没有真实域提升结论。** 已在 CPU 上通过结构、梯度、融合、重载、预测检查，以及 8 张仿真训练图 / 4 张验证图的一轮流程检查。

## 改了哪里

打开 [模块结构图](structure.svg)。核心实现见 [ibn_backbone.py](ibn_backbone.py)，新增处标记为 `【新增】`，替换处标记为 `【修改 IBN-xx】`，原写法用 `【原结构，注释保留】` 等注释保留在对应位置。

```text
原结构 model[2].cv1：Conv 1×1 (32→32) → BN(32) → SiLU
新结构 model[2].cv1：Conv 1×1 (32→32) → IBN    → SiLU
                                          ├─ 前 16 通道：IN
                                          └─ 后 16 通道：BN
                                   按原顺序拼回 32 通道
```

这里的 IN 按每张图、每个通道计算空间均值和方差，BN 保留另一半通道的批归一化处理。目的是尝试减少浅层特征对仿真亮度、对比度和纹理统计的依赖，同时保留部分原有判别信息。这是针对外观差异的实验假设，不能补齐仿真中缺失的坑形态、视角或尺度。

这是借鉴 IBN 的局部替换实验，不是完整复现 IBN-Net。在本模型的 C3k2 中，投影后的前半通道进入旁路，后半通道进入 Bottleneck。因此此版本的 IN 半支对应旁路，BN 半支对应 Bottleneck 输入，没有另外打乱通道。

其他 C3k2、下采样层、颈部和分割头的结构保留。此模型从原始仿真权重开始，未叠加 KAN 或边缘分支。

## 张量大小

以下是实际输入 `[B, 3, 1024, 1024]` 时的大小，已用 `B=1` 验证。高宽不同的输入按实际尺寸变化。

| 位置 | 张量大小 |
|---|---|
| `model[2]` 浅层 C3k2 输入 | `[B, 32, 256, 256]` |
| `model[2].cv1.conv` 的 1×1 卷积输出 | `[B, 32, 256, 256]` |
| IN 分支 / BN 分支 | 各 `[B, 16, 256, 256]` |
| IBN 拼接输出 | `[B, 32, 256, 256]` |
| `model[2]` 浅层 C3k2 输出 | `[B, 64, 256, 256]` |
| 原来接适配器的颈部 P3，即 `model[16]` 输出 | `[B, 64, 128, 128]` |
| 分割头原始 logits | `[B, 5, 128, 128]` |

预测程序会将分割结果还原到图像尺寸。IBN 不改变通道总数和空间尺寸，完整模型的可学习参数总数仍为 **1,631,082**。

## 权重与训练范围

- 起点：`D:\Projects\Sim2real\runs\yolo26n_sem_50epochs_1024\weights\best.pt`。
- 交付权重：`weights/yolo26n_sem_ibn_init.pt`，保存完整模型；它只完成结构替换和参数迁移，没有正式训练。
- 原 BN 的前半仿射参数复制给 IN，后半仿射参数和运行统计复制给 BN。其他 359 个状态张量在初始化时与原模型逐项相等。
- IN 改变了归一化统计的计算方式，所以新模型初始化输出不保证与原模型相同。
- **正式训练更新整个骨干、颈部和分割头，包括 IN/BN 的仿射参数；不冻结原骨干。** 这使 P3 之前的卷积也能适应新的归一化。
- BN 运行统计只在仿真训练中更新；预测时固定。IN 推理时计算当前图像的统计，不积累真实数据统计、不更新参数。
- 沿用原语义分割损失，没有加入新的坑边界损失。模型选择使用仿真验证集。

训练脚本使用完整已改造模型，避免普通 `YOLO(...).train()` 按旧 YAML 重建模型而丢失 IBN。`ConvWithIBN` 使用独立模块类型，避免预测阶段错误地将 IBN 当作普通 BN 融合进卷积。

## 数据与类别

只使用从 `D:\Projects\Sim2real\data\simulated_data\Traindata` 准备的现有数据：

```text
D:\Projects\Sim2real\data\simulated_data\yolo26_semantic_data\data.yaml
  train：8000 张
  val：1320 张
```

训练入口只读取上述仿真划分；不会扫描或加入真实图像。标签为单通道类别索引：`0 月壤、1 撞击坑、2 岩石、3 山体、4 天空`，`255` 为忽略区域。真实数据只用于训练后的预测和评价。

脚本会检查已有仿真 train/val 缓存是否与数据一致；发现不一致则停止，不在原数据目录自动重建缓存。

## 运行命令（PowerShell）

使用现有 Python 环境，不需要重新安装 Ultralytics。以下命令可从任意工作目录执行。

### 开始正式训练

```powershell
& 'D:\Projects\[Seg]\ultralytics-main\.venv-yolo\Scripts\python.exe' 'D:\Projects\Sim2real\imp\IBN\train_ibn.py' --train --epochs 30 --imgsz 1024 --batch 4 --device 0
```

默认 AdamW、学习率 `1e-4`、早停耐心值 `10`、固定随机种子 `42`、单 GPU、`workers=0`。为保持已验证的执行路径，默认 `amp=False`。1024 的完整模型训练显存需求还没有在 GPU 上实测，显存不足可将 `--batch` 降至 2 或 1。

日志和权重保存到 `runs/ibn_sim_1024/`；目录已存在时由训练器生成新的实验目录。`weights/best.pt` 是按仿真验证指标选择的**完整 YOLO + IBN 模型**，`last.pt` 是最后一轮完整模型。

如指定 `--weights`，必须使用本实验生成的 IBN 完整模型。此选项开始新的微调，不恢复旧优化器状态；脚本没有提供中断续训入口。

### CPU 小样本流程检查

```powershell
& 'D:\Projects\[Seg]\ultralytics-main\.venv-yolo\Scripts\python.exe' 'D:\Projects\Sim2real\imp\IBN\train_ibn.py' --smoke --device cpu
```

此命令复制 8 张仿真训练图和 4 张验证图到本目录，在 256 输入下训练 1 轮，输出至 `runs/smoke/`。它只检查程序，不用于比较模型精度。交付目录没有把这次小检查的 `best.pt` 冒充正式权重。

### 结构与加载检查

```powershell
& 'D:\Projects\[Seg]\ultralytics-main\.venv-yolo\Scripts\python.exe' 'D:\Projects\Sim2real\imp\IBN\verify_ibn.py'
```

使用 CPU 检查实际 1024 输入形状、参数迁移、梯度、预测融合和保存重载，报告写入 `checks/verification.json`。梯度更新只发生在内存中的检查模型上，不修改交付初始权重。

### 用训练后的 best.pt 预测

```powershell
& 'D:\Projects\[Seg]\ultralytics-main\.venv-yolo\Scripts\python.exe' 'D:\Projects\Sim2real\imp\IBN\predict_ibn.py' 'D:\Projects\Sim2real\data\real_data\CE3' --weights 'D:\Projects\Sim2real\imp\IBN\runs\ibn_sim_1024\weights\best.pt' --imgsz 1024 --device cpu --name ce3_ibn
```

请按实际训练目录填写 `--weights`。输出在 `predictions/ce3_ibn/`，包括原预测可视化和 `class_masks/` 下的 0～4 索引掩膜。索引掩膜不等于真实数据的 RGB 标注颜色，此命令不计算评价指标。空闲 GPU 可用 `--device 0`。

## 文件用途

| 文件 | 用途 |
|---|---|
| `ibn_backbone.py` | IBN 结构、BN 参数迁移、插入位置检查 |
| `build_ibn.py` | 从原始 best.pt 构建完整初始化模型；拒绝覆盖已存在输出 |
| `train_ibn.py` | 仿真训练与小样本流程检查 |
| `predict_ibn.py` | 载入自定义 IBN 模型并预测 |
| `verify_ibn.py` | CPU 结构、梯度、融合与重载检查 |
| `runtime.py` | 现有源码、Python 环境及数据路径配置 |
| `weights/build_audit.json` | 初始权重来源、原文件哈希及参数迁移记录 |
| `checks/verification.json` | 实际结构及功能检查结果 |
| `checks/smoke_training_audit.json`、`smoke_results.csv` | 已完成的一轮小样本流程检查记录 |
| `delivery_manifest.json` | 交付文件及 SHA-256 校验清单 |

检查记录中的工作区路径保留当时的实际测试位置。小样本数据和权重仍保留在 `D:\Projects\[Seg]\ultralytics-main\sim2real_ibn`，没有复制为正式实验结果。

自定义权重依赖 `ibn_backbone.py` 及本项目的语义分割源码。请保留整个程序目录，使用提供的入口加载；不是只把 `.pt` 拿到普通 pip Ultralytics 中就能运行。移动项目时可通过 `IBN_ULTRALYTICS_ROOT` 指定源码根目录，其他数据路径在 `runtime.py` 中。

## 参考与验证边界

IBN 思路来自 [Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net（ECCV 2018）](https://arxiv.org/abs/1807.09441)，分通道实现参考 [作者官方代码](https://github.com/XingangPan/IBN-Net/blob/master/ibnnet/modules.py)。许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。此项目采用的是浅层 C3k2 内的一处局部改造，不是论文完整架构。

目前验证了程序可运行，未验证真实撞击坑召回率改善。下一步应在相同仿真划分和训练预算下，与“原结构全模型继续训练”对照，避免把继续训练的收益误归因于 IBN。真实开发集可以用于分析，但反复据其结果选择方案后，最终结论还需独立真实测试集。
