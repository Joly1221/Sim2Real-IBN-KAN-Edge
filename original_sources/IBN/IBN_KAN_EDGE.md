# IBN + 单层 KAN 适配器 + 边缘分支

【新增说明，2026-11-12】这是一个组合版本。按原有 `YOLO26n-sem + 单层 KAN + 边缘分支` 的相同位置，在当前已训练 IBN 模型的 P3 后、原语义分类器前接入两项模块。

**底座 IBN 已经完成仿真训练；新 KAN 和边缘分支目前只初始化，尚未正式训练。** 交付的初始权重不包含小样本检查产生的更新。

## 结构与位置

[打开结构示意图](ibn_kan_edge_structure.svg)

```text
图像 → 已训练的 IBN 骨干和颈部 → P3 特征 x
                                      │
              ┌───────────────────────┤
              │                       ↓
              │         1×1 Conv 64→16 → SiLU
              │         → 单层 KAN(16→16) → 1×1 Conv 16→64
              └───────────────────────＋ → z = x + KAN(x)
                                      │
              ┌───────────────────────┤
              │                       ↓
              │         3×3 Conv 64→16 → SiLU → 边缘特征 f
              │                                  ├→ 1×1 Conv 16→1 → 边界损失
              │                                  └→ 1×1 Conv 16→64
              └──────────────────────────────────＋ → z + 边缘残差
                                                   ↓
                                           原语义分类器 → 五类分割
```

主路径公式：`z = x + KAN_adapter(x)`，`logits = 原分类器(z + edge_residual(z))`。边缘分支的单通道输出只提供训练辅助监督；边缘残差仍参与实际分割预测。

- P3 为颈部 `model[16]` 的输出。
- 两项模块在 `model[17].classifier` 内，分别位于 `kan_classifier.adapter` 和 `edge`。
- IBN 原位置仍为浅层 `model[2].cv1.bn`，没有再次替换或移动。
- 完全复用之前的 `kan_adapter.py`、`edge_branch.py` 结构和边界损失实现，旧文件没有修改。

## 输入输出张量

下表对应 `[B,3,1024,1024]` 输入。已用 `B=1` 实际验证。

| 位置 | 张量大小 |
|---|---|
| P3 / KAN 输入 x | `[B,64,128,128]` |
| KAN 的 1×1 降维后 | `[B,16,128,128]` |
| 单层官方 KAN 输入与输出 | `[B×16384,16]` |
| KAN 升维输出 / 残差相加后的 z | `[B,64,128,128]` |
| 边缘分支 3×3 卷积特征 f | `[B,16,128,128]` |
| 单通道边界 logits | `[B,1,128,128]` |
| 边缘残差 / 原分类器输入 | `[B,64,128,128]` |
| 原分类器输出 logits | `[B,5,128,128]` |

这里始终只有 **一层 KANLayer**，不是两个 KAN 串联。其网格与原版一致：16→16，5 个区间、三次样条、初始区间 `[-1,1]`。

## 初始化、损失与冻结

起点为：`D:\Projects\Sim2real\imp\IBN\runs\ibn_sim_1024\weights\best.pt`。

此 IBN 模型的 366 个状态张量全部原样保留。原分类器只改变在模型树中的路径，参数没有重新初始化。新 KAN 输出投影和新边缘残差投影均零初始化，因此接入后的初始分割输出与 IBN 底座相同。这里复用旧模块的结构，不装入它们在旧骨干上训练得到的权重。

默认配套训练方式：

| 部分 | 是否更新 |
|---|---|
| IBN、骨干、颈部、原语义分类器 | 冻结 |
| 原 BN 运行统计量 | 固定，保持 eval |
| KAN 的卷积、样条系数及可学习缩放 | 更新 |
| KAN 网格和固定掩码 | 固定 |
| 边缘特征卷积、边界头、残差投影 | 更新 |

IN 的仿射参数固定，但按其定义仍会为当前输入计算逐图统计。整个模型共有 **1,646,555 个参数**，其中本阶段更新 **15,025 个**：KAN 4,688，边缘分支 10,337。KAN 的固定网格和掩码计入总参数，不计入可训练参数。

边界损失沿用原版本：从增强后的仿真类别标签中提取撞击坑（类别 1）的四邻接边界；忽略 255 区域及其直接邻接位置。边界 logits 双线性插值到标签尺寸，使用带正类权重的 BCE，正类权重为负/正样本数之比并限制在 `[1,20]`。

原语义损失已经按 batch 累计，因此代码中的总损失为：

`L_batch = L_semantic_batch + 0.1 × B × L_edge_mean`

新训练器每轮恢复冻结模式，优化器只接收两项新分支的可训练参数，并保护 EMA 中的冻结状态。保存沿用原框架 FP16 格式；重载检查按相同存储精度比较冻结张量。

## 文件与权重

- [组合模块与冻结控制](ibn_kan_edge.py)：新增及修改逻辑有 `【新增】` / `【修改】` 标注，旧路径以注释保留。
- [构建入口](build_ibn_kan_edge.py)：生成完整初始模型，拒绝覆盖已有输出。
- [训练入口](train_ibn_kan_edge.py)：只用仿真数据联合训练 KAN 与边缘分支。
- [预测入口](predict_ibn_kan_edge.py)：注册已有自定义类并加载完整模型。
- [验证入口](verify_ibn_kan_edge.py)：CPU 的结构、冻结、梯度、融合和预测检查。
- [完整初始权重](weights/ibn_kan_edge_init.pt)：包含已训练 IBN 底座及尚未训练的两项新分支。
- [结构和梯度检查](checks/ibn_kan_edge_verification.json)。
- [一轮小样本训练检查](checks/ibn_kan_edge_smoke_audit.json)。
- [构建和来源记录](weights/ibn_kan_edge_build_audit.json)。

旧 IBN、KAN、edge 程序及旧权重均保留。新程序依赖本机原项目和原官方 pykan 目录，不需要再次安装依赖。请使用专用入口，普通 YOLO 训练入口按旧 YAML 重建时可能丢失组合结构。

## 后续正式训练命令

**本次尚未执行此命令。** 只使用 `Traindata` 对应的现有仿真划分（8,000 张训练图、1,320 张验证图），不加入真实图像。

```powershell
& 'D:\Projects\[Seg]\ultralytics-main\.venv-yolo\Scripts\python.exe' 'D:\Projects\Sim2real\imp\IBN\train_ibn_kan_edge.py' --train --epochs 30 --imgsz 1024 --batch 4 --device 0
```

AdamW，学习率 `1e-4`，早停耐心值 10。输出位于 `runs/ibn_kan_edge_sim_1024`，同名目录已存在时自动生成新目录。`best.pt` 保存完整组合模型，依据仿真验证 mIoU 选取。`--weights` 可指定其他本组合完整模型进行新的微调，不提供恢复优化器的中断续训入口。

CPU 小样本检查：

```powershell
& 'D:\Projects\[Seg]\ultralytics-main\.venv-yolo\Scripts\python.exe' 'D:\Projects\Sim2real\imp\IBN\train_ibn_kan_edge.py' --smoke --device cpu
```

它使用 8 张仿真训练图、4 张验证图、256 输入训练 1 轮。检查已在工作区执行通过，只证明流程正确，不能用于判断真实域效果；小样本权重保留在工作区，不冒充正式训练权重。

未来训练完成后预测：

```powershell
& 'D:\Projects\[Seg]\ultralytics-main\.venv-yolo\Scripts\python.exe' 'D:\Projects\Sim2real\imp\IBN\predict_ibn_kan_edge.py' 'D:\Projects\Sim2real\data\real_data\CE3' --weights 'D:\Projects\Sim2real\imp\IBN\runs\ibn_kan_edge_sim_1024\weights\best.pt' --imgsz 1024 --device 0
```

请使用实际生成的训练目录。新分支尚未正式训练，因此现在还没有本组合的真实测试提升结论。
