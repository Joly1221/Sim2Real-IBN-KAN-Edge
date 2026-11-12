# 当前完整模型工程：YOLO26n-sem＋IBN＋单层 KAN＋边缘分支

本目录是 2026-11-12 的独立代码副本。原程序、数据、训练目录和权重全部保留。
本次只整理代码，没有重新训练。最新完整模型是真实 train 微调累计第 30 轮后的 EMA `last.pt`，不是零样本模型，也不是按测试精度选择的 best.pt。

## 从哪里看起

| 文件或目录 | 内容 |
|---|---|
| `ibn_kan_edge.py` | 组合入口和冻结逻辑：IBN 骨干/颈部 → P3 → KAN 残差 → 边缘残差 → 原五类分类器 |
| `ibn_backbone.py` | 浅层 C3k2 的 IN/BN 归一化 |
| `kan_adapter.py` | 64→16→官方单层 KAN(16→16)→64 的适配器 |
| `edge_branch.py` | 边缘特征分支、坑边界生成和损失 |
| `train_ibn.py` | 仿真数据训练 IBN 主模型 |
| `train_ibn_kan_edge.py` | 仿真数据仅训练 KAN＋边缘分支；包含共用训练器 |
| `prepare_ibn_real_train.py` / `train_ibn_kan_edge_real.py` | 真实标注转换及仅训练 KAN＋边缘分支 |
| `predict_ibn_kan_edge.py` / `predict_ce3_full.py` | 完整模型通用预测 / CE3 两张图预测 |
| `evaluate_real_adapter_test.py` | 46 张 Kaggle test 评价、五类指标展示及图像浏览页 |
| `verify_bundle.py` | 本副本导入、检查点、冻结范围与前向检查 |
| `ultralytics_source/` | 本项目实际使用的完整 Ultralytics 源码、配置、文档和测试；不是另装官方最新版 |
| `official_pykan/` | 实际使用的官方 pykan 源码及文档，原样复制 |
| `dependencies/` | 原 pykan 局部依赖，包含 Windows CPython 3.10 二进制 |
| `original_sources/` | 整理前自定义程序的原样快照 |
| `COPY_MANIFEST.json` | 每个复制文件的来源、原始/副本 SHA-256 和路径修改记录 |

代码中的 `【整理修改 CODE-PACKAGE-02】` 标出本次路径调整。原语句保留注释，完整原文件另存于 `original_sources/`。模型结构、损失与训练设置没有改变。导入不再依赖旧 IBN/KAN/edge 代码目录。历史运行参数和文档中的原路径保留为来源记录，以本 README 的运行命令为准。

## 已复制的权重

- `runs/ibn_kan_edge_real_train_1024_resume/weights/last.pt`：**当前完整模型**，第30轮，IBN＋KAN＋边缘分支，SHA-256 `5eba98606a50a81b9de7ac4f99446ac07a37c8274d0bad013b0360c18cfe7c31`。
- `runs/ibn_sim_1024/weights/best.pt`：仿真训练后的 IBN 底座。
- `weights/ibn_kan_edge_init.pt`：加入零残差 KAN＋边缘分支后的完整初始化模型。
- `weights/yolo26n_sem_ibn_init.pt`：IBN 初始化模型。
- `weights/yolo26n_sem_sim_best.pt`：最初的仿真 YOLO26n-sem 权重。
- `runs/ibn_kan_edge_real_train_1024_v2/weights/last.pt`：第9轮旧检查点，保留用于恢复验证和实验追溯。

## 环境与最短运行方法

在 PowerShell 中执行。以下使用本机已经验证的 Python 环境；无需修改或重新安装原环境。

```powershell
Set-Location -LiteralPath 'D:\Projects\Sim2real\code'
$pythonExe = 'D:\Projects\[Seg]\ultralytics-main\.venv-yolo\Scripts\python.exe'
& $pythonExe -B -X utf8 '.\verify_bundle.py'
& $pythonExe -B -X utf8 '.\predict_ibn_kan_edge.py' 'D:\Projects\Sim2real\data\real_data\CE3' --weights '.\runs\ibn_kan_edge_real_train_1024_resume\weights\last.pt' --device cpu --imgsz 1024 --name ce3_complete
```

生成 CE3 对照浏览页：`& $pythonExe -B -X utf8 '.\predict_ce3_full.py'`，输出 `ce3_full_model_20261112/output/index.html`。

评价已有 46 张真实 test：`& $pythonExe -B -X utf8 '.\evaluate_real_adapter_test.py' --device cpu`，输出 `kaggle_full_model_test_20261112/output/index.html`。
这些专用脚本拒绝覆盖已有输出；重复评价请用 `--out` 指定新目录（CE3 专用脚本需调整 OUT，或使用通用预测入口指定新 name）。

Python 主环境没有复制。版本见 `environment_snapshot.json` 和 `requirements-runtime.txt`。新机器需要建立兼容环境；没有声称已完成异机/新环境验证。局部 sklearn 二进制适用于 Windows/Python3.10。迁移到不同平台时应在对应环境安装兼容依赖，不能直接使用这些二进制。

## 数据和重新训练

数据集不重复复制，仍使用 `D:\Projects\Sim2real\data`。仿真预处理集为 `simulated_data/yolo26_semantic_data`（来自 Traindata），由 `runtime.py` 中 SIM 指定。真实训练默认 `real_data/kaggle_dataset/test/train`；真实测试为其相邻 test；CE3 使用 `real_data/CE3`。换数据位置时通过入口参数或修改这些数据路径。

如需重新进行真实训练，先准备本副本自己的数据副本，再从组合初始化权重开始：

```powershell
& $pythonExe -B -X utf8 '.\prepare_ibn_real_train.py'
& $pythonExe -B -X utf8 '.\train_ibn_kan_edge_real.py' --train --epochs 30 --imgsz 1024 --batch 4 --device 0 --name real_train_new
```

训练范围是 KAN＋边缘分支的 15,025 个参数；IBN、原骨干、颈部、分类器、BN 统计和 KAN 固定网格保持冻结。150 张真实 train 全量训练，不使用 test、不划验证集、不早停，保存完整 last.pt。此设置不是零样本 sim-to-real。

默认不自动启动任何训练。`--smoke` 是小样本流程检查。旧第9轮检查点只含 EMA；历史恢复并非与不中断训练逐位相同。历史检查点 train_args 保留原数据路径，恢复时须保持原训练计划；已完成第30轮的权重用作预测或新的实验起点，不应当作未完成训练直接 resume。

## 当前效果和评价边界

真实数据的标签只有坑、岩石、天空；未标注/跨类重叠像素为255。当前 Kaggle 46张测试的已标注三类 mIoU 为63.34%，不是全图五类 mIoU。月壤和山体无真值指标显示“无法计算”。真实微调模型在 CE3 出现大面积预测为坑，不能把高坑召回当作整体精度高。CE3没有真值，不能计算精度。

本次复制的是当前模型相关完整代码及运行所需模块，不包含历史 Conv-Adapter/MLP 等独立实验目录、全部预测结果、数据集和虚拟环境。当前 IBN 程序原样快照完整保留。图示见 `ibn_kan_edge_structure.svg`，第三方许可随源码及 `THIRD_PARTY_NOTICES.md` 保留。
