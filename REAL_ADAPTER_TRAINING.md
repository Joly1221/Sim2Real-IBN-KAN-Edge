# 仅使用真实 train 微调适配器

【新增，2026-11-12】按照用户指定，`D:\Projects\Sim2real\data\real_data\kaggle_dataset\test\train` 中的 150 张图像全部用于训练。不从中划分验证集，不调用旁边 test 数据，不做逐轮或训练结束后的数据集验证。

起点为 `weights/ibn_kan_edge_init.pt`：已经仿真训练过的 IBN 底座，加零残差初始化的单层 KAN 和边缘分支。它不是此前旧骨干上的 KAN 权重，也不包含小样本检查的更新。

## 冻结与训练

- 固定 IBN 的 IN/BN 仿射参数、原骨干、颈部、原语义分类器和所有原 BN 运行统计。
- 更新单层 KAN 适配器及边缘分支，共 15,025 个参数。KAN 网格、固定掩码不更新。
- 每轮记录 `freeze_checks.jsonl`，逐项检查冻结参数和缓冲区；EMA 的冻结部分也精确保留。
- 只使用真实 train 图像和标注，这次实验属于有监督真实域微调，不属于零样本迁移。

## 标签与边界损失

COCO 类别映射：Craters→1、Rocks→2、Sky→4。原五类输出保持不变。未标注或跨类别重叠区域为 255，不把它们当作月壤。原图和原始 annotations.json 不修改，训练副本位于 `real_adapter_train_data`。

沿用之前的边界损失：只在可靠的已标注相邻像素上构造坑边界，未知区域及直接邻域排除。此稀疏标注集的可靠正边界像素很少，绝大部分多边形外围邻接未知区域，不能将这次训练解释为完整坑边界监督；边缘残差分支同时接收分割主损失。

## 训练与保存

默认 30 轮、1024 输入、batch 4、AdamW、学习率 `1e-4`、单 GPU、无验证、无早停。名义有效 batch 设置为当前 batch，避免小数据集每轮只有少数优化器更新。旧的仿真训练入口保持不变。

专用训练器显式跳过框架最后一轮和训练结束时的默认验证，只保存每轮更新的完整 FP32 `weights/last.pt`（最后一轮 EMA 模型），不生成 `best.pt`。没有独立验证指标，不能宣称这个权重是泛化表现最佳的一轮。`results.csv` 仅记录训练损失和学习率。

数据 YAML 中的 val 指向 train 只是框架必需的配置占位；验证调用已在专用训练器中关闭。不会把训练集分数标成验证精度。

```powershell
& 'D:\Projects\[Seg]\ultralytics-main\.venv-yolo\Scripts\python.exe' 'D:\Projects\Sim2real\imp\IBN\prepare_ibn_real_train.py'
& 'D:\Projects\[Seg]\ultralytics-main\.venv-yolo\Scripts\python.exe' 'D:\Projects\Sim2real\imp\IBN\train_ibn_kan_edge_real.py' --train --epochs 30 --imgsz 1024 --batch 4 --device 0
```

数据准备程序拒绝覆盖已有数据副本；已有副本不需要重复运行。训练输出为 `runs/ibn_kan_edge_real_train_1024`，重名时生成新目录。

`--smoke --device cpu` 只取上述 train 的 8 张图、256 输入、1 轮来检查程序与冻结；它不验证模型泛化能力。正式训练从原组合初始权重重新开始。

实时状态见实验目录下 `progress.json`，每轮冻结检查见 `freeze_checks.jsonl`，训练结束后生成 `real_adapter_audit.json`。新权重可用 `predict_ibn_kan_edge.py --weights <last.pt>` 加载。

## 2026-11-12 进度写入修复与恢复

【修改 IBN-REAL-04】`ibn_kan_edge_real_train_1024_v2` 已完成第 9 轮。进入第 10 轮时，Windows 拒绝替换 `progress.json`，程序退出；第 9 轮完整检查点和优化器状态保存成功。现在进度文件使用唯一临时文件，替换失败会重试，持续占用时写入 `progress_events.jsonl` 并继续训练。检查点保存仍是关键操作，重试后无法保存会报错。

【新增 IBN-REAL-05】恢复入口 `--resume-from <last.pt>` 保留优化器、EMA 计数和累计 30 轮学习率计划。第 9 轮旧检查点只包含 EMA 模型，恢复时以该模型继续，不声称与不中断训练逐位相同。新检查点另存 `resume_model` 实时训练模型，供后续恢复使用。数据、冻结范围和禁用验证设置保持原计划。

```powershell
& 'D:\Projects\[Seg]\ultralytics-main\.venv-yolo\Scripts\python.exe' 'D:\Projects\Sim2real\imp\IBN\train_ibn_kan_edge_real.py' --train --resume-from 'D:\Projects\Sim2real\imp\IBN\runs\ibn_kan_edge_real_train_1024_v2\weights\last.pt' --epochs 30 --imgsz 1024 --batch 4 --device 0 --name ibn_kan_edge_real_train_1024_resume
```

旧运行目录保留。恢复后的目录保存第 10—30 轮记录，`resume_restore.json` 记录恢复轮次、优化器和 EMA 计数，最终审计报告记录累计完成轮次。
