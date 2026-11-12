"""【新增 CODE-PACKAGE-01】复制当前完整模型工程；只修改新副本的代码路径。"""
from __future__ import annotations

import ast
from datetime import datetime
import hashlib
from importlib import metadata
import json
from pathlib import Path
import shutil
import sys

SOURCE=Path(r'D:\Projects\Sim2real\imp\IBN')
KAN=Path(r'D:\Projects\[Seg]\ultralytics-main\sim2real_kan_adapter')
EDGE=Path(r'D:\Projects\Sim2real\imp\edge')
ULTRA=Path(r'D:\Projects\[Seg]\ultralytics-main\ultralytics-main')
DEST=Path(r'D:\Projects\Sim2real\code')
SKIP={'.git','__pycache__','.ipynb_checkpoints','.pytest_cache','.ruff_cache'}
COPIED=[]
PATCHED=[]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_file(source: Path, relative: str | Path) -> None:
    target=DEST/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    before=digest(source)
    shutil.copy2(source,target)
    assert digest(target)==before
    COPIED.append({'source':str(source),'copy':target.relative_to(DEST).as_posix(),'source_sha256':before})


def copy_tree(source: Path, relative: str) -> None:
    for path in sorted(source.rglob('*')):
        if path.is_file() and not any(part in SKIP for part in path.relative_to(source).parts):
            if path.is_symlink():
                raise RuntimeError(f'检查软链接后再复制：{path}')
            copy_file(path,Path(relative)/path.relative_to(source))


def patch(name: str, old: str, new: str, count: int=1, retain: bool=True) -> None:
    path=DEST/name
    text=path.read_text(encoding='utf-8')
    assert text.count(old)==count,(name,old,text.count(old))
    if retain:
        replacement='\n'.join('# 【整理前代码保留】 '+line for line in old.splitlines())+'\n# 【整理修改 CODE-PACKAGE-02】使用本副本中的模块与资源。\n'+new
    else:
        replacement=new
    path.write_text(text.replace(old,replacement),encoding='utf-8')
    if name not in PATCHED:
        PATCHED.append(name)


def main() -> None:
    if DEST.exists() and any(DEST.iterdir()):
        raise FileExistsError('目标目录非空，拒绝覆盖已有文件：'+str(DEST))
    DEST.mkdir(parents=True,exist_ok=True)
    # 当前 IBN 工程顶层全部程序、说明和结构图原样留档，再创建可运行副本。
    for source in sorted(SOURCE.iterdir()):
        if source.is_file() and source.suffix.lower() in {'.py','.md','.svg','.json'}:
            copy_file(source,Path('original_sources/IBN')/source.name)
            relative=Path('docs/original_IBN_README.md') if source.name=='README.md' else Path(source.name)
            copy_file(source,relative)
    for root,name in [(KAN,'kan_adapter.py'),(EDGE,'edge_branch.py')]:
        copy_file(root/name,Path('original_sources')/root.name/name)
        copy_file(root/name,name)
    copy_tree(ULTRA,'ultralytics_source')
    copy_tree(KAN/'official_pykan','official_pykan')
    copy_tree(KAN/'dependencies','dependencies')
    copy_tree(SOURCE/'weights','weights')
    for run in ['ibn_sim_1024','ibn_kan_edge_real_train_1024_v2','ibn_kan_edge_real_train_1024_resume']:
        copy_tree(SOURCE/'runs'/run,'runs/'+run)
    copy_file(Path(r'D:\Projects\Sim2real\runs\yolo26n_sem_50epochs_1024\weights\best.pt'),'weights/yolo26n_sem_sim_best.pt')
    copy_file(Path(r'D:\Projects\Sim2real\data\simulated_data\yolo26_semantic_data\masks\train\1693274727326823936.png'),'assets/lunar_palette.png')
    copy_file(SOURCE/'real_adapter_train_data/provenance.json','metadata/real_train_provenance.json')
    copy_file(Path(r'D:\Projects\[Seg]\ultralytics-main\sim2real_conv_adapter\apollo_test_20261110\output\source_hashes.json'),'metadata/apollo_200_source_hashes.json')
    copy_file(Path(__file__),'tools/package_current_code.py')
    copy_file(Path(__file__).with_name('verify_bundle.py'),'verify_bundle.py')
    copy_file(Path(__file__).with_name('original_forward_reference.json'),'metadata/original_forward_reference.json')
    copy_file(KAN/'使用说明.md','docs/original_KAN_README.md')

    patch('runtime.py',"REPO = Path(os.environ.get('IBN_ULTRALYTICS_ROOT', r'D:\\Projects\\[Seg]\\ultralytics-main\\ultralytics-main'))","REPO = ROOT / 'ultralytics_source'")
    patch('runtime.py',"BASE = Path(r'D:\\Projects\\Sim2real\\runs\\yolo26n_sem_50epochs_1024\\weights\\best.pt')","BASE = ROOT / 'weights/yolo26n_sem_sim_best.pt'\nPALETTE = ROOT / 'assets/lunar_palette.png'")
    patch('runtime.py',"    paths += [REPO.parent / p for p in ('sim2real_edge_branch/edge_branch.py',\n                                      'sim2real_edge_branch/train_edge.py',\n                                      'sim2real_conv_adapter/train_adapter.py')]",
          "    paths += [ROOT / p for p in ('kan_adapter.py', 'edge_branch.py', 'ibn_backbone.py', 'train_ibn.py')]")
    patch('ibn_kan_edge.py',"KAN_ROOT = Path(r'D:\\Projects\\[Seg]\\ultralytics-main\\sim2real_kan_adapter')\nEDGE_ROOT = Path(r'D:\\Projects\\Sim2real\\imp\\edge')","KAN_ROOT = ROOT\nEDGE_ROOT = ROOT")
    patch('ibn_kan_edge.py',"BASE_IBN = Path(r'D:\\Projects\\Sim2real\\imp\\IBN\\runs\\ibn_sim_1024\\weights\\best.pt')","BASE_IBN = ROOT / 'runs/ibn_sim_1024/weights/best.pt'")
    patch('ibn_kan_edge.py',"    paths = [BASE_IBN, KAN_ROOT / 'kan_adapter.py', EDGE_ROOT / 'edge_branch.py',\n             KAN_ROOT / 'runs/kan_sim_1024/weights/best.pt',\n             EDGE_ROOT / 'runs/edge_sim_1024-2/weights/best.pt',\n             Path(r'D:\\Projects\\Sim2real\\imp\\IBN') / 'ibn_backbone.py',\n             Path(r'D:\\Projects\\Sim2real\\imp\\IBN') / 'train_ibn.py']",
          "    paths = [BASE_IBN, ROOT / 'kan_adapter.py', ROOT / 'edge_branch.py',\n             ROOT / 'ibn_backbone.py', ROOT / 'train_ibn.py', ROOT / 'ibn_kan_edge.py']")
    patch('edge_branch.py',"KAN_ROOT = Path(r'D:\\Projects\\[Seg]\\ultralytics-main\\sim2real_kan_adapter')","KAN_ROOT = Path(__file__).resolve().parent")
    patch('evaluate_kaggle_test.py',"OLD_HASHES = Path(r'D:\\Projects\\[Seg]\\ultralytics-main\\sim2real_conv_adapter\\apollo_test_20261110\\output\\source_hashes.json')","OLD_HASHES = ROOT / 'metadata/apollo_200_source_hashes.json'")
    for name in ['evaluate_kaggle_test.py','evaluate_real_adapter_test.py','predict_ce3_full.py']:
        path=DEST/name
        lines=path.read_text(encoding='utf-8').splitlines()
        import_line=next(line for line in lines if line.startswith('from runtime import '))
        patch(name,import_line,import_line+', PALETTE')
        text=path.read_text(encoding='utf-8')
        for expression in ["SIM / 'masks/train/1693274727326823936.png'","SIM/'masks/train/1693274727326823936.png'"]:
            if expression in text:
                # 表达式位于函数内部，仅改资源位置；原完整文件已保存在 original_sources。
                patch(name,expression,'PALETTE',retain=False)
    patch('evaluate_real_adapter_test.py',"    provenance_path=ROOT/'real_adapter_train_data/provenance.json'","    provenance_path=ROOT/'metadata/real_train_provenance.json'")
    patch('verify_real_training_recovery.py','from ibn_kan_edge import YOLO, torch, INITIAL, branches_only_mode, frozen_state, check_frozen',
          'from ibn_kan_edge import YOLO, torch, INITIAL, branches_only_mode, frozen_state, check_frozen\nfrom runtime import ROOT')
    patch('verify_real_training_recovery.py',"    path = Path(r'D:\\Projects\\Sim2real\\imp\\IBN\\runs\\ibn_kan_edge_real_train_1024_v2\\weights\\last.pt')",
          "    path = ROOT/'runs/ibn_kan_edge_real_train_1024_v2/weights/last.pt'")
    for path in DEST.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'),filename=str(path))

    # 不复制整个 Python 环境；记录本机运行版本，单独复制 pykan 局部依赖。
    packages=['torch','torchvision','numpy','scipy','matplotlib','opencv-python','Pillow','PyYAML','sympy','pandas',
              'tqdm','seaborn','psutil','polars','requests','ultralytics-thop','huggingface-hub','setuptools','filelock','fsspec']
    versions={}
    for name in packages:
        try:
            versions[name]=metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    (DEST/'requirements-runtime.txt').write_text('# 本机已验证 Python 3.10 环境；CUDA/PyTorch 请选匹配平台的安装包。\n'+
        '\n'.join(name+'=='+version for name,version in versions.items())+'\n',encoding='utf-8')
    (DEST/'environment_snapshot.json').write_text(json.dumps({'python':sys.version,'executable':sys.executable,
        'packages':versions,'local_pykan_dependencies':{'scikit-learn':'1.5.2','joblib':'1.4.2','threadpoolctl':'3.5.0'},
        'local_dependency_platform':'Windows CPython 3.10 (copied existing binaries)'},ensure_ascii=False,indent=2),encoding='utf-8')
    (DEST/'README.md').write_text(README,encoding='utf-8')
    for row in COPIED:
        assert digest(Path(row['source']))==row['source_sha256'],'源文件在整理期间发生变化：'+row['source']
        row['copied_sha256']=digest(DEST/row['copy'])
        row['modified_for_local_paths']=row['copied_sha256']!=row['source_sha256']
    manifest={'created_at':datetime.now().astimezone().isoformat(),'destination':str(DEST),'source_files_unchanged':True,
              'patched_files':PATCHED,'files':COPIED,'file_count':len(COPIED),
              'copied_bytes':sum((DEST/row['copy']).stat().st_size for row in COPIED),
              'datasets_copied':False,'python_environment_copied':False,'verification':'pending'}
    (DEST/'COPY_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='files'},ensure_ascii=False,indent=2),flush=True)


README=r'''# 当前完整模型工程：YOLO26n-sem＋IBN＋单层 KAN＋边缘分支

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
'''


if __name__=='__main__':
    main()
