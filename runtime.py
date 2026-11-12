"""【新增文件 IBN-00】独立实验运行环境；不修改原 Ultralytics 或旧实验。"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# 【整理前代码保留】 REPO = Path(os.environ.get('IBN_ULTRALYTICS_ROOT', r'D:\Projects\[Seg]\ultralytics-main\ultralytics-main'))
# 【整理修改 CODE-PACKAGE-02】使用本副本中的模块与资源。
REPO = ROOT / 'ultralytics_source'
if not (REPO / 'ultralytics/nn/tasks.py').is_file():
    raise FileNotFoundError(f'找不到项目使用的 Ultralytics 源码：{REPO}')
sys.path.insert(0, str(REPO))
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'yolo-config'))
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')

import torch  # noqa: E402
import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402

if not Path(ultralytics.__file__).resolve().is_relative_to(REPO.resolve()):
    raise RuntimeError('载入了其他 Ultralytics 安装，请使用本项目 Python 环境')

# 【整理前代码保留】 BASE = Path(r'D:\Projects\Sim2real\runs\yolo26n_sem_50epochs_1024\weights\best.pt')
# 【整理修改 CODE-PACKAGE-02】使用本副本中的模块与资源。
BASE = ROOT / 'weights/yolo26n_sem_sim_best.pt'
PALETTE = ROOT / 'assets/lunar_palette.png'
INITIAL = ROOT / 'weights/yolo26n_sem_ibn_init.pt'
SIM = Path(r'D:\Projects\Sim2real\data\simulated_data\yolo26_semantic_data')
NAMES = ['lunar_regolith', 'crater', 'rock', 'mountain', 'sky']


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_hashes() -> dict[str, str]:
    """【新增】记录原权重和关键源文件哈希，验证它们未被本实验修改。"""
    paths = [BASE] + [REPO / p for p in (
        'ultralytics/nn/modules/conv.py', 'ultralytics/nn/modules/block.py',
        'ultralytics/nn/tasks.py', 'ultralytics/engine/trainer.py',
        'ultralytics/models/yolo/semantic/train.py',
        'ultralytics/cfg/models/26/yolo26-sem.yaml')]
# 【整理前代码保留】     paths += [REPO.parent / p for p in ('sim2real_edge_branch/edge_branch.py',
# 【整理前代码保留】                                       'sim2real_edge_branch/train_edge.py',
# 【整理前代码保留】                                       'sim2real_conv_adapter/train_adapter.py')]
# 【整理修改 CODE-PACKAGE-02】使用本副本中的模块与资源。
    paths += [ROOT / p for p in ('kan_adapter.py', 'edge_branch.py', 'ibn_backbone.py', 'train_ibn.py')]
    return {str(p): sha256(p) for p in paths}
